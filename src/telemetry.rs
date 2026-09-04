//! Temperature readings for GPU and CPU.
//!
//! GPU: NVML (`nvml.dll` / `libnvidia-ml.so`) ships with the NVIDIA driver,
//! so no toolkit is needed - the same principle as the PTX solution for the
//! kernel. The library is loaded dynamically and a missing NVML just gives
//! `None`.
//!
//! CPU: there is no portable API. Linux has hwmon in sysfs; on Windows a real
//! core temperature requires a signed driver (LibreHardwareMonitor and
//! others), which we do not want to bundle with a miner. So we read what can
//! be had without extra privileges and otherwise show "-".

#[cfg(feature = "cuda")]
use nvml_wrapper::Nvml;

/// An opened telemetry source. `None` fields mean "not available here".
pub struct Telemetry {
    #[cfg(feature = "cuda")]
    nvml: Option<Nvml>,
}

/// One reading; any field may be missing depending on platform/hardware.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Reading {
    pub gpu_temp_c: Option<u32>,
    pub gpu_fan_pct: Option<u32>,
    pub gpu_power_w: Option<f64>,
    pub cpu_temp_c: Option<u32>,
}

impl Telemetry {
    /// Open the available sources. The NVML loader can panic if the library
    /// is missing - the same guard as in the CUDA detection.
    pub fn open() -> Self {
        #[cfg(feature = "cuda")]
        {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let nvml = std::panic::catch_unwind(Nvml::init).ok().and_then(|r| r.ok());
            std::panic::set_hook(prev);
            return Self { nvml };
        }
        #[cfg(not(feature = "cuda"))]
        Self {}
    }

    /// Read current values. Errors are swallowed - telemetry must never
    /// disturb mining.
    pub fn read(&self, gpu_index: u32) -> Reading {
        let mut r = Reading {
            cpu_temp_c: read_cpu_temp(),
            ..Default::default()
        };

        #[cfg(feature = "cuda")]
        if let Some(nvml) = &self.nvml {
            if let Ok(dev) = nvml.device_by_index(gpu_index) {
                r.gpu_temp_c = dev
                    .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                    .ok();
                r.gpu_fan_pct = dev.fan_speed(0).ok();
                // NVML reports milliwatts.
                r.gpu_power_w = dev.power_usage().ok().map(|mw| mw as f64 / 1000.0);
            }
        }
        let _ = gpu_index;
        r
    }
}

/// CPU temperature where the platform exposes it without extra privileges.
#[cfg(target_os = "linux")]
fn read_cpu_temp() -> Option<u32> {
    // Look for an hwmon that reports package/core temperature.
    let dir = std::fs::read_dir("/sys/class/hwmon").ok()?;
    for entry in dir.flatten() {
        let base = entry.path();
        let name = std::fs::read_to_string(base.join("name")).unwrap_or_default();
        let name = name.trim();
        if !matches!(name, "coretemp" | "k10temp" | "zenpower" | "cpu_thermal") {
            continue;
        }
        // temp1_input is the package on coretemp/k10temp; value in millicelsius.
        if let Ok(v) = std::fs::read_to_string(base.join("temp1_input")) {
            if let Ok(milli) = v.trim().parse::<i64>() {
                return u32::try_from(milli / 1000).ok();
            }
        }
    }
    None
}

/// Windows: `MSAcpi_ThermalZoneTemperature` is the only source that does not
/// require a kernel driver of our own. Many motherboards report nothing at all
/// there - then the answer is `None` and the GUI shows "-".
///
/// Reading it means starting PowerShell, which is why this is cached and why a
/// probe that comes back empty turns the whole thing off permanently. See
/// `CPU_TEMP_REFRESH` and `CpuTempCache` below.
#[cfg(target_os = "windows")]
fn read_cpu_temp() -> Option<u32> {
    let mut cache = CPU_TEMP.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    if cache.needs_probe(now) {
        cache.record(probe_cpu_temp(), now);
    }
    cache.value
}

#[cfg(target_os = "windows")]
static CPU_TEMP: std::sync::Mutex<CpuTempCache> = std::sync::Mutex::new(CpuTempCache::new());

/// Full path rather than the bare name.
///
/// `Command::new("powershell")` resolves through PATH and (on Windows) the
/// working directory, which is the same search-order hole `harden_dll_search_path`
/// in main.rs closes for DLLs. The CLI is typically unpacked and run straight
/// out of Downloads, where a dropped `powershell.exe` would then be executed
/// as the user.
#[cfg(target_os = "windows")]
fn powershell_path() -> std::path::PathBuf {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    std::path::Path::new(&root).join(r"System32\WindowsPowerShell\v1.0\powershell.exe")
}

#[cfg(target_os = "windows")]
fn probe_cpu_temp() -> Option<u32> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let out = std::process::Command::new(powershell_path())
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature \
              -ErrorAction SilentlyContinue | Select-Object -First 1).CurrentTemperature",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let decikelvin: f64 = text.trim().parse().ok()?;
    if decikelvin <= 0.0 {
        return None;
    }
    // The value is in tenths of a kelvin.
    let celsius = decikelvin / 10.0 - 273.15;
    (0.0..150.0).contains(&celsius).then(|| celsius.round() as u32)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_cpu_temp() -> Option<u32> {
    None
}

// ----------------------------------------------------------------------
// Probe throttling for the Windows CPU temperature.
//
// Compiled under `test` on every platform as well, so the rule below is
// covered by CI - which has no Windows runner for `cargo test` on Linux
// images, and where the probe itself can never run.
// ----------------------------------------------------------------------

/// How long a reading is reused before PowerShell is started again.
///
/// The stats tick is 5 s by default and 3 s under the GUI, so this used to be
/// about 20 process launches a minute, each costing hundreds of milliseconds
/// of CPU and tens of megabytes. That competes with the GPU feeder thread,
/// which has to be scheduled to queue the next kernel in time - on exactly the
/// laptops this telemetry exists for. A CPU package temperature does not move
/// fast enough to be worth any of it. A binary that repeatedly spawns
/// PowerShell is also a shape EDR heuristics flag, and a miner starts out with
/// no benefit of the doubt.
#[cfg(any(target_os = "windows", test))]
const CPU_TEMP_REFRESH: std::time::Duration = std::time::Duration::from_secs(60);

#[cfg(any(target_os = "windows", test))]
use std::time::Instant;

#[cfg(any(target_os = "windows", test))]
struct CpuTempCache {
    /// `None` until the first probe has run.
    probed_at: Option<Instant>,
    value: Option<u32>,
    /// Cleared for the life of the process when the FIRST probe yields
    /// nothing usable. `MSAcpi_ThermalZoneTemperature` needs admin on most
    /// consumer machines and returns nothing there, so for those users the
    /// answer will never change and every further attempt is pure cost.
    keep_probing: bool,
}

#[cfg(any(target_os = "windows", test))]
impl CpuTempCache {
    const fn new() -> Self {
        Self { probed_at: None, value: None, keep_probing: true }
    }

    fn needs_probe(&self, now: Instant) -> bool {
        match self.probed_at {
            None => true,
            Some(_) if !self.keep_probing => false,
            Some(at) => now.duration_since(at) >= CPU_TEMP_REFRESH,
        }
    }

    fn record(&mut self, value: Option<u32>, now: Instant) {
        // Only the first probe latches it off. A machine that has answered
        // once can still have a transient failure, and giving up on telemetry
        // for the rest of the run over one blip would lose a reading that
        // works - and by then the cost of retrying is one process a minute,
        // not one per tick.
        if self.probed_at.is_none() && value.is_none() {
            self.keep_probing = false;
        }
        self.probed_at = Some(now);
        self.value = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_telemetry_never_panics() {
        // Without an NVIDIA driver this must give an empty source, not crash.
        let t = Telemetry::open();
        let r = t.read(0);
        // No requirements on the values - only that the call survives.
        let _ = (r.gpu_temp_c, r.cpu_temp_c, r.gpu_power_w, r.gpu_fan_pct);
    }

    /// A machine that reports nothing must be probed exactly once.
    ///
    /// This is the whole point of the cache: `MSAcpi_ThermalZoneTemperature`
    /// needs admin on most consumer hardware, so for most users every probe
    /// after the first is a PowerShell process started for a value that will
    /// never arrive.
    #[test]
    fn a_first_probe_that_finds_nothing_is_never_repeated() {
        let t0 = Instant::now();
        let mut c = CpuTempCache::new();
        assert!(c.needs_probe(t0));
        c.record(None, t0);
        assert!(!c.needs_probe(t0));
        assert!(!c.needs_probe(t0 + CPU_TEMP_REFRESH * 100));
        assert_eq!(c.value, None);
    }

    /// A working reading is reused between ticks and refreshed on a timer,
    /// not on every tick.
    #[test]
    fn a_working_probe_is_cached_and_refreshed_on_a_timer() {
        let t0 = Instant::now();
        let mut c = CpuTempCache::new();
        c.record(Some(52), t0);
        assert_eq!(c.value, Some(52));
        assert!(!c.needs_probe(t0));
        assert!(!c.needs_probe(t0 + CPU_TEMP_REFRESH / 2));
        assert!(c.needs_probe(t0 + CPU_TEMP_REFRESH));
    }

    /// Only the first probe latches probing off; one blip on a machine that
    /// does report a temperature must not cost it telemetry for the run.
    #[test]
    fn a_later_failure_does_not_disable_probing() {
        let t0 = Instant::now();
        let mut c = CpuTempCache::new();
        c.record(Some(52), t0);
        let t1 = t0 + CPU_TEMP_REFRESH;
        c.record(None, t1);
        assert_eq!(c.value, None);
        assert!(c.needs_probe(t1 + CPU_TEMP_REFRESH));
    }

    #[test]
    fn reading_serializes_with_null_for_missing() {
        let json = serde_json::to_string(&Reading::default()).unwrap();
        assert_eq!(
            json,
            r#"{"gpu_temp_c":null,"gpu_fan_pct":null,"gpu_power_w":null,"cpu_temp_c":null}"#
        );
    }
}
