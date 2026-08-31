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
#[cfg(target_os = "windows")]
fn read_cpu_temp() -> Option<u32> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let out = std::process::Command::new("powershell")
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

    #[test]
    fn reading_serializes_with_null_for_missing() {
        let json = serde_json::to_string(&Reading::default()).unwrap();
        assert_eq!(
            json,
            r#"{"gpu_temp_c":null,"gpu_fan_pct":null,"gpu_power_w":null,"cpu_temp_c":null}"#
        );
    }
}
