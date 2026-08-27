//! Temperaturavläsning för GPU och CPU.
//!
//! GPU: NVML (`nvml.dll` / `libnvidia-ml.so`) följer med NVIDIA-drivrutinen,
//! så inget toolkit behövs — samma princip som PTX-lösningen för kerneln.
//! Biblioteket laddas dynamiskt och saknad NVML ger bara `None`.
//!
//! CPU: ingen portabel API finns. Linux har hwmon i sysfs; på Windows kräver
//! riktig kärntemperatur en signerad drivrutin (LibreHardwareMonitor m.fl.),
//! vilket vi inte vill bunta med en miner. Vi läser därför det som går utan
//! extra rättigheter och visar annars "—".

#[cfg(feature = "cuda")]
use nvml_wrapper::Nvml;

/// Öppnad telemetrikälla. `None`-fälten betyder "inte tillgängligt här".
pub struct Telemetry {
    #[cfg(feature = "cuda")]
    nvml: Option<Nvml>,
}

/// En avläsning; alla fält kan saknas beroende på plattform/hårdvara.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Reading {
    pub gpu_temp_c: Option<u32>,
    pub gpu_fan_pct: Option<u32>,
    pub gpu_power_w: Option<f64>,
    pub cpu_temp_c: Option<u32>,
}

impl Telemetry {
    /// Öppna tillgängliga källor. NVML-laddaren kan panika om biblioteket
    /// saknas — samma skydd som i CUDA-detekteringen.
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

    /// Läs aktuella värden. Fel svälјs — telemetri får aldrig störa mining.
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
                // NVML rapporterar milliwatt.
                r.gpu_power_w = dev.power_usage().ok().map(|mw| mw as f64 / 1000.0);
            }
        }
        let _ = gpu_index;
        r
    }
}

/// CPU-temperatur där plattformen exponerar den utan extra rättigheter.
#[cfg(target_os = "linux")]
fn read_cpu_temp() -> Option<u32> {
    // Leta upp ett hwmon som rapporterar paket-/kärntemperatur.
    let dir = std::fs::read_dir("/sys/class/hwmon").ok()?;
    for entry in dir.flatten() {
        let base = entry.path();
        let name = std::fs::read_to_string(base.join("name")).unwrap_or_default();
        let name = name.trim();
        if !matches!(name, "coretemp" | "k10temp" | "zenpower" | "cpu_thermal") {
            continue;
        }
        // temp1_input är paketet på coretemp/k10temp; värdet är millicelsius.
        if let Ok(v) = std::fs::read_to_string(base.join("temp1_input")) {
            if let Ok(milli) = v.trim().parse::<i64>() {
                return u32::try_from(milli / 1000).ok();
            }
        }
    }
    None
}

/// Windows: `MSAcpi_ThermalZoneTemperature` är den enda källan som inte
/// kräver en egen kärndrivrutin. Många moderkort rapporterar inget alls där —
/// då blir svaret `None` och GUI:t visar "—".
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
    // Värdet är tiondels kelvin.
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
        // Utan NVIDIA-drivrutin ska detta ge en tom källa, inte krascha.
        let t = Telemetry::open();
        let r = t.read(0);
        // Inga krav på värden — bara att anropet överlever.
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
