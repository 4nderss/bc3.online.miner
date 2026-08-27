//! Maskinläsbar utdata (`--json`): en JSON-rad per händelse på stdout.
//!
//! GUI:t startar CLI-binären som barnprocess och läser dessa rader. Formatet
//! är avsiktligt platt och stabilt — ett fält får läggas till, aldrig byta
//! betydelse.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sätts av `--json`; styr om utdata är JSON-rader eller människotext.
static JSON_MODE: AtomicBool = AtomicBool::new(false);

pub fn set_json_mode(on: bool) {
    JSON_MODE.store(on, Ordering::Relaxed);
}

pub fn json_mode() -> bool {
    JSON_MODE.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Event {
    /// Uppstart: vilka backends och enheter som hittades.
    Startup {
        version: String,
        backend: String,
        gpus: Vec<String>,
        cpu_threads: usize,
    },
    /// Anslutningsläge mot poolen.
    Status {
        state: StatusState,
        message: String,
    },
    /// Periodisk statistik (samma takt som textutskriften).
    Stats {
        hashrate: f64,
        accepted: u64,
        rejected: u64,
        /// Förväntad tid till block i sekunder; `null` innan hashrate finns.
        eta_secs: Option<f64>,
        network_difficulty: f64,
        /// Temperatur/effekt där plattformen exponerar det (annars null).
        #[serde(flatten)]
        telemetry: crate::telemetry::Reading,
    },
    /// En share skickades in och poolen svarade.
    Share { accepted: bool },
    /// Vi hittade ett block (hash i display-ordning).
    Block { hash: String },
    /// Svar på `--probe`: tillgänglig hårdvara (GUI:t frågar innan start).
    Probe {
        gpus: Vec<String>,
        cpu_cores: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusState {
    Connecting,
    Authorized,
    Mining,
    Error,
}

/// Skriv en händelse som JSON-rad (no-op när `--json` inte är satt).
pub fn emit(ev: &Event) {
    if !json_mode() {
        return;
    }
    if let Ok(line) = serde_json::to_string(ev) {
        println!("{line}");
    }
}

/// Skriv människoläsbar text — men bara när JSON-läget är av, så att
/// stdout förblir ren JSON för GUI:t.
#[macro_export]
macro_rules! human {
    ($($arg:tt)*) => {
        if !$crate::ipc::json_mode() {
            println!($($arg)*);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_serialize_with_type_tag() {
        let s = serde_json::to_string(&Event::Share { accepted: true }).unwrap();
        assert_eq!(s, r#"{"type":"share","accepted":true}"#);

        let s = serde_json::to_string(&Event::Status {
            state: StatusState::Authorized,
            message: "ok".into(),
        })
        .unwrap();
        assert_eq!(s, r#"{"type":"status","state":"authorized","message":"ok"}"#);

        let s = serde_json::to_string(&Event::Stats {
            hashrate: 1.5,
            accepted: 3,
            rejected: 0,
            eta_secs: None,
            network_difficulty: 42.0,
            telemetry: crate::telemetry::Reading {
                gpu_temp_c: Some(64),
                ..Default::default()
            },
        })
        .unwrap();
        assert!(s.starts_with(r#"{"type":"stats","hashrate":1.5"#));
        assert!(s.contains(r#""eta_secs":null"#));
        // Telemetrin plattas ut i samma objekt.
        assert!(s.contains(r#""gpu_temp_c":64"#));
        assert!(s.contains(r#""cpu_temp_c":null"#));
    }

    #[test]
    fn startup_lists_devices() {
        let s = serde_json::to_string(&Event::Startup {
            version: "0.1.0".into(),
            backend: "cuda".into(),
            gpus: vec!["RTX 3050 Ti".into()],
            cpu_threads: 0,
        })
        .unwrap();
        assert!(s.contains(r#""type":"startup""#));
        assert!(s.contains(r#""gpus":["RTX 3050 Ti"]"#));
    }

    #[test]
    fn json_mode_toggles() {
        set_json_mode(false);
        assert!(!json_mode());
        set_json_mode(true);
        assert!(json_mode());
        set_json_mode(false);
    }
}
