//! Machine-readable output (`--json`): one JSON line per event on stdout.
//!
//! The GUI starts the CLI binary as a child process and reads these lines.
//! The format is deliberately flat and stable - a field may be added, never
//! change meaning.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set by `--json`; decides whether output is JSON lines or human text.
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
    /// Startup: which backends and devices were found.
    Startup {
        version: String,
        backend: String,
        gpus: Vec<String>,
        cpu_threads: usize,
    },
    /// Connection state towards the pool.
    Status {
        state: StatusState,
        message: String,
    },
    /// Periodic statistics (same cadence as the text output).
    Stats {
        hashrate: f64,
        /// Split per backend (0 when that backend is not in use).
        hashrate_gpu: f64,
        hashrate_cpu: f64,
        accepted: u64,
        rejected: u64,
        /// Highest share difficulty reached during this run.
        best_share: f64,
        /// Number of blocks found during this run.
        blocks: u64,
        /// Expected time to a block in seconds; `null` before there is a
        /// hashrate.
        eta_secs: Option<f64>,
        network_difficulty: f64,
        /// The block height the pool's latest job applies to - shows that the
        /// client is in sync with the pool and mining on the right block.
        /// 0 = no job yet.
        job_height: u32,
        /// Temperature/power where the platform exposes it (else null).
        #[serde(flatten)]
        telemetry: crate::telemetry::Reading,
    },
    /// A share was submitted and the pool answered.
    Share { accepted: bool },
    /// We found a block (hash in display order).
    Block { hash: String },
    /// The pool moved on to a new block (height from the job's coinbase).
    NewBlockHeight { height: u32 },
    /// Answer to `--probe`: available hardware (the GUI asks before start).
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

/// Write an event as a JSON line (no-op when `--json` is not set).
pub fn emit(ev: &Event) {
    if !json_mode() {
        return;
    }
    if let Ok(line) = serde_json::to_string(ev) {
        println!("{line}");
    }
}

/// Write human-readable text - but only when JSON mode is off, so that
/// stdout stays pure JSON for the GUI.
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
            hashrate_gpu: 1.5,
            hashrate_cpu: 0.0,
            accepted: 3,
            rejected: 0,
            best_share: 1234.5,
            blocks: 0,
            eta_secs: None,
            network_difficulty: 42.0,
            job_height: 59_342,
            telemetry: crate::telemetry::Reading {
                gpu_temp_c: Some(64),
                ..Default::default()
            },
        })
        .unwrap();
        assert!(s.starts_with(r#"{"type":"stats","hashrate":1.5"#));
        assert!(s.contains(r#""eta_secs":null"#));
        // The telemetry is flattened into the same object.
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
