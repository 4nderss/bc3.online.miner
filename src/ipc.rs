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
        /// CPU threads that only mine when the GPUs cannot (BC3Hash).
        cpu_fallback_threads: usize,
    },
    /// The pool's jobs moved to another PoW (and on the first job): which one,
    /// and where it is hashed - "gpu", "cpu", "gpu+cpu" or "none".
    Pow { algo: String, devices: String },
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
        /// Highest share difficulty reached during this run, with the
        /// current PoW. 0 before the first share.
        best_share: f64,
        /// The block difficulty of the job the best share was found on: what
        /// it would have needed to be a block. Not `network_difficulty` -
        /// with ASERT that moves every block. 0 before the first share.
        best_share_network_difficulty: f64,
        /// Seconds since the best share was found; `null` before one.
        best_share_age_secs: Option<f64>,
        /// Number of blocks found during this run.
        blocks: u64,
        /// Expected time to a block in seconds; `null` before there is a
        /// hashrate.
        eta_secs: Option<f64>,
        /// Difficulty of the block being mined, from the latest job's nBits.
        /// 0 before the first job.
        network_difficulty: f64,
        /// The block height the pool's latest job applies to - shows that the
        /// client is in sync with the pool and mining on the right block.
        /// 0 = no job yet.
        job_height: u32,
        /// Latest round trip of a request to the pool in milliseconds - a
        /// share submit, or the handshake before the first share. `null`
        /// until the pool has answered anything.
        pool_rtt_ms: Option<f64>,
        /// Temperature/power where the platform exposes it (else null).
        #[serde(flatten)]
        telemetry: crate::telemetry::Reading,
    },
    /// A share was submitted and the pool answered.
    Share { accepted: bool },
    /// We found a block (hash in display order).
    Block { hash: String },
    /// The pool moved on to a new block (height from the job's coinbase), and
    /// the difficulty that block needs (from the job's nBits) - so the two
    /// never show as a mismatched pair until the next stats.
    NewBlockHeight { height: u32, network_difficulty: f64 },
    /// The BC3Hash table window a GPU mines at (by slot), the memory its
    /// tables take, and whether it is still tuning it (src/tuning.rs). As a
    /// measurement ends: the window it tried, and how much faster than 16
    /// that hashed, in percent.
    Tables { gpu: usize, window: u32, mib: u64, tuning: bool, tried: Option<u32>, gain_pct: Option<f64> },
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
            best_share: 1.46,
            best_share_network_difficulty: 11.28,
            best_share_age_secs: Some(720.0),
            blocks: 0,
            eta_secs: None,
            network_difficulty: 11.31,
            job_height: 59_342,
            pool_rtt_ms: Some(34.5),
            telemetry: crate::telemetry::Reading {
                gpu_temp_c: Some(64),
                ..Default::default()
            },
        })
        .unwrap();
        assert!(s.starts_with(r#"{"type":"stats","hashrate":1.5"#));
        assert!(s.contains(r#""eta_secs":null"#));
        assert!(s.contains(r#""pool_rtt_ms":34.5"#));
        assert!(s.contains(
            r#""best_share":1.46,"best_share_network_difficulty":11.28,"best_share_age_secs":720.0"#
        ));
        assert!(s.contains(r#""network_difficulty":11.31"#));
        // The telemetry is flattened into the same object.
        assert!(s.contains(r#""gpu_temp_c":64"#));
        assert!(s.contains(r#""cpu_temp_c":null"#));
    }

    #[test]
    fn a_new_block_comes_with_its_difficulty() {
        let s = serde_json::to_string(&Event::NewBlockHeight {
            height: 64_368,
            network_difficulty: 11.28,
        })
        .unwrap();
        assert_eq!(s, r#"{"type":"newblockheight","height":64368,"network_difficulty":11.28}"#);
    }

    #[test]
    fn startup_lists_devices() {
        let s = serde_json::to_string(&Event::Startup {
            version: "0.1.0".into(),
            backend: "cuda".into(),
            gpus: vec!["RTX 3050 Ti".into()],
            cpu_threads: 0,
            cpu_fallback_threads: 20,
        })
        .unwrap();
        assert!(s.contains(r#""type":"startup""#));
        assert!(s.contains(r#""gpus":["RTX 3050 Ti"]"#));
        assert!(s.contains(r#""cpu_fallback_threads":20"#));
    }

    #[test]
    fn pow_event_names_the_algorithm_and_devices() {
        let s = serde_json::to_string(&Event::Pow {
            algo: "BC3Hash".into(),
            devices: "cpu".into(),
        })
        .unwrap();
        assert_eq!(s, r#"{"type":"pow","algo":"BC3Hash","devices":"cpu"}"#);
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
