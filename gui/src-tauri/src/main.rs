//! bc3.online miner - the GUI shell.
//!
//! The GUI does NOT run the mining code itself: it starts the CLI binary
//! (`bc3-miner`) as a child process with `--json` and streams its JSON lines
//! on to the frontend as Tauri events. The benefit is isolation - a GPU crash
//! never fells the window - and the mining core stays testable on its own.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Settings from the GUI when the user presses Start.
#[derive(Debug, Clone, Deserialize)]
pub struct StartOptions {
    /// BC3 address (bc1...).
    address: String,
    /// Rig name; empty => "gui".
    rig: String,
    /// "pplns" or "solo".
    mode: String,
    /// Optional pool override (host:port); empty => bc3.online with the mode's port.
    pool: String,
    /// "gpu", "cpu" or "dual".
    hardware: String,
    /// Intensity 1-100 %.
    intensity: Option<u32>,
}

/// Translate the hardware choice into CLI flags.
///
/// - gpu: `--backend auto` (the CLI starts no CPU threads when a GPU exists)
/// - cpu: `--backend cpu` + all cores
///
/// Note: combined GPU+CPU mining ("dual") is deliberately absent from the GUI -
/// measurement showed a lower total hashrate than GPU alone, since CPU threads
/// crowd out the GPU's feeder thread and share the power/heat budget with the
/// card. Anyone who still wants to try reaches it via the CLI's `--threads N`.
fn hardware_args(hardware: &str) -> Vec<String> {
    match hardware {
        "cpu" => vec!["--backend".into(), "cpu".into(), "--threads".into(), "0".into()],
        _ => vec!["--backend".into(), "auto".into()],
    }
}

/// The running miner process (None when stopped).
#[derive(Default)]
pub struct MinerState(Mutex<Option<CommandChild>>);

#[derive(Debug, Clone, Serialize)]
struct LogLine {
    text: String,
}

fn endpoint(opts: &StartOptions) -> String {
    if !opts.pool.trim().is_empty() {
        return opts.pool.trim().to_string();
    }
    match opts.mode.as_str() {
        "solo" => "bc3.online:3112".into(),
        _ => "bc3.online:3111".into(),
    }
}

/// Simple client-side validation - the pool is the final authority.
///
/// Both address families are accepted, because the pool pays either: bech32 and
/// bech32m (bc1..., segwit v0 and taproot) and legacy base58check (1... P2PKH,
/// 3... P2SH).
///
/// Requiring the bc1 prefix here was the bug behind the report that the miner
/// "doesn't let address other than bc1 to mine". v1.1.1 fixed the JavaScript
/// check and missed this one, so a legacy address passed the form, lit up the
/// Start button, and only then failed - which reads as a broken miner rather
/// than a rejected address.
fn validate_address(addr: &str) -> Result<(), String> {
    let incomplete = || Err("That doesn't look like a complete BC3 address".into());
    let a = addr.trim();
    if a.is_empty() {
        return Err("Enter your BC3 address".into());
    }
    // BIP173 permits an all-uppercase bech32 address, so match the prefix
    // case-insensitively. Length only: the pool decides the rest.
    let lower = a.to_lowercase();
    if lower.starts_with("bc1") {
        return if (23..=90).contains(&lower.len()) { Ok(()) } else { incomplete() };
    }
    // Base58 IS case-sensitive, so this one checks the original string. The
    // alphabet omits 0, O, I and l precisely because they are easy to confuse.
    if a.starts_with('1') || a.starts_with('3') {
        const B58: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        let ok = (26..=40).contains(&a.len()) && a.chars().all(|c| B58.contains(c));
        return if ok { Ok(()) } else { incomplete() };
    }
    Err("A BC3 address starts with bc1, 1 or 3".into())
}

#[tauri::command]
fn start_mining(
    app: AppHandle,
    state: State<'_, MinerState>,
    opts: StartOptions,
) -> Result<(), String> {
    validate_address(&opts.address)?;
    {
        let guard = state.0.lock().map_err(|_| "state poisoned")?;
        if guard.is_some() {
            return Err("Already mining".into());
        }
    }

    let rig = if opts.rig.trim().is_empty() { "gui" } else { opts.rig.trim() };
    let user = format!("{}.{}", opts.address.trim(), rig);
    let mut args = vec![
        "--pool".to_string(),
        endpoint(&opts),
        "--user".to_string(),
        user,
        "--json".to_string(),
        "--stats-interval".to_string(),
        "3".to_string(),
    ];
    args.extend(hardware_args(&opts.hardware));
    let intensity = opts.intensity.unwrap_or(100).clamp(1, 100);
    args.push("--intensity".into());
    args.push(intensity.to_string());

    let sidecar = app
        .shell()
        .sidecar("bc3-miner")
        .map_err(|e| format!("miner binary not found: {e}"))?
        .args(args);
    let (mut rx, child) = sidecar
        .spawn()
        .map_err(|e| format!("could not start the miner: {e}"))?;

    *state.0.lock().map_err(|_| "state poisoned")? = Some(child);

    // Stream the child's output on to the frontend.
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    // JSON lines are forwarded as "miner-event"; everything
                    // else (e.g. panics) ends up in the log panel.
                    match serde_json::from_str::<serde_json::Value>(&line) {
                        Ok(v) => {
                            let _ = handle.emit("miner-event", v);
                        }
                        Err(_) => {
                            let _ = handle.emit("miner-log", LogLine { text: line });
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).trim().to_string();
                    if !text.is_empty() {
                        let _ = handle.emit("miner-log", LogLine { text });
                    }
                }
                CommandEvent::Terminated(payload) => {
                    let _ = handle.emit("miner-stopped", payload.code);
                    if let Some(state) = handle.try_state::<MinerState>() {
                        if let Ok(mut guard) = state.0.lock() {
                            *guard = None;
                        }
                    }
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn stop_mining(state: State<'_, MinerState>) -> Result<(), String> {
    let child = state.0.lock().map_err(|_| "state poisoned")?.take();
    if let Some(child) = child {
        child.kill().map_err(|e| format!("could not stop the miner: {e}"))?;
    }
    Ok(())
}

#[tauri::command]
fn is_mining(state: State<'_, MinerState>) -> bool {
    state.0.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Ask the CLI which GPUs exist (a short run with `--probe`), so that the
/// hardware buttons can show the card's name before you start.
#[tauri::command]
async fn probe_hardware(app: AppHandle) -> Result<serde_json::Value, String> {
    let output = app
        .shell()
        .sidecar("bc3-miner")
        .map_err(|e| format!("miner binary not found: {e}"))?
        .args(["--probe", "--json"])
        .output()
        .await
        .map_err(|e| format!("probe failed: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if v["type"] == "probe" {
                return Ok(v);
            }
        }
    }
    Err("probe produced no result".into())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(MinerState::default())
        .invoke_handler(tauri::generate_handler![
            start_mining,
            stop_mining,
            is_mining,
            probe_hardware
        ])
        .run(tauri::generate_context!())
        .expect("kunde inte starta bc3.online miner");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(mode: &str, pool: &str) -> StartOptions {
        StartOptions {
            address: "bc1qerclmenj87gc2r3hfeyd0v0rxze32pnxygnt6p".into(),
            rig: "rig".into(),
            mode: mode.into(),
            pool: pool.into(),
            hardware: "gpu".into(),
            intensity: None,
        }
    }

    #[test]
    fn hardware_maps_to_cli_flags() {
        assert_eq!(hardware_args("gpu"), vec!["--backend", "auto"]);
        assert_eq!(
            hardware_args("cpu"),
            vec!["--backend", "cpu", "--threads", "0"]
        );
        // Unknown value (incl. an old saved "dual") => the GPU behavior.
        assert_eq!(hardware_args("dual"), vec!["--backend", "auto"]);
        assert_eq!(hardware_args("nonsense"), vec!["--backend", "auto"]);
    }

    #[test]
    fn endpoint_defaults_per_mode() {
        assert_eq!(endpoint(&opts("solo", "")), "bc3.online:3112");
        assert_eq!(endpoint(&opts("pplns", "")), "bc3.online:3111");
        // The override always wins.
        assert_eq!(endpoint(&opts("solo", "127.0.0.1:13112")), "127.0.0.1:13112");
    }

    #[test]
    fn address_validation() {
        assert!(validate_address("bc1qerclmenj87gc2r3hfeyd0v0rxze32pnxygnt6p").is_ok());
        // Uppercase is valid in bech32.
        assert!(validate_address("BC1QERCLMENJ87GC2R3HFEYD0V0RXZE32PNXYGNT6P").is_ok());
        // Legacy base58check. Rejecting these was the v1.1.1 bug: the pool pays
        // them, and the CLI has never refused them.
        assert!(validate_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa").is_ok());
        assert!(validate_address("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy").is_ok());
        // Base58 is case-sensitive and has no 0, O, I or l.
        assert!(validate_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfN0").is_err());
        assert!(validate_address("").is_err());
        assert!(validate_address("bc1short").is_err());
        assert!(validate_address("1short").is_err());
        assert!(validate_address("xyz1qerclmenj87gc2r3hfeyd0v0rxze32pnxygnt6p").is_err());
    }
}
