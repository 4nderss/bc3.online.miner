//! bc3.online miner - the GUI shell.
//!
//! The GUI does NOT run the mining code itself: it starts the CLI binary
//! (`bc3-miner`) as a child process with `--json` and streams its JSON lines
//! on to the frontend as Tauri events. The benefit is isolation - a GPU crash
//! never fells the window - and the mining core stays testable on its own.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
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
    const INCOMPLETE: &str = "That doesn't look like a complete BC3 address";
    let a = addr.trim();
    if a.is_empty() {
        return Err("Enter your BC3 address".into());
    }
    // BIP173 permits an all-uppercase bech32 address, so match the prefix
    // case-insensitively. Length only: the pool decides the rest.
    let lower = a.to_lowercase();
    if lower.starts_with("bc1") {
        if lower.len() < 23 || lower.len() > 90 {
            return Err(INCOMPLETE.into());
        }
        return Ok(());
    }
    // Base58 IS case-sensitive, so this one checks the original string. The
    // alphabet omits 0, O, I and l precisely because they are easy to confuse.
    if a.starts_with('1') || a.starts_with('3') {
        const B58: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        if a.len() < 26 || a.len() > 40 || !a.chars().all(|c| B58.contains(c)) {
            return Err(INCOMPLETE.into());
        }
        return Ok(());
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

/// The GUI's settings - address, rig name, mode, hardware, intensity - as a
/// JSON file in the app's local data folder (`%LOCALAPPDATA%\online.bc3.miner`
/// on Windows), next to the WebView's own data.
///
/// They used to live in the WebView's localStorage. Chromium writes that to
/// disk a few seconds after a change, so a kill or a crash in between lost
/// the change, and a failed write was never reported. On one machine a new
/// address reverted to the old one on every start until the whole folder was
/// deleted. This file is written at once, and a failure reaches the log.
fn settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("no app data folder: {e}"))?;
    Ok(dir.join("settings.json"))
}

/// The saved settings, or `None` before anything has been saved.
#[tauri::command]
fn load_settings(app: AppHandle) -> Result<Option<serde_json::Value>, String> {
    read_settings(&settings_path(&app)?)
}

/// A plain (not async) command: Tauri runs those on the main thread, one at
/// a time, so two saves never write the temporary file at once. The window
/// sends one save at a time and the newest last (saveSettings in main.js).
#[tauri::command]
fn save_settings(app: AppHandle, settings: serde_json::Value) -> Result<(), String> {
    write_settings(&settings_path(&app)?, &settings)
}

fn read_settings(path: &Path) -> Result<Option<serde_json::Value>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(v) if v.is_object() => Ok(Some(v)),
        Ok(_) => Err(format!("{} does not hold settings", path.display())),
        Err(e) => Err(format!("{} is damaged: {e}", path.display())),
    }
}

/// Written to a temporary file, flushed, and renamed over the old one: a
/// crash or power loss mid-write leaves the previous settings, never half a
/// file.
fn write_settings(path: &Path, settings: &serde_json::Value) -> Result<(), String> {
    if !settings.is_object() {
        return Err("settings must be a JSON object".into());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    }
    let body = serde_json::to_vec_pretty(settings).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&body)?;
        f.sync_all()
    };
    write().map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("could not save {}: {e}", path.display()))
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
            probe_hardware,
            load_settings,
            save_settings
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

    /// A folder of its own under the system temp dir, removed again.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bc3-gui-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn settings_round_trip() {
        let dir = scratch("settings");
        // The folder does not exist yet: nothing saved, and a save creates it.
        let path = dir.join("settings.json");
        assert_eq!(read_settings(&path), Ok(None));
        let first = serde_json::json!({ "address": "bc1qold", "rig": "a", "mode": "pplns" });
        write_settings(&path, &first).unwrap();
        assert_eq!(read_settings(&path), Ok(Some(first)));
        // A second save replaces the first, and leaves no temporary file.
        let second = serde_json::json!({ "address": "bc1qnew", "rig": "b", "mode": "solo" });
        write_settings(&path, &second).unwrap();
        assert_eq!(read_settings(&path), Ok(Some(second)));
        assert!(!path.with_extension("json.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn damaged_settings_are_an_error_not_defaults() {
        let dir = scratch("damaged");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        std::fs::write(&path, "{\"address\": \"bc1q").unwrap();
        assert!(read_settings(&path).unwrap_err().contains("damaged"));
        std::fs::write(&path, "[1, 2]").unwrap();
        assert!(read_settings(&path).is_err());
        assert!(write_settings(&path, &serde_json::json!("text")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
