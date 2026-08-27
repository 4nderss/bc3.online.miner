//! bc3.online miner — GUI-skalet.
//!
//! GUI:t kör INTE mining-koden själv: det startar CLI-binären (`bc3-miner`)
//! som barnprocess med `--json` och strömmar dess JSON-rader vidare till
//! frontend som Tauri-events. Fördelen är isolering — en GPU-krasch fäller
//! aldrig fönstret — och att mining-kärnan förblir testbar för sig.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Inställningar från GUI:t när användaren trycker Start.
#[derive(Debug, Clone, Deserialize)]
pub struct StartOptions {
    /// BC3-adress (bc1...).
    address: String,
    /// Riggnamn; tomt ⇒ "gui".
    rig: String,
    /// "pplns" eller "solo".
    mode: String,
    /// Valfri pool-override (host:port); tomt ⇒ bc3.online med lägets port.
    pool: String,
    /// "gpu", "cpu" eller "dual".
    hardware: String,
    /// Intensitet 1–100 %.
    intensity: Option<u32>,
}

/// Översätt hårdvaruvalet till CLI-flaggor.
///
/// - gpu: `--backend auto` (CLI:n startar inga CPU-trådar när en GPU finns)
/// - cpu: `--backend cpu` + alla kärnor
///
/// Obs: kombinerad GPU+CPU-mining ("dual") finns medvetet inte i GUI:t —
/// mätning visade lägre total hashrate än enbart GPU, eftersom CPU-trådarna
/// konkurrerar ut GPU:ns matartråd och delar effekt/värmebudget med kortet.
/// Den som ändå vill prova når det via CLI:n med `--threads N`.
fn hardware_args(hardware: &str) -> Vec<String> {
    match hardware {
        "cpu" => vec!["--backend".into(), "cpu".into(), "--threads".into(), "0".into()],
        _ => vec!["--backend".into(), "auto".into()],
    }
}

/// Den körande minerprocessen (None när stoppad).
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

/// Enkel klientvalidering — poolen är den slutgiltiga auktoriteten.
fn validate_address(addr: &str) -> Result<(), String> {
    let a = addr.trim();
    if a.is_empty() {
        return Err("Enter your BC3 address".into());
    }
    let lower = a.to_lowercase();
    if !lower.starts_with("bc1") {
        return Err("A BC3 address starts with bc1…".into());
    }
    if lower.len() < 26 || lower.len() > 90 {
        return Err("That doesn't look like a complete BC3 address".into());
    }
    Ok(())
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

    // Strömma barnets utdata vidare till frontend.
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    // JSON-rader vidarebefordras som "miner-event"; allt annat
                    // (t.ex. panics) hamnar i loggpanelen.
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

/// Fråga CLI:n vilka GPU:er som finns (kort körning med `--probe`), så att
/// hårdvaruknapparna kan visa kortets namn innan man startar.
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
        // Okänt värde (inkl. gammalt sparat "dual") ⇒ GPU-beteendet.
        assert_eq!(hardware_args("dual"), vec!["--backend", "auto"]);
        assert_eq!(hardware_args("nonsense"), vec!["--backend", "auto"]);
    }

    #[test]
    fn endpoint_defaults_per_mode() {
        assert_eq!(endpoint(&opts("solo", "")), "bc3.online:3112");
        assert_eq!(endpoint(&opts("pplns", "")), "bc3.online:3111");
        // Override vinner alltid.
        assert_eq!(endpoint(&opts("solo", "127.0.0.1:13112")), "127.0.0.1:13112");
    }

    #[test]
    fn address_validation() {
        assert!(validate_address("bc1qerclmenj87gc2r3hfeyd0v0rxze32pnxygnt6p").is_ok());
        // Versaler är giltiga i bech32.
        assert!(validate_address("BC1QERCLMENJ87GC2R3HFEYD0V0RXZE32PNXYGNT6P").is_ok());
        assert!(validate_address("").is_err());
        assert!(validate_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa").is_err());
        assert!(validate_address("bc1short").is_err());
    }
}
