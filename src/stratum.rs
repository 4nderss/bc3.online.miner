//! Stratum v1-klient: håller anslutningen till poolen, tar emot jobb och
//! skickar in funna shares. Återansluter med backoff vid avbrott.

use crate::consensus::{swab32, target_for_difficulty};
use crate::shared::{FoundShare, MinerJob, Shared};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::TcpStream;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

pub struct StratumConfig {
    pub pool: String,
    pub user: String,
}

pub fn run_client(shared: Arc<Shared>, submit_rx: Receiver<FoundShare>, cfg: StratumConfig) {
    let mut backoff = 1u64;
    loop {
        match session(&shared, &submit_rx, &cfg) {
            Ok(()) => backoff = 1,
            Err(e) => {
                eprintln!("[pool] anslutningsfel: {e} — återansluter om {backoff}s");
                crate::ipc::emit(&crate::ipc::Event::Status {
                    state: crate::ipc::StatusState::Connecting,
                    message: format!("connection lost, retrying in {backoff}s"),
                });
                shared.clear_job();
                std::thread::sleep(Duration::from_secs(backoff));
                backoff = (backoff * 2).min(60);
            }
        }
    }
}

fn session(
    shared: &Arc<Shared>,
    submit_rx: &Receiver<FoundShare>,
    cfg: &StratumConfig,
) -> std::io::Result<()> {
    crate::human!("[pool] ansluter till {} som {}", cfg.pool, cfg.user);
    crate::ipc::emit(&crate::ipc::Event::Status {
        state: crate::ipc::StatusState::Connecting,
        message: format!("connecting to {}", cfg.pool),
    });
    let stream = TcpStream::connect(&cfg.pool)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_millis(200)))?;
    let mut writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);

    let send = |w: &mut TcpStream, v: Value| -> std::io::Result<()> {
        w.write_all((v.to_string() + "\n").as_bytes())
    };
    send(&mut writer, json!({"id": 1, "method": "mining.subscribe",
        "params": [format!("bc3-miner/{}", env!("CARGO_PKG_VERSION"))]}))?;
    send(&mut writer, json!({"id": 2, "method": "mining.authorize",
        "params": [cfg.user, "x"]}))?;

    let mut extranonce1: Vec<u8> = vec![];
    let mut extranonce2_size = 4usize;
    let mut difficulty = 1.0f64;
    let mut next_submit_id: u64 = 100;
    let mut line = String::new();

    loop {
        // 1) Skicka in väntande shares.
        while let Ok(share) = submit_rx.try_recv() {
            if share.is_block_candidate {
                    crate::human!("[MINER] ★ BLOCKKANDIDAT {} ★", share.hash_display);
                crate::ipc::emit(&crate::ipc::Event::Block {
                    hash: share.hash_display.clone(),
                });
            }
            send(&mut writer, json!({"id": next_submit_id, "method": "mining.submit", "params": [
                cfg.user, share.job_id, hex::encode(&share.extranonce2),
                format!("{:08x}", share.ntime), format!("{:08x}", share.nonce),
            ]}))?;
            next_submit_id += 1;
        }

        // 2) Läs ev. rad från poolen (timeout 200 ms håller loopen igång).
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    ErrorKind::ConnectionAborted,
                    "poolen stängde anslutningen",
                ))
            }
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                continue
            }
            Err(e) => return Err(e),
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };

        if msg["id"] == json!(1) {
            let r = &msg["result"];
            if r.is_null() {
                return Err(std::io::Error::new(ErrorKind::InvalidData, "subscribe avvisades"));
            }
            extranonce1 = hex::decode(r[1].as_str().unwrap_or("")).unwrap_or_default();
            extranonce2_size = r[2].as_u64().unwrap_or(4) as usize;
            continue;
        }
        if msg["id"] == json!(2) {
            if msg["result"] != json!(true) {
                eprintln!("[pool] AUKTORISERING AVVISAD: {}", msg["error"]);
                crate::ipc::emit(&crate::ipc::Event::Status {
                    state: crate::ipc::StatusState::Error,
                    message: format!("pool rejected the address: {}", msg["error"]),
                });
                return Err(std::io::Error::new(ErrorKind::PermissionDenied, "authorize"));
            }
            crate::human!("[pool] auktoriserad");
            crate::ipc::emit(&crate::ipc::Event::Status {
                state: crate::ipc::StatusState::Mining,
                message: "authorized — mining".into(),
            });
            continue;
        }
        // Submit-svar.
        if let Some(id) = msg["id"].as_u64() {
            if id >= 100 {
                use std::sync::atomic::Ordering;
                let accepted = msg["result"] == json!(true);
                if accepted {
                    shared.stats.accepted.fetch_add(1, Ordering::Relaxed);
                } else {
                    shared.stats.rejected.fetch_add(1, Ordering::Relaxed);
                    eprintln!("[pool] share avvisad: {}", msg["error"]);
                }
                crate::ipc::emit(&crate::ipc::Event::Share { accepted });
                continue;
            }
        }

        match msg["method"].as_str() {
            Some("mining.set_difficulty") => {
                difficulty = msg["params"][0].as_f64().unwrap_or(1.0);
            }
            Some("mining.notify") => {
                let p = msg["params"].as_array().cloned().unwrap_or_default();
                if p.len() < 9 || extranonce1.is_empty() {
                    continue;
                }
                match parse_notify(&p, &extranonce1, extranonce2_size, difficulty) {
                    Some(job) => shared.publish_job(job),
                    None => eprintln!("[pool] ogiltigt notify-meddelande"),
                }
            }
            Some("client.reconnect") => {
                return Err(std::io::Error::new(ErrorKind::ConnectionReset, "reconnect begärd"));
            }
            _ => {}
        }
    }
}

fn parse_notify(
    p: &[Value],
    extranonce1: &[u8],
    extranonce2_size: usize,
    difficulty: f64,
) -> Option<MinerJob> {
    let prev_swab: [u8; 32] = hex::decode(p[1].as_str()?).ok()?.try_into().ok()?;
    let steps = p[4]
        .as_array()?
        .iter()
        .map(|s| {
            hex::decode(s.as_str().unwrap_or(""))
                .ok()
                .and_then(|v| <[u8; 32]>::try_from(v).ok())
        })
        .collect::<Option<Vec<_>>>()?;
    Some(MinerJob {
        job_id: p[0].as_str()?.to_string(),
        extranonce1: extranonce1.to_vec(),
        extranonce2_size,
        coinb1: hex::decode(p[2].as_str()?).ok()?,
        coinb2: hex::decode(p[3].as_str()?).ok()?,
        merkle_steps: steps,
        version: u32::from_str_radix(p[5].as_str()?, 16).ok()?,
        prev_hash: swab32(&prev_swab),
        bits: u32::from_str_radix(p[6].as_str()?, 16).ok()?,
        ntime: u32::from_str_radix(p[7].as_str()?, 16).ok()?,
        share_target: target_for_difficulty(difficulty),
    })
}
