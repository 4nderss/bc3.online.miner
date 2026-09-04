//! Stratum v1 client: keeps the connection to the pool, receives jobs and
//! submits found shares. Reconnects with backoff when the link drops.

use crate::consensus::{swab32, target_for_difficulty};
use crate::shared::{FoundShare, MinerJob, Shared};
use serde_json::{json, Value};
use std::io::{BufReader, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Longest line we accept from the pool.
///
/// A pool that streams bytes without ever sending a newline would otherwise
/// grow this buffer without bound. Real messages are a few kB; the largest is
/// a PPLNS `mining.notify`, whose coinbase carries one output per window
/// participant - about 32 kB at the pool's 500-payout cap.
const MAX_LINE_BYTES: usize = 64 * 1024;

/// Silence longer than this means the link is dead, whatever TCP thinks.
///
/// A half-open connection - the pool restarted, a NAT table expired, the rig
/// moved networks - never produces an error or an EOF. The read simply times
/// out for ever while the miner keeps hashing the last job it was given,
/// submitting into a socket that goes nowhere and showing "Mining" the whole
/// time. The pool sends a job at least every 20 seconds, so silence this long
/// is not something a healthy link does.
const POOL_SILENCE_TIMEOUT: Duration = Duration::from_secs(90);

/// Read one complete line, keeping any partial one in `buf` between calls.
/// `Ok(None)` means nothing complete has arrived yet.
///
/// The partial buffer is the point. `BufRead::read_line` appends what it has
/// read and then, when the socket's read timeout fires mid-message, returns
/// an error - and the bytes it had already taken off the socket are gone with
/// the discarded String. One retransmit inside a multi-segment `mining.notify`
/// was enough: the tail arrived alone, failed to parse, and the rig carried on
/// mining the previous block until the next job, with every share rejected.
fn next_line<R: Read>(reader: &mut R, buf: &mut Vec<u8>) -> std::io::Result<Option<String>> {
    loop {
        if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = buf.drain(..=pos).collect();
            line.pop(); // the newline itself
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
        }
        if buf.len() > MAX_LINE_BYTES {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "the pool sent a line longer than we accept",
            ));
        }
        let mut chunk = [0u8; 8192];
        match reader.read(&mut chunk) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    ErrorKind::ConnectionAborted,
                    "pool closed the connection",
                ))
            }
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                return Ok(None)
            }
            Err(e) => return Err(e),
        }
    }
}

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
                eprintln!("[pool] connection error: {e} - reconnecting in {backoff}s");
                crate::ipc::emit(&crate::ipc::Event::Status {
                    state: crate::ipc::StatusState::Connecting,
                    message: format!("connection lost, retrying in {backoff}s"),
                });
                shared.clear_job();
                // Drop shares that were queued but never sent.
                //
                // They were found under the old connection's extranonce1,
                // which goes into the coinbase. After reconnecting we get a
                // new one, so the pool rebuilds a different coinbase, gets a
                // different hash, and rejects them as below target - a
                // handful of confusing rejections per reconnect, for work
                // that can no longer be claimed under any job.
                let dropped = submit_rx.try_iter().count();
                if dropped > 0 {
                    crate::human!("[pool] dropped {dropped} queued share(s) from the previous connection");
                }
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
    crate::human!("[pool] connecting to {} as {}", cfg.pool, cfg.user);
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
    // Kept across reads so a message split by a timeout is not lost.
    let mut rx_buf: Vec<u8> = Vec::with_capacity(8192);
    let mut last_rx = Instant::now();

    loop {
        // 1) Submit pending shares.
        while let Ok(share) = submit_rx.try_recv() {
            if share.is_block_candidate {
                crate::human!("[miner] * BLOCK CANDIDATE {} *", share.hash_display);
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

        // 2) Read any line from the pool (the 200 ms timeout keeps the loop
        // running).
        let line = match next_line(&mut reader, &mut rx_buf)? {
            Some(line) => {
                last_rx = Instant::now();
                line
            }
            None => {
                if last_rx.elapsed() > POOL_SILENCE_TIMEOUT {
                    return Err(std::io::Error::new(
                        ErrorKind::TimedOut,
                        format!(
                            "no word from the pool in {} s - the connection is dead",
                            POOL_SILENCE_TIMEOUT.as_secs()
                        ),
                    ));
                }
                continue;
            }
        };
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };

        if msg["id"] == json!(1) {
            let r = &msg["result"];
            if r.is_null() {
                return Err(std::io::Error::new(ErrorKind::InvalidData, "subscribe rejected"));
            }
            extranonce1 = hex::decode(r[1].as_str().unwrap_or("")).unwrap_or_default();
            // Clamp what the pool asks for. Zero would give every worker the
            // same (empty) extranonce2, collapsing the partitioning so all of
            // them grind identical headers; a huge value makes
            // `encode_extranonce2` allocate that many bytes per attempt and
            // abort the process. Eight bytes is already more space than any
            // rig can search.
            extranonce2_size = r[2].as_u64().unwrap_or(4) as usize;
            if !(1..=16).contains(&extranonce2_size) {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("pool asked for an extranonce2 of {extranonce2_size} bytes"),
                ));
            }
            continue;
        }
        if msg["id"] == json!(2) {
            if msg["result"] != json!(true) {
                eprintln!("[pool] AUTHORIZATION REJECTED: {}", msg["error"]);
                crate::ipc::emit(&crate::ipc::Event::Status {
                    state: crate::ipc::StatusState::Error,
                    message: format!("pool rejected the address: {}", msg["error"]),
                });
                return Err(std::io::Error::new(ErrorKind::PermissionDenied, "authorize"));
            }
            crate::human!("[pool] authorized");
            crate::ipc::emit(&crate::ipc::Event::Status {
                state: crate::ipc::StatusState::Mining,
                message: "authorized - mining".into(),
            });
            continue;
        }
        // Submit responses.
        if let Some(id) = msg["id"].as_u64() {
            if id >= 100 {
                use std::sync::atomic::Ordering;
                let accepted = msg["result"] == json!(true);
                if accepted {
                    shared.stats.accepted.fetch_add(1, Ordering::Relaxed);
                } else {
                    shared.stats.rejected.fetch_add(1, Ordering::Relaxed);
                    eprintln!("[pool] share rejected: {}", msg["error"]);
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
                    None => eprintln!("[pool] invalid notify message"),
                }
            }
            Some("client.reconnect") => {
                return Err(std::io::Error::new(ErrorKind::ConnectionReset, "reconnect requested"));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader that hands out pre-baked chunks and reports WouldBlock between
    /// them - the shape a socket with a read timeout actually has.
    struct Chunks(Vec<Option<&'static [u8]>>);

    impl Read for Chunks {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            match self.0.first().copied() {
                // None models the read timeout firing.
                Some(None) => {
                    self.0.remove(0);
                    Err(std::io::Error::new(ErrorKind::WouldBlock, "timeout"))
                }
                Some(Some(data)) => {
                    self.0.remove(0);
                    let n = data.len().min(out.len());
                    out[..n].copy_from_slice(&data[..n]);
                    Ok(n)
                }
                None => Ok(0), // EOF
            }
        }
    }

    /// The bug this function exists for: a message split across a read timeout
    /// must survive it. `read_line` dropped the bytes it had already taken off
    /// the socket, so the tail arrived alone and failed to parse - and a lost
    /// `mining.notify` means the rig keeps mining the previous block.
    #[test]
    fn a_line_split_by_a_timeout_is_not_lost() {
        let mut r = Chunks(vec![
            Some(b"{\"id\":1,\"met"),
            None, // timeout in the middle of the message
            Some(b"hod\":\"mining.notify\"}\n"),
        ]);
        let mut buf = Vec::new();

        // First call: only half of it has arrived.
        assert_eq!(next_line(&mut r, &mut buf).unwrap(), None);
        // Second call: the rest arrives and the WHOLE line comes back.
        assert_eq!(
            next_line(&mut r, &mut buf).unwrap().as_deref(),
            Some("{\"id\":1,\"method\":\"mining.notify\"}")
        );
        assert!(buf.is_empty());
    }

    /// Several lines in one packet must all be delivered, one per call.
    #[test]
    fn a_packet_with_several_lines_yields_them_all() {
        let mut r = Chunks(vec![Some(b"{\"a\":1}\n{\"b\":2}\r\n{\"c\":3}\n")]);
        let mut buf = Vec::new();
        for want in ["{\"a\":1}", "{\"b\":2}", "{\"c\":3}"] {
            assert_eq!(next_line(&mut r, &mut buf).unwrap().as_deref(), Some(want));
        }
        // CRLF must not leave a stray carriage return behind.
        assert!(buf.is_empty());
    }

    /// A pool that never sends a newline must not be able to grow the buffer
    /// without bound.
    #[test]
    fn an_endless_line_is_refused() {
        let filler: &'static [u8] = &[b'x'; 8192];
        let mut r = Chunks(vec![Some(filler); MAX_LINE_BYTES / 8192 + 2]);
        let mut buf = Vec::new();
        let err = loop {
            match next_line(&mut r, &mut buf) {
                Ok(Some(_)) => panic!("there is no newline in the stream"),
                Ok(None) => panic!("the reader ran out before the cap was hit"),
                Err(e) => break e,
            }
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(buf.len() <= MAX_LINE_BYTES + 8192);
    }

    /// A closed socket is an error, not an endless stream of empty lines.
    #[test]
    fn eof_ends_the_session() {
        let mut r = Chunks(vec![]);
        let mut buf = Vec::new();
        let err = next_line(&mut r, &mut buf).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConnectionAborted);
    }
}
