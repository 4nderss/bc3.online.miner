//! Stratum v1 client: keeps the connection to the pool, receives jobs and
//! submits found shares. Reconnects with backoff when the link drops.

use crate::consensus::{bip34_height, swab32, target_for_difficulty, PowAlgo};
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
    /// The share difficulty to ask the pool to start at, if any - see
    /// `suggested_difficulty`.
    pub suggest_difficulty: Option<f64>,
}

/// BC3Hash share difficulty to suggest per CUDA card. A pool that has no
/// history of the rig starts it where a browser or a CPU belongs: bc3.online
/// at 2e-6, which an RTX 4090 at 2.5 MH/s beats ~290 times a second. The pool
/// takes 40 shares in 10 s and raises the difficulty x16 per 12 of them, so
/// what the card had found at the old difficulty meanwhile, its queued
/// launches included, came back "too many shares per second" - 155 rejects in
/// the first ten seconds on an RTX 4090. At 1e-4 the RTX 4090 starts at ~6
/// shares a second, and the pool has raised it after two seconds, long
/// before its limit; a card of 20 kH/s, slower than any NVIDIA card the
/// kernel is likely to meet, gets a share every ~20 s, near the pool's
/// target of 15.
const SUGGESTED_DIFFICULTY_PER_CARD: f64 = 1e-4;

/// The share difficulty to suggest to the pool (`mining.suggest_difficulty`)
/// for `cuda_cards` NVIDIA cards mining at `intensity` percent. None without
/// a CUDA card: a CPU belongs where the pool starts anyway, and an OpenCL
/// device may be an integrated GPU of a few kH/s, which a suggestion made
/// for discrete cards would leave without a share for minutes.
pub fn suggested_difficulty(cuda_cards: usize, intensity: u32) -> Option<f64> {
    (cuda_cards > 0).then(|| {
        SUGGESTED_DIFFICULTY_PER_CARD * cuda_cards as f64 * f64::from(intensity.clamp(1, 100)) / 100.0
    })
}

/// Requests the pool has not answered yet, so that its answer can be timed.
///
/// The round trip of a share submit is the one latency a miner operator
/// cares about - it is the time a share is exposed to going stale - so that
/// is what the "pool" figure in the stats shows. Subscribe and authorize are
/// timed too, so the figure is there from the first second rather than after
/// the first share. Requests the pool never answers are forgotten after a
/// while; a pool that silently drops submits must not grow this for ever.
struct Outstanding {
    sent: std::collections::VecDeque<(u64, Instant)>,
}

const OUTSTANDING_MAX_AGE: Duration = Duration::from_secs(120);

impl Outstanding {
    fn new() -> Self {
        Self { sent: std::collections::VecDeque::new() }
    }

    fn sent(&mut self, id: u64, now: Instant) {
        while let Some((_, t)) = self.sent.front() {
            if now.duration_since(*t) > OUTSTANDING_MAX_AGE {
                self.sent.pop_front();
            } else {
                break;
            }
        }
        self.sent.push_back((id, now));
    }

    /// The round trip for `id`, or None if it was never sent (or already
    /// answered, or forgotten).
    fn answered(&mut self, id: u64, now: Instant) -> Option<Duration> {
        let pos = self.sent.iter().position(|(i, _)| *i == id)?;
        let (_, t) = self.sent.remove(pos)?;
        Some(now.duration_since(t))
    }
}

/// A reject with code 21, stratum's "job not found": the pool calls a share
/// stale when the work it was found on was over before it arrived - a new
/// block, mostly ("job not found (stale)", "stale: the proof of work has
/// changed", "job is on a stale chain tip"). Lost like any reject, but no
/// fault of the rig, so the GUI and the stats count it apart.
fn is_stale(error: &Value) -> bool {
    error[0] == json!(21)
}

/// How long repeats of one reject reason are held back before they are
/// counted in one line.
const REJECT_SUMMARY_EVERY: Duration = Duration::from_secs(10);

/// Says why shares are rejected once per reason, and then how many more
/// every `REJECT_SUMMARY_EVERY` while it goes on - not a line per share.
///
/// Right after the fork the pool started every connection at a share
/// difficulty an RTX 4090 beat ~145 times a second, and answered all but 4
/// (then 40) per 10 s with "too many shares per second" until vardiff had
/// caught up. A line each buried everything else said at the fork - and in
/// the GUI, whose log keeps 300 lines, the kernel and PoW lines were gone
/// within a second. The miner suggests a starting difficulty now
/// (`suggested_difficulty`), but not every pool takes it, nor every device
/// gets one.
#[derive(Default)]
struct RejectLog {
    /// Per reason: when its window started, and the rejects held back in it.
    open: Vec<(String, Instant, u64)>,
}

impl RejectLog {
    /// A share was rejected for `reason`: the line to print now, if any.
    fn reject(&mut self, reason: &str, now: Instant) -> Option<String> {
        if let Some(open) = self.open.iter_mut().find(|(r, _, _)| r == reason) {
            open.2 += 1;
            return None;
        }
        self.open.push((reason.to_string(), now, 0));
        Some(format!("[pool] share rejected: {reason}"))
    }

    /// The counts of the windows that are over. A reason that went quiet
    /// for a whole window is forgotten, so its next reject is said at once.
    fn due(&mut self, now: Instant) -> Vec<String> {
        let mut lines = Vec::new();
        self.open.retain_mut(|(reason, since, held)| {
            let window = now.duration_since(*since);
            if window < REJECT_SUMMARY_EVERY {
                return true;
            }
            if *held == 0 {
                return false;
            }
            lines.push(format!(
                "[pool] share rejected: {reason} - {held} more in the last {} s",
                window.as_secs()
            ));
            (*since, *held) = (now, 0);
            true
        });
        lines
    }
}

/// How long a session must have been authorized to count as healthy: the
/// next failure after one starts the reconnect backoff from the bottom.
const HEALTHY_SESSION: Duration = Duration::from_secs(60);

/// Seconds between reconnect attempts: 1, 2, 4 ... 60, and back to 1 after a
/// healthy session.
///
/// The ladder used to go back to 1 only when `session` returned Ok - which
/// it never does; it runs until the link fails. So every reconnect in the
/// life of the process doubled the wait for good, and from the seventh on a
/// rig sat out a full minute after every pool restart, however long ago the
/// one before had been. At a fork, when the pool is most likely to restart,
/// that minute is lost on every rig that has been up for a while.
struct Backoff {
    next: u64,
}

impl Backoff {
    const MAX_SECS: u64 = 60;

    fn new() -> Self {
        Self { next: 1 }
    }

    /// The wait before the next attempt, after a session that was `healthy`
    /// or not.
    fn after(&mut self, healthy: bool) -> u64 {
        if healthy {
            self.next = 1;
        }
        let wait = self.next;
        self.next = (self.next * 2).min(Self::MAX_SECS);
        wait
    }
}

pub fn run_client(shared: Arc<Shared>, submit_rx: Receiver<FoundShare>, cfg: StratumConfig) {
    let mut ladder = Backoff::new();
    loop {
        let mut authorized_at = None;
        let result = session(&shared, &submit_rx, &cfg, &mut authorized_at);
        let healthy = authorized_at.is_some_and(|t: Instant| t.elapsed() >= HEALTHY_SESSION);
        let backoff = ladder.after(healthy);
        match result {
            Ok(()) => {}
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
            }
        }
    }
}

/// One connection to the pool, until it fails. `authorized_at` is set when
/// the pool accepts the address - how the caller tells a session that worked
/// from one that never got going.
fn session(
    shared: &Arc<Shared>,
    submit_rx: &Receiver<FoundShare>,
    cfg: &StratumConfig,
    authorized_at: &mut Option<Instant>,
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
    let mut outstanding = Outstanding::new();
    send(&mut writer, json!({"id": 1, "method": "mining.subscribe",
        "params": [format!("bc3-miner/{}", env!("CARGO_PKG_VERSION"))]}))?;
    outstanding.sent(1, Instant::now());
    // Before authorize, so that the first job already carries it. The pool
    // keeps what the rig's own recent shares say instead, where it has them;
    // a pool that does not know the method answers with an error, which
    // nothing here reads.
    if let Some(d) = cfg.suggest_difficulty {
        send(&mut writer, json!({"id": 3, "method": "mining.suggest_difficulty", "params": [d]}))?;
        outstanding.sent(3, Instant::now());
    }
    send(&mut writer, json!({"id": 2, "method": "mining.authorize",
        "params": [cfg.user, "x"]}))?;
    outstanding.sent(2, Instant::now());

    let mut extranonce1: Vec<u8> = vec![];
    let mut subscribed = false;
    let mut extranonce2_size = 4usize;
    let mut difficulty = 1.0f64;
    let mut warned_difficulty = false;
    let mut next_submit_id: u64 = 100;
    // Kept across reads so a message split by a timeout is not lost.
    let mut rx_buf: Vec<u8> = Vec::with_capacity(8192);
    let mut last_rx = Instant::now();
    let mut rejects = RejectLog::default();

    loop {
        for line in rejects.due(Instant::now()) {
            eprintln!("{line}");
        }
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
            outstanding.sent(next_submit_id, Instant::now());
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

        // Any answer to something we sent is a round-trip measurement,
        // whatever it says.
        if let Some(id) = msg["id"].as_u64() {
            if let Some(rtt) = outstanding.answered(id, Instant::now()) {
                shared.record_pool_rtt(rtt);
            }
        }

        if msg["id"] == json!(1) {
            let r = &msg["result"];
            if r.is_null() {
                return Err(std::io::Error::new(ErrorKind::InvalidData, "subscribe rejected"));
            }
            // An extranonce1 that does not decode used to become an empty one,
            // and every notify after it was ignored: connected, no error, and
            // nothing mined for as long as the session lasted.
            extranonce1 = parse_extranonce1(&r[1]).ok_or_else(|| {
                std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("pool sent an unusable extranonce1: {}", r[1]),
                )
            })?;
            subscribed = true;
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
            *authorized_at = Some(Instant::now());
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
                    if let Some(line) = rejects.reject(&msg["error"].to_string(), Instant::now()) {
                        eprintln!("{line}");
                    }
                }
                let stale = !accepted && is_stale(&msg["error"]);
                if stale {
                    shared.stats.stale.fetch_add(1, Ordering::Relaxed);
                }
                crate::ipc::emit(&crate::ipc::Event::Share { accepted, stale });
                continue;
            }
        }

        match msg["method"].as_str() {
            Some("mining.set_difficulty") => match parse_difficulty(&msg["params"][0]) {
                Some(d) => difficulty = d,
                // A value that is not one used to become difficulty 1 - for
                // BC3Hash a million times what the pool meant, so no share for
                // the rest of the job - and 0 or less the easiest target there
                // is, half of all hashes a share. Keep the last good one.
                None => {
                    if !std::mem::replace(&mut warned_difficulty, true) {
                        eprintln!(
                            "[pool] ignoring an invalid share difficulty {} - keeping {difficulty}",
                            msg["params"][0]
                        );
                    }
                }
            },
            Some("mining.notify") => {
                let p = msg["params"].as_array().cloned().unwrap_or_default();
                if p.len() < 9 || !subscribed {
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

/// The share difficulty in a `mining.set_difficulty`: a finite number above
/// zero, or None. JSON has no NaN or infinity, but it has strings, nulls,
/// zero and negative numbers.
fn parse_difficulty(v: &Value) -> Option<f64> {
    v.as_f64().filter(|d| d.is_finite() && *d > 0.0)
}

/// Longest extranonce1 we take. Pools use 4 to 8 bytes; it goes into every
/// coinbase every worker builds.
const MAX_EXTRANONCE1: usize = 32;

/// The extranonce1 in a `mining.subscribe` result: hex of at most
/// `MAX_EXTRANONCE1` bytes. Empty is legal.
fn parse_extranonce1(v: &Value) -> Option<Vec<u8>> {
    hex::decode(v.as_str()?).ok().filter(|e| e.len() <= MAX_EXTRANONCE1)
}

fn parse_notify(
    p: &[Value],
    extranonce1: &[u8],
    extranonce2_size: usize,
    difficulty: f64,
) -> Option<MinerJob> {
    let prev_swab: [u8; 32] = hex::decode(p[1].as_str()?).ok()?.try_into().ok()?;
    let coinb1 = hex::decode(p[2].as_str()?).ok()?;
    let version = u32::from_str_radix(p[5].as_str()?, 16).ok()?;
    // The version bits decide the PoW, exactly as the node's GetHash does -
    // decided here, once per job, so no worker has to. The height is only
    // for showing the user which block is being mined.
    let height = bip34_height(&coinb1);
    let algo = PowAlgo::for_version(version);
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
        coinb1,
        coinb2: hex::decode(p[3].as_str()?).ok()?,
        merkle_steps: steps,
        version,
        prev_hash: swab32(&prev_swab),
        bits: u32::from_str_radix(p[6].as_str()?, 16).ok()?,
        ntime: u32::from_str_radix(p[7].as_str()?, 16).ok()?,
        share_target: target_for_difficulty(difficulty),
        height,
        algo,
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
    /// A reply is timed against the request it answers, once.
    #[test]
    fn an_answer_yields_its_round_trip() {
        let t0 = Instant::now();
        let mut o = Outstanding::new();
        o.sent(100, t0);
        o.sent(101, t0 + Duration::from_millis(10));
        assert_eq!(o.answered(7, t0 + Duration::from_millis(20)), None, "never sent");
        assert_eq!(
            o.answered(101, t0 + Duration::from_millis(45)),
            Some(Duration::from_millis(35))
        );
        assert_eq!(o.answered(101, t0 + Duration::from_millis(50)), None, "already answered");
        assert_eq!(
            o.answered(100, t0 + Duration::from_millis(60)),
            Some(Duration::from_millis(60))
        );
    }

    /// A pool that never answers must not make the list grow without bound.
    #[test]
    fn unanswered_requests_are_forgotten() {
        let t0 = Instant::now();
        let mut o = Outstanding::new();
        for id in 0..1000u64 {
            o.sent(id, t0);
        }
        o.sent(1000, t0 + OUTSTANDING_MAX_AGE + Duration::from_secs(1));
        assert_eq!(o.sent.len(), 1, "the stale ones are gone");
        assert_eq!(o.answered(5, t0 + OUTSTANDING_MAX_AGE + Duration::from_secs(2)), None);
        assert!(o.answered(1000, t0 + OUTSTANDING_MAX_AGE + Duration::from_secs(2)).is_some());
    }

    /// A mining.notify as the pool sends it, for a block at `height`.
    fn notify_params(height: u32, version: u32) -> Vec<Value> {
        let mut coinb1 = Vec::new();
        coinb1.extend_from_slice(&2u32.to_le_bytes());
        coinb1.push(0x01);
        coinb1.extend_from_slice(&[0u8; 32]);
        coinb1.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        coinb1.push(24);
        coinb1.push(0x04);
        coinb1.extend_from_slice(&height.to_le_bytes());
        coinb1.extend_from_slice(b"/bc3.online/");
        vec![
            json!("job1"),
            json!(hex::encode([0x11u8; 32])),
            json!(hex::encode(&coinb1)),
            json!("ffffffff"),
            json!([]),
            json!(format!("{version:08x}")),
            json!("1d00ffff"),
            json!("6a000000"),
            json!(true),
        ]
    }

    /// Every job carries its height and the PoW its version bits pick - the
    /// height plays no part in the choice.
    #[test]
    fn notify_decides_the_jobs_pow() {
        for (height, version, algo) in [
            (64_359, 0x2000_1000, PowAlgo::Sha3t),
            // What getblocktemplate hands out from the fork height on.
            (64_360, 0x2000_1800, PowAlgo::Bc3Hash),
            (64_360, 0x2000_0800, PowAlgo::Bc3Hash),
            // A SHA3-256t version past the height is still SHA3-256t.
            (64_361, 0x2000_1000, PowAlgo::Sha3t),
            (10, 0x2000_0000, PowAlgo::Sha256d),
        ] {
            let job = parse_notify(&notify_params(height, version), &[1, 2], 4, 1.0).unwrap();
            assert_eq!(job.height, Some(height));
            assert_eq!(job.version, version);
            assert_eq!(job.algo, algo, "height {height}, version {version:08x}");
        }
    }

    /// Only a finite difficulty above zero is one; anything else keeps the
    /// last good value.
    #[test]
    fn share_difficulties_must_be_positive_numbers() {
        assert_eq!(parse_difficulty(&json!(2e-6)), Some(2e-6));
        assert_eq!(parse_difficulty(&json!(16)), Some(16.0));
        assert_eq!(parse_difficulty(&json!(1e300)), Some(1e300));
        for bad in [json!(0), json!(-1.5), json!("0.5"), json!(null), json!([1]), json!(true)] {
            assert_eq!(parse_difficulty(&bad), None, "{bad}");
        }
    }

    /// An extranonce1 is hex of a sane length, or the session is refused -
    /// not silently left without work.
    #[test]
    fn extranonce1_must_be_hex_of_a_sane_length() {
        assert_eq!(parse_extranonce1(&json!("0a0b0c0d")), Some(vec![10, 11, 12, 13]));
        assert_eq!(parse_extranonce1(&json!("")), Some(vec![]));
        assert_eq!(parse_extranonce1(&json!("xyz")), None);
        assert_eq!(parse_extranonce1(&json!(12)), None);
        assert_eq!(parse_extranonce1(&json!("00".repeat(MAX_EXTRANONCE1))).map(|e| e.len()), Some(MAX_EXTRANONCE1));
        assert_eq!(parse_extranonce1(&json!("00".repeat(MAX_EXTRANONCE1 + 1))), None);
    }

    #[test]
    fn eof_ends_the_session() {
        let mut r = Chunks(vec![]);
        let mut buf = Vec::new();
        let err = next_line(&mut r, &mut buf).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConnectionAborted);
    }

    /// The reconnect waits double up to a minute, and a healthy session puts
    /// them back at the bottom. They never did: the reset hung on `session`
    /// returning Ok, which it never does.
    #[test]
    fn the_reconnect_backoff_starts_over_after_a_healthy_session() {
        let mut b = Backoff::new();
        let waits: Vec<u64> = (0..8).map(|_| b.after(false)).collect();
        assert_eq!(waits, [1, 2, 4, 8, 16, 32, 60, 60]);
        assert_eq!(b.after(true), 1, "a healthy session must reset the ladder");
        assert_eq!(b.after(false), 2);
    }

    /// Rejects are said once per reason and then counted every ten seconds,
    /// the way the fork's burst of rate-limited shares should read: one line
    /// at once, one count, and no line per share.
    #[test]
    fn rejects_are_said_once_per_reason_and_then_counted() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut log = RejectLog::default();
        let limited = r#"[20,"too many shares per second - difficulty will be raised",null]"#;
        let stale = r#"[21,"stale: the proof of work has changed",null]"#;
        assert_eq!(log.reject(limited, at(0)).as_deref(), Some(&*format!("[pool] share rejected: {limited}")));
        for i in 1..=140 {
            assert_eq!(log.reject(limited, at(i * 7)), None, "a line per share");
        }
        assert!(log.reject(stale, at(1_000)).is_some(), "another reason is said at once");
        assert!(log.due(at(9_999)).is_empty());
        let lines = log.due(at(10_000));
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("140 more in the last 10 s") && lines[0].contains("too many shares"));
        // The stale reason had nothing held back: no line, and forgotten.
        assert!(log.due(at(11_000)).is_empty());
        assert!(log.reject(stale, at(12_000)).is_some(), "said again after a quiet window");
        // The rate limit went quiet for a whole window: forgotten too.
        assert!(log.due(at(20_000)).is_empty());
        assert!(log.reject(limited, at(21_000)).is_some());
    }

    /// The pool's three stale answers are stale; its other rejects are not.
    #[test]
    fn stale_is_code_21() {
        for why in
            ["job not found (stale)", "stale: the proof of work has changed", "job is on a stale chain tip"]
        {
            assert!(is_stale(&json!([21, why, null])), "{why}");
        }
        assert!(!is_stale(&json!([20, "too many shares per second - difficulty will be raised", null])));
        assert!(!is_stale(&json!([23, "share above target (difficulty too low)", null])));
        assert!(!is_stale(&json!([22, "duplicate share", null])));
        assert!(!is_stale(&Value::Null));
    }

    /// The pool's end of a test connection.
    struct FakePool {
        stream: TcpStream,
        buf: Vec<u8>,
    }

    impl FakePool {
        fn accept(listener: &std::net::TcpListener) -> Self {
            let (stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_millis(50))).unwrap();
            Self { stream, buf: Vec::new() }
        }

        fn send(&mut self, v: Value) {
            self.stream.write_all((v.to_string() + "\n").as_bytes()).unwrap();
        }

        /// The next message from the miner; None when none came in time.
        fn next(&mut self) -> Option<Value> {
            next_line(&mut self.stream, &mut self.buf)
                .unwrap()
                .map(|l| serde_json::from_str(&l).unwrap())
        }
    }

    /// A job as the pool keeps it, to check the shares it gets back.
    #[derive(Clone, Copy)]
    struct PoolJob {
        id: &'static str,
        height: u32,
        version: u32,
        /// In stratum's word-swapped order, as notify carries it.
        prev: [u8; 32],
        bits: u32,
        difficulty: f64,
    }

    const NTIME: u32 = 0x6a00_0000;
    const EN1: [u8; 4] = [0x0a, 0x0b, 0x0c, 0x0d];

    impl PoolJob {
        fn coinb1(&self) -> Vec<u8> {
            let mut coinb1 = Vec::new();
            coinb1.extend_from_slice(&2u32.to_le_bytes());
            coinb1.push(0x01);
            coinb1.extend_from_slice(&[0u8; 32]);
            coinb1.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
            coinb1.push(24);
            coinb1.push(0x04);
            coinb1.extend_from_slice(&self.height.to_le_bytes());
            coinb1.extend_from_slice(b"/bc3.online/");
            coinb1
        }

        /// set_difficulty and notify, in the order the pool sends them.
        fn announce(&self, pool: &mut FakePool, clean: bool) {
            pool.send(json!({"id": null, "method": "mining.set_difficulty", "params": [self.difficulty]}));
            pool.send(json!({"id": null, "method": "mining.notify", "params": [
                self.id, hex::encode(self.prev), hex::encode(self.coinb1()), "ffffffff", [],
                format!("{:08x}", self.version), format!("{:08x}", self.bits),
                format!("{NTIME:08x}"), clean,
            ]}));
        }

        /// Does a submit meet this job's target under the job's own PoW -
        /// the header rebuilt from the job and the submit alone.
        fn check(&self, en2: &[u8], ntime: u32, nonce: u32) -> bool {
            let mut coinbase = self.coinb1();
            coinbase.extend_from_slice(&EN1);
            coinbase.extend_from_slice(en2);
            coinbase.extend_from_slice(&[0xff; 4]);
            let header = crate::consensus::BlockHeader {
                version: self.version,
                prev_hash: swab32(&self.prev),
                merkle_root: crate::consensus::sha256d(&coinbase),
                time: ntime,
                bits: self.bits,
                nonce,
            }
            .serialize();
            let pow = PowAlgo::for_version(self.version).hash_fn().unwrap();
            crate::consensus::hash_meets_target(&pow(&header), &target_for_difficulty(self.difficulty))
        }
    }

    #[test]
    fn the_suggested_difficulty_scales_with_the_cuda_cards_and_intensity() {
        assert_eq!(suggested_difficulty(0, 100), None);
        assert_eq!(suggested_difficulty(1, 100), Some(1e-4));
        let rig = suggested_difficulty(8, 50).unwrap();
        assert!((rig - 4e-4).abs() < 1e-12, "{rig}");
        // Intensity is 1-100; nothing outside it may zero or inflate it.
        assert!(suggested_difficulty(1, 0).unwrap() > 0.0);
        assert_eq!(suggested_difficulty(1, 500), Some(1e-4));
    }

    /// The suggestion goes between subscribe and authorize, so that the
    /// pool's first job carries it - and a pool that does not know the
    /// method costs nothing: its error is not a rejected share, and the
    /// session goes on.
    #[test]
    fn the_difficulty_is_suggested_before_authorize() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let cfg = StratumConfig {
            pool: listener.local_addr().unwrap().to_string(),
            user: "bc1qtest.rig".into(),
            suggest_difficulty: Some(2e-4),
        };
        let (tx, rx) = std::sync::mpsc::sync_channel(crate::shared::SUBMIT_QUEUE);
        let shared = Arc::new(Shared::new(tx, Default::default()));
        let s = shared.clone();
        let miner = std::thread::spawn(move || {
            let mut authorized_at = None;
            let result = session(&s, &rx, &cfg, &mut authorized_at);
            (result, authorized_at)
        });
        let mut pool = FakePool::accept(&listener);
        let mut asked = Vec::new();
        while asked.len() < 3 {
            if let Some(m) = pool.next() {
                asked.push(m);
            }
        }
        let methods: Vec<&str> = asked.iter().map(|m| m["method"].as_str().unwrap_or("")).collect();
        assert_eq!(methods, ["mining.subscribe", "mining.suggest_difficulty", "mining.authorize"]);
        assert_eq!(asked[1]["params"][0].as_f64(), Some(2e-4));
        pool.send(json!({"id": 1, "error": null, "result": [
            [["mining.set_difficulty", "0a0b0c0d"], ["mining.notify", "0a0b0c0d"]], "0a0b0c0d", 4]}));
        pool.send(json!({"id": 3, "result": null, "error": [20, "unknown method", null]}));
        pool.send(json!({"id": 2, "error": null, "result": true}));
        // Answered and authorized: then the pool goes away.
        let deadline = Instant::now() + Duration::from_secs(5);
        while shared.pool_rtt_ms().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        std::thread::sleep(Duration::from_millis(300));
        drop(pool);
        let (result, authorized_at) = miner.join().unwrap();
        assert!(result.is_err(), "the session ends when the pool goes away");
        assert!(authorized_at.is_some(), "the error to the suggestion must not stop the session");
        assert_eq!(shared.stats.rejected.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    /// The fork over a real connection, as the pool runs it: a SHA3-256t
    /// job, as from a pool that is not past the fork, which the miner must
    /// not mine at all - not a hash, not a share. Then at the fork height a
    /// job with bit 11 alone and the BC3Hash start difficulty -
    /// `set_difficulty` before `notify`, which is what binds the difficulty
    /// to the job. A CPU worker mines throughout; every share is checked like
    /// the pool checks it. The miner must start on BC3Hash at once, never
    /// submit a share that fails its job, and take "too many shares per
    /// second" while vardiff catches up as rejects to count, not as a reason
    /// to drop the connection or stop. Then a retarget, and the pool going
    /// away.
    #[cfg(feature = "bc3hash")]
    #[test]
    fn the_fork_over_a_pool_connection() {
        use crate::shared::{MiningPlan, SUBMIT_QUEUE};
        use std::sync::atomic::Ordering;
        const RATE_LIMITED: &str = "too many shares per second - difficulty will be raised";

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let cfg = StratumConfig {
            pool: listener.local_addr().unwrap().to_string(),
            user: "bc1qtest.rig".into(),
            suggest_difficulty: None,
        };
        let (tx, rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let plan = MiningPlan { cpu_threads: 1, ..Default::default() };
        let shared = Arc::new(Shared::new(tx, plan));
        let s = shared.clone();
        let miner = std::thread::spawn(move || {
            let mut authorized_at = None;
            let result = session(&s, &rx, &cfg, &mut authorized_at);
            (result, authorized_at)
        });
        let s = shared.clone();
        std::thread::spawn(move || {
            crate::worker::run_worker(s, 0, 1, crate::worker::CpuRole::All)
        });

        let mut pool = FakePool::accept(&listener);
        let mut asked = Vec::new();
        while asked.len() < 2 {
            if let Some(m) = pool.next() {
                asked.push(m["method"].as_str().unwrap_or("").to_string());
            }
        }
        assert_eq!(asked, ["mining.subscribe", "mining.authorize"]);
        pool.send(json!({"id": 1, "error": null, "result": [
            [["mining.set_difficulty", "0a0b0c0d"], ["mining.notify", "0a0b0c0d"]], "0a0b0c0d", 4]}));
        pool.send(json!({"id": 2, "error": null, "result": true}));

        let sha3 = PoolJob {
            id: "s1",
            height: 64_359,
            version: 0x2000_1000,
            prev: [0x11; 32],
            bits: 0x1d00_ffff,
            difficulty: 1e-5,
        };
        let fork = PoolJob {
            id: "b1",
            height: 64_360,
            version: 0x2000_0800,
            prev: [0x22; 32],
            bits: 0x1c0f_fff0,
            difficulty: 2.5e-8,
        };
        // A vardiff retarget: the same template at another difficulty.
        let retarget = PoolJob { id: "b2", difficulty: 5e-8, ..fork };

        #[derive(Default, Debug)]
        struct Tally {
            bc3_ok: u32,
            limited: u32,
            retarget_ok: u32,
            bad: Vec<String>,
        }
        let mut t = Tally::default();
        let mut first_bc3: Option<Duration> = None;
        let mut retargeted = false;

        // Not past the fork: nothing to mine, and nothing is.
        sha3.announce(&mut pool, true);
        let quiet = Instant::now() + Duration::from_millis(1500);
        while Instant::now() < quiet {
            if let Some(m) = pool.next() {
                panic!("the miner sent something for a SHA3-256t job: {m}");
            }
        }
        assert_eq!(shared.current_algo(), Some(PowAlgo::Sha3t));
        assert_eq!(shared.stats.hashes.load(Ordering::Relaxed), 0, "a SHA3-256t job was hashed");

        fork.announce(&mut pool, true);
        let forked = Instant::now();
        let deadline = forked + Duration::from_secs(120);
        while t.retarget_ok < 2 {
            assert!(Instant::now() < deadline, "the fork never got going: {t:?}");
            if !retargeted && t.bc3_ok >= 3 && t.limited >= 3 {
                retarget.announce(&mut pool, false);
                retargeted = true;
            }
            let Some(m) = pool.next() else { continue };
            assert_eq!(m["method"], "mining.submit", "{m}");
            let p = &m["params"];
            // No version rolling, ever: the PoW bits stay the job's.
            assert_eq!(p.as_array().map(Vec::len), Some(5), "a submit with version bits: {m}");
            let en2 = hex::decode(p[2].as_str().unwrap()).unwrap();
            let ntime = u32::from_str_radix(p[3].as_str().unwrap(), 16).unwrap();
            let nonce = u32::from_str_radix(p[4].as_str().unwrap(), 16).unwrap();
            let job = match p[1].as_str().unwrap() {
                "b1" => &fork,
                "b2" => &retarget,
                "s1" => panic!("a share for the SHA3-256t job: {m}"),
                other => panic!("a share for a job never sent: {other}"),
            };
            if !job.check(&en2, ntime, nonce) {
                t.bad.push(format!("{m}"));
            }
            let answer = if job.id == "b2" {
                t.retarget_ok += 1;
                Ok(())
            } else if t.bc3_ok < 3 {
                first_bc3.get_or_insert(forked.elapsed());
                t.bc3_ok += 1;
                Ok(())
            } else {
                t.limited += 1;
                Err((20, RATE_LIMITED))
            };
            match answer {
                Ok(()) => pool.send(json!({"id": m["id"], "error": null, "result": true})),
                Err((code, why)) => {
                    pool.send(json!({"id": m["id"], "result": null, "error": [code, why, null]}))
                }
            }
        }
        assert!(t.bad.is_empty(), "shares that fail their job's target: {:?}", t.bad);
        println!("fork: first BC3Hash share {first_bc3:?} after the switch; {t:?}");

        // Every answer was counted, and the connection was not given up.
        let answered = t.bc3_ok + t.limited + t.retarget_ok;
        let until = Instant::now() + Duration::from_secs(10);
        while shared.stats.accepted.load(Ordering::Relaxed) + shared.stats.rejected.load(Ordering::Relaxed)
            < u64::from(answered)
        {
            assert!(Instant::now() < until, "the miner did not take every answer");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(
            shared.stats.accepted.load(Ordering::Relaxed),
            u64::from(t.bc3_ok + t.retarget_ok)
        );
        assert_eq!(shared.stats.rejected.load(Ordering::Relaxed), u64::from(t.limited));
        assert!(!miner.is_finished(), "the miner dropped the connection over rejects");
        assert_eq!(shared.current_algo(), Some(PowAlgo::Bc3Hash));

        // The pool goes away: the session ends with an error, and says it
        // had been authorized, so a long one resets the reconnect backoff.
        drop(pool);
        let (result, authorized_at) = miner.join().unwrap();
        assert!(result.is_err());
        assert!(authorized_at.is_some());
        shared.clear_job();
    }
}
