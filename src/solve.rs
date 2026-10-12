//! `--solve <job.json>`: mine one fixed block template offline.
//!
//! For the hard fork's first block the developer published the template as a
//! file - height, previous hash, version, time, bits, the text for the
//! coinbase's scriptSig and the coinbase outputs - rather than as a pool job:
//! no node outside his has the blocks before it yet, so no node of ours can
//! hand out a `getblocktemplate` for it. This mines that template on the
//! ordinary workers - the same job dispatch, the same GPU kernels, every GPU
//! hit checked on the CPU - with no pool, no node and no network, and writes
//! the whole block, ready for `submitblock`, when a hash meets the target.
//!
//! The coinbase is the template's: the BIP34 height, the scriptSig text, then
//! 4 + 4 bytes of extranonce, then the outputs exactly as listed, version 2,
//! final sequence, lock time 0. The extranonce is what the workers roll, as
//! with a pool. Each process draws its own extranonce1, so two machines - or
//! a restart - never search the same space.

use crate::consensus::{
    compact_to_target, hash_meets_target, hash_to_display, sha256d, target_for_difficulty,
    BlockHeader, PowAlgo,
};
use crate::shared::{FoundShare, MinerJob, Shared};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Extranonce bytes in the scriptSig: 4 per process, 4 that the workers roll.
const EXTRANONCE1: usize = 4;
const EXTRANONCE2: usize = 4;
/// Consensus limits on a coinbase scriptSig (`bad-cb-length`).
const MIN_SCRIPT_SIG: usize = 2;
const MAX_SCRIPT_SIG: usize = 100;
/// Seconds between progress lines.
const REPORT_EVERY: Duration = Duration::from_secs(30);
/// Seconds between progress shares the share difficulty aims for. The shares
/// only prove that the GPU's results check out on the CPU; the block is what
/// counts.
const SHARE_EVERY: f64 = 15.0;

pub struct Config {
    pub path: PathBuf,
    /// Fixed progress-share difficulty; None sets it from the measured rate.
    pub difficulty: Option<f64>,
    pub gpus: usize,
    pub cpu_threads: usize,
}

/// The job file as published (eu1.bc3pool.org/job.json). Fields it has that
/// are not read here (`nNonce`, `versionRollingMask`) are ignored: the
/// extranonce gives the workers all the room they need.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobFile {
    network: Option<String>,
    height: u32,
    n_version: String,
    hash_prev_block: String,
    n_time: u32,
    n_bits: String,
    script_sig_data: String,
    payouts: Vec<Payout>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Payout {
    script_pub_key: String,
    satoshis: u64,
}

/// The template, decoded: the header fields and the coinbase around the
/// extranonce.
#[derive(Debug, Clone)]
pub struct Template {
    pub network: Option<String>,
    pub height: u32,
    pub version: u32,
    /// Header (internal) byte order.
    pub prev_hash: [u8; 32],
    pub time: u32,
    pub bits: u32,
    pub coinb1: Vec<u8>,
    pub coinb2: Vec<u8>,
    pub script_sig_text: String,
    pub outputs: usize,
    pub payout_total: u64,
}

fn parse_u32(field: &str, s: &str) -> Result<u32, String> {
    let s = s.trim();
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => s.parse::<u32>(),
    };
    parsed.map_err(|e| format!("{field}: {s:?} is not a number ({e})"))
}

fn unhex(field: &str, s: &str) -> Result<Vec<u8>, String> {
    hex::decode(s.trim()).map_err(|e| format!("{field}: not hex ({e})"))
}

/// Bitcoin's CompactSize.
fn push_varint(out: &mut Vec<u8>, n: u64) {
    match n {
        0..=0xfc => out.push(n as u8),
        0xfd..=0xffff => {
            out.push(0xfd);
            out.extend_from_slice(&(n as u16).to_le_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(0xfe);
            out.extend_from_slice(&(n as u32).to_le_bytes());
        }
        _ => {
            out.push(0xff);
            out.extend_from_slice(&n.to_le_bytes());
        }
    }
}

/// `CScript() << height`, the prefix BIP34 requires of the coinbase scriptSig:
/// the height as a minimal little-endian script number, pushed. Heights up to
/// 16 would be an opcode instead; no block this mines is that low.
fn bip34_height(height: u32) -> Result<Vec<u8>, String> {
    if height <= 16 {
        return Err(format!("height {height}: too low for a pushed BIP34 height"));
    }
    let mut num = Vec::new();
    let mut h = height;
    while h > 0 {
        num.push(h as u8);
        h >>= 8;
    }
    // A set top bit would make the number negative.
    if num.last().is_some_and(|b| b & 0x80 != 0) {
        num.push(0);
    }
    let mut out = vec![num.len() as u8];
    out.extend_from_slice(&num);
    Ok(out)
}

impl Template {
    pub fn parse(json: &str) -> Result<Template, String> {
        let f: JobFile = serde_json::from_str(json).map_err(|e| format!("job file: {e}"))?;
        let version = parse_u32("nVersion", &f.n_version)?;
        let bits = parse_u32("nBits", &f.n_bits)?;
        if compact_to_target(bits).is_none() {
            return Err(format!("nBits 0x{bits:08x} is not a valid target"));
        }
        let mut prev: [u8; 32] = unhex("hashPrevBlock", &f.hash_prev_block)?
            .try_into()
            .map_err(|_| "hashPrevBlock: not 32 bytes".to_string())?;
        // Published in display order, like every explorer; the header holds
        // it reversed.
        prev.reverse();
        let text = unhex("scriptSigData", &f.script_sig_data)?;
        let height = bip34_height(f.height)?;
        let script_sig_len = height.len() + text.len() + EXTRANONCE1 + EXTRANONCE2;
        if !(MIN_SCRIPT_SIG..=MAX_SCRIPT_SIG).contains(&script_sig_len) {
            return Err(format!(
                "the coinbase scriptSig would be {script_sig_len} bytes (height {} + text {} + \
                 extranonce {}); consensus allows {MIN_SCRIPT_SIG}-{MAX_SCRIPT_SIG}",
                height.len(),
                text.len(),
                EXTRANONCE1 + EXTRANONCE2
            ));
        }
        if f.payouts.is_empty() {
            return Err("payouts: a coinbase needs at least one output".into());
        }

        // Coinbase up to the extranonce: version 2, one input spending the
        // null outpoint, the scriptSig's length, height and text.
        let mut coinb1 = 2u32.to_le_bytes().to_vec();
        coinb1.push(1);
        coinb1.extend_from_slice(&[0u8; 32]);
        coinb1.extend_from_slice(&[0xff; 4]);
        push_varint(&mut coinb1, script_sig_len as u64);
        coinb1.extend_from_slice(&height);
        coinb1.extend_from_slice(&text);

        // After it: final sequence, the outputs as listed, lock time 0.
        let mut coinb2 = vec![0xff; 4];
        push_varint(&mut coinb2, f.payouts.len() as u64);
        let mut total: u64 = 0;
        for (i, p) in f.payouts.iter().enumerate() {
            let script = unhex(&format!("payouts[{i}].scriptPubKey"), &p.script_pub_key)?;
            total = total
                .checked_add(p.satoshis)
                .ok_or_else(|| "payouts: the total overflows".to_string())?;
            coinb2.extend_from_slice(&p.satoshis.to_le_bytes());
            push_varint(&mut coinb2, script.len() as u64);
            coinb2.extend_from_slice(&script);
        }
        coinb2.extend_from_slice(&0u32.to_le_bytes());

        Ok(Template {
            network: f.network,
            height: f.height,
            version,
            prev_hash: prev,
            time: f.n_time,
            bits,
            coinb1,
            coinb2,
            script_sig_text: String::from_utf8_lossy(&text).into_owned(),
            outputs: f.payouts.len(),
            payout_total: total,
        })
    }

    pub fn algo(&self) -> PowAlgo {
        PowAlgo::for_version(self.version)
    }

    /// The template as a job for the workers, at a progress-share difficulty.
    pub fn job(&self, id: u64, extranonce1: &[u8], difficulty: f64) -> MinerJob {
        MinerJob {
            job_id: format!("solve-{id}"),
            extranonce1: extranonce1.to_vec(),
            extranonce2_size: EXTRANONCE2,
            coinb1: self.coinb1.clone(),
            coinb2: self.coinb2.clone(),
            merkle_steps: vec![],
            version: self.version,
            prev_hash: self.prev_hash,
            bits: self.bits,
            ntime: self.time,
            share_target: target_for_difficulty(difficulty),
            height: Some(self.height),
            algo: self.algo(),
        }
    }
}

/// The block a share makes: its header, and the header followed by the one
/// transaction, as `submitblock` takes it.
pub fn assemble(job: &MinerJob, share: &FoundShare) -> ([u8; 80], Vec<u8>) {
    let mut coinbase = job.coinb1.clone();
    coinbase.extend_from_slice(&job.extranonce1);
    coinbase.extend_from_slice(&share.extranonce2);
    coinbase.extend_from_slice(&job.coinb2);
    let header = BlockHeader {
        version: job.version,
        prev_hash: job.prev_hash,
        // One transaction: the merkle root is its txid.
        merkle_root: sha256d(&coinbase),
        time: share.ntime,
        bits: job.bits,
        nonce: share.nonce,
    }
    .serialize();
    let mut block = header.to_vec();
    push_varint(&mut block, 1);
    block.extend_from_slice(&coinbase);
    (header, block)
}

/// Difficulty of a compact target, relative to diff1 (0x1d00ffff).
fn difficulty_of_bits(bits: u32) -> f64 {
    let exponent = (bits >> 24) as i32;
    let mantissa = (bits & 0x007f_ffff) as f64;
    if mantissa == 0.0 {
        return f64::INFINITY;
    }
    65535.0 / mantissa * 256f64.powi(0x1d - exponent)
}

fn random_extranonce1() -> Vec<u8> {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        ^ (std::process::id() as u128) << 64;
    let mixed = sha256d(&seed.to_le_bytes());
    mixed[..EXTRANONCE1].to_vec()
}

fn format_duration(secs: f64) -> String {
    if !secs.is_finite() {
        return "-".into();
    }
    let s = secs.max(0.0) as u64;
    match s {
        0..=119 => format!("{s} s"),
        120..=7199 => format!("{} min", s / 60),
        _ => format!("{:.1} h", s as f64 / 3600.0),
    }
}

pub fn run(shared: Arc<Shared>, rx: Receiver<FoundShare>, cfg: Config) -> i32 {
    let text = match std::fs::read_to_string(&cfg.path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[solve] cannot read {}: {e}", cfg.path.display());
            return 2;
        }
    };
    let template = match Template::parse(&text) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[solve] {}: {e}", cfg.path.display());
            return 2;
        }
    };
    let algo = template.algo();
    let block_target = compact_to_target(template.bits).expect("checked when parsed");
    let block_difficulty = difficulty_of_bits(template.bits);
    let block_work = block_difficulty * 4_294_967_296.0;
    let mut prev_display = template.prev_hash;
    prev_display.reverse();
    println!(
        "[solve] bc3-miner {} - solving a fixed block template offline: no pool, no node, \
         nothing leaves this machine",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "[solve] block {} on {} ({}), version 0x{:08x} -> {}, bits 0x{:08x} = difficulty {}",
        template.height,
        hex::encode(prev_display),
        template.network.as_deref().unwrap_or("?"),
        template.version,
        algo.name(),
        template.bits,
        block_difficulty
    );
    println!(
        "[solve] time {} ({}), coinbase: {} output(s), {:.8} BC3, scriptSig text {:?}",
        template.time,
        chrono_utc(template.time),
        template.outputs,
        template.payout_total as f64 / 1e8,
        template.script_sig_text
    );
    if let Some(why) = crate::worker::cpu_unavailable(algo, true) {
        eprintln!("[solve] cannot check results on the CPU: {why}");
        return 1;
    }

    let extranonce1 = random_extranonce1();
    println!(
        "[solve] this run's extranonce1 is {} - another machine or a restart searches its own space",
        hex::encode(&extranonce1)
    );
    let guess = if cfg.gpus > 0 { cfg.gpus as f64 * 150e3 } else { cfg.cpu_threads.max(1) as f64 * 1e3 };
    let mut difficulty = cfg.difficulty.unwrap_or_else(|| share_difficulty(guess, block_difficulty));
    let mut jobs: HashMap<String, MinerJob> = HashMap::new();
    let mut job_count = 0u64;
    let mut publish = |difficulty: f64, jobs: &mut HashMap<String, MinerJob>| {
        job_count += 1;
        let job = template.job(job_count, &extranonce1, difficulty);
        jobs.insert(job.job_id.clone(), job.clone());
        shared.publish_job(job);
    };
    publish(difficulty, &mut jobs);

    let started = Instant::now();
    let hashes_at_start = shared.stats.hashes.load(std::sync::atomic::Ordering::Relaxed);
    let mut last_report = Instant::now();
    let mut last_hashes = hashes_at_start;
    let (mut ok, mut invalid) = (0u64, 0u64);
    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(share) => {
                let Some(job) = jobs.get(&share.job_id) else { continue };
                let (header, block) = assemble(job, &share);
                let hash = match algo.hash_fn() {
                    Some(pow) => pow(&header),
                    None => {
                        eprintln!("[solve] this build cannot hash {}", algo.name());
                        return 1;
                    }
                };
                if !hash_meets_target(&hash, &job.share_target) {
                    invalid += 1;
                    println!(
                        "[solve] INVALID result: nonce {:08x} - the CPU reference does not agree",
                        share.nonce
                    );
                    continue;
                }
                ok += 1;
                if hash_meets_target(&hash, &block_target) {
                    let code = found(&template, &hash, &header, &block, started.elapsed());
                    quiesce(&shared);
                    return code;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("[solve] the workers have stopped");
                return 1;
            }
        }
        if last_report.elapsed() >= REPORT_EVERY {
            let now = shared.stats.hashes.load(std::sync::atomic::Ordering::Relaxed);
            let rate = (now - last_hashes) as f64 / last_report.elapsed().as_secs_f64();
            let done = (now - hashes_at_start) as f64;
            last_hashes = now;
            last_report = Instant::now();
            println!(
                "[solve] {} | {} | {:.3e} hashes | results {ok} ok, {invalid} invalid | \
                 expected {} per block at this rate, {:.1} % chance by now",
                format_duration(started.elapsed().as_secs_f64()),
                crate::bench::format_rate(rate),
                done,
                format_duration(block_work / rate),
                100.0 * (1.0 - (-done / block_work).exp())
            );
            if cfg.difficulty.is_none() && rate > 0.0 {
                let wanted = share_difficulty(rate, block_difficulty);
                if (wanted / difficulty - 1.0).abs() > 0.5 {
                    difficulty = wanted;
                    publish(difficulty, &mut jobs);
                }
            }
        }
    }
}

/// Take the job away and wait until the workers have stopped hashing. The
/// process exits right after, and exit runs the C++ side's static
/// destructors: a CPU thread still inside BC3Hash then reads freed tables
/// and the miner dies with a segfault - after printing the block, but still
/// a crash to explain.
fn quiesce(shared: &Shared) {
    shared.clear_job();
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut last = shared.stats.hashes.load(std::sync::atomic::Ordering::Relaxed);
    let mut still = 0;
    while still < 3 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(300));
        let now = shared.stats.hashes.load(std::sync::atomic::Ordering::Relaxed);
        still = if now == last { still + 1 } else { 0 };
        last = now;
    }
}

/// A progress-share difficulty: one every `SHARE_EVERY` seconds, and always
/// well below the block's, so that every block solution is also a share.
fn share_difficulty(rate: f64, block_difficulty: f64) -> f64 {
    (rate * SHARE_EVERY / 4_294_967_296.0)
        .max(1.0 / (1u64 << 31) as f64)
        .min(block_difficulty / 64.0)
}

fn found(template: &Template, hash: &[u8; 32], header: &[u8; 80], block: &[u8], took: Duration) -> i32 {
    let hash_display = hash_to_display(hash);
    let header_hex = hex::encode(header);
    let block_hex = hex::encode(block);
    let name = format!("bc3-block-{}-{}.txt", template.height, &hash_display[..16]);
    let contents = format!(
        "height {}\nhash {}\nheader {}\nblock {}\n",
        template.height, hash_display, header_hex, block_hex
    );
    let saved = std::fs::write(&name, &contents);
    println!();
    println!("[solve] ==================== BLOCK {} SOLVED ====================", template.height);
    println!("[solve] after {}", format_duration(took.as_secs_f64()));
    println!("[solve] hash   {hash_display}");
    println!("[solve] header {header_hex}");
    println!("[solve] block  {block_hex}");
    match saved {
        Ok(()) => println!("[solve] saved to {name}"),
        Err(e) => eprintln!("[solve] could NOT save to {name}: {e} - copy the block line above"),
    }
    println!(
        "[solve] submit with: bitcoinIII-cli submitblock <the block line> - on a node that \
         has block {}",
        template.height.saturating_sub(1)
    );
    0
}

/// "2026-10-12 01:24:02 UTC" without a date crate.
fn chrono_utc(unix: u32) -> String {
    let secs = unix as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The job file the developer published for the fork block, verbatim.
    const FORK_JOB: &str = r#"{
  "network": "main",
  "height": 64360,
  "nVersion": "0x20000800",
  "hashPrevBlock": "00000000000008cc8b3b25f3d32ea394097e8234a340ce2bda1a20cff2ce1992",
  "nTime": 1791768242,
  "nBits": "0x1c0ffff0",
  "nNonce": 0,
  "versionRollingMask": "0x1fffe000",
  "scriptSigData": "534841332d32353674202b20323031362d626c6f636b20444141202d3e2042433348617368202b20617365727469392d3264207c20464f524b20424c4f434b21207c20f09f8e89e29aa1efb88ff09f92b8",
  "payouts": [
    { "scriptPubKey": "6a375468652042433348617368202b20617365727469392d32642065726120626567696e7320616e64206973206865726520746f2073746179", "satoshis": 21000000 },
    { "scriptPubKey": "6a0ef09f8e89e29aa1efb88ff09f92b8", "satoshis": 21000000 },
    { "scriptPubKey": "6a1ef09f9396f09faa99f09f92b0f09f9497f09f9490f09f9fa9e299a5efb88f", "satoshis": 21000000 },
    { "scriptPubKey": "6a1df09f9491e29b93efb88ff09f92b5f09f9a80f09f8eaff09fa587e29ca8", "satoshis": 21000000 },
    { "scriptPubKey": "51", "satoshis": 100000000 },
    { "scriptPubKey": "4104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5fac", "satoshis": 2100000000 }
  ]
}"#;

    #[test]
    fn bip34_heights_are_minimal_script_numbers() {
        assert_eq!(bip34_height(64_360).unwrap(), vec![3, 0x68, 0xfb, 0x00]);
        assert_eq!(bip34_height(17).unwrap(), vec![1, 17]);
        assert_eq!(bip34_height(128).unwrap(), vec![2, 0x80, 0x00]);
        assert_eq!(bip34_height(32_767).unwrap(), vec![2, 0xff, 0x7f]);
        assert_eq!(bip34_height(8_388_608).unwrap(), vec![4, 0x00, 0x00, 0x80, 0x00]);
        assert!(bip34_height(16).is_err());
    }

    /// The fork job decodes to the header fields it states and a coinbase
    /// with the height, the text, room for the extranonce and the six
    /// outputs in order.
    #[test]
    fn the_fork_job_decodes_to_its_header_and_coinbase() {
        let t = Template::parse(FORK_JOB).unwrap();
        assert_eq!(t.height, 64_360);
        assert_eq!(t.version, 0x2000_0800);
        assert_eq!(t.algo(), PowAlgo::Bc3Hash);
        assert_eq!(t.bits, 0x1c0f_fff0);
        assert_eq!(difficulty_of_bits(t.bits), 16.0);
        assert_eq!(t.time, 1_791_768_242);
        assert_eq!(chrono_utc(t.time), "2026-10-12 01:24:02 UTC");
        // Display order reversed into the header.
        assert_eq!(t.prev_hash[31], 0x00);
        assert_eq!(t.prev_hash[0], 0x92);
        assert_eq!(t.outputs, 6);
        assert_eq!(t.payout_total, 2_284_000_000);
        assert!(t.script_sig_text.starts_with("SHA3-256t + 2016-block DAA -> BC3Hash"));
        // version 2, 1 input, null outpoint, scriptSig length 4 + 81 + 8.
        assert_eq!(&t.coinb1[..5], &[2, 0, 0, 0, 1]);
        assert_eq!(&t.coinb1[5..37], &[0u8; 32]);
        assert_eq!(&t.coinb1[37..41], &[0xff; 4]);
        assert_eq!(t.coinb1[41], 93);
        assert_eq!(&t.coinb1[42..46], &[3, 0x68, 0xfb, 0x00]);
        assert_eq!(t.coinb1.len(), 46 + 81);
        // Final sequence, six outputs, the first 0.21 BC3, lock time 0.
        assert_eq!(&t.coinb2[..5], &[0xff, 0xff, 0xff, 0xff, 6]);
        assert_eq!(&t.coinb2[5..13], &21_000_000u64.to_le_bytes());
        assert_eq!(&t.coinb2[t.coinb2.len() - 4..], &[0, 0, 0, 0]);
    }

    /// A share becomes the header and the block `submitblock` takes: the
    /// header's merkle root is the coinbase's txid, then one transaction.
    #[test]
    fn a_share_becomes_a_header_and_a_whole_block() {
        let t = Template::parse(FORK_JOB).unwrap();
        let job = t.job(1, &[1, 2, 3, 4], 1e-6);
        let share = FoundShare {
            job_id: job.job_id.clone(),
            extranonce2: vec![9, 8, 7, 6],
            ntime: t.time,
            nonce: 0xdead_beef,
            hash_display: String::new(),
            is_block_candidate: false,
        };
        let (header, block) = assemble(&job, &share);
        assert_eq!(&block[..80], &header);
        assert_eq!(block[80], 1);
        let coinbase = &block[81..];
        assert_eq!(coinbase.len(), t.coinb1.len() + 8 + t.coinb2.len());
        assert_eq!(&coinbase[t.coinb1.len()..t.coinb1.len() + 8], &[1, 2, 3, 4, 9, 8, 7, 6]);
        assert_eq!(&header[36..68], &sha256d(coinbase));
        assert_eq!(&header[0..4], &0x2000_0800u32.to_le_bytes());
        assert_eq!(&header[4..36], &t.prev_hash);
        assert_eq!(&header[68..72], &t.time.to_le_bytes());
        assert_eq!(&header[72..76], &0x1c0f_fff0u32.to_le_bytes());
        assert_eq!(&header[76..80], &0xdead_beefu32.to_le_bytes());
    }

    #[test]
    fn a_coinbase_scriptsig_over_the_limit_is_refused() {
        let long = FORK_JOB.replace(
            "\"scriptSigData\": \"5348",
            "\"scriptSigData\": \"00000000000000005348",
        );
        let e = Template::parse(&long).unwrap_err();
        assert!(e.contains("101 bytes"), "{e}");
    }

    /// End to end on the CPU reference: the fork template at an easy target
    /// yields a block whose BC3Hash meets it.
    #[cfg(feature = "bc3hash")]
    #[test]
    fn an_easy_template_solves_to_a_valid_block() {
        let easy = FORK_JOB.replace("0x1c0ffff0", "0x207fffff");
        let t = Template::parse(&easy).unwrap();
        let job = t.job(1, &[0xaa; 4], 1e-9);
        let target = compact_to_target(t.bits).unwrap();
        let pow = PowAlgo::Bc3Hash.hash_fn().unwrap();
        let mut share = FoundShare {
            job_id: job.job_id.clone(),
            extranonce2: vec![0; 4],
            ntime: t.time,
            nonce: 0,
            hash_display: String::new(),
            is_block_candidate: false,
        };
        loop {
            let (header, _) = assemble(&job, &share);
            if hash_meets_target(&pow(&header), &target) {
                break;
            }
            share.nonce += 1;
            assert!(share.nonce < 64, "a 1-in-2 target took too long");
        }
        let (header, block) = assemble(&job, &share);
        assert!(hash_meets_target(&pow(&header), &target));
        assert_eq!(&block[..80], &header);
    }
}
