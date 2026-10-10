//! BC3Hash hash rate, the way the CPU worker calls it: one 80-byte header per
//! call, nonce patched in place.
//!
//!     cargo run --release --example bench -- <compact|fast> <threads> [seconds]
//!
//! The mode is fixed for the life of a process, so compare modes by running
//! it twice - and alternate the runs: a laptop's clock drifts with heat by
//! more than the difference being measured.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = match args.get(1).map(String::as_str) {
        Some("fast") => bc3hash::Mode::Fast,
        Some("compact") | None => bc3hash::Mode::Compact,
        Some(other) => panic!("mode must be compact or fast, not {other}"),
    };
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let seconds: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    bc3hash::configure(mode);
    let t0 = Instant::now();
    bc3hash::initialize().expect("BC3Hash tables");
    let init = t0.elapsed();

    let stop = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(threads + 1));
    let workers: Vec<_> = (0..threads)
        .map(|t| {
            let stop = stop.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                let mut header = [0u8; 80];
                header[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
                header[4..8].copy_from_slice(&(t as u32).to_le_bytes());
                let mut nonce = 0u32;
                let mut sink = 0u8;
                start.wait();
                let t0 = Instant::now();
                while !stop.load(Ordering::Relaxed) {
                    header[76..80].copy_from_slice(&nonce.to_le_bytes());
                    sink ^= bc3hash::bc3hash(&header)[31];
                    nonce = nonce.wrapping_add(1);
                }
                std::hint::black_box(sink);
                nonce as f64 / t0.elapsed().as_secs_f64()
            })
        })
        .collect();
    start.wait();
    std::thread::sleep(Duration::from_secs(seconds));
    stop.store(true, Ordering::Relaxed);
    let rates: Vec<f64> = workers.into_iter().map(|w| w.join().unwrap()).collect();

    let total: f64 = rates.iter().sum();
    let min = rates.iter().copied().fold(f64::INFINITY, f64::min);
    let max = rates.iter().copied().fold(0.0, f64::max);
    println!(
        "{:<7} tables {:>6.1} MiB, init {:>6.2} s | {threads:>2} thread(s): total {total:>8.1} H/s, \
         per thread avg {:>7.1} min {min:>7.1} max {max:>7.1}",
        format!("{mode:?}").to_lowercase(),
        bc3hash::table_bytes() as f64 / (1 << 20) as f64,
        init.as_secs_f64(),
        total / threads as f64,
    );
}
