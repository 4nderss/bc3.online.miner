//! bc3-miner — CPU/GPU-miner för BC3 (bc3.online-klienten).
//!
//! Denna version minar med CPU (flertrådad SHA3-256t). GPU-backends (CUDA,
//! OpenCL) ansluts i samma jobbpipeline.

mod consensus;
mod shared;
mod stats;
mod stratum;
mod worker;

use clap::Parser;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "bc3-miner", version, about = "BC3 (SHA3-256t) miner — bc3.online")]
struct Args {
    /// Pooladress (host:port). PPLNS: bc3.online:3111, solo: bc3.online:3112.
    #[arg(long, default_value = "bc3.online:3111")]
    pool: String,

    /// Din BC3-adress, valfritt med riggnamn: adress.riggnamn
    #[arg(long)]
    user: String,

    /// Antal CPU-trådar (0 = alla kärnor).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Sekunder mellan statistikrader.
    #[arg(long, default_value_t = 5)]
    stats_interval: u64,
}

fn main() {
    let args = Args::parse();
    let threads = if args.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        args.threads
    };

    println!("bc3-miner {} — {} CPU-trådar", env!("CARGO_PKG_VERSION"), threads);

    let (submit_tx, submit_rx) = std::sync::mpsc::channel();
    let shared = Arc::new(shared::Shared::new(submit_tx));

    for i in 0..threads {
        let s = shared.clone();
        std::thread::Builder::new()
            .name(format!("worker-{i}"))
            .spawn(move || worker::run_worker(s, i, threads))
            .expect("kunde inte starta arbetstråd");
    }
    {
        let s = shared.clone();
        let interval = args.stats_interval;
        std::thread::spawn(move || stats::run_reporter(s, interval));
    }

    stratum::run_client(
        shared,
        submit_rx,
        stratum::StratumConfig {
            pool: args.pool,
            user: args.user,
        },
    );
}
