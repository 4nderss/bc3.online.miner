//! bc3-miner — CPU/GPU-miner för BC3 (bc3.online-klienten).
//!
//! GPU-backends (CUDA via NVRTC, OpenCL) och CPU-trådar delar samma
//! jobbpipeline: stratum-klienten publicerar jobb, arbetarna partitionerar
//! extranonce2-rymden disjunkt och rapporterar shares på samma kanal.

mod backend;
mod consensus;
mod gpu_worker;
mod ipc;
mod telemetry;
mod shared;
mod stats;
mod stratum;
mod worker;

use backend::BackendKind;
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

    /// Backend: auto = CUDA om möjligt, annars OpenCL, annars CPU.
    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    backend: BackendKind,

    /// Använd bara en specifik GPU (index i den detekterade listan).
    #[arg(long)]
    gpu_id: Option<usize>,

    /// Antal CPU-trådar (0 = alla kärnor). Med GPU-backend är standard att
    /// inga CPU-trådar startas — ange flaggan för att mina med båda.
    #[arg(long)]
    threads: Option<usize>,

    /// Intensitet 1–100 %. Under 100 vilar arbetarna mellan pass, vilket
    /// sänker värme/strömförbrukning och gör datorn användbar under mining.
    #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=100))]
    intensity: u32,

    /// Sekunder mellan statistikrader.
    #[arg(long, default_value_t = 5)]
    stats_interval: u64,

    /// Skriv maskinläsbara JSON-rader på stdout i stället för text (GUI:t).
    #[arg(long)]
    json: bool,

    /// Lista tillgänglig hårdvara och avsluta (GUI:t frågar innan start).
    #[arg(long)]
    probe: bool,
}

fn main() {
    let args = Args::parse();
    ipc::set_json_mode(args.json);

    let all_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    if args.probe {
        let gpus: Vec<String> = backend::detect_gpus(BackendKind::Auto, None)
            .iter()
            .map(|g| g.describe())
            .collect();
        ipc::emit(&ipc::Event::Probe {
            gpus: gpus.clone(),
            cpu_cores: all_cores,
        });
        human!("GPUs: {gpus:?}, CPU cores: {all_cores}");
        return;
    }

    let gpus = if matches!(args.backend, BackendKind::Cpu) {
        vec![]
    } else {
        backend::detect_gpus(args.backend, args.gpu_id)
    };
    if gpus.is_empty() {
        match args.backend {
            BackendKind::Cuda => {
                eprintln!("bc3-miner: no CUDA device found");
                std::process::exit(1);
            }
            BackendKind::Opencl => {
                eprintln!("bc3-miner: no OpenCL GPU found");
                std::process::exit(1);
            }
            BackendKind::Auto => human!("[gpu] no GPU found — mining on CPU"),
            BackendKind::Cpu => {}
        }
    }

    let threads = match (args.threads, gpus.is_empty()) {
        (Some(0), _) => all_cores,
        (Some(n), _) => n,
        (None, true) => all_cores, // ren CPU-mining som förr
        (None, false) => 0,        // GPU-läge: inga CPU-trådar om inte begärt
    };

    human!(
        "bc3-miner {} — {} GPU(s), {} CPU thread(s)",
        env!("CARGO_PKG_VERSION"),
        gpus.len(),
        threads
    );
    for gpu in &gpus {
        human!("[gpu] found: {}", gpu.describe());
    }
    ipc::emit(&ipc::Event::Startup {
        version: env!("CARGO_PKG_VERSION").to_string(),
        backend: if gpus.is_empty() { "cpu".into() } else { "gpu".into() },
        gpus: gpus.iter().map(|g| g.describe()).collect(),
        cpu_threads: threads,
    });

    let (submit_tx, submit_rx) = std::sync::mpsc::channel();
    let shared = Arc::new(shared::Shared::new(submit_tx));
    shared.set_intensity(args.intensity);
    if args.intensity < 100 {
        human!("[miner] intensity {}%", args.intensity);
    }

    // CPU-trådar och GPU:er delar en gemensam disjunkt extranonce2-partition.
    let total_workers = threads + gpus.len();
    for i in 0..threads {
        let s = shared.clone();
        std::thread::Builder::new()
            .name(format!("worker-{i}"))
            .spawn(move || worker::run_worker(s, i, total_workers))
            .expect("kunde inte starta arbetstråd");
    }
    for (g, gpu) in gpus.into_iter().enumerate() {
        let s = shared.clone();
        std::thread::Builder::new()
            .name(format!("gpu-{g}"))
            .spawn(move || gpu_worker::run_gpu_worker(s, gpu, threads + g, total_workers))
            .expect("kunde inte starta GPU-tråd");
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
