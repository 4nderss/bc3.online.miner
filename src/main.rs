//! bc3-miner - CPU/GPU miner for BC3 (the bc3.online client).
//!
//! The GPU backends (CUDA via NVRTC, OpenCL) and the CPU threads share the
//! same job pipeline: the stratum client publishes jobs, the workers
//! partition the extranonce2 space disjointly and report shares on the same
//! channel.

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
    /// Payout mode. Picks the pool port: pplns 3111, solo 3112.
    #[arg(long, value_enum, default_value_t = Mode::Pplns, env = "BC3_MODE")]
    mode: Mode,

    /// Pool address (host:port). Overrides --mode when given.
    #[arg(long, env = "BC3_POOL")]
    pool: Option<String>,

    /// Your BC3 address, optionally with a rig name: address.rigname.
    /// Required for mining, but not for `--probe`.
    #[arg(long, env = "BC3_USER")]
    user: Option<String>,

    /// Your BC3 address, without a rig name. Use with --rig instead of --user
    /// when the two come from separate places - a container environment, say.
    #[arg(long, env = "BC3_WALLET")]
    wallet: Option<String>,

    /// Rig name, combined with --wallet. Ignored when --user is given.
    #[arg(long, env = "BC3_WORKER")]
    rig: Option<String>,

    /// Backend: auto = CUDA if possible, else OpenCL, else CPU.
    #[arg(long, value_enum, default_value_t = BackendKind::Auto, env = "BC3_BACKEND")]
    backend: BackendKind,

    /// Use only one specific GPU (index into the detected list).
    #[arg(long, env = "BC3_GPU_ID")]
    gpu_id: Option<usize>,

    /// Number of CPU threads (0 = all cores). With a GPU backend the default
    /// is to start no CPU threads - pass the flag to mine with both.
    #[arg(long, env = "BC3_THREADS")]
    threads: Option<usize>,

    /// Intensity 1-100 %. Below 100 the workers idle between passes, which
    /// lowers heat/power draw and keeps the machine usable while mining.
    #[arg(long, default_value_t = 100, env = "BC3_INTENSITY",
          value_parser = clap::value_parser!(u32).range(1..=100))]
    intensity: u32,

    /// Seconds between statistics lines.
    #[arg(long, default_value_t = 5)]
    stats_interval: u64,

    /// Print machine-readable JSON lines on stdout instead of text (the GUI).
    #[arg(long)]
    json: bool,

    /// List available hardware and exit (the GUI asks before starting).
    #[arg(long)]
    probe: bool,
}

/// Payout mode. Only the port differs, but nobody should have to remember
/// which port is which.
#[derive(Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    Pplns,
    Solo,
}

impl Mode {
    fn default_pool(self) -> &'static str {
        match self {
            Mode::Pplns => "bc3.online:3111",
            Mode::Solo => "bc3.online:3112",
        }
    }
}

/// Promote any casing of our environment variables to the canonical name.
///
/// Environment variables are case-sensitive on Linux, and Docker passes them
/// through verbatim, so `-e bc3_wallet=...` silently does nothing while
/// `BC3_WALLET` works. That failure looks like the miner ignoring you. Accept
/// either, and let clap read the canonical name.
///
/// Only our own BC3_* names are touched, and only when the canonical one is
/// unset, so nothing else in the environment can be shadowed.
fn normalise_env() {
    const VARS: [&str; 9] = [
        "BC3_WALLET", "BC3_WORKER", "BC3_USER", "BC3_MODE", "BC3_POOL",
        "BC3_BACKEND", "BC3_THREADS", "BC3_INTENSITY", "BC3_GPU_ID",
    ];
    let present: Vec<(String, String)> = std::env::vars().collect();
    for want in VARS {
        if std::env::var_os(want).is_some() {
            continue;
        }
        if let Some((_, v)) = present.iter().find(|(k, _)| k.eq_ignore_ascii_case(want)) {
            std::env::set_var(want, v);
        }
    }
}

fn main() {
    normalise_env();
    let args = Args::parse();
    ipc::set_json_mode(args.json);

    let all_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    if args.probe {
        // The probe is a diagnostic: list BOTH paths, not just the one Auto
        // would have picked. On an NVIDIA card CUDA wins in Auto, and then
        // you would never see whether the OpenCL runtime works too.
        let mut gpus: Vec<String> = backend::detect_gpus(BackendKind::Cuda, None)
            .iter()
            .map(|g| g.describe())
            .collect();
        gpus.extend(
            backend::detect_gpus(BackendKind::Opencl, None)
                .iter()
                .map(|g| g.describe()),
        );
        ipc::emit(&ipc::Event::Probe {
            gpus: gpus.clone(),
            cpu_cores: all_cores,
        });
        human!("GPUs: {gpus:?}, CPU cores: {all_cores}");
        return;
    }

    // --user wins; otherwise build it from --wallet and --rig, which is the
    // shape a container environment hands us.
    let user = args.user.or_else(|| {
        args.wallet.as_ref().map(|w| match args.rig.as_deref() {
            Some(r) if !r.is_empty() => format!("{w}.{r}"),
            _ => w.clone(),
        })
    });
    let Some(user) = user else {
        eprintln!(
            "bc3-miner: --user <BC3-address[.rig]> is required for mining              (or --wallet, optionally with --rig)"
        );
        std::process::exit(2);
    };
    let pool = args.pool.clone().unwrap_or_else(|| args.mode.default_pool().to_string());

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

    // "All cores" (0) means different things with and without a GPU: every
    // GPU has a feeder thread that must get CPU time to launch the next
    // kernel in time. If we let CPU mining take every core the feeders starve
    // and the GPU hashrate drops more than the CPU adds. So we reserve one
    // core per GPU plus one more.
    let threads = match (args.threads, gpus.is_empty()) {
        (Some(0), true) => all_cores,
        (Some(0), false) => all_cores.saturating_sub(gpus.len() + 1).max(1),
        (Some(n), _) => n,
        (None, true) => all_cores, // pure CPU mining, as before
        (None, false) => 0,        // GPU mode: no CPU threads unless asked for
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

    // CPU threads and GPUs share one common disjoint extranonce2 partition.
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
            pool,
            user,
        },
    );
}
