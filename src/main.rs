//! bc3-miner - CPU/GPU miner for BC3 (the bc3.online client).
//!
//! The GPU backends (CUDA via NVRTC, OpenCL) and the CPU threads share the
//! same job pipeline: the stratum client publishes jobs, the workers
//! partition the extranonce2 space disjointly and report shares on the same
//! channel.

mod backend;
mod bench;
mod consensus;
mod gpu_worker;
mod ipc;
mod telemetry;
mod shared;
mod stats;
mod stratum;
mod worker;

use backend::BackendKind;
use worker::CpuRole;
use clap::Parser;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "bc3-miner", version, about = "BC3 miner - bc3.online")]
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

    /// Backend: auto = CUDA if possible, else OpenCL, else CPU. In auto the
    /// CPU also mines whatever the GPU cannot (BC3Hash on OpenCL cards, or a
    /// CUDA card whose BC3Hash kernel failed its self-test); with cuda or
    /// opencl it mines only if --threads asks it to.
    #[arg(long, value_enum, default_value_t = BackendKind::Auto, env = "BC3_BACKEND")]
    backend: BackendKind,

    /// Use only one specific GPU (index into the detected list).
    #[arg(long, env = "BC3_GPU_ID")]
    gpu_id: Option<usize>,

    /// Number of CPU threads (0 = all cores), mining every job. Without the
    /// flag a GPU in --backend auto leaves the CPU idle until a job comes that
    /// the GPU has no kernel for, and then mines it on all cores; with
    /// --backend cuda or opencl the CPU does not mine at all.
    #[arg(long, env = "BC3_THREADS")]
    threads: Option<usize>,

    /// BC3Hash lookup tables for the CPU: fast (~256 MiB of RAM) or compact
    /// (~7 MiB, 10-20 % slower). Both find exactly the same shares. Default:
    /// compact when a GPU mines BC3Hash and the CPU only checks its shares,
    /// fast otherwise.
    #[cfg(feature = "bc3hash")]
    #[arg(long, value_enum, env = "BC3_BC3HASH_MODE")]
    bc3hash_mode: Option<Bc3HashMode>,

    /// Table window of the GPU's BC3Hash kernel in bits: 10 (7 MiB), 12
    /// (22 MiB), 14 (76 MiB) or 16 (256 MiB of GPU memory). Bigger is faster
    /// as long as the card's L2 keeps up; `--benchmark` without it compares
    /// all four. Default 16.
    #[cfg(feature = "bc3hash")]
    #[arg(long, env = "BC3_BC3HASH_WINDOW", value_parser = parse_bc3_window)]
    bc3_window: Option<u32>,

    /// Benchmark instead of mining: an in-process job source serves random
    /// jobs to the normal workers - same dispatch, same kernels, every GPU
    /// hit re-checked - and checks every share with the CPU reference. No
    /// pool, no network, no address needed.
    #[arg(long)]
    benchmark: bool,

    /// Benchmark: the PoW to measure.
    #[arg(long, value_enum, default_value_t = BenchAlgo::Bc3hash)]
    algo: BenchAlgo,

    /// Benchmark: seconds to measure. Comparing the BC3Hash windows, each
    /// window runs twice for this long (default 20); otherwise 60.
    #[arg(long)]
    seconds: Option<u64>,

    /// Benchmark: share difficulty. By default it is set from the measured
    /// hashrate so that a share comes every few seconds.
    #[arg(long)]
    difficulty: Option<f64>,

    /// Intensity 1-100 %. Below 100 the workers idle between passes, which
    /// lowers heat/power draw and keeps the machine usable while mining.
    #[arg(long, default_value_t = 100, env = "BC3_INTENSITY",
          value_parser = clap::value_parser!(u32).range(1..=100))]
    intensity: u32,

    /// Seconds between statistics lines.
    ///
    /// Bounded below at 1: 0 made the reporter `sleep(0)`, i.e. a tight loop
    /// that spins a core and re-reads telemetry as fast as it can - which on
    /// Windows meant starting a PowerShell process per iteration.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    stats_interval: u64,

    /// Print machine-readable JSON lines on stdout instead of text (the GUI).
    #[arg(long)]
    json: bool,

    /// Exit instead of falling back to CPU when no GPU is found.
    ///
    /// For rented GPU machines: a container that quietly mines on CPU is
    /// still billed at GPU prices, and the only symptom is a hashrate that
    /// looks like someone unplugged something. Failing makes the host's
    /// restart policy or alerting notice.
    #[arg(long, env = "BC3_REQUIRE_GPU")]
    require_gpu: bool,

    /// List available hardware and exit (the GUI asks before starting).
    #[arg(long)]
    probe: bool,
}

/// The PoW `--benchmark` measures.
#[derive(Copy, Clone, PartialEq, Eq, Debug, clap::ValueEnum)]
enum BenchAlgo {
    Bc3hash,
    Sha3t,
}

impl From<BenchAlgo> for consensus::PowAlgo {
    fn from(a: BenchAlgo) -> Self {
        match a {
            BenchAlgo::Bc3hash => consensus::PowAlgo::Bc3Hash,
            BenchAlgo::Sha3t => consensus::PowAlgo::Sha3t,
        }
    }
}

#[cfg(feature = "bc3hash")]
fn parse_bc3_window(s: &str) -> Result<u32, String> {
    match s.parse::<u32>() {
        Ok(w) if BC3_WINDOWS.contains(&w) => Ok(w),
        _ => Err(format!("must be one of {BC3_WINDOWS:?}")),
    }
}

/// The table windows there is a GPU kernel for (see build.rs).
#[cfg(feature = "bc3hash")]
const BC3_WINDOWS: [u32; 4] = [10, 12, 14, 16];

/// Size of the BC3Hash tables - see `bc3hash::Mode`.
#[cfg(feature = "bc3hash")]
#[derive(Copy, Clone, PartialEq, Eq, Debug, clap::ValueEnum)]
enum Bc3HashMode {
    Compact,
    Fast,
}

#[cfg(feature = "bc3hash")]
impl From<Bc3HashMode> for bc3hash::Mode {
    fn from(mode: Bc3HashMode) -> Self {
        match mode {
            Bc3HashMode::Compact => bc3hash::Mode::Compact,
            Bc3HashMode::Fast => bc3hash::Mode::Fast,
        }
    }
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
    const VARS: [&str; 11] = [
        "BC3_WALLET", "BC3_WORKER", "BC3_USER", "BC3_MODE", "BC3_POOL",
        "BC3_BACKEND", "BC3_THREADS", "BC3_INTENSITY", "BC3_GPU_ID", "BC3_BC3HASH_MODE",
        "BC3_BC3HASH_WINDOW",
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

/// Take the current directory and PATH out of the DLL search order.
///
/// Everything GPU-related is loaded by name at run time: `OpenCL.dll` by our
/// own loader, and `nvcuda.dll` by cudarc - which first tries `cuda.dll`,
/// `cuda64.dll`, `cuda64_12.dll` and a handful more that exist on no machine
/// at all, so every start walks the whole search order looking for them. With
/// the legacy order that walk includes the current directory and every PATH
/// entry, and the CLI is typically unpacked and run straight from Downloads.
/// A `cuda.dll` dropped there would be loaded and executed as the user.
///
/// `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` keeps the application directory and
/// System32 - where the drivers install all three libraries - and drops the
/// rest. Linux is unaffected: dlopen never searched the working directory.
#[cfg(windows)]
fn harden_dll_search_path() {
    #[link(name = "kernel32")]
    extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
    }
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
    // Fails only on a Windows too old to have the flag; then the loader keeps
    // the legacy order and there is nothing more we can do about it here.
    unsafe { SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) };
}

#[cfg(not(windows))]
fn harden_dll_search_path() {}

fn main() {
    harden_dll_search_path();
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
    // shape a container environment hands us. The benchmark needs neither.
    let user = args.user.clone().or_else(|| {
        args.wallet.as_ref().map(|w| match args.rig.as_deref() {
            Some(r) if !r.is_empty() => format!("{w}.{r}"),
            _ => w.clone(),
        })
    });
    let user = match user {
        Some(user) => user,
        None if args.benchmark => String::new(),
        None => {
            eprintln!("bc3-miner: set --user <address[.rig]>, or --wallet with an optional --rig");
            std::process::exit(2);
        }
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
            BackendKind::Auto if args.require_gpu => {
                eprintln!("bc3-miner: no GPU found, and --require-gpu is set.");
                eprintln!("  On a container host this usually means the NVIDIA runtime");
                eprintln!("  is not in use, so no device ever reached the container.");
                std::process::exit(1);
            }
            BackendKind::Auto => human!("[gpu] no GPU found - mining on CPU"),
            BackendKind::Cpu => {}
        }
    }

    let (threads, role) = plan_cpu(args.backend, gpus.len(), args.threads, all_cores);
    let plan = shared::MiningPlan {
        gpus: gpus.len(),
        cpu_threads: if role == CpuRole::All { threads } else { 0 },
        cpu_fallback_threads: if role == CpuRole::GpuFallback { threads } else { 0 },
        gpu_kernel: consensus::PowAlgo::ALL
            .map(|a| gpus.iter().any(|g| backend::device_has_kernel(g, a))),
    };
    let gpu_bc3 = plan.gpu_kernel[consensus::PowAlgo::Bc3Hash.index()];
    #[cfg(not(feature = "bc3hash"))]
    let _ = gpu_bc3;
    let gpu_names: Vec<String> = gpus.iter().map(|g| g.describe()).collect();
    let gpu_bc3_slots: Vec<bool> = gpus
        .iter()
        .map(|g| backend::device_has_kernel(g, consensus::PowAlgo::Bc3Hash))
        .collect();

    human!(
        "bc3-miner {} - {} GPU(s), {} CPU thread(s)",
        env!("CARGO_PKG_VERSION"),
        gpus.len(),
        plan.cpu_threads
    );
    for gpu in &gpus {
        human!("[gpu] found: {}", gpu.describe());
    }
    if plan.cpu_fallback_threads > 0 {
        human!(
            "[cpu] {} thread(s) standing by for jobs the GPU has no kernel for (BC3Hash)",
            plan.cpu_fallback_threads
        );
    }
    ipc::emit(&ipc::Event::Startup {
        version: env!("CARGO_PKG_VERSION").to_string(),
        backend: if gpus.is_empty() { "cpu".into() } else { "gpu".into() },
        gpus: gpus.iter().map(|g| g.describe()).collect(),
        cpu_threads: plan.cpu_threads,
        cpu_fallback_threads: plan.cpu_fallback_threads,
    });

    // The BC3Hash tables are built now, on their own thread, rather than by
    // the first hash: the fast ones take ~11 s, which would otherwise land in
    // the middle of mining - at the fork, the moment it matters. The same call
    // runs the build's self-test against the node's vectors. A GPU that mines
    // BC3Hash needs them too: the CPU checks its self-test and every hit, a
    // few hashes a second, which the compact tables do as well as the fast
    // ones. Without CPU threads or such a GPU, nothing here ever hashes
    // BC3Hash, so nothing is built.
    #[cfg(feature = "bc3hash")]
    {
        let mode = args.bc3hash_mode.unwrap_or(if gpu_bc3 {
            Bc3HashMode::Compact
        } else {
            Bc3HashMode::Fast
        });
        bc3hash::configure(mode.into());
        if threads > 0 || gpu_bc3 {
            std::thread::Builder::new()
                .name("bc3hash-init".into())
                .spawn(move || {
                    let t0 = std::time::Instant::now();
                    // No exit: SHA3-256t, on the GPU above all, mines on
                    // regardless. The CPU threads see the same result through
                    // worker::cpu_supports and leave BC3Hash alone - shares
                    // from a wrong build would only ever be rejected.
                    if let Err(e) = bc3hash::initialize() {
                        let message = match e {
                            bc3hash::InitError::Tables { .. } => format!(
                                "BC3Hash: NOT mining - {e} With little memory, start with \
                                 --bc3hash-mode compact."
                            ),
                            bc3hash::InitError::WrongResult { .. } => {
                                format!("BC3Hash: NOT mining - {e}")
                            }
                        };
                        eprintln!("bc3-miner: {message}");
                        ipc::emit(&ipc::Event::Status {
                            state: ipc::StatusState::Error,
                            message,
                        });
                        return;
                    }
                    human!(
                        "[cpu] BC3Hash tables ready and self-tested ({}, {:.0} MiB) in {:.1} s",
                        format!("{mode:?}").to_lowercase(),
                        bc3hash::table_bytes() as f64 / (1u64 << 20) as f64,
                        t0.elapsed().as_secs_f64()
                    );
                })
                .expect("could not start the BC3Hash table thread");
        }
    }

    // Bunden kö: se Shared::submit.
    let (submit_tx, submit_rx) = std::sync::mpsc::sync_channel(shared::SUBMIT_QUEUE);
    let shared = Arc::new(shared::Shared::new(submit_tx, plan));
    shared.set_intensity(args.intensity);
    if args.intensity < 100 {
        human!("[miner] intensity {}%", args.intensity);
    }
    #[cfg(feature = "bc3hash")]
    if let Some(window) = args.bc3_window {
        shared.set_bc3_window(window);
    }

    // CPU threads and GPUs share one common disjoint extranonce2 partition.
    let total_workers = threads + gpus.len();
    for i in 0..threads {
        let s = shared.clone();
        std::thread::Builder::new()
            .name(format!("worker-{i}"))
            .spawn(move || worker::run_worker(s, i, total_workers, role))
            .expect("could not start a worker thread");
    }
    // Counted here, before the threads start: a worker that has not been
    // scheduled yet must not look the same to the watchdog as one that died.
    shared
        .stats
        .gpu_workers_alive
        .store(gpus.len() as u32, std::sync::atomic::Ordering::Relaxed);
    for (g, gpu) in gpus.into_iter().enumerate() {
        let s = shared.clone();
        std::thread::Builder::new()
            .name(format!("gpu-{g}"))
            .spawn(move || gpu_worker::run_gpu_worker(s, gpu, g, threads + g, total_workers))
            .expect("could not start a GPU thread");
    }
    if !args.benchmark {
        let s = shared.clone();
        let interval = args.stats_interval;
        std::thread::spawn(move || stats::run_reporter(s, interval));
    }

    // Watchdog: a run where every GPU worker has stopped and the CPU does not
    // mine the current PoW either is not mining, however healthy the process
    // looks. It stays connected to the pool, the GUI keeps saying "Mining",
    // and the hashrate is zero - on a rented GPU host that is billed at GPU
    // prices. Exit instead, so a restart policy or an alert has something to
    // react to. Stand-in CPU threads only count while the PoW is theirs: on
    // BC3Hash they are the miner, and a dead GPU changes nothing.
    if plan.gpus > 0 {
        let s = shared.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            if s.stats.gpu_workers_alive.load(std::sync::atomic::Ordering::Relaxed) > 0 {
                continue;
            }
            let algo = s.current_algo();
            // Before the first job the PoW is unknown, and any CPU thread
            // might turn out to be the one mining it.
            let cpu_mining = match algo {
                Some(a) => s.devices_for(a).cpu,
                None => s.plan.cpu_threads + s.plan.cpu_fallback_threads > 0,
            };
            if cpu_mining {
                continue;
            }
            eprintln!(
                "bc3-miner: every GPU worker has stopped and the CPU is not mining {} - \n\
                 nothing is being mined. Exiting with an error so this is noticed.\n\
                 Common causes: a driver too old for the kernel, or a GPU that fell off the bus.",
                algo.map(|a| a.name()).unwrap_or("this job")
            );
            std::process::exit(1);
        });
    }

    if args.benchmark {
        #[cfg(feature = "bc3hash")]
        let windows = match args.bc3_window {
            Some(w) => vec![w],
            None if args.algo == BenchAlgo::Bc3hash && gpu_bc3 => BC3_WINDOWS.to_vec(),
            None => vec![shared.bc3_window()],
        };
        #[cfg(not(feature = "bc3hash"))]
        let windows = vec![shared.bc3_window()];
        let compare = windows.len() > 1;
        let code = bench::run(
            shared,
            submit_rx,
            bench::Config {
                algo: args.algo.into(),
                seconds: args.seconds.unwrap_or(if compare { 20 } else { 60 }),
                difficulty: args.difficulty,
                windows,
                gpu_names,
                gpu_bc3: gpu_bc3_slots,
                cpu_threads: threads,
            },
        );
        std::process::exit(code);
    }

    stratum::run_client(shared, submit_rx, stratum::StratumConfig { pool, user });
}

/// How many CPU threads to start, and what they mine.
///
/// "All cores" (0) means different things with and without a GPU: every GPU
/// has a feeder thread that must get CPU time to launch the next kernel in
/// time. If we let CPU mining take every core next to a busy GPU the feeders
/// starve and the GPU hashrate drops more than the CPU adds. So `--threads 0`
/// reserves one core per GPU plus one more.
///
/// The stand-ins are different. They only run when the GPU has nothing to do,
/// on BC3Hash, which has no kernel; then its feeder needs no core, and they
/// get all of them.
fn plan_cpu(
    backend: BackendKind,
    gpus: usize,
    threads: Option<usize>,
    all_cores: usize,
) -> (usize, CpuRole) {
    match (threads, gpus) {
        (Some(0), 0) => (all_cores, CpuRole::All),
        (Some(0), g) => (all_cores.saturating_sub(g + 1).max(1), CpuRole::All),
        (Some(n), _) => (n, CpuRole::All),
        // Pure CPU mining, as before.
        (None, 0) => (all_cores, CpuRole::All),
        // A GPU in auto: no CPU mining next to it, but every core stands by
        // for what it cannot do. Since the fork that is everything.
        (None, _) if backend == BackendKind::Auto => (all_cores, CpuRole::GpuFallback),
        // A GPU backend the user chose by name: the CPU stays out of it.
        (None, _) => (0, CpuRole::All),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fallback decision for every way the miner can be started. The case
    /// the fork hinges on: auto with a GPU and no --threads stands every core
    /// by for BC3Hash, while a GPU backend chosen by name leaves the CPU alone.
    #[test]
    fn cpu_threads_and_their_role() {
        use BackendKind::*;
        assert_eq!(plan_cpu(Auto, 1, None, 20), (20, CpuRole::GpuFallback));
        assert_eq!(plan_cpu(Auto, 2, None, 20), (20, CpuRole::GpuFallback));
        assert_eq!(plan_cpu(Cuda, 1, None, 20), (0, CpuRole::All));
        assert_eq!(plan_cpu(Opencl, 1, None, 20), (0, CpuRole::All));
        // --threads always means "mine with these, everything".
        assert_eq!(plan_cpu(Auto, 1, Some(4), 20), (4, CpuRole::All));
        assert_eq!(plan_cpu(Cuda, 1, Some(4), 20), (4, CpuRole::All));
        assert_eq!(plan_cpu(Auto, 2, Some(0), 20), (17, CpuRole::All));
        assert_eq!(plan_cpu(Auto, 2, Some(0), 2), (1, CpuRole::All));
        // No GPU: plain CPU mining on every core.
        assert_eq!(plan_cpu(Auto, 0, None, 20), (20, CpuRole::All));
        assert_eq!(plan_cpu(Cpu, 0, None, 20), (20, CpuRole::All));
        assert_eq!(plan_cpu(Cpu, 0, Some(0), 20), (20, CpuRole::All));
    }
}
