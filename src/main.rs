//! bc3-miner - CPU/GPU miner for BC3 (the bc3.online client).
//!
//! The GPU backends (CUDA via NVRTC, OpenCL) and the CPU threads share the
//! same job pipeline: the stratum client publishes jobs, the workers
//! partition the extranonce2 space disjointly and report shares on the same
//! channel.

mod backend;
// The internal tools: only with the `internal-tools` feature - see internal.rs.
#[cfg(feature = "internal-tools")]
mod bench;
mod consensus;
mod crash;
mod gpu_worker;
#[cfg(feature = "internal-tools")]
mod internal;
mod ipc;
mod telemetry;
#[cfg(feature = "bc3hash")]
mod tuning;
mod shared;
#[cfg(feature = "internal-tools")]
mod solve;
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

    /// Password sent with the address when logging in to the pool.
    /// bc3.online ignores it; other pools and proxies may want one.
    #[arg(short = 'p', long, visible_alias = "pass", default_value = "x",
          env = "BC3_PASSWORD", hide_env_values = true)]
    password: String,

    /// Backend: auto = CUDA if possible, else OpenCL, else CPU. In auto the
    /// CPU also mines whatever the GPU cannot (BC3Hash on a card whose
    /// BC3Hash kernel failed its self-test); with cuda or opencl it mines
    /// only if --threads asks it to.
    #[arg(long, value_enum, default_value_t = BackendKind::Auto, env = "BC3_BACKEND")]
    backend: BackendKind,

    /// Use only one specific GPU (index into the detected list).
    #[arg(long, env = "BC3_GPU_ID")]
    gpu_id: Option<usize>,

    /// Number of CPU threads (0 = all cores), mining every job. Without the
    /// flag a GPU in --backend auto leaves the CPU idle unless the GPU cannot
    /// mine (its BC3Hash kernel failed), and then mines on all cores; with
    /// --backend cuda or opencl the CPU does not mine at all.
    #[arg(long, env = "BC3_THREADS")]
    threads: Option<usize>,

    // --benchmark, --solve and their options - internal builds only. Here,
    // not at the end, so such a build lists its options in the order they
    // always had.
    #[cfg(feature = "internal-tools")]
    #[command(flatten)]
    internal: internal::Tools,

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
    const VARS: &[&str] = &[
        "BC3_WALLET", "BC3_WORKER", "BC3_USER", "BC3_MODE", "BC3_POOL",
        "BC3_PASSWORD", "BC3_BACKEND", "BC3_THREADS", "BC3_INTENSITY", "BC3_GPU_ID",
        #[cfg(feature = "internal-tools")]
        "BC3_BC3HASH_MODE",
        #[cfg(feature = "internal-tools")]
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
    // First: a crash anywhere after this - our code, the node's C++, a
    // driver library - says where on stderr (see crash.rs).
    crash::install();
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

    // Mining for no pool: the benchmark or --solve of an internal build.
    #[cfg(feature = "internal-tools")]
    let offline = args.internal.offline();
    #[cfg(not(feature = "internal-tools"))]
    let offline = false;

    // --user wins; otherwise build it from --wallet and --rig, which is the
    // shape a container environment hands us. Offline needs neither.
    let user = args.user.clone().or_else(|| {
        args.wallet.as_ref().map(|w| match args.rig.as_deref() {
            Some(r) if !r.is_empty() => format!("{w}.{r}"),
            _ => w.clone(),
        })
    });
    let user = match user {
        Some(user) => user,
        None if offline => String::new(),
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
    // What the internal tools report on, taken before the workers own the GPUs.
    #[cfg(feature = "internal-tools")]
    let devices = internal::Devices::new(&gpus, threads);

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
            "[cpu] {} thread(s) standing by, to mine BC3Hash should a GPU's kernel fail",
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
        let mode = if gpu_bc3 { bc3hash::Mode::Compact } else { bc3hash::Mode::Fast };
        #[cfg(feature = "internal-tools")]
        let mode = args.internal.bc3hash_mode().unwrap_or(mode);
        bc3hash::configure(mode);
        if threads > 0 || gpu_bc3 {
            std::thread::Builder::new()
                .name("bc3hash-init".into())
                .spawn(move || {
                    let t0 = std::time::Instant::now();
                    // No exit: the error stays on screen, in the log and in
                    // the GUI, while nothing is mined. The CPU threads see the
                    // same result through worker::cpu_supports and leave
                    // BC3Hash alone, and so do the GPUs, which need this
                    // reference for their self-test and for every hit -
                    // shares from a wrong build would only ever be rejected.
                    if let Err(e) = bc3hash::initialize() {
                        use bc3hash::InitError::{Tables, WrongResult};
                        let hint = match e {
                            // Only an internal build can choose the tables.
                            Tables { .. } if cfg!(feature = "internal-tools") => {
                                " With little memory, start with --bc3hash-mode compact."
                            }
                            Tables { .. } | WrongResult { .. } => "",
                        };
                        let message = format!("BC3Hash: NOT mining - {e}{hint}");
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

    // Bounded queue: see Shared::submit.
    let (submit_tx, submit_rx) = std::sync::mpsc::sync_channel(shared::SUBMIT_QUEUE);
    let shared = Arc::new(shared::Shared::new(submit_tx, plan));
    shared.set_intensity(args.intensity);
    if args.intensity < 100 {
        human!("[miner] intensity {}%", args.intensity);
    }
    #[cfg(feature = "internal-tools")]
    args.internal.configure(&shared);
    if !gpus.is_empty() && shared.gpu_queue() != backend::QUEUE_DEPTH {
        human!("[gpu] {} kernel launch(es) queued per GPU", shared.gpu_queue());
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
    // Taken before the workers own the devices: see stratum::suggested_difficulty.
    let suggest_difficulty =
        stratum::suggested_difficulty(gpus.iter().filter(|g| g.is_cuda()).count(), args.intensity);
    // The stats thread loads NVML only when an NVIDIA card mines: see
    // Telemetry::without_gpu.
    let nvidia = gpus.iter().any(|g| g.is_cuda());
    // For the stall guard's messages, by slot: the workers own the devices.
    let gpu_names: Vec<String> = gpus.iter().map(|g| g.describe()).collect();
    for (g, gpu) in gpus.into_iter().enumerate() {
        let s = shared.clone();
        std::thread::Builder::new()
            .name(format!("gpu-{g}"))
            .spawn(move || gpu_worker::run_gpu_worker(s, gpu, g, threads + g, total_workers))
            .expect("could not start a GPU thread");
    }
    if !offline {
        let s = shared.clone();
        let interval = args.stats_interval;
        std::thread::spawn(move || stats::run_reporter(s, interval, nvidia));
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

    // Stall guard: a GPU that holds the PoW and finishes nothing gives it to
    // the stand-in CPU threads until it hashes again (see
    // Shared::check_gpu_stalls) - and says so on stderr, which the GUI shows
    // in its Activity log.
    if plan.gpus > 0 {
        let s = shared.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            for change in s.check_gpu_stalls(s.now_us(), shared::GPU_STALL_SECS * 1_000_000) {
                let algo = s.current_algo();
                let cpu = algo.is_some_and(|a| s.devices_for(a).cpu);
                let name = algo.map(|a| a.name()).unwrap_or("the job");
                eprintln!("{}", stall_message(change, &gpu_names, name, cpu));
            }
        });
    }

    #[cfg(feature = "internal-tools")]
    if offline {
        std::process::exit(args.internal.run(shared, submit_rx, devices));
    }

    let password = args.password.clone();
    stratum::run_client(shared, submit_rx, stratum::StratumConfig { pool, user, password, suggest_difficulty });
}

/// What the user reads when the stall guard sets a GPU aside or takes it back.
/// `cpu`: whether CPU threads mine `algo` while the card does not.
fn stall_message(change: shared::GpuStall, names: &[String], algo: &str, cpu: bool) -> String {
    let card = |slot: usize| names.get(slot).cloned().unwrap_or_else(|| format!("GPU {slot}"));
    match change {
        shared::GpuStall::Stalled { slot, idle_secs } => format!(
            "[gpu] {}: no launch finished in {idle_secs} s - its kernel is still loading, or the card \
             or its driver stopped answering. {} It takes {algo} back as soon as a launch finishes. \
             If this keeps happening, report it with this log.",
            card(slot),
            if cpu {
                format!("The CPU mines {algo} meanwhile.")
            } else {
                "No CPU threads stand in for it (they do in --backend auto).".to_string()
            }
        ),
        shared::GpuStall::Resumed { slot } => {
            format!("[gpu] {}: a launch finished again - it mines {algo} again.", card(slot))
        }
    }
}

/// How many CPU threads to start, and what they mine.
///
/// "All cores" (0) means different things with and without a GPU: every GPU
/// has a feeder thread that must get CPU time to launch the next kernel in
/// time. If we let CPU mining take every core next to a busy GPU the feeders
/// starve and the GPU hashrate drops more than the CPU adds. So `--threads 0`
/// reserves one core per GPU plus one more.
///
/// The stand-ins are different. They only run when the GPU has nothing to do -
/// its BC3Hash kernel failed - and then its feeder needs no core, and they
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
        // for what it cannot do - BC3Hash, should its kernel fail.
        (None, _) if backend == BackendKind::Auto => (all_cores, CpuRole::GpuFallback),
        // A GPU backend the user chose by name: the CPU stays out of it.
        (None, _) => (0, CpuRole::All),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fallback decision for every way the miner can be started. The case
    /// that matters most: auto with a GPU and no --threads stands every core
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

    /// Mining-OS templates and other miners' command lines pass `-p` (issue
    /// #2), and clap refuses an option it does not know - so all three
    /// spellings must parse, and none given is the "x" every pool accepts.
    #[test]
    fn the_pool_password_takes_the_usual_spellings() {
        let password = |argv: &[&str]| {
            let argv = [&["bc3-miner", "--user", "bc1qtest"], argv].concat();
            Args::try_parse_from(argv).unwrap().password
        };
        assert_eq!(password(&[]), "x");
        assert_eq!(password(&["-p", "d=0.5"]), "d=0.5");
        assert_eq!(password(&["--password", "secret"]), "secret");
        assert_eq!(password(&["--pass", "secret"]), "secret");
    }
}
