//! The internal tools: `--benchmark`, `--solve`, and the knobs only they
//! need - the GPU's BC3Hash table window, the CPU's table size, the depth of
//! the launch queue.
//!
//! Compiled in only with the `internal-tools` feature, which no release
//! enables. They are for measuring the miner and for one-off jobs, not for
//! the people running it: `--solve` mines a block template that is not the
//! pool's, and the knobs exist to be compared, with defaults that are already
//! the measured best. A release build has none of these options, not even
//! hidden - clap rejects them as unexpected arguments. See AGENTS.md.

use crate::shared::{FoundShare, Shared};
use crate::{backend, bench, consensus, solve};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

/// The options of the internal tools, flattened into the miner's own. Not
/// called `Args`: clap names the argument group of a derived struct after it,
/// and a second `Args` panics in clap's debug checks.
#[derive(clap::Args)]
pub struct Tools {
    /// BC3Hash lookup tables for the CPU: fast (~256 MiB of RAM) or compact
    /// (~7 MiB, 10-20 % slower). Both find exactly the same shares. Default:
    /// compact when a GPU mines BC3Hash and the CPU only checks its shares,
    /// fast otherwise.
    #[cfg(feature = "bc3hash")]
    #[arg(long, value_enum, env = "BC3_BC3HASH_MODE")]
    bc3hash_mode: Option<Bc3HashMode>,

    /// Table window of the GPU's BC3Hash kernel in bits: 16 (256 MiB of GPU
    /// memory), 18 (0.9 GiB), 19 (1.8 GiB), 20 (3.3 GiB), 22 (8 GiB) or 24
    /// (28 GiB) - 22 and 24 on CUDA cards only. Fixes the window: without it
    /// every card starts at 16 and tunes its own while it mines
    /// (src/tuning.rs). `--benchmark` without it compares them all.
    #[cfg(feature = "bc3hash")]
    #[arg(long, env = "BC3_BC3HASH_WINDOW", value_parser = parse_bc3_window)]
    bc3_window: Option<u32>,

    /// Benchmark instead of mining: an in-process job source serves random
    /// jobs to the normal workers - same dispatch, same kernels, every GPU
    /// hit re-checked - and checks every share with the CPU reference. No
    /// pool, no network, no address needed.
    #[arg(long)]
    benchmark: bool,

    /// Solve one fixed block template offline instead of mining for a pool: a
    /// job file with the block's height, previous hash, version, time, bits,
    /// coinbase text and outputs, as published for the hard fork's first
    /// block. No pool, no node, no network; the whole block is printed and
    /// saved to a file when it is found.
    #[arg(long, value_name = "JOB.json", conflicts_with = "benchmark")]
    solve: Option<std::path::PathBuf>,

    /// Benchmark: leave the window to each card's tuning, as a release does,
    /// instead of comparing windows - to watch the tuning without a pool.
    /// Set BC3_TUNING_FILE to a new file to measure instead of recalling.
    #[cfg(feature = "bc3hash")]
    #[arg(long, requires = "benchmark", conflicts_with = "bc3_window")]
    tune: bool,

    /// Benchmark: seconds to measure. Comparing the BC3Hash windows, each
    /// window runs twice for this long (default 20); with --tune 540, time
    /// for the tuning to finish; otherwise 60.
    #[arg(long)]
    seconds: Option<u64>,

    /// Benchmark and --solve: share difficulty. By default it is set from the
    /// measured hashrate so that a share comes every few seconds.
    #[arg(long)]
    difficulty: Option<f64>,

    /// Kernel launches each GPU keeps queued. 2 keeps the card busy while the
    /// host collects and checks the previous launch; 1 is the old
    /// one-at-a-time loop, kept to compare the two with --benchmark.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=2),
          hide = true)]
    gpu_queue: u32,
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
const BC3_WINDOWS: [u32; 6] = [16, 18, 19, 20, 22, 24];

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

/// What the tools report on, per GPU slot - taken in main before the worker
/// threads own the devices.
pub struct Devices {
    names: Vec<String>,
    /// Per GPU slot: does it have a BC3Hash kernel?
    bc3: Vec<bool>,
    cpu_threads: usize,
}

impl Devices {
    pub fn new(gpus: &[backend::GpuDevice], cpu_threads: usize) -> Self {
        Self {
            names: gpus.iter().map(|g| g.describe()).collect(),
            bc3: gpus
                .iter()
                .map(|g| backend::device_has_kernel(g, consensus::PowAlgo::Bc3Hash))
                .collect(),
            cpu_threads,
        }
    }
}

impl Tools {
    /// A run that mines for no pool - the benchmark or `--solve`: it needs no
    /// address and reports no pool statistics.
    pub fn offline(&self) -> bool {
        self.benchmark || self.solve.is_some()
    }

    /// The CPU's BC3Hash tables, where the command line chose them.
    #[cfg(feature = "bc3hash")]
    pub fn bc3hash_mode(&self) -> Option<bc3hash::Mode> {
        self.bc3hash_mode.map(Into::into)
    }

    /// The GPU settings, before the workers start.
    pub fn configure(&self, shared: &Shared) {
        shared.set_gpu_queue(self.gpu_queue);
        #[cfg(feature = "bc3hash")]
        if let Some(window) = self.bc3_window {
            shared.set_bc3_window(window);
        }
    }

    /// Run `--solve` or `--benchmark` (only when `offline`) on the workers
    /// main has started; the process exit code.
    pub fn run(&self, shared: Arc<Shared>, rx: Receiver<FoundShare>, devices: Devices) -> i32 {
        if let Some(path) = self.solve.clone() {
            return solve::run(
                shared,
                rx,
                solve::Config {
                    path,
                    difficulty: self.difficulty,
                    gpus: devices.names.len(),
                    cpu_threads: devices.cpu_threads,
                },
            );
        }
        // Window 0: the benchmark leaves the window alone, and each card
        // tunes its own (--tune).
        #[cfg(feature = "bc3hash")]
        let windows = match self.bc3_window {
            Some(w) => vec![w],
            None if self.tune => vec![0],
            None if devices.bc3.contains(&true) => BC3_WINDOWS.to_vec(),
            None => vec![shared.bc3_window()],
        };
        #[cfg(not(feature = "bc3hash"))]
        let windows = vec![shared.bc3_window()];
        let compare = windows.len() > 1;
        // The tuning takes about seven minutes: the bigger window's tables
        // and kernel, a minute and a half of warm-up after them, a minute
        // each of 16, it and 16 again, half a minute's settling after each
        // switch.
        let tuning = windows == [0];
        bench::run(
            shared,
            rx,
            bench::Config {
                seconds: self.seconds.unwrap_or(if compare { 20 } else if tuning { 540 } else { 60 }),
                difficulty: self.difficulty,
                windows,
                gpu_names: devices.names,
                gpu_bc3: devices.bc3,
                cpu_threads: devices.cpu_threads,
            },
        )
    }
}
