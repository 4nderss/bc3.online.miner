//! The BC3Hash table window of each GPU, chosen while it mines.
//!
//! A bigger secp256k1 table window means fewer point additions per scalar
//! multiplication - 16 digits at 16 bits, 13 at 20 - for more GPU memory:
//! 256 MiB of tables at 16 bits, 3.3 GiB at 20. On an RTX 4090 (24 GiB, 72 MiB
//! of L2, at its 450 W limit throughout) window 20 hashed 5.6 % faster than 16
//! (2.66 against 2.52 MH/s, 60 s runs), 19 3.6 % and 18 1.6 %; on an RTX 3050
//! Ti laptop (4 GiB) 19 was about 5 % faster. Whether a card gains depends on
//! its memory and its caches, and nobody can try every card, so the miner
//! measures rather than guesses: it mines at 16 at once, builds the tables of
//! the largest window that fits while it does, switches, and keeps that
//! window only if the card hashes faster with it. The choice is remembered
//! per card model, memory size, driver and miner version, so the next start
//! goes straight there; a new driver or miner version measures again.
//!
//! This module is the decision, driven by events and free of I/O but for the
//! file the choices are kept in - so that it can be tested without a GPU.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Every card starts here: small enough for any card, its tables built in
/// about a second.
pub const START_WINDOW: u32 = 16;

/// The windows tried above START_WINDOW, largest first. 21 has the same 13
/// digits as 20 and so no fewer additions; 22 (12 digits) would need 12.9 GiB
/// of tables, built in host memory first.
pub const BIGGER: [u32; 3] = [20, 19, 18];

/// GPU memory left free beside the tables: the kernel's own local memory
/// (3.4 KiB a thread, ~170 MiB on a big card), the launch buffers, the
/// driver, and whatever else runs on the card - a desktop, a browser.
const VRAM_SPARE: u64 = 1 << 30;
/// Host memory left free beside the tables while they are built: the CPU's
/// own BC3Hash tables, the GUI, the operating system.
const RAM_SPARE: u64 = 2 << 30;

/// How long each window is measured, in GPU time. A minute keeps one launch
/// (~0.2-0.4 s) under 1 % of a rate, and the RTX 4090's rates held within
/// 0.5 % over its 60 s runs.
pub const MEASURE: Duration = Duration::from_secs(60);
/// Mining before the first measurement starts: the card warms up, and a
/// power-limited card settles its clock.
pub const WARM_UP: Duration = Duration::from_secs(30);
/// After a switch, before the new window is measured.
pub const SETTLE: Duration = Duration::from_secs(10);
/// The bigger window must be at least this much faster to be kept: its
/// tables cost memory and start-up time, and measuring has a spread too.
pub const MIN_GAIN: f64 = 0.005;

/// Size of the GPU tables at `window` (bc3hash::gpu_tables): the three power
/// tables, then four curves of (256 / w rounded up) rows of 2^w - 1 points of
/// 64 bytes.
pub fn table_bytes(window: u32) -> u64 {
    let rows = u64::from(256u32.div_ceil(window));
    let pow = (8 * 256 + 16 * 256 * 2 + 32 * 256 * 4) * 8;
    pow + 4 * rows * ((1u64 << window) - 1) * 64
}

/// The largest window above START_WINDOW whose tables fit on the card and,
/// while they are built, in the host's memory - with room to spare in both.
/// None when nothing bigger fits, or the free host memory is unknown.
pub fn largest_fitting(free_vram: u64, avail_ram: Option<u64>) -> Option<u32> {
    let avail_ram = avail_ram?;
    BIGGER.iter().copied().find(|&w| {
        let t = table_bytes(w);
        t + VRAM_SPARE <= free_vram && t + RAM_SPARE <= avail_ram
    })
}

/// The host memory available to a new allocation, where the platform says.
pub fn available_ram() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        let mut s = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: a correctly sized, writable struct with its length set, as
        // the function requires; kernel32 is always linked on Windows.
        (unsafe { GlobalMemoryStatusEx(&mut s) } != 0).then_some(s.avail_phys)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// A window chosen for a card, and the rates it was chosen on (hashes per
/// second of GPU time; 0 when it was not measured).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub window: u32,
    pub start_rate: f64,
    pub tried: u32,
    pub tried_rate: f64,
}

/// What a choice is remembered by: a different card, memory size, driver or
/// miner version measures again.
pub fn key(gpu: &str, total_vram: u64, driver: Option<&str>) -> String {
    format!(
        "{gpu} | {} MiB | driver {} | bc3-miner {}",
        total_vram >> 20,
        driver.unwrap_or("unknown"),
        env!("CARGO_PKG_VERSION")
    )
}

/// The file the choices are kept in: BC3_TUNING_FILE, or tuning.json in the
/// user's cache directory (LOCALAPPDATA on Windows, XDG_CACHE_HOME or
/// ~/.cache elsewhere).
pub fn file() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("BC3_TUNING_FILE") {
        return Some(PathBuf::from(p));
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    };
    Some(base?.join("bc3-miner").join("tuning.json"))
}

static FILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The remembered choice for `key`, if any. A missing or unreadable file is
/// no choice: the card measures again.
pub fn recall(key: &str) -> Option<Choice> {
    recall_in(&file()?, key)
}

fn recall_in(path: &std::path::Path, key: &str) -> Option<Choice> {
    let _guard = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    read_all(path).remove(key)
}

/// Remember `choice` for `key`, keeping the other cards' choices. Failing to
/// write only means measuring again next time.
pub fn remember(key: &str, choice: Choice) -> Result<(), String> {
    remember_in(&file().ok_or("no cache directory")?, key, choice)
}

fn remember_in(path: &std::path::Path, key: &str, choice: Choice) -> Result<(), String> {
    let _guard = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut all = read_all(path);
    all.insert(key.to_string(), choice);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(&all).map_err(|e| e.to_string())?;
    // Written aside and renamed, so a crash cannot leave half a file.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_all(path: &std::path::Path) -> HashMap<String, Choice> {
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

/// What the worker should do next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    /// Nothing.
    Wait,
    /// Start building the tables of this window in the background (and
    /// compiling its kernel ahead), and say when they are ready.
    Build(u32),
    /// Switch the card to this window now: its tables are ready.
    Switch(u32),
    /// The tuning is over; remember this.
    Done(Choice),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    /// Mining at START_WINDOW, its warm-up not over yet.
    WarmUp { mined: Duration },
    /// Measuring START_WINDOW (the target's tables are being built meanwhile).
    MeasureStart { rate: Rate },
    /// Start measured; waiting for the target's tables.
    AwaitBuilt { start: f64 },
    /// Switched to the target, letting it settle.
    Settle { start: f64, mined: Duration },
    /// Measuring the target.
    MeasureTarget { start: f64, rate: Rate },
    /// Going back to START_WINDOW, the target having lost.
    Revert { choice: Choice },
    /// Going straight to a remembered window, unmeasured.
    Recalled { window: u32 },
    Finished,
}

/// Hashes over GPU time, accumulated.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Rate {
    hashes: u64,
    busy: Duration,
}

impl Rate {
    fn add(&mut self, hashes: u64, busy: Duration) {
        self.hashes += hashes;
        self.busy += busy;
    }
    fn per_second(&self) -> f64 {
        self.hashes as f64 / self.busy.as_secs_f64().max(1e-9)
    }
}

/// The tuning of one card, from its first BC3Hash launch.
#[derive(Debug)]
pub struct Tuner {
    target: u32,
    built: bool,
    phase: Phase,
}

impl Tuner {
    /// `fitting`: the largest window above START_WINDOW that fits (see
    /// `largest_fitting`); `recalled`: what was chosen for this card before.
    /// Returns the tuner and its first action. None when there is nothing
    /// to tune: nothing bigger fits, or 16 was the choice before.
    pub fn start(fitting: Option<u32>, recalled: Option<Choice>) -> Option<(Self, Action)> {
        let fitting = fitting?;
        if let Some(c) = recalled {
            // A remembered bigger window still has to fit today.
            return (c.window > START_WINDOW && c.window <= fitting).then(|| {
                let t = Tuner { target: c.window, built: false, phase: Phase::Recalled { window: c.window } };
                (t, Action::Build(c.window))
            });
        }
        let t = Tuner { target: fitting, built: false, phase: Phase::WarmUp { mined: Duration::ZERO } };
        // The tables take seconds to build; the warm-up and the measurement
        // of the start window run meanwhile.
        Some((t, Action::Build(fitting)))
    }

    /// The window being tried, or gone to.
    pub fn target(&self) -> u32 {
        self.target
    }

    /// A launch came back: `hashes` hashed at `window` in `busy` of GPU time.
    pub fn launched(&mut self, window: u32, hashes: u64, busy: Duration) -> Action {
        match &mut self.phase {
            Phase::WarmUp { mined } if window == START_WINDOW => {
                *mined += busy;
                if *mined >= WARM_UP {
                    self.phase = Phase::MeasureStart { rate: Rate::default() };
                }
                Action::Wait
            }
            Phase::MeasureStart { rate } if window == START_WINDOW => {
                rate.add(hashes, busy);
                if rate.busy >= MEASURE {
                    let start = rate.per_second();
                    self.phase = Phase::AwaitBuilt { start };
                    if self.built {
                        return self.switch_to_target(start);
                    }
                }
                Action::Wait
            }
            Phase::Settle { start, mined } if window == self.target => {
                *mined += busy;
                if *mined >= SETTLE {
                    let start = *start;
                    self.phase = Phase::MeasureTarget { start, rate: Rate::default() };
                }
                Action::Wait
            }
            Phase::MeasureTarget { start, rate } if window == self.target => {
                rate.add(hashes, busy);
                if rate.busy < MEASURE {
                    return Action::Wait;
                }
                let (start, tried_rate) = (*start, rate.per_second());
                let keep = tried_rate > start * (1.0 + MIN_GAIN);
                let choice = Choice {
                    window: if keep { self.target } else { START_WINDOW },
                    start_rate: start,
                    tried: self.target,
                    tried_rate,
                };
                if keep {
                    self.phase = Phase::Finished;
                    Action::Done(choice)
                } else {
                    self.phase = Phase::Revert { choice };
                    Action::Switch(START_WINDOW)
                }
            }
            Phase::Revert { choice } if window == START_WINDOW => {
                let choice = *choice;
                self.phase = Phase::Finished;
                Action::Done(choice)
            }
            _ => Action::Wait,
        }
    }

    /// The target's tables are built (`ok`) or could not be.
    pub fn built(&mut self, ok: bool) -> Action {
        if !ok {
            // Nothing to switch to: stay at the start window, and do not
            // remember it - the next start may have the memory.
            self.phase = Phase::Finished;
            return Action::Wait;
        }
        self.built = true;
        match self.phase {
            Phase::AwaitBuilt { start } => self.switch_to_target(start),
            Phase::Recalled { window } => {
                self.phase = Phase::Finished;
                Action::Switch(window)
            }
            _ => Action::Wait,
        }
    }

    /// The card switched to `window` (`ok`), or the switch failed - its
    /// self-test, its memory. A failed target falls back to the start window
    /// and is remembered as such.
    pub fn switched(&mut self, window: u32, ok: bool) -> Action {
        if ok || window == START_WINDOW {
            return Action::Wait;
        }
        self.phase = Phase::Finished;
        Action::Done(Choice { window: START_WINDOW, start_rate: 0.0, tried: window, tried_rate: 0.0 })
    }

    /// Whether the tuning is over.
    pub fn finished(&self) -> bool {
        self.phase == Phase::Finished
    }

    fn switch_to_target(&mut self, start: f64) -> Action {
        self.phase = Phase::Settle { start, mined: Duration::ZERO };
        Action::Switch(self.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn table_sizes_match_the_gpu_tables() {
        // bc3hash's gpu_table_build_times: 256.3, 960.3, 1792.3, 3328.3 MiB.
        let mib = |w| table_bytes(w) as f64 / (1u64 << 20) as f64;
        assert!((mib(16) - 256.3).abs() < 0.1);
        assert!((mib(18) - 960.3).abs() < 0.1);
        assert!((mib(19) - 1792.3).abs() < 0.1);
        assert!((mib(20) - 3328.3).abs() < 0.1);
    }

    #[test]
    fn the_largest_window_that_fits_both_memories() {
        // A 24 GiB card with 22 free and plenty of RAM: 20.
        assert_eq!(largest_fitting(22 * GIB, Some(32 * GIB)), Some(20));
        // 4 GiB free: 20 needs 3.25 + 1 spare, 19 needs 1.75 + 1.
        assert_eq!(largest_fitting(4 * GIB, Some(32 * GIB)), Some(19));
        // Plenty of VRAM, but 4 GiB of RAM: 19 needs 1.75 + 2.
        assert_eq!(largest_fitting(22 * GIB, Some(4 * GIB)), Some(19));
        // 3 GiB of RAM: 18 (0.94 + 2).
        assert_eq!(largest_fitting(22 * GIB, Some(3 * GIB)), Some(18));
        assert_eq!(largest_fitting(22 * GIB, Some(2 * GIB)), None);
        // 2 GiB free on the card: 18 (0.94 + 1); 1.5 GiB: nothing above 16.
        assert_eq!(largest_fitting(2 * GIB, Some(32 * GIB)), Some(18));
        assert_eq!(largest_fitting(3 * GIB / 2, Some(32 * GIB)), None);
        // Unknown host memory: do not risk it.
        assert_eq!(largest_fitting(22 * GIB, None), None);
    }

    /// Drive a tuner through launches of `busy` each at the window the card
    /// is at, applying its actions; `built_after` launches until the tables
    /// are ready. Returns the actions seen.
    fn run(tuner: &mut Tuner, first: Action, rate_at: impl Fn(u32) -> f64, built_after: usize) -> Vec<Action> {
        let busy = Duration::from_millis(250);
        let mut window = START_WINDOW;
        let mut actions = vec![first];
        for n in 0..2000 {
            if n == built_after {
                let a = tuner.built(true);
                if a != Action::Wait {
                    actions.push(a);
                }
                if let Action::Switch(w) = a {
                    window = w;
                    assert_eq!(tuner.switched(w, true), Action::Wait);
                }
            }
            let hashes = (rate_at(window) * busy.as_secs_f64()) as u64;
            let a = tuner.launched(window, hashes, busy);
            if a != Action::Wait {
                actions.push(a);
            }
            match a {
                Action::Switch(w) => {
                    window = w;
                    assert_eq!(tuner.switched(w, true), Action::Wait);
                }
                Action::Done(_) => break,
                _ => {}
            }
        }
        actions
    }

    #[test]
    fn a_faster_bigger_window_is_kept() {
        let (mut t, first) = Tuner::start(Some(20), None).unwrap();
        assert_eq!(first, Action::Build(20));
        let actions = run(&mut t, first, |w| if w == 20 { 2.66e6 } else { 2.52e6 }, 40);
        assert_eq!(actions[1], Action::Switch(20));
        let Action::Done(c) = *actions.last().unwrap() else { panic!("{actions:?}") };
        assert_eq!(c.window, 20);
        assert!((c.start_rate - 2.52e6).abs() < 1e3 && (c.tried_rate - 2.66e6).abs() < 1e3);
        assert!(t.finished());
    }

    #[test]
    fn a_window_that_is_not_faster_goes_back() {
        let (mut t, first) = Tuner::start(Some(19), None).unwrap();
        // 0.2 % faster is within the spread: not worth 1.75 GiB.
        let actions = run(&mut t, first, |w| if w == 19 { 2.505e6 } else { 2.50e6 }, 10);
        assert_eq!(actions[1], Action::Switch(19));
        assert_eq!(actions[2], Action::Switch(START_WINDOW));
        let Action::Done(c) = *actions.last().unwrap() else { panic!("{actions:?}") };
        assert_eq!((c.window, c.tried), (START_WINDOW, 19));
    }

    #[test]
    fn slow_tables_are_waited_for() {
        // The start window is measured long before the tables are ready.
        let (mut t, first) = Tuner::start(Some(20), None).unwrap();
        let actions = run(&mut t, first, |w| if w == 20 { 2.6e6 } else { 2.5e6 }, 1500);
        assert_eq!(actions[1], Action::Switch(20));
        assert!(matches!(actions.last(), Some(Action::Done(Choice { window: 20, .. }))));
    }

    #[test]
    fn a_remembered_window_is_gone_to_unmeasured() {
        let c = Choice { window: 19, start_rate: 1.0, tried: 19, tried_rate: 2.0 };
        let (mut t, first) = Tuner::start(Some(20), Some(c)).unwrap();
        assert_eq!(first, Action::Build(19));
        assert_eq!(t.launched(START_WINDOW, 1000, Duration::from_secs(100)), Action::Wait);
        assert_eq!(t.built(true), Action::Switch(19));
        assert!(t.finished());
        // ...but only if it still fits.
        assert!(Tuner::start(Some(18), Some(c)).is_none());
        // And a remembered 16 means nothing to do.
        let c16 = Choice { window: START_WINDOW, ..c };
        assert!(Tuner::start(Some(20), Some(c16)).is_none());
    }

    #[test]
    fn nothing_fitting_means_no_tuning() {
        assert!(Tuner::start(None, None).is_none());
    }

    #[test]
    fn a_failed_build_or_switch_stays_at_the_start_window() {
        let (mut t, _) = Tuner::start(Some(20), None).unwrap();
        assert_eq!(t.built(false), Action::Wait);
        assert!(t.finished());
        let (mut t, _) = Tuner::start(Some(20), None).unwrap();
        assert!(matches!(t.switched(20, false), Action::Done(Choice { window: START_WINDOW, tried: 20, .. })));
    }

    #[test]
    fn choices_survive_a_round_trip_through_the_file() {
        let dir = std::env::temp_dir().join(format!("bc3-tuning-test-{}", std::process::id()));
        let path = dir.join("tuning.json");
        let a = Choice { window: 20, start_rate: 2.52e6, tried: 20, tried_rate: 2.66e6 };
        let ka = key("NVIDIA GeForce RTX 4090", 24 * GIB, Some("596.49"));
        let kb = key("NVIDIA GeForce RTX 3050 Ti Laptop GPU", 4 * GIB, None);
        assert_eq!(recall_in(&path, &ka), None);
        remember_in(&path, &ka, a).unwrap();
        remember_in(&path, &kb, Choice { window: 19, ..a }).unwrap();
        assert_eq!(recall_in(&path, &ka), Some(a));
        assert_eq!(recall_in(&path, &kb).map(|c| c.window), Some(19));
        assert!(ka.contains("24576 MiB") && ka.contains("596.49") && ka.contains(env!("CARGO_PKG_VERSION")));
        // A file that is not ours is no choice, not an error.
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(recall_in(&path, &ka), None);
        let _ = std::fs::remove_dir_all(dir);
    }
}
