//! The BC3Hash table window of each GPU, chosen while it mines.
//!
//! A bigger secp256k1 table window means fewer point additions per scalar
//! multiplication - 16 digits at 16 bits, 13 at 20, 11 at 24 - for more GPU
//! memory: 256 MiB of tables at 16 bits, 3.3 GiB at 20, 28 GiB at 24. On an
//! RTX 4090 (24 GiB, 72 MiB of L2, at its 450 W limit throughout) window 20
//! hashed 5.6 % faster than 16 (2.66 against 2.52 MH/s, 60 s runs), 19 3.6 %
//! and 18 1.6 % - about 2 % a digit; on an RTX 3050 Ti laptop (4 GiB) 19 was
//! 2-5 % faster. Whether a card gains depends on its memory and its caches,
//! and nobody can try every card, so the miner measures rather than guesses:
//! it mines at 16 at once, then measures 16, the largest window that fits,
//! and 16 again, and keeps the bigger window only if the card hashes faster
//! with it than with 16 on average. The choice is remembered per card model,
//! memory size, driver and miner version, so the next start goes straight
//! there; a new driver or miner version measures again.
//!
//! Only CUDA cards tune (see `MiningBackend::memory`); the others stay at 16.
//! A CUDA card builds its tables itself, at the switch (the kernel's
//! `bc3hash_build_row`), so a window needs the card's memory and nothing of
//! the host's.
//!
//! This module is the decision, driven by events and free of I/O but for the
//! file the choices are kept in - so that it can be tested without a GPU -
//! and the tables' layout, which decides their size.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Every card starts here: small enough for any card, its tables built in
/// well under a second.
pub const START_WINDOW: u32 = 16;

/// The windows tried above START_WINDOW, largest first. 21 has the same 13
/// digits as 20 and so no fewer additions; 22 and 24 are mixed schedules
/// (`digits`), 12 and 11 digits in 8 and 28 GiB - a 32 GB card for 24.
pub const BIGGER: [u32; 5] = [24, 22, 20, 19, 18];

/// GPU memory left free beside the tables, of what is free while the card
/// mines at the start window - its local memory and launch buffers already
/// taken: room for whatever else comes to run on the card, a desktop or a
/// browser, and for the driver.
const VRAM_SPARE: u64 = 1 << 30;

/// How long each window is measured, in GPU time. A minute keeps one launch
/// (~0.2-0.4 s) under 1 % of a rate, and the RTX 4090's rates held within
/// 0.5 % over its 60 s runs.
pub const MEASURE: Duration = Duration::from_secs(60);
/// Mining between the bigger window being ready - its kernel compiled ahead,
/// where the card has no machine code for it - and the first measurement:
/// the card warms up, and its clock falls with it. In 2.1 the host built the
/// bigger window's tables meanwhile, on every core, and in a laptop the CPU
/// and the GPU share one power budget and one cooling system, which the GPU
/// gets back slowly: an RTX 3050 Ti laptop (i9-12900HK, the GPU holding at
/// 74 C) dropped from 40 to 27 W while the build ran, and was back at 40 W
/// only 90 s after it. Its first measurement of 16 right after the build
/// came out 20 % low, and window 19 looked 11.5 % faster; with this warm-up
/// the tuning measured +3.7 %, where alternated runs give 2-5 %. The card
/// builds its tables itself now, at the switch.
pub const WARM_UP: Duration = Duration::from_secs(90);
/// After a switch, before the new window is measured. The switch idles the
/// card for a second or two, and a power-limited card comes back faster than
/// it settles: an RTX 3050 Ti laptop ran 2 % faster for half a minute.
pub const SETTLE: Duration = Duration::from_secs(30);
/// The bigger window must be at least this much faster to be kept: its
/// tables cost memory and start-up time, and measuring has a spread too.
pub const MIN_GAIN: f64 = 0.005;
/// How far the two measurements of the start window - before the bigger
/// window and after it - may differ for the choice to be remembered. Their
/// mean cancels a clock that drifts as the card warms up; a bigger
/// difference means something else moved the rate, so the choice is used
/// but the next start measures again. An RTX 3050 Ti laptop drifted 2-4 %
/// in a few minutes of mining; a measurement the build's power sharing
/// still held down (see WARM_UP) was 20 % off.
pub const MAX_DRIFT: f64 = 0.05;

/// The widths of the digits a 256-bit scalar is cut into at `window`, least
/// significant first: one row of each curve's table per digit, in this order.
/// A uniform window w has 256 / w digits, rounded up, of w bits, the last one
/// holding what is left; 22 and 24 are mixed schedules, digits of two widths
/// that cover the 256 bits exactly - the fewest digits for the memory: 12 of
/// 21 and 22 bits are 8 GiB of tables where 12 of 22 bits would be 12 GiB.
/// What the kernel is built with for each (`BC3_WINDOW`, `BC3_WIDE`) is in
/// tools/ptx.sh; the card's self-test catches the two disagreeing. None for a
/// window there is no kernel for.
pub fn digits(window: u32) -> Option<Vec<u32>> {
    match window {
        2..=20 => Some(vec![window; 256u32.div_ceil(window) as usize]),
        22 => Some([vec![21; 8], vec![22; 4]].concat()),
        24 => Some([vec![23; 8], vec![24; 3]].concat()),
        _ => None,
    }
}

/// One row of a curve's table: the lowest bit of its digit, the places in it
/// (2^width - 1 points of 64 bytes, b * 2^pos * P for b = 1, 2, ...) and how
/// many of them hold a point - fewer only in the last row of a uniform
/// window, whose digit is shorter; zeros fill the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
    pub pos: u32,
    pub places: u64,
    pub points: u64,
}

/// The rows of each curve's table at `window` (see `digits`).
pub fn rows(window: u32) -> Option<Vec<Row>> {
    let mut pos = 0;
    let rows = digits(window)?
        .into_iter()
        .map(|width| {
            let bits = width.min(256 - pos);
            let row = Row { pos, places: (1u64 << width) - 1, points: (1u64 << bits) - 1 };
            pos += bits;
            row
        })
        .collect();
    Some(rows)
}

/// The power tables at the start of the buffer, in u64 (pow64: 8 rows of 256,
/// pow128: 16 of 256 x 2, pow256: 32 of 256 x 4) - `BC3_TABLE_CURVE` of the
/// kernel. Four curves follow them (`rows`), x[4] and y[4] a point.
pub const POWER_TABLES_U64: u64 = 8 * 256 + 16 * 256 * 2 + 32 * 256 * 4;

/// Size of the GPU tables at `window`, 0 for a window there is no kernel for.
pub fn table_bytes(window: u32) -> u64 {
    rows(window).map_or(0, |rows| (POWER_TABLES_U64 + 4 * rows.iter().map(|r| r.places).sum::<u64>() * 8) * 8)
}

/// The largest window above START_WINDOW whose tables fit on the card with
/// room to spare, `free_vram` counting the start window's tables, which go
/// before the bigger ones come. None when nothing bigger fits.
pub fn largest_fitting(free_vram: u64) -> Option<u32> {
    BIGGER.iter().copied().find(|&w| table_bytes(w) + VRAM_SPARE <= free_vram)
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
    /// Get this window ready to switch to - its kernel compiled ahead where
    /// the card has no machine code for it - and say when it is (`ready`).
    /// Its tables the card builds at the switch.
    Build(u32),
    /// Switch the card to this window now.
    Switch(u32),
    /// The tuning is over: the card stays at `choice.window`. `remember`:
    /// whether the choice is sound enough for the next start to go straight
    /// to it - not when the start window measured differently twice.
    Done { choice: Choice, remember: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    /// Mining at START_WINDOW while the target is made ready, and then for
    /// the warm-up; only the time after it is ready counts.
    WarmUp { mined: Duration },
    /// Measuring START_WINDOW.
    MeasureStart { rate: Rate },
    /// Switched to the target, letting it settle.
    Settle { before: f64, mined: Duration },
    /// Measuring the target.
    MeasureTarget { before: f64, rate: Rate },
    /// Back at START_WINDOW, letting it settle.
    SettleBack { before: f64, tried: f64, mined: Duration },
    /// Measuring START_WINDOW again.
    MeasureBack { before: f64, tried: f64, rate: Rate },
    /// Going to the target for good, it having won.
    Return { choice: Choice, remember: bool },
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
    /// The smaller windows that fit, largest first: tried in turn when the
    /// card cannot switch to the target - its tables did not fit after all
    /// (28 GiB on a 32 GB card that drives a desktop), or its kernel failed
    /// the self-test.
    fallback: Vec<u32>,
    ready: bool,
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
                let phase = Phase::Recalled { window: c.window };
                let t = Tuner { target: c.window, fallback: Vec::new(), ready: false, phase };
                (t, Action::Build(c.window))
            });
        }
        let fallback = BIGGER.iter().copied().filter(|&w| w < fitting).collect();
        let t = Tuner { target: fitting, fallback, ready: false, phase: Phase::WarmUp { mined: Duration::ZERO } };
        // The card mines at the start window while the target is made ready.
        Some((t, Action::Build(fitting)))
    }

    /// The window being tried, or gone to.
    pub fn target(&self) -> u32 {
        self.target
    }

    /// A launch came back: `hashes` hashed at `window` in `busy` of GPU time.
    /// A launch at a window the phase does not expect - one still in flight
    /// across a switch - counts for nothing.
    pub fn launched(&mut self, window: u32, hashes: u64, busy: Duration) -> Action {
        let target = self.target;
        match &mut self.phase {
            Phase::WarmUp { mined } if window == START_WINDOW && self.ready => {
                *mined += busy;
                if *mined >= WARM_UP {
                    self.phase = Phase::MeasureStart { rate: Rate::default() };
                }
                Action::Wait
            }
            Phase::MeasureStart { rate } if window == START_WINDOW => {
                rate.add(hashes, busy);
                if rate.busy < MEASURE {
                    return Action::Wait;
                }
                self.phase = Phase::Settle { before: rate.per_second(), mined: Duration::ZERO };
                Action::Switch(target)
            }
            Phase::Settle { before, mined } if window == target => {
                *mined += busy;
                if *mined >= SETTLE {
                    self.phase = Phase::MeasureTarget { before: *before, rate: Rate::default() };
                }
                Action::Wait
            }
            Phase::MeasureTarget { before, rate } if window == target => {
                rate.add(hashes, busy);
                if rate.busy < MEASURE {
                    return Action::Wait;
                }
                self.phase = Phase::SettleBack { before: *before, tried: rate.per_second(), mined: Duration::ZERO };
                Action::Switch(START_WINDOW)
            }
            Phase::SettleBack { before, tried, mined } if window == START_WINDOW => {
                *mined += busy;
                if *mined >= SETTLE {
                    self.phase = Phase::MeasureBack { before: *before, tried: *tried, rate: Rate::default() };
                }
                Action::Wait
            }
            Phase::MeasureBack { before, tried, rate } if window == START_WINDOW => {
                rate.add(hashes, busy);
                if rate.busy < MEASURE {
                    return Action::Wait;
                }
                let (before, after, tried) = (*before, rate.per_second(), *tried);
                let start_rate = (before + after) / 2.0;
                let remember = (before - after).abs() <= start_rate * MAX_DRIFT;
                let keep = tried > start_rate * (1.0 + MIN_GAIN);
                let choice = Choice {
                    window: if keep { target } else { START_WINDOW },
                    start_rate,
                    tried: target,
                    tried_rate: tried,
                };
                if keep {
                    self.phase = Phase::Return { choice, remember };
                    Action::Switch(target)
                } else {
                    self.phase = Phase::Finished;
                    Action::Done { choice, remember }
                }
            }
            _ => Action::Wait,
        }
    }

    /// The target is ready to switch to (`Action::Build`).
    pub fn ready(&mut self) -> Action {
        self.ready = true;
        match self.phase {
            Phase::Recalled { window } => Action::Switch(window),
            _ => Action::Wait,
        }
    }

    /// The card switched to `window` (`ok`), or the switch failed - its
    /// self-test, its memory. A first switch to the target that fails tries
    /// the next smaller window that fits, measured against the same 16; any
    /// other failure ends the tuning at the start window.
    pub fn switched(&mut self, window: u32, ok: bool) -> Action {
        let unmeasured = |window, tried| Choice { window, start_rate: 0.0, tried, tried_rate: 0.0 };
        let done = match self.phase {
            Phase::Return { choice, remember } if ok && window == self.target => Action::Done { choice, remember },
            // Already in the file: nothing to write again.
            Phase::Recalled { .. } if ok => Action::Done { choice: unmeasured(window, window), remember: false },
            _ if ok => return Action::Wait,
            Phase::Settle { before, .. } if window == self.target && !self.fallback.is_empty() => {
                self.target = self.fallback.remove(0);
                self.phase = Phase::Settle { before, mined: Duration::ZERO };
                return Action::Switch(self.target);
            }
            // A target that fails is remembered as such: the card stays at the
            // start window until a new driver or miner version.
            _ if window != START_WINDOW => Action::Done { choice: unmeasured(START_WINDOW, window), remember: true },
            // The start window itself, on the way back: no answer at all.
            _ => Action::Done { choice: unmeasured(START_WINDOW, self.target), remember: false },
        };
        self.phase = Phase::Finished;
        done
    }

    /// Whether the tuning is over.
    #[cfg(test)]
    pub fn finished(&self) -> bool {
        self.phase == Phase::Finished
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
        // 8 x (2^21 - 1) + 4 x (2^22 - 1) and 8 x (2^23 - 1) + 3 x (2^24 - 1)
        // points a curve.
        assert!((mib(22) - 8192.3).abs() < 0.1);
        assert!((mib(24) - 28672.3).abs() < 0.1);
        assert_eq!(table_bytes(21), 0, "no kernel for 21");
    }

    #[test]
    fn every_window_cuts_the_256_bits_into_its_rows() {
        for w in [2, 16, 18, 19, 20, 22, 24] {
            let rows = rows(w).unwrap();
            let bits: u32 = rows.iter().map(|r| r.points.trailing_ones()).sum();
            assert_eq!(bits, 256, "window {w}");
            assert_eq!(rows[0].pos, 0);
            for pair in rows.windows(2) {
                assert_eq!(pair[1].pos, pair[0].pos + pair[0].points.trailing_ones(), "window {w}");
            }
            // Every place holds a point, but in the last row of a uniform
            // window that does not divide 256.
            for r in &rows[..rows.len() - 1] {
                assert_eq!(r.points, r.places, "window {w}");
            }
        }
        assert_eq!(digits(20).unwrap().len(), 13);
        assert_eq!(rows(20).unwrap()[12], Row { pos: 240, places: (1 << 20) - 1, points: (1 << 16) - 1 });
        assert_eq!(digits(22).unwrap(), [21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22]);
        assert_eq!(digits(24).unwrap(), [23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24]);
        assert_eq!(rows(24).unwrap()[8].pos, 184);
        // The kernels there are (build.rs) and those the tuning tries.
        assert!(BIGGER.iter().all(|&w| digits(w).is_some()));
        assert!(BIGGER.windows(2).all(|p| table_bytes(p[0]) > table_bytes(p[1])), "largest first");
    }

    #[test]
    fn the_largest_window_that_fits() {
        // A 32 GB RTX 5090 with nothing else on it: 24 (28 GiB + 1 spare).
        assert_eq!(largest_fitting(30 * GIB), Some(24));
        // A 24 GiB card with 22 free: 22 (8 + 1); 8 free: 20 (3.25 + 1).
        assert_eq!(largest_fitting(22 * GIB), Some(22));
        assert_eq!(largest_fitting(8 * GIB), Some(20));
        // 4 GiB free: 20 needs 3.25 + 1 spare, 19 needs 1.75 + 1.
        assert_eq!(largest_fitting(4 * GIB), Some(19));
        // 2 GiB free on the card: 18 (0.94 + 1); 1.5 GiB: nothing above 16.
        assert_eq!(largest_fitting(2 * GIB), Some(18));
        assert_eq!(largest_fitting(3 * GIB / 2), None);
    }

    /// Drive a tuner through launches of `busy` each at the window the card
    /// is at, applying its actions; `ready_after` launches until the target
    /// is ready, `rate_at(window, launch)` the card's rate. Returns the
    /// actions seen.
    fn run(tuner: &mut Tuner, first: Action, rate_at: impl Fn(u32, usize) -> f64, ready_after: usize) -> Vec<Action> {
        let busy = Duration::from_millis(250);
        let mut window = START_WINDOW;
        let mut actions = vec![first];
        let apply = |a: Action, window: &mut u32, tuner: &mut Tuner, actions: &mut Vec<Action>| {
            if a != Action::Wait {
                actions.push(a);
            }
            if let Action::Switch(w) = a {
                *window = w;
                let after = tuner.switched(w, true);
                if after != Action::Wait {
                    actions.push(after);
                }
            }
        };
        for n in 0..5000 {
            if n == ready_after {
                let a = tuner.ready();
                apply(a, &mut window, tuner, &mut actions);
            }
            if tuner.finished() {
                break;
            }
            let hashes = (rate_at(window, n) * busy.as_secs_f64()) as u64;
            let a = tuner.launched(window, hashes, busy);
            apply(a, &mut window, tuner, &mut actions);
            if tuner.finished() {
                break;
            }
        }
        actions
    }

    fn done(actions: &[Action]) -> (Choice, bool) {
        match actions.last() {
            Some(&Action::Done { choice, remember }) => (choice, remember),
            _ => panic!("not done: {actions:?}"),
        }
    }

    #[test]
    fn a_faster_bigger_window_is_kept() {
        let (mut t, first) = Tuner::start(Some(20), None).unwrap();
        assert_eq!(first, Action::Build(20));
        let actions = run(&mut t, first, |w, _| if w == 20 { 2.66e6 } else { 2.52e6 }, 40);
        // 16, 20, 16 again, and back to 20 for good.
        assert_eq!(actions[1..4], [Action::Switch(20), Action::Switch(START_WINDOW), Action::Switch(20)]);
        let (c, remember) = done(&actions);
        assert_eq!(c.window, 20);
        assert!((c.start_rate - 2.52e6).abs() < 1e3 && (c.tried_rate - 2.66e6).abs() < 1e3);
        assert!(remember);
        assert!(t.finished());
    }

    #[test]
    fn a_window_that_is_not_faster_goes_back() {
        let (mut t, first) = Tuner::start(Some(19), None).unwrap();
        // 0.2 % faster is within the spread: not worth 1.75 GiB.
        let actions = run(&mut t, first, |w, _| if w == 19 { 2.505e6 } else { 2.50e6 }, 10);
        assert_eq!(actions[1..3], [Action::Switch(19), Action::Switch(START_WINDOW)]);
        let (c, remember) = done(&actions);
        assert_eq!((c.window, c.tried), (START_WINDOW, 19));
        assert!(remember, "a steady loss is as good an answer as a win");
        assert_eq!(actions.len(), 4, "no switch after the decision: {actions:?}");
    }

    #[test]
    fn a_drifting_clock_cancels_out() {
        // Both windows equally fast, the clock rising 0.5 % a minute: 16
        // measured once, before, would make the bigger window look 0.7 %
        // faster - over MIN_GAIN. Measured before and after, it is not.
        let rising = |_: u32, n: usize| 2.5e6 * (1.0 + 2e-5 * n as f64);
        let (mut t, first) = Tuner::start(Some(20), None).unwrap();
        let (c, remember) = done(&run(&mut t, first, rising, 10));
        assert_eq!(c.window, START_WINDOW, "{c:?}");
        assert!((c.tried_rate / c.start_rate - 1.0).abs() < 1e-3, "{c:?}");
        assert!(remember, "1.4 % between the two is drift, not a disturbance");
    }

    #[test]
    fn a_disturbed_measurement_is_used_but_not_remembered() {
        // 16 at 1.82 the first time and 2.15 the second, as the laptop's
        // power sharing had it: the choice stands for now, the next start
        // measures again.
        let (mut t, first) = Tuner::start(Some(19), None).unwrap();
        let rate = |w: u32, n: usize| match (w, n < 700) {
            (START_WINDOW, true) => 1.82e6,
            (START_WINDOW, false) => 2.15e6,
            _ => 2.2e6,
        };
        let (c, remember) = done(&run(&mut t, first, rate, 10));
        assert_eq!(c.window, 19);
        assert!((c.start_rate - 1.985e6).abs() < 1e3, "{c:?}");
        assert!(!remember);
    }

    #[test]
    fn a_slow_compile_ahead_is_waited_for() {
        // The target is ready long after the warm-up - the driver compiling
        // its kernel ahead; the start window's rate meanwhile (here 2.0, the
        // CPU busy) must not count.
        let (mut t, first) = Tuner::start(Some(20), None).unwrap();
        let ready_after = 1000;
        let rate = |w: u32, n: usize| match (w, n < ready_after) {
            (START_WINDOW, true) => 2.0e6,
            (START_WINDOW, false) => 2.5e6,
            _ => 2.6e6,
        };
        let actions = run(&mut t, first, rate, ready_after);
        let (c, _) = done(&actions);
        assert_eq!(c.window, 20);
        assert!((c.start_rate - 2.5e6).abs() < 1e3, "start rate {} includes the compile", c.start_rate);
    }

    #[test]
    fn the_start_window_is_measured_after_the_target_is_ready_and_the_warm_up() {
        // Launches of 0.25 s: the warm-up is 360 of them after the target is
        // ready, the measurement 240 more.
        let (mut t, _) = Tuner::start(Some(20), None).unwrap();
        let busy = Duration::from_millis(250);
        let ready_after = 1000;
        for n in 0..ready_after + 360 + 240 {
            if n == ready_after {
                assert_eq!(t.ready(), Action::Wait);
            }
            let a = t.launched(START_WINDOW, 625_000, busy);
            let expect = if n + 1 == ready_after + 360 + 240 { Action::Switch(20) } else { Action::Wait };
            assert_eq!(a, expect, "launch {n}");
        }
    }

    #[test]
    fn a_launch_from_before_a_switch_counts_for_nothing() {
        let (mut t, first) = Tuner::start(Some(20), None).unwrap();
        // Every launch at the window the phase does not expect - as one in
        // flight across a switch would be - at an absurd rate.
        let actions = run(&mut t, first, |w, _| if w == 20 { 2.6e6 } else { 2.5e6 }, 10);
        let (c, _) = done(&actions);
        let (mut u, _) = Tuner::start(Some(20), None).unwrap();
        assert_eq!(u.ready(), Action::Wait);
        let busy = Duration::from_millis(250);
        let mut window = START_WINDOW;
        let mut result = None;
        for _ in 0..5000 {
            let stray = if window == START_WINDOW { 20 } else { START_WINDOW };
            assert_eq!(u.launched(stray, u64::MAX / 8, busy), Action::Wait);
            let rate = if window == 20 { 2.6e6 } else { 2.5e6 };
            match u.launched(window, (rate * 0.25) as u64, busy) {
                Action::Switch(w) => {
                    window = w;
                    if let Action::Done { choice, .. } = u.switched(w, true) {
                        result = Some(choice);
                        break;
                    }
                }
                Action::Done { choice, .. } => {
                    result = Some(choice);
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(result, Some(c));
    }

    #[test]
    fn a_remembered_window_is_gone_to_unmeasured() {
        let c = Choice { window: 19, start_rate: 1.0, tried: 19, tried_rate: 2.0 };
        let (mut t, first) = Tuner::start(Some(20), Some(c)).unwrap();
        assert_eq!(first, Action::Build(19));
        assert_eq!(t.launched(START_WINDOW, 1000, Duration::from_secs(100)), Action::Wait);
        assert_eq!(t.ready(), Action::Switch(19));
        // Already in the file: nothing to write again.
        assert!(matches!(t.switched(19, true), Action::Done { choice: Choice { window: 19, .. }, remember: false }));
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
    fn a_target_the_card_cannot_switch_to_gives_way_to_the_next() {
        // 24 does not load - 28 GiB that did not fit after all - and 22 is
        // measured in its place, against the 16 measured before.
        let (mut t, first) = Tuner::start(Some(24), None).unwrap();
        assert_eq!(first, Action::Build(24));
        let busy = Duration::from_millis(250);
        assert_eq!(t.ready(), Action::Wait);
        let mut n = 0;
        let a = loop {
            n += 1;
            let a = t.launched(START_WINDOW, 625_000, busy);
            if a != Action::Wait {
                break a;
            }
        };
        assert_eq!((a, n), (Action::Switch(24), 360 + 240));
        assert_eq!(t.switched(24, false), Action::Switch(22));
        assert_eq!(t.switched(22, true), Action::Wait);
        assert_eq!(t.target(), 22);
        let mut window = 22;
        let choice = loop {
            let rate = if window == 22 { 2.7e6 } else { 2.5e6 };
            match t.launched(window, (rate * 0.25) as u64, busy) {
                Action::Switch(w) => {
                    window = w;
                    if let Action::Done { choice, .. } = t.switched(w, true) {
                        break choice;
                    }
                }
                Action::Done { choice, .. } => break choice,
                Action::Wait => {}
                a => panic!("{a:?}"),
            }
        };
        assert_eq!((choice.window, choice.tried), (22, 22));
        assert!((choice.start_rate - 2.5e6).abs() < 1e3, "{choice:?}");
        // Every smaller window failing too: 16, the last one tried remembered.
        let (mut t, _) = Tuner::start(Some(19), None).unwrap();
        t.ready();
        while t.launched(START_WINDOW, 625_000, busy) == Action::Wait {}
        assert_eq!(t.switched(19, false), Action::Switch(18));
        assert!(matches!(
            t.switched(18, false),
            Action::Done { choice: Choice { window: START_WINDOW, tried: 18, .. }, remember: true }
        ));
    }

    #[test]
    fn a_failed_switch_otherwise_stays_at_the_start_window() {
        // A switch that failed - the self-test, the card's memory - is
        // remembered: the card stays at 16 until a new driver or version.
        let (mut t, _) = Tuner::start(Some(18), None).unwrap();
        t.ready();
        while t.launched(START_WINDOW, 625_000, Duration::from_millis(250)) == Action::Wait {}
        assert!(matches!(
            t.switched(18, false),
            Action::Done { choice: Choice { window: START_WINDOW, tried: 18, .. }, remember: true }
        ));
        assert!(t.finished());
        // So is a remembered window that fails: it was measured on this card
        // before, and something has changed since.
        let c = Choice { window: 19, start_rate: 1.0, tried: 19, tried_rate: 2.0 };
        let (mut t, _) = Tuner::start(Some(20), Some(c)).unwrap();
        assert_eq!(t.ready(), Action::Switch(19));
        assert!(matches!(
            t.switched(19, false),
            Action::Done { choice: Choice { window: START_WINDOW, tried: 19, .. }, remember: true }
        ));
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
