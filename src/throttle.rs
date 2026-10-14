//! Optional GPU throttle: keep the drafting GPU load off the ceiling so the
//! machine stays usable while a draft runs (watching a video in a browser,
//! say).
//!
//! # Why a duty cycle, and only a duty cycle
//!
//! The encoder's GPU call is `encode_window` (`src/encode.rs`): one forward per
//! window on a FIXED frame grid. Two facts close every other door:
//!
//! * `bundle::session()` registers the execution providers with NO
//!   `provider_options`, so there is no CUDA stream priority or SM-occupancy
//!   knob to turn from here; and
//! * V-JEPA 2.1's rotary embedding pins the token count, so the window (and
//!   with it the batch) is fixed -- "a short window is not a thing it can be
//!   asked for".
//!
//! What is left is the TIME between windows. The encoder runs a window and then
//! waits; the wait is ours to set. That wait is expressed as a DUTY CYCLE: at
//! `duty = 50` a window is followed by a nap as long as itself, and the encode
//! of that window spreads over twice the wall clock with the GPU idle half the
//! time -- which is exactly the room a browser needs to composite a frame.
//!
//! # The loop
//!
//! A background thread samples the whole-device 3D utilization through NVML
//! (one call, ~1.7 us -- `nvidia-smi` as a subprocess is ~2 s, five orders of
//! magnitude more) every `interval` seconds and accumulates the samples. Once
//! `avg_window` seconds have gone by, it forms ONE error and nudges the duty
//! toward the target:
//!
//! ```text
//!   error    = mean(last avg_window of samples) - gpu_target
//!   delta    = clamp(-kp * error, -fall_limit .. rise_limit)
//!   duty     = clamp(duty + delta, duty_min .. 100)
//! ```
//!
//! Over the target, error is positive, duty falls, windows get longer naps and
//! the GPU load comes off; under it, duty climbs back. The loop is SINGLE
//! LAYER: the device ceiling `gpu_target` is the one controlled quantity.
//! `script_max` is an OPTIONAL second clamp (default off) for the rare case
//! where the machine ceiling alone lets the script take more than wanted; it
//! clamps duty from above without touching the loop.
//!
//! # Why the average is in TIME and not in samples
//!
//! This is the one non-obvious thing here, and both earlier versions got it
//! wrong. Because a window is `work` ms of saturation followed by a nap, the
//! load this loop creates is a PULSE TRAIN: measured, `work = 606 ms` against a
//! 2430 ms nap at duty 20, a ~3 s period. NVML's own figure is a ~1 s sliding
//! window, so one sample reads ~100 % inside a pulse and ~40 % in the nap --
//! the same run reports both. A loop that decides on a sample therefore decides
//! on the phase of its own output: 100 is "over target", so it lengthens the
//! nap, which makes the next pulse no smaller, until the duty is pinned at the
//! floor and the mean sits at 53 % against a 90 % target.
//!
//! Averaging over `avg_window` (9 s, three periods) removes the phase and the
//! reading becomes the duty. Sampling stays fast because the average wants
//! points; the DECISION is what is slow.
//!
//! # Not oscillating
//!
//! The load being measured is the load being changed, so a naive loop can hunt.
//! The guards, all in `recompute`:
//!
//! 1. **deadband** -- an error inside +-`deadzone` is not acted on at all, so
//!    the browser's own jitter never crosses the line;
//! 2. **slow both ways** -- a correction that RAISES duty is capped at
//!    `rise_limit` per decision and one that LOWERS it at `fall_limit`: with a
//!    slow decision rate (one per `avg_window`) an uncapped proportional move
//!    overshoots in either direction, and a down-overshoot puts the loop in the
//!    floor, from which it then has to climb back at `rise_limit` each time;
//! 3. **the average window** -- see above; this is what keeps the loop from
//!    reading its own phase;
//! 4. **dwell** -- after any change, `dwell` seconds must pass before a change
//!    in the OPPOSITE direction is allowed.
//!
//! Nothing here acts on a SINGLE sample. That is deliberate, and the earlier
//! `spike_cut` guard (a single over-target sample cutting duty immediately) was
//! removed for exactly the reason the average exists: at this sampling rate the
//! encoder's own pulse and a real external burst are the same shape, so a
//! per-sample branch fires on the phase of the loop's own output and floor-locks
//! it. A real external burst is answered one `avg_window` later instead.
//!
//! # Fail-open
//!
//! If NVML cannot be loaded or read, the throttle DISARMS (duty 100, no naps)
//! and says so once; a draft must never be blocked by an optional feature. The
//! throttle is off unless asked for: `--throttle` / `--no-throttle` (default).
//! An explicit `--throttle-script-max` turns on the optional second clamp.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;

/// Preset names, as accepted by `--throttle`.
pub const PRESETS: &[&str] = &["gentle", "balanced", "aggressive", "custom"];

/// The presets the PICKER's P key cycles through, in order, starting from OFF.
///
/// `custom` is deliberately absent: it means "start from balanced and take the
/// CLI's overrides", and a key press has no overrides to take. The picker
/// therefore offers the three that are complete on their own, and its OFF is
/// `None` rather than a name (see `settings::Settings::throttle`).
pub const PICKER_CYCLE: &[&str] = &["gentle", "balanced", "aggressive"];

/// The next picker preset after `cur` (`None` = off), wrapping back to `None`.
///
/// A name that is not in `PICKER_CYCLE` -- a hand-edited `custom`, or a preset
/// a future build drops -- restarts the cycle, so one press always lands on a
/// preset the build actually has rather than stepping into nothing.
pub fn next_preset(cur: Option<&str>) -> Option<&'static str> {
    match cur {
        None => Some(PICKER_CYCLE[0]),
        Some(c) => match PICKER_CYCLE.iter().position(|p| *p == c) {
            Some(i) if i + 1 < PICKER_CYCLE.len() => Some(PICKER_CYCLE[i + 1]),
            // the last preset, or an unknown one: back to off
            _ => None,
        },
    }
}

/// Two or three words for what `name` does, for the POST line. `None` (or an
/// unknown name) is the off state.
pub fn describe(name: Option<&str>) -> String {
    match name {
        None => "off".into(),
        Some("custom") => "custom".into(),
        Some(n) => match Config::preset(n) {
            Some(c) => format!("{n}, hold <= {:.0}%", c.gpu_target),
            None => format!("{n} (unknown, ignored)"),
        },
    }
}

/// Everything the loop reads. All of it is fixed at start; the loop only
/// carries `duty` and the smoothing `history` forward.
///
/// `Copy` because every field is a scalar and `preset` uses `..d` in several
/// match arms at once.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Whole-device 3D utilization to hold. Above this the throttle pulls the
    /// duty down.
    pub gpu_target: f32,
    /// Seconds between samples.
    pub interval: f32,
    /// Seconds of samples averaged before ONE control decision is made.
    ///
    /// This is the field that makes the loop work at all, and it is not a
    /// smoothing constant in the usual sense. A window takes `work` ms of
    /// solid GPU and is followed by a nap (see `rest_after`), so the load the
    /// throttle creates is a TRAIN OF PULSES, not a level: measured on the
    /// shipped grid, `work = 606 ms` against a 2430 ms nap at duty 20, i.e. a
    /// ~3 s period of which ~0.6 s is saturated. `nvmlDeviceGetUtilization-
    /// Rates` reports a ~1 s sliding window, so a sample either lands inside a
    /// pulse and reads ~100 %, or lands in the nap and reads ~40 % -- the
    /// SAME run reports both. A loop that acts on one sample is therefore
    /// acting on the phase of its own output, and it floor-locks itself: every
    /// pulse reads 100, every 100 is "over target", every correction lengthens
    /// the nap and makes the next pulse no smaller. That is exactly what the
    /// 0.5 s build did, and why its mean sat at 53 % against a 90 % target.
    ///
    /// Averaging over a window longer than the period removes the phase: at
    /// `avg_window` 9 s a pulse train of 0.6/3.0 reads ~20 %, which IS the
    /// duty. The controller then regulates the only thing it can actually
    /// control, and the pulses stop being visible to it at all.
    pub avg_window: f32,
    /// Loop gain: duty moves `kp * error` per decision.
    pub kp: f32,
    /// Error inside +-this is ignored (in utilization points).
    pub deadzone: f32,
    /// Duty floor: the throttle never starves the encode below this.
    pub duty_min: f32,
    /// Cap on a single UPWARD duty move, per decision (in duty points).
    pub rise_limit: f32,
    /// Cap on a single DOWNWARD duty move, per decision (in duty points).
    ///
    /// The mirror of `rise_limit`, and not optional. Without it a large error
    /// at a low target floors the loop in one decision -- at `gpu_target` 50,
    /// `kp` 3, an error of 37 asked for a move of 111 points, and the duty went
    /// 64 -> 20 in a single step, then had to climb back at `rise_limit` 4 per
    /// decision. Big downward moves overshoot exactly like big upward ones.
    pub fall_limit: f32,
    /// Seconds a change must hold before the opposite direction is allowed.
    pub dwell: f32,
    /// RETIRED. Kept only so an existing `--throttle-spike-margin` still
    /// parses. The spike branch was removed: at this sampling rate the
    /// encoder's own pulse is indistinguishable from a real burst, so acting on
    /// one sample acted on the loop's own phase. No effect.
    #[allow(dead_code)]
    pub spike_margin: f32,
    /// RETIRED with `spike_margin`. No effect.
    #[allow(dead_code)]
    pub spike_cut_max: f32,
    /// RETIRED with `spike_margin`. No effect.
    #[allow(dead_code)]
    pub spike_hold: f32,
    /// Optional second clamp: the script never exceeds this duty (100 = off).
    pub script_max: f32,
    /// Samples averaged for the error. SUPERSEDED by `avg_window`: the loop
    /// now averages in TIME, not in count, because the plant's pulse period is
    /// a time. Kept so an existing `--throttle-smooth` still parses; it has no
    /// effect.
    #[allow(dead_code)]
    pub smooth: usize,
    /// Seconds at the start during which no correction is applied, so the
    /// first windows (which are also the warmup) are not throttled.
    pub grace: f32,
    /// The duty the loop arms at, instead of 100.
    ///
    /// Starting at 100 means the first `avg_window` seconds are spent at full
    /// pelt no matter what, because the loop has no evidence yet. The encoder
    /// is the only real load and it is entirely predictable, so starting where
    /// the answer usually is costs nothing and skips the one window that the
    /// user is most likely to be watching. The loop walks up from here if the
    /// machine turns out to be idle.
    pub duty_start: f32,
}

impl Default for Config {
    fn default() -> Self {
        // `balanced`: the shipped default.
        //
        // TUNED AGAINST MEASUREMENT on an RTX 3080 Ti, 2m43s 1080p clip,
        // browser alongside. Three rounds, and the third is the one that
        // understands the plant:
        //
        //   no throttle    mean 78%,  median 98%,  60% of samples over 90%
        //   round one      mean 71%,  median 75%,  43% over 90%
        //                  (interval 5 s, kp 2, grace 30 -- too slow)
        //   round two      mean 56%,  median 48%,  18% over 90%
        //                  (interval 0.5 s, kp 6, uncapped spike cut)
        //   round three    this config
        //
        // Rounds one and two both LOOKED like a tuning problem and were not.
        // The per-window trace is what settled it: `[window] work=606ms
        // rest=2430ms duty=20%`. One encode window is 606 ms of solid GPU,
        // followed by a nap -- so at any duty the load is a PULSE TRAIN, ~0.6 s
        // saturated out of a ~3 s period at duty 20. NVML reports a ~1 s window,
        // so a sample lands in a pulse and reads 100, or in the nap and reads
        // 40: the run reports both, and rounds one and two were reading the
        // PHASE OF THEIR OWN OUTPUT. Every 100 was then "over target", so every
        // correction lengthened the nap, which made the next pulse no smaller
        // -- floor lock, and the mean sank to 53 % against a 90 % target with
        // the duty pinned at 20.
        //
        // The fix is `avg_window`: decisions are made on ~9 s of samples, which
        // is three pulse periods, so the phase averages out and the reading IS
        // the duty. Sampling stays fast (0.5 s) because the average wants many
        // points; it is the DECISION that is slow. `smooth` is 1 because
        // `avg_window` already is the average.
        //
        // The rest of the guards are kept for the one disturbance the average
        // cannot see: a real burst from something else on the machine. Those
        // are rare and short, so the spike branch fires on a single sample with
        // a CAPPED cut (`spike_cut_max`) -- uncapped, one burst would floor-lock
        // the loop exactly as the pulse train did.
        Self {
            gpu_target: 90.0,
            interval: 0.5,
            avg_window: 9.0,
            kp: 3.0,
            deadzone: 3.0,
            duty_min: 20.0,
            rise_limit: 4.0,
            fall_limit: 20.0,
            dwell: 4.0,
            spike_margin: 3.0,
            spike_cut_max: 25.0,
            spike_hold: 2.5,
            script_max: 100.0,
            smooth: 1,
            grace: 12.0,
            duty_start: 60.0,
        }
    }
}

impl Config {
    /// The named preset's config, or `None` for an unknown name.
    pub fn preset(name: &str) -> Option<Self> {
        let d = Self::default();
        Some(match name {
            // gentle: below the target and slower to act -- for a machine that
            // is otherwise busy and should not be hurried.
            "gentle" => Self {
                gpu_target: 85.0,
                interval: 0.5,
                avg_window: 12.0,
                kp: 2.0,
                deadzone: 4.0,
                duty_min: 15.0,
                rise_limit: 3.0,
                fall_limit: 12.0,
                dwell: 6.0,
                spike_margin: 5.0,
                spike_cut_max: 18.0,
                spike_hold: 3.0,
                ..d
            },
            "balanced" => d,
            // aggressive: right up against the ceiling, and quicker to give
            // room back. The shorter average is deliberate -- it tracks a
            // changing load faster, at the cost of some of the phase rejection
            // `avg_window` exists for. `spike_hold` is shorter because a high
            // target leaves less room between "at target" and "over the line".
            "aggressive" => Self {
                gpu_target: 95.0,
                interval: 0.5,
                avg_window: 6.0,
                kp: 4.0,
                deadzone: 2.0,
                duty_min: 35.0,
                rise_limit: 6.0,
                fall_limit: 25.0,
                dwell: 3.0,
                spike_margin: 2.0,
                spike_cut_max: 35.0,
                spike_hold: 2.0,
                ..d
            },
            // `custom` starts from the default and is then overridden
            // field-by-field from the CLI. It is accepted here so that
            // `--throttle custom` needs no special case upstream.
            "custom" => d,
            _ => return None,
        })
    }

    /// Replace any field the CLI set explicitly (`Some`). Used for
    /// `--throttle custom`.
    pub fn apply_overrides(&mut self, o: &Overrides) {
        if let Some(v) = o.gpu_target {
            self.gpu_target = v;
        }
        if let Some(v) = o.interval {
            self.interval = v;
        }
        if let Some(v) = o.avg_window {
            self.avg_window = v;
        }
        if let Some(v) = o.kp {
            self.kp = v;
        }
        if let Some(v) = o.deadzone {
            self.deadzone = v;
        }
        if let Some(v) = o.duty_min {
            self.duty_min = v;
        }
        if let Some(v) = o.duty_start {
            self.duty_start = v;
        }
        if let Some(v) = o.rise_limit {
            self.rise_limit = v;
        }
        if let Some(v) = o.fall_limit {
            self.fall_limit = v;
        }
        if let Some(v) = o.dwell {
            self.dwell = v;
        }
        if let Some(v) = o.spike_margin {
            self.spike_margin = v;
        }
        if let Some(v) = o.spike_cut_max {
            self.spike_cut_max = v;
        }
        if let Some(v) = o.spike_hold {
            self.spike_hold = v;
        }
        if let Some(v) = o.script_max {
            self.script_max = v;
        }
        if let Some(v) = o.smooth {
            self.smooth = v;
        }
        if let Some(v) = o.grace {
            self.grace = v;
        }
    }
}

/// The raw CLI values for `--throttle custom`. Every field is optional; the
/// ones left unset keep the preset/default value.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub gpu_target: Option<f32>,
    pub interval: Option<f32>,
    pub avg_window: Option<f32>,
    pub kp: Option<f32>,
    pub deadzone: Option<f32>,
    pub duty_min: Option<f32>,
    pub duty_start: Option<f32>,
    pub rise_limit: Option<f32>,
    pub fall_limit: Option<f32>,
    pub dwell: Option<f32>,
    pub spike_margin: Option<f32>,
    pub spike_cut_max: Option<f32>,
    pub spike_hold: Option<f32>,
    pub script_max: Option<f32>,
    pub smooth: Option<usize>,
    pub grace: Option<f32>,
}

/// The loop's mutable state. Shared behind a mutex; `duty` is also mirrored in
/// an atomic so the encode hot path reads it without locking.
struct State {
    cfg: Config,
    duty: f32,
    /// The samples inside the current accumulation window. Emptied once a
    /// window closes; see `recompute` for why the window exists at all.
    acc: Vec<f32>,
    /// When the current accumulation window opened.
    acc_start: Instant,
    /// The last window's mean, kept only so the trace can show what the
    /// decision was made on.
    last_mean: f32,
    /// Set by the picker's P key cycling round to OFF. The sampler keeps its
    /// tick (the thread cannot be joined) but every decision is skipped, so the
    /// duty stays at 100 and this process drafts exactly as an unthrottled
    /// build would.
    off: bool,
    last_change: Instant,
    last_dir: i8, // -1 down, 0 none, +1 up: which way the last move went
    started: Instant,
    warned: bool,
    /// Dump every decision to stderr (`GS_THROTTLE_TRACE=1`).
    trace: bool,
}

/// The process-wide throttle, if one was installed.
static THROTTLE: OnceLock<Mutex<State>> = OnceLock::new();
/// `duty` mirrored for the encode path, scaled by 100 (as an integer) so it
/// fits an atomic. 100 = full speed, no nap. When no throttle is installed
/// this stays at 10000 and the naps are zero.
static DUTY_X100: AtomicU32 = AtomicU32::new(10000);
/// Set when the loop's thread has been spawned, so `install` is idempotent.
static RUNNING: AtomicBool = AtomicBool::new(false);
/// Set by the sampler thread when it wants the encode loop to stop (never in
/// this design; kept for symmetry with `cancel`).
static _STOP: AtomicBool = AtomicBool::new(false);

/// Install the throttle with `cfg` and start its sampler thread. A no-op if one
/// is already installed or if `enabled` is false.
pub fn install(enabled: bool, cfg: Config) {
    if !enabled {
        return;
    }
    if RUNNING.swap(true, Ordering::SeqCst) {
        return; // already running
    }
    let now = Instant::now();
    let interval = Duration::from_secs_f32(cfg.interval.max(0.2));
    let script_max = cfg.script_max;
    let gpu_target = cfg.gpu_target;
    let interval_s = cfg.interval;
    let avg_window = cfg.avg_window;
    let duty_floor = cfg.duty_min;
    let duty_start = cfg.duty_start.clamp(cfg.duty_min, 100.0);

    let st = State {
        cfg,
        duty: duty_start,
        acc: Vec::with_capacity(64),
        acc_start: now,
        last_mean: 0.0,
        off: false,
        last_change: now,
        last_dir: 0,
        started: now,
        warned: false,
        trace: std::env::var_os("GS_THROTTLE_TRACE").is_some_and(|v| v != "0"),
    };
    let published = st.duty.min(st.cfg.script_max).clamp(1.0, 100.0);
    if THROTTLE.set(Mutex::new(st)).is_err() {
        return; // someone else won the race
    }
    // Arm the encode path at `duty_start`, not at 100: the sampler thread's
    // first `publish` is one tick away, and until then the encoder would run at
    // full pelt.
    DUTY_X100.store((published * 100.0) as u32, Ordering::Relaxed);

    // Say what we are doing, once, on stderr so a piped stdout stays clean.
    eprintln!(
        "[throttle] on  target {gpu_target:.0}%  sample {interval_s:.1}s  \
         average {avg_window:.0}s  start {duty_start:.0}%  duty floor {duty_floor:.0}%{}",
        if script_max < 100.0 {
            format!("  script-max {script_max:.0}%")
        } else {
            String::new()
        }
    );

    std::thread::Builder::new()
        .name("gs-throttle".into())
        .spawn(move || sampler_loop(interval))
        .ok();
}

/// Change the running throttle's settings in place, or turn it on and off.
///
/// `install` is one-shot per process -- it owns the once-only statics -- but the
/// PICKER can change its mind between batches, so this is how a preset reaches a
/// loop that is already sampling. Everything except the sampler's tick rate is
/// taken from `cfg`: the tick is fixed at install because the thread's sleep is
/// baked into its loop, and the presets all sample at the same 0.5 s anyway. A
/// `None` is the off state.
pub fn reconfigure(cfg: Option<Config>) {
    let Some(t) = THROTTLE.get() else {
        // Never installed in this process (a CLI run that found no flag, then
        // something asks for one): install instead.
        if let Some(cfg) = cfg {
            install(true, cfg);
        }
        return;
    };
    if let Ok(mut s) = t.lock() {
        match cfg {
            None => {
                // Off. The sampler keeps ticking -- one NVML read costs nothing
                // -- but `recompute` returns immediately on this flag, so the
                // duty is never walked back off 100.
                s.off = true;
                DUTY_X100.store(10000, Ordering::Relaxed);
            }
            Some(cfg) => {
                let duty_start = cfg.duty_start.clamp(cfg.duty_min, 100.0);
                s.cfg = cfg;
                s.duty = duty_start;
                s.off = false;
                // Drop the accumulated window: its samples were taken under the
                // old target, and averaging across the change would apply the
                // new target to the old load for one window.
                s.acc.clear();
                s.acc_start = Instant::now();
                s.last_dir = 0;
                s.last_change = Instant::now();
                s.started = Instant::now(); // restart grace for the new target
                publish(&s);
            }
        }
    }
}

/// Turn the throttle off for the rest of the session -- duty 100, no naps.
pub fn disable() {
    reconfigure(None);
}

/// The sampler: read the GPU, feed the loop, until the process goes away.
///
/// NVML is loaded lazily and once; if it is not there the thread exits after
/// saying so and the duty stays at 100 (fail-open).
fn sampler_loop(interval: Duration) {
    let nvml = match nvml_util::Nvml::open() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[throttle] NVML unavailable ({e}); throttle disarmed");
            DUTY_X100.store(10000, Ordering::Relaxed);
            return;
        }
    };
    let mut next_bad = 0u32;
    loop {
        std::thread::sleep(interval);
        match nvml.utilization() {
            Ok(u) => {
                next_bad = 0;
                recompute(u as f32);
            }
            Err(e) => {
                // A single failed read is nothing; a run of them means the
                // driver went away, and then holding duty at 100 (no naps) is
                // the safe answer.
                next_bad += 1;
                if next_bad >= 3 {
                    if let Some(t) = THROTTLE.get() {
                        if let Ok(mut s) = t.lock() {
                            if !s.warned {
                                s.warned = true;
                                eprintln!(
                                    "[throttle] NVML read failed repeatedly ({e}); \
                                     releasing throttle"
                                );
                            }
                        }
                    }
                    DUTY_X100.store(10000, Ordering::Relaxed);
                    return;
                }
            }
        }
    }
}

/// One tick of the loop: push the sample, and -- once the accumulation window
/// has closed -- form the error and move the duty.
///
/// The split between SAMPLING and DECIDING is the whole design. Sampling is
/// fast (0.5 s) because the average wants many points; deciding is slow
/// (`avg_window`, 9 s) because the plant being measured is a pulse train whose
/// period is ~3 s (see `avg_window`). One decision per pulse would be a
/// decision made on the phase of the loop's own output.
fn recompute(gpu: f32) {
    let Some(t) = THROTTLE.get() else { return };
    let Ok(mut s) = t.lock() else { return };

    // Off since the last tick (the picker's P key came back round to OFF).
    // Stopping here rather than publishing 100 is what keeps `publish` from
    // walking the duty back up on the next closed window.
    if s.off {
        return;
    }

    // `GS_THROTTLE_TRACE=1` dumps every DECISION -- the window mean, the duty
    // it produced, and which branch fired -- to stderr. It exists because the
    // only way to tell a control loop from a sawtooth is to see the command
    // and the response on the SAME time base; a GPU trace alone cannot say
    // whether a spike is a disturbance the loop missed or a correction the
    // loop made.
    macro_rules! trace {
        ($act:expr) => {
            if s.trace {
                eprintln!(
                    "[throttle] t={:.1} mean={:.0} duty={:.1} {}",
                    s.started.elapsed().as_secs_f32(),
                    s.last_mean,
                    s.duty,
                    $act
                );
            }
        };
    }

    // NO per-sample branch lives here any more, and that is the design, not an
    // omission. Two versions tried to keep one -- first a single sample over
    // the target, then a run of such samples -- and both were defeated by the
    // same fact: while the encoder runs, a sample is ~100 % or ~40 % depending
    // only on where in the pulse train it landed (606 ms of load every ~3 s at
    // duty 20; the pulse gets DENSER as the duty rises, so the "run of
    // over-target samples" test passes trivially at high duty). A per-sample
    // branch is therefore a branch on the phase of the loop's own output, and
    // nothing in the NVML reading can separate the encoder's pulse from a
    // genuine external burst -- they are the same shape at this sampling rate.
    //
    // The average branch below is the whole controller. Its response to a real
    // external burst is one `avg_window` slower than a per-sample cut would be,
    // which is the honest price of not floor-locking on every pulse.

    // Accumulate, and do nothing else until the window closes.
    s.acc.push(gpu);
    let done = s.acc_start.elapsed().as_secs_f32() >= s.cfg.avg_window;
    if !done {
        return;
    }

    // The grace window: no correction for the first `grace` seconds. The
    // accumulation still runs, so the first decision has a full window behind
    // it rather than three samples.
    let n = s.acc.len() as f32;
    let mean: f32 = s.acc.iter().sum::<f32>() / n.max(1.0);
    s.acc.clear();
    s.acc_start = Instant::now();
    s.last_mean = mean;

    if s.started.elapsed().as_secs_f32() < s.cfg.grace {
        publish(&s);
        trace!("grace");
        return;
    }

    let error = mean - s.cfg.gpu_target;

    // Guard 1, the deadband.
    if error.abs() <= s.cfg.deadzone {
        publish(&s);
        trace!("dead");
        return;
    }

    // Guard 4, dwell: a reversal is not allowed until the last move has held.
    let dir: i8 = if error > 0.0 { -1 } else { 1 };
    if dir != s.last_dir
        && s.last_dir != 0
        && s.last_change.elapsed().as_secs_f32() < s.cfg.dwell
    {
        publish(&s);
        trace!("dwell");
        return;
    }

    // The correction: plain proportional, then clamped on BOTH sides.
    //
    // There is no downward "boost" any more. An earlier version multiplied the
    // down-move by up to 3x when the overshoot was large, on the theory that
    // walking duty 100 -> 60 should take a few decisions and not a few minutes.
    // Measured against a low target it did exactly that and more: at
    // `gpu_target` 50, `kp` 3 and a deadzone of 1, an error of 37 points turned
    // into a boost of 13 (clamped to 3) and the very first decision moved the
    // duty from 64 to 20 -- straight through the plant and into the floor, in
    // one step, from which the loop then had to climb back at `rise_limit` 4
    // per decision. Overshoot on the DOWN side is the same disease as overshoot
    // on the up side, and the cure is the same: cap the step.
    //
    // `kp * error` is already aggressive enough at the shipped gains -- at the
    // default `kp` 3 an error of 10 points asks for 30 duty points, which is a
    // third of the whole range in one move.
    let mut delta = -s.cfg.kp * error;
    // Guard 2, fast down / slow up. Down stays uncapped in PROPORTION but is
    // still bounded by `fall_limit`, so a large error cannot floor-lock the
    // loop; up is bounded by `rise_limit`, the cautious direction.
    if delta > 0.0 {
        if delta > s.cfg.rise_limit {
            delta = s.cfg.rise_limit;
        }
    } else if delta < -s.cfg.fall_limit {
        delta = -s.cfg.fall_limit;
    }
    let before = s.duty;
    s.duty = (s.duty + delta).clamp(s.cfg.duty_min, 100.0);
    if (s.duty - before).abs() > 0.01 {
        s.last_change = Instant::now();
        s.last_dir = if s.duty < before { -1 } else { 1 };
    }
    publish(&s);
    trace!(format!("{delta:+.1}"));
}

/// Mirror the duty into the atomic the encode path reads, with the optional
/// `script_max` clamp applied here (so the hot path is one atomic load).
fn publish(s: &State) {
    let d = s.duty.min(s.cfg.script_max).clamp(1.0, 100.0);
    DUTY_X100.store((d * 100.0) as u32, Ordering::Relaxed);
}

/// The current duty, 1..=100. One atomic load; no lock.
#[inline]
pub fn duty() -> f32 {
    DUTY_X100.load(Ordering::Relaxed) as f32 / 100.0
}

/// How long to nap after a window that took `window: Duration`, at the current
/// duty.
///
/// `rest = window * (100 - duty) / duty`, so the pair (work, rest) averages to
/// `duty` percent busy. A duty of 100 gives zero, which the caller skips.
#[inline]
pub fn rest_after(window: Duration) -> Duration {
    let d = duty();
    if d >= 100.0 {
        return Duration::ZERO;
    }
    let d = d.clamp(1.0, 100.0);
    window.mul_f32((100.0 - d) / d)
}

/// Whether a throttle is installed and running.
///
/// `install` reports itself on stderr when it arms, so nothing in the app has
/// to call this to find out; it stays as the predicate a test or a future
/// caller would use.
#[allow(dead_code)]
pub fn active() -> bool {
    RUNNING.load(Ordering::Relaxed)
}

/// Parse `--throttle`'s value into a config. `custom` needs the overrides.
/// Unknown names are an error with the list of presets.
pub fn parse(name: &str, ov: &Overrides) -> Result<Config> {
    let mut cfg = Config::preset(name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown --throttle preset {name:?}; expected one of: {}",
            PRESETS.join(", ")
        )
    })?;
    cfg.apply_overrides(ov);
    Ok(cfg)
}

// --------------------------------------------------------------------------
// NVML, loaded by hand.
//
// `nvml-wrapper` would do this, but it pins a whole `nvml-wrapper-sys`
// dependency for three calls, and the DLL is present on every machine that can
// run the CUDA/DirectML build this throttle exists for. Loading it here keeps
// the dependency graph unchanged and makes "no NVML" a state we handle rather
// than a build-time decision.
// --------------------------------------------------------------------------
mod nvml_util {
    use anyhow::{anyhow, Result};
    use std::ffi::c_void;

    #[repr(C)]
    struct Utilization {
        gpu: u32,
        memory: u32,
    }

    type InitFn = unsafe extern "C" fn() -> i32;
    type HandleByIndexFn = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
    type UtilizationFn = unsafe extern "C" fn(*mut c_void, *mut Utilization) -> i32;

    pub struct Nvml {
        _lib: libloading::Library,
        handle: *mut c_void,
        utilization: UtilizationFn,
    }

    // The handle is a device pointer NVML owns; the calls that use it are
    // synchronous reads. Moving the wrapper between threads is fine.
    unsafe impl Send for Nvml {}
    unsafe impl Sync for Nvml {}

    impl Nvml {
        pub fn open() -> Result<Self> {
            // nvml.dll ships with the driver and lives in System32.
            unsafe {
                let lib = libloading::Library::new("nvml.dll")
                    .map_err(|e| anyhow!("loading nvml.dll: {e}"))?;

                // Every symbol below is fetched, dereferenced to a plain fn
                // pointer and the `Symbol` (which borrows `lib`) dropped before
                // the move of `lib` into `Self`. That is why the fn pointers
                // are copied out in a block of their own rather than the
                // `Symbol`s being held.
                let init: InitFn = *lib
                    .get(b"nvmlInit_v2\0")
                    .or_else(|_| lib.get(b"nvmlInit\0"))
                    .map_err(|e| anyhow!("nvmlInit: {e}"))?;
                if init() != 0 {
                    return Err(anyhow!("nvmlInit returned an error"));
                }
                let by_index: HandleByIndexFn = *lib
                    .get(b"nvmlDeviceGetHandleByIndex_v2\0")
                    .or_else(|_| lib.get(b"nvmlDeviceGetHandleByIndex\0"))
                    .map_err(|e| anyhow!("nvmlDeviceGetHandleByIndex: {e}"))?;
                let utilization: UtilizationFn = *lib
                    .get(b"nvmlDeviceGetUtilizationRates\0")
                    .map_err(|e| anyhow!("nvmlDeviceGetUtilizationRates: {e}"))?;

                let mut handle: *mut c_void = std::ptr::null_mut();
                if by_index(0, &mut handle) != 0 || handle.is_null() {
                    return Err(anyhow!("no NVIDIA device at index 0"));
                }
                Ok(Self {
                    _lib: lib,
                    handle,
                    utilization,
                })
            }
        }

        /// Whole-device 3D utilization, percent.
        pub fn utilization(&self) -> Result<u32> {
            let mut u = Utilization { gpu: 0, memory: 0 };
            let rc = unsafe { (self.utilization)(self.handle, &mut u) };
            if rc != 0 {
                return Err(anyhow!("nvmlDeviceGetUtilizationRates returned {rc}"));
            }
            Ok(u.gpu)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The P key's cycle, which is the one piece of this module a user meets
    /// directly. Off must be reachable from the last preset and ONLY from it in
    /// one step, or the key would walk past the setting the user wanted.
    #[test]
    fn picker_cycles_off_through_every_preset_back_to_off() {
        let mut cur: Option<&str> = None;
        let mut seen = Vec::new();
        for _ in 0..PICKER_CYCLE.len() + 1 {
            cur = next_preset(cur);
            seen.push(cur);
        }
        assert_eq!(
            seen,
            vec![
                Some("gentle"),
                Some("balanced"),
                Some("aggressive"),
                None,
            ]
        );
        // and one more press is back to the top of the lap, not a dead end
        assert_eq!(next_preset(cur), Some("gentle"));
    }

    /// A name this build does not cycle through -- a hand-edited `custom`, or a
    /// preset a future build drops -- must land on a preset that EXISTS on the
    /// next press, not step into nothing.
    #[test]
    fn unknown_name_restarts_the_cycle() {
        assert_eq!(next_preset(Some("custom")), None);
        assert_eq!(next_preset(Some("not-a-preset")), None);
    }

    /// `describe` is what the POST line prints, so it must never panic and must
    /// say something for every input a settings file could hold.
    #[test]
    fn describe_survives_anything_a_file_could_hold() {
        assert_eq!(describe(None), "off");
        assert_eq!(describe(Some("custom")), "custom");
        let b = describe(Some("balanced"));
        assert!(b.contains("balanced") && b.contains("90"), "{b}");
        assert!(describe(Some("garbage")).contains("garbage"));
    }

    /// The step caps are the fix for the floor-lock: a proportional move is
    /// clipped on BOTH sides. This pins the arithmetic the low-target run
    /// depends on, without a GPU.
    #[test]
    fn a_large_error_cannot_floor_the_duty_in_one_step() {
        let cfg = Config::preset("balanced").unwrap();
        // error of 37 points at kp 3 asks for 111 duty points -- a third of the
        // range more than the whole floor-to-ceiling span
        let error = 37.0f32;
        let raw = -cfg.kp * error;
        assert!(raw < -cfg.fall_limit, "test premise: {raw} exceeds the cap");
        let delta = raw.max(-cfg.fall_limit);
        let duty = (60.0f32 + delta).clamp(cfg.duty_min, 100.0);
        assert_eq!(duty, 40.0, "one step from 60 lands on 40, not on the floor");
    }

    /// A preset this module ships must always be armable: the config the picker
    /// builds has to pass every check `arm_throttle` makes in `main`, or the P
    /// key would offer a key that fails the moment it is pressed.
    #[test]
    fn shipped_presets_are_self_consistent() {
        for name in PICKER_CYCLE {
            let c = Config::preset(name).unwrap_or_else(|| panic!("no preset {name}"));
            assert!((1.0..=100.0).contains(&c.gpu_target), "{name} target");
            assert!((1.0..=100.0).contains(&c.duty_min), "{name} duty_min");
            assert!((1.0..=100.0).contains(&c.duty_start), "{name} duty_start");
            assert!(c.duty_start >= c.duty_min, "{name} starts below its own floor");
            assert!(c.rise_limit > 0.0 && c.fall_limit > 0.0, "{name} step caps");
            assert!(c.avg_window > 0.0 && c.interval > 0.0, "{name} times");
            assert!((0.0..=100.0).contains(&c.script_max), "{name} script_max");
        }
    }
}
