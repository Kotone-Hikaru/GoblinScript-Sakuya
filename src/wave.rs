//! Waveforms for manual spans: the shapes, as DATA.
//!
//! A wave is a list of segments walked from the start of a cycle -- each with a
//! relative duration, a level to reach and an easing curve. `src/waves.json`
//! holds them and documents the format; this file only evaluates it.
//!
//! Three rules keep a shape cheap to change, and they are the reason the shapes
//! live outside the code:
//!
//!   * `eval` is a pure function of `(t, params)` returning `[0,1]`. It knows
//!     nothing about the lower/upper bounds, the cadence, the repeat count, the
//!     phase or the two mirror switches -- a caller applies all of those. So the
//!     preview thumbnail, the export and the tests all run the SAME code, and a
//!     wave cannot mean one thing on screen and another in the file.
//!   * Weights are relative. `rise 13 / fall 37 / floor 50` redistributes itself
//!     when one of them moves, which is what those numbers read as.
//!   * A param that is absent falls back to the declared default, so a wave that
//!     grows a new knob does not invalidate spans saved before it existed.
//!
//! `waves.json` is read from BESIDE THE EXE when it is there, and from the copy
//! baked in at build time otherwise -- same arrangement as the review page, for
//! the same reason: changing a shape should cost an edit and a page reload, not
//! a rebuild. Read per request rather than cached, so replacing the file is all
//! it takes.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A number, or the name of a param. Untagged so the JSON can write either
/// bare: `"w": 13` and `"w": "rise"` both parse.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Val {
    Num(f64),
    Str(String),
}

impl Val {
    pub fn num(&self) -> Option<f64> {
        match self {
            Val::Num(v) => Some(*v),
            Val::Str(_) => None,
        }
    }
    fn text(&self) -> Option<&str> {
        match self {
            Val::Num(_) => None,
            Val::Str(s) => Some(s),
        }
    }
}

/// Resolved parameters for one span. Values keep the units the file declares
/// (`%` params stay in percent), so what the page shows is what the file says.
pub type Params = HashMap<String, Val>;

/// The easing curves a segment may name. A segment's `ease` that matches none
/// of these is read as a PARAM name instead -- that is how `glide` offers its
/// curve as a dropdown rather than hard-coding one.
pub const EASINGS: &[&str] = &["linear", "sine", "smooth", "in", "in3", "out", "out3"];

/// `f(0) = 0`, `f(1) = 1`, monotone. The wave's shape comes from mixing these,
/// so a new shape usually needs a new line HERE and none anywhere else.
pub fn ease(name: &str, u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    match name {
        "sine" => 0.5 - 0.5 * (std::f64::consts::PI * u).cos(),
        "smooth" => u * u * (3.0 - 2.0 * u),
        "in" => u * u,
        "in3" => u * u * u,
        "out" => 1.0 - (1.0 - u) * (1.0 - u),
        "out3" => 1.0 - (1.0 - u).powi(3),
        _ => u,
    }
}

/// One knob on a wave. The page builds the control from this alone, so adding
/// a knob is one entry in `waves.json` and no front-end work at all.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParamSpec {
    pub key: String,
    /// i18n key.
    pub label: String,
    /// `"num"` (default) or `"choice"`.
    #[serde(default = "kind_num")]
    pub kind: String,
    #[serde(rename = "def")]
    pub default: Val,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub unit: String,
    #[serde(default)]
    pub options: Vec<String>,
}

fn kind_num() -> String {
    "num".to_string()
}

/// One leg of a cycle.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Segment {
    /// Relative duration: a number, or a param name. Normalized across the
    /// wave, so the numbers can be read as shares of a cycle.
    pub w: Val,
    /// Level reached at the END of the leg -- 0 to 100 on the wave's own scale,
    /// where 0 is the lower bound and 100 the upper -- or a param name, read on
    /// that same scale.
    pub to: Val,
    /// An [`EASINGS`] name, or a param name whose choice picks one.
    pub ease: Val,
}

impl Segment {
    fn weight(&self, p: &Params) -> f64 {
        match &self.w {
            Val::Num(v) => *v,
            Val::Str(k) => p.get(k).and_then(Val::num).unwrap_or(1.0),
        }
        .max(0.0)
    }
    fn target(&self, p: &Params) -> f64 {
        let raw = match &self.to {
            Val::Num(v) => *v,
            Val::Str(k) => p.get(k).and_then(Val::num).unwrap_or(0.0),
        };
        raw / 100.0
    }
    fn ease_name(&self, p: &Params) -> String {
        let named = match self.ease.text() {
            Some(s) => s,
            None => return "linear".to_string(),
        };
        if EASINGS.contains(&named) {
            return named.to_string();
        }
        // not a curve: it names a param holding one (the glide wave's dropdown)
        p.get(named)
            .and_then(Val::text)
            .unwrap_or("linear")
            .to_string()
    }
}

/// One waveform.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Wave {
    /// Stable id, stored in a span. Never an index: inserting a wave into the
    /// middle of the list must not re-point spans that already exist.
    pub id: String,
    /// i18n key for the picker.
    pub name: String,
    #[serde(default)]
    pub hint: String,
    /// Wrapped around (almost every wave), or drawn once across the span
    /// (glide, which never comes back). A one-shot wave ignores the cadence.
    #[serde(default = "yes")]
    pub periodic: bool,
    #[serde(default)]
    pub params: Vec<ParamSpec>,
    pub segments: Vec<Segment>,
}

fn yes() -> bool {
    true
}

impl Wave {
    /// The normalized shape: `t` is position in one cycle, `v` is the level
    /// between the bounds, 0 at the lower one and 1 at the upper.
    ///
    /// A cycle always STARTs at 0, which is what makes "start from the lower
    /// bound" the natural default and the mirror switches a caller's problem.
    pub fn eval(&self, t: f64, p: &Params) -> f64 {
        let last = match self.segments.len() {
            0 => return 0.0,
            n => n - 1,
        };
        let weights: Vec<f64> = self.segments.iter().map(|s| s.weight(p)).collect();
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) {
            return 0.0;
        }
        let t = t.clamp(0.0, 1.0);
        let mut acc = 0.0;
        let mut from = 0.0;
        for (i, seg) in self.segments.iter().enumerate() {
            let span = weights[i] / total;
            let to = seg.target(p);
            if i == last || t < acc + span {
                let u = if span > 0.0 { (t - acc) / span } else { 1.0 };
                return from + (to - from) * ease(&seg.ease_name(p), u);
            }
            acc += span;
            from = to;
        }
        from
    }

    /// `n` samples across one cycle, for the panel's thumbnail. Sampled by the
    /// server so the page never has to own a second copy of the maths.
    pub fn sample(&self, p: &Params, n: usize) -> Vec<f64> {
        let n = n.max(2);
        (0..n).map(|i| self.eval(i as f64 / n as f64, p)).collect()
    }

    /// Every declared param at its default.
    pub fn defaults(&self) -> Params {
        self.params
            .iter()
            .map(|s| (s.key.clone(), s.default.clone()))
            .collect()
    }

    /// The defaults, with `given` merged over them: what a span with a few
    /// saved knobs evaluates with.
    pub fn resolve(&self, given: &Params) -> Params {
        let mut p = self.defaults();
        for (k, v) in given {
            p.insert(k.clone(), v.clone());
        }
        p
    }
}

/// A manual span's recipe: which wave, over what band, at what cadence.
///
/// This is the whole of what a user dialled in -- the SHAPE still comes from
/// `waves.json`, so re-tuning a shape or adding a knob never touches a stored
/// span. Absent fields fall back, so a span saved by an older page still
/// evaluates; `params` is keyed and merged over the wave's own defaults for the
/// same reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveSpec {
    pub id: String,
    /// Band floor and ceiling in position units (0..100). Absolute, not
    /// relative to anything: what the panel's two boxes say is what the wave
    /// sweeps between.
    #[serde(default = "band_lo")]
    pub lo: f64,
    #[serde(default = "band_hi")]
    pub hi: f64,
    /// ONE CYCLE, in seconds -- the wave's own clock. The panel offers other
    /// units, but this is what gets stored.
    #[serde(default = "one_sec")]
    pub period_s: f64,
    #[serde(default)]
    pub params: Params,
    /// Mirror in time: the cycle runs backwards. Phase-LOCKED waves (a ramp, a
    /// ring-down) change shape under it, which is the point.
    #[serde(default)]
    pub mirror_time: bool,
    /// Mirror in amplitude: the band is walked upside down.
    #[serde(default)]
    pub mirror_amp: bool,
}

fn band_lo() -> f64 {
    25.0
}
fn band_hi() -> f64 {
    75.0
}
fn one_sec() -> f64 {
    1.0
}

impl WaveSpec {
    /// The level this wave asks for, in 0..100, `t` seconds into the span.
    ///
    /// Bounds, cadence, mirror and phase are all applied HERE and nowhere
    /// else, which is what lets `Wave::eval` stay a pure shape.
    pub fn level_at(&self, w: &Wave, t: f64, eff: &Params) -> f64 {
        let period = if self.period_s.is_finite() && self.period_s >= 0.02 {
            self.period_s
        } else {
            1.0
        };
        let mut u = t / period;
        u -= u.floor();
        if self.mirror_time {
            u = (1.0 - u).rem_euclid(1.0);
        }
        let mut v = w.eval(u, eff);
        if self.mirror_amp {
            v = 1.0 - v;
        }
        let (a, b) = if self.lo <= self.hi {
            (self.lo, self.hi)
        } else {
            (self.hi, self.lo)
        };
        (a + (b - a) * v).clamp(0.0, 100.0)
    }
}

/// Overwrite rows `[lo, hi)` of a level track with `spec`'s wave, and hand
/// `vmarg` the wave's own speed over those rows.
///
/// Both tracks, not just `level`: the gap filler hunts rows whose |vmarg| is
/// under `still_eps`, and a wave's own floor is exactly where it is slowest --
/// so a manual fill that only wrote levels would offer its own valleys up as
/// "still" and get rhythm stamped over them. False when the spec names a wave
/// this build does not have, or when the range is empty.
pub fn fill_rows(
    level: &mut [f64],
    vmarg: &mut [f64],
    times: &[f64],
    lo: usize,
    hi: usize,
    spec: &WaveSpec,
) -> bool {
    let set = load();
    let w = match set.get(&spec.id) {
        Some(w) => w,
        None => return false,
    };
    let n = level.len().min(vmarg.len()).min(times.len());
    let hi = hi.min(n);
    if hi <= lo {
        return false;
    }
    let eff = w.resolve(&spec.params);
    let t0 = times[lo];
    let dt = if hi > lo + 1 {
        (times[hi - 1] - t0) / (hi - lo - 1) as f64 / 1000.0
    } else {
        1.0 / 30.0
    };
    let dt = if dt > 1e-6 { dt } else { 1.0 / 30.0 };
    for i in lo..hi {
        level[i] = spec.level_at(w, (times[i] - t0) / 1000.0, &eff);
    }
    for i in lo..hi {
        let nxt = if i + 1 < hi { level[i + 1] } else { level[i] };
        vmarg[i] = ((nxt - level[i]) / dt).abs();
    }
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WaveSet {
    pub version: u32,
    #[serde(default)]
    pub waves: Vec<Wave>,
}

impl WaveSet {
    pub fn get(&self, id: &str) -> Option<&Wave> {
        self.waves.iter().find(|w| w.id == id)
    }
}

/// The wave file beside the exe if there is one, else the baked copy. Never
/// fails: an unreadable or unparseable loose file falls back rather than
/// taking the server down over a typo in a data file.
pub fn load() -> WaveSet {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p: PathBuf = dir.join("waves.json");
            if let Ok(s) = std::fs::read_to_string(&p) {
                if let Ok(set) = serde_json::from_str::<WaveSet>(&s) {
                    return set;
                }
            }
        }
    }
    serde_json::from_str(include_str!("waves.json")).expect("the baked waves.json must parse")
}

/// The whole set, for the page to build its picker and controls from.
pub fn value() -> serde_json::Value {
    serde_json::to_value(load()).unwrap_or_else(|_| serde_json::json!({ "waves": [] }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_baked_set_parses_and_ids_are_unique() {
        let set = load();
        assert!(!set.waves.is_empty(), "no waves in the baked file");
        let mut seen = std::collections::HashSet::new();
        for w in &set.waves {
            assert!(!w.id.is_empty(), "a wave has no id");
            assert!(!w.segments.is_empty(), "{} has no segments", w.id);
            assert!(seen.insert(w.id.clone()), "duplicate wave id {}", w.id);
            assert!(w.name.starts_with("wave."), "{} has no i18n name", w.id);
        }
    }

    #[test]
    fn every_wave_stays_between_the_bounds_and_closes_the_cycle() {
        for w in load().waves {
            let p = w.defaults();
            let v = w.sample(&p, 128);
            for (i, x) in v.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(x),
                    "{} sample {i} = {x} left the bounds",
                    w.id
                );
            }
            assert!(v[0].abs() < 1e-9, "{} must start at the lower bound", w.id);
            // a periodic wave has to come back, or repeating it would snap
            if w.periodic {
                assert!(
                    v[127].abs() < 0.05,
                    "{} ends at {} instead of near the lower bound",
                    w.id,
                    v[127]
                );
            }
        }
    }

    #[test]
    fn a_missing_param_is_not_a_shape_change() {
        // the whole point of defaults: a span saved before a knob existed still
        // evaluates, and evaluates to the same shape
        for w in load().waves {
            let empty = w.resolve(&Params::new());
            let full = w.defaults();
            let a = w.sample(&empty, 64);
            let b = w.sample(&full, 64);
            for i in 0..64 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-12,
                    "{} differs at {i} once defaults are filled in",
                    w.id
                );
            }
        }
    }

    #[test]
    fn a_choice_param_can_drive_the_easing() {
        let set = load();
        let g = set.get("glide").expect("the glide wave is part of the set");
        let mut p = g.defaults();
        let smooth = g.eval(0.5, &p);
        p.insert("ease".to_string(), Val::Str("linear".to_string()));
        let linear = g.eval(0.5, &p);
        assert!(
            (smooth - 0.5).abs() < 1e-9 && (linear - 0.5).abs() < 1e-9,
            "both curves pass the midpoint"
        );
        let q = g.eval(0.25, &p);
        assert!((q - 0.25).abs() < 1e-9, "linear is exactly linear");
    }

    #[test]
    fn weights_are_shares_so_moving_one_moves_the_others() {
        let set = load();
        let w = set.get("single_peak").expect("single_peak is part of the set");
        let p = w.defaults();
        let peak_at = |p: &Params| {
            let v = w.sample(p, 1000);
            let mut best = 0;
            for i in 1..v.len() {
                if v[i] > v[best] {
                    best = i;
                }
            }
            best as f64 / 1000.0
        };
        let before = peak_at(&p);
        let mut stretched = p.clone();
        stretched.insert("rise".to_string(), Val::Num(26.0)); // double the rise
        let after = peak_at(&stretched);
        assert!(
            after > before + 0.05,
            "doubling the rise weight should push the peak later ({before} -> {after})"
        );
    }
}
