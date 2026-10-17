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

/// A tiny arithmetic expression over a wave's params: numbers, param names,
/// `+ - * /`, parentheses and a unary minus.
///
/// It exists so a SHAPE can stay DATA. A knob that moves a vertex -- the apex
/// of a triangle, the corner radius of a rounded one -- is a RATIO of the
/// legs, and writing that ratio as a number would freeze it at one setting:
/// `"1-peak"` and `"round/2"` let `waves.json` say what it means, which is the
/// file's own rule (a new knob is one entry, and no code moves). A bare name
/// is still a bare name: `"rise"` resolves through this same table and
/// evaluates to the number it always did, so every wave written before this
/// parses to exactly the shape it had.
fn eval_expr(src: &str, p: &Params) -> Option<f64> {
    let mut e = Expr { s: src.as_bytes(), i: 0, p };
    let v = e.sum()?;
    e.ws();
    if e.i == e.s.len() {
        Some(v)
    } else {
        None
    }
}

struct Expr<'a> {
    s: &'a [u8],
    i: usize,
    p: &'a Params,
}

impl Expr<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn sum(&mut self) -> Option<f64> {
        let mut v = self.prod()?;
        loop {
            self.ws();
            match self.s.get(self.i).copied() {
                Some(b'+') => {
                    self.i += 1;
                    v += self.prod()?;
                }
                Some(b'-') => {
                    self.i += 1;
                    v -= self.prod()?;
                }
                _ => return Some(v),
            }
        }
    }
    fn prod(&mut self) -> Option<f64> {
        let mut v = self.unary()?;
        loop {
            self.ws();
            match self.s.get(self.i).copied() {
                Some(b'*') => {
                    self.i += 1;
                    v *= self.unary()?;
                }
                Some(b'/') => {
                    self.i += 1;
                    let d = self.unary()?;
                    // a zero divisor is a malformed shape, not a zero leg: the
                    // caller falls back rather than shipping a NaN width
                    if d == 0.0 {
                        return None;
                    }
                    v /= d;
                }
                _ => return Some(v),
            }
        }
    }
    fn unary(&mut self) -> Option<f64> {
        self.ws();
        match self.s.get(self.i).copied() {
            Some(b'-') => {
                self.i += 1;
                Some(-self.unary()?)
            }
            Some(b'+') => {
                self.i += 1;
                self.unary()
            }
            _ => self.atom(),
        }
    }
    fn atom(&mut self) -> Option<f64> {
        self.ws();
        let c = *self.s.get(self.i)?;
        if c == b'(' {
            self.i += 1;
            let v = self.sum()?;
            self.ws();
            if self.s.get(self.i) != Some(&b')') {
                return None;
            }
            self.i += 1;
            return Some(v);
        }
        if c.is_ascii_digit() || c == b'.' {
            let start = self.i;
            while self.i < self.s.len()
                && (self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.')
            {
                self.i += 1;
            }
            return std::str::from_utf8(&self.s[start..self.i]).ok()?.parse().ok();
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = self.i;
            while self.i < self.s.len()
                && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
            {
                self.i += 1;
            }
            let name = std::str::from_utf8(&self.s[start..self.i]).ok()?;
            return self.p.get(name).and_then(Val::num);
        }
        None
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
            Val::Str(k) => eval_expr(k, p).unwrap_or(1.0),
        }
        .max(0.0)
    }
    fn target(&self, p: &Params) -> f64 {
        let raw = match &self.to {
            Val::Num(v) => *v,
            Val::Str(k) => eval_expr(k, p).unwrap_or(0.0),
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

/// The WHOLE range is the default the panel offers too. A wave that silently
/// swept 25..75 would spend two thirds of the hand-drawn travel on nothing, and
/// the number the user reads back would not be the number they asked for.
fn band_lo() -> f64 {
    0.0
}
fn band_hi() -> f64 {
    100.0
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

/// Vertical-deviation Douglas-Peucker over `(ms, level)` samples, the same
/// rule `style::rdp` applies to a written action list -- so the vertices a
/// drawing resolves to and the vertices RDP would keep of the same drawing
/// agree about what "the same shape" means.
fn rdp_points(pts: &[(f64, f64)], eps: f64) -> Vec<(f64, f64)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let n = pts.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b - a < 2 {
            continue;
        }
        let (ta, pa) = pts[a];
        let (tb, pb) = pts[b];
        let dt = (tb - ta).max(1e-9);
        let mut best = a + 1;
        let mut bd = -1.0f64;
        for (o, p) in pts[a + 1..b].iter().enumerate() {
            let line = pa + (pb - pa) * (p.0 - ta) / dt;
            let d = (p.1 - line).abs();
            if d > bd {
                bd = d;
                best = a + 1 + o;
            }
        }
        if bd > eps {
            keep[best] = true;
            stack.push((a, best));
            stack.push((best, b));
        }
    }
    pts.iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| *p)
        .collect()
}

/// How far a solved vertex may sit off the wave, position units. The panel's
/// own accuracy need, not the device's: two units is under a fiftieth of the
/// travel and under what a hand can place, while an eased wave still keeps its
/// arc (a sine comes out around thirteen points a cycle; a straight leg two).
pub const VERTEX_EPS: f64 = 2.0;

/// The vertices of one hand-drawn piece, SOLVED from the wave rather than
/// sampled on the row grid. Returns `(milliseconds into the span, level 0..100)`
/// in ascending time; empty when the spec names a wave this build lacks.
///
/// Why this replaces the row scan. A row scan can only report the wave at row
/// times, and at a cadence whose half-period is not a whole number of rows the
/// top of every cycle falls BETWEEN two rows: the peak is then written at
/// whatever the neighbouring rows hold (93 for a 0.5 s triangle on a 30 rows/s
/// grid) and the piece never reaches the band it was given. The same scan hands
/// the vertex picker a marginal that flips sign wherever integer rounding
/// jitters, which it reads as extra reversals in the middle of the piece.
///
/// A solved piece has neither problem. Every leg boundary is a vertex at its
/// exact time and its exact level -- a boundary is where one easing hands over
/// to the next, so a peak lands on the ceiling exactly -- and inside a leg the
/// curve is thinned to `eps` units, which keeps an eased arc and leaves a
/// straight leg as its two endpoints.
pub fn span_vertices(spec: &WaveSpec, span_ms: f64, eps: f64) -> Vec<(f64, f64)> {
    let set = load();
    let w = match set.get(&spec.id) {
        Some(w) => w,
        None => return Vec::new(),
    };
    let eff = w.resolve(&spec.params);
    let period = if spec.period_s.is_finite() && spec.period_s >= 0.02 {
        spec.period_s
    } else {
        1.0
    };
    let span_s = (span_ms / 1000.0).max(0.0);
    if span_s <= 0.0 {
        return Vec::new();
    }
    let weights: Vec<f64> = w.segments.iter().map(|s| s.weight(&eff)).collect();
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return Vec::new();
    }
    let mut shares: Vec<f64> = vec![0.0];
    let mut acc = 0.0;
    for x in &weights {
        acc += x / total;
        shares.push(acc);
    }
    // Where a boundary sits in a cycle, on the SPAN's clock. `mirror_time` is
    // applied to `u` by the caller, so a boundary at share `b` is reached at
    // `1 - b` of the cycle; the boundary set is the same one walked backwards.
    let mut knots: Vec<f64> = Vec::new();
    let cycles = (span_s / period).ceil() as usize + 1;
    for k in 0..cycles {
        for b in &shares {
            let f = if spec.mirror_time {
                (1.0 - b).rem_euclid(1.0)
            } else {
                *b
            };
            let t = k as f64 * period + f * period;
            if t <= span_s + 1e-9 {
                knots.push(t.clamp(0.0, span_s));
            }
        }
    }
    knots.push(0.0);
    knots.push(span_s);
    knots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    knots.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in 0..knots.len() {
        let ta = knots[i];
        // the boundary is exact and is never thinned away
        out.push((ta, spec.level_at(w, ta, &eff)));
        let tb = match knots.get(i + 1) {
            Some(&tb) => tb,
            None => continue,
        };
        if tb - ta <= 1e-9 {
            continue;
        }
        // 240 Hz of sampling under the thinner: far finer than any leg an
        // easing can draw, so the thinned result is the curve and not a
        // property of the grid it was measured on
        let n = (((tb - ta) * 240.0).ceil() as usize).max(8);
        let leg: Vec<(f64, f64)> = (0..=n)
            .map(|j| {
                let t = ta + (tb - ta) * j as f64 / n as f64;
                (t, spec.level_at(w, t, &eff))
            })
            .collect();
        // skip(1): the leg's start is the boundary already emitted above
        out.extend(rdp_points(&leg, eps).into_iter().skip(1));
    }
    // the walk above is on the wave's own clock, which is SECONDS; the caller
    // places these on a span's millisecond clock, so hand them over in ms
    for p in &mut out {
        p.0 *= 1000.0;
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
    out
}

/// One hand-drawn piece as the styler needs it: the span's clock, its recipe,
/// and the two switches the panel offers beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct ManualSpan {
    pub lo_ms: f64,
    pub hi_ms: f64,
    pub spec: WaveSpec,
    /// Solve the piece at its own slopes and do not run the device's speed cap
    /// over it. ON by default: choosing to draw a piece by hand is taking
    /// responsibility for it, and the cap is what flattened a fast drawing's
    /// peaks.
    pub no_limit: bool,
    /// Leave the shot cuts that fall inside the piece unmarked.
    pub no_cuts: bool,
}

/// Overwrite rows `[lo, hi)` of a level track with `spec`'s wave, and hand
/// `vmarg` the wave's own speed over those rows.
///
/// Both tracks, and BOTH SIGNS. `level` is what the piece IS; `vmarg` is what
/// makes the styler see it: `compose` cuts a shot into strokes on this track's
/// sign flips (`style::crossings`), so a wave handed over as a magnitude reads
/// as one endless up-stroke and is composited back out as a single ramp -- the
/// amplitude survives, the shape the user drew does not. The tracked `.abs()`
/// readers (the gap filler's stillness test, `stroke_veto`) take their own
/// magnitude, so a signed track costs them nothing. False when the spec names
/// a wave this build does not have, or when the range is empty.
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
    // SIGNED forward difference, off the wave itself rather than off the row
    // that follows, so the last row of a piece has a real slope (the wave is a
    // function of time, so its next value exists past `hi`) and the track never
    // ends on a fabricated zero.
    for i in lo..hi {
        let t = (times[i] - t0) / 1000.0;
        vmarg[i] = (spec.level_at(w, t + dt, &eff) - level[i]) / dt;
    }
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WaveSet {
    pub version: u32,
    #[serde(default)]
    pub waves: Vec<Wave>,
    /// One-click fills for the strip's context menu. Data in here for the same
    /// reason the waves are: a new preset is an entry, not a code path.
    #[serde(default)]
    pub fills: Vec<Fill>,
}

/// A one-click fill: the whole recipe for a manual piece, so a stretch of
/// script becomes it in a single gesture.
///
/// `cell` is the menu grid position, written "row-column" ("1-1" is the top
/// left, "3-3" the bottom right). The page lays its cells out by that rule, so
/// an unused cell stays a deliberate hole instead of shifting everything up.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Fill {
    pub cell: String,
    /// i18n key for the menu label.
    pub label: String,
    /// The wave this preset draws with.
    pub wave: String,
    /// Absolute band in position units (0..100) -- the same numbers the
    /// panel's two boxes carry.
    pub lo: f64,
    pub hi: f64,
    /// Cycles per second, because that is how a preset is described out loud.
    /// Converted to the span's own seconds-per-cycle in `spec`.
    pub rate: f64,
    /// Knob overrides, merged over the wave's own defaults.
    #[serde(default)]
    pub params: Params,
}

impl Fill {
    /// The span this preset writes into a piece: the named wave, this preset's
    /// band and cadence, and BOTH exemptions on -- a fill is drawn rather than
    /// decoded, so neither the speed ceiling nor the shot cuts should be
    /// allowed to redraw it.
    pub fn spec(&self, set: &WaveSet) -> Option<WaveSpec> {
        let w = set.get(&self.wave)?;
        let mut p = Params::new();
        for s in &w.params {
            p.insert(s.key.clone(), s.default.clone());
        }
        for (k, v) in &self.params {
            p.insert(k.clone(), v.clone());
        }
        Some(WaveSpec {
            id: self.wave.clone(),
            lo: self.lo,
            hi: self.hi,
            period_s: if self.rate > 0.0 { 1.0 / self.rate } else { 1.0 },
            params: p,
            mirror_time: false,
            mirror_amp: false,
        })
    }
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

/// The fill presets, ready for the page: the grid cell, the label key, and the
/// finished wave spec to drop into a piece. Assembled here so the page never
/// keeps a second copy of what "a sine at 7 a second" means.
pub fn fills_json() -> serde_json::Value {
    let set = load();
    let items: Vec<serde_json::Value> = set
        .fills
        .iter()
        .filter_map(|f| {
            let spec = f.spec(&set)?;
            Some(serde_json::json!({ "cell": f.cell, "label": f.label, "wave": spec }))
        })
        .collect();
    serde_json::json!(items)
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

    /// `fill_rows` hands the styler a SIGNED marginal, and this is why.
    ///
    /// `compose` cuts a shot into strokes on that track's sign flips
    /// (`style::crossings`). A wave handed over as a magnitude has no flips at
    /// all, so the whole piece reads as one endless up-stroke and composites
    /// back out as a single ramp: the amplitude survives, the shape the user
    /// drew does not. That was the bug this test pins.
    #[test]
    fn fill_rows_writes_both_signs_so_a_drawn_wave_keeps_its_shape() {
        let n = 30; // one second on a 30 rows/s grid
        let times: Vec<f64> = (0..n).map(|i| i as f64 * 1000.0 / 30.0).collect();
        let spec = WaveSpec {
            id: "triangle".to_string(),
            lo: 0.0,
            hi: 100.0,
            period_s: 1.0,
            params: Params::new(),
            mirror_time: false,
            mirror_amp: false,
        };
        let mut level = vec![50.0; n];
        let mut vmarg = vec![0.0; n];
        assert!(fill_rows(&mut level, &mut vmarg, &times, 0, n, &spec));
        // the band the user typed, not a model's: 0..100 starts at the floor
        assert!(level[0].abs() < 1e-6, "starts at the floor, got {}", level[0]);
        assert!(
            level.iter().cloned().fold(f64::NEG_INFINITY, f64::max) > 95.0,
            "the triangle must reach the ceiling it was given"
        );
        // both directions present
        assert!(vmarg.iter().any(|&v| v > 1e-6), "no rising rows");
        assert!(
            vmarg.iter().any(|&v| v < -1e-6),
            "a magnitude track reads as one endless up-stroke"
        );
        let flips = vmarg.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(flips >= 1, "the marginal never changed sign, so one stroke");
        // and it is the DRAWN slope, not a fabricated one
        let dt = 1.0 / 30.0;
        for i in 0..n - 1 {
            let want = (level[i + 1] - level[i]) / dt;
            assert!(
                (vmarg[i] - want).abs() < 1e-6,
                "row {i}: vmarg {} vs the drawn step {want}",
                vmarg[i]
            );
        }
    }

    /// The arithmetic in `w`/`to`, which is what lets a knob move a vertex
    /// without any code moving: `"1-peak"`, `"round/2"`, `"apex*(100-round)"`.
    #[test]
    fn a_leg_expresses_itself_arithmetically() {
        let mut p = Params::new();
        p.insert("peak".to_string(), Val::Num(0.2));
        p.insert("round".to_string(), Val::Num(25.0));
        // a bare name is still a bare name
        assert_eq!(eval_expr("peak", &p), Some(0.2));
        assert_eq!(eval_expr("round", &p), Some(25.0));
        // the ratios shapes are actually written in
        let close = |got: Option<f64>, want: f64| match got {
            Some(v) => (v - want).abs() < 1e-9,
            None => false,
        };
        assert!(close(eval_expr("1-peak", &p), 0.8), "{:?}", eval_expr("1-peak", &p));
        assert!(close(eval_expr("1 - peak", &p), 0.8));
        assert!(close(eval_expr("round/2", &p), 12.5));
        assert!(close(eval_expr("peak*2", &p), 0.4));
        assert!(close(eval_expr("(100-round)*peak*2", &p), 30.0));
        assert!(close(eval_expr("round*(1-peak)*2", &p), 40.0));
        // precedence and sign: `-peak*2` is (-peak)*2, and parens win
        assert!(close(eval_expr("-peak*2", &p), -0.4));
        assert!(close(eval_expr("-(peak*2)", &p), -0.4));
        assert!(close(eval_expr("2*(3+4)", &p), 14.0));
        assert!(close(eval_expr("2.5", &p), 2.5));
        // a name nothing defines is not a number, and neither is a malformed
        // shape: both fall back at the call site rather than shipping NaN
        assert_eq!(eval_expr("nope", &p), None);
        assert_eq!(eval_expr("peak*", &p), None);
        assert_eq!(eval_expr("(peak", &p), None);
        assert_eq!(eval_expr("peak 2", &p), None);
        assert_eq!(eval_expr("1/0", &p), None);
        assert_eq!(eval_expr("", &p), None);
    }

    #[test]
    fn every_leg_in_the_baked_file_says_something_evaluable() {
        // a typo in waves.json would otherwise degrade silently to a leg of
        // weight 1 / target 0, which is a shape nobody asked for
        for w in load().waves {
            let p = w.defaults();
            for (i, s) in w.segments.iter().enumerate() {
                if let Val::Str(k) = &s.w {
                    assert!(
                        eval_expr(k, &p).is_some(),
                        "{} segment {i} weight {k:?} does not evaluate",
                        w.id
                    );
                }
                if let Val::Str(k) = &s.to {
                    assert!(
                        eval_expr(k, &p).is_some(),
                        "{} segment {i} target {k:?} does not evaluate",
                        w.id
                    );
                }
            }
        }
    }

    fn spec(id: &str, period_s: f64, params: &[(&str, f64)]) -> WaveSpec {
        let mut p = Params::new();
        for (k, v) in params {
            p.insert((*k).to_string(), Val::Num(*v));
        }
        WaveSpec {
            id: id.to_string(),
            lo: 0.0,
            hi: 100.0,
            period_s,
            params: p,
            mirror_time: false,
            mirror_amp: false,
        }
    }

    /// The bug this whole solver exists for: a triangle at a cadence whose
    /// half-period is not a whole number of 30 Hz rows used to be written at
    /// 93 because the peak fell BETWEEN two rows, and the row scan reported
    /// whatever the neighbours held.
    #[test]
    fn a_solved_triangle_reaches_its_ceiling_and_floor_exactly() {
        // 0.5 s a cycle on a 30 rows/s grid: a peak every 7.5 rows, so a row
        // scan can never land on one
        let v = span_vertices(&spec("triangle", 0.5, &[]), 10_300.0, VERTEX_EPS);
        assert!(!v.is_empty(), "the triangle solved to nothing");
        let top = v.iter().cloned().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let bot = v.iter().cloned().map(|p| p.1).fold(f64::INFINITY, f64::min);
        assert!(
            (top - 100.0).abs() < 1e-9,
            "the peak must reach the band's ceiling, got {top}"
        );
        assert!(
            (bot - 0.0).abs() < 1e-9,
            "the floor must reach the band's floor, got {bot}"
        );
        // and the vertices land ON those times: a triangle turns at its floor
        // and its peak, so a corner every half period, which at 0.5 s a cycle
        // is 250 ms. The span's own end is the one point that is not a corner:
        // the piece stops when the piece stops.
        let last = v.len() - 1;
        for (i, p) in v.iter().enumerate() {
            if i == last {
                continue;
            }
            let in_cycle = p.0 % 250.0;
            assert!(
                in_cycle < 1e-6 || (250.0 - in_cycle) < 1e-6,
                "a vertex at {} is off the half-period grid",
                p.0
            );
        }
    }

    /// "a triangle should only need three points": with the leg boundaries
    /// exact and the legs straight, thinner and boundary agree and nothing
    /// survives between them.
    #[test]
    fn a_straight_leg_keeps_only_its_endpoints() {
        let v = span_vertices(&spec("triangle", 0.5, &[]), 2_000.0, VERTEX_EPS);
        // four whole cycles, a floor and a peak each, plus the closing floor
        assert_eq!(
            v.len(),
            9,
            "a triangle must be its corners and nothing else, got {v:?}"
        );
        // and none of them is redundant: an interior point IS a turn, so it
        // must sit well off the line joining its neighbours
        for w in v.windows(3) {
            let [(t0, p0), (t1, p1), (t2, p2)] = [w[0], w[1], w[2]];
            let line = p0 + (p2 - p0) * (t1 - t0) / (t2 - t0);
            assert!(
                (p1 - line).abs() > VERTEX_EPS,
                "a point at {t1} ms is within eps of the line: it should have been thinned away"
            );
            assert!(
                (t2 - t1 - 250.0).abs() < 1e-6,
                "a triangle's corners are half a period apart, got {}",
                t2 - t1
            );
        }
    }

    #[test]
    fn a_solved_curve_stays_within_eps_of_the_wave() {
        // the arc a sine should keep: thinned, but never off by more than eps
        for id in ["sine", "rounded_tri", "single_peak", "pulse"] {
            let s = spec(id, 1.0, &[]);
            let w = load().get(id).expect("wave").clone();
            let eff = w.resolve(&s.params);
            let v = span_vertices(&s, 3_000.0, VERTEX_EPS);
            assert!(v.len() > 2, "{id} solved to {} points", v.len());
            // walk the solved polyline and sample the truth against it
            let mut worst: f64 = 0.0;
            let mut k = 0;
            for i in 0..=3000 {
                let t = i as f64;
                while k + 2 < v.len() && v[k + 1].0 < t {
                    k += 1;
                }
                let (ta, pa) = v[k];
                let (tb, pb) = v[(k + 1).min(v.len() - 1)];
                let line = if tb > ta {
                    pa + (pb - pa) * (t - ta) / (tb - ta)
                } else {
                    pa
                };
                worst = worst.max((line - s.level_at(&w, t / 1000.0, &eff)).abs());
            }
            assert!(
                worst <= VERTEX_EPS + 1e-6,
                "{id} strays {worst} units off the wave"
            );
        }
    }

    /// The apex knob moves the triangle's point, which is the whole of
    /// requirement four for that wave.
    #[test]
    fn the_apex_knob_moves_the_triangles_point() {
        // the returned clock is milliseconds into the span
        let v = span_vertices(&spec("triangle", 1.0, &[("peak", 0.25)]), 1_000.0, VERTEX_EPS);
        let peak = v
            .iter()
            .find(|p| (p.1 - 100.0).abs() < 1e-9)
            .expect("the triangle has a peak");
        assert!(
            (peak.0 - 250.0).abs() < 1e-6,
            "an apex of 0.25 belongs a quarter of the way in, got {}",
            peak.0
        );
        // and the knob is a share of the cycle, so it holds at any cadence
        let v = span_vertices(&spec("triangle", 2.0, &[("peak", 0.75)]), 2_000.0, VERTEX_EPS);
        let peak = v
            .iter()
            .find(|p| (p.1 - 100.0).abs() < 1e-9)
            .expect("the triangle has a peak");
        assert!((peak.0 - 1_500.0).abs() < 1e-6, "got {}", peak.0);
    }

    /// `round` was a slider nothing read: the shape was two half-parabolas
    /// whatever it said. Now it is the knee of a six-leg corner, and its two
    /// ends are the two shapes the wave's own hint promises.
    #[test]
    fn the_round_slider_actually_changes_the_shape() {
        let flat = span_vertices(&spec("rounded_tri", 1.0, &[("round", 0.0)]), 1_000.0, VERTEX_EPS);
        let tri = span_vertices(&spec("triangle", 1.0, &[]), 1_000.0, VERTEX_EPS);
        assert_eq!(
            flat.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>(),
            tri.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>(),
            "a corner radius of zero IS the triangle, or the knob is a lie"
        );
        let soft = span_vertices(
            &spec("rounded_tri", 1.0, &[("round", 25.0)]),
            1_000.0,
            VERTEX_EPS,
        );
        assert_ne!(
            soft.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>(),
            tri.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>(),
            "the slider changed but the shape did not"
        );
        // the knee sits at half the radius, and the flat run in between is
        // what keeps the wave's promise of an even middle
        assert!(
            soft.iter().any(|p| (p.1 - 12.5).abs() < 1e-9),
            "a radius of 25% puts its knee at 12.5 units, got {:?}",
            soft.iter().map(|p| p.1).collect::<Vec<_>>()
        );
        assert!(
            soft.iter().any(|p| (p.1 - 87.5).abs() < 1e-9),
            "and the far knee at 87.5"
        );
    }
}
