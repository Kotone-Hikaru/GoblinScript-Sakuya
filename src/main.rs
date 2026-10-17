//! GoblinScript -- video in, funscript out. Made by real goblins.
//!
//! A frozen V-JEPA 2.1 encoder and a trained head, wrapped so that nothing
//! about either leaks into the interface: you hand a goblin a video, it hands
//! you a funscript. Needs `ffmpeg` and `ffprobe` on PATH; everything else --
//! the goblin, the PCA basis, the deploy constants -- is baked into the binary.

mod artifacts;
mod autocrop;
mod bios;
mod boundaries;
mod bundle;
mod cache;
mod cancel;
mod chrome;
mod cropedit;
mod dl;
mod encode;
mod errlog;
mod exposure;
mod mascot;
mod prefetch;
mod ffmpeg;
mod heads;
mod lang;
mod review;
mod serve;
mod settings;
mod sound;
mod style;
mod theme;
mod throttle;
mod tui;
mod viz;
mod vr;
mod wave;

use anyhow::{Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser};
use console::style;
use indicatif::ProgressBar;
use theme::{con, theme};
use std::collections::{BTreeSet, HashMap};
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Stamped into every funscript's `metadata.author`, so a written file carries
/// the tool and build that made it -- "GoblinScript v0.2.0".
const AUTHOR: &str = concat!("GoblinScript v", env!("CARGO_PKG_VERSION"));

/// Stamped into every funscript's `metadata.tags`, marking provenance so a
/// player or library can tell a machine-drafted script from a hand-authored
/// one at a glance -- the author string names the tool, these name the class.
const TAGS: &[&str] = &["ai-generated"];

// Status chips, in the same bracketed shape the startup report uses -- one
// vocabulary from POST to the last written script. Plain ASCII, so they need no
// terminal-capability fallback and survive a redirect into a log file intact.
const TICK: &str = "[ OK ]";
const ARROW: &str = ">>";
const CROSS: &str = "[FAIL]";
const SKIP: &str = "[SKIP]";
use std::path::{Path, PathBuf};
use std::time::Instant;

use bundle::Bundle;
use cache::{Cache, CacheMeta};

#[derive(Parser, Debug, Clone)]
#[command(name = "goblinscript", version, about = "Video in, funscript out. Made by real goblins.")]
struct Cli {
    /// Video file(s) to draft, a FOLDER (every video in it, sub-folders not
    /// searched), or http(s) link(s) to fetch and then draft (that needs yt-dlp
    /// on PATH -- we do not install it for you). With none given (e.g. the exe
    /// was double-clicked), an interactive picker opens instead.
    #[arg(value_name = "VIDEO|DIR|URL")]
    videos: Vec<PathBuf>,

    /// Where the .funscript goes (default: next to the video).
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,

    /// Where a video fetched from a link is saved (default: the working
    /// directory, or the folder the picker is browsing). The script then lands
    /// beside it as usual, which is the point of keeping the two together.
    /// Needs yt-dlp on PATH; without it, links are simply not offered.
    #[arg(long, value_name = "DIR")]
    dl_dir: Option<PathBuf>,

    /// Anything after `--` goes to yt-dlp verbatim, e.g.
    /// `goblinscript "<link>" -- --cookies-from-browser firefox`. Ours are
    /// passed first, so where yt-dlp takes the last word these win. They also
    /// reach the picker, so a run started with no videos can paste links under
    /// the same options.
    #[arg(last = true, value_name = "YT-DLP ARG", allow_hyphen_values = true)]
    dl_args: Vec<String>,

    /// Where the working files live DURING a draft (default: `cache/` next to
    /// the executable). Ephemeral: a video's working folder is removed once
    /// its funscript is written -- but always kept after a crash or Ctrl-C,
    /// so the next run resumes instead of starting over.
    #[arg(long, value_name = "DIR")]
    cache: Option<PathBuf>,

    /// Keep the working files (transcode + latents, ~1 MB per second of
    /// video) after a successful draft too: re-drafting the same video, e.g.
    /// with a future version, then skips the heavy work.
    #[arg(long)]
    keep_cache: bool,

    /// Overwrite an existing .funscript. Without this a video whose script
    /// already exists is SKIPPED -- the file there may be hand-made or
    /// hand-corrected, which is not ours to replace.
    #[arg(long)]
    force: bool,

    /// How confident the model must be to park a hold at its level: cautious
    /// parks only sure ones (the safer choice on unfamiliar material), eager
    /// parks more.
    #[arg(long, value_enum, default_value_t = style::Dwells::Normal)]
    dwells: style::Dwells,

    /// How aggressively near-still passages hold flat instead of minting
    /// tiny strokes: low keeps micro-motion, high kills phantom strokes
    /// harder.
    #[arg(long, value_enum, default_value_t = style::Stillness::Normal)]
    stillness: style::Stillness,

    /// Make a small variation of the same script. The goblins select how deep
    /// each stroke goes, and a different seed makes them select differently.
    /// The change is small: on one video, stroke depths moved by 1 of 100 on
    /// average, and 99% of the changes of direction stayed at the same time.
    /// Each seed gives the same result every time you use it.
    #[arg(long, value_name = "N")]
    env_seed: Option<u64>,

    /// EXPERT: where the dwell lock's confidence ramp starts, as a raw
    /// number (0..1), overriding --dwells. This is the numeric knob behind the
    /// review page's expert mode.
    #[arg(long, value_name = "P")]
    dwell_ramp: Option<f64>,

    /// EXPERT: the stillness velocity floor as a raw number (pos-units/s),
    /// overriding --stillness.
    #[arg(long, value_name = "EPS")]
    still_eps: Option<f64>,

    /// Position synthesis: composed (the production decode -- level +
    /// envelope + rails + locks) or level (strokes around the predicted
    /// level with the model's own travel -- steadier speed, less reach)
    #[arg(long, value_enum, default_value_t = style::Style::Composed)]
    style: style::Style,

    /// Keep stroke depth consistent: pull each reversal's reach toward the
    /// running median of nearby same-direction reversals, so tops and bottoms
    /// settle onto steady levels. Trades the model's velocity-driven amplitude
    /// for a more uniform depth; off = the model's own reach.
    #[arg(long, value_enum, default_value_t = style::DepthUniformity::Off)]
    depth_uniformity: style::DepthUniformity,

    /// EXPERT: depth-uniformity blend as a raw number (0..1, overrides the
    /// preset's) -- 0 leaves the reach, 1 snaps it to the running median.
    #[arg(long, value_name = "DOSE")]
    depth_dose: Option<f64>,

    /// EXPERT: depth-uniformity window in seconds (overrides the preset's) --
    /// how far the running median reaches; short = local, long = whole-passage.
    #[arg(long, value_name = "SEC")]
    depth_window: Option<f64>,

    /// Stroke amplitude scale, 0.5..4.0 (the review page's unlock boxes raise
    /// this to 10 or 100); 1.0 = the model's own amplitude. Multiplies every
    /// position's distance from mid-range, so 1.5 writes strokes half again as
    /// far from the middle.
    #[arg(long, default_value_t = 1.0)]
    intensity: f64,

    /// Confine positions to a span, e.g. 10-90 for a device that should
    /// never hit its physical ends.
    #[arg(long, value_name = "LO-HI", default_value = "0-100", value_parser = parse_range)]
    range: (f64, f64),

    /// Cap the whole-device GPU load while a draft runs, so the machine stays
    /// usable (a browser video keeps playing). A background loop samples the
    /// GPU every --throttle-interval seconds and stretches the gap between
    /// encoder windows until the load settles at --throttle-gpu-target.
    /// Presets: gentle (target 85, gentle corrections), balanced (target 90,
    /// the shipped default), aggressive (target 95, holds speed), custom
    /// (start from balanced and set any --throttle-* knob below). Off by
    /// default: a plain run is byte-identical to a build without this flag.
    #[arg(long, value_name = "PRESET")]
    throttle: Option<String>,

    /// Do not throttle. The default, stated so a script can be explicit.
    #[arg(long)]
    no_throttle: bool,

    /// Fraction of the GPU load to hold, as a percentage (--throttle target).
    /// Over the target the gaps between windows lengthen; under it they
    /// shorten. 90 leaves a browser video ~10% of headroom.
    #[arg(long, value_name = "PCT")]
    throttle_gpu_target: Option<f32>,

    /// Seconds between GPU samples. NVML reads cost ~2 us, so this is a
    /// responsiveness choice, not a cost one. The driver's own utilization is
    /// a ~1 s sliding average, and the encoder's load is a pulse train whose
    /// period is ~3 s, so a single sample is mostly the phase of that train;
    /// this is why the loop DECIDES on --throttle-avg-window, not on one
    /// sample. Sampling fast is still right -- the average wants points.
    #[arg(long, value_name = "SEC")]
    throttle_interval: Option<f32>,

    /// Seconds of samples averaged before ONE duty correction. Must exceed the
    /// encoder's pulse period (~3 s here) or the loop regulates its own phase:
    /// see the module docs. 9 s is three periods and the shipped default.
    #[arg(long, value_name = "SEC")]
    throttle_avg_window: Option<f32>,

    /// Loop gain: duty moves this many points per point of error. Higher
    /// settles faster and hunts more.
    #[arg(long, value_name = "K")]
    throttle_kp: Option<f32>,

    /// Error inside +-this many points is ignored, so the machine's own
    /// jitter never moves the loop.
    #[arg(long, value_name = "PCT")]
    throttle_deadzone: Option<f32>,

    /// Duty floor, as a percentage: the throttle never starves the encode
    /// below this, so a draft always finishes.
    #[arg(long, value_name = "PCT")]
    throttle_duty_min: Option<f32>,

    /// Duty the loop arms at, instead of 100. Starting at 100 spends the first
    /// average window at full pelt no matter what, and that window is the one
    /// the user is most likely to be watching; the loop walks up from here if
    /// the machine is idle.
    #[arg(long, value_name = "PCT")]
    throttle_duty_start: Option<f32>,

    /// Cap on a single upward duty move per tick, in points. Raising duty is
    /// the cautious direction (it re-tightens the throttle), so it moves
    /// slowly; see --throttle-fall-limit for the other side.
    #[arg(long, value_name = "PCT")]
    throttle_rise_limit: Option<f32>,

    /// Cap on a single downward duty move per tick, in points. Without it a
    /// large error relative to a low --throttle-gpu-target floors the loop in
    /// one move and it then has to climb back at --throttle-rise-limit per
    /// tick.
    #[arg(long, value_name = "PCT")]
    throttle_fall_limit: Option<f32>,

    /// Seconds a duty change must hold before the loop may move the other
    /// way. Kills slow hunting that smoothing alone would ride.
    #[arg(long, value_name = "SEC")]
    throttle_dwell: Option<f32>,

    /// RETIRED, no effect. A single over-target sample can no longer move the
    /// loop: at this sampling rate the encoder's own pulse (~606 ms of load
    /// every ~3 s) and a real external burst are the same shape, so the branch
    /// that acted on one sample acted on the loop's own phase and floor-locked
    /// it. Accepted so an older command line still runs.
    #[arg(long, value_name = "PCT")]
    throttle_spike_margin: Option<f32>,

    /// RETIRED with --throttle-spike-margin. No effect.
    #[arg(long, value_name = "PCT")]
    throttle_spike_cut_max: Option<f32>,

    /// RETIRED with --throttle-spike-margin. No effect.
    #[arg(long, value_name = "SEC")]
    throttle_spike_hold: Option<f32>,

    /// OPTIONAL second clamp: hold the SCRIPT itself to this duty ceiling,
    /// independent of the machine target. The plain loop controls the whole
    /// device and lets the script use whatever is free; this bounds the
    /// script even when the device is idle. Off unless set.
    #[arg(long, value_name = "PCT")]
    throttle_script_max: Option<f32>,

    /// SUPERSEDED by --throttle-avg-window and has no effect: the loop now
    /// averages over a span of TIME (which is what the plant's pulse period
    /// is), not over a count of samples. Accepted so an older command line
    /// still runs.
    #[arg(long, value_name = "N")]
    throttle_smooth: Option<usize>,

    /// Seconds at the start with no correction, so encoder warmup is not
    /// throttled.
    #[arg(long, value_name = "SEC")]
    throttle_grace: Option<f32>,

    /// Cap position speed (pos-units/s). The default is the speed a funscript
    /// player can actually deliver: a faster transition is depth the device
    /// never performs, so the endpoint is pulled in until the move is
    /// playable (timing never moves). Raise it for hardware that outruns it,
    /// or pass 0 to write the model's reach uncapped.
    #[arg(long, default_value_t = style::MAX_POS_RATE)]
    max_speed: f64,

    /// Smooth the jolt at a shot cut (pos-units/s, 0 = off). Styling runs
    /// per shot, so each cut gets two points about one frame apart and the
    /// device must make the whole level change of the cut in that time. Above
    /// this speed those points go -- the one before the cut first, then the
    /// one after it if the move is still too fast -- so the move runs between
    /// real changes of direction across the cut, which is what a human writer
    /// does. If that move is also too fast, its depth decreases until its
    /// speed obeys the limit. Each shot always keeps a point. No time moves,
    /// and no move across a cut is faster than this limit.
    #[arg(long, default_value_t = style::CUT_EASE)]
    cut_ease: f64,

    /// Replace passages the model calls motion-free with a background
    /// filler rhythm. off = none; subtle = sparse and gentle (20 s gaps,
    /// 30/min, +-10); steady = the standard rhythm (10 s, 40/min, +-15);
    /// bursts = stroke bursts with rests (10 s, 50/min, +-18). Garbage
    /// output inside a gap (weak short wiggles) is replaced too; the
    /// rhythm sits at the model's own level and cross-fades at the seams.
    /// Every number below overrides its preset value.
    #[arg(long, value_enum, default_value_t = style::FillerPreset::Off)]
    filler: style::FillerPreset,

    /// EXPERT: still seconds (summed across a gap's kept interruptions)
    /// before filler starts (overrides the preset's; setting this with
    /// --filler off turns filler on at this gap).
    #[arg(long, value_name = "SEC")]
    filler_gap: Option<f64>,

    /// Motion inside a gap is kept (interrupting the rhythm) only if it
    /// sustains --filler-real-v for at least this much time in total;
    /// anything with less evidence is garbage and is replaced.
    #[arg(long, value_name = "SEC", default_value_t = 1.0)]
    filler_min_real: f64,

    /// EXPERT: the confidence bar (pos-units/s of smoothed predicted
    /// velocity) motion inside a gap must sustain to count as real.
    #[arg(long, value_name = "V", default_value_t = 45.0)]
    filler_real_v: f64,

    /// How much the model's output shapes a filled gap (0..1): at 1
    /// islands survive at the evidence bar and the rhythm rides the
    /// smoothed level; at 0 the rhythm runs pure -- no sub-bridge island
    /// survives and the base is one constant anchor per gap (its
    /// entry/exit levels). Islands at least --filler-max-bridge long
    /// always survive, at every setting.
    #[arg(long, value_name = "W", default_value_t = 1.0)]
    filler_model_w: f64,

    /// EXPERT: only motion SHORTER than this many seconds can be replaced
    /// as garbage -- a longer passage always survives, however slow, and
    /// ends the gap; kept motion shorter than this only pauses the rhythm
    /// (the gap's still time keeps counting toward --filler-gap).
    #[arg(long, value_name = "SEC", default_value_t = 5.0)]
    filler_max_bridge: f64,

    /// Filler rhythm rate, strokes per minute (overrides the preset's).
    #[arg(long, value_name = "SPM")]
    filler_rate: Option<f64>,

    /// Filler rhythm amplitude, position units above and below the held
    /// level (overrides the preset's).
    #[arg(long, value_name = "UNITS")]
    filler_amp: Option<f64>,

    /// EXPERT: seam cross-fade seconds at each end of a filled gap;
    /// 0 = one stroke leg (60/rate).
    #[arg(long, value_name = "SEC", default_value_t = 0.0)]
    filler_ramp: f64,

    /// EXPERT: fractional amplitude sway of the rhythm (0 = metronome),
    /// deterministic per gap.
    #[arg(long, value_name = "FRAC", default_value_t = 0.15)]
    filler_sway: f64,

    /// EXPERT: the sway's period, seconds.
    #[arg(long, value_name = "SEC", default_value_t = 16.0)]
    filler_sway_s: f64,

    /// Rhythm shape (overrides the preset's): a continuous triangle, or
    /// bursts of --filler-burst strokes with --filler-rest between them.
    #[arg(long, value_enum, value_name = "SHAPE")]
    filler_pattern: Option<style::FillerPattern>,

    /// Strokes per burst (pattern burst).
    #[arg(long, value_name = "N")]
    filler_burst: Option<usize>,

    /// Seconds of rest between bursts (pattern burst).
    #[arg(long, value_name = "SEC")]
    filler_rest: Option<f64>,

    /// After drafting, stay open and review in the browser: the video plays
    /// with the scripted motion overlaid, and a parameter change rewrites the
    /// script in milliseconds (the picker does this automatically).
    #[arg(long)]
    review: bool,

    /// Run as an API server for another program -- a funscript editor, a
    /// script -- instead of drafting a list of videos. Binds to localhost only
    /// and prints one JSON line with the port and the access token, then
    /// answers requests until its standard input closes or a client asks it
    /// to quit. Every drafting option on this command line is the
    /// configuration each session drafts under. `goblinscript --serve --help`
    /// is this text; the routes are documented in README.md.
    #[arg(long)]
    serve: bool,

    /// The port `--serve` listens on; 0 picks a free one, which the handshake
    /// line then names.
    #[arg(long, value_name = "PORT", default_value_t = 0)]
    port: u16,

    /// Aim VR sources before drafting them, in a browser page: point one flat
    /// viewport at the action (and optionally trim the part worth drafting),
    /// and the goblins see ordinary 2D footage from there on.
    ///
    /// Normally this needs no flag -- a VR-looking source is detected and the
    /// page opens by itself. Pass this to force the page open on a source the
    /// detector called flat (or to aim one on a run with no terminal to ask
    /// in); pass `--no-vr` to skip the step entirely and draft the raw frames.
    #[arg(long)]
    vr: bool,

    /// Never open the VR prep page: draft every source exactly as it comes.
    /// A VR source drafted this way produces nonsense -- the encoder has never
    /// seen an equirectangular frame -- so this is for a false positive on the
    /// detector, or a batch that must not stop to ask.
    #[arg(long, conflicts_with = "vr")]
    no_vr: bool,

    /// Zoom the goblins onto the action before encoding: a sparse probe runs
    /// the clip's attention (the bundle's own mask net) and picks one
    /// conservative crop rect per shot -- rects change only at detected cuts,
    /// the crop rides the decode chain (never a re-encode, so the clock
    /// cannot move), and a clip whose attention wants the whole frame is left
    /// uncropped. Helps wide framing (VR flats, wide shots); measured free on
    /// tight footage. ON by default -- pass this only to demand it of a
    /// bundle that may not carry a mask net, where it becomes an error
    /// instead of a skip.
    #[arg(long, conflicts_with = "no_autocrop")]
    autocrop: bool,

    /// Draft from the whole frame: skip the auto-crop probe entirely. For
    /// footage already framed on the action, where the probe is time spent to
    /// be told what it started with.
    #[arg(long)]
    no_autocrop: bool,

    /// Check the crop rects in a browser before the goblins read the video.
    /// The page draws each shot's rect on the picture and lets you drag it;
    /// what you draw is kept for that video from then on, through a new
    /// goblin and a new release. ON by default when you are at a terminal --
    /// pass this only to demand it where it would otherwise be skipped, which
    /// is a piped run with nobody to answer it.
    #[arg(long, conflicts_with_all = ["no_autocrop", "no_crop_edit"])]
    crop_edit: bool,

    /// Take the crop the goblins picked, without showing it to you first.
    #[arg(long)]
    no_crop_edit: bool,

    /// Draft the picture as-is: skip the exposure correction (one gamma that
    /// maps a clip outside the corpus's luma band back into it; a clip
    /// inside the band is never touched either way).
    #[arg(long)]
    no_exposure: bool,

    /// Who decodes the source while it is being normalized. The goblins write
    /// the same normalized copy either way, byte for byte -- this only changes
    /// how long the normalize stage takes. The card removes the decode but
    /// adds a trip across the bus, so it is quicker on a video that decodes
    /// expensively (an 8K panorama: approximately 1.6 times as quick) and
    /// slower on one that does not (4K H.264: approximately 3 times slower).
    /// Because that is a property of your video, `auto` times both ways on the
    /// video itself, which reads approximately 100 frames and adds one or two
    /// seconds. Any AMD, Intel or NVIDIA card can do it: the goblins ask the
    /// operating system, not the make of the card. If no decoder can take the
    /// video, or if the number of bits a colour cannot be read, the goblins
    /// decode on the processor.
    #[arg(long, value_enum, default_value_t = ffmpeg::HwAccel::Auto, value_name = "WHO")]
    hwaccel: ffmpeg::HwAccel,

    /// Colour scheme for every screen the goblins draw: green phosphor, an
    /// amber monitor, CGA cyan/magenta, or plain white phosphor. The picker
    /// cycles it with T and remembers the choice.
    #[arg(long, value_enum, value_name = "NAME")]
    theme: Option<theme::Palette>,

    /// The language the goblins speak, as a tag naming a file in `languages/`
    /// beside the exe -- `--lang zh-CN` reads `languages/zh-CN.json`. Without
    /// this the system's own language is used when a file for it is installed,
    /// and English otherwise. The picker cycles it with G and remembers the
    /// choice. To add a language, copy `languages/en-US.json`, translate the
    /// right-hand side of every line, and save it under its own tag.
    #[arg(long, value_name = "TAG")]
    lang: Option<String>,

    /// Silence the chimes and blips (the picker toggles this with M).
    #[arg(long)]
    mute: bool,

    /// Skip the startup report -- for scripted runs that want the drafting log
    /// and nothing above it.
    #[arg(long)]
    quiet: bool,

    /// Play the background music while the app is open. Needed on a
    /// named-video run, which is silent by default; the picker plays it
    /// already and toggles it with M (that choice is remembered, and applies
    /// to the picker only). Music always stops for the duration of a review.
    #[arg(long)]
    music: bool,

    /// How loud the music sits under the work. The picker cycles it with V and
    /// remembers the choice.
    #[arg(long, value_enum, value_name = "LEVEL")]
    volume: Option<sound::Volume>,

    /// DEVELOPMENT: run everything on the CPU instead of the graphics card.
    /// Hidden from `--help` because the shipped bundle has no CPU path at all:
    /// its attention runs in a packed layout with no CPU kernel, so this flag
    /// refuses that bundle up front (a `--cpu-attn` re-export is the developer
    /// escape hatch). Even on a CPU-capable bundle the goblins are ~74x slower,
    /// so this is a bisection tool for a GPU that miscomputes, never a way to
    /// run without a card.
    #[arg(long, hide = true)]
    cpu: bool,

    /// Load the ONNX bundle from a directory instead of the baked-in one.
    #[arg(long, value_name = "DIR")]
    bundle: Option<PathBuf>,

    /// Draft with this model instead of the bundle's default. A bundle
    /// carries one perception and several models, each named for the
    /// release that shipped it; `--dump-bundle` writes them out under
    /// packs/, and the API's /health names them.
    #[arg(long, value_name = "NAME")]
    model: Option<String>,

    /// Draft every named model over ONE encode of each video, comma
    /// separated. The chosen model (`--model`, else the default) writes
    /// <stem>.funscript as always, and every other one
    /// <stem>.<model>.funscript beside it. A review has no use for it: the
    /// page offers every model of the bundle, prepared while drafting.
    #[arg(long, value_name = "A,B", value_delimiter = ',', conflicts_with = "review")]
    models: Vec<String>,

    /// The name a second model's draft carries between the stem and the
    /// extension; set by the batch for every model after the chosen one,
    /// never from the command line.
    #[arg(skip)]
    out_suffix: Option<String>,

    /// Write the goblins out to a folder and stop: the model baked into this
    /// exe, as the files it was packed from. A GoblinScript built from source
    /// has no goblins in it -- hand it this folder with --bundle and it drafts
    /// exactly as this one does.
    #[arg(long, value_name = "DIR")]
    dump_bundle: Option<PathBuf>,

    /// Time the encoder and exit (development).
    #[arg(long, hide = true)]
    bench: bool,

    /// DEVELOPMENT: print what every soundtrack file measures and how the
    /// levelling will treat it, then exit. Run this after adding music: the
    /// median it reports is what `TARGET_LOUDNESS` should be.
    #[arg(long, hide = true)]
    music_levels: bool,

    /// DEVELOPMENT: draft strictly one stage at a time, without normalizing the
    /// next video in this one's GPU-bound shadow. The measurement baseline for
    /// that overlap, and the switch to flip if it is ever suspected of
    /// starving a stage for CPU.
    #[arg(long, hide = true)]
    no_prefetch: bool,

    /// DEVELOPMENT: treat the source as already normalized and skip the
    /// transcode. Only correct for a clip that already IS 480p/30/crf23 -- it
    /// exists so parity runs read the exact same pixels the Python pipeline
    /// did, instead of a re-encode of them.
    #[arg(long, hide = true)]
    no_transcode: bool,

    /// DEVELOPMENT: bound the draft to the first N minutes.
    #[arg(long, hide = true, value_name = "MIN")]
    minutes: Option<f64>,

    /// DEVELOPMENT: take the auto-crop's attention placement as-is, without
    /// the confidence-guided placement search -- the A/B control for what
    /// the search buys. The shipped search is margin-gated and skips
    /// high-confidence shots (the A/B's damaging moves rode small
    /// margins). The two arms share a plan cache key, so
    /// never mix them under `--keep-cache`.
    #[arg(long, hide = true)]
    no_crop_search: bool,

    /// DEVELOPMENT: don't launch the browser for the review page (the server
    /// still runs and prints its URL) -- for driving the endpoints headless.
    #[arg(long, hide = true)]
    no_browser: bool,

    /// Open the review page in our own browser window, aimed at the discrete
    /// GPU. Off unless asked for, and remembered once asked: it costs a private
    /// browser profile (a window carrying none of your bookmarks or logins),
    /// and only helps a machine whose browser would otherwise land on the
    /// integrated chip.
    #[arg(long)]
    gpu_browser: bool,

    /// DEVELOPMENT: run the VR prep step on the named videos and exit, without
    /// loading the model. Aiming needs no GPU, so this is the way to drive the
    /// prep endpoints (and to check a detection) without occupying one.
    #[arg(long, hide = true)]
    vr_only: bool,

    /// DEVELOPMENT: skip every stage before the head and run it over a latent
    /// file that already exists -- raw int8 rows of dim*grid*grid bytes in
    /// video order, the layout the encoder writes to `latents.i8`. Writes the
    /// tracks to --dump-tracks and exits.
    #[arg(long, hide = true, value_name = "FILE")]
    from_latents: Option<PathBuf>,

    /// DEVELOPMENT: the cut times --from-latents runs against, as the
    /// `cuts_ms` array of a boundaries JSON. Without it the timeline is one
    /// uncut shot, which is not what the Python side scored.
    #[arg(long, hide = true, value_name = "FILE")]
    cuts_json: Option<PathBuf>,

    /// DEVELOPMENT: where --from-latents writes its tracks.
    #[arg(long, hide = true, value_name = "FILE", default_value = "tracks.json")]
    dump_tracks: PathBuf,
}

/// `--range 10-90` -> `(10.0, 90.0)`, validated inside 0..100.
fn parse_range(s: &str) -> Result<(f64, f64), String> {
    let (lo, hi) = s
        .split_once('-')
        .ok_or_else(|| format!("expected LO-HI, e.g. 10-90, got {s:?}"))?;
    let (lo, hi): (f64, f64) = (
        lo.trim().parse().map_err(|_| format!("bad number {lo:?}"))?,
        hi.trim().parse().map_err(|_| format!("bad number {hi:?}"))?,
    );
    if !(0.0..=100.0).contains(&lo) || !(0.0..=100.0).contains(&hi) || lo >= hi {
        return Err(format!("need 0 <= LO < HI <= 100, got {lo}-{hi}"));
    }
    Ok((lo, hi))
}

fn load_bundle(bundle: Option<&Path>) -> Result<Bundle> {
    // Init the runtime right before the session that needs it, so in picker mode
    // this whole (slow) call runs on the loader thread under the startup intro.
    if !ort::init().commit() {
        anyhow::bail!("failed to initialize ONNX Runtime");
    }
    match bundle {
        Some(dir) => Bundle::from_dir(dir),
        None => {
            #[cfg(feature = "embed")]
            {
                Bundle::embedded()
            }
            #[cfg(not(feature = "embed"))]
            {
                anyhow::bail!(
                    "this build has no bundle baked in -- pass --bundle <DIR> \
                     (build with --features embed to embed one)"
                )
            }
        }
    }
}

// --- the live status line ---------------------------------------------------
//
// The processing UI draws exactly ONE line in place. That is the whole trick to
// being resize-proof: a single line is rewritten with a carriage return and
// truncated to the terminal's *current* width every frame, so it can never wrap
// -- and a line that never wraps carries no multi-row bookkeeping for a resize
// to invalidate. (indicatif's MultiProgress orphans lines on resize because it
// erases its block by moving the cursor up a line count captured at the old
// width; there is no resize listener to correct it. One `\r` line sidesteps the
// entire problem.) Completed stages print as ordinary scrollback above the live
// line; the goblin animates as the line's leading glyph. All stdout writes are
// funnelled through the single render thread, so nothing interleaves.

/// The live block's frame time. Nothing on screen moves faster than this: the
/// goblin's pick, the progress bar, and the viewport's own repaint all ride it.
const FRAME_MS: u64 = 120;

/// The pick swings up -> down -> back, and the goblin blinks once a cycle;
/// paced in MILLISECONDS rather than frames, so it keeps its own gait whatever
/// the frame rate is. Every frame is the same display width, so the line's
/// layout never jitters.
const GOBLIN_MS: u64 = 120;
const GOBLIN: [&str; 8] = [
    "(o_o)/", "(o_o)|", "(o_o)\\", "(o_o)|",
    "(o_o)/", "(-_-)|", "(o_o)\\", "(o_o)|",
];

/// The active stage the render thread paints. `pb` is a HIDDEN indicatif bar --
/// never drawn, used only as a position/speed/ETA estimator so `boundaries`,
/// `encode` and `heads` keep their `&ProgressBar` signatures and just call
/// `set_position`. `units_per_sec` converts its rate into "x realtime".
struct Stage {
    label: String,
    /// The measurable half of the stage -- `None` while it is still setting up.
    /// Getting a graph onto the GPU takes tens of seconds (DirectML compiles
    /// it), and there is no progress to report over that: a 0% bar is a bar
    /// that looks hung, so `note` is painted in its place instead.
    pb: Option<ProgressBar>,
    /// Meaningful only alongside `pb`.
    units_per_sec: f64,
    /// What the setup is waiting on, in place of the bar.
    note: String,
}

struct LiveState {
    stage: Option<Stage>,
    /// Permanent lines waiting to be emitted above the live line.
    pending: Vec<String>,
    /// When the display opened -- the goblin's animation is paced off the wall
    /// clock (`GOBLIN_MS`) rather than off a frame counter, so its gait is a
    /// property of the mascot and not of how often the block is repainted.
    opened: Option<Instant>,
    /// Visible width of every line of the live BLOCK as last drawn, top row
    /// first: the viewport's rows (`viz.rs`), then the status line. If the
    /// terminal is now narrower than one of them, that line has reflowed into
    /// several physical rows and the erase must climb over all of them -- which
    /// is what keeps a shrink from leaving orphans. Empty means nothing is
    /// drawn; one entry is the status line alone, which is the whole display
    /// until the encoder has produced a latent row.
    last_vis: Vec<usize>,
    /// Terminal width the block was last drawn at. A frame that finds it
    /// unchanged can overwrite the block in place; a frame that does not has
    /// lines reflowed by the resize under it and has to wipe before drawing.
    last_width: usize,
}

/// Raw mode for as long as the guard lives, restored on every exit path --
/// a panic that leaves the console raw eats the user's next shell.
pub struct RawMode(bool);

impl RawMode {
    pub fn enable() -> RawMode {
        RawMode(ratatui::crossterm::terminal::enable_raw_mode().is_ok())
    }
    /// Are keys actually reaching us? False on a console that refused raw mode,
    /// where the render thread must not try to read events.
    pub fn on(&self) -> bool {
        self.0
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        if self.0 {
            let _ = ratatui::crossterm::terminal::disable_raw_mode();
        }
    }
}

/// Where a draft's stage events go besides the console. The API server
/// (`serve`) hands one to `draft` so a client can read the stage in flight
/// and the ones that cleared; a console run passes none. `stage` hands over
/// the hidden bar the stage drives, so the listener reads progress off the
/// same counter the live line does. `page` is a stage that waits on a
/// person: the browser page at `url` is open, and the stage ends when its
/// Done button is pressed.
pub trait StageSink: Send + Sync {
    fn setup(&self, key: &str);
    fn stage(&self, key: &str, pb: &ProgressBar);
    fn page(&self, key: &str, url: &str);
    fn done(&self, key: &str, info: &str);
}

/// A single-line, resize-proof progress display. On a non-terminal (piped/logged)
/// it degrades to plain permanent lines and draws no live line at all.
struct Live {
    state: Arc<Mutex<LiveState>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    tty: bool,
    sink: Option<Arc<dyn StageSink>>,
}

impl Live {
    fn new(sink: Option<Arc<dyn StageSink>>) -> Live {
        // A draft opens with no stage in flight. The marker is otherwise left
        // set by a stage that was interrupted rather than finished, and the
        // next failure would inherit it.
        errlog::stage(None);
        let tty = std::io::stdout().is_terminal();
        let state = Arc::new(Mutex::new(LiveState {
            stage: None,
            pending: Vec::new(),
            opened: None,
            last_vis: Vec::new(),
            last_width: 0,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = if tty {
            // Raw mode for the length of the draft: it is what lets the render
            // thread read M/V/N while the goblins work, instead of the console
            // holding every keystroke back until Enter. It also means Ctrl-C
            // stops being a signal and starts being a key -- `render_loop`
            // handles it, and the guard restores the console on every exit
            // path, panics included.
            let raw = RawMode::enable();
            let state = state.clone();
            let stop = stop.clone();
            Some(std::thread::spawn(move || {
                let _raw = raw;
                render_loop(&state, &stop)
            }))
        } else {
            None
        };
        Live { state, stop, handle, tty, sink }
    }

    /// Open a stage that cannot be measured yet -- the graph it needs is still
    /// loading. `stage` then replaces it with the bar, under the same label, so
    /// the line reads as one stage settling into gear.
    ///
    /// `key` is the stage's catalog key, not its words: the line is drawn in
    /// the user's language and remembered in English for the failure log, and
    /// taking the key is what stops those two from ever naming different
    /// stages.
    fn setup(&self, key: &str, note: &str) {
        errlog::stage(Some(lang::en(key)));
        if let Some(s) = &self.sink {
            s.setup(key);
        }
        self.state.lock().unwrap().stage = Some(Stage {
            label: lang::t(key).to_string(),
            pb: None,
            units_per_sec: 1.0,
            note: note.to_string(),
        });
    }

    /// Begin a stage with a determinate length. Returns the hidden bar the stage
    /// worker drives via `set_position`.
    ///
    /// Called once the stage can actually make progress: the bar's clock starts
    /// here, so setup time lands in neither the rate nor the ETA.
    fn stage(&self, key: &str, len: u64, units_per_sec: f64) -> ProgressBar {
        errlog::stage(Some(lang::en(key)));
        let pb = ProgressBar::hidden();
        pb.set_length(len.max(1));
        if let Some(s) = &self.sink {
            s.stage(key, &pb);
        }
        self.state.lock().unwrap().stage = Some(Stage {
            label: lang::t(key).to_string(),
            pb: Some(pb.clone()),
            units_per_sec,
            note: String::new(),
        });
        pb
    }

    /// A stage that waits on a person at a browser page. The console prints
    /// the URL itself; this tells the listener, which is how an editor learns
    /// where the page is.
    fn page(&self, key: &str, url: &str) {
        if let Some(s) = &self.sink {
            s.page(key, url);
        }
    }

    /// End the current stage and record its summary line, laid out with the
    /// same dot leader and right-hand chip as the startup report.
    fn done(&self, key: &str, info: &str) {
        // Between stages there is no stage: a failure in the gap belongs to
        // the draft, not to the stage that just cleared.
        errlog::stage(None);
        if let Some(s) = &self.sink {
            s.done(key, info);
        }
        let line = done_line(lang::t(key), info);
        self.state.lock().unwrap().stage = None;
        self.println(line);
        // one blip per cleared stage: audible from across the room, and quiet
        // enough that four of them per video is not a tune
        sound::play_stage();
    }

    /// A permanent line above the live line. On a terminal it is queued for the
    /// render thread (which erases the live line, prints this, then repaints);
    /// piped, there is no live line to protect, so it prints straight through.
    fn println(&self, line: String) {
        if self.tty {
            self.state.lock().unwrap().pending.push(line);
        } else {
            println!("{line}");
        }
    }

    /// Stop the render thread, wipe the live line, and flush anything queued.
    fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let mut st = self.state.lock().unwrap();
        st.stage = None;
        let mut out = std::io::stdout().lock();
        if self.tty {
            let width = console::Term::stdout().size().1.max(20) as usize;
            erase_live(&mut out, &st.last_vis, width);
            st.last_vis.clear();
        }
        for line in st.pending.drain(..) {
            let _ = writeln!(out, "{line}");
        }
        let _ = out.flush();
    }
}

/// Physical rows a block of lines with these visible widths occupies on a
/// terminal this wide -- one per line, plus any extra a narrowed terminal has
/// since wrapped that line into.
fn block_rows(last_vis: &[usize], width: usize) -> usize {
    last_vis
        .iter()
        .map(|v| 1 + v.saturating_sub(1) / width.max(1))
        .sum()
}

/// Put the cursor back on the first row of the live block, ready to overwrite
/// it. Nothing is cleared: a frame that blanks the block before refilling it is
/// a frame the terminal can present half-drawn, which is what a 13-row viewport
/// flickering looks like. Each line erases its own tail as it is written
/// instead (`\x1b[K`), and only a resize or a block that SHRANK needs a wipe.
fn home_live(out: &mut impl Write, last_vis: &[usize], width: usize) {
    let rows = block_rows(last_vis, width);
    if rows == 0 {
        return;
    }
    if rows > 1 {
        let _ = write!(out, "\x1b[{}A", rows - 1);
    }
    let _ = write!(out, "\r");
}

/// Erase the live block outright -- for tearing the display DOWN, where there
/// is no repaint coming and the rows have to go. `\x1b[J` clears from the
/// cursor to the end of the screen, which is safe because the block is always
/// the bottom-most content.
fn erase_live(out: &mut impl Write, last_vis: &[usize], width: usize) {
    if last_vis.is_empty() {
        return;
    }
    home_live(out, last_vis, width);
    let _ = write!(out, "\x1b[J");
}

impl Drop for Live {
    /// If a draft bails out early (any `?`), `finish` never runs -- stop the
    /// render thread here so it can't keep writing to a dead terminal.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
            if self.tty {
                let width = console::Term::stdout().size().1.max(20) as usize;
                let last_vis = self.state.lock().unwrap().last_vis.clone();
                let mut out = std::io::stdout().lock();
                erase_live(&mut out, &last_vis, width);
                let _ = out.flush();
            }
        }
    }
}

/// One rendered frame: put the cursor back on the block's first row, flush any
/// pending permanent lines into scrollback, then overwrite the block -- the
/// viewport's rows, then the status line -- each truncated to the live width so
/// none of them wraps. All stdout writes for the draft funnel through here, so
/// nothing races.
///
/// **Nothing is cleared before it is redrawn.** Every line is written over the
/// one under it and erases only its own tail, so there is no moment where the
/// block is blank -- which is what a 13-row viewport being cleared and refilled
/// eight times a second looks like from across the room. The two cases that
/// still need a wipe are the ones where overwriting cannot be enough: a resize,
/// which has reflowed the rows under us, and a block that got SHORTER, which
/// leaves its old bottom rows standing.
///
/// The block is otherwise redrawn in place, so the log above it never moves. It
/// grows once, when the encoder produces its first latent row and the viewport
/// appears.
fn render_loop(state: &Arc<Mutex<LiveState>>, stop: &Arc<AtomicBool>) {
    let term = console::Term::stdout();
    while !stop.load(Ordering::Relaxed) {
        {
            let mut st = state.lock().unwrap();
            let mut out = std::io::stdout().lock();
            let (screen_h, w) = term.size();
            let width = w.max(20) as usize;
            // 1. back to the top of the block, without blanking it
            let prev_rows = block_rows(&st.last_vis, width);
            let resized = width != st.last_width && !st.last_vis.is_empty();
            home_live(&mut out, &st.last_vis, width);
            if resized {
                let _ = write!(out, "\x1b[J");
            }
            st.last_vis.clear();
            st.last_width = width;
            // 2. permanent lines scroll into history where the block was.
            // CRLF, not LF: the console is in raw mode for the draft, so nothing
            // returns the carriage for us and a bare \n stair-steps the log.
            let n_pending = st.pending.len();
            for line in st.pending.drain(..) {
                let _ = write!(out, "{line}\x1b[0m\x1b[K\r\n");
            }
            // 3. the viewport, above the status line and on the same clock
            for line in viz::render(width, chrome::log_rows(screen_h) as usize) {
                let line = console::truncate_str(&line, width.saturating_sub(1), "");
                let _ = write!(out, "{line}\x1b[0m\x1b[K\r\n");
                st.last_vis.push(measure(&line));
            }
            // 4. repaint the status line
            let t = theme();
            let opened = *st.opened.get_or_insert_with(Instant::now);
            let swing = (opened.elapsed().as_millis() / GOBLIN_MS as u128) as usize;
            let goblin = style(GOBLIN[swing % GOBLIN.len()]).fg(con(t.logo)).bold();
            let body = match &st.stage {
                None => style(t!("console.idle"))
                    .fg(con(t.text))
                    .bold()
                    .to_string(),
                Some(s) => stage_body(s, width),
            };
            let line = format!("  {goblin}  {body}");
            // truncate to width-1 (ANSI-aware) so no cell wraps to a second row
            let line = console::truncate_str(&line, width.saturating_sub(1), "");
            let _ = write!(out, "{line}\x1b[0m\x1b[K");
            st.last_vis.push(measure(&line));
            // 5. a block that got shorter has left its old bottom rows on
            // screen -- those, and only those, need the wipe
            if n_pending + block_rows(&st.last_vis, width) < prev_rows {
                let _ = write!(out, "\x1b[J");
            }
            let _ = out.flush();
        }
        // The frame's worth of waiting is spent listening instead of sleeping.
        pump_keys(Duration::from_millis(FRAME_MS));
    }
}

/// The draft's keyboard, drained on the render thread's own clock.
///
/// The picker's vocabulary, minus everything that would mean changing what is
/// being computed: audio and theme are presentation, and presentation stays
/// live while the goblins work. Anything else is ignored -- a stray key must
/// never cost an hour of encoding, which is also why quitting is Ctrl-C and not
/// `q`.
fn pump_keys(budget: Duration) {
    use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    let deadline = Instant::now() + budget;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match event::poll(left) {
            Ok(true) => {}
            // Poll failing means no console to read (a redirected run reaches
            // here only if raw mode somehow took): wait the budget out rather
            // than spinning on the error.
            Ok(false) => return,
            Err(_) => {
                std::thread::sleep(left);
                return;
            }
        }
        let ev = match event::read() {
            Ok(ev) => ev,
            Err(_) => return,
        };
        let Event::Key(k) = ev else { continue };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        match k.code {
            KeyCode::Char('c') | KeyCode::Char('C')
                if k.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                // First ask: unwind at the next chunk boundary, which keeps the
                // working files consistent for the resume. Second ask: the user
                // has waited through a chunk and wants out NOW -- and since
                // every artifact is written atomically, leaving on the spot
                // costs nothing but the chunk in flight.
                if cancel::request() > 1 {
                    let _ = ratatui::crossterm::terminal::disable_raw_mode();
                    let mut out = std::io::stdout().lock();
                    let _ = write!(out, "\r\x1b[J");
                    // leaving by `exit` runs no destructor, so the pinned
                    // header's scroll region has to be given back by hand
                    chrome::release(&mut out);
                    let _ = out.flush();
                    std::process::exit(130);
                }
            }
            KeyCode::Char('m') | KeyCode::Char('M') => {
                sound::cycle_audio();
                sound::play_click();
            }
            KeyCode::Char('v') | KeyCode::Char('V') if sound::music_on() => {
                sound::cycle_volume();
                sound::play_click();
            }
            KeyCode::Char('n') | KeyCode::Char('N') if sound::music_on() => {
                sound::next_track();
                sound::play_click();
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                theme::cycle();
                sound::play_click();
            }
            _ => {}
        }
    }
}

/// Where a stage line's dot leader ends, and how wide its detail field is
/// before the status chip -- the processing log and the startup report share a
/// column grid so a finished run reads as one continuous printout.
const STAGE_LEADER: usize = 14;
const STAGE_INFO: usize = 38;

/// A cleared stage's permanent line: `normalize .... reused ... [ OK ]`.
///
/// Both fields are laid out in CELLS, not characters. A translated stage name
/// is one column per latin letter but two per CJK glyph, and `{:<N}` pads by
/// characters -- measuring it that way walks the chip out of the column the
/// startup report shares with it the moment the console speaks anything else.
fn done_line(what: &str, info: &str) -> String {
    let t = theme();
    let dots = STAGE_LEADER.saturating_sub(measure(what));
    let pad = " ".repeat(STAGE_INFO.saturating_sub(measure(info)));
    format!(
        "  {} {} {} {}",
        style(what.to_string()).fg(con(t.text)),
        style(".".repeat(dots)).fg(con(t.muted)),
        style(format!("{info}{pad}")).fg(con(t.muted)),
        style(TICK).fg(con(t.ok)).bold(),
    )
}

/// The stage portion of the live line:
/// `ENCODE ..... [████████░░░░░░]  62%  4.2x, 9s left`.
/// The bar width flexes to what is left after the fixed fields so the whole line
/// stays within the terminal.
fn stage_body(s: &Stage, width: usize) -> String {
    let t = theme();
    let label = s.label.to_uppercase();
    let label_w = measure(&label);
    let dots = STAGE_LEADER.saturating_sub(label_w);
    let head = format!(
        "{} {}",
        style(&label).fg(con(t.text)).bold(),
        style(".".repeat(dots)).fg(con(t.muted)),
    );
    // Still loading the graph: no bar, no numbers, just what it is waiting on.
    let Some(pb) = &s.pb else {
        return format!("{head} {}", style(s.note.as_str()).fg(con(t.muted)));
    };
    let pos = pb.position();
    let len = pb.length().unwrap_or(1).max(1);
    let frac = (pos as f64 / len as f64).clamp(0.0, 1.0);
    // No rate until the estimator has seen one position update, and indicatif
    // reports the absent rate and ETA as ZERO. Printing those verbatim gives
    // "0.0x, 0s left" -- a stage that has not finished its first chunk claiming
    // it is done in no time. Say what is true instead: not measured yet.
    let rate = pb.per_sec();
    let tail = if rate > 0.0 {
        format!(
            "{:>3}%  {:.1}x, {} left",
            (frac * 100.0) as u64,
            rate / s.units_per_sec.max(1e-9),
            fmt_dur(pb.eta().as_secs_f64()),
        )
    } else {
        format!("{:>3}%  taking a first measure...", (frac * 100.0) as u64)
    };
    // width taken by the goblin, the prefix, the labelled leader, the brackets
    // and the tail; the bar gets the rest, clamped to a sane band so it neither
    // vanishes nor overruns.
    let fixed =
        2 + measure(GOBLIN[0]) + 2 + label_w + 1 + dots + 2 + 2 + measure(&tail) + 2;
    let bar_w = width.saturating_sub(fixed).clamp(6, 32);
    let filled = (frac * bar_w as f64).floor() as usize;
    let full = "\u{2588}".repeat(filled.min(bar_w)); // full block
    let empty = "\u{2591}".repeat(bar_w.saturating_sub(filled)); // light shade
    format!(
        "{head} [{}{}]  {}",
        style(full).fg(con(t.bar)),
        style(empty).fg(con(t.bar_dim)),
        style(tail).fg(con(t.muted)),
    )
}

/// Display width of a string, ignoring ANSI escapes.
fn measure(s: &str) -> usize {
    console::measure_text_width(s)
}

/// "1h 23m" / "4m 06s" / "48s" -- durations the way a person says them.
fn fmt_dur(secs: f64) -> String {
    let s = secs.round().max(0.0) as u64;
    if s >= 3600 {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// One separator style per platform -- `--out` given with forward slashes
/// otherwise leaks a mixed `C:/dir\file` into every printed path.
fn nice(p: &Path) -> String {
    let s = p.display().to_string();
    if cfg!(windows) {
        s.replace('/', "\\")
    } else {
        s
    }
}

/// Everything a re-style needs, held after the draft: the head's tracks and
/// the row clock. A few MB per video -- what makes the review loop free.
struct DraftOut {
    name: String,
    /// The source video -- the review page streams it when a browser can.
    src: PathBuf,
    /// The shot cuts the tracks were decoded against, video milliseconds on
    /// the draft's clock: what the flag channel and `shot_edges` were built
    /// from. A re-decode at other cuts replaces both together.
    cuts_ms: Vec<f64>,
    /// The one normalized file this draft decoded from, when a transcode
    /// happened -- the review page's fallback (and, for VR, only) picture.
    norm: Option<PathBuf>,
    dst: PathBuf,
    tracks: style::Tracks,
    /// The trunk `tracks` were decoded by (`Bundle::trunk`). A model on
    /// another trunk needs its own head pass; a model on this one is a
    /// restyle.
    trunk: String,
    /// The authorship seed `tracks` were decoded at. A review that moves the
    /// seed has to re-run the head stage -- styling cannot reach the
    /// envelope -- and this is what says whether it must.
    tracks_seed: u64,
    /// Tracks this draft made and set aside, by (trunk, seed): what a switch
    /// back to a model or a seed already visited takes instead of a head pass.
    shelf: HashMap<(String, u64), style::Tracks>,
    /// The cuts the shelf and the tracks in hand were decoded at. `trunk_key`
    /// covers the graphs and the seed but NOT the cuts, so this is what tells
    /// an edited cut list that nothing cached may be trusted and the head has
    /// to run again. Equal to `cuts_ms` until a review moves a shot boundary.
    shelf_cuts: Vec<f64>,
    shot_edges: Vec<usize>,
    times: Vec<f64>,
    cache_dir: PathBuf,
    dur_ms: f64,
    /// Where this draft's clock starts on the SOURCE clock. Non-zero only for
    /// a VR clip whose prep trimmed a range: everything downstream of the
    /// transcode -- rows, times, tracks, the review page's video -- runs on the
    /// trimmed clip's clock, and this is what puts the written funscript back
    /// on the clock of the file the user actually plays.
    t0_ms: f64,
    /// This draft came from a reprojected VR source, so the ONLY video that
    /// matches it is the normalized copy -- the original is still two
    /// equirectangular eyes, and reviewing the script against that would be
    /// judging it on a picture the goblins never saw.
    is_vr: bool,
    /// The auto-crop plan as fractions of the frame, when a crop was actually
    /// taken: what the review page draws over the video. `None` = the goblins
    /// saw the whole frame -- the probe skipped, or its answer was identity.
    crop: Option<autocrop::View>,
    /// The model that drafted it, as the bundle's pack name: what the review
    /// re-decodes and re-styles it with, and what its tab says beside a
    /// name another model's draft shares.
    pack: String,
}

/// A draft as the review page sees it. Probe-and-fallback source choice: the
/// ORIGINAL when every browser plays it (full quality), the cache's
/// normalized copy when not. A VR draft has exactly one matching picture,
/// the reprojected copy: never the original, however playable -- it is two
/// equirect eyes.
fn review_clip(o: &DraftOut, b: &Bundle) -> Result<review::Clip> {
    let man = &b.manifest;
    let norm = o.norm.clone().filter(|p| p.exists());
    let (video, fallback, is_original) = match norm {
        Some(n) if o.is_vr => (n, None, false),
        n if ffmpeg::browser_playable(&o.src) => (o.src.clone(), n, true),
        Some(n) => (n, None, false),
        None => (o.src.clone(), None, true), // --no-transcode dev runs
    };
    Ok(review::Clip {
        name: o.name.clone(),
        script: o.dst.clone(),
        video,
        fallback,
        is_original,
        duration_ms: o.dur_ms,
        row_hz: man.row_hz(),
        row0_ms: man.row_ms(0),
        t0_ms: o.t0_ms,
        crop: o.crop.clone(),
        model: o.pack.clone(),
        models: b.models()?,
        compare: o.cache_dir.join("compare.funscript"),
        // the cuts this draft was DECODED against: the page's shot layer, and
        // what its "reset" goes back to. The session's own edit rides the
        // Choice, never the draft.
        cuts_ms: o.cuts_ms.clone(),
    })
}

/// One video's verdict. The payload is large next to `Skipped`, and boxing it
/// would be ceremony: this is constructed once per video, against minutes of
/// work each, and the whole of it is moved straight into the review's list.
#[allow(clippy::large_enum_variant)]
enum Outcome {
    Drafted(DraftOut),
    Skipped,
}

/// Each video's VR aim, for the batch about to run. Absent = flat, drafted
/// exactly as it comes.
type VrMap = std::collections::HashMap<PathBuf, vr::Config>;

/// The aim to apply to `video`, or `None` when it is flat (or the human looked
/// at the prep page and said "not VR after all").
fn vr_for<'a>(map: &'a VrMap, video: &Path) -> Option<&'a vr::Config> {
    map.get(video).filter(|c| !c.skip)
}

/// The style config a parameter set produces from the manifest's frozen
/// constants. At `Params::default()` this IS the manifest.
fn style_cfg(man: &bundle::Manifest, p: &style::Params) -> style::StyleCfg {
    style::StyleCfg {
        fps: man.row_hz(),
        // an expert numeric override wins over the preset it stands in for
        still_eps: p.still_eps.unwrap_or_else(|| p.stillness.still_eps(man.still_eps)),
        ext_snap: man.ext_snap,
        amp_cap_x: man.amp_cap_x,
        amp_cap_f0: man.amp_cap_f0,
        // per-row overrides are a mixed review's business; a plain draft
        // leaves them empty so the scalars above cover the whole clip
        amp_cap_x_rows: Vec::new(),
        amp_cap_f0_rows: Vec::new(),
        manual_ms: Vec::new(),
        env_gain_p: man.env_gain_p,
        plat_thr: man.plat_thr,
        plat_lo: man.plat_lo,
        plat_peak: man.plat_peak,
        plat_veto: man.plat_veto,
        plat_rail_track: man.plat_rail_track,
        // the dwell control and its expert number move the SAME axis -- the
        // ramp's start -- so the box beside the dropdown overrides the thing
        // the dropdown selects, and the manifest keeps the ramp's top
        plat_soft: {
            let man_soft = (man.plat_soft[0], man.plat_soft[1]);
            match p.dwell_ramp {
                Some(p0) => (p0, man_soft.1),
                None => p.dwells.plat_soft(man_soft),
            }
        },
        plat_shift_cap: man.plat_shift_cap,
        // the manifest speaks SECONDS; rows are formed here, once
        rev_snap: if man.rev_snap_s > 0.0 { man.rows_at(man.rev_snap_s) } else { 0 },
        rev_viterbi: man.rev_source == "viterbi",
        rev_gap_rows: man.rows_at(man.rev_gap_s).max(1),
        rev_gap_k: man.rev_gap_k,
        period_cap_rows: man.rows_at(man.period_cap_s).max(1),
        rev_gap_prior: man.rev_gap_prior,
        speed_ref_s: man.speed_ref_s,
        rev_smooth_s: man.rev_smooth_s,
        bias_fit_rows: man.bias_fit_rows(),
        subframe_rev: man.subframe == "rev",
        // depth uniformity resolves entirely from the params (its presets carry
        // fixed constants, none of them manifest-tuned)
        depth: p.depth_params(),
        // filler is a user authorship knob, never a manifest
        // constant -- it resolves entirely from the params
        filler_gap_s: p.filler_gap_s,
        filler_min_real_s: p.filler_min_real_s,
        filler_real_v: p.filler_real_v,
        filler_model_w: p.filler_model_w,
        filler_rate: p.filler_rate,
        filler_amp: p.filler_amp,
        filler_ramp_s: p.filler_ramp_s,
        filler_max_bridge_s: p.filler_max_bridge_s,
        filler_sway: p.filler_sway,
        filler_sway_s: p.filler_sway_s,
        filler_pattern: p.filler_pattern,
        filler_burst: p.filler_burst,
        filler_rest_s: p.filler_rest_s,
        cut_ease: p.cut_ease,
    }
}

/// A written funscript. Field order is deliberate: the small keys and
/// `metadata` come before the long `actions` array, so a text editor shows the
/// author at the top of the file instead of after thousands of points. A
/// derived struct serializes in declaration order (unlike a serde_json object
/// map, which sorts keys), which is what pins that order.
#[derive(serde::Serialize)]
struct Funscript<'a> {
    version: &'static str,
    inverted: bool,
    range: i64,
    metadata: FunMeta<'a>,
    actions: &'a [style::Action],
}

/// Funscript `metadata` -- the author stamp plus provenance tags, always
/// present. `filler` (ms ranges the background rhythm replaced) appears
/// only when the filler stage ran; `artifacts` (ms ranges whose action
/// sequence reads hot under the corpus rarity prior, artifacts.rs) when
/// any exist. The review page reads both to draw timeline bands.
#[derive(serde::Serialize)]
struct FunMeta<'a> {
    author: &'static str,
    tags: &'static [&'static str],
    /// the model and the flags that wrote the script, so it can be made again
    model: &'a str,
    settings: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    filler: Vec<[i64; 2]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    artifacts: Vec<[i64; 2]>,
    /// the run-share phase reading of a composed decode (style::PhaseRuns)
    #[serde(skip_serializing_if = "Option::is_none")]
    phase_runs: Option<style::PhaseRuns>,
}

/// The seed a style decodes at on this model: the params' own, or the
/// model's (`u64::MAX` is "whatever the bundle says").
fn seed_for(b: &Bundle, p: &style::Params) -> u64 {
    if p.env_seed == u64::MAX {
        b.manifest.env_seed
    } else {
        p.env_seed
    }
}

/// Put the tracks of `b`'s model at `seed` in hand, from the latents the
/// draft left behind.
///
/// A model is a trunk and a set of decode constants. The constants are the
/// styler's, so a model on the trunk in hand is only a new name. The seed
/// feeds `env_step` and the trunk is the head graph itself, so either one
/// moving needs tracks made by the HEAD stage -- unless this draft made them
/// before and shelved them. Everything the head pass needs is already held:
/// the latents sit in the draft's own cache until the review is finished,
/// and the cuts and the crop they were encoded under belong to the video,
/// not to a model. Seconds to a minute per clip against milliseconds for a
/// restyle, which is why the page and the API treat it as work rather than
/// as a knob.
fn retrack(d: &mut DraftOut, b: &Bundle, seed: u64) -> Result<()> {
    d.pack = b.manifest.pack.clone();
    let key = (b.trunk(), seed);
    if d.trunk == key.0 && d.tracks_seed == seed {
        return Ok(());
    }
    let tracks = match d.shelf.remove(&key) {
        Some(t) => t,
        None => head_pass(d, b, seed)?,
    };
    let old = std::mem::replace(&mut d.tracks, tracks);
    let old_key = (std::mem::replace(&mut d.trunk, key.0), d.tracks_seed);
    d.shelf.insert(old_key, old);
    d.tracks_seed = seed;
    Ok(())
}

/// Shelve the tracks of every model of the bundle for a review: one head
/// pass per trunk the draft does not hold yet, at the seed the page opens on,
/// and the drafted model back in hand. What makes every model on the page a
/// restyle rather than a wait.
fn pretrack(d: &mut DraftOut, b: &Bundle, params: &style::Params) -> Result<()> {
    for m in b.pack_names() {
        let mb = b.select(&m)?;
        let key = (mb.trunk(), seed_for(&mb, params));
        if (d.trunk == key.0 && d.tracks_seed == key.1) || d.shelf.contains_key(&key) {
            continue;
        }
        println!("  {}", style(t!("console.pretrack", model = m)).fg(con(theme().muted)));
        retrack(d, &mb, key.1)?;
    }
    retrack(d, b, seed_for(b, params))
}

/// The head stage alone, over the draft's cached latents, on `b`'s trunk at
/// `seed`: the tracks the encode stage would have made alongside them.
fn head_pass(d: &DraftOut, b: &Bundle, seed: u64) -> Result<style::Tracks> {
    let mut man = b.manifest.clone();
    man.env_seed = seed;
    let head = b.head_session()?;
    let env = b
        .env_session()?
        .context("this bundle has no envelope graph -- composed styling needs one")?;
    let n = d.shot_edges.last().copied().unwrap_or(0);
    let cut_rows = boundaries::cut_rows(&d.cuts_ms, |i| man.row_ms(i));
    let mut heads = heads::Heads::new(head, env, &man, cut_rows);
    let lat = encode::Latents {
        path: d.cache_dir.join("latents.i8"),
        rows: n,
        row_bytes: man.dim * man.grid * man.grid,
    };
    let pb = indicatif::ProgressBar::hidden();
    heads::stream_cache(&lat, &mut heads, None, &man, &pb)?;
    heads.finish(n, &d.shot_edges)
}

/// One styled list, on the SOURCE clock, with the readouts the file and the
/// API carry beside it.
struct Styled {
    actions: Vec<style::Action>,
    /// ms ranges the background rhythm replaced
    filler: Vec<[i64; 2]>,
    /// ms ranges whose action sequence reads hot under the rarity prior
    artifacts: Vec<[i64; 2]>,
    art_rate: f64,
    /// the composed decode's run-share phase reading
    phase_runs: Option<style::PhaseRuns>,
}

/// Compose + shape + write: everything downstream of the head's tracks.
/// This is the whole cost of changing a styling parameter -- milliseconds --
/// so the review loop can rewrite on every keypress. Returns the action
/// count plus the artifact-rarity spans/rate (artifacts.rs) of the
/// written list, for the batch summary readout.
fn restyle(
    d: &DraftOut,
    man: &bundle::Manifest,
    params: &style::Params,
) -> Result<(usize, Vec<[i64; 2]>, f64)> {
    let s = style_draft(d, man, params, &[]);
    write_styled(&d.dst, &s, man, params)?;
    Ok((s.actions.len(), s.artifacts, s.art_rate))
}

/// `restyle` on caller-supplied tracks: the mixed review's write, the same
/// file path and the same readout, only the rows come from a splice.
fn restyle_mixed(
    d: &DraftOut,
    man: &bundle::Manifest,
    params: &style::Params,
    tracks: &style::Tracks,
    cap_x: &[f64],
    cap_f0: &[f64],
    manual_ms: &[crate::wave::ManualSpan],
) -> Result<(usize, Vec<[i64; 2]>, f64)> {
    let s = style_draft_with(d, man, params, &[], tracks, cap_x, cap_f0, manual_ms);
    write_styled(&d.dst, &s, man, params)?;
    Ok((s.actions.len(), s.artifacts, s.art_rate))
}

/// One styled list as a funscript at `path`.
fn write_styled(path: &Path, s: &Styled, man: &bundle::Manifest, p: &style::Params) -> Result<()> {
    let script = Funscript {
        version: "1.0",
        inverted: false,
        range: 100,
        metadata: FunMeta {
            author: AUTHOR,
            tags: TAGS,
            model: &man.pack,
            settings: style::flags_line(p),
            filler: s.filler.clone(),
            artifacts: s.artifacts.clone(),
            phase_runs: s.phase_runs,
        },
        actions: &s.actions,
    };
    std::fs::write(path, serde_json::to_vec(&script)?)
        .with_context(|| format!("could not write {}", path.display()))
}

/// The review's re-style of one draft: its model's tracks in hand at the
/// params' seed, the script written, and -- when the page compares -- the
/// other model's line at the same knobs, written to `compare`. The compared
/// model is styled first so the draft ends with its own model's tracks in
/// hand. Returns the script's action count and its model's confidence.
fn review_style(
    d: &mut DraftOut,
    b: &Bundle,
    p: &style::Params,
    choice: &review::Choice,
    compare: &Path,
) -> Result<review::Restyled> {
    with_cuts(d, b, p, choice.cuts_ms.as_deref(), |d| {
        if let Some(m) = &choice.compare {
            let cb = b.select(m)?;
            retrack(d, &cb, seed_for(&cb, p))?;
            write_styled(compare, &style_draft(d, &cb.manifest, p, &[]), &cb.manifest, p)?;
        }
        let bb = b.select(&choice.model)?;
        retrack(d, &bb, seed_for(&bb, p))?;
        // The rows the script is written from: the chosen model's tracks over the
        // whole clip, each span's own model spliced in on its rows, with the
        // per-row amplitude bound that follows (each pack keeps its own `amp_cap_x`).
        let (mixed, cap_x, cap_f0, conf, manual_ms) = mixed_tracks(d, b, p, choice, &bb)?;
        let (actions, _, _) =
            restyle_mixed(d, &bb.manifest, p, &mixed, &cap_x, &cap_f0, &manual_ms)?;
        Ok(review::Restyled { actions, conf })
    })
}

/// Decode the draft at a session's edited cut list for the length of `body`,
/// then hand the draft its own cuts back.
///
/// An edit here is not a knob: the shot cuts reach the head as its `cut` input
/// channel AND are what `shot_edges` is built from, so the tracks in hand and
/// every shelved trunk were decoded for the OLD boundaries. `trunk_key` covers
/// the graphs and the seed but not the cuts, so nothing cached is reused: the
/// shelf is dropped, the in-hand key poisoned, and `pretrack` refills it -- one
/// head pass per trunk, seconds each. `shelf_cuts` is what says the shelf is
/// already the right shape so a run of edits with one cut set rebuilds ONCE.
///
/// `cuts_ms` and `shot_edges` are the draft's own and are restored on every
/// path, so the page's "reset" always has the original to ask for and the rest
/// of the program never sees a half-edited draft.
fn with_cuts<R>(
    d: &mut DraftOut,
    b: &Bundle,
    p: &style::Params,
    cuts: Option<&[f64]>,
    body: impl FnOnce(&mut DraftOut) -> Result<R>,
) -> Result<R> {
    // `None` = the draft's own cuts; `Some` = this session's edit
    let want: Vec<f64> = match cuts {
        Some(c) => c.to_vec(),
        None => d.cuts_ms.clone(),
    };
    let old_cuts = std::mem::replace(&mut d.cuts_ms, want.clone());
    let old_edges = std::mem::replace(&mut d.shot_edges, Vec::new());
    let n = old_edges.last().copied().unwrap_or(0);
    // the body always runs with cuts and edges in step, whichever branch below
    d.shot_edges = boundaries::shot_edges(&d.cuts_ms, n, |i| b.manifest.row_ms(i));
    let res = (|| -> Result<R> {
        if want != d.shelf_cuts {
            // nothing was decoded at these cuts: the shelf and the hand are stale
            d.shelf.clear();
            d.trunk.clear();
            d.tracks_seed = u64::MAX;
            pretrack(d, b, p)?;
            d.shelf_cuts = want.clone();
        }
        body(d)
    })();
    d.cuts_ms = old_cuts;
    d.shot_edges = old_edges;
    res
}

/// The tracks a review's chosen model and spans make, plus the per-row
/// amplitude bound they carry and the confidence the page reads back.
///
/// With no spans this is exactly the base model's tracks and empty cap rows,
/// so a review that never used the razor is byte-for-byte what it was. With
/// spans: the base model's tracks fill the clip, then each span's model's rows
/// are spliced in. The spliced tracks are a CLONE -- the shelf and the draft's
/// own hand are left alone, so a mixed review never evicts the base model.
fn mixed_tracks(
    d: &mut DraftOut,
    b: &Bundle,
    p: &style::Params,
    choice: &review::Choice,
    base: &Bundle,
) -> Result<(
    style::Tracks,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<crate::wave::ManualSpan>,
)> {
    let n = d.tracks.vmarg.len();
    let none = Vec::new();
    if choice.spans.is_empty() {
        return Ok((
            d.tracks.clone(),
            Vec::new(),
            Vec::new(),
            d.tracks.conf.clone(),
            none,
        ));
    }
    let mut mixed = d.tracks.clone();
    // the base model's bound over the clip; each span overwrites its own rows
    let mut cap_x = vec![base.manifest.amp_cap_x; n];
    let mut cap_f0 = vec![base.manifest.amp_cap_f0; n];
    // Manual pieces, in the clip's milliseconds. Reported back so the styler
    // can keep the model off them (the output-domain intensity scale, and the
    // gap filler, whose whole job is to stamp rhythm over rows that look still
    // -- which a wave's own floor always does) and so the finished list can be
    // rewritten here from the waves themselves.
    let mut manual_ms: Vec<crate::wave::ManualSpan> = Vec::new();
    // Resolve every span's tracks FIRST (this can need a head pass, which
    // borrows `d` mutably), then splice from the owned copies.
    let mut pieces: Vec<(usize, usize, style::Tracks, f64, f64)> = Vec::new();
    for sp in &choice.spans {
        if sp.model == review::MANUAL {
            continue; // written over the splice below, from its own recipe
        }
        // a span naming a model this bundle no longer has is skipped, not fatal
        let sb = match b.select(&sp.model) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let t = tracks_for(d, &sb, seed_for(&sb, p))?;
        let (lo, hi) = span_rows(&d.times, sp.lo_ms, sp.hi_ms);
        if hi <= lo {
            continue;
        }
        pieces.push((lo, hi, t, sb.manifest.amp_cap_x, sb.manifest.amp_cap_f0));
    }
    for (lo, hi, t, cx, cf) in pieces {
        mixed.splice_from(&t, lo, hi);
        let hi = hi.min(cap_x.len());
        for i in lo..hi {
            cap_x[i] = cx;
            cap_f0[i] = cf;
        }
    }
    // Manual LAST, so a model splice can never land on top of a hand-drawn
    // piece. These rows owe no model anything: no shelf key, no head pass.
    for sp in &choice.spans {
        if sp.model != review::MANUAL {
            continue;
        }
        let spec = match &sp.wave {
            Some(w) => w,
            None => continue,
        };
        let (lo, hi) = span_rows(&d.times, sp.lo_ms, sp.hi_ms);
        if hi <= lo {
            continue;
        }
        if crate::wave::fill_rows(&mut mixed.level, &mut mixed.vmarg, &d.times, lo, hi, spec) {
            manual_ms.push(crate::wave::ManualSpan {
                lo_ms: sp.lo_ms,
                hi_ms: sp.hi_ms,
                spec: spec.clone(),
                no_limit: sp.no_limit,
                no_cuts: sp.no_cuts,
            });
        }
    }
    // Each ordinary piece's own intensity, about its own mean. Applied to the
    // raw spliced level, so a knob dragged to 2 and back to 1 restores the
    // model's own output instead of compounding, and so the extrema / RDP /
    // seam / speed stages downstream still get the last word. Hand-drawn
    // pieces are left out on purpose: their band is what the user typed, not a
    // model's guess to be nudged.
    let pieces: Vec<style::PieceScale> = choice
        .spans
        .iter()
        .filter(|sp| sp.model != review::MANUAL && (sp.scale - 1.0).abs() > 1e-9)
        .map(|sp| style::PieceScale {
            lo_ms: sp.lo_ms,
            hi_ms: sp.hi_ms,
            k: sp.scale,
        })
        .collect();
    if !pieces.is_empty() {
        style::scale_pieces(&mut mixed.level, &d.times, &pieces);
    }
    let conf = mixed.conf.clone();
    Ok((mixed, cap_x, cap_f0, conf, manual_ms))
}

/// The tracks `b`'s model at `seed` would make, WITHOUT disturbing the
/// draft's own: a hit in the shelf or in hand is a clone, otherwise a head
/// pass. Cloning (a few MB) rather than shelf-swapping keeps a mixed review
/// from evicting the model the page is otherwise working on.
fn tracks_for(d: &mut DraftOut, b: &Bundle, seed: u64) -> Result<style::Tracks> {
    let key = (b.trunk(), seed);
    if d.trunk == key.0 && d.tracks_seed == seed {
        return Ok(d.tracks.clone());
    }
    if let Some(t) = d.shelf.get(&key) {
        return Ok(t.clone());
    }
    head_pass(d, b, seed)
}

/// The row range a millisecond span covers: `times[i]` is row `i`'s ms on the
/// clip's clock, so both ends are a partition point and a span that starts and
/// ends inside one row yields an empty range the caller skips.
fn span_rows(times: &[f64], lo_ms: f64, hi_ms: f64) -> (usize, usize) {
    let n = times.len();
    let lo = times.partition_point(|&t| t < lo_ms).min(n);
    let hi = times.partition_point(|&t| t < hi_ms).min(n);
    (lo, hi)
}

/// Compose + shape, and the author's pins over the result: the styled list
/// itself, which `restyle` writes and the API answers with. `pins` are
/// vertices on the source clock (`style::pin_actions`); a console draft has
/// none.
fn style_draft(
    d: &DraftOut,
    man: &bundle::Manifest,
    params: &style::Params,
    pins: &[style::Action],
) -> Styled {
    style_draft_with(d, man, params, pins, &d.tracks, &[], &[], &[])
}

/// `style_draft` on tracks the caller supplies instead of the draft's own --
/// what a mixed review needs: the base model's tracks with a span's model's
/// rows spliced in, plus that splice's per-row amplitude bound (`cap_x` /
/// `cap_f0`, one entry per row; empty = the manifest's scalar covers the
/// clip, which is every ordinary draft). Everything downstream is the same
/// styler, so the knobs, intensity and throttle stay whole-clip whatever the
/// rows came from.
#[allow(clippy::too_many_arguments)]
fn style_draft_with(
    d: &DraftOut,
    man: &bundle::Manifest,
    params: &style::Params,
    pins: &[style::Action],
    tracks: &style::Tracks,
    cap_x: &[f64],
    cap_f0: &[f64],
    manual_ms: &[crate::wave::ManualSpan],
) -> Styled {
    let mut cfg = style_cfg(man, params);
    if !cap_x.is_empty() {
        cfg.amp_cap_x_rows = cap_x.to_vec();
        cfg.amp_cap_f0_rows = cap_f0.to_vec();
    }
    cfg.manual_ms = style::manual_pairs(manual_ms);
    // Level styling is carrier-only: no event segmentation, no sub-frame
    // times, no forced apexes -- exactly jepa_infer's non-composed path.
    let (p, sub, force, filled, phase_runs) = match params.style {
        style::Style::Composed => style::compose(tracks, &d.shot_edges, &cfg, &d.times),
        style::Style::Level => {
            // carrier-only decode, but the carrier-agnostic authorship
            // stages (depth uniformity, gap filler) apply here too --
            // a level draft must not silently ignore those knobs
            let (p, filled) =
                style::compose_level_full(tracks, &d.shot_edges, &cfg, &d.times);
            (p, std::collections::HashMap::new(), Vec::new(), filled, None)
        }
    };
    let mut actions = style::extrema_actions(
        &p,
        &d.times,
        &d.shot_edges,
        Some(&sub),
        cfg.fps,
        cfg.rev_smooth_s,
        &force,
        cfg.cut_ease,
    );
    style::shape_actions(&mut actions, params, manual_ms);
    // The drawing's own vertices LAST, before the clock moves: a piece written
    // by hand is solved from its wave rather than read off the row scan above,
    // so its peaks land on the band it was given and its middle stays as clean
    // as the shape is. On the CLIP's clock, like `manual_ms` itself.
    style::splice_manual(
        &mut actions,
        manual_ms,
        &d.cuts_ms,
        params,
        1000.0 / cfg.fps,
    );
    // Back onto the source clock. A VR prep that trimmed a range drafted a
    // CLIP, and every row above is on the clip's clock -- but the file the user
    // plays is the whole video, so the script that lands beside it has to be
    // too. Done at write time rather than earlier so the file on disk is right
    // at every moment, including after an interrupted run.
    if d.t0_ms != 0.0 {
        let shift = d.t0_ms.round() as i64;
        for a in &mut actions {
            a.at += shift;
        }
    }
    // filled row spans -> ms on the source clock, like the actions above
    let filler: Vec<[i64; 2]> = filled
        .iter()
        .filter(|(a, b)| *b > *a && *a < d.times.len())
        .map(|&(a, b)| {
            let b1 = (b - 1).min(d.times.len() - 1);
            [
                (d.times[a] + d.t0_ms).round() as i64,
                (d.times[b1] + d.t0_ms).round() as i64,
            ]
        })
        .collect();
    // the author's pins, on the source clock like the actions they join --
    // after the device cap on purpose: a pin is the author's placement, and
    // a transition they asked for is theirs to keep
    let actions = style::pin_actions(&actions, pins);
    // the rarity instrument runs on the final written list (post-shape,
    // source clock), exactly what jepa_infer's panel line scores
    let (artifacts, art_rate) = artifacts::artifact_events(&actions);
    Styled { actions, filler, artifacts, art_rate, phase_runs }
}

/// One video, end to end.
/// Where a video's funscript lands: `--out` if given, else beside the source.
fn script_dst(cli: &Cli, video: &Path) -> PathBuf {
    let out_dir = cli
        .out
        .clone()
        .unwrap_or_else(|| video.parent().unwrap_or(Path::new(".")).to_path_buf());
    // The stem is carried as an OS string the whole way. Going through a
    // `String` would put U+FFFD where a byte the platform allows but Unicode
    // does not used to be, and the script would land beside the video under a
    // name that is not the video's -- which is also the name the skip rule
    // looks for next run, so the draft would be made again every time.
    let mut name = video.file_stem().unwrap_or_default().to_os_string();
    if let Some(model) = &cli.out_suffix {
        name.push(".");
        name.push(model);
    }
    name.push(".funscript");
    out_dir.join(name)
}

/// The model the run drafts with: `--model`, else the `remembered` choice
/// when this bundle still carries it, else the bundle's default, and every
/// `--models` name checked against the bundle before a video is touched, so a
/// wrong name fails here and not after an encode.
fn pick_model(b: Bundle, cli: &Cli, remembered: Option<&str>) -> Result<Bundle> {
    for m in &cli.models {
        b.select(m)?;
    }
    match (&cli.model, remembered) {
        (Some(m), _) => b.select(m),
        (None, Some(m)) if b.pack_names().iter().any(|n| n == m) => b.select(m),
        _ => Ok(b),
    }
}

/// What made the script already sitting beside a video. Shown wherever an
/// existing script blocks a draft, because the two cases call for opposite
/// answers: an AI script is worth re-drafting with a newer build, someone's
/// hand work is not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScriptKind {
    /// Carries a machine-provenance stamp -- ours (`AUTHOR` / `TAGS`) or
    /// another drafting tool's.
    Ai,
    /// No such stamp. Hand-authored scripts never carry one -- though neither
    /// does a generator that stamps nothing, so this is strictly "unstamped".
    Hand,
    /// A file is there but nothing readable came back from it.
    Unknown,
}

impl ScriptKind {
    /// The one-word label, short enough to sit after a filename.
    pub fn label(self) -> &'static str {
        match self {
            ScriptKind::Ai => t!("script.kind.ai"),
            ScriptKind::Hand => t!("script.kind.hand"),
            ScriptKind::Unknown => t!("script.kind.unknown"),
        }
    }
}

/// Metadata substrings that mark a script as machine-drafted, lowercased. Ours
/// is the `ai-generated` tag `TAGS` writes; another tool's stamp lands in its
/// author string instead.
const AI_MARKERS: &[&str] = &["ai-generated", "ai generated", "goblinscript"];

/// Classify the script at `path` by sampling BOTH ends of the file. `metadata`
/// sits at whichever end the writing tool left it -- we write it first (see
/// `Funscript`), a tool that serializes its keys sorted lands it after the
/// actions -- and the array in between is numeric, so it can match no marker
/// and is not worth the megabytes it would cost to read.
pub fn script_kind(path: &Path) -> ScriptKind {
    use std::io::{Read, Seek, SeekFrom};
    const WIN: usize = 16 * 1024;
    let Ok(mut f) = std::fs::File::open(path) else {
        return ScriptKind::Unknown;
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    let mut sample = vec![0u8; WIN.min(len)];
    if f.read_exact(&mut sample).is_err() {
        return ScriptKind::Unknown;
    }
    if len > WIN {
        let mut tail = vec![0u8; WIN];
        if f.seek(SeekFrom::End(-(WIN as i64))).is_ok() && f.read_exact(&mut tail).is_ok() {
            sample.extend_from_slice(&tail);
        }
    }
    let text = String::from_utf8_lossy(&sample).to_lowercase();
    if !text.contains("actions") && !text.contains("metadata") {
        return ScriptKind::Unknown; // not a funscript we can read
    }
    if AI_MARKERS.iter().any(|m| text.contains(m)) {
        ScriptKind::Ai
    } else {
        ScriptKind::Hand
    }
}

/// Would this video be skipped for already having a script? Checked by `draft`
/// before it does anything, and by the batch loop before it hands a video to
/// the prefetcher -- transcoding ahead for a video nobody will draft is the one
/// way a head start can cost more than it saves.
fn will_skip(cli: &Cli, video: &Path) -> bool {
    script_dst(cli, video).exists() && !cli.force
}

/// The crop page, on the runs that can show one.
///
/// It opens by default, because the crop is the one decision in the pipeline
/// a person can make better than the goblins in a glance -- they can see what
/// the video is of. Two runs are skipped rather than asked: a PIPED one with
/// no API client has nobody to answer the page (`--crop-edit` demands it
/// anyway, for driving the endpoints), and a re-run whose plan came off the
/// cache already carries the answer the first run gave, hand-drawn or
/// accepted.
fn crop_page(
    cli: &Cli,
    live: &Live,
    norm: &Path,
    plan: &autocrop::Plan,
    probed: bool,
    dur_ms: f64,
    man: &bundle::Manifest,
) -> Result<Option<autocrop::Plan>> {
    let someone = live.tty || live.sink.is_some();
    if !(cli.crop_edit || (!cli.no_crop_edit && someone && probed)) {
        return Ok(None);
    }
    let (w, h) = ffmpeg::dims(norm)?;
    live.setup("console.stage.cropedit", t!("console.cropedit.waiting"));
    let edited = cropedit::edit(
        norm,
        plan,
        dur_ms,
        man.grid_fps,
        w as f64 / h.max(1) as f64,
        !cli.no_browser,
        &|url| {
            live.page("console.stage.cropedit", url);
            live.println(format!(
                "  {} {}  {}",
                t!("console.cropedit.label"),
                style(url).cyan().bold(),
                style(t!("console.cropedit.hint")).dim()
            ));
        },
    )?;
    if edited.is_none() {
        live.done("console.stage.cropedit", t!("console.cropedit.kept"));
    }
    Ok(edited)
}

/// The ONE normalize height for a source: crop-capable
/// (`autocrop::crop_norm_height` -- 576 at the shipped bundle), capped by
/// the source's own lines -- a normalize never invents detail -- and never
/// below the bundle's spec. A 480p source at the shipped spec reads 480:
/// the single file is then byte-identical to a spec-height normalize and
/// nothing softens.
fn norm_height(video: &Path, man: &bundle::Manifest) -> u32 {
    let spec = man.transcode.height;
    let src_h = ffmpeg::dims(video).map(|(_, h)| h as u32).unwrap_or(spec);
    autocrop::crop_norm_height(man.enc_res as u32).min(src_h).max(spec)
}

/// The containers a goblin will open. Extension only: probing every file in a
/// folder to find out would be a process launch per name, on the one screen the
/// picker opens to.
const VIDEO_EXT: &[&str] = &[
    "mp4", "m4v", "mkv", "avi", "mov", "wmv", "webm", "mpg", "mpeg", "m2ts", "ts", "flv",
];

pub fn is_video(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXT.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// The videos a folder holds DIRECTLY, in the order the picker lists them.
///
/// One level and no deeper, wherever a folder stands in for its contents -- the
/// picker's Space on a folder row, and a folder named on the command line. A
/// recursive walk would turn "draft this drop" into a library-wide job that
/// looks identical at the moment it is started, so the depth is a property of
/// the function rather than a flag someone can be talked into.
pub fn videos_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .map(|e| e.path())
        .filter(|p| is_video(p))
        .collect();
    out.sort_by_key(|p| p.file_name().unwrap_or_default().to_string_lossy().to_lowercase());
    out
}

/// A folder in the batch becomes the videos inside it, in place. Empty is an
/// error rather than a quiet nothing: the user pointed at that folder because
/// they believe videos are in it, and a silent drop to zero would reopen the
/// picker as though they had asked for nothing.
fn expand_dirs(cli: &mut Cli) -> Result<()> {
    if !cli.videos.iter().any(|v| v.is_dir()) {
        return Ok(());
    }
    let mut out: Vec<PathBuf> = Vec::new();
    for v in std::mem::take(&mut cli.videos) {
        if !v.is_dir() {
            out.push(v);
            continue;
        }
        let found = videos_in(&v);
        if found.is_empty() {
            anyhow::bail!(
                "no videos in {} -- sub-folders are not searched, so name the folder the videos \
                 are actually in",
                nice(&v)
            );
        }
        out.extend(found);
    }
    cli.videos = out;
    Ok(())
}

/// Turn any links in the batch into local files, in place, so that everything
/// downstream -- the skip rule, VR prep, the cache key -- sees only paths.
///
/// A link on the command line is a REQUEST, so a missing yt-dlp is an error here.
/// That is the one place it is: everywhere else the feature is simply not
/// offered, because a tool the user never installed is not a failure they need
/// reporting at them.
fn resolve_links(cli: &mut Cli) -> Result<()> {
    if !cli.videos.iter().any(|v| dl::is_url(&v.to_string_lossy())) {
        return Ok(());
    }
    dl::have_tool()?;
    let dir = match &cli.dl_dir {
        Some(d) => d.clone(),
        // the shell's own directory: the user typed the link there, and the
        // script will land beside the video it fetches
        None => std::env::current_dir().context("no working directory to fetch into")?,
    };
    let th = theme();
    // cloned so the loop below can hold `videos` mutably without also borrowing
    // the rest of `cli`
    let extra = cli.dl_args.clone();
    for v in cli.videos.iter_mut() {
        let url = v.to_string_lossy().to_string();
        if !dl::is_url(&url) {
            continue;
        }
        println!(
            "  {} {}",
            style("[ LINK ]").fg(con(th.accent)).bold(),
            style(t!("console.link.into", dir = dir.display())).fg(con(th.muted))
        );
        // yt-dlp draws its own progress straight onto this terminal -- it is the
        // bar the user already knows, and reprinting it worse would be a choice
        let got = dl::fetch_blocking(&url, &dir, &extra)?;
        println!(
            "  {} {}",
            style("[ LINK ]").fg(con(th.accent)).bold(),
            style(nice(&got)).fg(con(th.accent))
        );
        *v = got;
    }
    Ok(())
}

/// Find the VR sources in a batch and aim them -- all of them, in ONE browser
/// session, before any drafting starts.
///
/// Up front is not a preference. The batch loop normalizes the next video while
/// this one is on the GPU (`prefetch`), and a transcode cannot start for a video
/// whose aim is still a question -- so every question gets asked first, and the
/// batch then runs unattended the way an overnight queue is supposed to.
///
/// An aim already on disk is reused silently: re-running a video must not
/// re-ask, and the sidecar is keyed on the source bytes so it survives
/// everything except the file changing.
///
/// `marked` are videos the user marked VR by hand in the picker (B). A mark is
/// the newest human answer there is, so it outranks both the detector and an
/// earlier "not VR" -- it reopens the aiming page on a video the run would
/// otherwise have drafted flat.
fn vr_prep(
    cli: &Cli,
    videos: &[PathBuf],
    cache_root: &Path,
    marked: &BTreeSet<PathBuf>,
) -> Result<VrMap> {
    let mut map = VrMap::new();
    if cli.no_vr {
        return Ok(map);
    }
    let th = theme();
    let mut clips: Vec<vr::Clip> = Vec::new();
    let mut fresh = 0usize;
    // hand marks that reached a clip actually about to be drafted -- the user
    // asking for the page in as many words, so nothing below asks them again
    let mut hand = 0usize;

    for v in videos {
        if will_skip(cli, v) {
            continue; // already scripted -- nothing here will be drafted
        }
        let hand_marked = marked.contains(v);
        let Some(need) = vr_need(v, cache_root, hand_marked, cli.vr) else {
            continue;
        };
        if need.fresh {
            fresh += 1;
        }
        if let Some(why) = &need.why {
            println!(
                "  {} {}  {}",
                style("[ VR ]").fg(con(th.accent)).bold(),
                style(v.file_name().unwrap_or_default().to_string_lossy()).fg(con(th.accent)),
                style(why).fg(con(th.muted))
            );
        }
        if hand_marked {
            hand += 1;
        }
        clips.push(need.clip);
    }

    if clips.is_empty() {
        return Ok(map);
    }
    // The human asked for the page outright: by flag, or by marking a video VR
    // in the picker. Either way the questions below are already answered.
    let asked = cli.vr || hand > 0;
    // Everything already answered for: reuse in silence. `--vr` reopens the
    // page anyway -- that is what asking for it means.
    let unanswered = clips.iter().filter(|c| !c.cfg.touched).count();
    if unanswered == 0 && !cli.vr {
        for c in clips {
            map.insert(c.src.clone(), c.cfg);
        }
        return Ok(map);
    }

    let ask = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if !asked && !ask {
        // Nobody to ask and nothing forced: say so loudly rather than spend an
        // hour drafting equirectangular frames into nonsense.
        eprintln!(
            "  {} {}",
            style(SKIP).fg(con(th.warn)).bold(),
            style(t!("console.vr.noterminal", n = fresh)).fg(con(th.muted))
        );
        return Ok(map);
    }
    if !asked {
        print!(
            "\n  {} ",
            style(t!("console.vr.ask", n = unanswered))
                .fg(con(th.accent))
                .bold()
        );
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        let _ = std::io::stdin().read_line(&mut line);
        if line.trim().eq_ignore_ascii_case("n") {
            println!(
                "  {}",
                style(t!("console.vr.asis")).fg(con(th.muted))
            );
            return Ok(map);
        }
    }

    // Music always stops for a browser step: the page is where the attention is.
    let was_music = sound::music_on();
    sound::set_music(false);
    let done = vr::prep(clips, !cli.no_browser, &|url| {
        println!(
            "\n  {} {}  {}",
            style("[ VR ]").fg(con(th.accent)).bold(),
            style(url).fg(con(th.accent)).bold(),
            style(t!("console.vr.hint")).fg(con(th.muted))
        );
    });
    if was_music {
        sound::set_music(true);
    }
    for c in done? {
        map.insert(c.src.clone(), c.cfg);
    }
    Ok(map)
}

/// A video that wants the aiming page, or would if nobody had answered
/// for it yet. `None` is a flat video: the detector saw nothing and nobody
/// says otherwise.
struct VrNeed {
    clip: vr::Clip,
    /// Nobody has answered for this video: no sidecar, or a "not VR" that a
    /// hand mark just reopened.
    fresh: bool,
    /// The detector's reason, when this run is the first to see it as VR.
    why: Option<String>,
}

/// What `video` needs before it is drafted. `marked` is a hand mark on the
/// video -- the picker's, or an API client's -- and `force` is `--vr`: both
/// say the video is VR whatever the detector thinks.
///
/// A saved sidecar is the user's own earlier answer -- including "not VR",
/// which must keep meaning that on every later run. A hand mark just now is
/// a NEWER answer, so it reopens the question: the aim (if it ever had one)
/// is kept, the "not VR" is not.
fn vr_need(video: &Path, cache_root: &Path, marked: bool, force: bool) -> Option<VrNeed> {
    let sidecar = vr::sidecar_path(cache_root, video).ok()?;
    let saved = vr::load(&sidecar);
    let probe = |d: &Option<vr::Detected>| {
        d.as_ref()
            .map(|d| (d.w, d.h, d.dur_ms))
            .unwrap_or((0, 0, ffmpeg::duration_ms(video).unwrap_or(0.0)))
    };
    let (cfg, dur_ms, w, h, fresh, why) = match (saved, vr::detect(video)) {
        (Some(mut c), d) if marked && c.skip => {
            c.skip = false;
            c.touched = false;
            let (w, h, dur) = probe(&d);
            (c, dur, w, h, true, None)
        }
        (Some(c), d) => {
            let (w, h, dur) = probe(&d);
            (c, dur, w, h, false, None)
        }
        (None, Some(d)) => (d.cfg, d.dur_ms, d.w, d.h, true, Some(d.why)),
        // Not detected. A hand mark (or `--vr`) says the detector is wrong,
        // so offer it anyway on whatever the probe could tell us.
        (None, None) if force || marked => {
            let dur = ffmpeg::duration_ms(video).unwrap_or(0.0);
            (vr::Config::default(), dur, 0, 0, true, None)
        }
        (None, None) => return None,
    };
    Some(VrNeed {
        clip: vr::Clip {
            name: video.file_name().unwrap_or_default().to_string_lossy().to_string(),
            src: video.to_path_buf(),
            sidecar,
            cfg,
            dur_ms,
            src_w: w,
            src_h: h,
        },
        fresh,
        why,
    })
}

#[allow(clippy::too_many_arguments)]
fn draft(
    cli: &Cli,
    b: &Bundle,
    video: &Path,
    cache_root: &Path,
    params: &style::Params,
    idx: usize,
    total: usize,
    prefetch: &prefetch::Prefetch,
    // `next` is the following video in the batch, whose transcode runs in this
    // one's GPU-bound shadow. `None` on the last video (and a one-video run).
    next: Option<&Path>,
    // Every VR aim in the batch -- this video's, and the next one's, which the
    // head start needs before this video reaches the GPU.
    vrs: &VrMap,
    // Where the stage events go besides the console: the API server's
    // session, or nothing on a console run.
    sink: Option<Arc<dyn StageSink>>,
) -> Result<Outcome> {
    let man = &b.manifest;
    let vr_cfg = vr_for(vrs, video);
    let t0 = Instant::now();
    println!();
    // The per-video header prints to the console BEFORE the progress group is
    // built. A MultiProgress that has had a plain line pushed through it (an
    // `mp.println` with no bars yet) carries an orphan-line count that makes its
    // first bar draw twice on its opening tick. Keeping every pre-work line out
    // of `mp` lets the group start from a clean slate.
    let th = theme();
    println!(
        "{} {}{}",
        style("::").fg(con(th.logo)).bold(),
        style(video.file_name().unwrap_or_default().to_string_lossy())
            .fg(con(th.accent))
            .bold(),
        if total > 1 {
            style(format!("  [{}/{total}]", idx + 1))
                .fg(con(th.muted))
                .to_string()
        } else {
            String::new()
        }
    );

    // where the funscript will land -- decided (and checked) before any work,
    // because "already scripted" must cost zero minutes, not a transcode
    let dst = script_dst(cli, video);
    if will_skip(cli, video) {
        println!(
            "  {} {}",
            style(SKIP).fg(con(th.warn)).bold(),
            style(t!(
                "console.skip",
                kind = script_kind(&dst).label(),
                name = nice(&dst)
            ))
            .fg(con(th.muted))
        );
        sound::play_skip();
        // A skip costs no time, so the next video's head start has nothing to
        // hide behind -- but starting it here still overlaps it with whatever
        // skips follow, and with this one's own console output.
        if let Some(n) = next {
            let spec = bundle::Transcode {
                height: norm_height(n, &b.manifest),
                ..b.manifest.transcode.clone()
            };
            prefetch.start(n, cache_root, &spec, vr_for(vrs, n));
        }
        return Ok(Outcome::Skipped);
    }

    // The draft's timeline. A VR prep that trimmed a range makes it SHORTER
    // than the file: everything from the transcode on runs on the trimmed
    // clip's clock, and `t0_ms` is what puts the written script back on the
    // source's.
    let src_dur_ms = ffmpeg::duration_ms(video)?;
    let t0_ms = vr_cfg.map(|c| c.t0_ms(src_dur_ms)).unwrap_or(0.0);
    let mut dur_ms = match vr_cfg {
        Some(c) => {
            let (a, b) = c.range(src_dur_ms);
            b - a
        }
        None => src_dur_ms,
    };
    let max_s = cli.minutes.map(|m| m * 60.0);
    if let Some(t) = max_s {
        dur_ms = dur_ms.min(t * 1000.0);
    }

    // two clocks: decoded FRAMES (grid_fps -- the cut detector's budget) and
    // latent ROWS (row_hz -- the encoder's output and everything after it)
    let n_frames = (dur_ms / 1000.0 * man.grid_fps) as u64;
    let n_rows = (dur_ms / 1000.0 * man.row_hz()) as u64;
    println!(
        "  {} {}",
        style(fmt_dur(dur_ms / 1000.0)).fg(con(th.accent)).bold(),
        style(t!("console.ofvideo")).fg(con(th.muted))
    );
    // The keys the draft answers to, once per batch. Drafts run for hours; a
    // user who cannot find the volume or the exit sits through both.
    if idx == 0 && std::io::stdout().is_terminal() {
        println!(
            "  {}",
            style(t!("console.keys")).fg(con(th.muted))
        );
    }

    // Now the live area: an animated status line with the viewport above it,
    // resize-proof by redrawing the whole block in place every frame (see
    // `Live`). Every permanent line from here on goes through `live` so it
    // scrolls into history above the block instead of racing it.
    viz::clear(); // this video's viewport, never the last one's
    let live = Live::new(sink);

    let cache = Cache::open(cache_root, video, vr_cfg)?;

    // 1. normalize -- the encode spec every clip the model has seen went
    // through, at the clip's crop-capable height: ONE ffmpeg pass and ONE
    // cache video serve every stage. A crop reads the extra lines directly;
    // every uncropped read softens to the spec height inside its own decode
    // chain (`autocrop::soften_arg`), so a source with no lines to give
    // (or `--no-transcode`) is byte-identical to a spec-height normalize.
    let norm_h = norm_height(video, man);
    let norm_spec = bundle::Transcode { height: norm_h, ..man.transcode.clone() };
    // What the stages below decoded from, for the caches they leave behind:
    // this normalize, or the untouched source under `--no-transcode`. The two
    // are different pixels of the same clip and must never share an entry.
    let norm_id = if cli.no_transcode {
        cache::NORM_SOURCE.to_string()
    } else {
        cache::norm_key(&norm_spec)
    };
    // The span every cache below covers: the clip's own length, already cut
    // to `--minutes` where that is set.
    let span_ms = dur_ms.round() as i64;
    let norm = if cli.no_transcode {
        live.done("console.stage.normalize", t!("console.norm.notranscode"));
        video.to_path_buf()
    } else {
        let norm = cache.norm_video(&norm_spec);
        // A head start claimed here was begun while the PREVIOUS video was on
        // the GPU, so it is usually already done and this stage costs nothing.
        if let Some(job) = prefetch.claim(video) {
            let t = Instant::now();
            let pb = live.stage(
                "console.stage.normalize",
                1000,
                1000.0 / (dur_ms / 1000.0).max(1e-9),
            );
            let waited = job.wait(|p| pb.set_position(p))?;
            let s = t.elapsed().as_secs_f64();
            live.done("console.stage.normalize", &if waited.as_secs_f64() < 0.5 {
                t!("console.norm.ready").to_string()
            } else {
                t!("console.norm.headstart",
                   left = fmt_dur(s),
                   rate = format!("{:.1}", dur_ms / 1000.0 / s.max(1e-9)))
            });
        } else if norm.exists() {
            live.done("console.stage.normalize", t!("console.norm.reused"));
        } else {
            let t = Instant::now();
            let pb = live.stage(
                "console.stage.normalize",
                100,
                100.0 / (dur_ms / 1000.0).max(1e-9),
            );
            ffmpeg::transcode(video, &norm, &norm_spec, dur_ms, vr_cfg, cli.hwaccel, |f| {
                pb.set_position((f * 100.0) as u64)
            })?;
            let s = t.elapsed().as_secs_f64();
            live.done(
                "console.stage.normalize",
                &t!("console.norm.done",
                    height = norm_h,
                    took = fmt_dur(s),
                    rate = format!("{:.1}", dur_ms / 1000.0 / s)),
            );
        }
        norm
    };

    // A flat source whose shape is not the corpus's reaches the goblins
    // stretched, and every stage after this one inherits that. Said once,
    // whichever way the normalize above was satisfied -- a reused copy is
    // still the same stretched picture. A VR source is not asked: its viewport
    // has its own version of this complaint, on the page where it is aimed.
    if vr_cfg.is_none() {
        if let Some(note) = ffmpeg::dims(video)
            .ok()
            .and_then(|(w, h)| vr::flat_aspect_warning(w as u32, h as u32))
        {
            live.println(format!("  {}", style(note).fg(con(th.muted))));
        }
    }

    // 1.5 exposure -- the clip's one corrective gamma, read off the same file
    // every later stage decodes. It joins the decode chains that feed the
    // MODEL (crop probe + encode); cut detection reads the raw picture
    // (TransNet looks for transitions, not levels). An in-band clip reads
    // gamma 1.0, which is no filter and a byte-identical run.
    let expo = if cli.no_exposure {
        None
    } else {
        let e = exposure::probe(&norm, man.enc_res, man.grid_fps, n_frames as usize)?;
        live.done("console.stage.exposure", &e.stage_line());
        Some(e)
    };
    let photo = expo.as_ref().and_then(|e| e.filter());
    let gamma = expo.as_ref().map(|e| e.gamma).unwrap_or(1.0);

    // The CPU is now free and everything below this line is the GPU's: start
    // the next video's transcode into the gap. It runs against `cuts`,
    // `encode` and `model` and is normally finished before they are.
    if let Some(n) = next {
        let spec = bundle::Transcode { height: norm_height(n, man), ..man.transcode.clone() };
        prefetch.start(n, cache_root, &spec, vr_for(vrs, n));
    }

    // 2. cuts -- the model reads them as a flag channel, the styler as spans
    let cuts = match cache.read_cuts(man, &norm_id, span_ms) {
        Some(c) => {
            live.done("console.stage.cuts", &t!("console.cuts.reused", n = c.len()));
            c
        }
        None => {
            let t = Instant::now();
            live.setup("console.stage.cuts", t!("console.cuts.loading"));
            let mut sess = b.transnet_session()?;
            let pb = live.stage("console.stage.cuts", n_frames, man.grid_fps);
            let c = boundaries::find_cuts(&norm, &mut sess, man, n_frames, max_s, &pb)?;
            cache.write_cuts(man, &norm_id, span_ms, &c)?;
            let s = t.elapsed().as_secs_f64();
            live.done(
                "console.stage.cuts",
                &t!("console.cuts.done",
                    n = c.len(),
                    took = fmt_dur(s),
                    rate = format!("{:.1}", dur_ms / 1000.0 / s)),
            );
            c
        }
    };

    // 2.5 auto-crop (on by default) -- the mask net's attention picks one rect per
    // shot; the crop is applied inside the encode DECODE (SegmentedDecoder),
    // so the frame clock is untouched by construction. The probe is cached
    // per (clip, bundle) next to the latents.
    // On by default: `--no-autocrop` is the way past it. A bundle with no mask
    // graph cannot probe, and on the default that is a SKIP with a note -- an
    // old bundle must still draft. Only an explicit `--autocrop` makes it an
    // error, which is what that flag is now for.
    let want_crop = !cli.no_autocrop;
    let mut mask = if want_crop || live.tty { b.mask_session()? } else { None };
    let mut enc_sess: Option<ort::session::Session> = None;
    let mut crop_head: Option<ort::session::Session> = None;
    let plan = if want_crop && (mask.is_some() || cli.autocrop) {
        autocrop::require_mask(&mask)?;
        let mut probed = false;
        let p = match autocrop::read_cached(&cache.dir, man, gamma) {
            Some(p) => {
                live.done("console.stage.autocrop", &autocrop::stage_line(&p, true));
                p
            }
            None => {
                probed = true;
                let t = Instant::now();
                live.setup("console.stage.autocrop", t!("console.crop.loading"));
                enc_sess = Some(b.encoder_session()?);
                // the head judges the placements: the crop stage needs it
                // before the encode stage does, so it is built here and
                // handed on rather than opened twice
                if !cli.no_crop_search {
                    crop_head = Some(b.head_session()?);
                }
                let pb = live.stage("console.stage.autocrop", 1, 1.0);
                let p = autocrop::probe(
                    &norm,
                    enc_sess.as_mut().expect("created above"),
                    mask.as_mut().expect("require_mask passed"),
                    crop_head.as_mut(),
                    man,
                    &cuts,
                    n_frames as usize,
                    photo.as_deref(),
                    &pb,
                )?;
                autocrop::write_cached(&cache.dir, man, &p, gamma)?;
                let s = t.elapsed().as_secs_f64();
                live.done(
                    "console.stage.autocrop",
                    &t!("console.crop.timed",
                        what = autocrop::stage_line(&p, false),
                        took = fmt_dur(s)),
                );
                p
            }
        };
        // 2.6 the crop page (on by default at a terminal): the one stage a
        // person is asked to judge, placed where judging it is free. Nothing
        // downstream of the probe has run yet, so a corrected rect costs the
        // look it took and no GPU seconds; the same correction after a draft
        // would re-encode the whole video. It opens on the identity plan too
        // -- "the attention wants the whole frame" is exactly the answer
        // someone may want to overrule.
        let p = match crop_page(&cli, &live, &norm, &p, probed, dur_ms, man)? {
            Some(edited) => {
                autocrop::write_cached(&cache.dir, man, &edited, gamma)?;
                live.done("console.stage.cropedit", &autocrop::stage_line(&edited, false));
                edited
            }
            None => p,
        };
        // An identity plan is the uncropped decode under a different NAME: the
        // pixels match frame for frame, but carrying it in the cache key would
        // invalidate every latent tree drafted before the crop became the
        // default and re-encode hours to arrive at the same rows. The decision
        // is kept where it belongs -- the stage line said it, and the cached
        // plan means the probe does not repeat -- and the plan itself is
        // dropped here, so "the attention wants the whole frame" is byte-for-
        // byte the same run as never having asked.
        (!p.is_identity()).then_some(p)
    } else {
        if want_crop {
            live.done("console.stage.autocrop", t!("console.crop.nomask"));
        }
        None
    };

    // The crop's pixel identity: latents from a plan decoded off a taller
    // normalize differ from the same plan off a spec-height one, so the
    // height rides the key (the uncropped identity carries it in
    // `CacheMeta.norm_h` instead).
    let crop_key = plan.as_ref().map(|p| {
        let k = p.key();
        if norm_h > man.transcode.height { format!("{k}@{norm_h}") } else { k }
    });

    // 3. the expensive pass -- and the head, in the same stage.
    //
    // There is no model stage. The head is minutes of work against the
    // encoder's hours, so it runs on the rows the encoder has already made,
    // 1024 at a time: the chunk boundaries are the ones a whole-cache pass
    // would have used and the tracks are identical, but they arrive DURING the
    // long stage instead of after it -- which is what puts the strokes on
    // screen while the goblins are still working.
    let t = Instant::now();
    live.setup("console.stage.encode", t!("console.encode.loading"));
    // reuse the session the crop search already built, when it built one
    let head = match crop_head.take() {
        Some(h) => h,
        None => b.head_session()?,
    };
    let env = b
        .env_session()?
        .context("this bundle has no envelope graph -- composed styling needs one")?;
    let cut_rows = boundaries::cut_rows(&cuts, |i| man.row_ms(i));
    let mut heads = heads::Heads::new(head, env, man, cut_rows);
    // The viewport's own graph, and only when there is a viewport: on a piped
    // run nothing is drawn, so nothing is computed for it either.
    let mask = if live.tty { mask } else { None };

    let lat_path = cache.latents();
    let reused = cache.valid_latents(man, crop_key.as_deref(), &norm_id, gamma, span_ms);
    let lat = match &reused {
        // the rows are already on disk; the head still has to read them
        Some(m) => {
            let l = encode::Latents {
                path: lat_path,
                rows: m.rows,
                row_bytes: m.dim * m.grid * m.grid,
            };
            let pb = live.stage("console.stage.encode", l.rows as u64, man.row_hz());
            heads::stream_cache(&l, &mut heads, mask, man, &pb)?;
            l
        }
        None => {
            if enc_sess.is_none() {
                enc_sess = Some(b.encoder_session()?);
            }
            let sess = enc_sess.as_mut().expect("created above");
            // rects in grid cells -> pixels of the normalized clip, probed
            // per clip (the corpus is mostly but not only 16:9). Cropped
            // segments read the file's lines directly (what they are for);
            // every UNCROPPED read of a taller-than-spec file -- identity
            // segments, or the whole run without a plan -- softens to the
            // spec height inside its pipe, keeping the trained scale ratio.
            // The exposure correction rides every segment either way.
            let segs = {
                let (nw, nh) = ffmpeg::dims(&norm)?;
                let soften = autocrop::soften_arg(nh as u32, man.transcode.height);
                match &plan {
                    Some(p) => Some(
                        p.segments(nw, nh)
                            .into_iter()
                            .map(|(s, f)| {
                                let f = f.or_else(|| soften.clone());
                                (s, exposure::join_filters(photo.as_deref(), f.as_deref()))
                            })
                            .collect(),
                    ),
                    None => exposure::join_filters(photo.as_deref(), soften.as_deref())
                        .map(|f| vec![(0, Some(f))]),
                }
            };
            let pb = live.stage("console.stage.encode", n_rows, man.row_hz());
            let l = encode::encode(
                &norm, sess, mask, man, &lat_path, n_rows, max_s, segs, &pb, &mut heads,
            )?;
            cache.write_meta(&CacheMeta {
                rows: l.rows,
                dim: man.dim,
                grid: man.grid,
                basis_id: man.basis_id.clone(),
                encoder: man.encoder.clone(),
                crop: crop_key.clone(),
                norm: norm_id.clone(),
                gamma,
                decode_rev: ffmpeg::DECODE_REV,
                span_ms,
            })?;
            l
        }
    };

    // 4. the tail chunk, and the tracks composed styling reads
    let shot_edges = boundaries::shot_edges(&cuts, lat.rows, |i| man.row_ms(i));
    let tracks = heads.finish(lat.rows, &shot_edges)?;
    let s = t.elapsed().as_secs_f64();
    let rate = format!("{:.1}", dur_ms / 1000.0 / s);
    live.done("console.stage.encode", &if reused.is_some() {
        t!("console.encode.reused", took = fmt_dur(s), rate = rate)
    } else {
        t!("console.encode.done", took = fmt_dur(s), rate = rate)
    });

    // 5. composed styling

    // row -> video milliseconds, on the ROW clock
    let times: Vec<f64> = (0..lat.rows).map(|i| man.row_ms(i)).collect();
    if let Some(d) = dst.parent() {
        std::fs::create_dir_all(d)?;
    }
    let out = DraftOut {
        name: video
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        src: video.to_path_buf(),
        cuts_ms: cuts.clone(),
        norm: (!cli.no_transcode).then(|| norm.clone()),
        dst,
        tracks,
        trunk: b.trunk(),
        tracks_seed: man.env_seed,
        shelf: HashMap::new(),
        shelf_cuts: cuts,
        shot_edges,
        times,
        cache_dir: cache.dir.clone(),
        dur_ms,
        t0_ms,
        is_vr: vr_cfg.is_some(),
        // the plan is on the FRAME grid of the normalized clip, which is the
        // clock the review page's video runs on
        crop: plan.as_ref().map(|p| p.view(man.grid_fps)),
        pack: man.pack.clone(),
    };
    let (n_actions, art_spans, art_rate) = restyle(&out, man, params)?;

    // The model's own view of how well it read this clip. It is the one
    // readout that does not need a reference script, so it is what says
    // whether a change to what the encoder was SHOWN -- a crop above all --
    // helped or hurt on footage nobody has scripted.
    let conf: Vec<f64> = out.tracks.conf.iter().copied().filter(|v| v.is_finite()).collect();
    let conf_note = if conf.is_empty() {
        String::new()
    } else {
        t!("console.conf", v = format!("{:.3}", conf.iter().sum::<f64>() / conf.len() as f64))
    };
    // rarity readout (artifacts.rs): reference-free like conf, printed
    // against the authors' own rate so "too many" has a yardstick
    let art_note = if art_spans.is_empty() {
        String::new()
    } else {
        t!(
            "console.rare",
            n = art_spans.len(),
            rate = format!("{art_rate:.1}"),
            authors = format!("{:.1}", artifacts::prior().anchor_per_min),
        )
    };

    let secs = t0.elapsed().as_secs_f64();
    live.println(format!(
        "  {} {}  {}",
        style(ARROW).fg(con(th.ok)).bold(),
        style(nice(&out.dst)).fg(con(th.accent)).bold(),
        style(t!(
            "console.wrote",
            n = n_actions,
            notes = format!("{conf_note}{art_note}"),
            took = fmt_dur(secs),
            rate = format!("{:.1}", dur_ms / 1000.0 / secs),
        ))
        .fg(con(th.muted)),
    ));
    // the goblin rests: stop the animation, wipe the live line, flush the queue.
    live.finish();
    Ok(Outcome::Drafted(out))
}

/// `--cpu` against a packed-attention bundle cannot work: ONNX Runtime has no
/// CPU kernel for that layout, and the failure would otherwise land deep in the
/// encode stage as "Packed QKV of shape (B, L, N, 3, H) not implemented for
/// CPU". Refuse before the startup report claims a CPU chain that will not run.
/// The authorship seed the run decodes with: the CLI's, or the bundle's own.
///
/// The seed is not a quality knob. The flow envelope SAMPLES a stroke depth
/// rather than publishing the average of the depths it thinks are plausible,
/// and the seed picks which sample -- so a different one is a different valid
/// reading of the same video, at the same reversal times. It reaches the
/// envelope through the manifest because that is the one place both this
/// decoder and the training pipeline read it from, and it is a no-op on a
/// bundle whose envelope publishes a mean (`env_flow` false).
fn check_cpu_runnable(man: &bundle::Manifest) -> Result<()> {
    if bundle::force_cpu() && man.attn == "packed" {
        anyhow::bail!(
            "--cpu needs a bundle built for it. This one runs attention in the \
             packed layout, which is 2.4x faster on a graphics card and has no \
             CPU implementation at all.\nA CPU bundle has to be exported as one \
             (`--cpu-attn`); the shipped bundle is not it. Drop the --cpu and \
             let the goblins use the graphics card."
        );
    }
    Ok(())
}

/// What the encoder costs on this machine -- the heaviest GPU stage by far,
/// though no longer the largest slice of a draft's wall clock (the CPU-side
/// transcode ahead of it now costs as much or more).
/// DEVELOPMENT: the head alone, over latent rows that already exist.
///
/// `heads.rs` owns the chunking, the ctx padding and the envelope's stepped
/// decode, and nothing else drives them: `parity.py`'s tracks stage
/// REIMPLEMENTS that tiling in Python, and `compare_drafts.py` reaches this
/// code only end to end, through the encoder's int8 divergence. Feeding the
/// binary rows the Python pipeline already made isolates the Rust decode --
/// same rows in, tracks out, diffable row for row against the reference.
///
/// The tracks leave in the units styling reads them in: `vmarg` and `env` in
/// position units per second (already scaled by `v_std`, which rides along so
/// a normalized-unit reference can be put on the same scale), level and band
/// in 0..100, the dwell and reversal heads as probabilities. A row no chunk
/// covered is `null` for confidence and its head's neutral value elsewhere,
/// exactly as a draft would see it.
fn heads_only(b: &Bundle, lat: &Path, cuts_json: Option<&Path>, out: &Path) -> Result<()> {
    let man = &b.manifest;
    let row_bytes = man.dim * man.grid * man.grid;
    let len = std::fs::metadata(lat)
        .with_context(|| format!("no latent file at {}", lat.display()))?
        .len() as usize;
    if len == 0 || len % row_bytes != 0 {
        anyhow::bail!(
            "{} is {len} bytes, which is not a whole number of {row_bytes}-byte \
             rows (dim {} x grid {}x{}) -- these latents did not come from this \
             bundle's encoder",
            lat.display(), man.dim, man.grid, man.grid
        );
    }
    let rows = len / row_bytes;
    let cuts_ms: Vec<f64> = match cuts_json {
        Some(p) => {
            let txt = std::fs::read_to_string(p)
                .with_context(|| format!("could not read {}", p.display()))?;
            let v: serde_json::Value = serde_json::from_str(&txt)
                .with_context(|| format!("{} is not JSON", p.display()))?;
            v["cuts_ms"]
                .as_array()
                .with_context(|| format!("{} carries no cuts_ms array", p.display()))?
                .iter()
                .filter_map(|x| x.as_f64())
                .collect()
        }
        None => Vec::new(),
    };
    println!(
        "{rows} rows ({:.1} s at {:.4} rows/s), {} cut(s)",
        rows as f64 / man.row_hz(),
        man.row_hz(),
        cuts_ms.len()
    );

    let head = b.head_session()?;
    let env = b
        .env_session()?
        .context("this bundle has no envelope graph -- composed styling needs one")?;
    let cut_rows = boundaries::cut_rows(&cuts_ms, |i| man.row_ms(i));
    let mut heads = heads::Heads::new(head, env, man, cut_rows);
    let l = encode::Latents { path: lat.to_path_buf(), rows, row_bytes };
    let t = Instant::now();
    heads::stream_cache(&l, &mut heads, None, man, &ProgressBar::hidden())?;
    let shot_edges = boundaries::shot_edges(&cuts_ms, rows, |i| man.row_ms(i));
    let tr = heads.finish(rows, &shot_edges)?;
    println!("head + envelope in {:.1}s", t.elapsed().as_secs_f64());

    let mut j = serde_json::Map::new();
    j.insert("rows".into(), rows.into());
    j.insert("row_hz".into(), man.row_hz().into());
    j.insert("v_std".into(), man.v_std.into());
    j.insert("chunk".into(), man.chunk.into());
    j.insert("ctx".into(), man.ctx.into());
    let mut put = |k: &str, v: &[f64]| {
        j.insert(k.to_string(), serde_json::Value::from(v.to_vec()));
    };
    put("vmarg", &tr.vmarg);
    put("level", &tr.level);
    put("env", &tr.env);
    if !tr.conf.is_empty() {
        put("conf", &tr.conf);
    }
    if let Some((lo, hi)) = &tr.band {
        put("blo", lo);
        put("bhi", hi);
    }
    if let Some((top, bot)) = &tr.plat {
        put("plat_top", top);
        put("plat_bot", bot);
    }
    if let Some((top, bot)) = &tr.rev {
        put("rev_top", top);
        put("rev_bot", bot);
    }
    if !tr.period.is_empty() {
        put("period", &tr.period);
    }
    std::fs::write(out, serde_json::Value::Object(j).to_string())
        .with_context(|| format!("could not write {}", out.display()))?;
    println!("tracks -> {}", out.display());
    Ok(())
}

fn bench(b: &Bundle) -> Result<()> {
    let m = &b.manifest;
    println!(
        "bundle: {} (epoch {}), encoder {} @ {}px, grid {}x{}",
        m.checkpoint, m.epoch, m.encoder, m.enc_res, m.grid, m.grid
    );
    // which graphs can this machine's GPU actually take?
    for (name, r) in [
        ("transnet", b.transnet_session().map(|_| ())),
        ("head", b.head_session().map(|_| ())),
        ("env_step", b.env_session().map(|_| ())),
        ("mask", b.mask_session().map(|_| ())),
        ("encoder", b.encoder_session().map(|_| ())),
    ] {
        match r {
            Ok(()) => println!("  {name:<9} session OK"),
            Err(e) => println!("  {name:<9} FAILED: {}", format!("{e:#}").replace('\n', " ")),
        }
    }

    // The shot-cut detector runs once too: it is the other graph a GPU
    // provider can build and then refuse at the first node, and a draft
    // reaches it before the encoder.
    {
        let tn = &m.transnet;
        let shape = vec![1i64, tn.window as i64, tn.input_h as i64, tn.input_w as i64, 3];
        let n = tn.window * tn.input_h * tn.input_w * 3;
        let frames: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let mut sess = b.transnet_session()?;
        let x = ort::value::TensorRef::from_array_view((shape, &frames[..]))
            .map_err(bundle::ort_err)?;
        let out = sess.run(ort::inputs!["frames" => x]).map_err(bundle::ort_err)?;
        let (_s, prob) = out["prob"].try_extract_tensor::<f32>().map_err(bundle::ort_err)?;
        let n_bad = prob.iter().filter(|v| !v.is_finite()).count();
        if n_bad > 0 {
            anyhow::bail!("the shot-cut detector produced {n_bad} non-finite values");
        }
        println!("  transnet  forward -> {} probabilities, finite", prob.len());
    }

    let t0 = Instant::now();
    let mut sess = b.encoder_session()?;
    println!("encoder session up in {:.1}s", t0.elapsed().as_secs_f64());

    let shape = vec![1i64, m.clip_len as i64, m.enc_res as i64, m.enc_res as i64, 3];
    let n: usize = m.clip_len * m.enc_res * m.enc_res * 3;
    let frames: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();

    for i in 0..5 {
        let t = Instant::now();
        let x = ort::value::TensorRef::from_array_view((shape.clone(), &frames[..]))
            .map_err(bundle::ort_err)?;
        let out = sess
            .run(ort::inputs!["frames" => x])
            .map_err(bundle::ort_err)?;
        let (lat_shape, lat) = out["lat"]
            .try_extract_tensor::<f32>()
            .map_err(bundle::ort_err)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        // Latents are whitened: ~unit variance, finite, by construction. A
        // backend that returns NaN (or zeros) here still "runs" -- and the draft
        // it produces is flat. So the bench reads the values, not just the clock.
        let n_bad = lat.iter().filter(|v| !v.is_finite()).count();
        let mean = lat.iter().filter(|v| v.is_finite()).sum::<f32>() / lat.len() as f32;
        let var = lat.iter().filter(|v| v.is_finite())
            .map(|v| (v - mean).powi(2)).sum::<f32>() / lat.len() as f32;
        if n_bad > 0 || var.sqrt() < 0.1 {
            anyhow::bail!(
                "the encoder produced {n_bad} non-finite values and std {:.4}                  (expected ~1.0) -- this GPU backend is miscomputing the graph",
                var.sqrt()
            );
        }
        // a group of clip_len frames costs TWO forwards (one per alignment) and
        // yields one row per frame, so this is the rate a draft actually runs at
        let fps = m.clip_len as f64 / (2.0 * ms / 1000.0);
        println!(
            "  forward -> {:?}  {:>7.0} ms  = {:>5.0} f/s = {:.1}x realtime               (latent std {:.2}){}",
            lat_shape,
            ms,
            fps,
            fps / m.grid_fps,
            var.sqrt(),
            if i == 0 { "  (warmup)" } else { "" },
        );
    }
    Ok(())
}

/// Build a throttle config from whatever `--throttle-*` flags this run carries
/// and arm it, or do nothing if the command line asks for no throttle.
///
/// Split out of `main` because the picker needs the same validation on a
/// DIFFERENT source: its preset comes from `settings.json`, and the two paths
/// must agree on what a good config is or a value the picker accepts would
/// panic in `rest_after` where a CLI value would have been rejected up front.
fn arm_throttle_from_cli(cli: &Cli) -> anyhow::Result<()> {
    let ov = throttle::Overrides {
        gpu_target: cli.throttle_gpu_target,
        interval: cli.throttle_interval,
        avg_window: cli.throttle_avg_window,
        kp: cli.throttle_kp,
        deadzone: cli.throttle_deadzone,
        duty_min: cli.throttle_duty_min,
        duty_start: cli.throttle_duty_start,
        rise_limit: cli.throttle_rise_limit,
        fall_limit: cli.throttle_fall_limit,
        dwell: cli.throttle_dwell,
        spike_margin: cli.throttle_spike_margin,
        spike_cut_max: cli.throttle_spike_cut_max,
        spike_hold: cli.throttle_spike_hold,
        script_max: cli.throttle_script_max,
        smooth: cli.throttle_smooth,
        grace: cli.throttle_grace,
    };
    // `--throttle-script-max` on its own is a request to throttle, with the
    // balanced preset and only that clamp changed.
    if cli.no_throttle && (cli.throttle.is_some() || cli.throttle_script_max.is_some()) {
        anyhow::bail!("--no-throttle conflicts with --throttle / --throttle-script-max");
    }
    let want = !cli.no_throttle && (cli.throttle.is_some() || cli.throttle_script_max.is_some());
    if !want {
        return Ok(());
    }
    let name = cli.throttle.as_deref().unwrap_or("balanced");
    arm_throttle(name, &ov)
}

/// Validate `ov`-overridden `name` and install it. The one place a config
/// becomes a running throttle, so `--throttle` and the picker's P key cannot
/// drift apart on what is legal.
fn arm_throttle(name: &str, ov: &throttle::Overrides) -> anyhow::Result<()> {
    let cfg = throttle::parse(name, ov)?;
    if !(1.0..=100.0).contains(&cfg.gpu_target) {
        anyhow::bail!("throttle gpu_target {} is out of range (1..100)", cfg.gpu_target);
    }
    if cfg.duty_min < 1.0 || cfg.duty_min > 100.0 {
        anyhow::bail!("throttle duty_min {} is out of range (1..100)", cfg.duty_min);
    }
    if cfg.duty_start < 1.0 || cfg.duty_start > 100.0 {
        anyhow::bail!("throttle duty_start {} is out of range (1..100)", cfg.duty_start);
    }
    // A zero or negative cap would be a `delta` that cannot move, and a
    // `script_max` of 0 a duty clamped straight to the atomic's floor; both are
    // legal f32 but not legal controls.
    if cfg.script_max <= 0.0 || cfg.script_max > 100.0 {
        anyhow::bail!("throttle script_max {} is out of range (1..100]", cfg.script_max);
    }
    if cfg.rise_limit <= 0.0 || cfg.fall_limit <= 0.0 {
        anyhow::bail!(
            "throttle step caps must be positive (rise {}, fall {})",
            cfg.rise_limit,
            cfg.fall_limit
        );
    }
    if cfg.avg_window <= 0.0 {
        anyhow::bail!("throttle avg_window {} must be positive seconds", cfg.avg_window);
    }
    if cfg.interval <= 0.0 {
        anyhow::bail!("throttle interval {} must be positive seconds", cfg.interval);
    }
    throttle::install(true, cfg);
    Ok(())
}

fn main() -> std::process::ExitCode {
    use std::io::Write as _;
    // A double-clicked exe (no args) gets the picker -- and a pause before the
    // console window vanishes, whether the run succeeded or died on the spot.
    let interactive = std::env::args_os().len() == 1 && std::io::stdout().is_terminal();

    // Name the terminal tab, saving whatever it was so exit restores it. The
    // XTerm/Windows Terminal title stack does this: 22;2t pushes the current
    // window title, 23;2t (below) pops it back. Terminal-only, so no escape ever
    // reaches piped output.
    let own_title = std::io::stdout().is_terminal();
    if own_title {
        let mut out = std::io::stdout();
        let _ = write!(out, "\x1b[22;2t");
        let _ = ratatui::crossterm::execute!(
            out,
            ratatui::crossterm::terminal::SetTitle("GoblinScript")
        );
    }

    let code = match real_main() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        // Stopping is not failing: no red line, no error buzz, and the working
        // files stay for the resume. 130 is what a shell expects back from a
        // run its user interrupted.
        Err(e) if cancel::is_cancel(&e) => {
            let t = theme();
            println!(
                "\n{} {}",
                style(SKIP).fg(con(t.warn)).bold(),
                style(t!("console.stopped")).fg(con(t.muted))
            );
            std::process::ExitCode::from(130)
        }
        // A failure that reaches here ended the run, so it is the last thing
        // the window shows -- and on a double-clicked exe that window is about
        // to close. The log is the copy that survives it.
        Err(e) => {
            let t = theme();
            eprintln!(
                "{} {}",
                style(CROSS).fg(con(t.bad)).bold(),
                style(format!("{e:#}")).fg(con(t.bad))
            );
            if let Some(p) = errlog::record_fatal(&e) {
                eprintln!(
                    "  {}",
                    style(t!("console.failed.log", path = nice(&p))).fg(con(t.muted))
                );
            }
            sound::play_error();
            std::process::ExitCode::FAILURE
        }
    };
    if interactive {
        print!("\npress Enter to close ");
        let _ = std::io::stdout().flush();
        let mut s = String::new();
        let _ = std::io::stdin().read_line(&mut s);
    }
    // Restore the tab title the app was launched with -- the very last thing, so
    // it stays "GoblinScript" through the close prompt above.
    if own_title {
        let mut out = std::io::stdout();
        let _ = write!(out, "\x1b[23;2t");
        let _ = out.flush();
    }
    code
}

/// Which language to open in, from the three places that can say: the command
/// line, the remembered choice, then the machine's own. Returns the tag `--lang`
/// asked for and could not be given, which is the one case worth a complaint --
/// a remembered or system tag whose file has since gone is not something the
/// user asked for this time.
///
/// Called before the command line is fully parsed, because `--help` is written
/// in whatever this decides and clap builds that text as it parses.
fn choose_language(argv: &[String]) -> Option<String> {
    let asked = argv.iter().enumerate().find_map(|(i, a)| {
        a.strip_prefix("--lang=")
            .map(str::to_string)
            .or_else(|| (a == "--lang").then(|| argv.get(i + 1).cloned()).flatten())
    });
    if let Some(tag) = asked {
        return (!lang::set(&tag)).then_some(tag);
    }
    let remembered = settings::Settings::load().lang;
    if let Some(tag) = remembered.as_deref() {
        lang::set(tag);
    } else if let Some(tag) = lang::system_tag() {
        lang::set(&tag);
    }
    None
}

/// Clap's command with every piece of text the catalog has an opinion about
/// replaced. An argument the active language has not translated keeps the
/// English from its own doc comment, per key -- the same rule as everywhere
/// else, so a half-finished translation is a half-translated help screen rather
/// than a broken one.
///
/// Clap's own furniture ("Usage:", "Options:") is not reachable from here and
/// stays English.
fn localized_command() -> clap::Command {
    let mut cmd = Cli::command();
    if let Some(s) = lang::try_t("cli.about") {
        cmd = cmd.about(s);
    }
    let ids: Vec<clap::Id> = cmd.get_arguments().map(|a| a.get_id().clone()).collect();
    for id in ids {
        let help = lang::try_t(&format!("cli.{id}"));
        let long = lang::try_t(&format!("cli.{id}.long"));
        if help.is_none() && long.is_none() {
            continue;
        }
        cmd = cmd.mut_arg(id, |a| {
            let a = match help {
                Some(s) => a.help(s),
                None => a,
            };
            // A doc comment's later paragraphs are already the arg's long help,
            // and clap shows THAT for `--help`. So a translation that gives only
            // the short form replaces both, or `--help` would answer in English
            // while `-h` answered in the user's language.
            match long.or(help) {
                Some(s) => a.long_help(s),
                None => a,
            }
        });
    }
    cmd
}

fn real_main() -> Result<()> {
    // `args_os`, not `args`: the latter is a `String` iterator that PANICS on an
    // argument the platform allows but Unicode does not -- a filename in a
    // legacy codepage, a lone surrogate out of a Windows shell. The only thing
    // read here is a `--lang` tag, so a lossy reading of the rest costs nothing,
    // and clap parses the real arguments from the OS strings itself.
    let argv: Vec<String> =
        std::env::args_os().map(|a| a.to_string_lossy().into_owned()).collect();
    let bad_lang = choose_language(&argv);
    let mut cli = Cli::from_arg_matches_mut(&mut localized_command().get_matches())
        .unwrap_or_else(|e| e.exit());

    // A translation that is on disk and did not load, named before it can be
    // missed. The usual cause is the editor rather than the translation -- a
    // byte-order mark, a codepage, a comma too many -- and each of those is one
    // save away from fixed, but only for someone who knows which file to open.
    for bad in lang::unreadable() {
        eprintln!(
            "  {} {}",
            style("!").yellow().bold(),
            style(format!("languages/{bad} -- not loaded, so that language is not offered")).dim()
        );
    }

    // Said before anything else this run might do, including the paths that
    // return early: the user asked for a language and did not get it, which is
    // true whatever else they asked for. The two lines that fix it are here
    // because the fix needs no build and is entirely theirs to make.
    if let Some(tag) = bad_lang {
        let have: Vec<&str> = lang::available().iter().map(|c| c.code.as_str()).collect();
        eprintln!(
            "  {} {}",
            style("!").yellow().bold(),
            style(format!(
                "no language file for {tag:?} -- installed: {}. Copy languages/en-US.json to \
                 languages/{tag}.json and translate the right-hand side of every line.",
                have.join(", ")
            ))
            .dim()
        );
    }

    // Before the ffmpeg check: the soundtrack has nothing to do with video, and
    // a levelling pass should not need a media toolchain on PATH.
    if cli.music_levels {
        sound::print_levels();
        return Ok(());
    }

    // Handing over bytes that are already inside the exe needs none of what
    // follows: no ffmpeg, no graphics card, no cache. Whoever runs this is
    // usually standing in front of a build from source that cannot draft yet,
    // on a machine that may not even be the one that will do the drafting.
    //
    // English, like the log and for the same reason: this is a technical
    // handover meant to be read beside a build, and the paths in it are going
    // straight into somebody's command line.
    if let Some(dir) = cli.dump_bundle.clone() {
        let wrote = bundle::dump(&dir)?;
        let t = theme();
        println!(
            "{} {}",
            style(TICK).fg(con(t.ok)).bold(),
            style(format!("the goblins are out, in {}", dir.display())).bold()
        );
        for (name, len) in &wrote {
            let size = match *len {
                n if n >= 1_000_000 => format!("{:.1} MB", n as f64 / 1e6),
                n if n >= 1_000 => format!("{:.1} KB", n as f64 / 1e3),
                n => format!("{n} B"),
            };
            println!("       {:<18} {:>9}", name, style(size).fg(con(t.muted)));
        }
        println!(
            "       {}",
            style(format!("--bundle \"{}\" drafts with them", dir.display()))
                .fg(con(t.muted))
        );
        return Ok(());
    }

    // The optional GPU throttle. Armed here -- after every early-return path
    // that does no drafting (`--music-levels`, `--dump-bundle`, `--vr-only`)
    // and before any encode can start -- from the CLI flags when there are any,
    // and otherwise left to the picker, which reads its setting from
    // `settings.json` below. A run that asks for neither is byte-identical to a
    // build without any of this.
    arm_throttle_from_cli(&cli)?;

    ffmpeg::have_tools()?;

    // A folder stands for the videos in it from here on, so nothing below --
    // the picker decision, --vr-only, the skip rule, the batch loop -- ever
    // sees one. Before the picker decision in particular: a folder IS a batch,
    // and opening the picker on it would answer a command with a question.
    expand_dirs(&mut cli)?;

    // Aiming needs ffmpeg and nothing else -- no bundle, no GPU. Before every
    // check below, which are all about styling a draft this run will not make.
    if cli.vr_only {
        if cli.videos.is_empty() {
            anyhow::bail!("--vr-only needs videos: goblinscript --vr-only <VIDEO>...");
        }
        // aiming a link means fetching it first: the projection is measured off
        // the source frames, and there are none until the file is here
        resolve_links(&mut cli)?;
        let cache_root = cli.cache.clone().unwrap_or_else(|| PathBuf::from("cache"));
        let vrs = vr_prep(&cli, &cli.videos.clone(), &cache_root, &BTreeSet::new())?;
        for v in &cli.videos {
            match vrs.get(v) {
                Some(c) if c.skip => println!("{}: not VR (skipped)", nice(v)),
                Some(c) => println!(
                    "{}: {} yaw {:.1} pitch {:.1} hfov {} range {:?} -> {}",
                    nice(v),
                    c.layout,
                    c.aim.yaw,
                    c.aim.pitch,
                    c.h_fov,
                    c.range_ms,
                    c.filter_prefix(480)
                ),
                None => println!("{}: flat", nice(v)),
            }
        }
        return Ok(());
    }

    if !(crate::style::INTENSITY_MIN..=crate::style::INTENSITY_UNLOCKED_100).contains(&cli.intensity) {
        anyhow::bail!(
            "--intensity {} is out of range ({}..{})",
            cli.intensity,
            crate::style::INTENSITY_MIN,
            crate::style::INTENSITY_UNLOCKED_100
        );
    }
    if cli.max_speed < 0.0 {
        anyhow::bail!("--max-speed must be >= 0");
    }
    if cli.cut_ease < 0.0 {
        anyhow::bail!("--cut-ease must be >= 0");
    }
    if let Some(pk) = cli.dwell_ramp {
        if !(0.0..=1.0).contains(&pk) {
            anyhow::bail!("--dwell-ramp {pk} is out of range (0.0..1.0)");
        }
    }
    if let Some(se) = cli.still_eps {
        if !(0.0..=60.0).contains(&se) {
            anyhow::bail!("--still-eps {se} is out of range (0.0..60.0)");
        }
    }
    if let Some(dose) = cli.depth_dose {
        if !(0.0..=1.0).contains(&dose) {
            anyhow::bail!("--depth-dose {dose} is out of range (0.0..1.0)");
        }
    }
    if let Some(wsec) = cli.depth_window {
        if !(0.0..=60.0).contains(&wsec) {
            anyhow::bail!("--depth-window {wsec} is out of range (0.0..60.0 s)");
        }
    }
    let mut params = style::Params {
        style: cli.style,
        dwells: cli.dwells,
        stillness: cli.stillness,
        intensity: cli.intensity,
        range: cli.range,
        max_speed: cli.max_speed,
        cut_ease: cli.cut_ease,
        dwell_ramp: cli.dwell_ramp,
        still_eps: cli.still_eps,
        depth: cli.depth_uniformity,
        depth_dose: cli.depth_dose,
        depth_window: cli.depth_window,
        filler_min_real_s: cli.filler_min_real,
        filler_real_v: cli.filler_real_v,
        filler_model_w: cli.filler_model_w,
        filler_ramp_s: cli.filler_ramp,
        filler_max_bridge_s: cli.filler_max_bridge,
        filler_sway: cli.filler_sway,
        filler_sway_s: cli.filler_sway_s,
        ..style::Params::default()
    };
    // the preset writes its bundle, then any expert number overrides it
    cli.filler.apply(&mut params);
    if let Some(v) = cli.filler_gap {
        params.filler_gap_s = v;
    }
    if let Some(v) = cli.filler_rate {
        params.filler_rate = v;
    }
    if let Some(v) = cli.filler_amp {
        params.filler_amp = v;
    }
    if let Some(v) = cli.filler_pattern {
        params.filler_pattern = v;
    }
    if let Some(v) = cli.filler_burst {
        params.filler_burst = v;
    }
    if let Some(v) = cli.filler_rest {
        params.filler_rest_s = v;
    }
    if params.filler_gap_s < 0.0 {
        anyhow::bail!("--filler-gap must be >= 0 (seconds; 0 = off)");
    }
    if params.filler_gap_s > 0.0 && !(1.0..=300.0).contains(&params.filler_rate) {
        anyhow::bail!("--filler-rate {} is out of range (1..300 strokes/min)", params.filler_rate);
    }
    if params.filler_gap_s > 0.0 && !(1.0..=50.0).contains(&params.filler_amp) {
        anyhow::bail!("--filler-amp {} is out of range (1..50 units)", params.filler_amp);
    }
    if params.filler_min_real_s < 0.0
        || params.filler_real_v < 0.0
        || params.filler_ramp_s < 0.0
        || params.filler_max_bridge_s < 0.0
        || params.filler_rest_s < 0.0
    {
        anyhow::bail!("filler durations and the confidence bar must be >= 0");
    }
    if !(0.0..=1.0).contains(&params.filler_sway) || params.filler_sway_s <= 0.0 {
        anyhow::bail!("--filler-sway must be 0..1 and --filler-sway-s > 0");
    }
    if !(0.0..=1.0).contains(&params.filler_model_w) {
        anyhow::bail!("--filler-model-w must be 0..1 (1 = full model influence)");
    }
    if params.filler_burst == 0 {
        anyhow::bail!("--filler-burst must be >= 1");
    }

    // Picker (double-clicked) mode is a SESSION: draft a batch, review it, then
    // the picker reopens for the next drop of videos -- the app closes only when
    // the user quits the picker (Q). A command line that names videos runs that
    // batch once and exits. The chosen style carries from one batch to the next.
    let picker_mode = cli.videos.is_empty() && !cli.serve && std::io::stdout().is_terminal();
    let show_intro =
        picker_mode && !cli.bench && std::env::var_os("GOBLINSCRIPT_NO_INTRO").is_none();

    // The picker reopens where it was left -- across launches (settings.json
    // beside the exe), and within a session -- and carries the last batch's
    // outcome forward as a status line, so an instant, all-skipped start is not
    // silent. Loaded FIRST because it also decides how everything below looks
    // and sounds, startup report included.
    let mut settings = settings::Settings::load();
    theme::set(cli.theme.or(settings.theme).unwrap_or(theme::Palette::Phosphor));
    sound::set_muted(cli.mute || settings.muted);
    sound::set_volume(cli.volume.or(settings.volume).unwrap_or(sound::Volume::Normal));
    // The flag is one-way ON, like `--music`: someone who wants it OFF says so
    // in the picker's W, which is the only place a double-clicked exe's user
    // can reach it at all. Written back out below with the other choices.
    if cli.gpu_browser {
        settings.gpu_browser = true;
    }
    // Decided here, started later (after the boot sequence): the music only
    // plays where somebody is listening, so never on a piped run and never when
    // the whole audio layer is muted.
    //
    // And only where somebody OPENED the app rather than called it. The picker
    // is a place you sit -- the goblins put a record on, and M turns it off for
    // good. `goblinscript video.mp4` is a command in someone's terminal, very
    // possibly inside a script or beside something else they are listening to,
    // and a command that starts playing music is a command with a bug. That run
    // is silent unless `--music` asks, and the picker's remembered ON does not
    // leak into it.
    // The picker drafts with the model last chosen on its review page; a
    // named-video run takes the bundle's default unless `--model` says
    // otherwise, like every other remembered preference.
    let remembered_model = if picker_mode { settings.model.clone() } else { None };
    let want_music = (cli.music || (picker_mode && settings.music))
        && !sound::muted()
        && std::io::stdout().is_terminal();

    // What the throttle will actually be, resolved ONCE for the report below.
    // Precedence: `--no-throttle` > a `--throttle*` flag > the picker's
    // remembered choice. A run that was CALLED (a named video, a script) does
    // not read the file -- same rule as `music`, and for the same reason: a
    // command should not silently change behaviour because somebody pressed a
    // key in the picker last week.
    let throttle_name: Option<String> = if cli.no_throttle {
        None
    } else if cli.throttle.is_some() || cli.throttle_script_max.is_some() {
        Some(cli.throttle.clone().unwrap_or_else(|| "balanced".into()))
    } else if picker_mode {
        settings.throttle.clone()
    } else {
        None
    };
    // The picker's own choice is not on any command line, so it is armed here.
    // (A CLI run was already armed above, before `settings` had been read --
    // `install` is one-shot, and `reconfigure` is what reaches an installed
    // loop, so either path lands on the same place.)
    //
    // A stale or hand-edited name this build does not know is not worth a hard
    // failure on a double-click: say so and draft unthrottled, which is the
    // state the user was in anyway.
    if picker_mode {
        match throttle_name.as_deref() {
            None => {}
            Some(name) => match throttle::parse(name, &throttle::Overrides::default()) {
                Ok(cfg) => throttle::reconfigure(Some(cfg)),
                Err(e) => eprintln!("[throttle] {e}; drafting unthrottled"),
            },
        }
    }

    let cache_root = cli.cache.clone().unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("cache")))
            .unwrap_or_else(|| PathBuf::from("cache"))
    });

    // Before anything builds a session -- including the loader thread below and
    // the POST line that reports which chain it will walk.
    bundle::set_force_cpu(cli.cpu);
    bundle::pin_attention_kernel();

    // Startup runs in boot order: self-test, then the demo, then the picker --
    // the machine wakes up before it shows off.
    //
    // Loading the ONNX bundle is the slow part of startup, so in picker mode it
    // goes on a background thread and the POST prints over it. The report's own
    // pacing covers the load: the host lines need nothing from the bundle, and
    // the device lines join the loader first -- so a slow load reads as the
    // report enumerating devices, which is what a POST looks like anyway. The
    // intro then plays with everything already in hand.
    let mut b = if show_intro {
        let bundle_arg = cli.bundle.clone();
        let loader = std::thread::spawn(move || load_bundle(bundle_arg.as_deref()));
        let post = (!cli.quiet).then(|| {
            let p = bios::Post::begin(true);
            bios::system(&p, &cache_root);
            bios::throttle(&p, throttle_name.as_deref());
            p
        });
        let b = loader
            .join()
            .map_err(|_| anyhow::anyhow!("the bundle-loading goblin panicked"))??;
        let b = pick_model(b, &cli, remembered_model.as_deref())?;
        check_cpu_runnable(&b.manifest)?;
        b.probe_gpu()?;
        let b = b.with_seed(cli.env_seed);
        if let Some(p) = &post {
            bios::devices(p, &b.manifest);
            bios::ready(p);
        }
        tui::intro()?;
        b
    } else {
        let b = pick_model(load_bundle(cli.bundle.as_deref())?, &cli, remembered_model.as_deref())?;
        check_cpu_runnable(&b.manifest)?;
        b.probe_gpu()?;
        let b = b.with_seed(cli.env_seed);
        if !cli.quiet {
            // no animation here: this user is driving a tool, not booting a
            // machine, and the report is a header rather than a ceremony
            let p = bios::Post::begin(false);
            bios::system(&p, &cache_root);
            bios::throttle(&p, throttle_name.as_deref());
            bios::devices(&p, &b.manifest);
            bios::ready(&p);
        }
        b
    };

    if cli.bench {
        return bench(&b);
    }

    if let Some(p) = &cli.from_latents {
        return heads_only(&b, p, cli.cuts_json.as_deref(), &cli.dump_tracks);
    }

    if cli.serve {
        return serve::serve(cli, b, cache_root, params);
    }

    // The record goes on once the boot sequence is over -- after the startup
    // report and, in picker mode, after the intro. The intro has its own chime
    // playing over it, and two pieces of music at once is neither of them.
    if want_music {
        sound::set_music(true);
    }

    // The picker opens where `dir_mode` says: the remembered directory by
    // default, or the exe's own folder when the user pinned it there. The
    // exe's folder is the fallback, not the working directory -- a
    // double-clicked exe is handed an unrelated cwd by Explorer.
    let mut start_dir = settings.picker_start_dir();
    // In the picker (double-clicked, no flags) the last session's accepted style
    // is the starting point -- there is no command line to carry it otherwise.
    if picker_mode {
        if let Some(p) = settings.last_params {
            params = p;
        }
    }
    // What the batch that just ran left behind: the one-line outcome, and every
    // failure in full. Both belong to the PICKER rather than to the console --
    // it reopens into the alternate screen the moment a batch ends, and takes
    // whatever was printed away with it.
    let mut report = tui::Report::default();
    // videos the picker's user marked VR by hand, for THIS batch: replaced on
    // every trip round the loop, since the next batch is a new set of answers
    let mut vr_marks: BTreeSet<PathBuf> = BTreeSet::new();
    loop {
        if picker_mode {
            match tui::pick(
                cli.force,
                // auto-crop is on unless this launch or a past batch said no
                !cli.no_autocrop && settings.autocrop,
                // and so is the crop page it feeds
                !cli.no_crop_edit && settings.crop_edit,
                // the throttle, as the picker last left it (CLI flags win and
                // were armed above, but the picker still shows and re-saves
                // what it would use)
                throttle_name.clone(),
                // the auto-suggest switch, as the picker last left it: the page
                // reads it out of the settings file, so the picker only decides
                settings.auto_suggest,
                // and the GPU browser, on the same terms
                settings.gpu_browser,
                start_dir.clone(),
                std::mem::take(&mut report),
                cli.dl_dir.clone(),
                cli.dl_args.clone(),
            )? {
                Some(p) => {
                    cli.videos = p.videos;
                    cli.force = p.force;
                    // the picker's switch is the answer for this batch, as a
                    // PREFERENCE rather than a demand: switched on against a
                    // bundle with no mask net it skips the crop and drafts,
                    // where the explicit flag would refuse
                    cli.no_autocrop = !p.autocrop;
                    cli.no_crop_edit = !p.crop_edit;
                    vr_marks = p.vr;
                    start_dir = p.dir.clone();
                    // remember where videos were processed from, how the
                    // picker was left looking and sounding, and whether the
                    // batch was auto-cropped, for next launch
                    settings.last_dir = p.dir.as_ref().map(|d| d.display().to_string());
                    settings.autocrop = p.autocrop;
                    settings.crop_edit = p.crop_edit;
                    // The throttle, as the picker left it. Same terms as the
                    // switches above: a preference, remembered for next launch
                    // AND applied now, because the picker's user has no command
                    // line to apply it with. `reconfigure` reaches the loop the
                    // CLI already installed, or installs one the picker just
                    // asked for, or turns it off.
                    if p.throttle != settings.throttle {
                        match p.throttle.as_deref() {
                            None => throttle::disable(),
                            Some(name) => match throttle::parse(name, &throttle::Overrides::default())
                            {
                                Ok(cfg) => throttle::reconfigure(Some(cfg)),
                                // a name this build does not know (hand-edited
                                // file): draft unthrottled rather than refuse a
                                // batch over an optional feature
                                Err(e) => eprintln!("[throttle] {e}; drafting unthrottled"),
                            },
                        }
                    }
                    settings.throttle = p.throttle.clone();
                    // The confidence switch, on the same terms as the throttle:
                    // a standing preference, remembered and handed to the page
                    // through the settings file. Unlike the throttle there is
                    // nothing to arm here -- the suggestion runs in the browser,
                    // which reads this on its next load.
                    settings.auto_suggest = p.auto_suggest;
                    // The GPU browser window, on the same terms again: a
                    // standing preference, acted on the next time a page opens.
                    settings.gpu_browser = p.gpu_browser;
                    settings.remember_presentation();
                    settings.save();
                    // the picker's users have no flags to re-run with -- the
                    // review loop IS their parameter surface
                    cli.review = true;
                }
                None => {
                    // quit from the picker: the session is over, but a theme or
                    // audio change made on the way out still has to survive it
                    settings.remember_presentation();
                    settings.save();
                    return Ok(());
                }
            }
        } else if cli.videos.is_empty() {
            anyhow::bail!("feed the goblin a video: goblinscript <VIDEO>");
        }

        // Everything the batch needs before any of it runs: links become files,
        // so nothing below has to know the difference (the picker fetches its
        // own, so that is a no-op on this path), and every VR question is asked
        // and answered -- see `vr_prep`. A flat batch does nothing there but one
        // ffprobe per video.
        //
        // Run as one fallible step because they FAIL as one: an unreachable
        // link or an unreadable video is this batch's failure, and in a session
        // the user opened by double-clicking, a batch failing is not the app
        // closing. On a command line it still is -- that run asked for these
        // videos and nothing else.
        let prep = (|| -> Result<VrMap> {
            resolve_links(&mut cli)?;
            vr_prep(&cli, &cli.videos, &cache_root, &vr_marks)
        })();
        let vrs = match prep {
            Ok(v) => v,
            Err(e) if cancel::is_cancel(&e) || !picker_mode => return Err(e),
            Err(e) => {
                // English, like every other name in the log and like the
                // error's own words: this one names a step rather than a file.
                let f = errlog::Failure::of("preparing the batch", &e);
                report = tui::Report {
                    status: Some(format!(
                        "{}  {}",
                        t!("picker.status.failed", n = 1),
                        t!("picker.status.errors")
                    )),
                    log: errlog::record(&f),
                    failures: vec![f],
                };
                say_failed(&e, report.log.as_deref());
                sound::play_error();
                cli.videos.clear();
                continue;
            }
        };

        // One GPU job at a time: videos run in sequence, never in parallel. The
        // CPU-side transcode is the exception, and only ever one video ahead --
        // it uses hardware the drafting video is not using (see `prefetch`).
        let (mut drafted, mut skipped, mut failed) = (0, 0, 0);
        let mut outs: Vec<DraftOut> = Vec::new();
        // Every failure in full, and where they were written down. The picker
        // shows both back; a run from a command line has its console and needs
        // neither, but pays a `Vec` that stays empty for the simpler code.
        let mut failures: Vec<errlog::Failure> = Vec::new();
        let mut log: Option<PathBuf> = None;
        let pf = prefetch::Prefetch::new(!cli.no_transcode && !cli.no_prefetch, cli.hwaccel);
        // The picker's header, pinned over the log for the length of the batch:
        // the app opened as an app, so it keeps its goblins while it works. A
        // run that named videos on a command line keeps the plain console --
        // that terminal belongs to whoever typed into it, and taking it over to
        // animate a mascot is not a thing a command does.
        let header = picker_mode.then(chrome::Header::begin).flatten();
        for (i, v) in cli.videos.iter().enumerate() {
            // the next video that will actually be drafted -- one already
            // scripted is skipped, and normalizing it would be pure waste
            let next = cli.videos[i + 1..]
                .iter()
                .find(|n| !will_skip(&cli, n))
                .map(|n| n.as_path());
            // The chosen model first, with the next video's transcode in its
            // GPU-bound shadow; then every other `--models` model over the
            // same encode as a script of its own. A review switches models on
            // the page instead, so there every model's head pass runs onto the
            // one draft's shelf below.
            let mut runs: Vec<(Cli, Bundle, Option<&Path>)> =
                vec![(cli.clone(), b.select(&b.manifest.pack)?, next)];
            for m in cli.models.iter().filter(|m| m.as_str() != b.manifest.pack) {
                let mut c2 = cli.clone();
                c2.out_suffix = Some(m.clone());
                runs.push((c2, b.select(m)?, None));
            }
            // The working files of a drafted video go once every model has
            // read them: a second model over the same encode needs the cache
            // the first one would otherwise have swept.
            let mut sweep: Vec<PathBuf> = Vec::new();
            for (c, bb, nx) in &runs {
                match draft(c, bb, v, &cache_root, &params, i, cli.videos.len(), &pf, *nx, &vrs, None)
                {
                    Ok(Outcome::Drafted(mut out)) => {
                        drafted += 1;
                        if cli.review {
                            if let Err(e) = pretrack(&mut out, bb, &params) {
                                if cancel::is_cancel(&e) {
                                    return Err(e);
                                }
                                // the draft stands; a model that could not be
                                // prepared is prepared on the page instead
                                eprintln!("  {} {e:#}", style("!").yellow().bold());
                            }
                            outs.push(out); // cache + tracks live until the review closes
                        } else if !cli.keep_cache {
                            // working files exist for resume-after-interrupt;
                            // the draft is done, so they are done too (a
                            // failure never reaches this arm and keeps them)
                            sweep.push(out.cache_dir.clone());
                        }
                    }
                    Ok(Outcome::Skipped) => skipped += 1,
                    // The user stopped this one, so they have stopped the
                    // batch -- carrying on to the next video is the opposite
                    // of what Ctrl-C asked for.
                    Err(e) if cancel::is_cancel(&e) => return Err(e),
                    Err(e) => {
                        // Recorded before it is printed: the console is about
                        // to be taken back by the picker, and the log and the
                        // report it reads are what outlive that.
                        let f = errlog::Failure::of(&nice(v), &e);
                        log = errlog::record(&f).or(log);
                        failures.push(f);
                        say_failed(&e, log.as_deref());
                        eprintln!(
                            "  {}",
                            style(t!("console.failed.kept")).fg(con(theme().muted))
                        );
                        sound::play_error();
                        failed += 1;
                    }
                }
            }
            sweep.sort();
            sweep.dedup();
            for d in sweep {
                let _ = std::fs::remove_dir_all(&d);
            }
        }
        // The batch is over, so the screen goes back: the summary, the review
        // and the picker that follows it all want the whole of it.
        drop(header);
        if cli.videos.len() > 1 {
            let parts: Vec<String> = [
                (drafted, "console.sum.drafted"),
                (skipped, "console.sum.skipped"),
                (failed, "console.sum.failed"),
            ]
            .iter()
            .filter(|(n, _)| *n > 0)
            .map(|(n, k)| lang::fill(lang::t(k), &[("n", &n.to_string())]))
            .collect();
            println!("\n{}", style(parts.join(", ")).fg(con(theme().accent)).bold());
        }
        // A finished-processing chime, once per batch that actually drafted
        // something. (`sound` itself declines on a piped run -- there is no one
        // listening, and the process may exit before it plays.)
        if drafted > 0 {
            sound::play_done();
        }

        if cli.review && !outs.is_empty() {
            // The review page plays the video, and the script is judged against
            // what it sounds and looks like. Held across the terminal fallback
            // too -- and past the `?` in it, which is why this is a guard and
            // not a pair of calls.
            let _quiet = sound::pause_music();
            let man = &b.manifest;
            let clips: Vec<review::Clip> =
                outs.iter().map(|o| review_clip(o, &b)).collect::<Result<_>>()?;
            // one style per script -- reviewing is script-by-script
            let mut clip_params: Vec<style::Params> = vec![params; clips.len()];
            let mut choices: Vec<review::Choice> = clips
                .iter()
                .map(|c| review::Choice { model: c.model.clone(), compare: None, spans: Vec::new(), cuts_ms: None })
                .collect();
            // the manifest's "normal" values, so the page can show every preset's
            // number in brackets and seed the expert inputs
            let presets = review::Presets {
                dwell_ramp: man.plat_soft[0],
                still_eps: man.still_eps,
            };
            // the seed and the trunk reach the HEAD stage, upstream of styling,
            // so a clip whose seed or model moved onto tracks it has not made
            // runs the head pass over its own latents before it is styled.
            // `retrack` is a no-op for every turn of every other knob.
            // a Mutex rather than a RefCell: the styler runs on its own
            // thread, so the closure has to be Send
            let outs_cell = std::sync::Mutex::new(&mut outs);
            // The hint is a claim about what just happened, so it is chosen by
            // what will actually happen: `--no-browser` prints the address
            // without pretending a window appeared.
            let opened = !cli.no_browser;
            let page = move |url: &str| {
                println!(
                    "\n  {} {}",
                    t!("console.review.label"),
                    style(url).cyan().bold()
                );
                let hint = if opened {
                    t!("console.review.hint")
                } else {
                    t!("console.review.manual")
                };
                println!("  {}", style(hint).dim());
            };
            if let Err(e) = review::review(
                &clips,
                &mut clip_params,
                &mut choices,
                presets,
                !cli.no_browser,
                &page,
                |i, p, choice| {
                    let mut outs = outs_cell.lock().unwrap();
                    review_style(&mut outs[i], &b, p, choice, &clips[i].compare)
                },
            ) {
                // no server, no browser -- the terminal review screen still works
                eprintln!(
                    "  {} {}",
                    style("!").yellow().bold(),
                    style(t!("console.review.fallback", err = format!("{e:#}"))).dim()
                );
                let mut items: Vec<tui::ReviewItem> = outs
                    .iter()
                    .map(|o| tui::ReviewItem { name: o.name.clone(), actions: 0 })
                    .collect();
                tui::review(&mut items, &mut clip_params, |i, p| {
                    let bb = b.select(&outs[i].pack)?;
                    retrack(&mut outs[i], &bb, seed_for(&bb, p))?;
                    restyle(&outs[i], &bb.manifest, p).map(|(n, _, _)| n)
                })?;
            }
            // each script's accepted style, as flags that reproduce it; the
            // first carries forward as the next batch's seed
            for ((o, p), c) in outs.iter().zip(&clip_params).zip(&choices) {
                println!(
                    "{}",
                    style(format!("  {}: --model {} {}", o.name, c.model, style::flags_line(p))).dim()
                );
            }
            if let Some(p0) = clip_params.first() {
                params = *p0;
            }
            // and the first script's model drafts the next batch
            if let Some(c0) = choices.first() {
                b = b.select(&c0.model)?;
            }
            // remember the accepted style and model, so the next launch starts
            // from them
            // re-read first: the review writes the port it settled on while it
            // runs, and saving this older copy on top would erase it
            settings = settings::Settings::load();
            settings.last_params = Some(params);
            settings.model = Some(b.manifest.pack.clone());
            settings.save();
            if !cli.keep_cache {
                for o in &outs {
                    let _ = std::fs::remove_dir_all(&o.cache_dir);
                }
            }
        }

        // A named-video command line runs the one batch and is done; a failure
        // there is the process's failure. In picker mode the session goes on --
        // report the failures and return to the picker.
        if !picker_mode {
            if failed > 0 {
                // Plain, and with no path in it: this one goes through `main`,
                // which records it as the run's own last line and names the log
                // once, under every per-video entry already in it.
                anyhow::bail!("{failed} of {} videos failed", cli.videos.len());
            }
            return Ok(());
        }
        report = tui::Report {
            status: batch_status(drafted, skipped, failed, cli.force),
            failures,
            log,
        };
        cli.videos.clear(); // the next round starts from a fresh pick
    }
}

/// The red line a failure prints, and where its details were kept.
///
/// Printing is still worth doing -- a run from a command line has nothing else
/// -- but it is no longer the only copy: the picker's user gets the report
/// screen, and both of them get the log.
fn say_failed(e: &anyhow::Error, log: Option<&Path>) {
    let t = theme();
    eprintln!(
        "  {} {}",
        style(CROSS).fg(con(t.bad)).bold(),
        style(t!("console.failed", err = format!("{e:#}"))).fg(con(t.bad))
    );
    if let Some(p) = log {
        eprintln!(
            "  {}",
            style(t!("console.failed.log", path = nice(p))).fg(con(t.muted))
        );
    }
}

/// One-line feedback about the batch that just finished, shown in the reopened
/// picker. The key case is an all-skipped start: it is instant, so its console
/// message is wiped by the picker reopening -- this survives.
fn batch_status(drafted: i32, skipped: i32, failed: i32, force: bool) -> Option<String> {
    if drafted == 0 && skipped == 0 && failed == 0 {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if drafted > 0 {
        parts.push(t!("picker.status.drafted", n = drafted));
    }
    if skipped > 0 {
        parts.push(t!("picker.status.skipped", n = skipped));
    }
    if failed > 0 {
        parts.push(t!("picker.status.failed", n = failed));
    }
    let mut s = parts.join("  -  ");
    if skipped > 0 && drafted == 0 && failed == 0 && !force {
        s.push_str(&format!("  {}", t!("picker.status.overwrite")));
    } else if failed > 0 {
        // The console this used to point at is gone by the time anyone reads
        // this: the picker took the screen back. X is where the errors are.
        s.push_str(&format!("  {}", t!("picker.status.errors")));
    }
    Some(s)
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    /// clap validates a command's SHAPE when it parses, not when it compiles, and
    /// this one now has two variadic positionals (videos, and yt-dlp's arguments
    /// after `--`). `debug_assert` is clap's own check for exactly that class of
    /// mistake, and it belongs in the suite rather than in the first user's lap.
    #[test]
    fn the_command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    /// A `cli.<id>` key whose id is not an argument does NOTHING -- the help
    /// stays English and nobody is told why. That is the only way to get a
    /// translated CLI wrong, so every such key in every shipped catalog has to
    /// name an argument that exists (or the `.long` half of one).
    #[test]
    fn every_translated_argument_is_a_real_argument() {
        let ids: std::collections::HashSet<String> =
            Cli::command().get_arguments().map(|a| a.get_id().to_string()).collect();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("languages");
        for e in std::fs::read_dir(&dir).expect("languages/ is missing").flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let text = std::fs::read_to_string(&p).expect("unreadable catalog");
            let map: std::collections::HashMap<String, String> =
                serde_json::from_str(&text).expect("catalog is not valid JSON");
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            for key in map.keys() {
                let Some(rest) = key.strip_prefix("cli.") else { continue };
                if rest == "about" {
                    continue;
                }
                let id = rest.strip_suffix(".long").unwrap_or(rest);
                assert!(
                    ids.contains(id),
                    "{stem}.json translates {key:?}, but there is no --{} argument",
                    id.replace('_', "-")
                );
            }
        }
    }

    /// The script lands under the video's OWN name, character for character --
    /// including the names a filesystem allows and Unicode does not. A name
    /// read through a `String` comes back with U+FFFD standing where those
    /// were, and the script would then land beside the video under a name that
    /// is not the video's -- which is also the name the skip rule looks for, so
    /// the same clip would be drafted again on every run, forever.
    #[test]
    fn a_script_takes_the_video_name_even_when_it_is_not_unicode() {
        let cli = Cli::try_parse_from(["goblinscript"]).expect("a bare command parses");
        let named = |v: &str| {
            script_dst(&cli, Path::new(v)).file_name().unwrap().to_os_string()
        };
        assert_eq!(named("D:/v/clip.mp4"), std::ffi::OsString::from("clip.funscript"));
        assert_eq!(
            named("D:/v/Ostrov \u{00dc}n\u{00ef}c\u{00f8}d\u{00e9} \u{6f22}\u{5b57}.mkv"),
            std::ffi::OsString::from("Ostrov \u{00dc}n\u{00ef}c\u{00f8}d\u{00e9} \u{6f22}\u{5b57}.funscript")
        );
        // a name with more than one dot keeps every one of them but the last
        assert_eq!(named("D:/v/s01.e02.final.mp4"), std::ffi::OsString::from("s01.e02.final.funscript"));

        #[cfg(windows)]
        {
            use std::os::windows::ffi::{OsStrExt, OsStringExt};
            // NTFS stores a lone surrogate; Unicode has no character for one
            let odd = std::ffi::OsString::from_wide(&[
                0x0061, 0xD800, 0x002E, 0x006D, 0x0070, 0x0034,
            ]);
            let got = script_dst(&cli, Path::new(&odd));
            let want: Vec<u16> = [0x0061u16, 0xD800]
                .into_iter()
                .chain(std::ffi::OsStr::new(".funscript").encode_wide())
                .collect();
            assert_eq!(
                got.file_name().unwrap().encode_wide().collect::<Vec<u16>>(),
                want,
                "a name Unicode cannot spell survives the trip to the script"
            );
        }
    }

    /// The point of the `--` form is that hyphens after it are yt-dlp's business,
    /// not ours -- a flag we happen to share the name of must not be intercepted.
    #[test]
    fn yt_dlp_arguments_pass_through_after_a_dash_dash() {
        let c = Cli::try_parse_from([
            "goblinscript",
            "https://example.com/v",
            "--dl-dir",
            "D:/dl",
            "--",
            "--cookies-from-browser",
            "firefox",
            "--force",
        ])
        .expect("the passthrough form parses");
        assert_eq!(c.videos, vec![PathBuf::from("https://example.com/v")]);
        assert_eq!(c.dl_dir, Some(PathBuf::from("D:/dl")));
        assert_eq!(c.dl_args, ["--cookies-from-browser", "firefox", "--force"]);
        // --force after the -- went to yt-dlp, so ours is untouched
        assert!(!c.force);

        // and an ordinary run is unaffected by any of it
        let c = Cli::try_parse_from(["goblinscript", "a.mp4", "b.mp4", "--force"]).unwrap();
        assert_eq!(c.videos.len(), 2);
        assert!(c.force);
        assert!(c.dl_args.is_empty());
        assert!(c.dl_dir.is_none());
    }

    /// A folder on the command line is the videos IN it -- and the ones a level
    /// further down are somebody else's batch. The depth is the whole promise,
    /// so it is what this holds: the sub-folder's clip is created here purely
    /// to be left behind.
    #[test]
    fn a_folder_is_the_videos_in_it_and_nothing_deeper() {
        let dir = std::env::temp_dir().join("goblin_expand_dirs");
        let (deep, empty) = (dir.join("more"), dir.join("nothing"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::create_dir_all(&empty).unwrap();
        for name in ["b.mkv", "a.MP4", "notes.txt", "a.funscript"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        std::fs::write(deep.join("deeper.mp4"), b"").unwrap();

        // the folder, then a video named outright -- which happens to be the
        // one a level down, so naming it is the ONLY way it reaches the batch
        let named = deep.join("deeper.mp4");
        let mut cli = Cli::try_parse_from(["goblinscript", "x"]).unwrap();
        cli.videos = vec![dir.clone(), named.clone()];
        expand_dirs(&mut cli).expect("a folder with videos in it expands");
        assert_eq!(
            cli.videos,
            vec![dir.join("a.MP4"), dir.join("b.mkv"), named],
            "one level, video extensions only, in listing order"
        );

        // a folder the user believes has videos in it, that does not, is an
        // error -- silently drafting nothing is the worse answer
        cli.videos = vec![empty];
        let err = expand_dirs(&mut cli).expect_err("an empty folder is refused");
        assert!(format!("{err:#}").contains("no videos"), "{err:#}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Links are told from paths by scheme alone, and that is what decides whether
    /// yt-dlp is needed at all.
    #[test]
    fn only_links_ask_for_yt_dlp() {
        let c = Cli::try_parse_from(["goblinscript", r"D:\videos\clip.mp4"]).unwrap();
        assert!(!c.videos.iter().any(|v| dl::is_url(&v.to_string_lossy())));
        let c = Cli::try_parse_from(["goblinscript", "https://example.com/watch?v=x"]).unwrap();
        assert!(c.videos.iter().any(|v| dl::is_url(&v.to_string_lossy())));
    }
}

#[cfg(test)]
mod live_line_tests {
    use super::*;

    /// The escape sequence `erase_live` would send for a block of lines with
    /// these visible widths, on a terminal this wide.
    fn erased(widths: &[usize], width: usize) -> String {
        let mut buf: Vec<u8> = Vec::new();
        erase_live(&mut buf, widths, width);
        String::from_utf8(buf).unwrap()
    }

    // The erase has to climb over EVERY row the block occupies. Miscount it and
    // the clear starts in the middle of the block, which leaves the rows above
    // it stranded on screen for the rest of the run.
    #[test]
    fn the_erase_climbs_the_whole_block() {
        // nothing drawn yet: nothing to erase, and above all no stray climb
        assert_eq!(erased(&[], 80), "");
        // the status line alone, the display before the encoder has a row
        assert_eq!(erased(&[40], 80), "\r\x1b[J");
        // the viewport's 13 rows plus the status line, none of them wrapped
        assert_eq!(erased(&[70; 14], 80), "\x1b[13A\r\x1b[J");
    }

    // The flicker fix, held as a test because it is invisible in the code and
    // obvious on screen: getting back to the top of the block must MOVE the
    // cursor and nothing else. A `\x1b[J` here blanks 13 rows that are about to
    // be refilled, and the terminal is free to present that blank frame.
    #[test]
    fn homing_the_block_never_clears_it() {
        let mut buf: Vec<u8> = Vec::new();
        home_live(&mut buf, &[70; 14], 80);
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "\x1b[13A\r");
        assert!(!s.contains("[J") && !s.contains("[K"), "homing cleared: {s:?}");
        // and with nothing drawn it must not move at all -- there is no block
        // above the cursor to climb to
        let mut buf: Vec<u8> = Vec::new();
        home_live(&mut buf, &[], 80);
        assert!(buf.is_empty());
    }

    // Tearing the display down is the one path that DOES clear: no repaint is
    // coming behind it, so the rows have to go.
    #[test]
    fn erasing_the_block_clears_it() {
        assert_eq!(erased(&[70; 14], 80), "\x1b[13A\r\x1b[J");
    }

    // A terminal narrowed since the block was drawn has reflowed its lines, and
    // each one now stands on more rows than it was written on. The widths are
    // recorded for exactly this: 70 visible cells on a 40-column terminal is
    // two rows, not one.
    #[test]
    fn a_shrink_is_counted_row_by_row() {
        // 14 lines of 70 cells at width 40 -> 2 rows each = 28, climb 27
        assert_eq!(erased(&[70; 14], 40), "\x1b[27A\r\x1b[J");
        // a line exactly as wide as the terminal still stands on one row: the
        // cell that would wrap it is the one the painter truncates away
        assert_eq!(erased(&[40, 40], 40), "\x1b[1A\r\x1b[J");
        // mixed: 70 -> 2 rows, 20 -> 1, 41 -> 2. Five rows, climb four.
        assert_eq!(erased(&[70, 20, 41], 40), "\x1b[4A\r\x1b[J");
    }

    /// What the render thread would paint, stripped of colour.
    fn painted(s: &Stage) -> String {
        console::strip_ansi_codes(&stage_body(s, 100)).to_string()
    }

    fn measured(pos: u64, len: u64) -> Stage {
        let pb = ProgressBar::hidden();
        pb.set_length(len);
        // two updates a beat apart: enough for the estimator to hold a rate
        pb.set_position(pos / 2);
        std::thread::sleep(Duration::from_millis(20));
        pb.set_position(pos);
        Stage { label: "encode".into(), pb: Some(pb), units_per_sec: 15.0, note: String::new() }
    }

    // A stage that is still loading its graph has NOTHING to measure, and the
    // wait is tens of seconds on DirectML. It must say so -- no bar, and above
    // all no numbers, because indicatif answers "how fast / how long left" with
    // zeros before its first sample and that renders as a finished stage.
    #[test]
    fn a_loading_stage_shows_its_note_and_no_numbers() {
        let s = Stage {
            label: "encode".into(),
            pb: None,
            units_per_sec: 1.0,
            note: "loading the goblin...".into(),
        };
        let line = painted(&s);
        assert!(line.starts_with("ENCODE ......"), "{line}");
        assert!(line.contains("loading the goblin..."), "{line}");
        assert!(!line.contains('%') && !line.contains("left"), "no numbers: {line}");
    }

    // The same zeros reach the bar in the window between opening it and its
    // first position update (encode: one 32-frame window). "0.0x, 0s left" over
    // a stage that has not produced a row yet is the report of a hung bar.
    #[test]
    fn an_unmeasured_bar_never_claims_a_speed_or_an_eta() {
        let pb = ProgressBar::hidden();
        pb.set_length(1000);
        let s = Stage {
            label: "encode".into(),
            pb: Some(pb),
            units_per_sec: 15.0,
            note: String::new(),
        };
        let line = painted(&s);
        assert!(line.contains("0%"), "the bar is still shown, at zero: {line}");
        assert!(!line.contains("left"), "no ETA before a measurement: {line}");
        assert!(!line.contains("0.0x"), "no rate before a measurement: {line}");
    }

    // Once it has one, the numbers come back -- percent, rate as a multiple of
    // realtime, and an ETA.
    #[test]
    fn a_measured_bar_reports_percent_rate_and_eta() {
        let line = painted(&measured(500, 1000));
        assert!(line.contains("50%"), "{line}");
        assert!(line.contains('x') && line.contains("left"), "{line}");
    }

    /// The processing log and the startup report share a column grid, so the
    /// tick has to land in the same cell whatever the console is speaking. A
    /// CJK stage name is half as many CHARACTERS as it is columns wide -- pad
    /// by the wrong one and every translated line's chip sits four cells left
    /// of the English ones above it.
    #[test]
    fn a_translated_stage_line_puts_its_chip_in_the_same_column() {
        let where_tick = |line: &str| {
            let bare = console::strip_ansi_codes(line).to_string();
            bare.find(TICK).map(|b| measure(&bare[..b]))
        };
        let english = {
            let _lang = crate::lang::speaking("en-US");
            done_line(&t!("console.stage.normalize"), &t!("console.norm.reused"))
        };
        let chinese = {
            let _lang = crate::lang::speaking("zh-CN");
            done_line(&t!("console.stage.normalize"), &t!("console.norm.reused"))
        };
        assert_eq!(
            where_tick(&english),
            where_tick(&chinese),
            "the chip moved:\n{english}\n{chinese}"
        );
        // and the leader itself is measured in cells, not characters
        let _lang = crate::lang::speaking("zh-CN");
        let bare = console::strip_ansi_codes(&done_line(&t!("console.stage.encode"), "")).to_string();
        let leader = bare.trim_start().split(' ').take(2).collect::<Vec<_>>().join(" ");
        assert_eq!(measure(&leader), STAGE_LEADER + 1, "leader is off grid: {leader:?}");
    }

    /// A stage's live line is the same grid one frame earlier, and the bar
    /// width is what is LEFT of the terminal after the fixed fields. Measured
    /// in characters, a Chinese label under-reports and the row wraps.
    #[test]
    fn a_translated_live_line_stays_inside_the_terminal() {
        let _lang = crate::lang::speaking("zh-CN");
        let s = Stage {
            label: t!("console.stage.autocrop").into(),
            note: t!("console.crop.loading").into(),
            ..measured(500, 1000)
        };
        for width in [60usize, 80, 100, 120] {
            let line = console::strip_ansi_codes(&stage_body(&s, width)).to_string();
            assert!(measure(&line) < width, "{width}: {line}");
        }
    }
}

#[cfg(test)]
mod funscript_tests {
    use super::*;

    fn render() -> String {
        let acts = [style::Action { at: 0, pos: 0 }, style::Action { at: 100, pos: 90 }];
        let s = Funscript {
            version: "1.0",
            inverted: false,
            range: 100,
            metadata: FunMeta {
                author: AUTHOR,
                tags: TAGS,
                model: "v0.5.1",
                settings: style::flags_line(&style::Params { env_seed: 7, ..Default::default() }),
                filler: Vec::new(),
                artifacts: Vec::new(),
                phase_runs: None,
            },
            actions: &acts,
        };
        String::from_utf8(serde_json::to_vec(&s).unwrap()).unwrap()
    }

    // The whole point of the field order: metadata has to precede the long
    // actions array so it is readable at the top of a file.
    #[test]
    fn metadata_precedes_actions() {
        let out = render();
        let m = out.find("\"metadata\"").expect("metadata key present");
        let a = out.find("\"actions\"").expect("actions key present");
        assert!(m < a, "metadata must serialize before actions: {out}");
    }

    // The model and the flags that wrote the script, seed included, so the
    // file says how to make it again.
    #[test]
    fn model_and_settings_are_stamped() {
        let out = render();
        assert!(out.contains("\"model\":\"v0.5.1\""), "{out}");
        assert!(out.contains("--env-seed 7"), "{out}");
        // off is not the default, so it has to be written down
        let off = style::flags_line(&style::Params { max_speed: 0.0, cut_ease: 0.0, ..Default::default() });
        assert!(off.contains("--max-speed 0") && off.contains("--cut-ease 0"), "{off}");
    }

    // The author stamp is always written; it carries the tool + build number.
    #[test]
    fn author_always_present_and_versioned() {
        let out = render();
        assert!(out.contains(&format!("\"author\":\"{AUTHOR}\"")), "{out}");
        assert!(AUTHOR.contains(env!("CARGO_PKG_VERSION")), "author carries version");
    }

    // Provenance is stamped on every file: a generated script is marked as one.
    #[test]
    fn provenance_tag_always_present() {
        let out = render();
        assert!(out.contains("\"ai-generated\""), "{out}");
    }

    fn tmp(name: &str, body: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("goblinscript_kind_{name}.funscript"));
        std::fs::write(&p, body).unwrap();
        p
    }

    // The writer and the reader are one contract: what we stamp must be what
    // `script_kind` sees, or the picker calls our own drafts hand-made.
    #[test]
    fn our_own_script_reads_back_as_ai() {
        let p = tmp("ours", &render());
        assert_eq!(script_kind(&p), ScriptKind::Ai);
        let _ = std::fs::remove_file(&p);
    }

    // A tool that serializes its keys sorted puts `metadata` AFTER a
    // multi-megabyte actions array -- far past the head of the file, which is
    // why both ends get sampled. The filler here is bigger than that window.
    #[test]
    fn metadata_past_the_head_window_is_still_classified() {
        let acts: String =
            (0..40_000).map(|i| format!("{{\"at\":{i},\"pos\":50}},")).collect();
        let hand = format!(
            "{{\"actions\":[{}{{\"at\":9,\"pos\":0}}],\"metadata\":{{\"author\":\"OpenFunscripter\",\"tags\":[]}}}}",
            acts
        );
        let p = tmp("ofs", &hand);
        assert_eq!(script_kind(&p), ScriptKind::Hand);
        let _ = std::fs::remove_file(&p);

        let ai = hand.replace("\"tags\":[]", "\"tags\":[\"ai-generated\"]");
        let p = tmp("ofs_ai", &ai);
        assert_eq!(script_kind(&p), ScriptKind::Ai);
        let _ = std::fs::remove_file(&p);
    }

    // Nothing readable came back: say so rather than guessing "hand", which
    // would read as a claim about who wrote it.
    #[test]
    fn unreadable_is_unknown() {
        assert_eq!(script_kind(Path::new("no_such_file.funscript")), ScriptKind::Unknown);
        let p = tmp("junk", "not json at all");
        assert_eq!(script_kind(&p), ScriptKind::Unknown);
        let _ = std::fs::remove_file(&p);
    }
}

#[cfg(test)]
mod retrack_tests {
    use super::*;

    /// Every value of a track set, as bits: equal means bit-identical.
    fn bits(t: &style::Tracks) -> Vec<u64> {
        let mut v: Vec<f64> = Vec::new();
        for x in [&t.vmarg, &t.level, &t.env, &t.period, &t.conf] {
            v.extend(x.iter());
        }
        for (a, b) in [&t.band, &t.plat, &t.rev].into_iter().flatten() {
            v.extend(a.iter().chain(b.iter()));
        }
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn empty() -> style::Tracks {
        style::Tracks {
            vmarg: Vec::new(),
            level: Vec::new(),
            env: Vec::new(),
            band: None,
            plat: None,
            rev: None,
            period: Vec::new(),
            conf: Vec::new(),
        }
    }

    /// A model switch in the review is only worth offering if it is the draft
    /// the model would have made: the tracks a switch puts in hand are the
    /// head pass's own, bit for bit, whether they were made on the spot or
    /// taken off the shelf; a model on the trunk in hand changes the name
    /// and not the tracks; and the compared line leaves the script's model
    /// in hand.
    ///
    /// Synthetic latents through the real head graphs, on the CPU where the
    /// head always runs: no media, no GPU.
    #[test]
    #[ignore = "runs the head graphs of ./bundle on the CPU: cargo test --release -- --ignored retrack"]
    fn retrack_is_the_head_pass_whichever_way_it_arrives() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let b = Bundle::from_dir(&root.join("bundle")).unwrap();
        let models = b.models().unwrap();
        let a = models.iter().find(|m| m.name == b.default_pack()).unwrap().clone();
        let other = models.iter().find(|m| m.trunk != a.trunk).expect("a model on another trunk").clone();
        let sibling = models.iter().find(|m| m.trunk == a.trunk && m.name != a.name).cloned();
        let (ba, bo) = (b.select(&a.name).unwrap(), b.select(&other.name).unwrap());
        let man = &ba.manifest;

        // 1500 rows: past one 1024-row chunk, with a cut inside the second
        let dir = root.join("target/retrack_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        let rows = 1500usize;
        let row_bytes = man.dim * man.grid * man.grid;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let lat: Vec<u8> = (0..rows * row_bytes)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // latents sit near zero on the int8 grid, like the encoder's
                ((((state >> 33) % 41) as i64) - 20) as i8 as u8
            })
            .collect();
        std::fs::write(dir.join("latents.i8"), &lat).unwrap();
        let cuts = vec![man.row_ms(1200)];
        let shot_edges = boundaries::shot_edges(&cuts, rows, |i| man.row_ms(i));
        let mut d = DraftOut {
            name: "fixture".into(),
            src: dir.join("fixture.mp4"),
            cuts_ms: cuts.clone(),
            norm: None,
            dst: dir.join("fixture.funscript"),
            tracks: empty(),
            trunk: String::new(),
            tracks_seed: man.env_seed,
            shelf: HashMap::new(),
            shelf_cuts: cuts,
            shot_edges,
            times: (0..rows).map(|i| man.row_ms(i)).collect(),
            cache_dir: dir.clone(),
            dur_ms: man.row_ms(rows),
            t0_ms: 0.0,
            is_vr: false,
            crop: None,
            pack: a.name.clone(),
        };
        let seed = man.env_seed;
        let fresh_a = bits(&head_pass(&d, &ba, seed).unwrap());
        let fresh_o = bits(&head_pass(&d, &bo, bo.manifest.env_seed).unwrap());
        assert_ne!(fresh_a, fresh_o, "two trunks wrote the same tracks");

        d.tracks = head_pass(&d, &ba, seed).unwrap();
        d.trunk = ba.trunk();
        retrack(&mut d, &bo, bo.manifest.env_seed).unwrap();
        assert_eq!(bits(&d.tracks), fresh_o, "a head pass on switch is not the head pass");
        assert_eq!(d.pack, other.name);
        retrack(&mut d, &ba, seed).unwrap();
        assert_eq!(bits(&d.tracks), fresh_a, "the shelf did not give back the tracks it took");
        assert!(d.shelf.contains_key(&(bo.trunk(), bo.manifest.env_seed)));

        if let Some(s) = &sibling {
            let bs = b.select(&s.name).unwrap();
            retrack(&mut d, &bs, bs.manifest.env_seed).unwrap();
            assert_eq!(bits(&d.tracks), fresh_a, "a model on the same trunk moved the tracks");
            assert_eq!(d.pack, s.name);
        }

        // the review's style: the compared line first, the script's own model
        // left in hand and its script the plain restyle's
        let choice = review::Choice { model: a.name.clone(), compare: Some(other.name.clone()), spans: Vec::new(), cuts_ms: None };
        let p = style::Params::default();
        let compare = dir.join("compare.funscript");
        let r = review_style(&mut d, &b, &p, &choice, &compare).unwrap();
        assert_eq!(d.pack, a.name);
        assert_eq!(bits(&d.tracks), fresh_a);
        let script = std::fs::read(&d.dst).unwrap();
        restyle(&d, man, &p).unwrap();
        assert_eq!(std::fs::read(&d.dst).unwrap(), script, "the review's script is not the restyle's");
        assert!(std::fs::metadata(&compare).unwrap().len() > 0);
        assert_eq!(r.conf.len(), d.tracks.conf.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cut the page moved is a re-decode, and the draft must come back whole.
    ///
    /// Three claims, all through the real head graphs: an edited cut list
    /// changes the written script; a span naming another trunk's model changes
    /// it too, without moving the model in hand; and after either, `cuts_ms` /
    /// `shot_edges` are the draft's own again while `shelf_cuts` remembers what
    /// the shelf was rebuilt at -- so "reset" still has the original to ask for
    /// and nothing downstream sees a half-edited draft.
    #[test]
    #[ignore = "runs the head graphs of ./bundle on the CPU: cargo test --release -- --ignored moved_cut"]
    fn a_moved_cut_redecodes_and_leaves_the_draft_whole() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let b = Bundle::from_dir(&root.join("bundle")).unwrap();
        let models = b.models().unwrap();
        let a = models.iter().find(|m| m.name == b.default_pack()).unwrap().clone();
        let other = models.iter().find(|m| m.trunk != a.trunk).expect("another trunk").clone();
        let (ba, bo) = (b.select(&a.name).unwrap(), b.select(&other.name).unwrap());
        assert_ne!(ba.trunk(), bo.trunk(), "the two models share a trunk");
        let man = &ba.manifest;

        let dir = root.join("target/cuts_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        let rows = 1500usize;
        let row_bytes = man.dim * man.grid * man.grid;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let lat: Vec<u8> = (0..rows * row_bytes)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((((state >> 33) % 41) as i64) - 20) as i8 as u8
            })
            .collect();
        std::fs::write(dir.join("latents.i8"), &lat).unwrap();

        let cuts0 = vec![man.row_ms(800)];
        let edges0 = boundaries::shot_edges(&cuts0, rows, |i| man.row_ms(i));
        let mut d = DraftOut {
            name: "fixture".into(),
            src: dir.join("fixture.mp4"),
            cuts_ms: cuts0.clone(),
            norm: None,
            dst: dir.join("fixture.funscript"),
            tracks: empty(),
            trunk: String::new(),
            tracks_seed: man.env_seed,
            shelf: HashMap::new(),
            shelf_cuts: cuts0.clone(),
            shot_edges: edges0.clone(),
            times: (0..rows).map(|i| man.row_ms(i)).collect(),
            cache_dir: dir.clone(),
            dur_ms: man.row_ms(rows),
            t0_ms: 0.0,
            is_vr: false,
            crop: None,
            pack: a.name.clone(),
        };
        let p = style::Params::default();
        let compare = dir.join("compare.funscript");
        let base_choice = review::Choice {
            model: a.name.clone(), compare: None, spans: Vec::new(), cuts_ms: None,
        };
        review_style(&mut d, &b, &p, &base_choice, &compare).unwrap();
        let base = std::fs::read(&d.dst).unwrap();
        let tracks_base = bits(&d.tracks);

        // 1. a moved cut: the head is shown a different boundary, so the tracks
        //    in hand -- and the script the page reads back -- are not the same
        let cuts1 = vec![man.row_ms(300), man.row_ms(1100)];
        let moved = review::Choice {
            model: a.name.clone(), compare: None, spans: Vec::new(),
            cuts_ms: Some(cuts1.clone()),
        };
        review_style(&mut d, &b, &p, &moved, &compare).unwrap();
        let moved_script = std::fs::read(&d.dst).unwrap();
        let tracks_moved = bits(&d.tracks);
        assert_eq!(tracks_base.len(), tracks_moved.len(), "a moved cut changed the row count");
        assert_ne!(tracks_base, tracks_moved, "a moved cut did not re-decode the tracks");
        assert_ne!(base, moved_script, "a moved cut did not change the written script");
        // the draft is whole again, and the shelf remembers the edit
        assert_eq!(d.cuts_ms, cuts0, "the draft lost its own cuts");
        assert_eq!(d.shot_edges, edges0, "the draft lost its own shot edges");
        assert_eq!(d.shelf_cuts, cuts1, "the shelf did not record the rebuilt cuts");
        assert_eq!(d.pack, a.name, "the moved cut moved the model in hand");

        // 2. a span on the other trunk: the rows it covers are another model's,
        //    so the script changes -- and the model in hand does not, and the
        //    shelf is still the edited cut set (spans never touch cuts)
        let spanned = review::Choice {
            model: a.name.clone(), compare: None,
            spans: vec![review::Span {
                lo_ms: 100.0,
                hi_ms: man.row_ms(1100),
                model: other.name.clone(),
                // a model span, not a hand-drawn one: it names a bundle and
                // carries no waveform
                wave: None,
                no_limit: true,
                no_cuts: false,
                scale: 1.0,
            }],
            cuts_ms: Some(cuts1.clone()),
        };
        review_style(&mut d, &b, &p, &spanned, &compare).unwrap();
        let spanned_script = std::fs::read(&d.dst).unwrap();
        assert_ne!(moved_script, spanned_script, "a span did not change the written script");
        assert_eq!(bits(&d.tracks), tracks_moved, "a span disturbed the tracks in hand");
        assert_eq!(d.pack, a.name, "a span moved the model in hand");
        assert_eq!(d.cuts_ms, cuts0);
        assert_eq!(d.shot_edges, edges0);
        assert_eq!(d.shelf_cuts, cuts1);

        // 3. back to the draft's own cuts: the head runs at the original
        //    boundaries again, and the bytes are the base run's own
        review_style(&mut d, &b, &p, &base_choice, &compare).unwrap();
        assert_eq!(bits(&d.tracks), tracks_base, "reset did not re-decode at the draft's cuts");
        assert_eq!(std::fs::read(&d.dst).unwrap(), base, "reset did not restore the script");
        assert_eq!(d.shelf_cuts, cuts0, "reset did not rebuild the shelf at the draft's cuts");
        assert_eq!(d.cuts_ms, cuts0);
        assert_eq!(d.shot_edges, edges0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
