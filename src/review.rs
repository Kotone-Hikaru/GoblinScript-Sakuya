//! The browser review: a loopback-only HTTP server baked into the exe, so the
//! post-draft review needs no external player. The page (embedded, one file,
//! no external requests) plays the video with the scripted motion overlaid as
//! a simulator, and exposes the style knobs; a change POSTs back here, the
//! scripts on disk rewrite in milliseconds, and the page redraws.
//!
//! Video source is probe-and-fallback, decided in main: the ORIGINAL file
//! when every browser can play it (`ffmpeg::browser_playable`), the cache's
//! normalized copy (H.264+AAC MP4 -- plays everywhere, but 480p) otherwise.
//! The page keeps a client-side `onerror` fallback on top, so a codec
//! the probe was wrong about costs one reload, never a black screen.
//!
//! Control requests (params, done) run on the MAIN thread -- the re-style
//! closure is not `Send`, and must not race itself. File requests (video,
//! script) are handed to short-lived threads: a browser streaming a 4 GB
//! range must never block the Done button.

use anyhow::{Context, Result};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tiny_http::{Header, Method, Response, StatusCode};

use crate::style;

/// One drafted video, as the review page sees it.
pub struct Clip {
    pub name: String,
    /// The .funscript on disk -- rewritten by every re-style, served as-is.
    pub script: PathBuf,
    /// What `/video/N` streams: the original, or the normalized copy.
    pub video: PathBuf,
    /// The normalized copy, when `video` is the original -- the page swaps to
    /// it if playback still errors.
    pub fallback: Option<PathBuf>,
    /// False when the page is showing the 480p working copy (it says so).
    pub is_original: bool,
    pub duration_ms: f64,
    /// Row clock for the confidence track (`Restyled::conf`, per-row [0,1],
    /// NaN = unknown), as the rate and where row 0 sits: row i is at
    /// `row0_ms + i * 1000 / row_hz`. Both halves travel because the offset
    /// is not zero -- a row is its tubelet pair's midpoint, half a tubelet
    /// after the frame it is indexed by (`Manifest::row_ms`).
    pub row_hz: f64,
    pub row0_ms: f64,
    /// Where the VIDEO's clock starts on the SCRIPT's. Non-zero only for a VR
    /// clip whose prep trimmed a range: the page streams the trimmed copy, but
    /// the funscript on disk is for the full-length source (which is what makes
    /// it usable), so the page adds this to `currentTime` before it reads the
    /// script. 0 everywhere else, which is the whole flat path.
    pub t0_ms: f64,
    /// The auto-crop plan, when this draft took a crop: per-segment rects as
    /// fractions of the frame, on the VIDEO clock. The page dims the rest of
    /// the picture and draws the rect, so the user sees what the goblins
    /// actually looked at. `None` = the whole frame was drafted, and no crop
    /// UI appears at all.
    pub crop: Option<crate::autocrop::View>,
    /// The model that drafted it, the bundle's pack name: the page's model
    /// menu starts here.
    pub model: String,
    /// Every model the page may switch this script to, with its trunk.
    pub models: Vec<crate::bundle::ModelInfo>,
    /// Where the compared model's line is written: served as `/compare/N`,
    /// never the script itself.
    pub compare: PathBuf,
}

/// Which model writes a script, and which other model's line the page lays
/// under it. Both are part of the ask the styler works on: switching either
/// is a re-style, and a switch onto a trunk without tracks is a head pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub model: String,
    pub compare: Option<String>,
}

/// The manifest's "normal" values for the two preset axes. The page shows each
/// preset's resolved number in brackets (`normal (0.65)`) and seeds the expert
/// numeric inputs from them -- the constants for the other presets it derives
/// itself, but the "normal" ones are the manifest's and only known here.
#[derive(Clone, Copy)]
pub struct Presets {
    /// The manifest's own ramp start -- what `normal` resolves to.
    pub dwell_ramp: f64,
    pub still_eps: f64,
}

/// A hard ceiling per range response. Browsers ask open-ended
/// (`bytes=N-`) and read at their leisure; answering with the whole remainder
/// would pin a serving thread (and a few GB of socket traffic) on one
/// request. A short 206 is valid HTTP -- the browser just asks again.
const RANGE_CHUNK: u64 = 8 << 20;

/// Serve the review until the user finishes (Done in the page, or
/// Enter/Q/Esc in the console). `params` and `choices` start as the draft's
/// and end as what the page settled. `apply(i, params, choice)` re-styles clip `i`
/// under the chosen model (and writes the compared model's line) and returns
/// its action count. `page` hears the URL once the page is up -- the console
/// prints it, an API session reports it -- and `open` also launches the
/// default browser here.
pub fn review(
    clips: &[Clip],
    params: &mut [style::Params],
    choices: &mut [Choice],
    presets: Presets,
    open: bool,
    page: &dyn Fn(&str),
    apply: impl Fn(usize, &style::Params, &Choice) -> Result<Restyled> + Send + Sync,
) -> Result<()> {
    let server = tiny_http::Server::http("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("could not start the review server: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("review server has no TCP address")?
        .port();
    let url = format!("http://127.0.0.1:{port}/");

    // the draft's own params, snapshot before any re-style, so the page's
    // per-clip "reset" can restore exactly what the goblins first drafted.
    let defaults: Vec<style::Params> = params.to_vec();
    let mut first: Vec<Restyled> = Vec::with_capacity(clips.len());
    for (i, p) in params.iter().enumerate() {
        first.push(apply(i, p, &choices[i])?);
    }

    page(&url);
    if open {
        open_browser(&url);
    }

    let work = Arc::new(Work::new(first));
    // A scope, so the styler can borrow the draft's tracks rather than own a
    // copy of every clip's worth of them.
    std::thread::scope(|scope| {
        let styler = {
            let work = Arc::clone(&work);
            let apply = &apply;
            scope.spawn(move || work.style_until_done(apply))
        };
        let r = serve(&server, clips, params, choices, &defaults, presets, &work);
        // Whatever ended the review -- Done, a console key, an error -- the
        // styler is told before we leave, or the scope waits on it forever.
        work.finish();
        let _ = styler.join();
        r
    })
}

/// The request loop. Answers immediately, always: the only work it does itself
/// is reading a file or serialising state, and a re-style (seconds on a long
/// video) is handed to the styler thread instead. That is the whole point --
/// tiny_http accepts on a background thread but ANSWERS from this one, so a
/// re-style running here stalls the video stream, the timeline, and the Done
/// button along with it.
fn serve(
    server: &tiny_http::Server,
    clips: &[Clip],
    params: &mut [style::Params],
    choices: &mut [Choice],
    defaults: &[style::Params],
    presets: Presets,
    work: &Work,
) -> Result<()> {
    // No console (piped output, a service context): the page's Done button is
    // the only finish, which is exactly right there.
    let raw = crate::RawMode::enable();
    loop {
        // console side: Enter/Q/Esc (or Ctrl-C) finish, same as the page's Done
        while raw.on() && event::poll(Duration::ZERO)? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                        return Ok(())
                    }
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(())
                    }
                    _ => {}
                }
            }
        }
        // A re-style that failed is the user's business, not a silent revert:
        // it surfaces on the page's error line through `/state`.
        let req = match server.recv_timeout(Duration::from_millis(50))? {
            Some(r) => r,
            None => continue,
        };
        if handle(req, clips, params, choices, defaults, presets, work) == Flow::Finish {
            return Ok(());
        }
    }
}

#[derive(PartialEq)]
enum Flow {
    Continue,
    Finish,
}

/// The re-style queue, shared between the request loop and the styler thread.
///
/// At most ONE re-style is ever outstanding per clip. A knob turned while an
/// older change is still running replaces it rather than queueing behind it:
/// the older answer is a script nobody will read, and on a long video paying
/// for it would put every later change that many seconds further behind. So the
/// page can post on every twitch of a control and the styler always works on
/// the newest question asked.
struct Work {
    state: Mutex<Queue>,
    wake: Condvar,
}

struct Queue {
    /// clip -> the newest params and model choice asked for and not yet
    /// started.
    want: std::collections::BTreeMap<usize, (style::Params, Choice)>,
    /// The clip the styler has in hand, if any.
    running: Option<usize>,
    /// Action count per clip, from the last re-style that finished.
    actions: Vec<usize>,
    /// Confidence track per clip, from the same re-style: the trunk's, so a
    /// model switch moves it.
    conf: Vec<Vec<f64>>,
    /// What the last failed re-style said, for the page's error line.
    error: Option<String>,
    done: bool,
}

/// What one re-style leaves for the page: the script's action count and the
/// confidence track of the model that wrote it (empty without a conf head).
pub struct Restyled {
    pub actions: usize,
    pub conf: Vec<f64>,
}

impl Work {
    fn new(first: Vec<Restyled>) -> Self {
        let (actions, conf) = first.into_iter().map(|r| (r.actions, r.conf)).unzip();
        Work {
            state: Mutex::new(Queue {
                want: std::collections::BTreeMap::new(),
                running: None,
                actions,
                conf,
                error: None,
                done: false,
            }),
            wake: Condvar::new(),
        }
    }

    fn conf(&self, i: usize) -> Option<Vec<f64>> {
        self.state.lock().unwrap().conf.get(i).cloned()
    }

    /// Ask for clip `i` to be re-styled with `p` under `c`, replacing any
    /// request for that clip the styler has not picked up yet.
    fn request(&self, i: usize, p: style::Params, c: Choice) {
        let mut q = self.state.lock().unwrap();
        q.want.insert(i, (p, c));
        q.error = None;
        drop(q);
        self.wake.notify_one();
    }

    /// Is this clip's script still catching up with its knobs? True from the
    /// moment a change is posted until the styler has written the file, which
    /// is exactly the window the page must not re-read the script in.
    fn busy(&self, i: usize) -> bool {
        let q = self.state.lock().unwrap();
        q.running == Some(i) || q.want.contains_key(&i)
    }

    fn actions(&self) -> Vec<usize> {
        self.state.lock().unwrap().actions.clone()
    }

    fn error(&self) -> Option<String> {
        self.state.lock().unwrap().error.clone()
    }

    fn finish(&self) {
        self.state.lock().unwrap().done = true;
        self.wake.notify_all();
    }

    /// The styler thread: take the newest outstanding request, re-style it,
    /// publish the count, repeat until the review ends.
    fn style_until_done(
        &self,
        apply: &(impl Fn(usize, &style::Params, &Choice) -> Result<Restyled> + ?Sized),
    ) {
        loop {
            let (i, p, c) = {
                let mut q = self.state.lock().unwrap();
                loop {
                    if let Some(i) = q.want.keys().next().copied() {
                        let (p, c) = q.want.remove(&i).expect("key came from this map");
                        q.running = Some(i);
                        break (i, p, c);
                    }
                    // The queue is checked BEFORE `done`, so the review ending
                    // drains rather than abandons: someone who turns a knob and
                    // hits Done a second later must still get that knob's
                    // script on disk. `review` joins this thread, so nothing
                    // downstream runs until the drain is finished.
                    if q.done {
                        return;
                    }
                    q = self.wake.wait(q).unwrap();
                }
            };
            let r = apply(i, &p, &c);
            let mut q = self.state.lock().unwrap();
            q.running = None;
            match r {
                Ok(r) => {
                    let q = &mut *q;
                    if let (Some(a), Some(c)) = (q.actions.get_mut(i), q.conf.get_mut(i)) {
                        *a = r.actions;
                        *c = r.conf;
                    }
                }
                // Kept for the page rather than printed: the console behind the
                // browser is not where someone turning a knob is looking.
                Err(e) => q.error = Some(format!("{e:#}")),
            }
        }
    }
}

fn handle(
    mut req: tiny_http::Request,
    clips: &[Clip],
    params: &mut [style::Params],
    choices: &mut [Choice],
    defaults: &[style::Params],
    presets: Presets,
    work: &Work,
) -> Flow {
    let url = req.url().to_string();
    let mut parts = url.trim_matches('/').split('/');
    match (req.method().clone(), parts.next().unwrap_or("")) {
        (Method::Get, "") => {
            let _ = req.respond(
                Response::from_string(include_str!("review.html"))
                    .with_header(header("Content-Type", "text/html; charset=utf-8")),
            );
        }
        (Method::Get, "state") => {
            let _ = req.respond(json_response(&state_json(
                clips, params, choices, defaults, presets, work,
            )));
        }
        // The active catalog, for the page to dress itself in. Separate from
        // `/state` because it changes only when the language does, and the page
        // applies it once at boot rather than on every re-style.
        (Method::Get, "lang") => {
            let _ = req.respond(json_response(&crate::lang::catalog_json()));
        }
        (Method::Get, "script") => {
            if let Some(c) = parts.next().and_then(|i| i.parse::<usize>().ok()).and_then(|i| clips.get(i)) {
                serve_file_threaded(req, c.script.clone(), "application/json");
            } else {
                let _ = req.respond(Response::from_string("no such clip").with_status_code(404));
            }
        }
        // the compared model's line, written beside the script by the same
        // re-style that wrote the script
        (Method::Get, "compare") => {
            if let Some(c) = parts.next().and_then(|i| i.parse::<usize>().ok()).and_then(|i| clips.get(i)) {
                serve_file_threaded(req, c.compare.clone(), "application/json");
            } else {
                let _ = req.respond(Response::from_string("no such clip").with_status_code(404));
            }
        }
        (Method::Get, "conf") => {
            let i = parts.next().and_then(|i| i.parse::<usize>().ok());
            match i.and_then(|i| Some((clips.get(i)?, work.conf(i)?))) {
                Some((c, conf)) => {
                    let _ = req.respond(json_response(&conf_json(c, &conf)));
                }
                None => {
                    let _ = req.respond(Response::from_string("no such clip").with_status_code(404));
                }
            }
        }
        (Method::Get, "video") => {
            let clip = parts.next().and_then(|i| i.parse::<usize>().ok()).and_then(|i| clips.get(i));
            let path = clip.and_then(|c| match parts.next() {
                None => Some(c.video.clone()),
                Some("fallback") => c.fallback.clone(),
                Some(_) => None,
            });
            match path {
                Some(p) => {
                    let ctype = video_ctype(&p);
                    serve_file_threaded(req, p, ctype);
                }
                None => {
                    let _ =
                        req.respond(Response::from_string("no such clip").with_status_code(404));
                }
            }
        }
        (Method::Post, "params") => {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let parsed = parse_params(&body, |i| params.get(i).copied()).and_then(|(clip, p)| {
                let c = match (clips.get(clip), choices.get(clip)) {
                    (Some(clip), Some(base)) => parse_choice(&body, clip, base)?,
                    _ => anyhow::bail!("no clip {clip}"),
                };
                Ok((clip, p, c))
            });
            match parsed {
                Ok((clip, p, c)) if clip < params.len() => {
                    // per clip: only the addressed script re-styles. The answer
                    // goes back NOW, with the clip marked styling -- the work
                    // itself is the styler's, and the page waits for it by
                    // watching that flag rather than by holding a socket open.
                    params[clip] = p;
                    choices[clip] = c.clone();
                    work.request(clip, p, c);
                    let _ = req.respond(json_response(&state_json(
                        clips, params, choices, defaults, presets, work,
                    )));
                }
                Ok((clip, _, _)) => {
                    let _ = req.respond(
                        Response::from_string(format!("no clip {clip}")).with_status_code(400),
                    );
                }
                Err(e) => {
                    let _ = req
                        .respond(Response::from_string(format!("{e:#}")).with_status_code(400));
                }
            }
        }
        (Method::Post, "done") => {
            let _ = req.respond(Response::from_string("ok"));
            return Flow::Finish;
        }
        _ => {
            let _ = req.respond(Response::from_string("not found").with_status_code(404));
        }
    }
    Flow::Continue
}

/// The style knobs of one clip, as the page's controls read them.
/// `dwell_ramp` and `still_eps` are the raw EXPERT overrides -- `null` when the
/// clip is on a preset (the page then reads the preset's number from
/// `presets`), a number when it has been dialled in by hand. Each stands on the
/// same axis as the preset it replaces, so the box and the dropdown beside it
/// are two ways of setting one thing.
///
/// `env_seed` follows the same `null` rule for the same reason, and one more:
/// its "whatever the bundle says" sentinel is `u64::MAX`, which a browser
/// cannot hold -- JSON numbers are doubles there, so the sentinel would come
/// back rounded UP past `u64::MAX` and every knob the page posts after that
/// would be rejected. `null` is what the page already means by it, and what
/// the inbound `Option<u64>` already reads.
pub(crate) fn params_json(p: &style::Params) -> serde_json::Value {
    serde_json::json!({
        "style": p.style.label(),
        "dwells": p.dwells.label(),
        "stillness": p.stillness.label(),
        "intensity": p.intensity,
        "env_seed": (p.env_seed != u64::MAX).then_some(p.env_seed),
        "range": [p.range.0, p.range.1],
        "max_speed": p.max_speed,
        "cut_ease": p.cut_ease,
        "dwell_ramp": p.dwell_ramp,
        "still_eps": p.still_eps,
        // depth uniformity: the preset label plus its raw EXPERT overrides
        // (`null` on a preset, a number when dialled in)
        "depth": p.depth.label(),
        "depth_dose": p.depth_dose,
        "depth_window": p.depth_window,
        "filler_gap_s": p.filler_gap_s,
        "filler_min_real_s": p.filler_min_real_s,
        "filler_real_v": p.filler_real_v,
        "filler_model_w": p.filler_model_w,
        "filler_rate": p.filler_rate,
        "filler_amp": p.filler_amp,
        "filler_ramp_s": p.filler_ramp_s,
        "filler_max_bridge_s": p.filler_max_bridge_s,
        "filler_sway": p.filler_sway,
        "filler_sway_s": p.filler_sway_s,
        "filler_pattern": p.filler_pattern.label(),
        "filler_burst": p.filler_burst,
        "filler_rest_s": p.filler_rest_s,
    })
}

/// Each preset's resolved number, so the page can label the dropdowns
/// (`normal (0.65)`) and seed the expert inputs. Only "normal" is the
/// manifest's; the rest are the fixed constants the enums carry.
pub(crate) fn presets_json(presets: Presets) -> serde_json::Value {
    use style::{DepthUniformity, Dwells, Stillness};
    let depth = |d: DepthUniformity| {
        let p = d.base().expect("non-off depth preset");
        serde_json::json!({ "depth_dose": p.dose, "depth_window": p.window_s })
    };
    serde_json::json!({
        "dwells": {
            "cautious": Dwells::Cautious.ramp_start((presets.dwell_ramp, 1.0)),
            "normal": Dwells::Normal.ramp_start((presets.dwell_ramp, 1.0)),
            "eager": Dwells::Eager.ramp_start((presets.dwell_ramp, 1.0)),
        },
        "stillness": {
            "low": Stillness::Low.still_eps(presets.still_eps),
            "normal": Stillness::Normal.still_eps(presets.still_eps),
            "high": Stillness::High.still_eps(presets.still_eps),
        },
        // every depth preset is fixed constants (none manifest-tuned), so all
        // three resolve here for the page to seed its expert inputs from
        "depth": {
            "subtle": depth(DepthUniformity::Subtle),
            "even": depth(DepthUniformity::Even),
            "locked": depth(DepthUniformity::Locked),
        },
        // filler preset bundles (style.rs is the one source), so the page
        // seeds gap/rate/amp/pattern when a preset is picked
        "filler": {
            "subtle": filler_preset(style::FillerPreset::Subtle),
            "steady": filler_preset(style::FillerPreset::Steady),
            "bursts": filler_preset(style::FillerPreset::Bursts),
        },
    })
}

fn filler_preset(p: style::FillerPreset) -> serde_json::Value {
    let (gap, rate, amp, pat) = p.base().expect("non-off filler preset");
    serde_json::json!({
        "filler_gap_s": gap, "filler_rate": rate, "filler_amp": amp,
        "filler_pattern": pat.label(),
    })
}

/// What the page needs to (re)draw everything but the tracks themselves. Style
/// is per clip -- each carries its own params and reproducing flags.
fn state_json(
    clips: &[Clip],
    params: &[style::Params],
    choices: &[Choice],
    defaults: &[style::Params],
    presets: Presets,
    work: &Work,
) -> serde_json::Value {
    let actions = work.actions();
    serde_json::json!({
        "presets": presets_json(presets),
        // the active palette, as the custom properties the page overrides its
        // stylesheet with -- so the browser surface wears the same scheme as the
        // picker that launched it
        "theme": crate::theme::css_vars(),
        // what a re-style had to say for itself, when one failed
        "error": work.error(),
        "clips": clips.iter().enumerate().map(|(i, c)| serde_json::json!({
            "name": c.name,
            // the model writing this script, the one whose line lies under
            // it (null: none), and every model the menus offer
            "model": choices[i].model,
            "compare": choices[i].compare,
            "models": c.models,
            "actions": actions[i],
            // This clip's script is still catching up with its knobs. The page
            // waits on it before re-reading the file, and says so meanwhile --
            // on a two-hour video the styling is seconds of work, and a page
            // that looked finished would be showing the PREVIOUS answer.
            "styling": work.busy(i),
            "duration_ms": c.duration_ms,
            "video": format!("/video/{i}"),
            "fallback": c.fallback.as_ref().map(|_| format!("/video/{i}/fallback")),
            "original": c.is_original,
            // the video's zero on the script's clock (VR range trims only)
            "t0_ms": c.t0_ms,
            // the auto-crop rects to draw over the frame (null = no crop ran)
            "crop": c.crop,
            "params": params_json(&params[i]),
            // the draft's own params, so the page can offer a per-clip reset
            "defaults": params_json(&defaults[i]),
            "flags": style::flags_line(&params[i]),
        })).collect::<Vec<_>>(),
    })
}

/// One clip's confidence track for the review page: the row clock and the
/// per-row [0,1] score, rounded to keep the payload small, with `null` for the
/// unforwarded rows. `conf` is `null` (not `[]`) when the bundle has no conf
/// head, so the page can tell "unavailable" from "all-unknown" and hide the UI.
fn conf_json(c: &Clip, conf: &[f64]) -> serde_json::Value {
    let conf = if conf.is_empty() {
        serde_json::Value::Null
    } else {
        conf
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    serde_json::json!((v * 1000.0).round() / 1000.0)
                } else {
                    serde_json::Value::Null
                }
            })
            .collect::<Vec<_>>()
            .into()
    };
    serde_json::json!({ "fps": c.row_hz, "row0": c.row0_ms, "conf": conf })
}

/// The model choice a params message carries for `clip`: `model` absent keeps
/// the clip's own, `compare` absent keeps it and `null` clears it. A name the
/// clip does not offer is refused. A script switched onto the model it was
/// compared with compares with nothing: the two lines would be one.
fn parse_choice(body: &str, clip: &Clip, base: &Choice) -> Result<Choice> {
    #[derive(serde::Deserialize)]
    struct Msg {
        #[serde(default)]
        model: Option<String>,
        #[serde(default, deserialize_with = "nullable")]
        compare: Option<Option<String>>,
    }
    let m: Msg = serde_json::from_str(body).context("bad params payload")?;
    let model = m.model.unwrap_or_else(|| base.model.clone());
    let compare = m.compare.unwrap_or_else(|| base.compare.clone()).filter(|c| *c != model);
    for name in std::iter::once(&model).chain(compare.as_ref()) {
        anyhow::ensure!(
            clip.models.iter().any(|m| &m.name == name),
            "no model {name:?} for this script"
        );
    }
    Ok(Choice { model, compare })
}

/// The page's params message -> a clip index + a validated `Params`, same
/// bounds as the CLI. The clip index says which script the change is for;
/// `base(clip)` is what a field the message leaves out keeps.
fn parse_params(
    body: &str,
    base: impl Fn(usize) -> Option<style::Params>,
) -> Result<(usize, style::Params)> {
    #[derive(serde::Deserialize)]
    struct Which {
        clip: usize,
    }
    let w: Which = serde_json::from_str(body).context("bad params payload")?;
    let b = base(w.clip).with_context(|| format!("no clip {}", w.clip))?;
    Ok((w.clip, params_from_json(body, &b)?))
}

/// A params message -> a validated `Params`, same bounds as the CLI. Every
/// field is optional and falls back to `base`: the review page sends the
/// whole set, an API client sends the knobs it moved. Unknown fields are
/// ignored, so the message may carry more than params.
pub(crate) fn params_from_json(body: &str, base: &style::Params) -> Result<style::Params> {
    #[derive(serde::Deserialize)]
    struct Msg {
        #[serde(default)]
        style: Option<String>,
        #[serde(default)]
        dwells: Option<String>,
        #[serde(default)]
        stillness: Option<String>,
        #[serde(default)]
        intensity: Option<f64>,
        // absent keeps the base; `null` is the bundle's own seed
        #[serde(default, deserialize_with = "nullable")]
        env_seed: Option<Option<u64>>,
        #[serde(default)]
        range: Option<[f64; 2]>,
        #[serde(default)]
        max_speed: Option<f64>,
        #[serde(default)]
        cut_ease: Option<f64>,
        // absent keeps the base; `null` puts the clip back on its preset; a
        // number is the expert override
        #[serde(default, deserialize_with = "nullable")]
        dwell_ramp: Option<Option<f64>>,
        #[serde(default, deserialize_with = "nullable")]
        still_eps: Option<Option<f64>>,
        // depth uniformity; dose/window are overrides on the preset, with the
        // same absent/null/number reading as the expert knobs above
        #[serde(default)]
        depth: Option<String>,
        #[serde(default, deserialize_with = "nullable")]
        depth_dose: Option<Option<f64>>,
        #[serde(default, deserialize_with = "nullable")]
        depth_window: Option<Option<f64>>,
        // background filler rhythm (absent = off)
        #[serde(default)]
        filler_gap_s: Option<f64>,
        #[serde(default)]
        filler_min_real_s: Option<f64>,
        #[serde(default)]
        filler_real_v: Option<f64>,
        #[serde(default)]
        filler_model_w: Option<f64>,
        #[serde(default)]
        filler_rate: Option<f64>,
        #[serde(default)]
        filler_amp: Option<f64>,
        #[serde(default)]
        filler_ramp_s: Option<f64>,
        #[serde(default)]
        filler_max_bridge_s: Option<f64>,
        #[serde(default)]
        filler_sway: Option<f64>,
        #[serde(default)]
        filler_sway_s: Option<f64>,
        #[serde(default)]
        filler_pattern: Option<String>,
        #[serde(default)]
        filler_burst: Option<usize>,
        #[serde(default)]
        filler_rest_s: Option<f64>,
    }
    let m: Msg = serde_json::from_str(body).context("bad params payload")?;
    let sty = match m.style.as_deref() {
        None => base.style,
        Some(s) => style::Style::from_label(s)
            .with_context(|| format!("bad style {s:?}"))?,
    };
    let dwells = match m.dwells.as_deref() {
        None => base.dwells,
        Some(s) => style::Dwells::from_label(s).with_context(|| format!("bad dwells {s:?}"))?,
    };
    let stillness = match m.stillness.as_deref() {
        None => base.stillness,
        Some(s) => {
            style::Stillness::from_label(s).with_context(|| format!("bad stillness {s:?}"))?
        }
    };
    let intensity = m.intensity.unwrap_or(base.intensity);
    if !(0.5..=2.0).contains(&intensity) {
        anyhow::bail!("intensity {intensity} is out of range (0.5..2.0)");
    }
    let (lo, hi) = m.range.map(|r| (r[0], r[1])).unwrap_or(base.range);
    if !(0.0..=100.0).contains(&lo) || !(0.0..=100.0).contains(&hi) || lo >= hi {
        anyhow::bail!("need 0 <= LO < HI <= 100, got {lo}-{hi}");
    }
    let max_speed = m.max_speed.unwrap_or(base.max_speed);
    if max_speed < 0.0 {
        anyhow::bail!("max_speed must be >= 0");
    }
    let cut_ease = m.cut_ease.unwrap_or(base.cut_ease);
    if cut_ease < 0.0 {
        anyhow::bail!("cut_ease must be >= 0");
    }
    let dwell_ramp = m.dwell_ramp.unwrap_or(base.dwell_ramp);
    if let Some(pk) = dwell_ramp {
        if !(0.0..=1.0).contains(&pk) {
            anyhow::bail!("dwell_ramp {pk} is out of range (0.0..1.0)");
        }
    }
    let still_eps = m.still_eps.unwrap_or(base.still_eps);
    if let Some(se) = still_eps {
        if !(0.0..=60.0).contains(&se) {
            anyhow::bail!("still_eps {se} is out of range (0.0..60.0)");
        }
    }
    let depth = match m.depth.as_deref() {
        None => base.depth,
        Some(s) => style::DepthUniformity::from_label(s)
            .with_context(|| format!("bad depth {s:?}"))?,
    };
    let depth_dose = m.depth_dose.unwrap_or(base.depth_dose);
    if let Some(v) = depth_dose {
        if !(0.0..=1.0).contains(&v) {
            anyhow::bail!("depth_dose {v} is out of range (0.0..1.0)");
        }
    }
    let depth_window = m.depth_window.unwrap_or(base.depth_window);
    if let Some(v) = depth_window {
        if !(0.0..=60.0).contains(&v) {
            anyhow::bail!("depth_window {v} is out of range (0.0..60.0 s)");
        }
    }
    let filler_gap_s = m.filler_gap_s.unwrap_or(base.filler_gap_s);
    if filler_gap_s < 0.0 {
        anyhow::bail!("filler_gap_s must be >= 0 (seconds; 0 = off)");
    }
    let filler_min_real_s = m.filler_min_real_s.unwrap_or(base.filler_min_real_s);
    let filler_real_v = m.filler_real_v.unwrap_or(base.filler_real_v);
    let filler_model_w = m.filler_model_w.unwrap_or(base.filler_model_w);
    let filler_rate = m.filler_rate.unwrap_or(base.filler_rate);
    let filler_amp = m.filler_amp.unwrap_or(base.filler_amp);
    let filler_ramp_s = m.filler_ramp_s.unwrap_or(base.filler_ramp_s);
    let filler_max_bridge_s = m.filler_max_bridge_s.unwrap_or(base.filler_max_bridge_s);
    let filler_sway = m.filler_sway.unwrap_or(base.filler_sway);
    let filler_sway_s = m.filler_sway_s.unwrap_or(base.filler_sway_s);
    let filler_pattern = match m.filler_pattern.as_deref() {
        None => base.filler_pattern,
        Some(s) => style::FillerPattern::from_label(s)
            .with_context(|| format!("bad filler_pattern {s:?}"))?,
    };
    let filler_burst = m.filler_burst.unwrap_or(base.filler_burst).max(1);
    let filler_rest_s = m.filler_rest_s.unwrap_or(base.filler_rest_s);
    if filler_gap_s > 0.0 && !(1.0..=300.0).contains(&filler_rate) {
        anyhow::bail!("filler_rate {filler_rate} is out of range (1..300)");
    }
    if filler_gap_s > 0.0 && !(1.0..=50.0).contains(&filler_amp) {
        anyhow::bail!("filler_amp {filler_amp} is out of range (1..50)");
    }
    if filler_min_real_s < 0.0
        || filler_real_v < 0.0
        || filler_ramp_s < 0.0
        || filler_max_bridge_s < 0.0
        || filler_rest_s < 0.0
    {
        anyhow::bail!("filler durations and the confidence bar must be >= 0");
    }
    if !(0.0..=1.0).contains(&filler_sway) || filler_sway_s <= 0.0 {
        anyhow::bail!("filler_sway must be 0..1 and filler_sway_s > 0");
    }
    if !(0.0..=1.0).contains(&filler_model_w) {
        anyhow::bail!("filler_model_w must be 0..1 (1 = full model influence)");
    }
    Ok(style::Params {
            style: sty,
            dwells,
            stillness,
            intensity,
            env_seed: m.env_seed.map(|s| s.unwrap_or(u64::MAX)).unwrap_or(base.env_seed),
            range: (lo, hi),
            max_speed,
            cut_ease,
            dwell_ramp,
            still_eps,
            depth,
            depth_dose,
            depth_window,
            filler_gap_s,
            filler_min_real_s,
            filler_real_v,
            filler_model_w,
            filler_rate,
            filler_amp,
            filler_ramp_s,
            filler_max_bridge_s,
            filler_sway,
            filler_sway_s,
            filler_pattern,
            filler_burst,
            filler_rest_s,
    })
}

/// Absent and `null` are two different answers -- absent keeps the base,
/// `null` clears an override -- and serde folds both into `None` unless the
/// field is nested one level deeper. This is that level.
fn nullable<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    <Option<T> as serde::Deserialize>::deserialize(d).map(Some)
}

pub(crate) fn header(field: &str, value: &str) -> Header {
    Header::from_bytes(field.as_bytes(), value.as_bytes()).expect("static header")
}

pub(crate) fn json_response(v: &serde_json::Value) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(v.to_string().into_bytes())
        .with_header(header("Content-Type", "application/json"))
}

/// By this point the path is one of ours: an MP4-family original, a WebM-codec
/// matroska (a WebM to the browser -- the type tag is what makes Firefox try),
/// or the cache's normalized copy.
pub(crate) fn video_ctype(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()) {
        Some(e) if e == "webm" || e == "mkv" => "video/webm",
        _ => "video/mp4",
    }
}

/// Stream a file off the main thread. The thread lives for one response;
/// playback issues a handful of requests a second at most.
pub(crate) fn serve_file_threaded(req: tiny_http::Request, path: PathBuf, ctype: &'static str) {
    std::thread::spawn(move || {
        let _ = serve_file(req, &path, ctype);
    });
}

/// One file request, with the Range handling browsers need to seek: a
/// `Range: bytes=..` request gets a 206 with a Content-Range, capped at
/// `RANGE_CHUNK`; no Range gets the whole file.
pub(crate) fn serve_file(req: tiny_http::Request, path: &Path, ctype: &'static str) -> Result<()> {
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => {
            let _ = req.respond(Response::from_string("gone").with_status_code(404));
            return Ok(());
        }
    };
    let len = f.metadata()?.len();
    let range = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("range"))
        .and_then(|h| parse_range(h.value.as_str(), len));
    match range {
        Some((a, b)) => {
            let b = b.min(a + RANGE_CHUNK - 1);
            f.seek(SeekFrom::Start(a))?;
            let n = b - a + 1;
            let resp = Response::new(
                StatusCode(206),
                vec![
                    header("Content-Type", ctype),
                    header("Accept-Ranges", "bytes"),
                    header("Content-Range", &format!("bytes {a}-{b}/{len}")),
                ],
                f.take(n),
                Some(n as usize),
                None,
            );
            let _ = req.respond(resp);
        }
        None => {
            let resp = Response::new(
                StatusCode(200),
                vec![header("Content-Type", ctype), header("Accept-Ranges", "bytes")],
                f,
                Some(len as usize),
                None,
            );
            let _ = req.respond(resp);
        }
    }
    Ok(())
}

/// `Range: bytes=a-b` / `bytes=a-` / `bytes=-n` -> inclusive `(start, end)`
/// clamped into the file, or `None` for anything unusable (a plain 200 is a
/// legal answer to any Range request).
fn parse_range(value: &str, len: u64) -> Option<(u64, u64)> {
    if len == 0 {
        return None;
    }
    let spec = value.trim().strip_prefix("bytes=")?;
    let spec = spec.split(',').next()?.trim(); // multipart ranges: first only
    let (a, b) = spec.split_once('-')?;
    let (a, b) = (a.trim(), b.trim());
    if a.is_empty() {
        // suffix form: the last n bytes
        let n: u64 = b.parse().ok()?;
        if n == 0 {
            return None;
        }
        Some((len.saturating_sub(n), len - 1))
    } else {
        let start: u64 = a.parse().ok()?;
        if start >= len {
            return None;
        }
        let end = if b.is_empty() { len - 1 } else { b.parse::<u64>().ok()?.min(len - 1) };
        (start <= end).then_some((start, end))
    }
}

/// Best effort -- the URL is printed either way.
pub(crate) fn open_browser(url: &str) {
    #[cfg(windows)]
    let r = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(not(windows))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    let _ = r;
}

#[cfg(test)]
mod tests {
    use super::{parse_range, Choice, Work};
    use crate::style;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// The queue's two promises, both of which a long video is the reason for.
    ///
    /// SUPERSEDE: knobs turned while a style is in flight collapse to one run,
    /// not a backlog. Without it a page that posts on every control change
    /// would queue a two-hour style per twitch and fall minutes behind.
    ///
    /// DRAIN: the review ending does not abandon outstanding work. Someone who
    /// turns a knob and hits Done a second later must still find that knob's
    /// script on disk.
    #[test]
    fn the_queue_supersedes_in_flight_work_and_drains_on_the_way_out() {
        let none = || super::Restyled { actions: 0, conf: Vec::new() };
        let work = Arc::new(Work::new(vec![none(), none()]));
        let runs = Arc::new(AtomicUsize::new(0));
        // a styler that blocks on a gate, so "in flight" is a state the test
        // can actually hold the queue in
        let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));

        // bumped on ENTRY, so the test can wait for the styler to be inside a
        // run rather than merely for the request to be queued
        let entered = Arc::new(AtomicUsize::new(0));
        let styler = {
            let (work, runs) = (Arc::clone(&work), Arc::clone(&runs));
            let (gate, entered) = (Arc::clone(&gate), Arc::clone(&entered));
            std::thread::spawn(move || {
                work.style_until_done(&|_i: usize, p: &style::Params, _c: &Choice| {
                    entered.fetch_add(1, Ordering::SeqCst);
                    let (lk, cv) = &*gate;
                    let mut open = lk.lock().unwrap();
                    while !*open {
                        open = cv.wait(open).unwrap();
                    }
                    runs.fetch_add(1, Ordering::SeqCst);
                    // the action count stands in for "which params ran"
                    Ok(super::Restyled { actions: (p.intensity * 100.0) as usize, conf: Vec::new() })
                })
            })
        };

        let at = |v: f64| {
            let mut p = style::Params::default();
            p.intensity = v;
            p
        };
        let c = Choice { model: "m".into(), compare: None };
        work.request(0, at(0.1), c.clone()); // the styler picks this up and blocks in it
        while entered.load(Ordering::SeqCst) < 1 {
            std::thread::yield_now();
        }
        // three more changes to the SAME clip while the first is running
        work.request(0, at(0.2), c.clone());
        work.request(0, at(0.3), c.clone());
        work.request(0, at(0.4), c);

        work.finish(); // Done, with work still outstanding
        let (lk, cv) = &*gate;
        *lk.lock().unwrap() = true;
        cv.notify_all();
        styler.join().unwrap();

        // the first run, plus ONE for the three that piled up behind it
        assert_eq!(runs.load(Ordering::SeqCst), 2, "the pile-up did not collapse to one run");
        // and the run that survived is the NEWEST ask, not the oldest
        assert_eq!(work.actions()[0], 40, "the queue kept a stale change over the newest");
        assert!(!work.busy(0), "nothing is outstanding once the styler has drained");
        assert!(work.error().is_none());
    }

    /// The page posts the model and the compared model with every knob, so
    /// absent keeps, `null` clears the compare, a script switched onto its
    /// compared model compares with nothing, and a name the script does not
    /// offer is refused before the styler sees it.
    #[test]
    fn a_choice_keeps_clears_and_refuses_as_the_page_means() {
        let m = |n: &str, t: &str| crate::bundle::ModelInfo {
            name: n.into(),
            trunk: t.into(),
        };
        let clip = super::Clip {
            name: "c".into(),
            script: "s".into(),
            video: "v".into(),
            fallback: None,
            is_original: true,
            duration_ms: 0.0,
            row_hz: 30.0,
            row0_ms: 0.0,
            t0_ms: 0.0,
            crop: None,
            model: "a".into(),
            models: vec![m("a", "t1"), m("b", "t1"), m("c", "t2")],
            compare: "k".into(),
        };
        let base = Choice { model: "a".into(), compare: Some("c".into()) };
        let parse = |body: &str| super::parse_choice(body, &clip, &base);
        assert_eq!(parse(r#"{"clip":0}"#).unwrap(), base);
        assert_eq!(parse(r#"{"model":"b"}"#).unwrap(), Choice { model: "b".into(), compare: Some("c".into()) });
        assert_eq!(parse(r#"{"compare":null}"#).unwrap().compare, None);
        assert_eq!(parse(r#"{"model":"c"}"#).unwrap(), Choice { model: "c".into(), compare: None });
        assert!(parse(r#"{"model":"z"}"#).is_err());
        assert!(parse(r#"{"compare":"z"}"#).is_err());
    }

    #[test]
    fn closed_and_open_ranges() {
        assert_eq!(parse_range("bytes=0-1023", 10_000), Some((0, 1023)));
        assert_eq!(parse_range("bytes=500-", 10_000), Some((500, 9_999)));
        assert_eq!(parse_range("bytes=-100", 10_000), Some((9_900, 9_999)));
    }

    #[test]
    fn ranges_clamp_into_the_file() {
        assert_eq!(parse_range("bytes=0-999999", 100), Some((0, 99)));
        assert_eq!(parse_range("bytes=-200", 100), Some((0, 99)));
        assert_eq!(parse_range("bytes=100-", 100), None); // start past the end
        assert_eq!(parse_range("bytes=5-2", 100), None);
    }

    #[test]
    fn garbage_means_no_range() {
        assert_eq!(parse_range("bytes=", 100), None);
        assert_eq!(parse_range("bytes=a-b", 100), None);
        assert_eq!(parse_range("items=0-5", 100), None);
        assert_eq!(parse_range("bytes=0-5", 0), None);
    }
}
