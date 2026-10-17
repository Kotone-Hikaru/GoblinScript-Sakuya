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
    /// The cuts this draft was DECODED against, video milliseconds on the
    /// draft's clock. The page draws them as the shot layer, and they are what
    /// "reset" goes back to: the draft's own set never changes, only the
    /// session's edit to it does.
    pub cuts_ms: Vec<f64>,
}

/// Which model writes a script, and which other model's line the page lays
/// under it. Both are part of the ask the styler works on: switching either
/// is a re-style, and a switch onto a trunk without tracks is a head pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub model: String,
    pub compare: Option<String>,
    /// Stretches written by a model OTHER than `model`. Empty -- every
    /// ordinary review -- means `model`'s tracks fill the whole clip; each
    /// span splices its own model's rows in over that base. Session state:
    /// the cuts a draft was encoded against belong to the draft, a span
    /// belongs to the reviewer and is never written back.
    pub spans: Vec<Span>,
    /// The shot cuts to decode with, when this session has edited them.
    /// `None` -- every ordinary review -- means the draft's own set. A change
    /// here reaches the head as its `cut` input channel, so the page treats it
    /// as work: the head pass re-runs, several seconds per trunk.
    pub cuts_ms: Option<Vec<f64>>,
}

/// The span "model" that is not a model: a manual fill. A piece wearing it is
/// written by a waveform the user dialled in (`Span::wave`) instead of by any
/// bundle, so it needs no head pass, no latents and no model at all -- and the
/// leading `__` keeps it out of every real pack name's way.
pub const MANUAL: &str = "__manual";

/// One stretch of a script written by its own model, in video milliseconds on
/// the clip's clock (`lo_ms` inclusive, `hi_ms` exclusive). The page edits
/// these; the styler splices the named model's tracks in over those rows.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Span {
    pub lo_ms: f64,
    pub hi_ms: f64,
    pub model: String,
    /// Present exactly when `model` is [`MANUAL`]: the waveform to write this
    /// piece with. Absent on every ordinary span, so an older page's spans and
    /// every session before this existed still parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wave: Option<crate::wave::WaveSpec>,
    /// Manual only, and ON when absent: write the piece at its own slopes
    /// rather than running the device's speed cap over it. A page built before
    /// this existed sends no field, and ignoring the cap is what a drawing
    /// wants -- the cap is what used to flatten a fast piece's peaks.
    #[serde(default = "yes")]
    pub no_limit: bool,
    /// Manual only, and ON when absent: leave the video's own shot cuts out of
    /// this piece. A cut inside a drawing is a cut in the *footage* -- the
    /// author drew through it, so the seam protecting every other piece would
    /// only chew the drawing up. A page built before this existed sends no
    /// field, and ignoring the cuts is what a drawing wants.
    #[serde(default = "yes")]
    pub no_cuts: bool,
    /// This piece's own intensity, 0-5, where 1 is the model's own output.
    ///
    /// Scaling is about the piece's OWN mean rather than mid-range, so a factor
    /// above 1 cannot push a low-lying waveform toward the floor -- which is
    /// what "stronger made it smaller" was. It applies to ordinary spans: a
    /// hand-drawn piece has amplitude knobs of its own and is left out of it.
    /// Absent means 1, so a page built before this existed and every stored
    /// session read as "unchanged".
    #[serde(default = "one")]
    pub scale: f64,
}

/// The biggest piece intensity the panel offers. Refused above this rather than
/// clamped silently, because the number came from the page and a page sending
/// nonsense should hear about it.
pub const MAX_PIECE_SCALE: f64 = 5.0;

fn yes() -> bool {
    true
}

fn one() -> f64 {
    1.0
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
/// The word `/whoami` answers with -- the marker `already_serving` looks for.
const WHOAMI: &str = "goblinscript-review";

/// Can we have `port`? If not, a free one. The remembered address is a hint
/// that may have expired, not a demand, so a port taken by something else costs
/// the user a new URL and nothing more.
fn bind_review(want: Option<u16>) -> Result<tiny_http::Server> {
    if let Some(p) = want {
        if let Ok(s) = tiny_http::Server::http(("127.0.0.1", p)) {
            return Ok(s);
        }
    }
    tiny_http::Server::http("127.0.0.1:0").map_err(|e| anyhow::anyhow!("{e}"))
}

/// Is one of OUR review servers already answering on `port`?
///
/// A bare connect would say yes to anything holding the port, so the question
/// goes to the one endpoint only we serve: `/whoami` answers with a fixed word.
/// Anything else on the port -- another program, or a review from a build that
/// predates `/whoami` -- reads as "not ours", and we take a fresh port rather
/// than hand this session to a stranger.
fn already_serving(port: u16) -> bool {
    use std::io::Write;
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut s) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(900)));
    let ask = format!("GET /whoami HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if s.write_all(ask.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 256];
    let Ok(n) = s.read(&mut buf) else { return false };
    String::from_utf8_lossy(&buf[..n]).contains(WHOAMI)
}

/// Identifies THIS server run. The page keeps the first value it sees and
/// reloads when it changes, which is how a tab left open on the remembered port
/// walks itself into the next session instead of sitting on a stale page.
pub(crate) fn boot_id() -> &'static str {
    static BOOT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    BOOT.get_or_init(|| {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{ns:x}")
    })
}

pub fn review(
    clips: &[Clip],
    params: &mut [style::Params],
    choices: &mut [Choice],
    presets: Presets,
    open: bool,
    page: &dyn Fn(&str),
    apply: impl Fn(usize, &style::Params, &Choice) -> Result<Restyled> + Send + Sync,
) -> Result<()> {
    // ---- one review at a time, at a remembered address ---------------------
    //
    // The address is remembered so a browser tab left open on it is not
    // stranded: the next run is served from the same URL, and the page's own
    // poll sees a new `boot` id and reloads itself into the new session. A port
    // something else now holds is a stale hint rather than a fight -- take a
    // free one and write that back. And when one of OUR reviews is already
    // answering, say so and leave it alone: two of them share a cache
    // directory, which is the one outcome worth refusing.
    //
    // `GOBLIN_REVIEW_FORCE` skips that last check, for a developer running
    // back-to-back sessions where "one is already up" is the normal state.
    let mut settings = crate::settings::Settings::load();
    let force = std::env::var_os("GOBLIN_REVIEW_FORCE").is_some();
    // A review of ours already answering on the remembered port is NOT a reason
    // to end this one. It used to be, and in a session started from the picker
    // that meant the user never saw a page at all: the notice explaining why was
    // printed and then wiped by the TUI drawing itself back, so the batch simply
    // returned to the picker -- indistinguishable from a review that was skipped.
    // A leftover server from an earlier run is enough to cause it, and leftovers
    // are exactly what a long-lived port attracts. Losing the address costs one
    // reload; losing the review costs the batch.
    let mut want = settings.review_port;
    if let Some(p) = settings.review_port {
        if !force && already_serving(p) {
            println!(
                "\n  {} {}",
                crate::t!("console.review.label"),
                console::style(crate::t!("console.review.already", port = p)).dim()
            );
            want = None;
        }
    }
    let server = bind_review(want)
        .map_err(|e| anyhow::anyhow!("could not start the review server: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("review server has no TCP address")?
        .port();
    if settings.review_port != Some(port) {
        settings.review_port = Some(port);
        settings.save();
    }
    let url = format!("http://127.0.0.1:{port}/");

    // the draft's own params, snapshot before any re-style, so the page's
    // per-clip "reset" can restore exactly what the goblins first drafted.
    let defaults: Vec<style::Params> = params.to_vec();
    let mut first: Vec<Restyled> = Vec::with_capacity(clips.len());
    for (i, p) in params.iter().enumerate() {
        first.push(apply(i, p, &choices[i])?);
    }

    page(&url);
    // The crop page for the last video is probably still up in the one window
    // this run opened, waiting for exactly this address. Send it here instead
    // of opening a second window; if nobody is waiting, open one.
    crate::pagewin::deliver(&url, open);

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

/// The page served at `/`.
///
/// A `review.html` sitting BESIDE THE EXE wins over the copy baked in at build
/// time. That is the whole point: a dev exe can be handed the page as a loose
/// file, so a layout tweak costs a copy and a browser reload instead of a
/// rebuild. Detected per request, not cached, so replacing the file is enough.
///
/// Deliberately narrow: the exe's own directory only. Walking up into the
/// source tree would mean a SHIPPED exe quietly serving a page from a checkout
/// it happens to sit next to -- and there the page and the binary are free to
/// disagree, which is the one thing this must never allow.
fn page_html() -> String {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Ok(s) = std::fs::read_to_string(dir.join("review.html")) {
                return s;
            }
        }
    }
    include_str!("review.html").to_string()
}

/// Percent-decoding, for the two query strings this server reads. Values are
/// numbers and easing names -- all ASCII -- so this stays deliberately small,
/// but it still refuses to guess: a malformed `%` is left alone rather than
/// mangling the name it is part of.
fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
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
    // Strip the query BEFORE splitting: without this a route whose name would
    // be the first path segment arrives as `wave_preview?id=...` and matches
    // nothing at all.
    let path = url.split_once('?').map(|(p, _)| p).unwrap_or(&url);
    let mut parts = path.trim_matches('/').split('/');
    match (req.method().clone(), parts.next().unwrap_or("")) {
        (Method::Get, "") => {
            let _ = req.respond(
                Response::from_string(page_html())
                    .with_header(header("Content-Type", "text/html; charset=utf-8")),
            );
        }
        // The single-instance check's own endpoint: a fixed word, so a process
        // starting up can tell "one of ours is already here" from "something
        // else happens to hold the port". Nothing else reads it.
        (Method::Get, "whoami") => {
            let _ = req.respond(Response::from_string(WHOAMI));
        }
        // Just the id of the run answering, for the page's idle poll. Kept
        // separate from `/state` on purpose: this is asked every few seconds
        // and has to stay a few bytes.
        (Method::Get, "ping") => {
            let _ = req.respond(Response::from_string(boot_id()));
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
        // The waveforms as data: ids, knobs, ranges, the shapes themselves.
        // Read per request (see wave.rs), so editing waves.json and reloading
        // the page IS the edit loop for a shape -- no rebuild anywhere in it.
        (Method::Get, "waves") => {
            let _ = req.respond(json_response(&crate::wave::value()));
        }
        // One cycle of one wave, sampled, for the panel's thumbnail. The page
        // sends its current knob values, so the curve it draws is the curve the
        // export will write -- the page owns no second copy of the maths.
        (Method::Get, "wave_preview") => {
            let q = req
                .url()
                .split_once('?')
                .map(|(_, q)| q.to_string())
                .unwrap_or_default();
            let mut id = String::new();
            let mut n = 160usize;
            let mut given: crate::wave::Params = Default::default();
            for kv in q.split('&').filter(|s| !s.is_empty()) {
                let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                let v = url_decode(v);
                if k == "id" {
                    id = v;
                } else if k == "n" {
                    n = v.parse().unwrap_or(160);
                } else if let Ok(num) = v.parse::<f64>() {
                    given.insert(k.to_string(), crate::wave::Val::Num(num));
                } else {
                    given.insert(k.to_string(), crate::wave::Val::Str(v));
                }
            }
            let body = match crate::wave::load().get(&id) {
                Some(w) => {
                    let p = w.resolve(&given);
                    serde_json::json!({
                        "id": id,
                        "periodic": w.periodic,
                        "v": w.sample(&p, n.min(2048)),
                    })
                }
                None => serde_json::json!({ "error": format!("no wave {id}") }),
            };
            let _ = req.respond(json_response(&body));
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
        // The confidence suggestion's two standing preferences: which models
        // are allowed to compete, and whether the page should run the
        // suggestion by itself the moment a clip is up. Both go through
        // settings.json rather than page state, because both are about the
        // person rather than about this clip -- they have to survive a reload,
        // a different browser, and a cleared cache.
        (Method::Get, "conf_prefs") => {
            let s = crate::settings::Settings::load();
            let _ = req.respond(json_response(&conf_prefs_json(&s)));
        }
        (Method::Post, "conf_prefs") => {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            match parse_conf_prefs(&body) {
                Ok(p) => {
                    let mut s = crate::settings::Settings::load();
                    s.exclude_models = p.exclude;
                    // absent = the page is only speaking about exclusions
                    if let Some(a) = p.auto {
                        s.auto_suggest = a;
                    }
                    if let Some(g) = p.gamma {
                        s.conf_gamma = Some(g);
                    }
                    s.save();
                    let _ = req.respond(json_response(&conf_prefs_json(&s)));
                }
                Err(e) => {
                    let _ = req
                        .respond(Response::from_string(format!("{e:#}")).with_status_code(400));
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
        // The intensity slider's ceilings, so the page's unlock boxes and the
        // server's own accept-check are the same four numbers rather than two
        // copies that can drift. `base` is the everyday one a fresh page shows.
        "intensity": {
            "min": style::INTENSITY_MIN,
            "base": style::INTENSITY_MAX,
            "tiers": [style::INTENSITY_UNLOCKED_MAX, style::INTENSITY_UNLOCKED_100],
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

/// The page's two standing preferences about the confidence suggestion, in the
/// shape it reads them: `exclude` is the list of models taken out of the vote
/// (`null` for "none -- everything competes"), `auto` is whether the page runs
/// the suggestion on its own when a clip comes up.
pub(crate) fn conf_prefs_json(s: &crate::settings::Settings) -> serde_json::Value {
    serde_json::json!({
        "exclude": s.exclude_models,
        "auto": s.auto_suggest,
        // absent means "never chosen", which the page reads as an exponent of 1
        "gamma": s.conf_gamma,
    })
}

/// The same two fields coming back from the page.
///
/// `exclude` may be absent, `null`, or a list of names; anything else is a
/// mistake worth a 400 rather than a silent reset -- a typo that quietly
/// un-excludes a model would put it back on the ballot without saying so.
/// `auto` accepts a real boolean, or a string of the kinds a page written by
/// hand might send, and `None` means "not mine to say": the review page READS
/// the auto-suggest preference but no longer has a switch for it (it lives in
/// the picker now), so a page that leaves the field out must not be read as
/// turning it off behind the picker's back.
/// What the page may say about the suggestion's standing preferences.
///
/// Every field is an `Option` because "I am not speaking about this one" has to
/// be distinguishable from "set it to the default": the page owns some of these
/// and another surface owns others, and a view that stays silent must never
/// overwrite the one that spoke.
pub(crate) struct ConfPrefs {
    pub exclude: Option<Vec<String>>,
    pub auto: Option<bool>,
    pub gamma: Option<f64>,
}

pub(crate) fn parse_conf_prefs(body: &str) -> Result<ConfPrefs> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| anyhow::anyhow!("bad params: {e}"))?;
    let exclude = match v.get("exclude") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Array(a)) => Some(
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect::<Vec<_>>(),
        ),
        Some(_) => anyhow::bail!("exclude must be a list or null"),
    };
    let auto = match v.get("auto") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Bool(b)) => Some(*b),
        Some(serde_json::Value::String(s)) => {
            Some(matches!(s.as_str(), "1" | "true" | "yes" | "on"))
        }
        Some(_) => anyhow::bail!("auto must be a boolean"),
    };
    // The page's own trim on how loudly a model's confidence reads. Bounded to
    // the range the control offers: a value outside it is a bug somewhere, and
    // saying so beats clamping to a number nobody typed.
    let gamma = match v.get("gamma") {
        None | Some(serde_json::Value::Null) => None,
        Some(x) => match x.as_f64() {
            Some(g) if (0.25..=2.0).contains(&g) => Some(g),
            _ => anyhow::bail!("gamma must be a number between 0.25 and 2"),
        },
    };
    Ok(ConfPrefs { exclude, auto, gamma })
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
        // which server run this is: the page keeps the first one it sees and
        // reloads itself when it changes, so a tab left on the remembered port
        // follows the next session instead of going stale
        "boot": boot_id(),
        // the strip menu's one-click fills, wave specs already built
        "fills": crate::wave::fills_json(),
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
            "spans": choices[i].spans,
            // the cuts this draft was decoded against (the shot layer draws
            // them, and "reset" goes back to them), and this session's
            // replacement when it has one -- null means "the draft's own"
            "cuts_ms": c.cuts_ms,
            "cuts_cur": choices[i].cuts_ms,
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
        #[serde(default)]
        spans: Option<Vec<Span>>,
        // absent keeps, `null` goes back to the draft's own cuts, an array
        // replaces them
        #[serde(default, deserialize_with = "nullable")]
        cuts_ms: Option<Option<Vec<f64>>>,
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
    // spans absent keeps the clip's own; present replaces them whole (the
    // page sends the full list on every edit, so no per-span merging)
    let spans = match m.spans {
        Some(s) => normalize_spans(s, clip),
        None => base.spans.clone(),
    };
    let cuts_ms = match m.cuts_ms {
        Some(c) => c.map(|v| normalize_cuts(v, clip.duration_ms)),
        None => base.cuts_ms.clone(),
    };
    Ok(Choice { model, compare, spans, cuts_ms })
}

/// Clean a span list off a page message: clamp into the clip, drop empties,
/// drop a model the clip does not offer, and sort by start. The page's
/// contract is sorted and non-overlapping; the server never trusts a client
/// to hold it -- a later span simply wins its rows when they are spliced.
fn normalize_spans(spans: Vec<Span>, clip: &Clip) -> Vec<Span> {
    let dur = clip.duration_ms;
    let mut out: Vec<Span> = Vec::new();
    for mut s in spans {
        // A manual piece names no bundle, so it is exempt from the model test
        // -- but it has to carry a recipe, or there is nothing to write it
        // with and it would silently become a model piece.
        if s.model == MANUAL {
            match &s.wave {
                Some(w) => {
                    if w.id.is_empty() {
                        continue;
                    }
                }
                None => continue,
            }
        } else if !clip.models.iter().any(|m| m.name == s.model) {
            continue;
        }
        s.lo_ms = s.lo_ms.clamp(0.0, dur);
        s.hi_ms = s.hi_ms.clamp(0.0, dur);
        if s.hi_ms <= s.lo_ms {
            continue;
        }
        // A NaN from the page would poison the mean it is scaled about, and a
        // piece at 1.0 is the model's own output -- both ends of the range the
        // panel offers are real settings, so only the nonsensical ones are
        // pulled back in.
        if !s.scale.is_finite() {
            s.scale = 1.0;
        }
        s.scale = s.scale.clamp(0.0, MAX_PIECE_SCALE);
        out.push(s);
    }
    out.sort_by(|a, b| {
        a.lo_ms.partial_cmp(&b.lo_ms).unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Clean a cut list off a page message: finite, inside the clip, sorted, and
/// deduplicated. `shot_edges` clamps and dedupes too, but `cut_rows` does not,
/// and a NaN or a negative here would be a row the head is asked about that
/// does not exist -- so the scrubbing happens once, at the door. The order is
/// the page's to keep or not: this sorts before it collapses neighbours, so a
/// descending list is cleaned the same as an ascending one.
fn normalize_cuts(cuts: Vec<f64>, dur_ms: f64) -> Vec<f64> {
    let mut out: Vec<f64> = cuts
        .into_iter()
        .filter(|c| c.is_finite() && *c > 0.0 && *c < dur_ms)
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    out
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
    // The page's slider is the user's own guard rail: it stops at
    // `INTENSITY_MAX` until an unlock box is ticked, then reaches
    // `INTENSITY_UNLOCKED_MAX` or `INTENSITY_UNLOCKED_100`. The server accepts
    // up to the far ceiling for every request (it cannot see the boxes, and the
    // requests are the user's own), so the narrower everyday bound stays a UI
    // decision.
    if !(style::INTENSITY_MIN..=style::INTENSITY_UNLOCKED_100).contains(&intensity) {
        anyhow::bail!(
            "intensity {intensity} is out of range ({}..{})",
            style::INTENSITY_MIN,
            style::INTENSITY_UNLOCKED_100
        );
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
/// Open the review page in the machine's browser.
///
/// When the user has asked for the GPU browser (a standing preference in
/// settings.json), our own window is tried first and the system default is the
/// fallback -- so a machine with no browser we recognise still gets a page
/// rather than silence.
pub(crate) fn open_browser(url: &str) {
    if crate::settings::Settings::load().gpu_browser && open_browser_gpu(url) {
        return;
    }
    #[cfg(windows)]
    let r = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(not(windows))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    let _ = r;
}

/// The private profile the GPU window lives in.
///
/// NOT optional, and that is the whole trick: an already-running Chrome hands
/// the request to the running process and drops the command line on the floor,
/// so `--force_high_performance_gpu` would do nothing and the user would
/// conclude the switch is broken. A directory nothing else uses means the
/// browser we start is really ours. It sits beside the exe, so removing the
/// feature is removing one folder.
fn gpu_profile_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("gpu-browser")
}

/// The browser to launch, in the order we would rather have it.
///
/// Edge BEFORE Chrome, and not out of preference. `--force_high_performance_gpu`
/// is only a hint: a per-executable entry under
/// `HKCU\Software\Microsoft\DirectX\UserGpuPreferences` outranks it, and that
/// entry is exactly what someone does to Chrome on a hybrid machine when they
/// watch video on the iGPU and want the battery quiet. Measured here: the same
/// flag reported `AMD Radeon(TM) Graphics` on Chrome (its entry says
/// `GpuPreference=1`, power saving) and `NVIDIA GeForce RTX 3080 Ti` on Edge
/// (no entry). A browser we do not have to fight is worth more than a brand --
/// and the page cannot tell the difference, it is the same engine.
fn find_chromium() -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    let mut push = |base: Option<std::ffi::OsString>, rest: &[&str]| {
        if let Some(b) = base {
            let mut p = PathBuf::from(b);
            for r in rest {
                p.push(r);
            }
            cands.push(p);
        }
    };
    push(
        std::env::var_os("ProgramFiles(x86)"),
        &["Microsoft", "Edge", "Application", "msedge.exe"],
    );
    push(
        std::env::var_os("ProgramFiles"),
        &["Microsoft", "Edge", "Application", "msedge.exe"],
    );
    push(
        std::env::var_os("ProgramFiles"),
        &["Google", "Chrome", "Application", "chrome.exe"],
    );
    push(
        std::env::var_os("ProgramFiles(x86)"),
        &["Google", "Chrome", "Application", "chrome.exe"],
    );
    push(
        std::env::var_os("LOCALAPPDATA"),
        &["Google", "Chrome", "Application", "chrome.exe"],
    );
    cands.into_iter().find(|p| p.is_file())
}

/// Start a browser WE own, aimed at the discrete GPU, in app mode (no address
/// bar, no tabs) so it reads as one of our windows rather than another browser.
/// Returns false when there is no browser we know how to launch.
fn open_browser_gpu(url: &str) -> bool {
    #[cfg(not(windows))]
    {
        let _ = url;
        false
    }
    #[cfg(windows)]
    {
        let Some(exe) = find_chromium() else {
            return false;
        };
        let profile = gpu_profile_dir();
        let _ = std::fs::create_dir_all(&profile);
        std::process::Command::new(exe)
            .arg("--force_high_performance_gpu")
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg(format!("--user-data-dir={}", profile.display()))
            .arg(format!("--app={url}"))
            .spawn()
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_conf_prefs, parse_range, Choice, Work};
    use crate::style;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// The review page's exclusions may be posted without the auto-suggest
    /// switch -- it moved to the picker, and the page has no switch for it --
    /// so leaving the field out has to mean "no opinion", NOT "off".
    ///
    /// Reading an absent `auto` as `false` turns every exclusion tick into a
    /// silent switch-off of a preference the reviewer set in another window,
    /// which is exactly the kind of failure nobody would trace back here.
    #[test]
    fn a_page_that_says_nothing_about_auto_suggest_leaves_it_alone() {
        let p = parse_conf_prefs(r#"{"exclude":["anime-v1"]}"#).unwrap();
        assert_eq!(p.exclude.as_deref(), Some(&["anime-v1".to_string()][..]));
        assert_eq!(p.auto, None, "an absent switch must not read as OFF");
        assert_eq!(p.gamma, None, "nor may an absent exponent");
        // `null` is the same statement in JSON's own spelling
        let p = parse_conf_prefs(r#"{"exclude":null,"auto":null}"#).unwrap();
        assert_eq!(p.auto, None);
        // and where the page DOES speak, it is believed
        let p = parse_conf_prefs(r#"{"exclude":null,"auto":true}"#).unwrap();
        assert_eq!(p.auto, Some(true));
        let p = parse_conf_prefs(r#"{"exclude":null,"auto":"on"}"#).unwrap();
        assert_eq!(p.auto, Some(true), "a hand-posted string still reads");
    }

    /// The other half of the same contract: an `exclude` that is neither a
    /// list nor `null` is a 400 rather than a shrug. Quietly reading a typo as
    /// "nothing excluded" puts every model back on the ballot, and the
    /// suggestion the reviewer gets is then one they had already ruled out.
    #[test]
    fn a_broken_exclusion_list_is_refused_rather_than_ignored() {
        assert!(parse_conf_prefs(r#"{"exclude":7}"#).is_err());
        assert!(parse_conf_prefs(r#"{"exclude":"anime-v1"}"#).is_err());
        assert!(parse_conf_prefs("not json at all").is_err());
    }

    /// The confidence exponent is bounded to the range the control offers, and
    /// a value outside it is refused rather than silently clamped: a number
    /// that came back different from the one that was sent leaves the slider
    /// and the file disagreeing with nobody the wiser.
    #[test]
    fn the_confidence_exponent_is_bounded_and_refused_when_out_of_range() {
        assert_eq!(parse_conf_prefs(r#"{"gamma":0.5}"#).unwrap().gamma, Some(0.5));
        assert_eq!(parse_conf_prefs(r#"{"gamma":2}"#).unwrap().gamma, Some(2.0));
        assert!(parse_conf_prefs(r#"{"gamma":0.1}"#).is_err(), "below the floor");
        assert!(parse_conf_prefs(r#"{"gamma":4}"#).is_err(), "above the ceiling");
        assert!(
            parse_conf_prefs(r#"{"gamma":"2"}"#).is_err(),
            "a number, not a string"
        );
    }

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
        let c = Choice { model: "m".into(), compare: None, spans: Vec::new(), cuts_ms: None };
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
            cuts_ms: Vec::new(),
        };
        let base = Choice { model: "a".into(), compare: Some("c".into()), spans: Vec::new(), cuts_ms: None };
        let parse = |body: &str| super::parse_choice(body, &clip, &base);
        assert_eq!(parse(r#"{"clip":0}"#).unwrap(), base);
        assert_eq!(parse(r#"{"model":"b"}"#).unwrap(), Choice { model: "b".into(), compare: Some("c".into()), spans: Vec::new(), cuts_ms: None });
        assert_eq!(parse(r#"{"compare":null}"#).unwrap().compare, None);
        assert_eq!(parse(r#"{"model":"c"}"#).unwrap(), Choice { model: "c".into(), compare: None, spans: Vec::new(), cuts_ms: None });
        assert!(parse(r#"{"model":"z"}"#).is_err());
        assert!(parse(r#"{"compare":"z"}"#).is_err());
    }

    /// Spans ride the same message: absent keeps the clip's own, present
    /// replaces them whole, and the server clamps, drops unknown models and
    /// sorts before the styler ever sees them.
    #[test]
    fn a_choice_carries_spans_and_cleans_them() {
        let m = |n: &str| crate::bundle::ModelInfo { name: n.into(), trunk: "t".into() };
        let clip = super::Clip {
            name: "c".into(),
            script: "s".into(),
            video: "v".into(),
            fallback: None,
            is_original: true,
            duration_ms: 1000.0,
            row_hz: 30.0,
            row0_ms: 0.0,
            t0_ms: 0.0,
            crop: None,
            model: "a".into(),
            models: vec![m("a"), m("b")],
            compare: "k".into(),
            cuts_ms: Vec::new(),
        };
        let base = Choice {
            model: "a".into(),
            compare: None,
            spans: vec![super::Span { lo_ms: 0.0, hi_ms: 100.0, model: "b".into(), wave: None, no_limit: true, no_cuts: false, scale: 1.0 }],
            cuts_ms: None,
        };
        let parse = |body: &str| super::parse_choice(body, &clip, &base);
        // absent keeps the clip's own
        assert_eq!(parse(r#"{"model":"a"}"#).unwrap().spans, base.spans);
        // present replaces whole, and is cleaned: unknown model dropped,
        // inverted dropped, out-of-clip clamped, then sorted by start
        let got = parse(
            r#"{"spans":[
                {"lo_ms":900,"hi_ms":5000,"model":"b"},
                {"lo_ms":300,"hi_ms":400,"model":"zzz"},
                {"lo_ms":800,"hi_ms":900,"model":"b"},
                {"lo_ms":600,"hi_ms":700,"model":"a"}
            ]}"#,
        )
        .unwrap()
        .spans;
        assert_eq!(
            got,
            vec![
                super::Span { lo_ms: 600.0, hi_ms: 700.0, model: "a".into(), wave: None, no_limit: true, no_cuts: true, scale: 1.0 },
                super::Span { lo_ms: 800.0, hi_ms: 900.0, model: "b".into(), wave: None, no_limit: true, no_cuts: true, scale: 1.0 },
                super::Span { lo_ms: 900.0, hi_ms: 1000.0, model: "b".into(), wave: None, no_limit: true, no_cuts: true, scale: 1.0 },
            ]
        );
        // an empty list is a real answer: back to one model over the whole clip
        assert!(parse(r#"{"spans":[]}"#).unwrap().spans.is_empty());
    }

    /// Cuts ride the same message as spans: absent keeps the session's edit,
    /// `null` explicitly goes back to the draft's own set, and an array
    /// replaces it -- scrubbed (finite, inside the clip, sorted, collapsed)
    /// before the head is ever asked about a row that does not exist.
    #[test]
    fn a_choice_carries_cuts_and_cleans_them() {
        let m = |n: &str| crate::bundle::ModelInfo { name: n.into(), trunk: "t".into() };
        let clip = super::Clip {
            name: "c".into(),
            script: "s".into(),
            video: "v".into(),
            fallback: None,
            is_original: true,
            duration_ms: 2000.0,
            row_hz: 30.0,
            row0_ms: 0.0,
            t0_ms: 0.0,
            crop: None,
            model: "a".into(),
            models: vec![m("a")],
            compare: "k".into(),
            // the draft's own set: what "reset" goes back to
            cuts_ms: vec![500.0, 1500.0],
        };
        let base = Choice {
            model: "a".into(),
            compare: None,
            spans: Vec::new(),
            // this session has already moved a boundary
            cuts_ms: Some(vec![400.0, 1600.0]),
        };
        let parse = |body: &str| super::parse_choice(body, &clip, &base);
        // absent keeps the session's edit
        assert_eq!(parse(r#"{"model":"a"}"#).unwrap().cuts_ms, base.cuts_ms);
        // null goes back to the draft's own: the draft's set never changed
        assert_eq!(parse(r#"{"cuts_ms":null}"#).unwrap().cuts_ms, None);
        // an array replaces, cleaned: out-of-clip and negative dropped, a
        // descending list sorted, near-duplicates collapsed
        let got = parse(r#"{"cuts_ms":[1200, 300, 300.4, 5000, -10, 1200.2]}"#)
            .unwrap()
            .cuts_ms
            .unwrap();
        assert_eq!(got, vec![300.0, 1200.0]);
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

    /// The page's state lives in flush-left `let`s and its helpers read them.
    /// A helper that names one of its own locals after that state does not
    /// shadow "from there down" as it reads -- it shadows for the WHOLE body, so
    /// any read ABOVE the local declaration runs in the temporal dead zone and
    /// throws. That is exactly how the shot bench shipped dead: `shotLoad` opened
    /// with `state.clips[cur]` and closed with `const cur = ...`, so the very
    /// first line threw on every load and the only trace was a console error.
    ///
    /// So the rule this pins down: inside one function body, a local that
    /// shadows page state must not have that state read above it. Comments and
    /// blocks that shadow without reading the outer name first are fine, and a
    /// same-named local in a DIFFERENT function is fine too -- only the read
    /// that a shadowing declaration re-scopes is a hazard.
    ///
    /// Bodies are found the way the page writes them: a flush-left `function`,
    /// `async function`, or `... => {` opens one and the next flush-left `}`
    /// closes it. Parameters are not modelled; only `let`/`const`/`var`, which
    /// is what a name-the-local-after-state mistake produces.
    #[test]
    fn no_helper_shadows_the_pages_own_state() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let html = std::fs::read_to_string(dir.join("review.html")).expect("the review page is missing");
        let all: Vec<&str> = html.lines().collect();
        let open = all
            .iter()
            .position(|l| l.trim_start().starts_with("<script>"))
            .expect("review.html has no <script>");
        let close = all[open..]
            .iter()
            .position(|l| l.trim_start().starts_with("</script>"))
            .map(|i| open + i)
            .unwrap_or(all.len());

        fn word(line: &str, name: &str) -> bool {
            let wordish = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
            let mut from = 0;
            while from < line.len() {
                let Some(at) = line[from..].find(name).map(|p| from + p) else { return false };
                let before = at == 0 || !line[..at].chars().next_back().is_some_and(wordish);
                let after = at + name.len();
                let done = after == line.len() || !line[after..].chars().next().is_some_and(wordish);
                if before && done {
                    return true;
                }
                from = at + 1;
                while from < line.len() && !line.is_char_boundary(from) {
                    from += 1;
                }
            }
            false
        }
        fn declared(line: &str) -> Option<(&str, String)> {
            let t = line.trim_start();
            if t.len() == line.len() {
                return None; // flush-left: that is the state itself
            }
            let mut it = t.split_whitespace();
            let kw = it.next()?;
            if !matches!(kw, "let" | "const" | "var") {
                return None;
            }
            let name: String = it
                .next()?
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
                .collect();
            name.chars().next()?.is_alphabetic().then_some((kw, name))
        }

        /// Every name a flush-left declaration binds: `let a = [], b = [];`
        /// binds two. Only `=` at bracket depth zero counts, so the arrow in
        /// `let $ = (id) => ...` and anything inside an initializer are skipped.
        fn bound(line: &str) -> Vec<String> {
            let mut out = Vec::new();
            let mut depth = 0i32;
            for (i, b) in line.bytes().enumerate() {
                match b {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b'=' if depth == 0 => {
                        let head = line[..i].trim_end();
                        let name: String = head
                            .chars()
                            .rev()
                            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
                            .collect::<Vec<char>>()
                            .into_iter()
                            .rev()
                            .collect();
                        let lead = head
                            .chars()
                            .next_back()
                            .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '$');
                        // ...and the name has to sit in the binding list, so an
                        // arrow's parameter (`const $ = id => ...`) is not state
                        let tail = head[..head.len() - name.len()].trim_end();
                        let listed = tail.ends_with("let")
                            || tail.ends_with("const")
                            || tail.ends_with("var")
                            || tail.ends_with(',');
                        if lead && listed && !tail.ends_with('.') {
                            out.push(name);
                        }
                    }
                    _ => {}
                }
            }
            out
        }

        // page state: every flush-left declaration in the script
        let mut state: Vec<String> = Vec::new();
        for line in &all[open..close] {
            if line.starts_with(char::is_whitespace) {
                continue;
            }
            if !matches!(line.split_whitespace().next(), Some("let" | "const" | "var")) {
                continue;
            }
            for name in bound(line) {
                if !state.contains(&name) {
                    state.push(name);
                }
            }
        }

        let mut bad: Vec<String> = Vec::new();
        let mut i = open;
        while i < close {
            let head = all[i];
            let opens = !head.starts_with(char::is_whitespace)
                && (head.starts_with("function ")
                    || head.starts_with("async function ")
                    || head.contains("=> {"));
            if !opens {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < close && !all[j].starts_with('}') {
                j += 1;
            }
            let last = j.min(close - 1);
            for (k, line) in all[i..=last].iter().enumerate() {
                let ln = i + k;
                let Some((kw, name)) = declared(line) else { continue };
                if !state.contains(&name) {
                    continue;
                }
                if all[i..ln].iter().any(|above| word(above, &name)) {
                    bad.push(format!("review.html:{} `{kw} {name}` shadows state read above it", ln + 1));
                }
            }
            i = last + 1;
        }

        assert!(
            bad.is_empty(),
            "a local shadows the page's own state, so the read above it throws in the temporal \
             dead zone -- rename the local:\n  {}",
            bad.join("\n  ")
        );
    }

    /// A flush-left line that reads a binding the script declares FURTHER DOWN
    /// dies in the temporal dead zone, and a dead top-level line stops the WHOLE
    /// script -- `load()` included, so the page comes up empty and says nothing
    /// about why. That is how the crop page once shipped: a
    /// `tl.addEventListener(...)` sat above `const tl = $('tl')`, and every
    /// later step of that page was silently missing.
    ///
    /// The guard above catches the shadowing form of the same trap; this catches
    /// the ordering form. Only what a top-level statement reads WHERE IT STANDS
    /// is examined, so a binding used inside a function body or an arrow is left
    /// alone -- those run later, which is the normal way a page is written.
    #[test]
    fn no_top_level_line_reads_a_binding_declared_below_it() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

        fn ident(s: &str) -> &str {
            let n = s
                .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$'))
                .unwrap_or(s.len());
            &s[..n]
        }
        /// Every name a flush-left `let`/`const`/`var` line binds: `let a = [],
        /// b = []` binds two. Only commas at bracket depth zero separate, so a
        /// list inside an initializer is not mistaken for more bindings.
        fn names(line: &str) -> Vec<String> {
            let mut chunks = Vec::new();
            let mut depth = 0i32;
            let mut start = 0usize;
            for (i, b) in line.bytes().enumerate() {
                match b {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b',' if depth == 0 => {
                        chunks.push(&line[start..i]);
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            chunks.push(&line[start..]);
            let mut out = Vec::new();
            for c in chunks {
                let c = c.trim();
                let c = ["let ", "const ", "var "]
                    .iter()
                    .find_map(|k| c.strip_prefix(k))
                    .unwrap_or(c)
                    .trim();
                let name = ident(c);
                // a destructuring pattern binds names this cannot chase, and a
                // lone `$` is not a name anyone reads back out of a statement
                if name.chars().next().is_some_and(|c| c.is_alphabetic()) {
                    out.push(name.to_string());
                }
            }
            out
        }

        let mut bad = Vec::new();
        for page in ["review.html", "cropedit.html", "vr.html"] {
            let html = std::fs::read_to_string(dir.join(page)).expect("a page is missing");
            let all: Vec<&str> = html.lines().collect();
            let open = all
                .iter()
                .position(|l| l.trim_start().starts_with("<script>"))
                .expect("the page has no <script>");
            let close = all[open..]
                .iter()
                .position(|l| l.trim_start().starts_with("</script>"))
                .map(|i| open + i)
                .unwrap_or(all.len());

            let mut declared: Vec<(String, usize)> = Vec::new();
            for (i, line) in all[open..close].iter().enumerate() {
                if line.starts_with(char::is_whitespace) {
                    continue; // indented: not a top-level declaration
                }
                if !matches!(line.split_whitespace().next(), Some("let" | "const" | "var")) {
                    continue;
                }
                for name in names(line) {
                    declared.push((name, open + i));
                }
            }

            for (i, line) in all[open..close].iter().enumerate() {
                if line.starts_with(char::is_whitespace) {
                    continue; // indented: inside a body, so it runs later
                }
                let head = ident(line);
                // `x.y`, `x(...)`, `x[...]` is a read of x where it stands; a
                // bare word is a declaration, a label, or nothing at all
                if head.is_empty() || !line[head.len()..].starts_with(['.', '(', '[']) {
                    continue;
                }
                if let Some((_, at)) = declared.iter().find(|(n, _)| n == head) {
                    if *at > open + i {
                        bad.push(format!(
                            "{page}:{} reads `{head}` at top level, but it is declared at line {}",
                            open + i + 1,
                            at + 1
                        ));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "a top-level line reads a binding declared below it, so the script stops there in \
             the temporal dead zone and the page never finishes booting:\n  {}",
            bad.join("\n  ")
        );
    }
}
