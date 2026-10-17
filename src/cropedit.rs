//! The optional crop review: see what the goblins will be shown, and move it.
//!
//! `--crop-edit` puts the auto-crop plan in front of a person BEFORE the
//! expensive stage reads a single frame through it. The page plays the
//! normalized copy with each shot's rect drawn on it, and the rect is
//! draggable: a hand-drawn one replaces the attention's own for that shot, for
//! every shot, or nowhere at all.
//!
//! Three things make this cheap rather than a second pipeline:
//!
//! * **It runs between the probe and the encode.** The normalized copy already
//!   exists (the crop is applied to it), the plan already exists, and nothing
//!   downstream has been computed yet -- so a correction costs the person's
//!   attention and no GPU seconds at all. Correcting AFTER a draft would mean
//!   re-encoding the whole video.
//! * **The rect is already the page's language.** A plan carries fractions of
//!   the frame (`autocrop::Rect`), which is exactly what a browser overlay
//!   wants, so the drawing, the saving and the decode all read one number.
//! * **The correction outlives the recipe.** A hand-drawn plan is written to
//!   the same `autocrop.json` the probe writes, marked `manual`, and
//!   `autocrop::read_cached` then keeps it through a new checkpoint or a
//!   retuned constant. The aim was made against the picture, not against the
//!   recipe.
//!
//! What the page deliberately does NOT do is move a rect WITHIN a shot on its
//! own: a rect changes where a segment starts, and a segment start is a place
//! in the video a person has to name. So the page CAN add, move and remove
//! those starts -- a long shot whose subject walks out of the rect is exactly
//! the case a detector cannot call and a person can -- and what it still never
//! invents is a rect that drifts mid-segment.
//!
//! Those starts are the crop's own: they are seeded from the detected cuts, but
//! the models read the video through `cuts` themselves, so splitting a shot
//! here changes what a person is shown and nothing about what the goblins see.
//! Two different questions, two different lists, deliberately.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tiny_http::{Method, Response};

use crate::autocrop::{Plan, Rect, IDENTITY};
use crate::review::{header, json_response, serve_file_threaded, video_ctype};

/// What the page edits and what the caller gets back.
struct Session {
    /// One entry per shot, in plan order: where it starts (frame), and the
    /// rect it is to be drafted with.
    segs: Vec<(usize, Rect)>,
    /// The probe's own rects for those same shots -- what "auto" restores.
    auto: Vec<(usize, Rect)>,
    /// A rect here was drawn by hand rather than voted for.
    hand: Vec<bool>,
    /// A segment START was placed or moved here rather than detected. Only
    /// these make the page hand back a plan it otherwise would not: a plan
    /// whose rects are all the probe's is the cache entry that already exists.
    moved: bool,
    fps: f64,
    dur_ms: f64,
    /// The frame's own aspect (w/h), so the page can keep a drawn rect the
    /// shape of the source -- the shape the encoder's square squash expects.
    aspect: f64,
    done: bool,
    /// The page itself was asked for at least once. A run nobody is watching
    /// (`--no-browser`, no one at the address) has no window to steer, so it
    /// must not leave a finished server parked behind it.
    seen: bool,
}

impl Session {
    fn frames(&self) -> f64 {
        (self.dur_ms / 1000.0 * self.fps).max(1.0)
    }
    /// A start the page may place: on the frame grid, inside the clip, and
    /// strictly between its neighbours. Every edit lands through here, so the
    /// page cannot make a list `SegmentedDecoder::open` would refuse.
    fn clamp_frame(&self, i: usize, frame: f64) -> Option<usize> {
        if !frame.is_finite() {
            return None;
        }
        let last = self.frames() - 1.0;
        // segment 0 is the clip's own start and is not a boundary
        let lo = if i == 0 { 0.0 } else { self.segs[i - 1].0 as f64 + 1.0 };
        let hi = if i + 1 < self.segs.len() {
            self.segs[i + 1].0 as f64 - 1.0
        } else {
            last
        };
        if hi < lo {
            return None;
        }
        Some(frame.round().clamp(lo, hi) as usize)
    }
    fn json(&self) -> serde_json::Value {
        let segs: Vec<serde_json::Value> = self
            .segs
            .iter()
            .zip(&self.auto)
            .zip(&self.hand)
            .map(|(((f, r), (_, a)), &hand)| {
                serde_json::json!({
                    "t_ms": *f as f64 / self.fps * 1000.0,
                    "rect": [r.0, r.1, r.2, r.3],
                    "auto": [a.0, a.1, a.2, a.3],
                    "hand": hand,
                })
            })
            .collect();
        serde_json::json!({
            "segs": segs,
            "dur_ms": self.dur_ms,
            // the FRAME grid the segment starts are counted on -- the page
            // steps a frame at a time on it, so a step lands on a frame the
            // goblins will actually read
            "fps": self.fps,
            "aspect": self.aspect,
            "cap": crate::autocrop::MIN_SIDE_FRAC,
            // whether anything at all has been moved, so the page can say
            // "this is yours now" without diffing two lists
            "edited": self.moved || self.hand.iter().any(|&h| h),
        })
    }
}

/// Show `plan` for `video`, let the person correct it, and hand back what they
/// settled on. `None` when they left it exactly as the probe drew it -- the
/// caller then keeps the automatic plan, cache stamp and all.
///
/// `video` is the normalized copy: the picture the crop is actually applied
/// to, so a rect drawn here is a rect of the frames the encoder will read.
/// `page` hears the URL once the page is up; `open` also opens it in the
/// browser here.
pub fn edit(
    video: &Path,
    plan: &Plan,
    dur_ms: f64,
    fps: f64,
    aspect: f64,
    open: bool,
    page: &dyn Fn(&str),
) -> Result<Option<Plan>> {
    // `Arc`, because a finished page outlives this call: it is parked in the
    // browser window until the next stage gives it somewhere to go (see
    // `pagewin`). The socket closes when the last holder lets go, which is
    // what ends that wait.
    let server = Arc::new(
        tiny_http::Server::http("127.0.0.1:0")
            .map_err(|e| anyhow::anyhow!("could not start the crop page: {e}"))?,
    );
    let port = server
        .server_addr()
        .to_ip()
        .context("the crop page has no TCP address")?
        .port();
    let url = format!("http://127.0.0.1:{port}/");

    let state = Arc::new(Mutex::new(Session {
        segs: plan.segs.clone(),
        auto: if plan.auto.len() == plan.segs.len() {
            plan.auto.clone()
        } else {
            plan.segs.clone()
        },
        hand: vec![false; plan.segs.len()],
        moved: false,
        fps,
        dur_ms,
        aspect,
        done: false,
        seen: false,
    }));

    page(&url);
    // The window may already be up: the previous video's crop page is parked
    // in it waiting for exactly this address, and a batch should turn a page
    // rather than open a window per video. `deliver` hands it over when there
    // is one to hand over, and opens a browser when there is not.
    crate::pagewin::deliver(&url, open);

    // No console reading here: the draft's own display already owns the
    // keyboard (Ctrl-C included), so this stage waits on the page and on the
    // cancel flag that display sets.
    let path = video.to_path_buf();
    loop {
        crate::cancel::check()?;
        let Some(req) = server.recv_timeout(Duration::from_millis(100))? else {
            continue;
        };
        handle(req, &state, &path)?;
        if state.lock().unwrap().done {
            let out = finish(&state, plan);
            // This PAGE is finished; the WINDOW is not. Park it so the next
            // stage navigates it -- a crop page per video and a review at the
            // end should be one window turning pages, not a window opening,
            // closing and opening again. Only when someone actually opened the
            // page: a run with no window has nothing to hand over.
            if state.lock().unwrap().seen {
                crate::pagewin::park();
                let held = Arc::clone(&server);
                let state = Arc::clone(&state);
                let path = path.clone();
                std::thread::spawn(move || hold(held, state, path));
            }
            return Ok(out);
        }
    }
}

/// Keep a finished page answering until the window leaves for the next stage.
///
/// It serves the same routes: the picture is still on screen and a scrub after
/// Done is harmless. The one question that still matters is `/api/next`.
fn hold(server: Arc<tiny_http::Server>, state: Arc<Mutex<Session>>, video: PathBuf) {
    let deadline = Instant::now() + crate::pagewin::PATIENCE;
    while Instant::now() < deadline && crate::pagewin::parked() {
        let req = match server.recv_timeout(Duration::from_millis(200)) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(_) => return,
        };
        let leaving = req.url().starts_with("/api/left");
        if handle(req, &state, &video).is_err() || leaving {
            return;
        }
    }
    // Out of patience without being told the page left: the socket is about to
    // close under a page that is still asking, so say so rather than leave a
    // next address here that nobody will read. `/api/left` is the page's own
    // word and clears the slot itself, so this only fires on a timeout -- and
    // by then the slot, if any, is not ours to keep.
}

/// The edited plan, or `None` when nothing was actually moved. `src` is the
/// plan that went in: the rects are the page's, everything else is still the
/// probe's -- the grid they were read on, and the escape the probe measured
/// (which the review page prints, and which a drawn rect does not re-measure).
fn finish(state: &Arc<Mutex<Session>>, src: &Plan) -> Option<Plan> {
    let s = state.lock().unwrap();
    if !s.hand.iter().any(|&h| h) && !s.moved {
        return None;
    }
    // consecutive equal rects are one decode segment again: the merge the
    // probe does for its own rects, redone now that some of them moved. A
    // boundary the page placed between two halves that ended up with the SAME
    // rect is not a boundary the decoder can use, so it goes -- and it is not
    // lost work: a split exists to give the two halves different rects.
    let mut segs: Vec<(usize, Rect)> = Vec::with_capacity(s.segs.len());
    for &(start, r) in &s.segs {
        match segs.last() {
            Some(&(_, last)) if same(last, r) => {}
            _ => segs.push((start, r)),
        }
    }
    // ...and the same rule at the very first frame: a plan whose first segment
    // does not start the clip is refused by the decoder (is_runnable)
    if segs.first().map(|s| s.0) != Some(0) {
        if let Some(first) = segs.first_mut() {
            first.0 = 0;
        } else {
            return None;
        }
    }
    Some(Plan {
        auto: s.auto.clone(),
        segs,
        manual: true,
        grid: src.grid,
        escape_share: src.escape_share,
        placed: 0,
    })
}

fn same(a: Rect, b: Rect) -> bool {
    (a.0 - b.0).abs() < 1e-9
        && (a.1 - b.1).abs() < 1e-9
        && (a.2 - b.2).abs() < 1e-9
        && (a.3 - b.3).abs() < 1e-9
}

fn handle(mut req: tiny_http::Request, state: &Arc<Mutex<Session>>, video: &PathBuf) -> Result<()> {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("/").to_string();
    match (req.method().clone(), path.as_str()) {
        (Method::Get, "/") | (Method::Get, "/index.html") => {
            state.lock().unwrap().seen = true;
            let _ = req.respond(
                // No caching: this page is the program, and a stored copy would
                // outlive the version that wrote it.
                Response::from_string(include_str!("cropedit.html"))
                    .with_header(header("Content-Type", "text/html; charset=utf-8"))
                    .with_header(header("Cache-Control", "no-store")),
            );
        }
        // The page has done its job and asks where the run goes next; the next
        // stage leaves the address here. `null` means "not yet, keep asking".
        (Method::Get, "/api/next") => {
            let _ = req.respond(json_response(&serde_json::json!({
                "next": crate::pagewin::ask(),
            })));
        }
        // The page is navigating away: the window now belongs to that stage.
        (Method::Post, "/api/left") => {
            crate::pagewin::left();
            let _ = req.respond(json_response(&serde_json::json!({"ok": true})));
        }
        (Method::Get, "/api/state") => {
            let v = state.lock().unwrap().json();
            let _ = req.respond(json_response(&v));
        }
        (Method::Get, "/api/lang") => {
            let _ = req.respond(json_response(&crate::lang::catalog_json()));
        }
        (Method::Get, "/video") => {
            serve_file_threaded(req, video.clone(), video_ctype(video));
        }
        (Method::Post, "/api/rects") => {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let v = apply(state, &body);
            let _ = req.respond(json_response(&v));
        }
        (Method::Post, "/api/cuts") => {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let v = apply_cuts(state, &body);
            let _ = req.respond(json_response(&v));
        }
        (Method::Post, "/api/done") => {
            state.lock().unwrap().done = true;
            let _ = req.respond(json_response(&serde_json::json!({"ok": true})));
        }
        _ => {
            let _ = req.respond(Response::from_string("not here").with_status_code(404));
        }
    }
    Ok(())
}

/// One posted correction: `{"i": <segment or -1 for all>, "rect": [x,y,w,h] or
/// null for "back to the probe's own"}`.
///
/// Everything about a posted rect is checked here rather than trusted: the
/// page is a page, and a rect that left the frame or shrank past the zoom cap
/// would reach ffmpeg as a crop filter.
fn apply(state: &Arc<Mutex<Session>>, body: &str) -> serde_json::Value {
    let mut s = state.lock().unwrap();
    let v: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return serde_json::json!({"ok": false}),
    };
    let i = v.get("i").and_then(|x| x.as_i64()).unwrap_or(-1);
    let want: Option<Rect> = v.get("rect").and_then(|r| {
        let a = r.as_array()?;
        (a.len() == 4).then(|| {
            let n = |k: usize| a[k].as_f64().unwrap_or(f64::NAN);
            (n(0), n(1), n(2), n(3))
        })
    });
    let n = s.segs.len();
    let targets: Vec<usize> = if i < 0 { (0..n).collect() } else { vec![(i as usize).min(n - 1)] };
    for t in targets {
        match want {
            None => {
                s.segs[t].1 = s.auto[t].1;
                s.hand[t] = false;
            }
            Some(r) => {
                let Some(r) = legal(r) else { continue };
                s.segs[t].1 = r;
                s.hand[t] = true;
            }
        }
    }
    let v = s.json();
    drop(s);
    v
}

/// One posted change to the segment STARTS -- the boundaries between the rects,
/// which is the "where does this shot end" question the probe cannot always
/// answer and the person watching can.
///
/// `{"op":"add","frame":F}` splits the segment containing frame F in two;
/// `{"op":"move","i":i,"frame":F}` drags boundary i (segment i's own start);
/// `{"op":"del","i":i}` takes boundary i out and gives its frames to the shot
/// before it. Boundary 0 is the clip's own start and cannot be moved or
/// removed, only added to.
///
/// A new shot inherits the whole of the one it came out of -- its rect, the
/// probe's rect behind it and whether that rect was hand-drawn -- so splitting
/// is never a change to the picture on its own. Moving the boundary is.
fn apply_cuts(state: &Arc<Mutex<Session>>, body: &str) -> serde_json::Value {
    let mut s = state.lock().unwrap();
    let v: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return serde_json::json!({"ok": false}),
    };
    let frame = v.get("frame").and_then(|x| x.as_f64());
    let i = v.get("i").and_then(|x| x.as_i64()).unwrap_or(-1);
    match v.get("op").and_then(|x| x.as_str()) {
        Some("add") => {
            let Some(frame) = frame else {
                return serde_json::json!({"ok": false});
            };
            // which segment is this frame in? the last one that starts at or
            // before it -- the page's own rule, so the two agree
            let mut at = 0usize;
            for (k, &(f, _)) in s.segs.iter().enumerate() {
                if f as f64 <= frame {
                    at = k;
                }
            }
            let last = s.frames() - 1.0;
            if at == 0 && frame.round() <= 0.0 {
                return s.json(); // the clip's own start is not a split
            }
            if frame.round() >= last {
                return s.json();
            }
            if at + 1 < s.segs.len() && s.segs[at + 1].0 as f64 <= frame.round() {
                return s.json(); // a boundary is already there
            }
            // the split takes the frame the pointer named, so the new shot
            // begins exactly where the person pointed
            let f = frame.round().clamp(1.0, last) as usize;
            if s.segs.iter().any(|&(g, _)| g == f) {
                return s.json();
            }
            let (start, rect) = s.segs[at];
            let auto = s.auto.get(at).copied().unwrap_or((start, rect));
            let hand = s.hand.get(at).copied().unwrap_or(false);
            s.segs.insert(at + 1, (f, rect));
            s.auto.insert(at + 1, (f, auto.1));
            s.hand.insert(at + 1, hand);
            s.moved = true;
        }
        Some("move") => {
            let Some(frame) = frame else {
                return serde_json::json!({"ok": false});
            };
            if i <= 0 || i as usize >= s.segs.len() {
                return s.json();
            }
            let k = i as usize;
            let Some(f) = s.clamp_frame(k, frame) else {
                return s.json();
            };
            if f == s.segs[k].0 {
                return s.json();
            }
            s.segs[k].0 = f;
            // the probe's own rect for a boundary it never drew is the one
            // beside it: "auto" for a shot the person invented is the rect the
            // shot it came out of would get
            if let Some(a) = s.auto.get_mut(k) {
                a.0 = f;
            }
            s.moved = true;
        }
        Some("del") => {
            if i <= 0 || i as usize >= s.segs.len() {
                return s.json();
            }
            let k = i as usize;
            s.segs.remove(k);
            if k < s.auto.len() {
                s.auto.remove(k);
            }
            if k < s.hand.len() {
                s.hand.remove(k);
            }
            s.moved = true;
        }
        _ => return serde_json::json!({"ok": false}),
    }
    let v = s.json();
    drop(s);
    v
}

/// A rect the decode can be handed: inside the frame, no smaller than the zoom
/// cap on either axis, and made of real numbers. `None` refuses it outright --
/// the page keeps what it had rather than showing a rect nothing will honour.
fn legal(r: Rect) -> Option<Rect> {    let (x, y, w, h) = r;
    if ![x, y, w, h].iter().all(|v| v.is_finite()) {
        return None;
    }
    let cap = crate::autocrop::MIN_SIDE_FRAC;
    let w = w.clamp(cap, 1.0);
    let h = h.clamp(cap, 1.0);
    let x = x.clamp(0.0, 1.0 - w);
    let y = y.clamp(0.0, 1.0 - h);
    Some(if w >= 1.0 && h >= 1.0 { IDENTITY } else { (x, y, w, h) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A probe's plan and the page that opens on it -- one shot per rect,
    /// each starting 900 frames after the one before.
    fn opened(rects: &[Rect]) -> (Plan, Arc<Mutex<Session>>) {
        let segs: Vec<(usize, Rect)> =
            rects.iter().enumerate().map(|(i, &r)| (i * 900, r)).collect();
        let plan = Plan {
            auto: segs.clone(),
            segs: segs.clone(),
            manual: false,
            grid: 24,
            escape_share: 0.03,
            placed: 0,
        };
        let page = Arc::new(Mutex::new(Session {
            auto: segs.clone(),
            hand: vec![false; segs.len()],
            segs,
            moved: false,
            fps: 30.0,
            dur_ms: 60_000.0,
            aspect: 16.0 / 9.0,
            done: false,
            seen: false,
        }));
        (plan, page)
    }

    /// A page that posts nothing leaves the probe's plan exactly as it was --
    /// including its cache stamp, which a manual plan would have replaced.
    #[test]
    fn an_untouched_page_hands_back_no_plan() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        assert!(finish(&s, &src).is_none());
    }

    #[test]
    fn a_drawn_rect_replaces_one_shot_and_survives_the_merge() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), (0.1, 0.1, 0.7, 0.7), IDENTITY]);
        apply(&s, r#"{"i":1,"rect":[0.2,0.05,0.75,0.75]}"#);
        let p = finish(&s, &src).expect("a hand-drawn rect is a plan");
        assert!(p.manual);
        // three shots, three rects, none of them equal to its neighbour now
        assert_eq!(p.segs.len(), 3);
        assert!((p.segs[1].1 .0 - 0.2).abs() < 1e-12);
        assert_eq!(p.segs[1].0, 900);
        // and the probe's own rects are still there for the next page to show
        assert!((p.auto[1].1 .0 - 0.1).abs() < 1e-12);
    }

    #[test]
    fn one_rect_for_every_shot_merges_into_one_segment() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY, (0.2, 0.2, 0.7, 0.7)]);
        apply(&s, r#"{"i":-1,"rect":[0.15,0.15,0.7,0.7]}"#);
        let p = finish(&s, &src).expect("a hand-drawn rect is a plan");
        assert_eq!(p.segs.len(), 1, "one rect throughout is one decode segment");
        assert_eq!(p.segs[0].0, 0);
    }

    /// "Auto" is a real undo: the shot goes back to the probe's rect AND stops
    /// counting as hand-drawn, so a page fully reset hands back nothing.
    #[test]
    fn auto_puts_a_shot_back_and_a_full_reset_hands_back_nothing() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        apply(&s, r#"{"i":0,"rect":[0.2,0.2,0.7,0.7]}"#);
        assert!(finish(&s, &src).is_some());
        apply(&s, r#"{"i":0,"rect":null}"#);
        assert!(finish(&s, &src).is_none());
        assert!((s.lock().unwrap().segs[0].1 .0 - 0.1).abs() < 1e-12);
    }

    /// The page end to end, over its own socket: the state it publishes, a
    /// rect posted into it, and Done handing the plan back to the pipeline.
    /// The unit tests above reach past the server; this is the only thing that
    /// says the routes, the JSON and the finish are wired to each other.
    #[test]
    fn the_page_serves_its_state_takes_a_rect_and_finishes() {
        use std::io::{BufRead, BufReader, Write};
        // the page's loop reads the process-global stop flag; no test that
        // sets it may run beside this one
        let _guard = crate::cancel::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::cancel::reset();
        let (plan, _) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let worker = std::thread::spawn(move || {
            edit(Path::new("no-such-video.mp4"), &plan, 60_000.0, 30.0, 16.0 / 9.0, false, &|url| {
                let _ = tx.send(url.to_string());
            })
        });
        // the page hands over its URL, which is where the port is
        let url = rx.recv_timeout(Duration::from_secs(5)).expect("the page announces itself");
        let addr = url
            .strip_prefix("http://")
            .map(|a| a.trim_end_matches('/').to_string())
            .expect("an http address");

        // Lenient on purpose: the LAST request is Done, and the page tears
        // its server down the moment it has it -- a reset there is the stage
        // ending, not a failure.
        let ask = |req: &str| -> String {
            let mut s = std::net::TcpStream::connect(&addr).expect("the page is listening");
            if s.write_all(req.as_bytes()).is_err() {
                return String::new();
            }
            let mut r = BufReader::new(s);
            let mut len = 0usize;
            loop {
                let mut l = String::new();
                match r.read_line(&mut l) {
                    Ok(0) | Err(_) => return String::new(),
                    Ok(_) => {}
                }
                if let Some(n) = l.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = n.trim().parse().unwrap_or(0);
                }
                if l.trim().is_empty() {
                    break;
                }
            }
            let mut body = vec![0u8; len];
            match std::io::Read::read_exact(&mut r, &mut body) {
                Ok(()) => String::from_utf8_lossy(&body).to_string(),
                Err(_) => String::new(),
            }
        };
        let post = |path: &str, body: &str| {
            format!(
                "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        };

        let state: serde_json::Value =
            serde_json::from_str(&ask("GET /api/state HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"))
                .expect("state is json");
        assert_eq!(state["segs"].as_array().unwrap().len(), 2);
        assert_eq!(state["cap"].as_f64().unwrap(), crate::autocrop::MIN_SIDE_FRAC);
        // the page steps a frame at a time, so it is told the frame grid
        assert_eq!(state["fps"].as_f64().unwrap(), 30.0);
        assert_eq!(state["dur_ms"].as_f64().unwrap(), 60_000.0);
        assert!(!state["segs"][0]["hand"].as_bool().unwrap());

        let after: serde_json::Value =
            serde_json::from_str(&ask(&post("/api/rects", r#"{"i":1,"rect":[0.2,0.2,0.7,0.7]}"#)))
                .expect("the post answers with the new state");
        assert!(after["segs"][1]["hand"].as_bool().unwrap());
        assert!((after["segs"][1]["rect"][0].as_f64().unwrap() - 0.2).abs() < 1e-9);

        ask(&post("/api/done", "{}"));
        let got = worker.join().expect("the page thread ends").expect("no error");
        let got = got.expect("a drawn rect is a plan");
        assert!(got.manual);
        assert_eq!(got.segs.len(), 2);
        assert!((got.segs[1].1 .0 - 0.2).abs() < 1e-9);
    }

    /// The page is a page: every posted rect is checked before it can become a
    /// crop filter.
    #[test]
    fn an_impossible_rect_is_refused_or_brought_back_in() {
        assert_eq!(legal((0.0, 0.0, 1.0, 1.0)), Some(IDENTITY));
        assert!(legal((f64::NAN, 0.0, 0.5, 0.5)).is_none());
        // past the zoom cap on either axis: lifted to it
        let r = legal((0.4, 0.4, 0.2, 0.2)).unwrap();
        assert!((r.2 - crate::autocrop::MIN_SIDE_FRAC).abs() < 1e-12);
        // off the frame: brought back inside it, whole
        let r = legal((0.9, -0.5, 0.8, 0.8)).unwrap();
        assert!(r.0 + r.2 <= 1.0 + 1e-12 && r.1 >= 0.0);
    }

    // ---- the segment starts -------------------------------------------------
    // A shot is a run of frames with one rect. Splitting one is the edit the
    // probe cannot make -- a subject that walks out of the rect mid-shot is
    // exactly the case a person watching can call -- so these are the rules
    // that keep that edit from producing a plan the decoder would refuse.

    /// Splitting gives the new half the whole of the old one, so the picture
    /// does not change on its own; only the boundary does.
    #[test]
    fn a_split_inherits_the_shot_it_came_out_of() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        apply(&s, r#"{"i":0,"rect":[0.2,0.2,0.7,0.7]}"#);
        let after = apply_cuts(&s, r#"{"op":"add","frame":300}"#);
        assert_eq!(after["segs"].as_array().unwrap().len(), 3);
        // frames 300 and 900, the new one exactly where it was asked for
        assert_eq!(after["segs"][1]["t_ms"].as_f64().unwrap(), 10_000.0);
        assert_eq!(after["segs"][2]["t_ms"].as_f64().unwrap(), 30_000.0);
        // the half that came out of shot 0 carries shot 0's hand-drawn rect,
        // and is marked hand-drawn with it
        assert!((after["segs"][1]["rect"][0].as_f64().unwrap() - 0.2).abs() < 1e-9);
        assert!(after["segs"][1]["hand"].as_bool().unwrap());
        // the plan that comes back is the page's, and runnable
        let p = finish(&s, &src).expect("a placed boundary is a plan");
        assert!(p.manual);
        assert!(p.is_runnable());
    }

    /// A boundary is a place in the video, so it has to land on the frame grid
    /// and between its neighbours -- the page cannot make a list the decoder
    /// would refuse, whatever it posts.
    #[test]
    fn a_moved_boundary_stays_on_the_grid_and_between_its_neighbours() {
        let (_, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY, (0.2, 0.2, 0.7, 0.7)]);
        // 900 -> 1200, the last frame before the next boundary at 1800
        let a = apply_cuts(&s, r#"{"op":"move","i":1,"frame":1200.4}"#);
        assert_eq!(a["segs"][1]["t_ms"].as_f64().unwrap(), 40_000.0);
        // dragged past its right neighbour: pinned to one frame short of it
        let b = apply_cuts(&s, r#"{"op":"move","i":1,"frame":999999}"#);
        assert_eq!(b["segs"][1]["t_ms"].as_f64().unwrap(), 1799.0 / 30.0 * 1000.0);
        // ...and past its left neighbour, likewise
        let c = apply_cuts(&s, r#"{"op":"move","i":1,"frame":-50}"#);
        assert_eq!(c["segs"][1]["t_ms"].as_f64().unwrap(), 1.0 / 30.0 * 1000.0);
        // the clip's own start is not a boundary and never moves
        let d = apply_cuts(&s, r#"{"op":"move","i":0,"frame":500}"#);
        assert_eq!(d["segs"][0]["t_ms"].as_f64().unwrap(), 0.0);
    }

    /// A boundary is only a boundary if the two shots around it end up
    /// different: the plan that leaves is merged the way the probe merges its
    /// own, so splitting is not "one more segment" for its own sake.
    #[test]
    fn a_split_between_two_equal_rects_does_not_survive_the_merge() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        apply_cuts(&s, r#"{"op":"add","frame":300}"#);
        let p = finish(&s, &src).expect("a placed boundary is a plan on its own");
        assert_eq!(p.segs.len(), 2, "the two halves still share one rect");
        // ...but give one of them its own rect and the split is real
        apply(&s, r#"{"i":1,"rect":[0.35,0.1,0.7,0.7]}"#);
        let p = finish(&s, &src).unwrap();
        assert_eq!(p.segs.len(), 3);
        assert_eq!(p.segs[1].0, 300);
    }

    /// Removing a boundary gives its frames back to the shot before it, and
    /// nothing else about the plan changes.
    #[test]
    fn a_removed_boundary_hands_its_frames_back() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), (0.2, 0.2, 0.7, 0.7), IDENTITY]);
        let a = apply_cuts(&s, r#"{"op":"del","i":1}"#);
        assert_eq!(a["segs"].as_array().unwrap().len(), 2);
        // what was the third shot is now the second, at its own frame
        assert_eq!(a["segs"][1]["t_ms"].as_f64().unwrap(), 60_000.0);
        assert!((a["segs"][0]["rect"][0].as_f64().unwrap() - 0.1).abs() < 1e-9);
        // the clip's own start is not removable: the list is unchanged
        let b = apply_cuts(&s, r#"{"op":"del","i":0}"#);
        assert_eq!(b["segs"].as_array().unwrap().len(), 2);
        // and a boundary the page placed survives into the plan it hands over
        let p = finish(&s, &src).expect("a removed boundary is a plan too");
        assert_eq!(p.segs.len(), 2);
        assert_eq!(p.segs[1].0, 1800);
    }

    /// The page end to end over its own socket, for the boundary route: the
    /// state it publishes, a split posted into it, and Done handing back a
    /// plan with three shots in it.
    #[test]
    fn the_page_takes_a_split_and_hands_back_the_plan_it_makes() {
        use std::io::{BufRead, BufReader, Write};
        let _guard = crate::cancel::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::cancel::reset();
        let (plan, _) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let worker = std::thread::spawn(move || {
            edit(Path::new("no-such-video.mp4"), &plan, 60_000.0, 30.0, 16.0 / 9.0, false, &|url| {
                let _ = tx.send(url.to_string());
            })
        });
        let url = rx.recv_timeout(Duration::from_secs(5)).expect("the page announces itself");
        let addr = url
            .strip_prefix("http://")
            .map(|a| a.trim_end_matches('/').to_string())
            .expect("an http address");

        let ask = |req: &str| -> String {
            let mut s = std::net::TcpStream::connect(&addr).expect("the page is listening");
            if s.write_all(req.as_bytes()).is_err() {
                return String::new();
            }
            let mut r = BufReader::new(s);
            let mut len = 0usize;
            loop {
                let mut l = String::new();
                match r.read_line(&mut l) {
                    Ok(0) | Err(_) => return String::new(),
                    Ok(_) => {}
                }
                if let Some(n) = l.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = n.trim().parse().unwrap_or(0);
                }
                if l.trim().is_empty() {
                    break;
                }
            }
            let mut body = vec![0u8; len];
            match std::io::Read::read_exact(&mut r, &mut body) {
                Ok(()) => String::from_utf8_lossy(&body).to_string(),
                Err(_) => String::new(),
            }
        };
        let post = |path: &str, body: &str| {
            format!(
                "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        };

        let before: serde_json::Value = serde_json::from_str(&ask(
            "GET /api/state HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
        ))
        .expect("state is json");
        assert_eq!(before["segs"].as_array().unwrap().len(), 2);
        assert!(!before["edited"].as_bool().unwrap());

        let after: serde_json::Value =
            serde_json::from_str(&ask(&post("/api/cuts", r#"{"op":"add","frame":450}"#)))
                .expect("the post answers with the new state");
        assert_eq!(after["segs"].as_array().unwrap().len(), 3);
        assert!(after["edited"].as_bool().unwrap());
        // and the new shot is a real shot the page can then aim on its own
        let aimed: serde_json::Value =
            serde_json::from_str(&ask(&post("/api/rects", r#"{"i":1,"rect":[0.3,0.3,0.6,0.6]}"#)))
                .expect("the post answers with the new state");
        assert!(aimed["segs"][1]["hand"].as_bool().unwrap());

        ask(&post("/api/done", "{}"));
        let got = worker.join().expect("the page thread ends").expect("no error");
        let got = got.expect("a moved boundary is a plan");
        assert!(got.manual);
        assert_eq!(got.segs.len(), 3);
        assert_eq!(got.segs[1].0, 450);
        assert!(got.is_runnable());
    }

    /// A plan whose first segment does not start the clip is one the decoder
    /// refuses, so the page's own answer is never allowed to be one.
    #[test]
    fn the_handed_back_plan_always_starts_at_frame_zero() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        {
            let mut g = s.lock().unwrap();
            g.segs[0].0 = 12;   // as if something upstream had moved the start
            g.moved = true;
        }
        let p = finish(&s, &src).expect("a moved boundary is a plan");
        assert_eq!(p.segs[0].0, 0);
        assert!(p.is_runnable());
    }
}
