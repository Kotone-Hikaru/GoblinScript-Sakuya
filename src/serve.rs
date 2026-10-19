//! The API server: GoblinScript as a sidecar for another program.
//!
//! An editor that wants the goblins' draft under its own timeline does not
//! want a console, a review page or a funscript on disk. It wants the stages
//! run once per video, the tracks kept in memory, and a styled list back in
//! milliseconds every time the author moves a knob or places a point. That is
//! the review server's shape with the drafting stages placed under it and the
//! page taken away -- and it is the same `draft`, the same `retrack` and the
//! same `style_draft` the console runs, so a session drafts exactly what the
//! command line would.
//!
//! The decisions a person makes keep the browser pages the console opens
//! for them. Before the expensive stage: where a VR viewport points
//! (`vr.rs`) and which rect each shot is cropped to (`cropedit.rs`) -- a
//! session that reaches one reads `working` with the page's URL under
//! `page`, and the draft continues when Done is pressed there. After it: the
//! workbench (`review.rs`), the review page with the video, the simulator
//! and every styling knob, which a client opens whenever it likes and as
//! often as it likes. The knobs a person settles on there ARE the session's
//! style: every style call the client makes after that starts from them,
//! and `style_rev` in the status counts the changes so the client knows to
//! ask again. A page opens in the browser from here unless the client said
//! it would do that itself.
//!
//! One process, loopback only, one token. A session is one video, keyed by
//! its cache fingerprint, so a second client asking for the same file joins
//! the first. One worker thread runs the heavy stages one session at a time
//! (one GPU job at a time); the request thread answers a style call itself,
//! because it is the millisecond kind, and hands everything else to the
//! worker. The process ends when its standard input closes -- a client that
//! dies takes its goblins with it -- or when a client posts `quit`.
//!
//! Routes (every one wants the `x-goblin-token` header):
//!   GET    /health                        the build, the model, the session count
//!   GET    /manifest                      the deploy constants, defaults and presets
//!   POST   /sessions                      {video, vr?, crop_edit?, browser?, model?} -> the session, drafting
//!   GET    /sessions/ID                   its status: stage in flight, a page waiting, stages done, or ready
//!   GET    /sessions/ID/tracks?from_ms&to_ms   the per-row tracks on the video clock
//!   POST   /sessions/ID/style             {params..., pins?, from_ms?, to_ms?} -> actions
//!   POST   /sessions/ID/workbench         {browser?} -> {url}: the review page for this session
//!   DELETE /sessions/ID                   forget it (the cache stays)
//!   POST   /quit
use std::collections::HashMap;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use tiny_http::Method;

use crate::bundle::Manifest;
use crate::review::{self, json_response, params_from_json, params_json, presets_json};
use crate::{
    cache, cancel, draft, retrack, review_clip, review_style, seed_for, style, style_draft, vr,
    vr_need, Bundle,
    Cli, DraftOut, Outcome, StageSink, VrMap,
};

/// Where a session is in its life. `Working` covers the first draft and every
/// re-decode after it; the status says which stage.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Queued,
    Working,
    Ready,
    Failed,
}

impl State {
    fn label(self) -> &'static str {
        match self {
            State::Queued => "queued",
            State::Working => "working",
            State::Ready => "ready",
            State::Failed => "failed",
        }
    }
}

struct Status {
    state: State,
    /// The stage in flight and its bar, when one is measurable.
    stage: Option<(String, Option<ProgressBar>)>,
    /// A browser page the stage in flight waits on: its stage key and URL.
    page: Option<(String, String)>,
    /// The stages that cleared, with their summary lines, since the last
    /// job began.
    done: Vec<(String, String)>,
    error: Option<String>,
}

/// What the client said about VR when it opened the session.
enum VrAsk {
    /// Detect it; aim in the page when the video looks like VR and nobody
    /// has answered for it yet.
    Auto,
    /// Draft the raw frames whatever the detector thinks.
    Flat,
    /// Open the page whatever the detector thinks -- the hand mark, and the
    /// way to aim again a video that has an aim.
    Aim,
    /// Apply this sidecar as it is, no page.
    Fixed(vr::Config),
}

/// One video. `draft` is empty until the first draft lands and holds the
/// tracks from then on; a re-decode replaces them in place.
struct Session {
    id: String,
    video: PathBuf,
    ask: VrAsk,
    /// The crop page: `Some(true)` opens it always, `Some(false)` never,
    /// `None` when the probe ran fresh, as at a console.
    crop_edit: Option<bool>,
    /// Open a page in the browser from here; off when the client opens it.
    browser: bool,
    /// The model this session drafts with: a pack of the bundle, or the
    /// server's own (its `--model`, else the default) when absent.
    model: Option<String>,
    /// The aim the draft ran under, once decided. Absent: flat.
    vr: Mutex<Option<vr::Config>>,
    /// The session's style: the server's command line until the workbench
    /// moves it. Every style call starts from here.
    params: Mutex<style::Params>,
    /// Bumped on every change the workbench applies.
    style_rev: AtomicU64,
    /// The workbench's URL while its page is up.
    workbench: Mutex<Option<String>>,
    status: Mutex<Status>,
    draft: Mutex<Option<DraftOut>>,
    /// Set by a delete while a job is queued or running: the worker skips a
    /// queued job and unwinds a running one.
    cancelled: AtomicBool,
}

impl StageSink for Session {
    fn setup(&self, key: &str) {
        let mut s = self.status.lock().unwrap();
        s.state = State::Working;
        s.stage = Some((key.to_string(), None));
        s.page = None;
    }
    fn stage(&self, key: &str, pb: &ProgressBar) {
        let mut s = self.status.lock().unwrap();
        s.state = State::Working;
        s.stage = Some((key.to_string(), Some(pb.clone())));
        s.page = None;
    }
    fn page(&self, key: &str, url: &str) {
        let mut s = self.status.lock().unwrap();
        s.state = State::Working;
        s.stage = Some((key.to_string(), None));
        s.page = Some((key.to_string(), url.to_string()));
    }
    fn done(&self, key: &str, info: &str) {
        let mut s = self.status.lock().unwrap();
        s.stage = None;
        s.page = None;
        s.done.push((key.to_string(), console::strip_ansi_codes(info).to_string()));
    }
}

enum Job {
    Draft(Arc<Session>),
    /// Put this model's tracks at this seed in hand: a head pass unless the
    /// draft has them shelved.
    Retrack(Arc<Session>, String, u64),
}

type Sessions = Arc<Mutex<HashMap<String, Arc<Session>>>>;

/// Serve until standard input closes or a client posts `quit`.
pub fn serve(mut cli: Cli, b: Bundle, cache_root: PathBuf, params: style::Params) -> Result<()> {
    let server = tiny_http::Server::http(("127.0.0.1", cli.port))
        .map_err(|e| anyhow::anyhow!("could not start the API server: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("API server has no TCP address")?
        .port();
    let token = fresh_token();
    let man = b.manifest.clone();

    // The handshake: one line, first, and the only line a client has to
    // parse. Everything the console prints after it is the drafting log.
    {
        let hello = serde_json::json!({
            "goblinscript": env!("GS_VERSION"),
            "port": port,
            "token": token,
            "checkpoint": man.checkpoint,
            "row_hz": man.row_hz(),
        });
        let mut out = std::io::stdout().lock();
        writeln!(out, "{hello}")?;
        out.flush()?;
    }

    // A client that spawned this process holds its stdin; when the client
    // goes, the pipe closes and so does this. A terminal keeps its stdin for
    // the live line's key reader, so the watch is only armed on a pipe.
    if !std::io::stdin().is_terminal() {
        std::thread::spawn(|| {
            let mut sink = Vec::new();
            let _ = std::io::stdin().read_to_end(&mut sink);
            std::process::exit(0);
        });
    }

    // Every session drafts under this command line, minus the parts that
    // belong to a person at a console: the review, and the skip rule for a
    // script that already exists (the client owns its document; the file
    // this writes is the cache's own copy). The pages are each session's
    // own call, set per job below.
    cli.force = true;
    cli.review = false;
    cli.out = Some(cache_root.join("serve"));

    let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));
    let (tx, rx) = mpsc::channel::<Job>();
    // The request loop keeps its own handle on the bundle: the worker owns
    // the one it drafts with, and every pack's bytes are shared between the
    // two, so `/health`, `/manifest?model=` and a session's model check read
    // the same packs the worker drafts from.
    let api = b.select(&b.manifest.pack)?;
    let worker = {
        let root = cache_root.clone();
        std::thread::spawn(move || work(cli, b, root, params, rx))
    };

    for mut req in server.incoming_requests() {
        let authed = req
            .headers()
            .iter()
            .any(|h| h.field.equiv("x-goblin-token") && h.value.as_str() == token);
        if !authed {
            let _ = req.respond(
                json_response(&serde_json::json!({ "error": "bad token" })).with_status_code(401),
            );
            continue;
        }
        let url = req.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        let mut body = String::new();
        if matches!(req.method(), Method::Post | Method::Put) {
            let _ = req.as_reader().read_to_string(&mut body);
        }
        let reply = route(req.method(), &parts, query, &body, &sessions, &tx, &api, &man, &params);
        match reply {
            Reply::Json(v) => {
                let _ = req.respond(json_response(&v));
            }
            Reply::Error(code, msg) => {
                let _ = req.respond(
                    json_response(&serde_json::json!({ "error": msg })).with_status_code(code),
                );
            }
            Reply::Quit => {
                let _ = req.respond(json_response(&serde_json::json!({ "bye": true })));
                break;
            }
        }
    }
    // A job in flight is cancelled rather than waited for: nothing downstream
    // of a stage commits until it finishes, so an abrupt stop loses nothing
    // the cache cannot resume.
    cancel::request();
    drop(tx);
    let _ = worker.join();
    Ok(())
}

enum Reply {
    Json(serde_json::Value),
    Error(u16, String),
    Quit,
}

#[allow(clippy::too_many_arguments)]
fn route(
    method: &Method,
    parts: &[&str],
    query: &str,
    body: &str,
    sessions: &Sessions,
    tx: &mpsc::Sender<Job>,
    b: &Bundle,
    man: &Manifest,
    defaults: &style::Params,
) -> Reply {
    let find = |id: &str| sessions.lock().unwrap().get(id).cloned();
    match (method, parts) {
        (Method::Get, ["health"]) => Reply::Json(serde_json::json!({
            "goblinscript": env!("GS_VERSION"),
            "checkpoint": man.checkpoint,
            "basis_id": man.basis_id,
            "encoder": man.encoder,
            "row_hz": man.row_hz(),
            // the models this bundle carries, the one a session drafts
            // with when it names none, and the one the bundle ships as
            "model": man.pack,
            "models": b.pack_names(),
            // each model's trunk: models that share one switch by a restyle
            "trunks": b.models().map(|ms| ms.into_iter()
                .map(|m| (m.name, serde_json::Value::String(m.trunk)))
                .collect::<serde_json::Map<_, _>>()).unwrap_or_default(),
            "default": b.default_pack(),
            "sessions": sessions.lock().unwrap().len(),
        })),
        // `?model=NAME` reads another pack's constants; without it, the
        // server's own model
        (Method::Get, ["manifest"]) => match query_map(query)
            .get("model")
            .map(|m| b.select(m))
            .transpose()
        {
            Err(e) => Reply::Error(400, format!("{e:#}")),
            Ok(chosen) => {
            let bb = chosen.unwrap_or_else(|| b.select(&man.pack).expect("the server's own model"));
            manifest_json(&bb.manifest, &bb.trunk(), defaults)
        }
        },
        (Method::Post, ["sessions"]) => match open_session(body, sessions, tx, b, defaults) {
            Ok(s) => Reply::Json(session_json(&s, man)),
            Err(e) => Reply::Error(400, format!("{e:#}")),
        },
        (Method::Get, ["sessions", id]) => match find(id) {
            Some(s) => Reply::Json(session_json(&s, man)),
            None => Reply::Error(404, format!("no session {id}")),
        },
        (Method::Get, ["sessions", id, "tracks"]) => match find(id) {
            Some(s) => tracks(&s, query, man),
            None => Reply::Error(404, format!("no session {id}")),
        },
        (Method::Post, ["sessions", id, "style"]) => match find(id) {
            Some(s) => style_route(&s, body, tx, b),
            None => Reply::Error(404, format!("no session {id}")),
        },
        (Method::Post, ["sessions", id, "workbench"]) => match find(id) {
            Some(s) => workbench_route(&s, body, tx, b),
            None => Reply::Error(404, format!("no session {id}")),
        },
        (Method::Delete, ["sessions", id]) => {
            match sessions.lock().unwrap().remove(*id) {
                Some(s) => {
                    s.cancelled.store(true, Ordering::Relaxed);
                    if s.status.lock().unwrap().state == State::Working {
                        cancel::request();
                    }
                    Reply::Json(serde_json::json!({ "id": id, "deleted": true }))
                }
                None => Reply::Error(404, format!("no session {id}")),
            }
        }
        (Method::Post, ["quit"]) => Reply::Quit,
        _ => Reply::Error(404, "not found".to_string()),
    }
}

/// `GET /manifest`: the deploy constants of one model, the server's own
/// unless `?model=` names another pack of the bundle.
fn manifest_json(man: &Manifest, trunk: &str, defaults: &style::Params) -> Reply {
    Reply::Json(serde_json::json!({
        "bundle_version": man.bundle_version,
        "model": man.pack,
        "label": man.label,
        "trunk": trunk,
        "checkpoint": man.checkpoint,
        "epoch": man.epoch,
        "basis_id": man.basis_id,
        "encoder": man.encoder,
        "row_hz": man.row_hz(),
        "row0_ms": man.row_ms(0),
        "v_std": man.v_std,
        "env_seed": man.env_seed,
        "max_pos_rate": style::MAX_POS_RATE,
        "cut_ease": style::CUT_EASE,
        "pin_tol_ms": style::PIN_TOL_MS,
        "pin_horizon_ms": style::PIN_HORIZON_MS,
        "defaults": params_json(defaults),
        "presets": presets_json(crate::review::Presets {
            dwell_ramp: man.plat_soft[0],
            still_eps: man.still_eps,
        }),
    }))
}

/// The bundle a session drafts with: its own model when it named one,
/// else the server's, every pack's bytes shared either way.
fn for_session(b: &Bundle, s: &Session) -> Result<Bundle> {
    match &s.model {
        Some(m) => b.select(m),
        None => b.select(&b.manifest.pack),
    }
}

/// `POST /sessions`: attach to the session for this video, or open one and
/// queue its draft.
///
/// `vr` is absent to detect, `true` to aim in the page whatever the
/// detector thinks (and to aim again a video that has an aim), `false` to
/// draft the raw frames, or a sidecar object to apply as it is. `crop_edit`
/// is absent to open the crop page when the probe ran fresh, `true` to open
/// it always, `false` never. `browser` (default true) opens a page in the
/// browser from here; a client that opens the URL itself passes false.
fn open_session(
    body: &str,
    sessions: &Sessions,
    tx: &mpsc::Sender<Job>,
    b: &Bundle,
    defaults: &style::Params,
) -> Result<Arc<Session>> {
    #[derive(serde::Deserialize)]
    struct Msg {
        video: PathBuf,
        #[serde(default)]
        vr: serde_json::Value,
        #[serde(default)]
        crop_edit: Option<bool>,
        #[serde(default = "yes")]
        browser: bool,
        #[serde(default)]
        model: Option<String>,
    }
    fn yes() -> bool {
        true
    }
    let m: Msg = serde_json::from_str(body).context("bad session payload")?;
    if !m.video.is_file() {
        anyhow::bail!("no such video: {}", m.video.display());
    }
    if let Some(model) = &m.model {
        b.select(model)?;
    }
    let ask = match m.vr {
        serde_json::Value::Null => VrAsk::Auto,
        serde_json::Value::Bool(true) => VrAsk::Aim,
        serde_json::Value::Bool(false) => VrAsk::Flat,
        v => VrAsk::Fixed(serde_json::from_value(v).context("bad vr sidecar")?),
    };
    let id = cache::fingerprint(&m.video)?;
    let mut map = sessions.lock().unwrap();
    if let Some(s) = map.get(&id) {
        return Ok(s.clone());
    }
    let s = Arc::new(Session {
        id: id.clone(),
        video: m.video,
        ask,
        crop_edit: m.crop_edit,
        browser: m.browser,
        model: m.model,
        vr: Mutex::new(None),
        params: Mutex::new(*defaults),
        style_rev: AtomicU64::new(0),
        workbench: Mutex::new(None),
        status: Mutex::new(Status {
            state: State::Queued,
            stage: None,
            page: None,
            done: Vec::new(),
            error: None,
        }),
        draft: Mutex::new(None),
        cancelled: AtomicBool::new(false),
    });
    map.insert(id, s.clone());
    tx.send(Job::Draft(s.clone())).context("the worker is gone")?;
    Ok(s)
}

/// The status a client polls: where the work is, and once it has landed,
/// the clip's clock and shape. Times are on the VIDEO clock -- the source
/// file's -- which for a trimmed VR draft is `t0_ms` after the draft's own.
fn session_json(s: &Session, man: &Manifest) -> serde_json::Value {
    let st = s.status.lock().unwrap();
    let stage = st.stage.as_ref().map(|(key, pb)| {
        serde_json::json!({
            "key": key,
            "pos": pb.as_ref().map(|p| p.position()),
            "len": pb.as_ref().and_then(|p| p.length()),
        })
    });
    let page = st
        .page
        .as_ref()
        .map(|(key, url)| serde_json::json!({ "key": key, "url": url }));
    let done: Vec<serde_json::Value> = st
        .done
        .iter()
        .map(|(k, i)| serde_json::json!({ "key": k, "info": i }))
        .collect();
    // the draft is read only when it is not the worker's: a status poll during
    // a re-decode must answer, not wait
    let ready = s.draft.try_lock().ok().and_then(|d| {
        d.as_ref().map(|d| {
            let conf: Vec<f64> = d.tracks.conf.iter().copied().filter(|v| v.is_finite()).collect();
            serde_json::json!({
                "rows": d.times.len(),
                "row_hz": man.row_hz(),
                "row0_ms": man.row_ms(0) + d.t0_ms,
                "dur_ms": d.dur_ms,
                "t0_ms": d.t0_ms,
                "cuts_ms": d.cuts_ms.iter().map(|c| c + d.t0_ms).collect::<Vec<_>>(),
                "shot_edges": d.shot_edges,
                "env_seed": d.tracks_seed,
                "conf_mean": (!conf.is_empty()).then(|| conf.iter().sum::<f64>() / conf.len() as f64),
                "crop": d.crop,
            })
        })
    });
    let params = *s.params.lock().unwrap();
    serde_json::json!({
        "id": s.id,
        "video": s.video,
        // the model in hand once drafted, else the one the session asked for
        "model": s.draft.try_lock().ok().and_then(|d| d.as_ref().map(|d| d.pack.clone()))
            .unwrap_or_else(|| s.model.clone().unwrap_or_else(|| man.pack.clone())),
        "vr": *s.vr.lock().unwrap(),
        "state": st.state.label(),
        "stage": stage,
        "page": page,
        "done": done,
        "error": st.error,
        "ready": ready,
        "params": params_json(&params),
        "flags": style::flags_line(&params),
        "style_rev": s.style_rev.load(Ordering::Relaxed),
        "workbench": *s.workbench.lock().unwrap(),
    })
}

/// `GET /sessions/ID/tracks`: the per-row tracks, on the video clock, for
/// the rows inside `[from_ms, to_ms]` (the whole clip without a bound).
fn tracks(s: &Session, query: &str, man: &Manifest) -> Reply {
    let q = query_map(query);
    let guard = match s.draft.try_lock() {
        Ok(g) => g,
        Err(_) => return Reply::Error(409, "working".to_string()),
    };
    let d = match guard.as_ref() {
        Some(d) => d,
        None => return Reply::Error(409, "not drafted yet".to_string()),
    };
    let t0 = d.t0_ms;
    let from = q.get("from_ms").and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NEG_INFINITY);
    let to = q.get("to_ms").and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::INFINITY);
    let i0 = d.times.partition_point(|&t| t + t0 < from);
    let i1 = d.times.partition_point(|&t| t + t0 <= to);
    let (i0, i1) = (i0.min(d.times.len()), i1.min(d.times.len()).max(i0));
    let slice = |v: &[f64]| -> serde_json::Value {
        if v.is_empty() {
            return serde_json::Value::Null;
        }
        v[i0..i1]
            .iter()
            .map(|&x| {
                if x.is_finite() {
                    serde_json::json!((x * 1000.0).round() / 1000.0)
                } else {
                    serde_json::Value::Null
                }
            })
            .collect::<Vec<_>>()
            .into()
    };
    let pair = |p: &Option<(Vec<f64>, Vec<f64>)>| -> (serde_json::Value, serde_json::Value) {
        match p {
            Some((a, b)) => (slice(a), slice(b)),
            None => (serde_json::Value::Null, serde_json::Value::Null),
        }
    };
    let t = &d.tracks;
    let (blo, bhi) = pair(&t.band);
    let (plat_top, plat_bot) = pair(&t.plat);
    let (rev_top, rev_bot) = pair(&t.rev);
    Reply::Json(serde_json::json!({
        "first_row": i0,
        "rows": i1 - i0,
        "row_hz": man.row_hz(),
        "row0_ms": man.row_ms(0) + t0,
        "vmarg": slice(&t.vmarg),
        "level": slice(&t.level),
        "env": slice(&t.env),
        "conf": slice(&t.conf),
        "period": slice(&t.period),
        "blo": blo, "bhi": bhi,
        "plat_top": plat_top, "plat_bot": plat_bot,
        "rev_top": rev_top, "rev_bot": rev_bot,
    }))
}

/// Does the draft hold `trunk`'s tracks at `seed`, in hand or shelved?
fn holds(d: &DraftOut, trunk: &str, seed: u64) -> bool {
    (d.trunk == trunk && d.tracks_seed == seed) || d.shelf.contains_key(&(trunk.to_string(), seed))
}

/// `POST /sessions/ID/style`: the styled list for these params and pins, on
/// the session's model or the one `model` names, which becomes the
/// session's. A field the call leaves out keeps the session's own value --
/// the workbench's, once it has been used. A seed or a model whose tracks
/// the draft does not hold is head-stage work: the head pass is queued and
/// the answer says so, and the client asks again once the session is ready.
fn style_route(s: &Arc<Session>, body: &str, tx: &mpsc::Sender<Job>, b: &Bundle) -> Reply {
    #[derive(serde::Deserialize)]
    struct Msg {
        #[serde(default)]
        pins: Vec<[f64; 2]>,
        #[serde(default)]
        from_ms: Option<f64>,
        #[serde(default)]
        to_ms: Option<f64>,
        #[serde(default)]
        model: Option<String>,
    }
    let m: Msg = match serde_json::from_str(body) {
        Ok(m) => m,
        Err(e) => return Reply::Error(400, format!("bad style payload: {e}")),
    };
    let mut guard = match s.draft.try_lock() {
        Ok(g) => g,
        Err(_) => return Reply::Error(409, "working".to_string()),
    };
    let d = match guard.as_mut() {
        Some(d) => d,
        None => return Reply::Error(409, "not drafted yet".to_string()),
    };
    let base = *s.params.lock().unwrap();
    let params = match params_from_json(body, &base) {
        Ok(p) => p,
        Err(e) => return Reply::Error(400, format!("{e:#}")),
    };
    let model = m.model.unwrap_or_else(|| d.pack.clone());
    let bb = match b.select(&model) {
        Ok(bb) => bb,
        Err(e) => return Reply::Error(400, format!("{e:#}")),
    };
    let seed = seed_for(&bb, &params);
    if !holds(d, &bb.trunk(), seed) {
        drop(guard);
        queue(s, tx, Job::Retrack(s.clone(), model, seed));
        return Reply::Error(409, "redecoding".to_string());
    }
    // the tracks are held, so this is a swap and never a head pass
    if let Err(e) = retrack(d, &bb, seed) {
        return Reply::Error(500, format!("{e:#}"));
    }
    let pins: Vec<style::Action> = m
        .pins
        .iter()
        .map(|p| style::Action { at: p[0].round() as i64, pos: (p[1].round() as i64).clamp(0, 100) })
        .collect();
    let styled = style_draft(d, &bb.manifest, &params, &pins);
    let from = m.from_ms.unwrap_or(f64::NEG_INFINITY);
    let to = m.to_ms.unwrap_or(f64::INFINITY);
    let inside = |a: &style::Action| (a.at as f64) >= from && (a.at as f64) <= to;
    let actions: Vec<[i64; 2]> =
        styled.actions.iter().filter(|a| inside(a)).map(|a| [a.at, a.pos]).collect();
    Reply::Json(serde_json::json!({
        "n": actions.len(),
        "actions": actions,
        "artifacts": styled.artifacts,
        "art_rate": styled.art_rate,
        "filler": styled.filler,
        "model": d.pack,
        "env_seed": d.tracks_seed,
        "params": params_json(&params),
    }))
}

/// `POST /sessions/ID/workbench`: the review page for this session, on its
/// own port, for as long as it is open. The page's knobs and its model
/// choice land in the session through `apply_style`, and the funscript the
/// page plays is the cache's own copy, rewritten on every change. A second
/// call while the page is up answers the same URL.
fn workbench_route(s: &Arc<Session>, body: &str, tx: &mpsc::Sender<Job>, b: &Bundle) -> Reply {
    #[derive(serde::Deserialize)]
    struct Msg {
        #[serde(default = "yes")]
        browser: bool,
    }
    fn yes() -> bool {
        true
    }
    let body = if body.trim().is_empty() { "{}" } else { body };
    let m: Msg = match serde_json::from_str(body) {
        Ok(m) => m,
        Err(e) => return Reply::Error(400, format!("bad workbench payload: {e}")),
    };
    if let Some(url) = s.workbench.lock().unwrap().clone() {
        if m.browser {
            review::open_browser(&url);
        }
        return Reply::Json(serde_json::json!({ "url": url, "opened": false }));
    }
    let clip = match s.draft.try_lock() {
        Ok(g) => match g.as_ref() {
            Some(d) => match review_clip(d, b) {
                Ok(c) => c,
                Err(e) => return Reply::Error(500, format!("{e:#}")),
            },
            None => return Reply::Error(409, "not drafted yet".to_string()),
        },
        Err(_) => return Reply::Error(409, "working".to_string()),
    };
    let b2 = match b.select(&b.manifest.pack) {
        Ok(b2) => b2,
        Err(e) => return Reply::Error(500, format!("{e:#}")),
    };
    let (utx, urx) = mpsc::channel::<String>();
    let s2 = s.clone();
    let tx2 = tx.clone();
    std::thread::spawn(move || {
        let page = |url: &str| {
            *s2.workbench.lock().unwrap() = Some(url.to_string());
            let _ = utx.send(url.to_string());
        };
        let mut params = [*s2.params.lock().unwrap()];
        let man = &b2.manifest;
        let presets = review::Presets { dwell_ramp: man.plat_soft[0], still_eps: man.still_eps };
        let compare = clip.compare.clone();
        let mut choices = [review::Choice { model: clip.model.clone(), compare: None, spans: Vec::new(), cuts_ms: None }];
        let r = review::review(
            &[clip],
            &mut params,
            &mut choices,
            presets,
            m.browser,
            &page,
            |_, p, choice| apply_style(&s2, &tx2, &b2, p, choice, &compare),
        );
        *s2.workbench.lock().unwrap() = None;
        if let Err(e) = r {
            eprintln!("workbench: {e:#}");
        }
    });
    match urx.recv_timeout(Duration::from_secs(30)) {
        Ok(url) => Reply::Json(serde_json::json!({ "url": url, "opened": true })),
        Err(_) => Reply::Error(500, "the workbench did not start".to_string()),
    }
}

/// One knob turn or model choice on the workbench: the session's style and
/// model move, the file rewrites, and the count comes back for the page.
/// Tracks the draft does not hold are the worker's -- head-stage work, one
/// job at a time -- so this waits for each head pass rather than running it.
fn apply_style(
    s: &Arc<Session>,
    tx: &mpsc::Sender<Job>,
    b: &Bundle,
    p: &style::Params,
    choice: &review::Choice,
    compare: &std::path::Path,
) -> Result<review::Restyled> {
    let mut wants: Vec<(String, String, u64)> = Vec::new();
    for m in choice.compare.iter().chain(std::iter::once(&choice.model)) {
        let mb = b.select(m)?;
        wants.push((m.clone(), mb.trunk(), seed_for(&mb, p)));
    }
    loop {
        let missing = {
            let guard = s.draft.lock().unwrap();
            let d = guard.as_ref().context("not drafted yet")?;
            wants.iter().find(|(_, trunk, seed)| !holds(d, trunk, *seed)).cloned()
        };
        let Some((model, _, seed)) = missing else { break };
        queue(s, tx, Job::Retrack(s.clone(), model, seed));
        std::thread::sleep(Duration::from_millis(100));
        let st = s.status.lock().unwrap();
        if st.state == State::Failed {
            anyhow::bail!("{}", st.error.clone().unwrap_or_default());
        }
    }
    let mut guard = s.draft.lock().unwrap();
    let d = guard.as_mut().context("not drafted yet")?;
    let n = review_style(d, b, p, choice, compare)?;
    *s.params.lock().unwrap() = *p;
    s.style_rev.fetch_add(1, Ordering::Relaxed);
    Ok(n)
}

/// Queue head-stage work for a session, once: a session already queued or
/// working is left to finish.
fn queue(s: &Session, tx: &mpsc::Sender<Job>, job: Job) {
    let mut st = s.status.lock().unwrap();
    if matches!(st.state, State::Queued | State::Working) {
        return;
    }
    st.state = State::Queued;
    st.done.clear();
    st.error = None;
    let _ = tx.send(job);
}

/// The worker: one job at a time, in the order they were queued.
fn work(
    mut cli: Cli,
    b: Bundle,
    cache_root: PathBuf,
    params: style::Params,
    rx: mpsc::Receiver<Job>,
) {
    let pf = crate::prefetch::Prefetch::new(false, cli.hwaccel);
    while let Ok(job) = rx.recv() {
        // a cancel that belonged to the last job must not stop this one
        cancel::reset();
        let s = match &job {
            Job::Draft(s) | Job::Retrack(s, _, _) => s.clone(),
        };
        if s.cancelled.load(Ordering::Relaxed) {
            continue;
        }
        {
            let mut st = s.status.lock().unwrap();
            st.state = State::Working;
            st.stage = None;
            st.page = None;
            st.error = None;
        }
        let result: Result<()> = match job {
            Job::Draft(_) => {
                cli.videos = vec![s.video.clone()];
                cli.no_browser = !s.browser;
                (cli.crop_edit, cli.no_crop_edit) = match s.crop_edit {
                    Some(true) => (!cli.no_autocrop, false),
                    Some(false) => (false, true),
                    None => (false, false),
                };
                for_session(&b, &s).and_then(|bb| aim(&s, &cli, &cache_root).and_then(|cfg| {
                    let mut vrs = VrMap::new();
                    if let Some(c) = &cfg {
                        vrs.insert(s.video.clone(), c.clone());
                    }
                    *s.vr.lock().unwrap() = cfg;
                    let sink: Arc<dyn StageSink> = s.clone();
                    let out = draft(
                        &cli,
                        &bb,
                        &s.video,
                        &cache_root,
                        &params,
                        0,
                        1,
                        &pf,
                        None,
                        &vrs,
                        Some(sink),
                    )?;
                    match out {
                        Outcome::Drafted(d) => {
                            *s.draft.lock().unwrap() = Some(d);
                            Ok(())
                        }
                        Outcome::Skipped => anyhow::bail!("the draft was skipped"),
                    }
                }))
            }
            // The head pass shelves the tracks and leaves the model in hand
            // where it was: the style that asked for them switches, a job
            // does not.
            Job::Retrack(_, model, seed) => {
                s.setup("console.stage.encode");
                let mut guard = s.draft.lock().unwrap();
                match guard.as_mut() {
                    Some(d) => (|| {
                        let (was, was_seed) = (b.select(&d.pack)?, d.tracks_seed);
                        retrack(d, &b.select(&model)?, seed)?;
                        retrack(d, &was, was_seed)
                    })(),
                    None => Err(anyhow::anyhow!("nothing to re-decode")),
                }
            }
        };
        let mut st = s.status.lock().unwrap();
        st.stage = None;
        st.page = None;
        match result {
            Ok(()) => st.state = State::Ready,
            Err(e) => {
                st.state = State::Failed;
                st.error = Some(if cancel::is_cancel(&e) {
                    "cancelled".to_string()
                } else {
                    format!("{e:#}")
                });
            }
        }
    }
}

/// The VR aim a session drafts under: the console's own decision
/// (`vr_need`), with the client's word in place of the picker's mark and
/// `--vr`, and the aiming page in place of the console prompt. The sidecar
/// the page saves is the one a console run of the same video reads.
fn aim(s: &Session, cli: &Cli, cache_root: &std::path::Path) -> Result<Option<vr::Config>> {
    let force = match &s.ask {
        VrAsk::Flat => return Ok(None),
        VrAsk::Fixed(c) => return Ok(Some(c.clone())),
        VrAsk::Auto => false,
        VrAsk::Aim => true,
    };
    let Some(need) = vr_need(&s.video, cache_root, force, force) else {
        return Ok(None);
    };
    // answered before, and not asked to answer again: reuse in silence
    if need.clip.cfg.touched && !force {
        let cfg = need.clip.cfg;
        return Ok((!cfg.skip).then_some(cfg));
    }
    let mut done = vr::prep(vec![need.clip], !cli.no_browser, &|url| s.page("console.stage.vr", url))?;
    let cfg = done.pop().map(|c| c.cfg).context("the aiming page returned no clip")?;
    let info = if cfg.skip {
        "not VR".to_string()
    } else {
        format!(
            "{} {} yaw {:.1} pitch {:.1}{}",
            cfg.layout,
            cfg.projection,
            cfg.aim.yaw,
            cfg.aim.pitch,
            cfg.range_ms
                .map(|[a, b]| format!(" {:.1}-{:.1} s", a / 1000.0, b / 1000.0))
                .unwrap_or_default()
        )
    };
    s.done("console.stage.vr", &info);
    Ok((!cfg.skip).then_some(cfg))
}

fn query_map(q: &str) -> HashMap<String, String> {
    q.split('&')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// A token nobody outside this process can guess: the process, the clock
/// and the allocator's own randomness, hashed. Loopback keeps other machines
/// out; this keeps other programs on the same machine out.
fn fresh_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = blake3::Hasher::new();
    h.update(&std::process::id().to_le_bytes());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    h.update(&now.to_le_bytes());
    for _ in 0..4 {
        let r = std::collections::hash_map::RandomState::new().build_hasher().finish();
        h.update(&r.to_le_bytes());
    }
    h.finalize().to_hex()[..32].to_string()
}
