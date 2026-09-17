# The GoblinScript API

`goblinscript --serve` runs the goblins as a sidecar for another program:
anything that wants a funscript drafted from a video and re-styled around
its own edits. The server drafts a video once, keeps the per-row tracks in
memory, and answers a styled action list in milliseconds for every change
of a knob or a placed point. The decisions a person makes -- where a VR
viewport points, which rect each shot is cropped to, and the styling knobs
-- are made on the goblins' own browser pages, which the server opens on
your behalf or hands to you as a URL.

**The API is unstable.** Routes, fields and units change whenever a better
shape exists, without an alias or a version bump. The client that ships
beside a build is the one that fits it. Read the `goblinscript` field of
the handshake and refuse a build you were not written against.

## Starting the server

```
goblinscript --serve [--port N] [--quiet] [--cache DIR] [drafting options...]
```

Every drafting option on that command line (`--cache`, `--no-autocrop`,
`--dwells`, `--filler-gap` ...) is what each session drafts under, and the
styling options are each session's starting style. `--port 0` (the default)
takes any free port. The server binds to `127.0.0.1` only.

The first line on standard output is the handshake, one JSON object:

```json
{"checkpoint":"...","goblinscript":"0.6.0","port":60749,"row_hz":30.0,"token":"7c23a5aa..."}
```

The keys come in the server's own order, so parse the line and look for the
`goblinscript` key rather than matching a prefix. Every line after it is the
drafting log, which is worth keeping when a stage fails. `--quiet` keeps the
log short.

Every request carries the token in an `x-goblin-token` header. A request
without it answers `401`.

The server exits when its standard input closes -- spawn it with a pipe on
stdin and hold the pipe, and a client that dies takes its goblins with it --
or when a client posts `/quit`. Do not close stdin before you are done.

Every answer is JSON. An error is `{"error": "..."}` with a `4xx` or `5xx`
status.

## Routes

| Route | Body | Answer |
|---|---|---|
| `GET /health` | | the build, the models, the session count |
| `GET /manifest?model=` | | one model's deploy constants, the default params, the preset ladders |
| `POST /sessions` | `{video, vr?, crop_edit?, browser?, model?}` | the session, drafting |
| `GET /sessions/ID` | | the session's status |
| `GET /sessions/ID/tracks?from_ms&to_ms` | | the per-row tracks on the video clock |
| `POST /sessions/ID/style` | `{params..., pins?, from_ms?, to_ms?}` | the styled actions |
| `POST /sessions/ID/workbench` | `{browser?}` | `{url, opened}` |
| `DELETE /sessions/ID` | | `{id, deleted: true}` |
| `POST /quit` | | `{bye: true}` |

### `GET /health`

```json
{"goblinscript": "0.6.0", "checkpoint": "...", "basis_id": "...", "encoder": "...", "row_hz": 30.0,
 "model": "v0.5.1", "models": ["v0.3.0", "v0.4.0", "v0.5.1", "v0.5.1-snap10"],
 "trunks": {"v0.3.0": "…", "v0.4.0": "…", "v0.5.1": "…", "v0.5.1-snap10": "…"}, "default": "v0.5.1", "sessions": 1}
```

The bundle carries one perception and several models, each named for the
release that shipped it. `models` lists them, `default` is the one the
bundle ships as, and `model` is the one this server drafts with when a
session names none: its `--model`, else the default. `trunks` names each
model's trunk: models with the same trunk read the video identically and
differ only in styling. A session switches between any of them over the
same encode (`model` on a style call), so a client can offer the choice
and compare models' drafts of one video without a second encode.

### `GET /manifest`

`?model=NAME` answers another model's constants; without it, the server's
own. A name the bundle does not carry answers `400`.

The constants a client needs to draw and to place things: `row_hz` (rows
per second of the tracks), `row0_ms` (where row 0 sits on the clip's
clock), `max_pos_rate` (the device cap, position units per second),
`cut_ease` (ms), `pin_tol_ms` and `pin_horizon_ms` (the pins pass, below),
`env_seed` (the model's own authorship seed), the model's identity
(`model`, `label`, `trunk`, `checkpoint`, `epoch`, `basis_id`, `encoder`,
`bundle_version`, `v_std`),
`defaults` (the style params every session starts from, in the shape of
`params` below), and `presets`: each preset label's resolved number, so a
UI can write `normal (0.65)` next to a dropdown.

### `POST /sessions`

```json
{"video": "C:/videos/clip.mp4", "vr": true, "crop_edit": false, "browser": false, "model": "v0.3.0"}
```

Opens the session for `video`, or joins the one already open for it, and
queues its draft. A session is keyed by the video's content fingerprint,
so a second client asking for the same file joins the first, and the cache
under `--cache` is what makes a restart of the same video seconds instead
of minutes.

- `vr` -- absent to detect (a video that looks like VR opens the aiming
  page when nobody has answered for it yet); `true` to open the aiming
  page whatever the detector thinks, and to aim again a video that has an
  aim; `false` to draft the raw frames; or a sidecar object (the `cfg`
  the aiming page's own `/api/state` returns) to apply as it is, no page.
- `crop_edit` -- absent to open the crop page when the crop was just
  probed (a cached plan opens nothing); `true` to open it always; `false`
  never. Irrelevant under `--no-autocrop`.
- `browser` (default `true`) -- open a page in the default browser from
  the server. `false` leaves that to the client, which reads the URL off
  the status.
- `model` -- the model this session drafts with, one of `/health`'s
  `models`; absent, the server's own. A name the bundle does not carry
  answers `400`. A session is keyed by the video alone, so a second client
  naming another model for the same file joins the first session, and
  switches with a style call's `model`.

The answer is the status below, in state `queued` or `working`. A missing
video answers `400`.

### `GET /sessions/ID`

```json
{
  "id": "…", "video": "…", "model": "v0.5.1",
  "state": "working",
  "stage": {"key": "console.stage.encode", "pos": 1240, "len": 5400},
  "page": null,
  "done": [{"key": "console.stage.normalize", "info": "done at 576p (12.3s, 8.1x realtime)"}, "…"],
  "error": null,
  "ready": null,
  "vr": null,
  "params": {"…": "…"}, "flags": "--dwells normal …",
  "style_rev": 0,
  "workbench": null
}
```

- `model` -- the model whose tracks the session holds in hand: the last
  one a style call or the workbench chose, else the one it drafted with.
- `state` -- `queued`, `working`, `ready` or `failed`. `working` covers the
  first draft and every re-decode after it.
- `stage` -- the stage in flight: its key, and `pos`/`len` when it has a
  measurable bar (`null` when it does not). Keys are `console.stage.vr`,
  `console.stage.normalize`, `console.stage.exposure`, `console.stage.cuts`,
  `console.stage.autocrop`, `console.stage.cropedit` and
  `console.stage.encode`, and a client is free to name them as it likes.
- `page` -- `{key, url}` while the stage in flight is a browser page the
  goblins are waiting on (the VR aim or the crop check). The draft
  continues when Done is pressed on that page. Open the URL yourself when
  you passed `browser: false`.
- `done` -- the stages that cleared since the last job began, with their
  summary lines.
- `error` -- the failure's text once `state` is `failed`; `cancelled` for
  a session deleted mid-job.
- `ready` -- present once the tracks have landed and from then on:
  `rows`, `row_hz`, `row0_ms`, `dur_ms`, `t0_ms`, `cuts_ms`, `shot_edges`,
  `env_seed`, `conf_mean`, `crop`. Times are on the VIDEO clock, the source
  file's: `t0_ms` is where the drafted clip starts on it, non-zero only for
  a VR draft whose aiming page trimmed a range, and `row0_ms` already
  includes it. `cuts_ms` are the detected shot cuts. `crop` is the
  auto-crop plan as fractions of the frame per segment, or `null` when the
  whole frame was drafted.
- `vr` -- the sidecar the draft ran under (layout, projection, `aim:
  {yaw, pitch}`, `range_ms` ...), or `null` for a flat draft.
- `params` and `flags` -- the session's style, as the fields a style call
  takes and as the command-line flags that reproduce it.
- `style_rev` -- counts the changes the workbench applied. A client that
  caches styled lists re-asks when it moves.
- `workbench` -- the workbench's URL while its page is up, else `null`.

A poll during a re-decode answers from the status without waiting.

### `GET /sessions/ID/tracks?from_ms=A&to_ms=B`

The per-row tracks for the rows inside `[A, B]` on the video clock, the
whole clip without a bound. Row `i` of the answer sits at
`row0_ms + (first_row + i) * 1000 / row_hz`.

```json
{"first_row": 29, "rows": 61, "row_hz": 30.0, "row0_ms": 33.333,
 "vmarg": [...], "level": [...], "env": [...], "conf": [...], "period": [...],
 "blo": [...], "bhi": [...], "plat_top": [...], "plat_bot": [...], "rev_top": [...], "rev_bot": [...]}
```

`vmarg` is the marginal velocity (the phase), `level` the position level,
`env` the amplitude envelope, `conf` the model's confidence in `[0, 1]`,
`period` the stroke period, `blo`/`bhi` the band rails, `plat_top`/`plat_bot`
the dwell head, `rev_top`/`rev_bot` the reversal events. A value the model
did not emit is `null`; a track the bundle does not carry is `null` whole.
`409 working` while the tracks are being made, `409 not drafted yet` before
the first draft lands.

### `POST /sessions/ID/style`

```json
{"model": "v0.3.0", "dwells": "eager", "intensity": 1.2, "pins": [[2000, 90], [2500, 10]], "from_ms": 0, "to_ms": 30000}
```

Answers the styled action list for the session's tracks under these
params, milliseconds, in a few milliseconds:

```json
{"n": 14, "actions": [[at_ms, pos], ...], "artifacts": [[from_ms, to_ms], ...], "art_rate": 0.4,
 "filler": [[from_ms, to_ms], ...], "model": "v0.3.0", "env_seed": 0, "params": {"…": "…"}}
```

`model` (optional) styles under that model, with its own decode constants,
and makes it the session's; absent, the session's model.

Every styling field is optional and falls back to the session's style (the
workbench's, once it has been used). `from_ms`/`to_ms` keep only the
actions inside that span; the styling itself always runs on the whole
clip, so an action near the edge is the same one a whole-clip call gives.
`artifacts` are the spans whose action sequence reads hot under the rarity
prior, `art_rate` their count per minute, `filler` the spans the
background rhythm replaced.

**Pins.** `pins` is `[[ms, pos], ...]`, the vertices the author placed. A
pin replaces the model's vertices within `pin_tol_ms` of it and bends the
level around it toward the author's, interpolated between pins and fading
over `pin_horizon_ms` past the outermost. The model's other reversal times
stay where they were. A client that treats the author's points as pins and
draws the answer around them has the whole human-in-the-loop flow in one
call.

**A moved seed or model.** `env_seed` feeds the envelope and a model's
trunk is the head graph itself, both upstream of styling, so a seed or a
trunk the session holds no tracks for is head-stage work: the call answers
`409 redecoding`, the session reads `working` until the tracks are made,
and the same call then answers. The session keeps every set of tracks it
has made, so going back to a model or a seed is never a second head pass.
Only one job runs at a time.

The styling fields, all optional:

| Field | Values |
|---|---|
| `style` | `composed` (production), `level` |
| `dwells` | `cautious`, `normal`, `eager` |
| `stillness` | `low`, `normal`, `high` |
| `depth` | `off`, `subtle`, `even`, `locked` |
| `intensity` | 0.5 .. 2.0, stroke amplitude about mid-range; 1 = the model's own |
| `range` | `[lo, hi]`, 0 <= lo < hi <= 100, the position range the track maps onto |
| `max_speed` | >= 0, position units per second, the device cap |
| `cut_ease` | >= 0 ms, the seam ease at a shot cut |
| `env_seed` | an integer, or `null` for the bundle's own |
| `dwell_ramp` | 0 .. 1, or `null` to stand on the `dwells` preset |
| `still_eps` | 0 .. 60, or `null` to stand on the `stillness` preset |
| `depth_dose`, `depth_window` | 0 .. 1 and 0 .. 60 s, or `null` to stand on the `depth` preset |
| `filler_gap_s` | >= 0 s; a motion-free gap at least this long takes the filler rhythm; 0 = off |
| `filler_rate`, `filler_amp` | 1 .. 300 per minute, 1 .. 50 units, while the filler is on |
| `filler_pattern` | `steady`, `burst` |
| `filler_burst`, `filler_rest_s` | strokes per burst (>= 1), rest between bursts (>= 0 s) |
| `filler_model_w` | 0 .. 1, pure rhythm to model-shaped |
| `filler_min_real_s`, `filler_real_v`, `filler_ramp_s`, `filler_max_bridge_s` | >= 0 |
| `filler_sway`, `filler_sway_s` | 0 .. 1, and > 0 s |

A field absent keeps the session's value. A `null` where the table allows
it clears an expert override. A value out of range answers `400` with the
reason.

### `POST /sessions/ID/workbench`

```json
{"browser": false}
```

Opens the workbench for a session whose draft is ready: the goblins'
review page, with the video, the scripted motion as a simulator, and every
styling knob, on its own port for as long as it is open. Answers
`{"url": "http://127.0.0.1:PORT/", "opened": true}`; a second call while the
page is up answers the same URL with `opened: false` (and opens the browser
again unless `browser` is false). `409 not drafted yet` before the draft
lands.

The knobs settled on the workbench ARE the session's style from then on.
Every style call starts from them, the status carries them under `params`
and `flags`, and `style_rev` moves on every change, which is a client's
cue to re-ask for its styled lists. The page's Done button closes the
workbench and clears `workbench` in the status; open it again whenever
you like.

### `DELETE /sessions/ID`

Forgets the session. A job in flight is cancelled; a page waiting on a
person is closed. The cache stays, so the same video opens again in
seconds. `404` for an unknown id.

### `POST /quit`

Answers `{"bye": true}` and exits.

## The pages

Three browser pages belong to a session, and each has its own loopback
server on its own port, with no token: the VR aim and the crop check
before the encode, the workbench after it. A person uses them in a
browser. A client can drive them too, through the endpoints the page
itself uses, which is how the smoke test aims a clip without a person.

**The VR aim** (`page.key` = `console.stage.vr`): `GET /api/state` (the
current clip's `cfg`, `src_w`, `src_h`, `duration_ms`), `POST /api/config`
with any of `layout` (`sbs`, `tb`, `mono`), `projection` (`hequirect`,
`fisheye`, `equirect`), `ih_fov`, `h_fov`, `v_fov`, `roll`, `out_w`,
`out_h`, `range_ms` (`[t0, t1]` on the source clock, at least a second, or
`null` for the whole video), `POST /api/aim` with `{"aim": {"yaw", "pitch"}}`,
`POST /api/skip` (toggles "not VR after all"), `POST /api/done`. The
sidecar the page saves is filed under the video's fingerprint in the
cache, so a later session of the same video reuses it in silence.

**The crop check** (`console.stage.cropedit`): `GET /api/state` (`segs`
with each shot's `t_ms`, `rect`, `auto`, `hand`), `POST /api/rects` with
`{"i": segment or -1 for all, "rect": [x, y, w, h] as fractions of the
frame, or null for the probe's own}`, `POST /api/done`. A rect is kept
inside the frame and no smaller than the zoom cap.

**The workbench**: `GET /state` (the clip, its `params`, `defaults`,
`styling` flag, `error`), `POST /params` with `{"clip": 0, ...styling
fields...}`, `GET /script/0` (the funscript as styled), `GET /conf/0`,
`GET /video/0`, `POST /done`.

## A session, start to finish

```python
import json, subprocess, time, urllib.request

srv = subprocess.Popen(["goblinscript", "--serve", "--quiet"],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
hello = next(json.loads(l) for l in srv.stdout if l.startswith("{") and "goblinscript" in l)
base, tok = f"http://127.0.0.1:{hello['port']}", hello["token"]

def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"x-goblin-token": tok, "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

sess = call("POST", "/sessions", {"video": "clip.mp4"})
while (st := call("GET", f"/sessions/{sess['id']}"))["state"] not in ("ready", "failed"):
    if st["page"]:
        print("a page is waiting:", st["page"]["url"])   # opened in the browser by the server
    time.sleep(0.5)

styled = call("POST", f"/sessions/{sess['id']}/style",
              {"pins": [[12000, 85], [12400, 15]], "from_ms": 10000, "to_ms": 20000})
print(styled["n"], "actions between 10 and 20 s")

wb = call("POST", f"/sessions/{sess['id']}/workbench")   # the review page opens; tune there
call("POST", "/quit")
```

The style call is the one to make on every edit: send the author's points
as pins, draw what comes back. Poll the status while the workbench is open
and re-ask when `style_rev` moves.
