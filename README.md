> English | [简体中文](README.zh-CN.md)

# GoblinScript

[![CI](https://github.com/Kotone-Hikaru/GoblinScript-Sakuya/actions/workflows/ci.yml/badge.svg)](https://github.com/Kotone-Hikaru/GoblinScript-Sakuya/actions/workflows/ci.yml)

Drop in a video, get a funscript. **Made by real goblins.**

A goblin watches the video and scrawls a matching script: strokes where the
action is, timed to the motion on screen. The result lands next to the video
as `video.funscript`, ready for any player that reads funscripts. Treat it as
a solid first draft -- the goblin aims to nail the rhythm and the timing, and
you can fine-tune anything else in your favorite editor.

> **This is a fork of ofs69's GoblinScript.**
> [Kotone-Hikaru/GoblinScript-Sakuya](https://github.com/Kotone-Hikaru/GoblinScript-Sakuya)
> builds on [ofs69/GoblinScript](https://github.com/ofs69/GoblinScript) -- a GPU
> throttle, the spanning bench, hand-drawn wave pieces, a slider panel, a shelter
> for the picture, a batch crop pass and a window that comes forward when it
> wants you. The app, its pages, the training pipeline and the models are
> **ofs69's work**; everything that is not listed under
> [What this fork adds](#what-this-fork-adds) is theirs. See [Credits](#credits).

> **This repository is the inference client only.**
>
> It is the app that *runs* a trained model -- decode, perception, and the
> styling that turns what the goblins see into strokes. It is **not** the
> training pipeline, and it does **not** contain the models themselves. They
> are a separate ~250 MB bundle -- one perception and every model a release
> carries -- baked into released binaries; a build from source needs one
> passed with `--bundle`.
>
> The training pipeline is
> [GoblinTrainingGrounds](https://github.com/ofs69/GoblinTrainingGrounds).
> It trains a model on your own video/funscript pairs, compares it with the
> released one, and exports it as a pack this app loads.

## Using it

Double-click `goblinscript.exe` and a picker opens in the terminal: browse with
the arrow keys, select with Space, press **S** to set the goblins to work.
Or from a command line:

```
goblinscript video.mp4                  the script appears next to the video
goblinscript D:\clips                   every video in that folder
goblinscript video.mp4 --out D:\scripts put the scripts somewhere else
goblinscript video.mp4 --model v0.4.0  draft with another of the models it carries
goblinscript video.mp4 --models v0.5.1,v0.3.0
                                        both, from one look at the video:
                                        video.funscript and video.v0.3.0.funscript
goblinscript video.mp4 --throttle gentle
                                        keep the GPU usable; see the fork's notes below
goblinscript D:\clips --batch-crop      judge every crop at the end, then draft them all
```

A release carries several models, each named for the release that first
shipped it, and every one of them watches the video the same way -- so a
second model costs seconds, not a second encode. `--model` picks one, and
`--models` writes them side by side without a review. The review page reads
the video with every model while drafting, so it switches the model of a
script at once. Under the video, **script** names the model writing the
script and **vs** lays another line under it: another model's, written with
the same knobs, or the **pinned** script, held as it was with its model and
knobs. **⇄** (or **x**) swaps the script and the line under it, so a choice is
made by flipping between them while the video plays. Every script records the
model and the settings that wrote it.

Before the goblins start, a page opens in your browser with the part of the
picture they are about to watch drawn on each shot. Drag that rect onto the
action if they picked the wrong part, or click through if they got it right;
what you draw is kept for that video. Turn the page off with `--no-crop-edit`,
or with **K** in the picker, and a batch runs with nobody at the keyboard.

On a batch, judging every crop as each video is transcoded means sitting down
once per video. **`--batch-crop`** (or **J** in the picker) makes the run take
two passes instead: the first transcodes, cuts and asks about the crop for every
video, the second drafts them all with the answers from the cache and opens no
page at all. The crop is still decided *before* the expensive stage, so
correcting one is still free -- only *when* you are asked changes. It is off by
default because it costs machine time to save yours: see
[The batch crop pass](#the-batch-crop-pass).

`goblinscript --help` lists every option. A VR video opens a page in your
browser to aim it once; flat video needs no flags at all. English and Chinese
ship with it, and any `languages/*.json` you translate shows up on the **G**
key. Existing scripts are never overwritten without `--force`, and the video
file itself is never touched.

**Nothing on screen has to be readable while you work.** Both browser pages
carry a shelter -- two covers over the picture, reachable with their buttons or
by cycling **H**. See [The shelter](#the-shelter).

### Serving an editor

`goblinscript --serve` runs the goblins as a sidecar for another program: a
funscript editor that draws the draft under its own timeline and re-styles it
every time the author moves a point. It binds to localhost, prints one JSON
line with the `port` and the access `token`, then answers requests until its
standard input closes or a client posts `/quit`. Every drafting option on that
command line (`--cache`, `--no-autocrop`, `--dwells` ...) is what each session
drafts under.

**The API is unstable.** Routes, fields and units change whenever a better
shape exists, without an alias or a version bump; the client that ships beside
a build is the one that fits it. Every request carries the token in an
`x-goblin-token` header. `API.md` is the reference: every route, every
field, the pages, and a client in twenty lines.

| Route | What it does |
|---|---|
| `GET /health` | the build, the model, the session count |
| `GET /manifest` | the deploy constants, the default params, the preset ladders |
| `POST /sessions` `{video, vr?, crop_edit?, browser?}` | open (or join) the session for a video and draft it |
| `GET /sessions/ID` | the stage in flight, a page waiting, the stages done, or the clip's clock once ready |
| `GET /sessions/ID/tracks?from_ms&to_ms` | the per-row tracks on the video clock |
| `POST /sessions/ID/style` `{params..., pins?, from_ms?, to_ms?}` | the styled actions, milliseconds |
| `POST /sessions/ID/workbench` `{browser?}` | the review page for this session, as `{url}` |
| `DELETE /sessions/ID` | forget the session (its cache stays) |
| `POST /quit` | exit |

The decisions a person makes keep the browser pages the console opens for
them. Before the encode, the VR aiming page and the crop page: a session
that reaches one reads `working` with the page's `key` and `url` under
`page`, and the draft continues when Done is pressed there. `vr` is absent
to detect, `true` to open the aiming page whatever the detector thinks (and
to aim again a video that has an aim), `false` to draft the raw frames, or
a sidecar object to apply as it is. `crop_edit` is absent to open the crop
page when the crop was just probed, `true` to open it always, `false` never.
`browser: false` leaves opening a page to the client; the server opens it
otherwise.

After the draft, the **workbench**: the review page, with the video, the
simulator and every styling knob, on its own port for as long as it is
open. Open it as often as you like. The knobs settled there are the
session's style from then on -- every style call starts from them, the
status carries them under `params` and `flags`, and `style_rev` counts the
changes so a client knows to ask again.

A style call takes the same fields the review page posts, every one optional
and falling back to the session's style, plus `pins`: `[[ms, pos], ...]`
vertices the author placed. A pin replaces the model's vertices within 60 ms
of it and bends the level around it toward the author's; the model's other
reversal times stay where they were. A seed that moved is head-stage work:
the call answers `409 redecoding` and the session reads `working` until the
tracks are back.

## What this fork adds

Everything below is [Kotone-Hikaru](https://github.com/Kotone-Hikaru)'s work on top
of upstream `5bf8d6f`, written for the review page and the GPU it runs on. The
app and its pipeline are unchanged where these do not touch them. The bundle
carries two extra models, trained here rather than shipped by upstream -- see
[the two anime models](#the-two-anime-models), which also says why you should
keep using the default one unless your footage looks like their training set.

Three of these are about drafting -- how a script is cut into pieces, how a piece
can be drawn rather than modelled, and how the GPU is kept usable while it works.
The last three are about the other half of the job, the hours spent *watching and
deciding*: covering a picture that should not be read over your shoulder, judging
a batch's crops in one sitting, and a window that comes forward when it needs you.
They are the cheapest things here, and the first one is worth knowing before you
open a video at work.

### The spanning bench

The review page grows a second layer under the timeline: **spans**. A razor
cuts the draft into pieces and each piece names the model that writes it, so a
scene that one model reads badly can be handed to another without re-drafting
the rest. Pieces are edges you drag, and dragging an edge can snap to the
nearest shot cut (**S**). The video's own cuts are editable too (**B**) --
adding or removing one re-runs the head pass, because it changes what the
models were shown. Nothing in the bench touches the draft file: it is the
session's edit, like a knob.

Each piece also carries **its own intensity**, 0 to 5, where 1 is the model's
own answer. It scales about the piece's *own mean* rather than about mid-range,
so turning a quiet wave up widens it instead of pressing it against the floor --
and it moves the piece's vertices rather than the rows they were found on, so a
stronger setting can never cost the piece a point.

A **confidence suggestion** can propose a model per piece from each model's own
confidence, sampled with one instant restyle each. It is a proposal and nothing
else: it is off until asked for, and applying it is its own button. Which models
are even on the ballot is yours to set, per model, and the choice is remembered
in `settings.json`; a model you switch off is never sampled, so it cannot spend
your wait or win a shot on a self-assessment you do not trust.

Every step is undoable (**Ctrl+Z** / **Ctrl+Shift+Z**), one step per edit, with
the piece, cut, model or merge that it changed named in the tooltip.

### Hand-drawn wave pieces

A piece can be written from a waveform you draw instead of from the model. Six
waves ship -- triangle, sine, rounded triangle, single peak, pulse and a
one-way glide -- each with its own knobs (apex position, corner radius, rise
and fall shares, charge, depth...). The panel's every number is also a slider
you drag, and the two bounds are a single two-headed band you can also grab in
the middle and slide whole, mirroring the one-hand slider.

The waves live in `src/waves.json` as **data**, not code: a segment's duration
and level are arithmetic expressions over the wave's own parameters, so adding
or reshaping a knob is an edit to that file rather than a code change. The file
is baked into the binary, but a `waves.json` sitting **beside the exe** is read
in preference to it -- so a shape can be re-tuned by editing that copy and
reloading the page, with no rebuild and no reinstall. A drawn piece
is *solved* from its wave rather than scanned off the frame grid, so a triangle
reaches exactly 0 and 100 instead of stopping a row short. Two switches per
piece let it ignore the speed cap (**ignore limits**) and the video's shot cuts
(**ignore cuts**) -- the latter on by default, since a cut in the footage is not
a cut in your drawing.

### A GPU throttle

`--throttle` keeps the goblins off the whole GPU while they draft, so a machine
stays usable during a long batch: a closed loop samples GPU load and naps
between encoder windows for a duty-proportional fraction of the work it just
did. It is guarded so it cannot hunt -- a deadband, an averaged load, per-step
rise and fall limits, and a dwell before the loop may reverse -- and it fails
open, disarming itself if the GPU cannot be read rather than blocking a draft.
Three presets -- `gentle`, `balanced`, `aggressive` -- cycle with **P** in the
picker and are remembered in `settings.json`; `--throttle custom` with
`--throttle-*` flags tunes it by hand, and `--no-throttle` overrides a
remembered choice. The boot screen always reports which state it is in, so
"nothing is holding my GPU back" is answerable at a glance. `goblinscript --help`
lists the presets and every `--throttle-*` flag; the presets themselves are the
three the picker cycles.

### The shelter

Both pages that show the picture -- the crop page and the review page -- can
cover it. Not pause it, not stop the work: **cover it**, because a script is
written by watching a video and this material is not something to leave on a
screen that a passing colleague can read.

Two covers, each a toggle (pressing the lit one puts the picture back), so two
buttons carry three states and the state is visible:

* **still** -- an opaque card with a static texture. It costs nothing: the
  compositor blends it like any background.
* **frost** -- the frame, sampled into a canvas **one eighth** of the picture
  each way and blurred *at that size*, then scaled up by CSS.

**H** cycles off → still → frost → off. It is one key rather than a choice
because it is reached for suddenly, when someone walks in, and that is not a
moment to be picking between two covers. (The picker uses **H** for a different
switch -- the window raise below -- because a terminal and a page never have the
keyboard at the same time.)

Neither cover takes pointer events, so the native transport, the crop
rectangle and its handles, the simulator and every readout keep working behind
them -- a rectangle is judged by where it sits, which does not require reading
the pixels. And the frost follows a seek *while paused*, which is how both
pages are actually used: a cover that only refreshed during playback would leave
the previous frame on screen, which is exactly the leak it exists to prevent.

The frost is a canvas rather than `filter: blur()` on the video for one reason,
and it is the difference between cheap and free. A blur applied inside a small
canvas reaches the compositor as a finished bitmap. The obvious version -- a CSS
blur on the video inside a `transform: scale(8)` wrapper -- does not: the
compositor rasterises a filtered layer at the size it is finally drawn at, so
the blur re-runs at the magnified scale and the saving evaporates. The two
differ by about **64×** (the area). The blur radius itself is nearly free either
way: Skia's blur is three box passes, and a box blur sums with a sliding window,
so cost is independent of radius.

### The batch crop pass

`--batch-crop` (picker **J**) asks about every crop **at the end of the batch**
instead of one per video, by making the run take two passes:

| | what it does |
|---|---|
| **pass 1** | transcode → exposure → cuts → probe the crop → **open the page**. Stops there. |
| **pass 2** | the crops are decided. Take each answer from the cache and run the expensive stage. **No page opens.** |

The property that makes this a schedule change rather than a re-architecture is
one the pipeline already had: **the crop is decided before the expensive stage**,
which is what makes correcting it free. Batching changes *when* you are asked,
never *where* the question sits relative to the work it affects.

What it costs, and why it is off by default: the transcode of video *N+1* used
to run in the GPU shadow of video *N*'s expensive stage. Splitting the passes
removes that overlap, so a batch spends **more pure machine time and less of
yours**. That is a trade to choose, not one to discover.

The page itself changes to match: instead of one page per video in a fixed
order, the whole batch is listed down the left -- the original filename, the
number of it, and a dot on any video whose crop you have already changed by
hand. Any row can be picked at any time, in any order, and **next unjudged**
jumps to the first one still plain. The batch is submitted once, at the end, for
all of them. A single video gets no rail, because one row is a title bar.

A single video is never split into two passes -- there is nothing to batch, and
a second pass would cost a full walk over the cache to arrive at the same place.
The first pass's working files are deliberately **not** swept: the normalise,
the cut list and the crop plan are exactly what the second pass comes back for.

### A window that comes forward

When a stage stops and waits for a person -- the crop page, the review page --
the browser window is lifted to the front of the Z order if it has drifted
behind something else. A run blocked on a page nobody can see looks exactly like
a run still working, which is the failure this prevents.

It does **not** take the keyboard focus, and that is the whole design rather
than half of it. On Windows, taking the foreground is privileged: it is refused
unless the caller is already the foreground process, was started by it, or has
just received input. This program is a console app whose console window belongs
to the terminal, and by the time a stage needs attention the person has long
since moved elsewhere -- so it is essentially never eligible, and a focus grab
would silently degrade to a taskbar flash. Reordering the Z axis is *not*
privileged: toggling the topmost bit on and off, both halves with
`SWP_NOACTIVATE`, lifts the window over its neighbours without asking for
anything.

The price is small and stated plainly: the window comes up, the caret does not,
so you click it and carry on. The alternative -- a real focus grab -- can swallow
characters someone was typing into another window, and a tool that announces
itself has no business doing that.

The window is found by title, matched case-insensitively against the pages' own
`<title>`. A title alone is not enough, though, and finding that out cost a bug:
the console this pipeline runs in is titled with the program's name too, so a
bare substring search can match the black terminal box just as readily as the
browser, and which one won was an accident of what you had touched last. The
choice is therefore made on the window **class**: a browser class wins
(`Chrome_WidgetWin_*` covers Chrome, Edge, Brave, Vivaldi and Opera;
`MozillaWindowClass`, `IEFrame`), a terminal class is **never** raised whatever
its title says, and any other class that matches the title is still accepted so
an unfamiliar browser works instead of silently doing nothing.

The switch lives in the picker (**H**) with the other standing preferences and
is remembered in `settings.json`. It defaults **on**: unlike the switches that
spend a wait or rewrite pieces, it only decides whether a window is visible.

### The two anime models

The bundle carries two models that upstream does not ship. Both were trained
here, with ofs69's own training pipeline, on anime and MMD footage: a corpus of
**1,029 clips (77.6 hours)**, of which **128 clips (9.7 hours) were held out of
training** and are the only clips that every number below comes from.
`anime-v1` trained on the remaining 901, `anime-v1.1-from-0.3.0` on 899 after two
clips were dropped between the two runs. Both start from an upstream checkpoint.
They are `anime-v1` and `anime-v1.1-from-0.3.0`, and **the default is still
upstream's `v0.5.1`**. Nothing selects them for you.

They are not a general improvement, and on footage unlike their training set
they may well be worse than the default. Because the holdout is drawn from that
same anime corpus, what these numbers measure is "does this model fit anime
footage" -- not "is this model better":

| vs a reference arm | corr | travel | slow-section spikes `sl>3x` |
|---|---|---|---|
| upstream `v0.6.0` (reference) | 0.359 | 0.690 | 5.30 |
| `anime-v1` | 0.541 | 0.523 | **1.73** |
| `anime-v1.1-from-0.3.0` | **0.564** | 0.639 | 2.65 |

`corr` is agreement with the human script; `travel` is how much of the stroke's
range the script actually uses; `sl>3x` counts spikes three times too fast
during slow passages -- an artifact, where lower is better.

Note what the baseline is: `v0.6.0`, which is the checkpoint these two were
compared against during training. **The default `v0.5.1` was never run on this
holdout**, so the table does not say how these two compare to what you get out of
the box -- only how they compare to that one upstream arm.

Three things to take from that table. First, **`anime-v1` is the quiet one**: it
trades amplitude for calm -- travel down to 0.523, spikes down to a third of
upstream's. Second, **`anime-v1.1` wins on the averages** (best `corr`, travel
close to upstream) **but buys that with more spikes than `anime-v1`** -- 2.65
against 1.73, still better than upstream's 5.30 but no longer the quiet model.
It ships with `amp_cap_x=2.0`, an inference-time amplitude cap swept after
training: at the untuned default the same checkpoint reads 3.33, and with the
cap off it reads 4.19. Third, **none of these numbers was measured on anything
but anime**, so they say nothing about live action, 3D renders or whatever else
you point the app at. They are an argument for trying these models on anime, not
for believing they are better in general.

This is why upstream's `v0.5.1` remains the default and why `anime-v1.1` is not
called a finished model: on its own holdout it improves every average while
making the artifact floor worse, and by the project's own evaluation rules an
average bought with a worse artifact floor is a worse model. It is here because
the training was done and the evidence is real, not because it is recommended.

Use `--model anime-v1` (or pick it in the review page) if your footage is anime
or MMD. If it is not, leave the default alone.

### Smaller changes

* The **#strip** has two transports: *play*, where it follows the video and a
  click jumps, and *edit*, where the wheel slides, Ctrl+wheel zooms, dragging a
  boundary moves a cut and right-click picks the model (**E** switches; **Tab**
  does the same from anywhere).
* Both flank columns are **resizable**: drag the edge of the piece list or of
  the workbench. The two share one width -- they sit either side of the same
  stage, so widening the list to read the models in it widens the knobs to
  match -- and the width is remembered in `settings.json`, so it survives a
  reload, a cleared cache and a different browser.
* The **crop page** can add, move and remove the starts of its shots -- a long
  shot whose subject walks out of the rect is the case a detector cannot call
  and a person can (Enter at the playhead, Delete, or drag on the bar). Every
  edit is clamped to the frame grid and strictly between its neighbours, so the
  page cannot make a plan the segmented decoder would refuse. Boundary 0 (the
  clip's own start) can be added to, never moved or removed.
* The **speed readout** under the strip shows the rate at the playhead and the
  distance covered over the trailing second, painted on the same green-to-amber
  scale as the heat bar and going plain red once either passes the cap. Where
  the cap is *cutting* is washed red on the strip, which is the bar that colours
  itself by speed -- on the whole-clip timeline a red band would read as a
  property of the video, which is not what it means.
* The canvases repaint at 60 Hz while the video plays and **not at all** when it
  does not: a paused page costs nothing, and a gesture paints immediately
  instead of waiting for the next tick.
* Default playback volume is 15%, deliberately not remembered -- this is a
  visual editor, and a quiet default is friendlier than a loud one.
* The picker's option row carries every standing preference, so a double-clicked
  exe -- whose user has no command line -- can reach all of it: **P** throttle,
  **O** conf-suggest on open, **W** GPU browser, **H** raise the window (**J**
  batch crop). Each is remembered in `settings.json` and applied to the next
  batch.
* Closing a page is **cancelling**, and it stops the whole batch rather than
  leaving a process waiting on a window that no longer exists: a page says
  goodbye as it is torn down, and a run that has gone quiet far too long is
  treated as gone too. A goodbye is not acted on immediately, because a reload
  says goodbye exactly as a close does -- it is held for a moment and retracted
  by the page coming back, so **F5 cannot kill a batch**. Working files are kept,
  so re-running a video resumes it.
* The **version has four parts**, and the fourth is a **commit count**, not a
  build id: the last number is how many commits this fork has made since its own
  baseline tag. Rebuilding one commit ten times says the same thing ten times --
  which is the point, since a number that moves without the code moving tells you
  nothing. Every surface reads that one string (`--version`, the boot banner, the
  picker's wordmark, the error log, the `--serve` handshake), so they cannot
  disagree. A build with no repository -- a source tarball, no baseline tag --
  falls back to `.0` with a build warning rather than failing.
* Every new string ships in both English and Chinese (147 keys added on top of
  the fork's base, counted from `languages/en-US.json`).

## What you need

* **Windows 10 (64-bit) or 11**, or **Linux** on x86-64 with glibc 2.38 or
  newer (Ubuntu 24.04, Fedora 39, Debian 13, Arch, or the like).
* A graphics card with **8 GB of VRAM**: any **DirectX 12** card on Windows
  (NVIDIA, AMD, Intel), or an **NVIDIA** card on Linux with its display
  driver, version 525 or newer. There is no CPU-only mode: the goblins think
  in a form only a GPU can hold.
* The package for your card. `windows-dml` runs on every DirectX 12 card and
  is the small download. `windows-cuda` and `linux-cuda` want an NVIDIA card,
  and they carry NVIDIA's CUDA 12 and cuDNN 9 libraries with them, so CUDA is
  nothing you install.
* **8 GB of system memory**, or more.
* [ffmpeg](https://ffmpeg.org/) on your PATH -- `winget install --id Gyan.FFmpeg`
  and then a new terminal on Windows, `sudo apt install ffmpeg` on Linux.
* Keep every file from the zip in one folder with the goblinscript binary.
  The libraries are found there, and nowhere else.

Roughly half the video's running time on a fast GPU. Everything runs on your
machine: no uploads, no internet needed.

Funscripts drive adult toys, and adult video is what the goblins were taught
on. Nothing here is for anyone under 18.

## Building it

```
cargo build --release     # builds; no goblins in it yet
cargo test                # green on a bare checkout; see the runtime note below
```

The model is not in this repository, but a **released** `goblinscript.exe`
hands its own over. Ask it for one, and point your build at what it writes:

```
goblinscript --dump-bundle bundle          # from a release exe, into ./bundle
cargo run --release -- --bundle bundle VIDEO
```

Those are the same bytes the release drafts with, so your build now drafts
identically. To bake them in instead -- your own standalone exe, or a whole
release zip beside `LICENSE` and the third-party notices:

```
cargo build --release --features embed     # reads ./bundle at compile time
cargo xtask dist                           # -> dist/goblinscript-VER-windows-dml.zip on Windows,
                                           #    dist/goblinscript-VER-linux-cuda.zip on Linux
```

A Windows build fetches its ONNX Runtime through cargo and needs nothing set.
A Linux build links Microsoft's `onnxruntime-linux-x64-gpu_cuda12` package,
the one that reaches every NVIDIA card: unpack it, point `ORT_LIB_LOCATION` at
its `lib` folder and set `ORT_PREFER_DYNAMIC_LINK=1`. It also wants
`libasound2-dev` and `pkg-config` from the distribution. The crate tree is
cargo's to fetch on both.

A CUDA zip carries NVIDIA's CUDA 12 and cuDNN 9 libraries beside the binary,
which the pack reads from `cuda-libs/` (or `CUDA_LIBS_DIR`) and never
downloads for you. That is every Linux zip, and the Windows zip that
`cargo xtask dist --cuda` adds. `cargo xtask cuda-libs` prints the exact
names to put there, out of NVIDIA's redistributable archives at
<https://developer.download.nvidia.com/compute/>.

`music/` is empty in a checkout and the app runs silent; any General MIDI
`.mid` files dropped in there become the playlist.

The release zips are built by GitHub Actions from a version tag. The inputs a
checkout lacks sit on the permanent `bundle` release of this repository:
`bundle.zip`, the perception and every model both platforms run, and
`music.zip`. `bundle.sha256` pins the bundle a version embeds, and the
workflow refuses any other.

`DEVELOPMENT.md` is the map: what each stage does, why it is shaped that way,
and what is measured to keep it honest.

## Credits

**GoblinScript is [ofs69](https://github.com/ofs69)'s project.** The design, the
perception, the styling engine, the review and crop pages, the upstream model
bundles, the `--serve` sidecar, the build and release machinery and the
[GoblinTrainingGrounds](https://github.com/ofs69/GoblinTrainingGrounds) training
pipeline that makes the packs are all ofs69's work. This fork adds what is
listed above and nothing more; the upstream repository is the canonical project
and its releases are the ones most people want. Please star and follow the
original.

The one exception is the two `anime-*` models: those were trained here, with
ofs69's own training pipeline, from an upstream checkpoint. Every other model in
the bundle is upstream's.

The [Sakuya fork](https://github.com/Kotone-Hikaru/GoblinScript-Sakuya) was made
by [Kotone-Hikaru](https://github.com/Kotone-Hikaru), mostly as an experiment in
letting a coding agent (DeepSeek Harness) work on a real, non-trivial Rust
codebase end to end: reading the tree, proposing changes, writing tests, and
building. The spanning bench, the wave engine and the throttle were built that
way, reviewed and adjusted by hand along the way.

License: MIT, as upstream -- see [`LICENSE`](LICENSE), which carries ofs69's
original copyright. Third-party components are attributed in
[`THIRD-PARTY-NOTICES.txt`](THIRD-PARTY-NOTICES.txt).

## License

MIT -- see `LICENSE`. Third-party components (ONNX Runtime, DirectML, the model
weights, the Rust crate tree) are attributed in `THIRD-PARTY-NOTICES.txt`.
