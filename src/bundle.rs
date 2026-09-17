//! The ONNX bundle: the graphs, the manifest, and the GPU they run on.
//!
//! Everything the model IS lives here. `export_bundle.py` produces the bundle
//! from a champion checkpoint; a release build bakes it into the exe
//! (`--features embed`), a dev build reads it from a directory. Either way the
//! rest of the program sees the same thing: a `Manifest` of frozen constants
//! and a handful of `Session`s.

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Whether this run is pinned to the CPU (`--cpu`). Process-global because the
/// backend is a property of the RUN, while the choice is made deep inside a
/// per-graph call with no argument path back to `main` -- and every session,
/// wherever it is built, has to make the same one.
static FORCE_CPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Pin every graph to the CPU. Set from the command line before the first
/// session is built; a later change would leave a half-CPU run behind it.
pub fn set_force_cpu(on: bool) {
    FORCE_CPU.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Is this run pinned to the CPU?
pub fn force_cpu() -> bool {
    FORCE_CPU.load(std::sync::atomic::Ordering::Relaxed)
}

/// Which attention kernel ONNX Runtime is allowed to prefer on its own.
const CUDNN_SDPA_ENV: &str = "ORT_ENABLE_CUDNN_FLASH_ATTENTION";

/// Put every card on ONE attention kernel: ONNX Runtime's own flash attention.
///
/// The encoder is twelve `MultiHeadAttention` nodes over a 9216-token window,
/// and the runtime chooses their kernel per card. Left to itself it prefers
/// cuDNN's fused SDPA from compute capability 9.0 upward, and demotes flash
/// attention to reach it. cuDNN then builds a graph per node, and on an RTX 50
/// series card -- sm_120, and 120 counts as "9.0 and above" -- that build
/// fails: "Failed to build cuDNN graph for Flash Attention". Every draft on a
/// Blackwell card ends at the first window of the first shot.
///
/// `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0` turns the preference off, and the
/// runtime's own flash attention runs instead. That kernel is compiled for
/// every architecture the CUDA package reaches, sm_60 through sm_120a, and it
/// is the kernel an Ada card already runs -- so this is one kernel across the
/// whole range rather than one per GPU generation, which is also what makes a
/// draft the same draft on every card. The DirectML package carries no CUDA
/// provider and never reads the variable.
///
/// A variable already set is left alone: a machine can still be asked for
/// cuDNN's kernel by hand.
///
/// Called before the first session and before the first thread, because
/// the runtime reads this once, when it builds the provider.
pub fn pin_attention_kernel() {
    if std::env::var_os(CUDNN_SDPA_ENV).is_none() {
        std::env::set_var(CUDNN_SDPA_ENV, "0");
    }
}

/// Frozen perception + deploy config, written by `export_bundle.py`. Nothing
/// in here is a Rust default: a value that disagrees with the checkpoint is a
/// silently wrong draft, so every one of them is read, never assumed. The
/// manifest JSON carries more keys than this struct (values baked into the
/// graphs, e.g. `tcn_ch`, `pos_temp`); serde skips what Rust has no use for.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub bundle_version: u32,
    /// The pack this manifest was resolved from, and its one-line label. A
    /// format-5 bundle carries one perception and several packs, and
    /// `BundleFile::resolve` writes the chosen pack's name here.
    #[serde(default)]
    pub pack: String,
    #[serde(default)]
    pub label: String,
    pub checkpoint: String,
    pub epoch: i64,
    pub basis_id: String,
    pub encoder: String,
    /// Which `MultiHeadAttention` input layout the encoder was fused with.
    /// `packed` is 2.4x faster on DirectML and has no CPU kernel at all, so
    /// `--cpu` is refused up front rather than dying on an opaque kernel error
    /// once the encode stage is already running. Bundles exported before the
    /// field existed are the separate layout, which is what the default says.
    #[serde(default = "attn_separate")]
    pub attn: String,
    pub graphs: Graphs,
    pub transcode: Transcode,
    pub grid_fps: f64,
    /// Latent ROW rate (rows/s). Distinct from `grid_fps`, the DECODE frame
    /// rate: with tubelet stride k and n alignment passes the encoder emits
    /// `grid_fps * n / (2k)` rows/s, and every row->ms conversion, styling
    /// window and progress budget rides THIS clock, not the decode one.
    /// Bundles from before the field (all 15 rows/s: stride 1, alignments 2)
    /// deserialize to 0 and `row_hz()` falls back to that formula.
    #[serde(default)]
    pub row_hz: f64,
    pub enc_res: usize,
    pub grid: usize,
    pub dim: usize,
    pub clip_len: usize,
    pub tubelet_stride: usize,
    pub alignments: usize,
    pub int8_scale: f32,
    pub v_std: f64,
    pub chunk: usize,
    pub min_chunk: usize,
    /// Rows of real context forwarded on both sides of a chunk and cut from
    /// the output, so kept rows carry a warm receptive field and a warm
    /// envelope AR history across chunk boundaries. Pre-ctx bundles have no
    /// field, which deserializes to 0 = the bare tiling they were tested
    /// under.
    #[serde(default)]
    pub ctx: usize,
    /// Length of the envelope graph's recurrent state: every member's
    /// context buffer end to end. The head stage carries it opaque and
    /// reseeds it with zeros at each chunk.
    pub env_state: usize,
    /// The envelope head SAMPLES rather than publishing an expectation, so
    /// `env_step` takes a base draw per row. A categorical bundle has the
    /// field false (and a pre-flow one has no field at all, which
    /// deserializes to the same).
    #[serde(default)]
    pub env_flow: bool,
    /// Authorship seed the base draw is keyed by, beside the absolute row.
    #[serde(default)]
    pub env_seed: u64,
    /// Support of the base draw, `[0, env_base_hi)`.
    #[serde(default)]
    pub env_base_hi: f64,
    pub heads: Heads,
    pub still_eps: f64,
    pub ext_snap: f64,
    /// Composed styling's amplitude bound: a stroke's written excursion is
    /// at most this multiple of the marginal's own travel over the segment,
    /// so the written speed stays under that multiple of the speed the
    /// model reads there. 0 (absent = the serde default) = off.
    #[serde(default)]
    pub amp_cap_x: f64,
    /// Hz: the bound loosens with the segment's own frequency above this on
    /// the hedge curve `(f / f0) ^ env_gain_p`, flat below. 0 = flat.
    #[serde(default)]
    pub amp_cap_f0: f64,
    /// Exponent of the envelope's measured frequency hedge, which the bound
    /// loosens on. jepa_train's `ENV_GAIN_P`; a bundle without the field
    /// carries no bound, so the value is never reached.
    #[serde(default = "default_env_gain_p")]
    pub env_gain_p: f64,
    pub plat_thr: f64,
    pub plat_lo: f64,
    /// Dwell-call peak filter; 0 disables it (and pre-filter bundles have no
    /// field, which deserializes to the same off).
    #[serde(default)]
    pub plat_peak: f64,
    /// Dwell-call stroke veto (pos-units/s): calls whose ~2 s flanks both
    /// read below this mean predicted |vmarg| are dropped -- no adjoining
    /// stroke, no plateau. 0 disables it (pre-veto bundles have no field,
    /// which deserializes to the same off).
    #[serde(default)]
    pub plat_veto: f64,
    /// Per-row rail target inside a dwell lock (follows slow plateau
    /// drift) instead of one constant level. Absent (pre-variant bundles)
    /// deserializes to false = the constant-level lock.
    #[serde(default)]
    pub plat_rail_track: bool,
    /// Peak-confidence lock scaling [p0, p1]: each dwell's correction is
    /// scaled by clamp((peak - p0) / (p1 - p0), 0, 1). [0, 0] (absent =
    /// the serde default) = full strength, the pre-variant behavior.
    #[serde(default)]
    pub plat_soft: [f64; 2],
    /// Cap on the dwell lock's per-row mean correction, position units:
    /// a confident call shifting too far is half the gross-error rate.
    /// 0 (absent = the serde default, pre-cap bundles) = uncapped.
    #[serde(default)]
    pub plat_shift_cap: f64,
    /// Reversal-event crossing snap radius, SECONDS: composed styling moves
    /// each vmarg crossing to the rev head's direction-aware local argmax
    /// within this radius. 0 disables it.
    pub rev_snap_s: f64,
    /// Sub-frame reversal times for the written actions: "rev" = 3-point
    /// parabola on the event head around the snapped apex (the champion
    /// decode). Empty (pre-subframe bundles) = actions stay on the grid.
    #[serde(default)]
    pub subframe: String,
    /// Reversal SEGMENTATION source: "viterbi" = the alternating event
    /// decode over the rev head, which is what fast capture rides on;
    /// anything else (incl. the empty default of pre-graft bundles) = the
    /// marginal's zero crossings. The carrier is identical either way.
    #[serde(default)]
    pub rev_source: String,
    /// Pre-derivative crossing smoother, SECONDS. 0.1 is the operating
    /// point; 0.2 collapses fast capture.
    pub rev_smooth_s: f64,
    /// Event refractory, SECONDS. 0 = unconstrained.
    pub rev_gap_s: f64,
    /// Fit the emission prior separately on STILL and MOVING rows, so an
    /// unscripted gap's posterior mass cannot inflate the prior that governs
    /// moving rows. Defaults false, which is the single global fit that
    /// pre-gap-prior bundles carry.
    #[serde(default)]
    pub rev_gap_prior: bool,
    /// Box width, SECONDS, for the |vmarg| the gap prior reads its still /
    /// moving membership off. Defaults to jepa_infer's `SPEED_REF_S`.
    #[serde(default = "default_speed_ref_s")]
    pub speed_ref_s: f64,
    /// Per-row refractory off the local-period head: this multiple of the
    /// predicted period, floored at `rev_gap_s` and capped at
    /// `period_cap_s`. 0 (absent = the serde default) = the one-number
    /// refractory.
    #[serde(default)]
    pub rev_gap_k: f64,
    /// Cap on the per-row refractory, SECONDS: a period longer than this is
    /// a hold, and a hold holds the decode rather than freezing it.
    #[serde(default = "default_period_cap_s")]
    pub period_cap_s: f64,
    /// Emission-prior fit window, SECONDS (a cap: normally the whole clip).
    pub bias_fit_s: f64,
    pub transnet: Transnet,
}

/// jepa_infer's `SPEED_REF_S`. Only reached by a bundle exported before the
/// field existed; every current export writes it.
fn default_speed_ref_s() -> f64 {
    0.6
}

/// jepa_train's `ENV_GAIN_P`, for a bundle exported before the field.
fn default_env_gain_p() -> f64 {
    0.6
}

/// jepa_infer's `PERIOD_CAP_S`, for a bundle exported before the field.
fn default_period_cap_s() -> f64 {
    2.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Graphs {
    pub encoder: String,
    pub transnet: String,
    pub head: String,
    pub env_step: Option<String>,
    /// The mask net on its own, for the viewport. Absent in bundles exported
    /// before it, which draw latent band energy instead.
    pub mask: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Transcode {
    pub height: u32,
    pub fps: f64,
    pub crf: u32,
    pub preset: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Heads {
    pub pos: bool,
    pub env: bool,
    pub ext: bool,
    pub plat: bool,
    /// Reversal-event head (pre-rev bundles have no field -> false).
    #[serde(default)]
    pub rev: bool,
    /// Local-period head: the passage's fastest half-stroke period in
    /// seconds, per row (absent -> false).
    #[serde(default)]
    pub period: bool,
}

fn attn_separate() -> String {
    "separate".to_string()
}

/// The manifest format this build understands. A bump means field semantics
/// changed, and an old binary must refuse the bundle rather than misread it.
/// Version 3: every duration in the manifest is SECONDS -- `rev_snap_s` and
/// `rev_gap_s` replace the row-valued fields, and all three are required, so
/// a bundle that omits one is refused instead of silently defaulted.
/// Version 4: the envelope graph owns its recurrence (`state` in,
/// `state_next` out, `env_state` long) and the head graph may vote several
/// member trunks (`members`).
/// Version 5: one perception, several PACKS. The file is a `BundleFile`, the
/// perception at its top level and each model under `packs/<name>/`, and a
/// flat `Manifest` exists only as the resolve of one pack.
const BUNDLE_VERSION: u32 = 5;

impl Manifest {
    /// A manifest duration as ROWS on this bundle's grid -- the mirror of
    /// `common.rows_at`, and the ONE place seconds become rows.
    pub fn rows_at(&self, seconds: f64) -> usize {
        (seconds * self.row_hz()).round().max(1.0) as usize
    }

    /// The emission-prior fit window as ROWS on this bundle's grid.
    pub fn bias_fit_rows(&self) -> usize {
        self.rows_at(self.bias_fit_s)
    }

    /// The latent row rate: the stamped field, or (pre-field bundles) the
    /// grid formula over the stamped decode config.
    pub fn row_hz(&self) -> f64 {
        if self.row_hz > 0.0 {
            self.row_hz
        } else {
            self.grid_fps * self.alignments as f64
                / (2.0 * self.tubelet_stride as f64)
        }
    }

    /// Video milliseconds of latent row `i` -- THE row clock, and the one
    /// definition of it.
    ///
    /// A row is not an instant: it is the tubelet PAIR it was encoded from,
    /// frames `a` and `a + tubelet_stride`, so it sits at that pair's
    /// midpoint. That is what the training caches store as `times_ms` and
    /// therefore the clock every scorer, every lag fit and every target the
    /// heads were trained against already speak. Half a frame of it is not a
    /// rounding detail -- the whole draft is written on this clock, and an
    /// offset here moves every action in it.
    pub fn row_ms(&self, i: usize) -> f64 {
        let k = self.tubelet_stride as f64;
        // rows are 1:1 with frames when the alignments tile the window
        // (`alignments == 2k`); a coarser alignment count decimates them
        let frame = i as f64 * 2.0 * k / self.alignments as f64;
        (frame + k / 2.0) / self.grid_fps * 1000.0
    }


    /// Loud load-time refusal of a bundle this build cannot honor: a format
    /// version it does not understand, or a missing head the styler reads
    /// unconditionally (composed styling needs pos + env + ext; only the
    /// dwell head is optional at decode).
    fn validate(self) -> Result<Self> {
        anyhow::ensure!(
            self.bundle_version == BUNDLE_VERSION,
            "bundle format v{} (this build understands v{BUNDLE_VERSION})",
            self.bundle_version
        );
        anyhow::ensure!(
            self.heads.pos && self.heads.env && self.heads.ext,
            "bundle is missing heads the styler requires \
             (pos {}, env {}, ext {})",
            self.heads.pos,
            self.heads.env,
            self.heads.ext
        );
        // The preset ladders are compile-time constants standing beside these
        // runtime numbers, and this is the first and only point at which the
        // two meet. A bundle whose tuning has flattened a rung is a bundle
        // whose menus lie, so it does not load.
        if let Err(e) =
            crate::style::presets_ordered(self.still_eps, (self.plat_soft[0], self.plat_soft[1]))
        {
            anyhow::bail!("{e}");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct Transnet {
    /// TransNetV2's fixed input frame size.
    pub input_h: usize,
    pub input_w: usize,
    /// Sliding-window inference: `window` frames per forward, advanced by
    /// `step`; the middle `step` rows of each window's output are kept.
    pub window: usize,
    pub step: usize,
    /// A per-frame transition probability above `thr` marks a transition;
    /// consecutive such frames are one cut, no closer than `min_gap_s`.
    pub thr: f64,
    pub min_gap_s: f64,
}

/// The on-disk bundle, format 5: ONE perception -- the encoder and the
/// shot-cut detector, their graphs and the grid they share -- and a map of
/// PACKS, each a model: its head, mask and envelope graphs and every model
/// field the flat `Manifest` reads, plus a one-line label; and the name of
/// the default pack. A pack is named for the release that shipped its model,
/// because that is what a person compares by. No stage reads this struct:
/// `resolve` folds the perception and one pack into the flat `Manifest`
/// every stage consumes, so `man.<field>` means what it always meant.
#[derive(Debug, Clone, Deserialize)]
pub struct BundleFile {
    pub bundle_version: u32,
    pub packs: BTreeMap<String, serde_json::Value>,
    pub default: String,
    /// The perception: every top-level key that is not the packs map or the
    /// default, kept as JSON so a resolve merges by name.
    #[serde(flatten)]
    pub perception: serde_json::Map<String, serde_json::Value>,
}

impl BundleFile {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let f: Self = serde_json::from_slice(bytes)
            .context("manifest.json is not a bundle this build understands")?;
        anyhow::ensure!(
            f.bundle_version == BUNDLE_VERSION,
            "bundle format v{} (this build understands v{BUNDLE_VERSION})",
            f.bundle_version
        );
        anyhow::ensure!(!f.packs.is_empty(), "bundle carries no packs");
        anyhow::ensure!(
            f.packs.contains_key(&f.default),
            "bundle names default pack {:?}, which it does not carry (packs: {})",
            f.default,
            f.names().join(", ")
        );
        Ok(f)
    }

    /// The pack names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.packs.keys().cloned().collect()
    }

    /// The flat manifest for one pack: the perception's fields under the
    /// pack's, the pack's graph names under `packs/<name>/`, the
    /// perception's graph names as they are, the pack's name in `pack`.
    pub fn resolve(&self, name: &str) -> Result<Manifest> {
        use serde_json::{json, Value};
        let pack = self
            .packs
            .get(name)
            .with_context(|| {
                format!("no pack {name:?} in this bundle (packs: {})", self.names().join(", "))
            })?
            .as_object()
            .with_context(|| format!("pack {name:?} is not an object"))?;
        let mut m = self.perception.clone();
        m.insert("bundle_version".into(), json!(self.bundle_version));
        let mut graphs = m
            .get("graphs")
            .and_then(|g| g.as_object().cloned())
            .unwrap_or_default();
        for (k, v) in pack {
            if k == "graphs" {
                if let Some(pg) = v.as_object() {
                    for (gk, gv) in pg {
                        if let Some(s) = gv.as_str() {
                            graphs.insert(gk.clone(), json!(format!("packs/{name}/{s}")));
                        }
                    }
                }
            } else {
                m.insert(k.clone(), v.clone());
            }
        }
        m.insert("graphs".into(), Value::Object(graphs));
        m.insert("pack".into(), json!(name));
        serde_json::from_value::<Manifest>(Value::Object(m))
            .with_context(|| format!("pack {name:?}: manifest fields this build cannot read"))?
            .validate()
    }
}

/// What the head stage computes tracks with: the graphs' bytes, hashed once
/// at load, and the manifest fields `heads.rs` reads. Two packs with the same
/// key are one trunk -- the same frozen trunk and deploy heads -- under
/// different decode constants, and their tracks are interchangeable, so a
/// switch between them is a restyle. Every styler field stays out of it:
/// a key that moved with one would re-run the head pass for nothing.
pub fn trunk_key(graphs: &str, man: &Manifest) -> String {
    let fields = serde_json::json!({
        "v_std": man.v_std,
        "chunk": man.chunk,
        "min_chunk": man.min_chunk,
        "ctx": man.ctx,
        "env_state": man.env_state,
        "env_flow": man.env_flow,
        "env_base_hi": man.env_base_hi,
        "heads": man.heads,
        "dim": man.dim,
        "grid": man.grid,
    });
    let mut h = blake3::Hasher::new();
    h.update(graphs.as_bytes());
    h.update(fields.to_string().as_bytes());
    h.finalize().to_hex()[..16].to_string()
}

/// One pack's graph bytes, and the hash of the two the head stage runs.
struct PackBytes {
    head: Vec<u8>,
    env_step: Option<Vec<u8>>,
    mask: Option<Vec<u8>>,
    graphs: String,
}

impl PackBytes {
    fn new(head: Vec<u8>, env_step: Option<Vec<u8>>, mask: Option<Vec<u8>>) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(&(head.len() as u64).to_le_bytes());
        h.update(&head);
        if let Some(e) = &env_step {
            h.update(e);
        }
        let graphs = h.finalize().to_hex().to_string();
        PackBytes { head, env_step, mask, graphs }
    }
}

/// One model as a person chooses it: the pack's name and the trunk it runs,
/// so a menu can say which choices share their tracks.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub trunk: String,
}

/// Graph bytes + the manifest of the SELECTED pack. The perception's bytes
/// are held once and shared by every selection, and `select` yields the same
/// bundle looking at another pack. Sessions are built lazily: the boundary
/// pass and the encoder pass never need to be resident at the same time, and
/// the encoder alone is ~180 MB of weights on the GPU.
pub struct Bundle {
    pub manifest: Manifest,
    pub file: Arc<BundleFile>,
    encoder: Arc<Vec<u8>>,
    transnet: Arc<Vec<u8>>,
    packs: Arc<BTreeMap<String, PackBytes>>,
    /// The run's own authorship seed (`--env-seed`), when it named one:
    /// laid over every pack's `env_seed`, so a `select` keeps it.
    seed: Option<u64>,
}

#[cfg(feature = "embed")]
mod embedded {
    macro_rules! bundled {
        ($name:literal) => {
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/bundle/", $name))
        };
    }
    pub const MANIFEST: &[u8] = bundled!("manifest.json");
    pub const ENCODER: &[u8] = bundled!("vjepa_pca.onnx");
    pub const TRANSNET: &[u8] = bundled!("transnet.onnx");
    // `PACKS`: (name, head, mask, env_step) for every `bundle/packs/<name>/`,
    // enumerated by `build.rs` the way the soundtrack is
    include!(concat!(env!("OUT_DIR"), "/packs.rs"));
}

/// Write the bundle baked into this binary out to `dir`, byte for byte.
///
/// The graphs are a build product of the training tree and are not in the
/// repository, so a build from source has no goblins in it. A release binary
/// carries a set, and these are the same bytes `--bundle` reads: dump them
/// once and any build drafts exactly as that release does.
///
/// Named from the manifest rather than from a fixed list, so what lands in the
/// directory is what `from_dir` will look for -- every pack under its own
/// `packs/<name>/`, and no graph the manifest does not claim.
#[cfg(feature = "embed")]
pub fn dump(dir: &Path) -> Result<Vec<(String, usize)>> {
    let file = BundleFile::parse(embedded::MANIFEST)
        .context("the baked-in manifest.json is not one this build understands")?;
    std::fs::create_dir_all(dir)
        .with_context(|| format!("cannot make {}", dir.display()))?;
    let mut wrote: Vec<(String, usize)> = Vec::new();
    let mut put = |name: &str, bytes: &'static [u8]| -> Result<()> {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot make {}", parent.display()))?;
        }
        std::fs::write(&path, bytes)
            .with_context(|| format!("cannot write {}", path.display()))?;
        wrote.push((name.to_string(), bytes.len()));
        Ok(())
    };
    put("manifest.json", embedded::MANIFEST)?;
    let default = file.resolve(&file.default)?;
    put(&default.graphs.encoder, embedded::ENCODER)?;
    put(&default.graphs.transnet, embedded::TRANSNET)?;
    for &(name, head, mask, env_step) in embedded::PACKS {
        let man = file.resolve(name)?;
        put(&man.graphs.head, head)?;
        if let (Some(n), Some(b)) = (&man.graphs.mask, mask) {
            put(n, b)?;
        }
        if let (Some(n), Some(b)) = (&man.graphs.env_step, env_step) {
            put(n, b)?;
        }
    }
    Ok(wrote)
}

/// The same door, in a build that has nothing behind it.
#[cfg(not(feature = "embed"))]
pub fn dump(_dir: &Path) -> Result<Vec<(String, usize)>> {
    anyhow::bail!(
        "this build has no bundle baked into it, so there is nothing to write \
         out. --dump-bundle is for a released goblinscript.exe; a build from \
         source is already being handed a bundle directory with --bundle."
    )
}

/// What a machine without the goblins' GPU provider is missing, in the words
/// its owner needs. Said in two places -- `probe_gpu` at boot, and `session`
/// if a later session build is the first to notice -- so it is written once.
#[cfg(not(windows))]
const NO_GPU: &str = "no CUDA. GoblinScript needs an NVIDIA card and NVIDIA's \
     display driver, version 525 or newer. There is nothing else to install: \
     the CUDA 12 and cuDNN 9 libraries are in the package, and they are read \
     from the folder the binary is in, so keep them beside it. A remote \
     desktop or a virtual machine without GPU passthrough is the usual reason \
     a real card is not visible.";

#[cfg(all(windows, feature = "cuda"))]
const NO_GPU: &str = "no CUDA. This is the CUDA package, which needs an NVIDIA \
     card and NVIDIA's display driver, version 525 or newer. There is nothing \
     else to install: the CUDA 12 and cuDNN 9 libraries are in the package, \
     and they are read from the folder the exe is in, so keep them beside it. \
     On a card from another maker, take the -windows-dml package instead: it \
     runs on any DirectX 12 GPU and is a much smaller download.";

#[cfg(all(windows, not(feature = "cuda")))]
const NO_GPU: &str = "no graphics card the goblins can think on. GoblinScript \
     needs a DirectX 12 GPU -- any vendor, no runtime to install beyond the \
     display driver. A remote desktop or a virtual machine without GPU \
     passthrough is the usual reason a real card is not visible.";

impl Bundle {
    /// Load from a bundle directory (dev / `--bundle`), every pack's bytes
    /// read, the default pack selected.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            std::fs::read(dir.join(name))
                .with_context(|| format!("bundle: missing {name} in {}", dir.display()))
        };
        let file = BundleFile::parse(&read("manifest.json")?)
            .with_context(|| format!("bundle: {}", dir.display()))?;
        let default = file.resolve(&file.default)?;
        let encoder = Arc::new(read(&default.graphs.encoder)?);
        let transnet = Arc::new(read(&default.graphs.transnet)?);
        let mut packs = BTreeMap::new();
        for name in file.names() {
            let man = file.resolve(&name)?;
            packs.insert(
                name,
                PackBytes::new(
                    read(&man.graphs.head)?,
                    match &man.graphs.env_step {
                        Some(n) => Some(read(n)?),
                        None => None,
                    },
                    match &man.graphs.mask {
                        Some(n) => Some(read(n)?),
                        None => None,
                    },
                ),
            );
        }
        Ok(Self {
            manifest: default,
            file: Arc::new(file),
            encoder,
            transnet,
            packs: Arc::new(packs),
            seed: None,
        })
    }

    /// Load the bundle baked into this binary, the default pack selected.
    #[cfg(feature = "embed")]
    pub fn embedded() -> Result<Self> {
        let file = BundleFile::parse(embedded::MANIFEST)?;
        let mut packs = BTreeMap::new();
        for &(name, head, mask, env_step) in embedded::PACKS {
            anyhow::ensure!(
                file.packs.contains_key(name),
                "baked-in pack {name:?} is not in the baked-in manifest"
            );
            packs.insert(
                name.to_string(),
                PackBytes::new(
                    head.to_vec(),
                    env_step.map(|b| b.to_vec()),
                    mask.map(|b| b.to_vec()),
                ),
            );
        }
        for name in file.names() {
            anyhow::ensure!(
                packs.contains_key(&name),
                "the manifest names pack {name:?}, which this build did not bake in"
            );
        }
        Ok(Self {
            manifest: file.resolve(&file.default)?,
            file: Arc::new(file),
            encoder: Arc::new(embedded::ENCODER.to_vec()),
            transnet: Arc::new(embedded::TRANSNET.to_vec()),
            packs: Arc::new(packs),
            seed: None,
        })
    }

    /// The same bundle looking at another pack: its manifest resolved, the
    /// perception and every pack's bytes shared.
    pub fn select(&self, name: &str) -> Result<Self> {
        anyhow::ensure!(
            self.packs.contains_key(name),
            "no pack {name:?} in this bundle (packs: {})",
            self.pack_names().join(", ")
        );
        let mut manifest = self.file.resolve(name)?;
        if let Some(s) = self.seed {
            manifest.env_seed = s;
        }
        Ok(Self {
            manifest,
            file: Arc::clone(&self.file),
            encoder: Arc::clone(&self.encoder),
            transnet: Arc::clone(&self.transnet),
            packs: Arc::clone(&self.packs),
            seed: self.seed,
        })
    }

    /// The same bundle with the run's authorship seed laid over the pack's,
    /// on this pack and on every `select` after it. `None` leaves each
    /// pack's own.
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        if let Some(s) = seed {
            self.manifest.env_seed = s;
        }
        self
    }

    /// Every pack this bundle carries, sorted by name.
    pub fn pack_names(&self) -> Vec<String> {
        self.file.names()
    }

    /// The pack a draft with no choice made runs: the champion's.
    pub fn default_pack(&self) -> &str {
        &self.file.default
    }

    fn pack(&self) -> &PackBytes {
        &self.packs[&self.manifest.pack]
    }

    /// The selected pack's trunk (`trunk_key`).
    pub fn trunk(&self) -> String {
        trunk_key(&self.pack().graphs, &self.manifest)
    }

    /// Every model this bundle carries, sorted by name, with its trunk.
    pub fn models(&self) -> Result<Vec<ModelInfo>> {
        self.pack_names()
            .iter()
            .map(|n| {
                let bb = self.select(n)?;
                Ok(ModelInfo { name: n.clone(), trunk: bb.trunk() })
            })
            .collect()
    }

    pub fn encoder_session(&self) -> Result<Session> {
        session(&self.encoder, "encoder")
    }

    /// Can this machine build a GPU session at all?
    ///
    /// Answered by building one, on the SMALLEST graph the bundle carries:
    /// the question is whether the provider comes up, not what it runs, and
    /// the encoder is ~180 MB of weights to find that out with.
    ///
    /// This registers the platform's GPU provider ALONE, and with
    /// `error_on_failure`, which is the whole point. The chain `session`
    /// walks ends in the CPU provider, and a machine with no usable GPU does
    /// not fail there -- it succeeds, quietly, and then dies at the first
    /// attention node with "Packed QKV of shape (B, L, N, 3, H) not
    /// implemented for CPU". That message is true and it is not the problem.
    /// So the question is asked strictly, once, before any media is touched:
    /// a provider that cannot initialize says so here, in the time it takes
    /// to look for its libraries, instead of after a video has been probed,
    /// decoded and run through the shot-cut detector.
    pub fn probe_gpu(&self) -> Result<()> {
        if force_cpu() {
            return Ok(());
        }
        #[cfg(all(windows, not(feature = "cuda")))]
        use ort::ep::DirectML;
        #[cfg(any(feature = "cuda", not(windows)))]
        use ort::ep::CUDA;
        let pack = self.pack();
        let smallest = [pack.env_step.as_deref(), pack.mask.as_deref()]
            .into_iter()
            .flatten()
            .min_by_key(|g| g.len())
            .unwrap_or(self.transnet.as_slice());
        // The build's one provider is what gets asked, because it is also the
        // only one a draft can use.
        #[cfg(all(windows, not(feature = "cuda")))]
        let provider = DirectML::default().build().error_on_failure();
        #[cfg(any(feature = "cuda", not(windows)))]
        let provider = CUDA::default().build().error_on_failure();
        // One closure, so that NO_GPU reaches the error whichever step
        // produced it. Registering the provider is the step that usually
        // does -- a missing CUDA is a library that does not load, which
        // happens before any graph is read -- and a `?` on that line would
        // return the runtime's own words about a file it could not open.
        (|| -> Result<()> {
            Session::builder()
                .map_err(ort_err)?
                .with_execution_providers([provider])
                .map_err(ort_err)?
                .commit_from_memory(smallest)
                .map_err(ort_err)?;
            Ok(())
        })()
        .context(NO_GPU)
    }
    /// The shot-cut detector runs where the encoder does. Both providers the
    /// binary ships with have a Conv3d kernel -- DirectML and CUDA -- so the
    /// 48 layers of TransNetV2 need nothing said about them, and a machine
    /// with no GPU at all reaches the CPU through the same fallback every
    /// other graph does, which its size allows: 7.6M parameters over 27x48
    /// frames, a few seconds per clip.
    pub fn transnet_session(&self) -> Result<Session> {
        session(&self.transnet, "transnet")
    }
    /// The head runs on the CPU, and not by preference.
    ///
    /// DirectML BUILDS this graph and then crashes executing it (an access
    /// violation inside the EP, reproducible from Python with the same 5-D int8
    /// input, so it is not a binding bug on our side). The head is also the
    /// cheap half -- the encoder is ~95% of a draft's compute -- and the CPU
    /// path is the one the parity check validated to corr 1.00000000 against
    /// jepa_infer, so this costs accuracy nothing and wall clock little.
    pub fn head_session(&self) -> Result<Session> {
        cpu_session(&self.pack().head, "head")
    }
    /// The envelope's step graph runs on the CPU too, for its own reason: the
    /// decode is autoregressive -- one call per row, ~36k calls for a 40-minute
    /// video -- over three 1x1 convolutions. On a GPU that is pure dispatch
    /// overhead with no compute to amortize it.
    pub fn env_session(&self) -> Result<Option<Session>> {
        match &self.pack().env_step {
            Some(b) => Ok(Some(cpu_session(b, "env_step")?)),
            None => Ok(None),
        }
    }

    /// The mask net, for the viewport -- on the CPU, and pinned to ONE thread.
    ///
    /// It runs on the encode thread between windows, and everything around it
    /// is already spending the machine: the encoder has the GPU, and in a batch
    /// `prefetch` has libx264 on the cores. A decoration that took either would
    /// cost real wall clock. One thread is enough by a wide margin -- ~40 M
    /// multiply-adds against a window that arrives every ~120 ms -- so this is
    /// the cheap end of the trade, not a compromise.
    ///
    /// `None` when the bundle predates the graph; the viewport draws latent
    /// band energy instead and nothing else notices.
    pub fn mask_session(&self) -> Result<Option<Session>> {
        use ort::ep::CPU;
        let Some(bytes) = &self.pack().mask else { return Ok(None) };
        Ok(Some(
            Session::builder()
                .map_err(ort_err)?
                .with_execution_providers([CPU::default().build()])
                .map_err(ort_err)?
                .with_intra_threads(1)
                .map_err(ort_err)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_err)?
                .commit_from_memory(bytes)
                .map_err(ort_err)
                .context("failed to build the mask session")?,
        ))
    }
}

fn cpu_session(bytes: &[u8], what: &str) -> Result<Session> {
    use ort::ep::CPU;
    Session::builder()
        .map_err(ort_err)?
        .with_execution_providers([CPU::default().build()])
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .commit_from_memory(bytes)
        .map_err(ort_err)
        .with_context(|| format!("failed to build the {what} session"))
}

/// `ort`'s errors carry a recovery payload that is neither Send nor Sync, so
/// they do not convert into `anyhow::Error` on their own. The payload is a way
/// to get the half-built session back, which we never want: a session that
/// failed to build is a bug in the bundle, not something to recover from.
pub fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("{}", e.message())
}

/// One session on the best GPU we can reach.
///
/// A build has exactly ONE GPU provider, and which one is what separates the
/// packages. The default build is DirectML, the only backend that covers every
/// vendor's GPU on Windows, which is what makes that package distributable. A
/// `cuda` build is CUDA instead, and off Windows CUDA is the only choice there
/// is.
///
/// There is no fallback from one to the other, because the two do not live in
/// one ONNX Runtime: the CUDA packages are built by Microsoft, whose runtime
/// carries CUDA and TensorRT and no DirectML EP at all. Registering DirectML
/// there returns "Specified provider is not supported", so a `cuda` build that
/// cannot reach a card says so and stops, exactly as the Linux build does.
///
/// `error_on_failure` is what makes a machine without a usable card FAIL the
/// attempt rather than silently land on the CPU provider registered beside it.
///
/// CPU is registered last, and on a shipped bundle it can only ever serve the
/// small graphs: the encoder's packed attention has no CPU kernel at all, so a
/// GPU is not optional here. `main` refuses `--cpu` outright for that bundle.
///
/// `--cpu` skips the chain entirely, and needs an `--cpu-attn` export to work.
/// It exists because a GPU backend can MISCOMPUTE this graph rather than fail
/// it (the fp16 overflow `encode` guards against), and the CPU provider --
/// fp32, no vendor EP in the path -- is the reference a suspect draft gets
/// bisected against.
fn session(bytes: &[u8], what: &str) -> Result<Session> {
    if force_cpu() {
        return cpu_session(bytes, what);
    }
    use ort::ep::CPU;
    #[cfg(all(windows, not(feature = "cuda")))]
    use ort::ep::DirectML;
    #[cfg(any(feature = "cuda", not(windows)))]
    {
        use ort::ep::CUDA;
        let cuda = (|| -> Result<Session> {
            Session::builder()
                .map_err(ort_err)?
                .with_execution_providers([
                    CUDA::default().build().error_on_failure(),
                    CPU::default().build(),
                ])
                .map_err(ort_err)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_err)?
                .commit_from_memory(bytes)
                .map_err(ort_err)
        })();
        // Nothing is under this. The CPU provider registered beside it has no
        // kernel for the encoder's packed attention, so carrying on buys a
        // session that dies at its first node -- and it dies saying "Packed
        // QKV of shape (B, L, N, 3, H) not implemented for CPU", which tells
        // the reader nothing about the card or the driver they are missing.
        // Stop here and name it.
        return cuda.map_err(|e| e.context(NO_GPU));
    }
    #[cfg(all(windows, not(feature = "cuda")))]
    Session::builder()
        .map_err(ort_err)?
        .with_execution_providers([
            DirectML::default().build(),
            CPU::default().build(),
        ])
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .commit_from_memory(bytes)
        .map_err(ort_err)
        .with_context(|| format!("failed to build the {what} session"))
}

#[cfg(test)]
mod tests {
    use super::{trunk_key, BundleFile, Manifest};
    use std::path::PathBuf;

    fn fixture() -> BundleFile {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/manifest.json");
        BundleFile::parse(&std::fs::read(&path).unwrap()).unwrap()
    }

    /// The trunk key decides whether a model switch re-runs the head pass.
    /// A styler constant that moved it would cost a head pass per decode
    /// variant; a head-stage field that did not would hand a model another
    /// trunk's tracks.
    #[test]
    fn the_trunk_key_moves_with_the_head_stage_and_only_with_it() {
        let file = fixture();
        let man = file.resolve(&file.default).unwrap();
        let key = trunk_key("graphs", &man);
        let mut styler = man.clone();
        styler.still_eps += 1.0;
        styler.ext_snap += 10.0;
        styler.amp_cap_x += 4.0;
        styler.amp_cap_f0 += 1.0;
        styler.env_gain_p += 0.1;
        styler.plat_thr += 0.1;
        styler.plat_lo += 0.1;
        styler.plat_peak += 0.1;
        styler.plat_veto += 1.0;
        styler.plat_rail_track = !styler.plat_rail_track;
        styler.plat_soft = [0.1, 0.2];
        styler.plat_shift_cap += 1.0;
        styler.rev_snap_s += 0.1;
        styler.subframe = "none".into();
        styler.rev_source = "crossings".into();
        styler.rev_smooth_s += 0.1;
        styler.rev_gap_s += 0.1;
        styler.rev_gap_prior = !styler.rev_gap_prior;
        styler.speed_ref_s += 0.1;
        styler.rev_gap_k += 1.0;
        styler.period_cap_s += 1.0;
        styler.bias_fit_s += 1.0;
        styler.env_seed += 1;
        styler.pack = "other".into();
        styler.label = "other".into();
        assert_eq!(trunk_key("graphs", &styler), key, "a styler field moved the trunk");

        assert_ne!(trunk_key("other graphs", &man), key, "the graphs did not move the trunk");
        let moved: [fn(&mut Manifest); 10] = [
            |m| m.v_std += 1.0,
            |m| m.chunk += 1,
            |m| m.min_chunk += 1,
            |m| m.ctx += 1,
            |m| m.env_state += 1,
            |m| m.env_flow = !m.env_flow,
            |m| m.env_base_hi += 1.0,
            |m| m.heads.rev = !m.heads.rev,
            |m| m.dim += 1,
            |m| m.grid += 1,
        ];
        for (i, f) in moved.iter().enumerate() {
            let mut m = man.clone();
            f(&mut m);
            assert_ne!(trunk_key("graphs", &m), key, "head-stage field {i} did not move the trunk");
        }
    }

    /// Where row 0 sits on the shipped grid, in milliseconds. The mirror of
    /// the extractor's own arithmetic, which `grid_check.py` recomputes from
    /// the bundle manifest and compares against this line.
    const SHIPPED_ROW0_MS: f64 = 33.333333;

    /// The row clock a bundle on the shipped grid produces, against the clock
    /// the training caches were written on. Row `i` is the midpoint of the
    /// tubelet pair `(i, i + stride)`, which at the shipped 30 fps / stride 2
    /// puts row 0 at 33.333 ms and every row a frame later than the index
    /// alone would suggest. This is the one number that decides where every
    /// action in every draft is written, and there is nothing downstream that
    /// would notice it drifting.
    ///
    /// Read from the FIXTURE, which carries the shipped grid's constants: the
    /// arithmetic is the thing under test, and a bundle is a ~250 MB build
    /// product a source checkout does not have. That the bundle actually
    /// packed agrees with this is `cargo xtask dist`'s check, on the real
    /// manifest, at the moment it is about to ship.
    #[test]
    fn the_row_clock_is_the_tubelet_pairs_midpoint() {
        let file = fixture();
        let man = file.resolve(&file.default).unwrap();
        let step = 1000.0 / man.row_hz();
        for i in [0usize, 1, 17, 4096] {
            let want = man.row_ms(0) + i as f64 * step;
            assert!(
                (man.row_ms(i) - want).abs() < 1e-9,
                "row {i}: the clock is not uniform at 1/row_hz"
            );
        }
        assert!(
            (man.row_ms(0) - 1000.0 * man.tubelet_stride as f64 / 2.0 / man.grid_fps).abs() < 1e-9,
            "row 0 is not half a tubelet in"
        );
        assert!((man.row_ms(0) - SHIPPED_ROW0_MS).abs() < 1e-4,
                "the shipped grid's row 0 moved off {SHIPPED_ROW0_MS} ms (got {})",
                man.row_ms(0));
    }
}
