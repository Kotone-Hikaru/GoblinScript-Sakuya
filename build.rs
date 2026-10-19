//! Two build-time jobs: the Windows resources (app icon, manifest), and the
//! soundtrack manifest.
//!
//! The Windows resources are deliberately NON-FATAL: embedding needs a resource
//! compiler (the Windows SDK's `rc.exe`, or `llvm-rc`), and a machine without
//! one should still be able to build -- it just gets a binary with the default
//! icon and no manifest. A failure there is a warning, never a broken build.
//!
//! The soundtrack is enumerated rather than listed: every `.mid` in `music/`
//! becomes an entry in a generated `include_bytes!` table, so adding a track is
//! dropping a file in a folder and nothing else. `rerun-if-changed` on the
//! directory means cargo notices the new file on the next build.
use std::fmt::Write as _;

fn main() {
    #[cfg(windows)]
    {
        let ico = "assets/goblin.ico";
        println!("cargo:rerun-if-changed={ico}");
        let mut res = winresource::WindowsResource::new();
        res.set_icon(ico);
        res.set_manifest(MANIFEST);
        if let Err(e) = res.compile() {
            println!("cargo:warning=goblin resources not embedded ({e})");
        }
    }
    version();
    soundtrack();
    packs();
}

/// The four-part version this binary reports: `<cargo version>.<commit count>`.
///
/// Cargo's `version` accepts exactly three dot-separated numbers, so the fourth
/// part cannot live there. It is baked in here instead, as `GS_VERSION`, and
/// every place that shows a version reads that: `--version`, the BIOS banner,
/// the picker's wordmark, the marquee, the error log's header and the serve
/// handshake all agree because they all read one string.
///
/// The fourth part counts commits since the `spans-base` tag -- the fork's own
/// baseline -- so it measures THIS work, not upstream's history, and it does not
/// move when upstream is fetched. It is a count of commits, not of builds: three
/// rebuilds of one commit all say the same thing, which is the point.
///
/// Delay is not a failure: no git, no tag, or a source tarball with no `.git`
/// all fall back to `0` with a build warning. A checkout that cannot answer the
/// question still has to produce a binary.
fn version() {
    // Re-run when the commit history moves. `.git/HEAD` covers checkout/switch;
    // the branch ref covers a new commit; `logs/HEAD` covers amend and rebase,
    // which rewrite history without changing the count in an obvious file.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/logs/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(refname) = head.strip_prefix("ref: ").map(str::trim) {
            println!("cargo:rerun-if-changed=.git/{refname}");
        }
    }

    let base = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
    // An explicit override, for a build with no repository at all.
    let n = match std::env::var("GS_COMMITS") {
        Ok(v) => v.parse::<u32>().ok(),
        Err(_) => count_commits(),
    };
    let (n, part) = match n {
        Some(n) => (n, n.to_string()),
        None => {
            println!(
                "cargo:warning=commit count unavailable (no git, no `spans-base` tag, or no \
                 `.git`) -- the version will read {base}.0"
            );
            (0, "0".to_string())
        }
    };
    let _ = n;
    println!("cargo:rustc-env=GS_VERSION={base}.{part}");
}

/// Commits since `spans-base`, or `None` when git cannot answer.
///
/// `spans-base` rather than `origin/main`: the upstream ref is moved by any
/// `git fetch`, so a count taken from it can go DOWN after a fetch -- a version
/// number that runs backwards is worse than none. The tag is the fork's own
/// fixed anchor.
fn count_commits() -> Option<u32> {
    let out = std::process::Command::new("git")
        // Never let a credential prompt or a pager hang a build.
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .args(["rev-list", "--count", "spans-base..HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// Write `$OUT_DIR/packs.rs`: a `PACKS` table of (name, head, mask, env_step)
/// for every `bundle/packs/<name>/` that holds a head graph, sorted by name so
/// the build is reproducible. A pack is a model on the bundle's one
/// perception, and the embedded build carries every pack the manifest names;
/// enumerated, not listed, so adding a model is one `export_bundle --into`.
fn packs() {
    println!("cargo:rerun-if-changed=bundle/packs");
    println!("cargo:rerun-if-changed=bundle/manifest.json");
    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir("bundle/packs")
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.join("head.onnx").is_file())
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    let mut out = String::from(
        "// Generated by build.rs from bundle/packs/ -- do not edit.\n\
         pub static PACKS: &[(&str, &[u8], Option<&[u8]>, Option<&[u8]>)] = &[\n",
    );
    for p in &dirs {
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        let abs = abs.to_string_lossy().replace('\\', "/");
        let abs = abs.strip_prefix("//?/").unwrap_or(&abs).to_string();
        let opt = |f: &str| {
            if p.join(f).is_file() {
                format!("Some(include_bytes!({:?}))", format!("{abs}/{f}"))
            } else {
                "None".to_string()
            }
        };
        let _ = writeln!(
            out,
            "    ({name:?}, include_bytes!({:?}), {}, {}),",
            format!("{abs}/head.onnx"),
            opt("mask.onnx"),
            opt("env_step.onnx")
        );
    }
    out.push_str("];\n");
    let dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    std::fs::write(std::path::Path::new(&dir).join("packs.rs"), out)
        .expect("write the pack table");
}

/// What the exe tells Windows about itself.
///
/// `longPathAware` is the reason this exists. Without it every path the process
/// opens is cut off at 260 characters, and a video library is exactly the kind
/// of tree that runs past that -- a long series name inside a long folder name
/// on a mapped drive, and the draft fails on a file the user can see in
/// Explorer. (The machine also has to have long paths enabled, which Windows 10
/// 1607 and later do by policy; declaring it here is our half of that.)
///
/// The supported-OS list comes with it rather than as an extra: an executable
/// that carries a manifest and names no Windows version is assumed to be a
/// Vista-era program, and Windows starts lying to it about the version it is
/// running on and scaling its window itself. The four GUIDs are Windows 7, 8,
/// 8.1 and 10/11, which is every Windows this ships to.
#[cfg(windows)]
const MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"/>
      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"/>
      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"/>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
    </windowsSettings>
  </application>
</assembly>
"#;

/// Write `$OUT_DIR/soundtrack.rs`: a `SCORES` table of (name, bytes) for every
/// `.mid` in `music/`, sorted by filename so the build is reproducible and the
/// playlist order is the one visible in the folder.
fn soundtrack() {
    println!("cargo:rerun-if-changed=music");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir("music")
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("mid"))
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();

    let mut out = String::from(
        "// Generated by build.rs from the `music/` directory -- do not edit.\n\
         pub static SCORES: &[(&str, &[u8])] = &[\n",
    );
    for p in &files {
        // the display name is the stem, so the console can say what is playing
        let stem = p.file_stem().unwrap_or_default().to_string_lossy();
        let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        // forward slashes: a Windows path with backslashes is escape soup
        // inside a Rust string literal
        let abs = abs.to_string_lossy().replace('\\', "/");
        let abs = abs.strip_prefix("//?/").unwrap_or(&abs);
        let _ = writeln!(out, "    ({stem:?}, include_bytes!({abs:?})),");
    }
    out.push_str("];\n");

    // `has_music` is what lets the emptiness guard in `sound.rs` be a hard
    // assertion where there IS a soundtrack, without turning a source checkout
    // that has no `music/` into a failing test run. The soundtrack is not part
    // of the source distribution; a build with tracks in the folder still gets
    // the guard that a silent playlist never ships.
    println!("cargo::rustc-check-cfg=cfg(has_music)");
    if files.is_empty() {
        println!("cargo:warning=no .mid files in music/ -- the app will run silent");
    } else {
        println!("cargo::rustc-cfg=has_music");
    }
    let dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    std::fs::write(std::path::Path::new(&dir).join("soundtrack.rs"), out)
        .expect("write the soundtrack manifest");
}
