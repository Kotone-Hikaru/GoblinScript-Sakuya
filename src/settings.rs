//! A small JSON settings file next to the executable, so preferences survive
//! between runs. Best-effort throughout: any read/parse/write error just falls
//! back to (or keeps) the defaults -- settings are a convenience, never a
//! dependency of a draft.

use crate::style;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Field defaults come from `Default` (below), not from the derive: with
/// `#[serde(default)]` at container level, serde fills a missing field from it
/// too, so "never chosen" means the same thing whether the settings file is
/// absent, truncated, or simply predates the field.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// The directory the last batch was started from -- the picker reopens here
    /// on the next launch. `None` (or a path that no longer exists) falls back
    /// to the launch directory.
    pub last_dir: Option<String>,
    /// The style accepted at the end of the last review -- the picker seeds the
    /// next session with it, so a preferred recipe carries across launches.
    pub last_params: Option<style::Params>,
    /// The model the first script of the last review was accepted with -- the
    /// picker drafts the next batch with it. `None`, or a name the bundle no
    /// longer carries, means the bundle's default.
    pub model: Option<String>,
    /// The colour scheme, cycled with T in the picker. `None` = never chosen,
    /// which means the house phosphor green.
    pub theme: Option<crate::theme::Palette>,
    /// The language, as the tag of a file in `languages/` (cycled with G in the
    /// picker). `None` = never chosen, which means the machine's own language
    /// where a catalog for it is installed, and English otherwise -- so someone
    /// who never opens this file still gets their own language if it is there.
    pub lang: Option<String>,
    /// Sound effects off (M in the picker). Defaults to on -- a user who wants
    /// silence says so once and it sticks.
    #[serde(default)]
    pub muted: bool,
    /// Background music on. ON by default -- the goblins put a record on while
    /// they work. One press of M turns it off and that choice sticks, so the
    /// default costs an unwanted listener exactly one keystroke, once.
    ///
    /// Read by the PICKER only. A named-video run is silent unless `--music`
    /// says otherwise, so a preference set here never surprises someone who
    /// called the tool from a terminal rather than opening it.
    pub music: bool,
    /// How loud that music is (V in the picker). `None` = never chosen.
    pub volume: Option<crate::sound::Volume>,
    /// Auto-crop on (C in the picker): zoom the encoder onto the attention's
    /// region before drafting. ON by default, like the CLI -- a clip whose
    /// attention wants the whole frame is left alone anyway, so the default
    /// costs a probe and nothing else. The choice sticks when a batch is
    /// started with it. A named-video run reads the flags only.
    pub autocrop: bool,
    /// The crop check on (K in the picker): show each video's crop rects in
    /// the browser before the goblins read it, so a rect that is wrong can be
    /// dragged onto the action. ON by default -- the crop is the one decision
    /// a person makes better than the goblins in a glance, and it is free to
    /// make here. Switched off, a batch runs start to finish with nobody at
    /// the keyboard.
    pub crop_edit: bool,
    /// GPU throttle preset, cycled with P in the picker. `None` = never chosen,
    /// which means OFF -- a plain draft is unchanged, and someone who never
    /// touches this key is never throttled behind their back.
    ///
    /// The picker's users have no flags to re-run with, so this is the ONLY way
    /// to reach the throttle from a double-clicked exe; a named-video run reads
    /// `--throttle` instead, like every other remembered preference. A string
    /// (not an enum) so an unknown name from a hand-edited file, or a preset a
    /// future build drops, degrades to "off" rather than failing the parse for
    /// the whole file -- these settings are a convenience and never a
    /// dependency (see the module docs).
    pub throttle: Option<String>,
    /// Where the picker's file list starts. `None` (never chosen) means
    /// `last` -- the remembered directory, then the exe's OWN folder, then the
    /// launch directory. The two other names are:
    ///
    /// - `"last"` -- remember the last directory a batch was started from, and
    ///   reopen there. The shipped behaviour, and what `None` resolves to.
    /// - `"exe"`  -- always open in the folder the executable sits in, IGNORING
    ///   the remembered directory. For someone who double-clicks the exe from
    ///   Explorer: the process's working directory is then whatever Explorer
    ///   happened to hand it, which is why the picker used to open somewhere
    ///   unrelated. `exe` pins it to where the goblins actually live.
    ///
    /// A string, like `throttle`, so a hand-edited value this build does not
    /// know degrades to the default rather than failing the whole file.
    pub dir_mode: Option<String>,
    /// Models the reviewer took OUT of the confidence suggestion (the page's
    /// checkboxes on the drawer). `None` = never chosen, which means every model
    /// the bundle carries takes part -- a suggestion that silently ignored a
    /// model would be a lie about what was compared.
    ///
    /// Names, not indices: a bundle's model order is not a stable identity
    /// across packs, and a name this bundle no longer carries is simply not
    /// there to exclude. Empty is NOT the same as `None` -- it means the
    /// reviewer unchecked everything, which the page refuses to run with (a
    /// suggestion needs something to compare), so the two never collapse.
    pub exclude_models: Option<Vec<String>>,
    /// Run the confidence suggestion as soon as a clip is up, and keep the
    /// answer, rather than waiting to be asked for it.
    ///
    /// Off by default, and deliberately so: `false` is the only default that
    /// cannot surprise the user. With it on, opening a video costs a full
    /// sampling pass (one instant re-style per model) BEFORE the page is any
    /// use, and it rewrites the pieces without a click -- so it is a thing the
    /// reviewer turns on, never a thing they find already on.
    pub auto_suggest: bool,
    /// The port the review server answered on last time.
    ///
    /// Remembered so the SAME address serves the next run: a browser tab left
    /// open on it reloads itself into the new session (the page polls and
    /// compares `boot`) instead of stranding the user on a dead URL. `None`
    /// means never chosen -- take a free port and write it back.
    ///
    /// A hint, never a demand: a port something else has taken by the time the
    /// review starts is replaced with a fresh one rather than fought over.
    pub review_port: Option<u16>,
    /// Per-model multipliers on how loudly a model's confidence speaks in the
    /// suggestion's vote: `{"v0.5.0": 0.9}` makes that model's mean count for
    /// 90 % of what it said, so it still competes but no longer wins a shot on
    /// a self-assessment the reviewer does not trust on this material.
    ///
    /// This is the fine trim to `exclude_models`'s blunt one: excluding takes a
    /// model off the ballot entirely, and there is no middle setting there.
    /// `None` = never chosen = every model counts exactly what it said.
    ///
    /// Keyed by model NAME, never by position: a bundle's pack order is not a
    /// stable identity, and a weight that followed an index would silently land
    /// on a different model after any pack change.
    pub conf_weights: Option<std::collections::BTreeMap<String, f64>>,
    /// Launch our own browser on the review page, aimed at the discrete GPU.
    ///
    /// OFF by default. The flag only helps a machine whose browser would
    /// otherwise land on the integrated chip, and it is paid for with a
    /// separate browser profile (see `review::open_browser`) -- so it is a
    /// thing the user turns on, never a thing they find already on. A user who
    /// never touches it is never given a second browser.
    pub gpu_browser: bool,
    /// How wide the spanning bench (the left drawer, and the right workbench
    /// rail) was left, in CSS pixels.
    ///
    /// `None` = never dragged = the layout's own `clamp(272px, 23vw, 348px)`,
    /// which is the width that reads well on any screen. A number here is the
    /// page's own report of a drag, so it is honoured only within
    /// [`RAIL_MIN_W`]..=[`RAIL_MAX_W`]: the page clamps while dragging, but a
    /// hand-edited settings file must not be able to open a column wider than
    /// the window and leave no stage.
    ///
    /// ONE number for both flanks. The two columns sit on opposite sides of the
    /// same stage, so a review that widened the bench to read the piece list
    /// wants the knobs beside it just as wide -- and a single value is also
    /// what makes "drag either edge, the whole bench follows" true rather than
    /// a surprise.
    pub rail_width: Option<u16>,
    /// Raise the browser window when a stage is waiting on a person.
    ///
    /// A batch runs for minutes and then stops -- the crop page wants a
    /// decision, the review page wants a reading -- and a browser that has
    /// drifted behind something else gives no sign of it: the run looks busy
    /// while it is actually blocked. With this on, the window is lifted to the
    /// front of the Z order at those moments.
    ///
    /// It does NOT take the keyboard focus (see `crate::raise`), which is why
    /// this is safe to leave on: the window comes up, the caret stays where it
    /// was, and a person mid-sentence somewhere else is not interrupted.
    ///
    /// ON by default. Unlike the switches that spend a wait or rewrite pieces,
    /// this one only decides whether a window is visible, and a run blocked on
    /// a page nobody can see is the failure it prevents.
    pub raise_window: Option<bool>,
    /// Judge every crop at the END of the batch, instead of one per video as
    /// each is transcoded.
    ///
    /// A preference about the person's TIME rather than about how the work is
    /// done: with it on, a batch transcodes and probes every video first, asks
    /// about the crops in one sitting, then runs the expensive stages. The crop
    /// is still decided before the expensive stage, so correcting it is still
    /// free -- only WHEN the question is asked changes.
    ///
    /// OFF by default, for the same reason `auto_suggest` is: splitting the
    /// passes removes the transcode-overlap between videos, so the batch spends
    /// MORE pure machine time. That is a trade a person should choose, not one
    /// they should discover.
    pub batch_crop: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            last_dir: None,
            last_params: None,
            model: None,
            theme: None,
            lang: None,
            muted: false,
            music: true,
            volume: None,
            autocrop: true,
            crop_edit: true,
            throttle: None,
            dir_mode: None,
            exclude_models: None,
            auto_suggest: false,
            review_port: None,
            conf_weights: None,
            gpu_browser: false,
            rail_width: None,
            raise_window: None,
            batch_crop: false,
        }
    }
}

/// The narrowest a flank column may be dragged: below this the piece rows' own
/// number boxes stop fitting on one line.
pub const RAIL_MIN_W: u16 = 220;

/// The widest a flank column may be dragged. Wide enough for the fine-tuning
/// rows to sit on one line, narrow enough that a 1440-wide stage never loses
/// more than about a third of itself to one column.
pub const RAIL_MAX_W: u16 = 520;

/// `settings.json` beside the exe (where the cache lives too).
fn path() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(|d| d.join("settings.json"))
}

impl Settings {
    pub fn load() -> Self {
        path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let (Some(p), Ok(s)) = (path(), serde_json::to_string_pretty(self)) {
            let _ = std::fs::write(p, s);
        }
    }

    /// Capture the live presentation state (theme, mute, music) into the
    /// settings about to be saved. Those three live in process-wide globals
    /// because every surface reads them; this is the one place they are copied
    /// back out to disk.
    pub fn remember_presentation(&mut self) {
        self.theme = Some(crate::theme::active());
        self.lang = Some(crate::lang::active().to_string());
        self.muted = crate::sound::muted();
        self.music = crate::sound::music_on();
        self.volume = Some(crate::sound::volume());
    }

    /// Whether a waiting stage should lift the browser window to the front.
    ///
    /// `None` means never chosen, and the answer is YES: the switch only
    /// decides whether a window is visible, never where the caret goes, so its
    /// safe default is on. A settings file that says `false` is honoured.
    pub fn raises_window(&self) -> bool {
        self.raise_window.unwrap_or(true)
    }

    /// The remembered directory, if it still exists.
    pub fn start_dir(&self) -> Option<PathBuf> {
        self.last_dir
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
    }

    /// Where the picker's file list opens, resolving `dir_mode`.
    ///
    /// `"exe"` pins it to the executable's own folder and ignores the
    /// remembered directory entirely. Anything else (including the default
    /// `None`, and an unknown name from a hand-edited file) is the remembered
    /// directory first.
    ///
    /// The exe's folder is the fallback rather than the process's working
    /// directory: a double-clicked exe is handed some unrelated cwd by
    /// Explorer, and "somewhere the user did not choose" is exactly the bug
    /// this ordering avoids. The working directory is kept as the last resort,
    /// for a launch where even the exe's own path cannot be read.
    pub fn picker_start_dir(&self) -> Option<PathBuf> {
        let exe_dir = || {
            std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(PathBuf::from))
        };
        if self.dir_mode.as_deref() == Some("exe") {
            return exe_dir().or_else(|| std::env::current_dir().ok());
        }
        self.start_dir()
            .or_else(exe_dir)
            .or_else(|| std::env::current_dir().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole file is `#[serde(default)]`, and that is load-bearing: a user
    /// upgrading from a build that never heard of the throttle must keep every
    /// other field, not lose the file to a missing key.
    #[test]
    fn a_file_from_before_the_throttle_field_still_loads() {
        let old = r#"{
            "last_dir": "F://OSR6",
            "theme": "phosphor",
            "lang": "zh-CN",
            "muted": false,
            "music": true,
            "autocrop": true,
            "crop_edit": false
        }"#;
        let s: Settings = serde_json::from_str(old).expect("an older file must still parse");
        assert_eq!(s.last_dir.as_deref(), Some("F://OSR6"));
        assert_eq!(s.throttle, None, "no field means off, not a parse failure");
        assert!(s.autocrop && !s.crop_edit);
    }

    /// A hand-edited name this build does not ship must come back as a string
    /// and not fail the parse -- `describe` and the CLI both know what to do
    /// with a name they cannot resolve, and neither is worth losing the file
    /// over. (The empty file and the `null` are the other two shapes a user
    /// can leave behind.)
    #[test]
    fn an_unknown_or_nulled_throttle_is_data_not_a_failure() {
        for text in [
            r#"{"throttle": "some-future-preset"}"#,
            r#"{"throttle": null}"#,
            r#"{}"#,
        ] {
            let s: Settings = serde_json::from_str(text).expect("must parse");
            assert!(
                s.throttle.is_none() || s.throttle.as_deref() == Some("some-future-preset"),
                "unexpected value from {text}"
            );
        }
    }

    /// A round trip must preserve the throttle: it is the ONLY place the
    /// picker's P key is written down, so a field dropped on save would make
    /// the key look like it never stuck.
    #[test]
    fn a_saved_throttle_comes_back() {
        let mut s = Settings::default();
        assert_eq!(s.throttle, None, "off is the shipped state");
        s.throttle = Some("aggressive".into());
        let text = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&text).unwrap();
        assert_eq!(back.throttle.as_deref(), Some("aggressive"));
    }

    /// Which models take part in the confidence suggestion is a choice about
    /// the material, and it is remembered for the same reason the throttle is:
    /// a reviewer who has decided a model reads this kind of video badly should
    /// not have to say so again in the next clip, the next launch, or the next
    /// browser. A field dropped on save would silently restore every model to
    /// the ballot.
    #[test]
    fn the_models_left_out_of_the_vote_are_remembered() {
        let mut s = Settings::default();
        assert_eq!(s.exclude_models, None, "a fresh install compares everything");
        s.exclude_models = Some(vec!["v0.3.0".into(), "anime-v1".into()]);
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(
            back.exclude_models.as_deref(),
            Some(&["v0.3.0".to_string(), "anime-v1".to_string()][..])
        );
    }

    /// "Nothing excluded" and "everything excluded" are different states and
    /// must not collapse into each other through the file: `null`/absent means
    /// every model competes, `[]` means the reviewer unchecked them all. The
    /// page treats the second as "nothing to compare" rather than as "compare
    /// everything", which is the only reading that matches what was ticked.
    #[test]
    fn an_empty_vote_is_not_the_same_as_an_untouched_one() {
        let none: Settings = serde_json::from_str(r#"{"exclude_models": null}"#).unwrap();
        assert_eq!(none.exclude_models, None);
        let empty: Settings = serde_json::from_str(r#"{"exclude_models": []}"#).unwrap();
        assert_eq!(empty.exclude_models.as_deref(), Some(&[][..]));
        let absent: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(absent.exclude_models, None);
        // and a hand-edited file that is not a list is a parse error rather
        // than a silent "everything competes" -- the page turns a 400 into a
        // visible note, which is how a typo gets noticed
        assert!(serde_json::from_str::<Settings>(r#"{"exclude_models": 7}"#).is_err());
    }

    /// `auto_suggest` moves to the picker, and the page must not be able to
    /// turn it off behind the picker's back by leaving the field out of a POST
    /// -- so the field has to survive a round trip through the file on its own,
    /// and a page that says nothing about it must leave it exactly as it was.
    #[test]
    fn the_auto_suggest_switch_is_the_files_to_keep() {
        let mut s = Settings::default();
        assert!(!s.auto_suggest, "a fresh install does not restyle unasked");
        s.auto_suggest = true;
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert!(back.auto_suggest, "the switch did not survive being written");
        // an absent field is the Default, which is the OFF above -- the state a
        // file with no opinion is in, not a state a request put it in
        let absent: Settings = serde_json::from_str("{}").unwrap();
        assert!(!absent.auto_suggest);
    }

    /// The raise switch decides whether a blocked stage is VISIBLE, and it is
    /// the one switch here whose default is ON -- so both halves matter: a file
    /// that says nothing must raise (the person never asked for silence), and a
    /// file that says `false` must be honoured rather than overridden by the
    /// default.
    #[test]
    fn the_raise_switch_defaults_on_and_a_no_is_remembered() {
        let fresh = Settings::default();
        assert_eq!(
            fresh.raise_window, None,
            "a fresh install has said nothing about it"
        );
        assert!(fresh.raises_window(), "saying nothing means raise");

        let mut s = Settings::default();
        s.raise_window = Some(false);
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.raise_window, Some(false), "the no did not survive");
        assert!(!back.raises_window(), "an explicit no must win over the default");

        // and the ON answer survives a write/read too, so turning it back on is
        // not silently lost either
        s.raise_window = Some(true);
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert!(back.raises_window());
    }

    /// The per-model weights are the one place a reviewer's judgement about a
    /// model is written down for the NEXT batch, so a field dropped on save
    /// would quietly give a discounted model its full voice back -- and the
    /// reviewer would only notice as a worse suggestion, never as a lost
    /// setting.
    #[test]
    fn the_per_model_weights_are_remembered_by_name() {
        let mut s = Settings::default();
        assert_eq!(s.conf_weights, None, "a fresh install believes every model");
        let mut w = std::collections::BTreeMap::new();
        w.insert("v0.5.0".to_string(), 0.9);
        w.insert("anime-v1".to_string(), 1.25);
        s.conf_weights = Some(w);
        let text = serde_json::to_string(&s).unwrap();
        // keyed by name, so the file reads as a statement about models rather
        // than about positions in whatever bundle happened to be loaded
        assert!(text.contains("v0.5.0"), "the key is the model's name");
        let back: Settings = serde_json::from_str(&text).unwrap();
        let got = back.conf_weights.expect("the weights did not survive");
        assert_eq!(got.get("v0.5.0"), Some(&0.9));
        assert_eq!(got.get("anime-v1"), Some(&1.25));
        // absent and null both mean "nobody has an opinion", which is not the
        // same as an empty map: both are simply "count what was said"
        for t in [r#"{"conf_weights": null}"#, "{}"] {
            assert_eq!(
                serde_json::from_str::<Settings>(t).unwrap().conf_weights,
                None
            );
        }
    }

    /// The bench width is a layout preference the reviewer sets by dragging an
    /// edge, and it is worth exactly nothing if it does not come back on the
    /// next page load -- the drag is the whole feature. Absent means "never
    /// dragged", which is the stylesheet's `clamp()`, NOT a zero-width column.
    #[test]
    fn the_bench_width_is_the_files_to_keep() {
        let mut s = Settings::default();
        assert_eq!(s.rail_width, None, "a fresh install has never been dragged");
        s.rail_width = Some(412);
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.rail_width, Some(412), "the width did not survive being written");
        // a file with no opinion reads as the stylesheet's own width
        assert_eq!(serde_json::from_str::<Settings>("{}").unwrap().rail_width, None);
        // a width outside the range the page's splitter can produce did not come
        // from a drag; it is refused by `parse_ui_prefs` rather than stored, and
        // the two constants are what that check is written against
        assert!(RAIL_MIN_W < RAIL_MAX_W);
        assert_eq!(RAIL_MIN_W, 220);
        assert_eq!(RAIL_MAX_W, 520);
    }
}
