//! Bringing the page's window to the front when a stage is waiting on a person.
//!
//! A batch runs for minutes and then STOPS: the crop page wants a decision, the
//! review page wants a reading. If the browser has drifted behind something
//! else, nothing on screen says so -- the run looks like it is still working,
//! and it will wait until [`crate::pagewin::PATIENCE`] runs out.
//!
//! ## What this does, and what it deliberately does NOT do
//!
//! It raises the window's Z-ORDER and stops there. It does NOT take the
//! keyboard focus.
//!
//! That distinction is the whole design. On Windows, taking the foreground is
//! privileged: `SetForegroundWindow` is refused unless the caller is already the
//! foreground process, was started by it, or has just received input. This
//! program is a console app whose console window belongs to conhost, and by the
//! time a stage needs attention the person has long since moved elsewhere -- so
//! this process is essentially NEVER eligible, and a foreground grab would
//! silently degrade to a taskbar flash. Building the feature on it would be
//! building on a call that is refused nearly every time it matters.
//!
//! Reordering the Z axis is NOT privileged. The `topmost -> not-topmost` toggle
//! below, both halves with `SWP_NOACTIVATE`, lifts the window over its
//! neighbours without asking for anything.
//!
//! The price is honest and small: the window comes up, the caret does not. The
//! person clicks it and carries on. The alternative -- a real focus grab via
//! `AttachThreadInput` -- can swallow characters someone was typing into another
//! window, and this program has no business doing that to announce itself.
//!
//! ## Which window
//!
//! Found by TITLE, matched case-insensitively against the page's own
//! `<title>`, which is why the needle is the program name and not a page name:
//! the pages translate their titles (`GoblinScript review` / `GoblinScript
//! 校对`), but every one of them is prefixed with the program's own name, which
//! is not translated.
//!
//! ⚠️ A title match alone is NOT enough, and finding that out cost a bug: the
//! console this pipeline runs in is titled `GoblinScript` too (Windows Terminal
//! reports `CASCADIA_HOSTING_WINDOW_CLASS`, conhost `ConsoleWindowClass`), so a
//! bare substring search matches "the black box the run is printing into" just
//! as readily as the browser. `EnumWindows` order is Z order, so which one came
//! first was an accident of what the person had touched last.
//!
//! So the choice is made on the window CLASS, not only the title:
//!
//!   * a browser's top-level class is preferred (`Chrome_WidgetWin_1` covers
//!     Chrome, Edge, Brave, Vivaldi and Opera; `MozillaWindowClass` Firefox);
//!   * a terminal's class is NEVER chosen, whatever its title says -- raising a
//!     console at somebody tells them nothing about the page that is waiting;
//!   * any other class that matches the title is still accepted, so a browser
//!     this list has not heard of (a WebView2 host, an exotic build) works.
//!
//! A browser puts every tab in one top-level window, so raising is per window
//! and cannot pick a tab; the stage has already navigated the tab it owns, so
//! raising the window shows it.
//!
//! ## Failure is not an error
//!
//! Every call here returns quietly. A Windows API that refuses, a title that is
//! not on screen, a non-Windows build, or a person who turned the switch off all
//! mean the same thing to the caller: nothing was raised, carry on. This is a
//! courtesy, not a step in the pipeline.

/// Raise the browser window whose title contains `needle`, without taking the
/// keyboard focus. True when a window was found and asked to come up.
///
/// `needle` is matched case-insensitively as a substring, so `"goblinscript"`
/// covers every page this program opens.
pub fn raise_by_title(needle: &str) -> bool {
    imp::raise_by_title(needle)
}

/// Whether the switch is on and the platform can do it at all.
pub fn available() -> bool {
    imp::AVAILABLE
}

/// Is this window class a terminal -- the thing that must never be raised while
/// a page is waiting?
///
/// Kept here, outside the `windows` module and free of any API, so the decision
/// can be tested on any host. The cost of getting it wrong is not a crash: it is
/// a black console jumping in front of the person and the page staying hidden.
pub fn is_terminal_class(class: &str) -> bool {
    matches!(
        class,
        "ConsoleWindowClass"          // conhost
        | "CASCADIA_HOSTING_WINDOW_CLASS" // Windows Terminal
        | "PseudoConsoleWindow"       // ConPTY hosts
        | "mintty"                    // MSYS / Git Bash
        | "VSCodeWindowClass"         // the editor's own terminal panel host
    )
}

/// Is this window class a browser's top-level window?
///
/// `Chrome_WidgetWin_` is not one class but a family: Chrome, Edge, Brave,
/// Vivaldi and Opera all report it with a version suffix, which is why this is a
/// prefix test.
pub fn is_browser_class(class: &str) -> bool {
    class.starts_with("Chrome_WidgetWin_") || matches!(class, "MozillaWindowClass" | "IEFrame")
}

/// Which candidate to raise, given each window's `(class, title)` in
/// `EnumWindows` order. `None` when nothing qualifies -- raising nothing is the
/// right answer for "the page is not on screen".
///
/// Split out from the Win32 walk so the choice itself is a pure function: the
/// bug this replaces was in the CHOICE, and a choice that can only be exercised
/// by setting up a real desktop is a choice nobody checks.
pub fn choose(cands: &[(String, String)], needle: &str) -> Option<usize> {
    let needle = needle.to_lowercase();
    let matches_title = |c: &(String, String)| c.1.to_lowercase().contains(&needle);

    // A browser, first: that is the thing being waited on.
    if let Some(i) = cands
        .iter()
        .position(|c| matches_title(c) && is_browser_class(&c.0))
    {
        return Some(i);
    }
    // Anything else that is not a terminal: a browser class this list has not
    // heard of still gets raised rather than silently doing nothing.
    cands
        .iter()
        .position(|c| matches_title(c) && !is_terminal_class(&c.0))
}

#[cfg(windows)]
mod imp {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    pub const AVAILABLE: bool = true;

    type Hwnd = isize;
    type Bool = i32;

    // Declared by hand rather than pulling in a windows-* crate: the rest of the
    // program adds no dependencies for platform calls, and a handful of
    // declarations are cheaper than a new dependency tree.
    #[link(name = "user32")]
    extern "system" {
        fn EnumWindows(cb: extern "system" fn(Hwnd, isize) -> Bool, l: isize) -> Bool;
        fn GetWindowTextW(h: Hwnd, buf: *mut u16, n: i32) -> i32;
        fn GetWindowTextLengthW(h: Hwnd) -> i32;
        fn GetClassNameW(h: Hwnd, buf: *mut u16, n: i32) -> i32;
        fn IsWindowVisible(h: Hwnd) -> Bool;
        fn IsIconic(h: Hwnd) -> Bool;
        fn ShowWindow(h: Hwnd, cmd: i32) -> Bool;
        fn SetWindowPos(
            h: Hwnd,
            after: Hwnd,
            x: i32,
            y: i32,
            cx: i32,
            cy: i32,
            flags: u32,
        ) -> Bool;
    }

    const SW_SHOWNOACTIVATE: i32 = 4;
    const SW_RESTORE: i32 = 9;

    const HWND_TOPMOST: Hwnd = -1;
    const HWND_NOTOPMOST: Hwnd = -2;

    // SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_SHOWWINDOW
    // NOMOVE/NOSIZE keep the window where the person put it; NOACTIVATE is the
    // promise that this never steals the caret; SHOWWINDOW makes an
    // already-visible window actually get painted.
    const SWP_FLAGS: u32 = 0x0001 | 0x0002 | 0x0010 | 0x0040;

    /// Every visible window's `(hwnd, class, title)`, gathered in one pass.
    ///
    /// The whole list rather than the first hit, because the CHOICE needs to see
    /// more than one candidate: the console and the browser can both be titled
    /// with the program's name, and only their classes tell them apart.
    /// `EnumWindows` order is Z order, which is what makes "first" meaningless
    /// here -- it is whatever the person touched last.
    struct Search<'a> {
        found: Vec<(Hwnd, String, String)>,
        needle: &'a str,
    }

    fn wide_to_string(buf: &[u16], len: i32) -> String {
        if len <= 0 {
            return String::new();
        }
        let n = (len as usize).min(buf.len());
        OsString::from_wide(&buf[..n])
            .to_string_lossy()
            .into_owned()
    }

    extern "system" fn visit(h: Hwnd, l: isize) -> Bool {
        // SAFETY: `l` is the `&mut Search` passed to EnumWindows below, which
        // outlives this synchronous callback.
        let s = unsafe { &mut *(l as *mut Search) };
        // A hidden window is not something a person can be looking at, so it is
        // never the one to raise. (Not `!IsWindowVisible(h)`: a Win32 BOOL is
        // an i32, and `!` on it is a bitwise NOT -- it would reject every
        // visible window and accept every hidden one.)
        if unsafe { IsWindowVisible(h) } == 0 {
            return 1;
        }
        let tlen = unsafe { GetWindowTextLengthW(h) };
        if tlen <= 0 {
            return 1;
        }
        let mut tbuf = vec![0u16; tlen as usize + 1];
        let tgot = unsafe { GetWindowTextW(h, tbuf.as_mut_ptr(), tbuf.len() as i32) };
        let title = wide_to_string(&tbuf, tgot);
        // Cheap rejection before the class lookup: most windows are titled
        // something else entirely.
        if !title.to_lowercase().contains(s.needle) {
            return 1;
        }
        let mut cbuf = vec![0u16; 256];
        let cgot = unsafe { GetClassNameW(h, cbuf.as_mut_ptr(), cbuf.len() as i32) };
        let class = wide_to_string(&cbuf, cgot);
        s.found.push((h, class, title));
        1 // keep going: a later candidate may be the browser
    }

    pub fn raise_by_title(needle: &str) -> bool {
        let lower = needle.to_lowercase();
        let mut search = Search {
            found: Vec::new(),
            needle: &lower,
        };
        // SAFETY: the callback only reads/writes the `Search` it is handed, and
        // `EnumWindows` calls it synchronously before returning.
        unsafe {
            EnumWindows(visit, &mut search as *mut Search as isize);
        }
        let cands: Vec<(String, String)> = search
            .found
            .iter()
            .map(|(_, c, t)| (c.clone(), t.clone()))
            .collect();
        let idx = match super::choose(&cands, &lower) {
            Some(i) => i,
            None => return false,
        };
        let h = search.found[idx].0;

        // A minimised window has no place in the Z order to be raised to, so it
        // is restored first -- still without activation. SHOWNOACTIVATE so the
        // restore cannot take the caret either; fall back to SW_RESTORE only if
        // the window stays minimised, which is what some browsers do.
        if unsafe { IsIconic(h) } != 0 {
            unsafe { ShowWindow(h, SW_SHOWNOACTIVATE) };
            if unsafe { IsIconic(h) } != 0 {
                unsafe { ShowWindow(h, SW_RESTORE) };
            }
        }

        // The raise: toggle the topmost bit on and off. Both halves are
        // NOACTIVATE, so the window arrives over everything else with the
        // keyboard focus exactly where the person left it.
        unsafe {
            SetWindowPos(h, HWND_TOPMOST, 0, 0, 0, 0, SWP_FLAGS);
            SetWindowPos(h, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_FLAGS);
        }
        true
    }
}

#[cfg(not(windows))]
mod imp {
    pub const AVAILABLE: bool = false;

    /// Nothing to do off Windows: the windows this raises belong to a browser
    /// the program did not start, and every desktop answers that question
    /// differently. Saying so plainly beats a half-measure.
    pub fn raise_by_title(_needle: &str) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(class: &str, title: &str) -> (String, String) {
        (class.to_string(), title.to_string())
    }

    /// The title the pages actually set -- the match is on the page's own
    /// `<title>`, so a rename there must be caught here rather than discovered
    /// as "the window never comes up".
    #[test]
    fn the_page_titles_all_say_goblinscript() {
        for (name, html) in [
            ("review", include_str!("review.html")),
            ("crop", include_str!("cropedit.html")),
            ("vr", include_str!("vr.html")),
        ] {
            let lower = html.to_lowercase();
            let at = lower
                .find("<title>")
                .unwrap_or_else(|| panic!("{name} has no <title>"));
            let rest = &lower[at + 7..];
            let end = rest.find("</title>").expect("unterminated <title>");
            assert!(
                rest[..end].contains("goblinscript"),
                "{name}'s title is {:?}, which the raise would never match",
                &rest[..end]
            );
        }
    }

    /// THE BUG THIS REPLACED. The console this pipeline runs in is titled
    /// `GoblinScript`, exactly like the pages are, and `EnumWindows` hands
    /// windows over in Z order -- so "first title match" could raise the black
    /// console box and leave the page hidden. Both are in this list, the
    /// console first, and the browser must still win.
    #[test]
    fn the_console_never_wins_over_the_browser() {
        let cands = [
            pair("CASCADIA_HOSTING_WINDOW_CLASS", "GoblinScript"),
            pair("Chrome_WidgetWin_1", "GoblinScript 校对 - Google Chrome"),
        ];
        assert_eq!(choose(&cands, "goblinscript"), Some(1));

        // ...and the order must not matter: put the browser first and the answer
        // is still the browser, for the right reason rather than by luck
        let flipped = [cands[1].clone(), cands[0].clone()];
        assert_eq!(choose(&flipped, "goblinscript"), Some(0));
    }

    /// A terminal on its own is NOT raised. Bringing a console forward tells the
    /// person nothing about the page that is waiting, so "nothing happened" is
    /// the correct outcome.
    #[test]
    fn a_console_alone_is_never_raised() {
        for class in [
            "ConsoleWindowClass",
            "CASCADIA_HOSTING_WINDOW_CLASS",
            "PseudoConsoleWindow",
            "mintty",
        ] {
            let cands = [pair(class, "GoblinScript")];
            assert_eq!(
                choose(&cands, "goblinscript"),
                None,
                "{class} must never be raised"
            );
        }
    }

    /// A browser class this list has not heard of still works, so a WebView2
    /// host or an exotic build is not silently unsupported.
    #[test]
    fn an_unknown_browser_still_gets_raised() {
        let cands = [pair("SomeNewBrowserWindow", "GoblinScript review")];
        assert_eq!(choose(&cands, "goblinscript"), Some(0));
    }

    /// Titles are matched case-insensitively, and a window titled something else
    /// entirely is not a candidate at all.
    #[test]
    fn only_the_programs_own_windows_are_candidates() {
        let cands = [
            pair("Chrome_WidgetWin_1", "Some other tab - Google Chrome"),
            pair("Chrome_WidgetWin_1", "GOBLINSCRIPT CROP - Google Chrome"),
        ];
        assert_eq!(choose(&cands, "goblinscript"), Some(1));
        let none = [pair("Chrome_WidgetWin_1", "Some other tab - Google Chrome")];
        assert_eq!(choose(&none, "goblinscript"), None);
    }

    /// Every class this program cares about is recognised by name, not by
    /// accident of a substring: `Chrome_WidgetWin_1` is a browser while
    /// `ConsoleWindowClass` is a terminal, and neither is the other.
    #[test]
    fn the_class_families_do_not_overlap() {
        assert!(is_browser_class("Chrome_WidgetWin_1"));
        assert!(is_browser_class("Chrome_WidgetWin_0"));
        assert!(is_browser_class("MozillaWindowClass"));
        assert!(!is_browser_class("ConsoleWindowClass"));
        assert!(is_terminal_class("ConsoleWindowClass"));
        for b in ["Chrome_WidgetWin_1", "MozillaWindowClass"] {
            assert!(!is_terminal_class(b), "{b} is not a terminal");
        }
        for t in ["ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS"] {
            assert!(!is_browser_class(t), "{t} is not a browser");
        }
    }

    /// Asking for a title that cannot be on screen must be a quiet `false`, not
    /// a panic and not a stuck thread -- this runs on the pipeline's own path.
    #[test]
    fn a_title_nobody_has_raises_nothing() {
        assert!(!raise_by_title("no window is ever called this 7f3a1c"));
    }
}
