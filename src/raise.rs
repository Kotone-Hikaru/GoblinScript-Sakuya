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
//! `<title>` (`GoblinScript review`, `GoblinScript crop`, ...). Titles are what
//! the pages set themselves, they survive the window being reused across
//! stages, and matching them needs no handle kept from a previous stage -- which
//! is exactly what a long batch cannot promise.
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

#[cfg(windows)]
mod imp {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    pub const AVAILABLE: bool = true;

    type Hwnd = isize;
    type Bool = i32;

    // The four entry points this needs, declared by hand rather than pulling in
    // a windows-* crate: the rest of the program adds no dependencies for
    // platform calls, and five declarations are cheaper than a new tree.
    #[link(name = "user32")]
    extern "system" {
        fn EnumWindows(cb: extern "system" fn(Hwnd, isize) -> Bool, l: isize) -> Bool;
        fn GetWindowTextW(h: Hwnd, buf: *mut u16, n: i32) -> i32;
        fn GetWindowTextLengthW(h: Hwnd) -> i32;
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

    /// What `EnumWindows` is looking for, handed through the `lparam` as a raw
    /// pointer. The callback runs on this thread, synchronously, inside the
    /// `EnumWindows` call, so the borrow cannot outlive the struct.
    struct Search<'a> {
        needle: &'a str,
        found: Option<Hwnd>,
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
        let len = unsafe { GetWindowTextLengthW(h) };
        if len <= 0 {
            return 1;
        }
        let mut buf = vec![0u16; len as usize + 1];
        let got = unsafe { GetWindowTextW(h, buf.as_mut_ptr(), buf.len() as i32) };
        if got <= 0 {
            return 1;
        }
        buf.truncate(got as usize);
        let title = OsString::from_wide(&buf).to_string_lossy().to_lowercase();
        if title.contains(s.needle) {
            s.found = Some(h);
            return 0; // stop: the first match is the one to raise
        }
        1
    }

    pub fn raise_by_title(needle: &str) -> bool {
        let needle = needle.to_lowercase();
        let mut search = Search {
            needle: &needle,
            found: None,
        };
        // SAFETY: the callback only reads/writes the `Search` it is handed, and
        // `EnumWindows` calls it synchronously before returning.
        unsafe {
            EnumWindows(visit, &mut search as *mut Search as isize);
        }
        let h = match search.found {
            Some(h) => h,
            None => return false,
        };

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

    /// Asking for a title that cannot be on screen must be a quiet `false`, not
    /// a panic and not a stuck thread -- this runs on the pipeline's own path.
    #[test]
    fn a_title_nobody_has_raises_nothing() {
        assert!(!raise_by_title("no window is ever called this 7f3a1c"));
    }
}
