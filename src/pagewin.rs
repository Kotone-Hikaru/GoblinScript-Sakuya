//! One browser window for a whole run.
//!
//! A batch brings up a crop page per video and a review page after the last
//! one. Opening the browser at each of those is a window that appears,
//! disappears and appears again while the person watches -- and every page
//! left behind stays on screen attached to a server that has already gone.
//!
//! So a page that has finished its job PARKS instead of dying. It goes on
//! answering one question -- "where to next?" -- and the next stage hands its
//! address to it. The window navigates itself; one page turns into the next
//! and the person sees one window the whole way through.
//!
//! Nothing here assumes the person leaves that window open: a parked page that
//! stops asking is a closed window, [`hand_off`] then says no, and the caller
//! opens a fresh one. The swap costs an address, never a stage.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long a parked page's silence means its window is gone. The page asks
/// every 400 ms while it waits, so seconds of quiet is a closed window, not a
/// busy one.
const ALIVE: Duration = Duration::from_secs(4);

/// How long a parked page waits for a next stage at all. Longer than any stage
/// that runs between two pages, short enough that a run abandoned overnight
/// does not hold a listening socket until morning. A page that outlives it is
/// told the connection is gone (its own poll fails) -- which is the truth.
pub(crate) const PATIENCE: Duration = Duration::from_secs(4 * 60 * 60);

struct Waiting {
    /// Where the parked page should go, once a next stage knows its address.
    next: Option<String>,
    /// When that page last asked -- the window is up while it keeps asking.
    heard: Instant,
    /// The address was taken and the page is navigating out. There is nobody
    /// left to hand another one to, and saying otherwise would let a stage
    /// skip its own window for a page that has already gone.
    leaving: bool,
}

fn slot() -> &'static Mutex<Option<Waiting>> {
    static SLOT: OnceLock<Mutex<Option<Waiting>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// A page has handed its work back and is asking what comes next.
pub fn park() {
    *slot().lock().unwrap() = Some(Waiting {
        next: None,
        heard: Instant::now(),
        leaving: false,
    });
}

/// `GET /api/next`: where the parked page should go, or nothing yet.
///
/// Taking the address out is what makes this safe to poll: a page that gets
/// `null` asks again, and a page that gets an address leaves. It is never
/// handed the same stage twice.
pub fn ask() -> Option<String> {
    let mut s = slot().lock().unwrap();
    match s.as_mut() {
        Some(w) => {
            w.heard = Instant::now();
            let out = w.next.take();
            if out.is_some() {
                w.leaving = true;
            }
            out
        }
        None => None,
    }
}

/// Is a parked page still there -- i.e. is its window still up and asking?
pub fn parked() -> bool {
    match slot().lock().unwrap().as_ref() {
        Some(w) => w.heard.elapsed() < ALIVE,
        None => false,
    }
}

/// The page is navigating away: this window now serves the next stage.
pub fn left() {
    *slot().lock().unwrap() = None;
}

/// Point the one open window at `url`. True when a parked page took it -- the
/// window is already up, so nothing should open another.
pub fn hand_off(url: &str) -> bool {
    let mut s = slot().lock().unwrap();
    match s.as_mut() {
        Some(w) if !w.leaving && w.heard.elapsed() < ALIVE => {
            w.next = Some(url.to_string());
            true
        }
        Some(w) if w.leaving => false, // already on its way; nothing to read it
        Some(_) => {
            // It stopped asking: the window was closed. Leave nothing behind
            // for a page that will never read it.
            *s = None;
            false
        }
        None => false,
    }
}

/// Show `url` to the person: hand the open window over, or open a new one.
pub fn deliver(url: &str, open: bool) {
    if hand_off(url) {
        raise_soon();
        return;
    }
    if open {
        crate::review::open_browser(url);
    }
    raise_soon();
}

/// Wait this long before the first attempt. The delay is for the HAND-OFF case:
/// there the window is already up on the PREVIOUS page and is about to
/// navigate. Raising it immediately would still work, but the title the search
/// matches on belongs to the page being left, and a browser that has not yet
/// committed the new document can be mid-swap. A few hundred milliseconds puts
/// the raise after the page has arrived, so what comes to the front is the
/// stage that is actually waiting.
const RAISE_AFTER: Duration = Duration::from_millis(400);

/// How long to keep trying after that. A COLD launch -- no browser of ours
/// running yet, which is the first page of every batch -- takes longer than
/// [`RAISE_AFTER`] to put a window up at all: measured on this machine, the
/// window exists at 484-677 ms and not at 400 ms. A single attempt therefore
/// missed it every time, and the stage waited behind whatever the person had
/// open. Looking again for a few seconds is the whole fix; nothing else about
/// the raise changes.
const RAISE_FOR: Duration = Duration::from_secs(6);

/// How often to look while the window is not there yet. Short enough that the
/// raise lands within a frame or two of the window appearing, long enough that
/// the desktop walk is not a busy loop.
const RAISE_EVERY: Duration = Duration::from_millis(150);

/// Try to raise, retrying until it lands or `tries` runs out.
///
/// Split out from the thread so the RETRY itself can be exercised: a single
/// attempt was the bug, and a single attempt is exactly what a test cannot tell
/// apart from a loop that never gets a second turn.
fn keep_raising<F>(mut raise: F, tries: usize, every: Duration) -> bool
where
    F: FnMut() -> bool,
{
    for i in 0..tries {
        if i > 0 {
            std::thread::sleep(every);
        }
        if raise() {
            return true;
        }
    }
    false
}

/// Lift the page's window to the front, shortly, without blocking the caller.
///
/// A stage calls [`deliver`] and then gets on with its own work -- it must not
/// wait on the desktop, and on a build with no switch (or no Windows) there is
/// nothing to wait for at all. So this only runs when the setting says so, and
/// it runs on its own thread.
fn raise_soon() {
    if !crate::settings::Settings::load().raises_window() {
        return;
    }
    if !crate::raise::available() {
        return;
    }
    std::thread::spawn(|| {
        std::thread::sleep(RAISE_AFTER);
        // The answer is deliberately ignored: no window on screen, or a desktop
        // that refused, is not an error in the pipeline. The stage still runs.
        let tries = (RAISE_FOR.as_millis() / RAISE_EVERY.as_millis()).max(1) as usize;
        let _ = keep_raising(
            || crate::raise::raise_by_title("goblinscript"),
            tries,
            RAISE_EVERY,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slot is one per process; these tests own it while they run.
    fn own() -> std::sync::MutexGuard<'static, ()> {
        crate::cancel::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn a_parked_page_takes_the_next_address_exactly_once() {
        let _guard = own();
        left();
        park();
        assert!(hand_off("http://127.0.0.1:1/"));
        assert_eq!(ask().as_deref(), Some("http://127.0.0.1:1/"));
        // handed once, not a loop: the page is already on its way
        assert_eq!(ask(), None);
        // ...and a stage that arrives during that navigation must NOT think
        // the same window is still there to take its address
        assert!(!hand_off("http://127.0.0.1:2/"));
        left();
    }

    #[test]
    fn nothing_parked_means_nothing_is_taken() {
        let _guard = own();
        left();
        assert!(!hand_off("http://127.0.0.1:3/"));
    }

    #[test]
    fn a_page_that_stops_asking_is_a_window_that_closed() {
        let _guard = own();
        left();
        park();
        // backdate the last question past the grace period
        slot().lock().unwrap().as_mut().unwrap().heard =
            Instant::now() - ALIVE - Duration::from_millis(1);
        assert!(!parked());
        assert!(!hand_off("http://127.0.0.1:4/"));
        assert!(!parked(), "a dead window is not left behind");
        left();
    }

    /// THE BUG THIS REPLACES, in miniature. A cold launch puts its window up
    /// after the first attempt has already run, so a single attempt misses it
    /// and the stage waits behind whatever the person had open. The retry is
    /// what makes the raise land; one attempt is not.
    #[test]
    fn a_raise_that_arrives_late_still_lands() {
        let mut calls = 0;
        let landed = keep_raising(
            || {
                calls += 1;
                // the window is only there from the third look onwards
                calls >= 3
            },
            40,
            Duration::ZERO,
        );
        assert!(landed, "the third attempt should have found the window");
        assert_eq!(calls, 3, "it must stop looking once the raise lands");
    }

    /// The first attempt succeeding is the HAND-OFF case: the window is already
    /// up and must not be raised again and again.
    #[test]
    fn a_raise_that_lands_at_once_does_not_keep_looking() {
        let mut calls = 0;
        let landed = keep_raising(
            || {
                calls += 1;
                true
            },
            40,
            Duration::ZERO,
        );
        assert!(landed);
        assert_eq!(calls, 1, "one raise is enough when the window is there");
    }

    /// No window at all must end quietly after the budget, not loop forever.
    #[test]
    fn a_window_that_never_appears_gives_up() {
        let mut calls = 0;
        let landed = keep_raising(
            || {
                calls += 1;
                false
            },
            7,
            Duration::ZERO,
        );
        assert!(!landed);
        assert_eq!(calls, 7, "it looks exactly as many times as it is allowed");
    }

    /// The budget has to outlast a cold launch, which is what the measurement
    /// said it did not: the window was up at 484-677 ms while the only attempt
    /// ran at 400 ms. These are the numbers the fix rests on.
    #[test]
    fn the_raise_budget_outlasts_a_cold_launch() {
        assert!(
            RAISE_AFTER + RAISE_FOR > Duration::from_millis(700),
            "a raise budget of {RAISE_AFTER:?}+{RAISE_FOR:?} would still miss a \
             window that takes 677 ms to appear"
        );
        assert!(
            RAISE_EVERY < RAISE_FOR,
            "one look in the whole budget is the bug again"
        );
    }
}
