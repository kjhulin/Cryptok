//! A one-line progress bar on stderr for long jobs such as training.

use std::cell::RefCell;
use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

const WIDTH: usize = 30;
const LABEL_WIDTH: usize = 22;

pub struct Bar {
    start: Instant,
    tty: bool,
    state: RefCell<State>,
}

struct State {
    last_draw: Option<Instant>,
    /// Last tenth-of-the-way mark printed (plain output only).
    last_mark: i64,
    drawn: bool,
}

impl Bar {
    /// A live bar when stderr is a terminal; otherwise one plain line per 10% (so CI logs stay short).
    pub fn new() -> Self {
        Bar { start: Instant::now(), tty: std::io::stderr().is_terminal(), state: RefCell::new(State { last_draw: None, last_mark: -1, drawn: false }) }
    }

    /// Show `frac` (0.0 to 1.0, clamped) of the whole job done, working on `label`.
    pub fn set(&self, label: &str, frac: f64) {
        let frac = frac.clamp(0.0, 1.0);
        let mut st = self.state.borrow_mut();
        let now = Instant::now();
        if self.tty {
            let pct = (frac * 100.0).floor() as i64;
            let due = st.last_draw.map_or(true, |t| now.duration_since(t) >= Duration::from_millis(80));
            if !due && frac < 1.0 {
                return;
            }
            st.last_draw = Some(now);
            st.drawn = true;
            let filled = (frac * WIDTH as f64).round() as usize;
            let mut err = std::io::stderr().lock();
            let _ = write!(err, "\r{label:<LABEL_WIDTH$} [{}{}] {pct:>3}%  {:>4.0}s", "#".repeat(filled), "-".repeat(WIDTH - filled), self.start.elapsed().as_secs_f64());
            let _ = err.flush();
        } else {
            let mark = (frac * 10.0).floor() as i64;
            if mark > st.last_mark {
                st.last_mark = mark;
                eprintln!("[{:>3}%] {label}", mark * 10);
            }
        }
    }

    /// End the bar's line so later output starts cleanly.
    pub fn finish(&self) {
        if self.tty && self.state.borrow().drawn {
            eprintln!();
        }
    }
}
