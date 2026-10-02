//! Automatic solving: run the analyser, try every plausible solver, and rank the results.
//!
//! Attempts are compared by description length: the language-model score of the recovered
//! plaintext minus the cost of the key the solver was free to choose, per letter. A
//! substitution key costs ln(26!) nats, a Caesar shift almost nothing, and a running key
//! costs its own language-model score (it is a second English text).

use crate::analyze::analyze;
use crate::chain::{self, ChainOptions};
use crate::classic::{climb_ngram_size, solve_vigenere_with, Alphabet};
use crate::lm::LangModel;
use crate::periodic::{self, Mode};
use crate::polygraphic;
use crate::rkc::{self, RkcOptions};
use crate::subst;
use crate::text::{scrub, unscrub};
use crate::transpo::{self, Method};
use std::time::Instant;

#[derive(Clone)]
pub struct AutoOptions {
    /// Run every solver instead of only those the analyser suggests.
    pub exhaustive: bool,
    /// Beam width for the running key solver (it only runs for texts up to `rkc_max_len`).
    pub rkc_beam: usize,
    pub rkc_max_len: usize,
    pub max_period: usize,
    /// Keywords for keyed-alphabet (Quagmire III) Vigenère; empty = skip.
    pub alphabet_keywords: Vec<String>,
    /// Word model for the running key solver and its weight (a large gain on short texts).
    pub words: Option<(std::sync::Arc<crate::words::WordTrie>, f32)>,
}

impl Default for AutoOptions {
    fn default() -> Self {
        AutoOptions { exhaustive: false, rkc_beam: 20_000, rkc_max_len: 400, max_period: 20, alphabet_keywords: vec![], words: None }
    }
}

#[derive(Clone, Debug)]
pub struct Attempt {
    pub solver: String,
    pub detail: String,
    pub plain: Vec<u8>,
    /// Running key only: the other stream (key and plaintext are interchangeable).
    pub other: Option<Vec<u8>>,
    /// Full-model log-prob per letter of `plain`.
    pub per_letter: f32,
    /// Key description length in nats.
    pub cost: f32,
    pub secs: f64,
}

impl Attempt {
    /// Score used to compare attempts from different solvers.
    pub fn adjusted(&self) -> f32 {
        self.per_letter - self.cost / self.plain.len().max(1) as f32
    }
}

fn ln_factorial(k: usize) -> f32 {
    (2..=k).map(|i| (i as f32).ln()).sum()
}

/// Which solver groups to run for the analyser's suggestions.
fn wanted(commands: &[String], group: &str, exhaustive: bool) -> bool {
    exhaustive || commands.iter().any(|c| c == group)
}

pub fn auto_solve(lm: &LangModel, text: &str, opt: &AutoOptions) -> Vec<Attempt> {
    let cipher = scrub(text);
    let n = cipher.len();
    let mut out: Vec<Attempt> = Vec::new();
    if n < 8 {
        return out;
    }
    let report = analyze(text);
    let cmds: Vec<String> = report.suggestions.iter().map(|s| s.0.clone()).filter(|c| !c.is_empty()).collect();
    let exhaustive = opt.exhaustive || cmds.is_empty();
    let q = lm.dense(climb_ngram_size(lm, n));
    let mut push = |solver: &str, detail: String, plain: Vec<u8>, other: Option<Vec<u8>>, cost: f32, t: Instant| {
        let per_letter = lm.score_per_letter(&plain);
        out.push(Attempt { solver: solver.into(), detail, plain, other, per_letter, cost, secs: t.elapsed().as_secs_f64() });
    };

    if wanted(&cmds, "subst", exhaustive) {
        let t = Instant::now();
        if let Some(c) = subst::solve_affine_family(lm, &cipher, 1).into_iter().next() {
            push("affine", c.description, c.plain, None, 312f32.ln(), t);
        }
        let t = Instant::now();
        let c = subst::solve_substitution(lm, &q, &cipher, 200, 1);
        push("subst", c.description, c.plain, None, ln_factorial(26), t);
        // A transposition over a substitution has the same statistics as a plain
        // substitution, so try the supported chains too.
        let opts = ChainOptions::default();
        for steps in [[chain::Step::Columnar, chain::Step::Subst], [chain::Step::Rail, chain::Step::Subst]] {
            let t = Instant::now();
            if let Some(r) = chain::solve_chain(lm, &cipher, &steps, &opts).ok().and_then(|v| v.into_iter().next()) {
                push("chain", r.path.join(" -> "), r.plain, None, 70.0, t);
            }
        }
    }
    if wanted(&cmds, "transpose", exhaustive) {
        let t = Instant::now();
        if let Some(s) = transpo::solve_route(lm, &q, text, 60, 1).into_iter().next() {
            push("route", s.describe(), scrub(&s.text), None, 12.0, t);
        }
        let t = Instant::now();
        if let Some(s) = transpo::solve_columnar(lm, &q, text, 2, 12, 8, 1).into_iter().next() {
            let k = if let Method::Columnar { order } = &s.method { order.len() } else { 0 };
            push("columnar", s.describe(), scrub(&s.text), None, ln_factorial(k), t);
        }
        let t = Instant::now();
        if let Some(s) = transpo::solve_rail_fence(lm, text, 20, 1).into_iter().next() {
            push("rail", s.describe(), scrub(&s.text), None, 7.0, t);
        }
    }
    if wanted(&cmds, "periodic", exhaustive) {
        for mode in [Mode::Vigenere, Mode::Beaufort, Mode::VariantBeaufort, Mode::Porta, Mode::Gronsfeld] {
            let t = Instant::now();
            let sols = periodic::solve_periodic(lm, &q, &cipher, &mode, opt.max_period, 10);
            // `solve_periodic` already ranks with a length penalty; charge the key here too.
            if let Some(s) = sols.into_iter().next() {
                let cost = s.period as f32 * (mode.shifts() as f32).ln();
                push(&mode.name().to_lowercase().replace(' ', "-"), format!("period {} key {}", s.period, s.key), s.plain, None, cost, t);
            }
        }
        for kw in &opt.alphabet_keywords {
            let t = Instant::now();
            let a = Alphabet::from_keyword(kw);
            if let Some(s) = solve_vigenere_with(lm, &q, &cipher, &a, opt.max_period, 30).into_iter().next() {
                let cost = s.period as f32 * 26f32.ln();
                push("quagmire3", format!("alphabet {} period {} key {}", kw, s.period, unscrub(&s.key)), s.plain, None, cost, t);
            }
        }
    }
    if wanted(&cmds, "autokey", exhaustive) {
        let t = Instant::now();
        if let Some(s) = periodic::solve_autokey(lm, &q, &cipher, 12, 6).into_iter().next() {
            let kind = if s.kind == periodic::Autokey::Plaintext { "plaintext" } else { "ciphertext" };
            let cost = s.primer.len() as f32 * 26f32.ln();
            push("autokey", format!("{kind}, primer {}", unscrub(&s.primer)), s.plain, None, cost, t);
        }
    }
    if wanted(&cmds, "playfair", exhaustive) && n >= 20 {
        let t = Instant::now();
        let s = polygraphic::solve_playfair(lm, &q, &cipher, 500_000, 32, 1);
        push("playfair", format!("square {}", polygraphic::square_string(&s.square)), s.plain, None, ln_factorial(25), t);
    }
    if wanted(&cmds, "bifid", exhaustive) && n >= 20 {
        let t = Instant::now();
        let periods = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        if let Some(s) = polygraphic::solve_bifid(lm, &q, &cipher, &periods, 100_000, 4, 1).into_iter().next() {
            push("bifid", format!("period {} square {}", s.period, polygraphic::square_string(&s.square)), s.plain, None, ln_factorial(25), t);
        }
    }
    if wanted(&cmds, "hill", exhaustive) && n >= 8 {
        let t = Instant::now();
        let even = &cipher[..n / 2 * 2];
        if let Some(s) = polygraphic::solve_hill2(lm, &q, even, 1).into_iter().next() {
            push("hill", format!("key {:?}", s.matrix), s.plain, None, 157_248f32.ln(), t);
        }
    }
    if wanted(&cmds, "rkc", exhaustive) && n <= opt.rkc_max_len {
        let t = Instant::now();
        let o = RkcOptions { beam: opt.rkc_beam, results: 1, word_weight: opt.words.as_ref().map_or(0.0, |w| w.1), ..Default::default() };
        let trie = opt.words.as_ref().map(|w| &*w.0);
        if let Some(s) = rkc::solve_words(lm, trie, &cipher, &o, None, None).into_iter().next() {
            // Key and plaintext are interchangeable; treat the more fluent stream as "plain".
            let (a, b) = (lm.score_per_letter(&s.key), lm.score_per_letter(&s.plain));
            let (p, k) = if a >= b { (s.key, s.plain) } else { (s.plain, s.key) };
            // Description length: the key stream is paid for by its own (language-model) cost.
            let cost = -lm.score(&k);
            push("rkc", (if trie.is_some() { "running key (character + word model)" } else { "running key (character model)" }).into(), p, Some(k), cost, t);
        }
    }
    out.sort_by(|a, b| b.adjusted().total_cmp(&a.adjusted()));
    out
}

/// Fraction of expected-plaintext positions matched by `got` (compared from the start).
pub fn agreement(got: &[u8], expected: &[u8]) -> f64 {
    let n = expected.len().min(got.len());
    if expected.is_empty() {
        return 0.0;
    }
    got.iter().zip(expected).take(n).filter(|(a, b)| a == b).count() as f64 / expected.len() as f64
}

/// Accuracy of an attempt against known plaintext. A running-key solution is two streams
/// that may swap roles at any position, so a position counts if either stream has the
/// expected letter (chance level is then about 8%, not 4%).
pub fn accuracy(a: &Attempt, expected: &[u8]) -> f64 {
    let Some(o) = &a.other else { return agreement(&a.plain, expected) };
    if expected.is_empty() {
        return 0.0;
    }
    expected.iter().enumerate().filter(|&(i, &e)| a.plain.get(i) == Some(&e) || o.get(i) == Some(&e)).count() as f64 / expected.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{model, sample};

    #[test]
    fn picks_the_right_solver() {
        let lm = model();
        let p = sample(130_000, 300);
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("subst", subst::substitution_encrypt(&p, "QWERTYUIOPASDFGHJKLZXCVBNM")),
            ("affine", subst::affine_encrypt(&p, 7, 3)),
            ("beaufort", periodic::encrypt(&Mode::Beaufort, &p, &[2, 8, 15, 7, 4, 17])),
            ("rail", transpo::rail_fence_encrypt(&p, 4, 0)),
        ];
        for (want, c) in cases {
            let all = auto_solve(lm, &unscrub(&c), &AutoOptions::default());
            let best = &all[0];
            assert_eq!(best.solver, want, "{} -> {}", want, best.detail);
            assert!(accuracy(best, &p) > 0.95, "{want}");
        }
    }
}
