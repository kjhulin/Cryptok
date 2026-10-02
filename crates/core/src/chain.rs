//! Cipher chaining: undo several layers of different ciphers, outermost first.
//!
//! The difficulty is that an outer layer must be solved while the inner layers are still
//! scrambling the text, so ordinary language-model scores are useless. Each stage is
//! therefore ranked with a *proxy* that the remaining inner layers leave intact:
//!
//! * inner layers are all transpositions → the outer substitution is ranked by **unigram**
//!   likelihood (transposition keeps letter frequencies);
//! * inner layers are all monoalphabetic substitutions → the outer transposition is ranked
//!   by **bigram repetition** (substitution keeps which letter pairs repeat);
//! * the last stage is ranked by the full language model.
//!
//! A beam of the best candidates is carried from stage to stage. Other combinations
//! (for example two transpositions in a row) cannot be separated statistically and are
//! rejected with an explanation.

use crate::lm::{DenseNgram, LangModel};
use crate::periodic::{self, Mode};
use crate::polygraphic;
use crate::subst;
use crate::text::unscrub;
use crate::transpo::{self, Scorer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Affine,
    Subst,
    Periodic(PeriodicKind),
    Rail,
    Route,
    Columnar,
    Autokey,
    Hill,
    Playfair,
    Bifid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeriodicKind {
    Vigenere,
    Beaufort,
    VariantBeaufort,
    Porta,
    Gronsfeld,
}

pub const STEP_NAMES: &str = "affine, subst, vigenere, beaufort, variant-beaufort, porta, gronsfeld, rail, route, columnar, autokey, hill, playfair, bifid";

pub fn parse_step(s: &str) -> Option<Step> {
    use PeriodicKind::*;
    Some(match s.trim().to_ascii_lowercase().as_str() {
        "affine" | "caesar" | "atbash" => Step::Affine,
        "subst" | "substitution" => Step::Subst,
        "vigenere" => Step::Periodic(Vigenere),
        "beaufort" => Step::Periodic(Beaufort),
        "variant-beaufort" | "variant" => Step::Periodic(VariantBeaufort),
        "porta" => Step::Periodic(Porta),
        "gronsfeld" => Step::Periodic(Gronsfeld),
        "rail" => Step::Rail,
        "route" => Step::Route,
        "columnar" => Step::Columnar,
        "autokey" => Step::Autokey,
        "hill" => Step::Hill,
        "playfair" => Step::Playfair,
        "bifid" => Step::Bifid,
        _ => return None,
    })
}

impl Step {
    fn is_transposition(self) -> bool {
        matches!(self, Step::Rail | Step::Route | Step::Columnar)
    }
    fn is_mono(self) -> bool {
        matches!(self, Step::Affine | Step::Subst)
    }
    /// Substitution-like steps that a unigram proxy can rank.
    fn is_unigram_rankable(self) -> bool {
        self.is_mono() || matches!(self, Step::Periodic(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proxy {
    Full,
    Unigram,
    Pattern,
}

/// Choose the ranking proxy for each step (outermost first), or explain why the chain
/// cannot be solved.
pub fn plan(steps: &[Step]) -> Result<Vec<Proxy>, String> {
    if steps.is_empty() {
        return Err("a chain needs at least one step".into());
    }
    let mut out = Vec::new();
    for (i, &s) in steps.iter().enumerate() {
        let inner = &steps[i + 1..];
        let p = if inner.is_empty() {
            Proxy::Full
        } else if s.is_unigram_rankable() && inner.iter().all(|t| t.is_transposition()) {
            Proxy::Unigram
        } else if s.is_transposition() && inner.iter().all(|t| t.is_mono()) {
            Proxy::Pattern
        } else {
            return Err(format!(
                "cannot solve step {} ({s:?}) before {:?}: its output is still scrambled by the inner steps in a way no statistic survives. \
                 Supported: a substitution/periodic layer over transpositions, a transposition over monoalphabetic substitutions; \
                 autokey/hill/playfair/bifid only as the innermost step",
                i + 1,
                inner
            ));
        };
        out.push(p);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct ChainOptions {
    /// Candidates carried from stage to stage.
    pub beam: usize,
    pub max_period: usize,
    pub restarts: usize,
    pub max_width: usize,
    pub max_cols: usize,
    pub max_rails: usize,
}

impl Default for ChainOptions {
    fn default() -> Self {
        ChainOptions { beam: 5, max_period: 20, restarts: 10, max_width: 40, max_cols: 10, max_rails: 12 }
    }
}

#[derive(Clone, Debug)]
pub struct ChainResult {
    /// What each step found, outermost first.
    pub path: Vec<String>,
    pub plain: Vec<u8>,
    /// Mean full-model log-prob per letter.
    pub per_letter: f32,
}

#[derive(Clone)]
struct Cand {
    text: Vec<u8>,
    path: Vec<String>,
    /// Stage score used for pruning (per letter).
    score: f32,
}

/// Bigram repetition: sum of squared bigram counts per letter. Unchanged by any
/// monoalphabetic relabelling, and high for text that has been put back in order.
pub fn bigram_pattern(b: &[u8]) -> f32 {
    if b.len() < 2 {
        return 0.0;
    }
    let mut c = vec![0u32; 676];
    for w in b.windows(2) {
        c[w[0] as usize * 26 + w[1] as usize] += 1;
    }
    c.iter().map(|&x| (x * x) as f32).sum::<f32>() / b.len() as f32
}

struct Ctx<'a> {
    lm: &'a LangModel,
    quad: DenseNgram,
    uni: DenseNgram,
    opt: &'a ChainOptions,
}

impl Ctx<'_> {
    /// Per-letter stage score under `proxy`.
    fn per_letter(&self, proxy: Proxy, plain: &[u8]) -> f32 {
        let n = plain.len().max(1) as f32;
        match proxy {
            Proxy::Full => self.lm.score_per_letter(plain),
            Proxy::Unigram => self.uni.score(plain) / n,
            Proxy::Pattern => bigram_pattern(plain),
        }
    }

    fn dense_for(&self, proxy: Proxy) -> &DenseNgram {
        if proxy == Proxy::Unigram { &self.uni } else { &self.quad }
    }
}

fn run_step(ctx: &Ctx, step: Step, proxy: Proxy, text: &[u8]) -> Vec<Cand> {
    let beam = ctx.opt.beam.max(1);
    let mut out: Vec<Cand> = Vec::new();
    let mut add = |plain: Vec<u8>, desc: String, score: f32| out.push(Cand { text: plain, path: vec![desc], score });
    match step {
        Step::Affine => {
            for a in [1u8, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25] {
                for b in 0..26u8 {
                    let plain = subst::affine_decrypt(text, a, b);
                    let desc = match (a, b) {
                        (1, _) => format!("Caesar shift {b}"),
                        (25, 25) => "Atbash".to_string(),
                        _ => format!("Affine a={a} b={b}"),
                    };
                    let s = ctx.per_letter(proxy, &plain);
                    add(plain, desc, s);
                }
            }
        }
        Step::Subst => {
            let c = subst::solve_substitution(ctx.lm, ctx.dense_for(proxy), text, ctx.opt.restarts * 10, 1);
            let s = ctx.per_letter(proxy, &c.plain);
            add(c.plain, c.description, s);
        }
        Step::Periodic(kind) => {
            let mode = match kind {
                PeriodicKind::Vigenere => Mode::Vigenere,
                PeriodicKind::Beaufort => Mode::Beaufort,
                PeriodicKind::VariantBeaufort => Mode::VariantBeaufort,
                PeriodicKind::Porta => Mode::Porta,
                PeriodicKind::Gronsfeld => Mode::Gronsfeld,
            };
            let pen = (mode.shifts() as f32).ln();
            for s in periodic::solve_periodic(ctx.lm, ctx.dense_for(proxy), text, &mode, ctx.opt.max_period, ctx.opt.restarts) {
                // `solve_periodic` ranks with the full model; re-rank with the proxy.
                let per = ctx.per_letter(proxy, &s.plain);
                let adj = per - s.period as f32 * pen / text.len().max(1) as f32;
                add(s.plain, format!("{} period {} key {}", s.mode, s.period, s.key), adj);
            }
        }
        Step::Rail | Step::Route | Step::Columnar => {
            let letters = unscrub(text);
            let fast: Box<dyn Fn(&[u8]) -> f32 + Sync + '_> = match proxy {
                Proxy::Pattern => Box::new(|b: &[u8]| bigram_pattern(b) * b.len() as f32),
                _ => Box::new(|b: &[u8]| ctx.quad.score(b)),
            };
            let full: Box<dyn Fn(&[u8]) -> f32 + Sync + '_> = match proxy {
                Proxy::Pattern => Box::new(bigram_pattern),
                _ => Box::new(|b: &[u8]| ctx.lm.score_per_letter(b)),
            };
            let (fast, full): (Scorer, Scorer) = (&*fast, &*full);
            let sols = match step {
                Step::Rail => transpo::solve_rail_fence_with(full, &letters, ctx.opt.max_rails, beam),
                Step::Route => transpo::solve_route_with(fast, full, &letters, ctx.opt.max_width, beam),
                _ => transpo::solve_columnar_with(fast, full, &letters, 2, ctx.opt.max_cols, ctx.opt.restarts, beam),
            };
            for s in sols {
                let plain = crate::text::scrub(&s.text);
                add(plain, s.describe(), s.per_letter);
            }
        }
        Step::Autokey => {
            for s in periodic::solve_autokey(ctx.lm, ctx.dense_for(proxy), text, 12, ctx.opt.restarts).into_iter().take(beam) {
                let kind = if s.kind == periodic::Autokey::Plaintext { "plaintext" } else { "ciphertext" };
                let desc = format!("{kind} autokey, primer {}", unscrub(&s.primer));
                let sc = ctx.per_letter(proxy, &s.plain);
                add(s.plain, desc, sc);
            }
        }
        Step::Hill => {
            let mut t = text.to_vec();
            t.truncate(t.len() / 2 * 2);
            for s in polygraphic::solve_hill2(ctx.lm, ctx.dense_for(proxy), &t, beam) {
                let m = s.matrix;
                let sc = ctx.per_letter(proxy, &s.plain);
                add(s.plain, format!("Hill 2x2 key [{} {}; {} {}]", m[0], m[1], m[2], m[3]), sc);
            }
        }
        Step::Playfair => {
            let s = polygraphic::solve_playfair(ctx.lm, ctx.dense_for(proxy), text, 200_000, ctx.opt.restarts.min(8), 1);
            let sc = ctx.per_letter(proxy, &s.plain);
            add(s.plain, format!("Playfair square {}", polygraphic::square_string(&s.square)), sc);
        }
        Step::Bifid => {
            let periods = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
            for s in polygraphic::solve_bifid(ctx.lm, ctx.dense_for(proxy), text, &periods, 100_000, 4, 1).into_iter().take(beam) {
                let sc = ctx.per_letter(proxy, &s.plain);
                add(s.plain, format!("Bifid period {} square {}", s.period, polygraphic::square_string(&s.square)), sc);
            }
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.dedup_by(|a, b| a.text == b.text);
    out.truncate(beam);
    out
}

/// Undo `steps` (outermost first) on `cipher`, returning the best end results by full
/// language-model score. A beam of `opt.beam` candidates is kept between steps.
pub fn solve_chain(lm: &LangModel, cipher: &[u8], steps: &[Step], opt: &ChainOptions) -> Result<Vec<ChainResult>, String> {
    let proxies = plan(steps)?;
    let n = if cipher.len() < 150 { 5 } else { 4 }.min(lm.order() + 1);
    let ctx = Ctx { lm, quad: lm.dense(n), uni: lm.dense(1), opt };
    let mut beam = vec![Cand { text: cipher.to_vec(), path: vec![], score: 0.0 }];
    for (&step, &proxy) in steps.iter().zip(&proxies) {
        let mut next: Vec<Cand> = Vec::new();
        for c in &beam {
            for mut r in run_step(&ctx, step, proxy, &c.text) {
                let mut path = c.path.clone();
                path.append(&mut r.path);
                next.push(Cand { text: r.text, path, score: r.score });
            }
        }
        next.sort_by(|a, b| b.score.total_cmp(&a.score));
        next.truncate(opt.beam.max(1));
        beam = next;
    }
    let mut res: Vec<ChainResult> = beam
        .into_iter()
        .map(|c| ChainResult { per_letter: lm.score_per_letter(&c.text), path: c.path, plain: c.text })
        .collect();
    res.sort_by(|a, b| b.per_letter.total_cmp(&a.per_letter));
    Ok(res)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::encrypt;
    use crate::testutil::{model, sample};
    use crate::transpo::rail_fence_encrypt;

    fn rail(p: &[u8], rails: usize, off: usize) -> Vec<u8> {
        rail_fence_encrypt(p, rails, off)
    }

    fn columnar(p: &[u8], order: &[usize]) -> Vec<u8> {
        let k = order.len();
        let mut out = Vec::new();
        for &col in order {
            let mut i = col;
            while i < p.len() {
                out.push(p[i]);
                i += k;
            }
        }
        out
    }

    fn accuracy(sol: &ChainResult, p: &[u8]) -> f64 {
        sol.plain.iter().zip(p).filter(|(a, b)| a == b).count() as f64 / p.len() as f64
    }

    #[test]
    fn plan_rules() {
        use PeriodicKind::*;
        assert_eq!(plan(&[Step::Rail, Step::Subst]).unwrap(), vec![Proxy::Pattern, Proxy::Full]);
        assert_eq!(plan(&[Step::Periodic(Vigenere), Step::Columnar]).unwrap(), vec![Proxy::Unigram, Proxy::Full]);
        assert!(plan(&[Step::Rail, Step::Columnar]).is_err());
        assert!(plan(&[Step::Subst, Step::Subst]).is_err());
        assert!(plan(&[Step::Playfair, Step::Rail]).is_err());
        assert!(plan(&[]).is_err());
    }

    #[test]
    fn rail_over_substitution() {
        let lm = model();
        let p = sample(100_000, 300);
        let c = rail(&subst::substitution_encrypt(&p, "QWERTYUIOPASDFGHJKLZXCVBNM"), 4, 1);
        let r = &solve_chain(lm, &c, &[Step::Rail, Step::Subst], &ChainOptions::default()).unwrap()[0];
        assert!(accuracy(r, &p) > 0.95, "{:?} {}", r.path, accuracy(r, &p));
    }

    #[test]
    fn columnar_over_affine() {
        let lm = model();
        let p = sample(110_000, 300);
        let c = columnar(&subst::affine_encrypt(&p, 5, 9), &[2, 0, 4, 1, 3]);
        let r = &solve_chain(lm, &c, &[Step::Columnar, Step::Affine], &ChainOptions::default()).unwrap()[0];
        assert!(accuracy(r, &p) > 0.95, "{:?} {}", r.path, accuracy(r, &p));
    }

    #[test]
    fn vigenere_over_rail() {
        let lm = model();
        let p = sample(120_000, 400);
        let c = encrypt(&Mode::Vigenere, &rail(&p, 3, 0), &[10, 4, 24, 22, 14]);
        let steps = [Step::Periodic(PeriodicKind::Vigenere), Step::Rail];
        let r = &solve_chain(lm, &c, &steps, &ChainOptions::default()).unwrap()[0];
        assert!(accuracy(r, &p) > 0.95, "{:?} {}", r.path, accuracy(r, &p));
    }
}
