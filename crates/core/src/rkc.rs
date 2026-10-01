//! Running key cipher solver.
//!
//! Ciphertext `C = P + K (mod 26)` where both plaintext `P` and key `K` are
//! natural-language text. We search for the pair maximising
//! `log P_lm(K) + log P_lm(P)` with a Viterbi beam search:
//!
//! * **State merging.** The future score of a hypothesis depends only on its last
//!   `order` key letters (the matching plaintext letters are determined by the
//!   ciphertext), so hypotheses sharing that suffix are merged, keeping the best.
//!   This is what keeps the beam from filling with near-duplicates.
//! * **Mirror pruning.** `(K, P)` and `(P, K)` score identically. Without hints we
//!   only keep hypotheses whose key is lexicographically `<=` the plaintext.
//! * **Back-pointers** instead of copying key prefixes, and top-k by selection.
//! * Candidate expansion runs on all cores.

use crate::lm::LangModel;
use crate::map::U64Map;
use crate::text::dec;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct RkcOptions {
    /// Number of hypotheses kept after each step.
    pub beam: usize,
    /// Number of final solutions returned.
    pub results: usize,
    /// Optional fixed key letter per ciphertext letter position.
    pub key_hints: Vec<Option<u8>>,
    /// Optional fixed plaintext letter per ciphertext letter position.
    pub plain_hints: Vec<Option<u8>>,
    /// Worker threads (0 = all available cores).
    pub threads: usize,
}

impl Default for RkcOptions {
    fn default() -> Self {
        RkcOptions { beam: 100_000, results: 10, key_hints: vec![], plain_hints: vec![], threads: 0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RkcSolution {
    pub key: Vec<u8>,
    pub plain: Vec<u8>,
    /// Total log-probability (nats) of key + plaintext.
    pub score: f32,
}

impl RkcSolution {
    /// Mean log-probability per letter of each stream (comparable across lengths).
    pub fn per_letter(&self) -> f32 {
        if self.key.is_empty() {
            0.0
        } else {
            self.score / (2.0 * self.key.len() as f32)
        }
    }
}

/// Progress information passed to the step callback.
pub struct StepInfo<'a> {
    pub step: usize,
    pub total: usize,
    pub beam_size: usize,
    /// Current best partial solutions (prefixes), best first.
    pub best: &'a [RkcSolution],
}

#[derive(Clone, Copy)]
struct Hyp {
    kctx: u64,
    pctx: u64,
    score: f32,
    diverged: bool,
}

#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    parent: u32,
    letter: u8,
}

struct Back {
    parent: Vec<u32>,
    letter: Vec<u8>,
}

/// Allowed key letters at each position, derived from hints.
fn allowed_letters(cipher: &[u8], opts: &RkcOptions) -> Vec<Option<u8>> {
    (0..cipher.len())
        .map(|i| {
            let kh = opts.key_hints.get(i).copied().flatten();
            let ph = opts.plain_hints.get(i).copied().flatten().map(|p| dec(cipher[i], p));
            kh.or(ph)
        })
        .collect()
}

/// Solve a running key cipher. `cipher` holds letters in `0..26`.
///
/// `on_step` is called after every position with the current best partial solutions.
/// Setting `cancel` stops the search early and returns the best prefixes found so far.
pub fn solve(
    lm: &LangModel,
    cipher: &[u8],
    opts: &RkcOptions,
    mut on_step: Option<&mut dyn FnMut(&StepInfo)>,
    cancel: Option<&AtomicBool>,
) -> Vec<RkcSolution> {
    let n = cipher.len();
    if n == 0 {
        return vec![];
    }
    let k_order = lm.order();
    let modk = lm.pow(k_order);
    let deq = lm.deq();
    let allowed = allowed_letters(cipher, opts);
    let prune_mirror = allowed.iter().all(|a| a.is_none());
    let beam = opts.beam.max(1);
    let threads = if opts.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        opts.threads
    };

    let mut hyps = vec![Hyp { kctx: 0, pctx: 0, score: 0.0, diverged: false }];
    let mut backs: Vec<Back> = Vec::with_capacity(n);
    let mut seen = U64Map::with_capacity(beam * 2);
    let mut cands: Vec<Cand> = Vec::new();

    let mut done = n;
    for t in 0..n {
        if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
            done = t;
            break;
        }
        let c = cipher[t];
        let ctx_len = t.min(k_order);
        let fixed = allowed[t];

        // --- Expand every hypothesis by every allowed key letter (parallel). ---
        let expand = |chunk: &[Hyp], base: usize, out: &mut Vec<Cand>| {
            for (j, h) in chunk.iter().enumerate() {
                let krow = lm.row(h.kctx, ctx_len);
                let prow = lm.row(h.pctx, ctx_len);
                let parent = (base + j) as u32;
                let mut push = |k: u8| {
                    let p = dec(c, k);
                    if prune_mirror && !h.diverged && k > p {
                        return;
                    }
                    let s = h.score + deq[krow[k as usize] as usize] + deq[prow[p as usize] as usize];
                    out.push(Cand { score: s, parent, letter: k });
                };
                match fixed {
                    Some(k) => push(k),
                    None => (0..26u8).for_each(&mut push),
                }
            }
        };
        cands.clear();
        if threads <= 1 || hyps.len() < 4096 {
            expand(&hyps, 0, &mut cands);
        } else {
            let chunk = hyps.len().div_ceil(threads);
            let parts: Vec<Vec<Cand>> = std::thread::scope(|s| {
                let handles: Vec<_> = hyps
                    .chunks(chunk)
                    .enumerate()
                    .map(|(ci, ch)| {
                        let expand = &expand;
                        s.spawn(move || {
                            let mut v = Vec::with_capacity(ch.len() * 26);
                            expand(ch, ci * chunk, &mut v);
                            v
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for p in parts {
                cands.extend_from_slice(&p);
            }
        }

        // --- Keep the best candidates, merging identical states. ---
        let keep = (beam * 2).min(cands.len());
        if cands.len() > keep {
            cands.select_nth_unstable_by(keep, |a, b| b.score.total_cmp(&a.score));
            cands.truncate(keep);
        }
        cands.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

        seen.clear();
        let mut next = Vec::with_capacity(beam.min(cands.len()));
        let mut back = Back { parent: Vec::with_capacity(next.capacity()), letter: Vec::with_capacity(next.capacity()) };
        for cd in &cands {
            let h = &hyps[cd.parent as usize];
            let k = cd.letter;
            let p = dec(c, k);
            let kctx = (h.kctx * 26 + k as u64) % modk;
            let diverged = h.diverged || k < p;
            let state = kctx * 2 + diverged as u64;
            if !seen.insert_if_absent(state, 0) {
                continue; // a better hypothesis already owns this state
            }
            next.push(Hyp { kctx, pctx: (h.pctx * 26 + p as u64) % modk, score: cd.score, diverged });
            back.parent.push(cd.parent);
            back.letter.push(k);
            if next.len() == beam {
                break;
            }
        }
        hyps = next;
        backs.push(back);

        if let Some(cb) = on_step.as_deref_mut() {
            let best = backtrack(cipher, &backs, &hyps, 3);
            cb(&StepInfo { step: t + 1, total: n, beam_size: hyps.len(), best: &best });
        }
        if hyps.is_empty() {
            done = t + 1;
            break;
        }
    }
    let _ = done;
    backtrack(cipher, &backs, &hyps, opts.results)
}

fn backtrack(cipher: &[u8], backs: &[Back], hyps: &[Hyp], count: usize) -> Vec<RkcSolution> {
    let len = backs.len();
    (0..count.min(hyps.len()))
        .map(|i| {
            let mut key = vec![0u8; len];
            let mut idx = i;
            for t in (0..len).rev() {
                key[t] = backs[t].letter[idx];
                idx = backs[t].parent[idx] as usize;
            }
            let plain = key.iter().zip(cipher).map(|(&k, &c)| dec(c, k)).collect();
            RkcSolution { key, plain, score: hyps[i].score }
        })
        .collect()
}

/// Fraction of positions where the recovered (key, plain) pair matches the truth,
/// allowing the two streams to be swapped at any position (they are symmetric).
pub fn pair_accuracy(sol: &RkcSolution, key: &[u8], plain: &[u8]) -> f64 {
    let n = key.len().min(sol.key.len());
    if n == 0 {
        return 0.0;
    }
    let ok = (0..n)
        .filter(|&i| (sol.key[i] == key[i] && sol.plain[i] == plain[i]) || (sol.key[i] == plain[i] && sol.plain[i] == key[i]))
        .count();
    ok as f64 / key.len() as f64
}

/// A crib placed at a cipher position: the other stream there is `cipher - crib`.
#[derive(Clone, Debug)]
pub struct CribHit {
    pub pos: usize,
    pub other: Vec<u8>,
    /// Mean log-prob per letter of `other` (full model, no left context).
    pub score: f32,
}

/// Try a crib (a word believed to be in the key *or* the plaintext — the result is the
/// same) at every position and rank positions by how English the other stream looks.
pub fn crib_search(lm: &LangModel, cipher: &[u8], crib: &[u8]) -> Vec<CribHit> {
    let m = crib.len();
    if m == 0 || m > cipher.len() {
        return vec![];
    }
    let mut hits: Vec<CribHit> = (0..=cipher.len() - m)
        .map(|pos| {
            let other: Vec<u8> = (0..m).map(|i| dec(cipher[pos + i], crib[i])).collect();
            CribHit { pos, score: lm.score_per_letter(&other), other }
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{enc, scrub};

    #[test]
    fn solves_short_cipher_with_tiny_model() {
        let corpus = scrub(&"it was the best of times it was the worst of times we hold these truths to be self evident ".repeat(40));
        let (lm, _) = LangModel::train(&[corpus], 4);
        let p = scrub("wehold");
        let k = scrub("itwast");
        let c: Vec<u8> = p.iter().zip(&k).map(|(&a, &b)| enc(a, b)).collect();
        let opts = RkcOptions { beam: 2000, results: 5, threads: 1, ..Default::default() };
        let sols = solve(&lm, &c, &opts, None, None);
        assert_eq!(sols[0].key.len(), c.len(), "every letter must be decoded");
        assert!(pair_accuracy(&sols[0], &k, &p) > 0.99);
    }

    #[test]
    fn hints_are_respected() {
        let corpus = scrub(&"the cat sat on the mat and the dog ran ".repeat(40));
        let (lm, _) = LangModel::train(&[corpus], 3);
        let c = scrub("QWERTYUIOP");
        let mut opts = RkcOptions { beam: 500, results: 1, threads: 1, ..Default::default() };
        opts.plain_hints = vec![None; 10];
        opts.plain_hints[3] = Some(4); // 'E'
        let s = &solve(&lm, &c, &opts, None, None)[0];
        assert_eq!(s.plain[3], 4);
    }
}
