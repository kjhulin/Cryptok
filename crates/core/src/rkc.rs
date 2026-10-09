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
use crate::map::FxHashMap;
use crate::words::{StreamState, WordModel, WordTrie};
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
    /// Weight of the word-model score when a word trie is supplied (see [`solve_words`]).
    pub word_weight: f32,
    /// Number of trailing key letters that identify a state for merging
    /// (0 = the model order, or 10 when a word model is used). Must be at least the order.
    pub merge_len: usize,
    /// Soft cap on the bytes of back-pointers kept for traceback (0 = unlimited). When the
    /// search would exceed it, the beam is narrowed instead of the process growing without
    /// bound. Back-pointers cost 5 bytes per hypothesis per step, but entries that no
    /// surviving hypothesis descends from are released as the search goes (see
    /// [`solve_words`]), so the cap rarely binds.
    pub max_back_bytes: usize,
    /// Keep every back-pointer instead of releasing unreachable ones. Gives identical results
    /// (used by the tests); only worth enabling to debug.
    pub keep_all_backpointers: bool,
}

impl Default for RkcOptions {
    fn default() -> Self {
        RkcOptions { beam: 100_000, results: 10, key_hints: vec![], plain_hints: vec![], threads: 0, word_weight: 1.0, merge_len: 0, max_back_bytes: 0, keep_all_backpointers: false }
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
    /// Longer key history used to identify the state for merging.
    khist: u64,
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
    on_step: Option<&mut dyn FnMut(&StepInfo)>,
    cancel: Option<&AtomicBool>,
) -> Vec<RkcSolution> {
    solve_words(lm, None, cipher, opts, on_step, cancel)
}

/// Like [`solve`], but when `words` is given each hypothesis is also scored by the best
/// word segmentation of its key and plaintext so far (weighted by `opts.word_weight`),
/// so the beam prefers paths that spell words.
pub fn solve_words(
    lm: &LangModel,
    words: Option<&WordTrie>,
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

    let merge_len = if opts.merge_len > 0 { opts.merge_len } else if words.is_some() { 10 } else { k_order }.clamp(k_order, 13);
    let merge_mod = 26u64.pow(merge_len as u32);
    let ww = opts.word_weight;
    let mut hyps = vec![Hyp { kctx: 0, khist: 0, pctx: 0, score: 0.0, diverged: false }];
    // Word-segmentation state of (key, plaintext) per hypothesis; empty without a word model.
    let mut wst: Vec<(StreamState, StreamState)> = if words.is_some() { vec![(StreamState::new(), StreamState::new())] } else { vec![] };
    // Traceback memory is the big cost: 5 bytes x beam x steps. But the surviving hypotheses
    // descend from very few ancestors a few steps back (typically two: the key/plaintext
    // mirror pair), so back-pointer entries nothing alive descends from are dropped every
    // few steps (`release_unreachable`). Exact: such entries can never appear in a traceback.
    let mut backs: Vec<Back> = Vec::new();
    let mut live_bytes = 0usize;
    let mut beam = beam;
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
                    let mut s = h.score + deq[krow[k as usize] as usize] + deq[prow[p as usize] as usize];
                    if let Some(wt) = words {
                        let (ks, ps) = &wst[base + j];
                        s += ww * (wt.gain(ks, k) + wt.gain(ps, p));
                    }
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
                            // Only the best `2 * beam` of all candidates survive, so each part
                            // can already discard everything below its own best `2 * beam`:
                            // the merged set is 3x smaller and the later selection cheaper.
                            let keep = (beam * 2).min(v.len());
                            if v.len() > keep {
                                v.select_nth_unstable_by(keep, |a, b| b.score.total_cmp(&a.score));
                                v.truncate(keep);
                                v.shrink_to_fit();
                            }
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
        let mut next_wst: Vec<(StreamState, StreamState)> = Vec::new();
        let mut back = Back { parent: Vec::with_capacity(next.capacity()), letter: Vec::with_capacity(next.capacity()) };
        for cd in &cands {
            let h = &hyps[cd.parent as usize];
            let k = cd.letter;
            let p = dec(c, k);
            let kctx = (h.kctx * 26 + k as u64) % modk;
            let diverged = h.diverged || k < p;
            let khist = (h.khist * 26 + k as u64) % merge_mod;
            let state = khist * 2 + diverged as u64;
            if !seen.insert_if_absent(state, 0) {
                continue; // a better hypothesis already owns this state
            }
            if let Some(wt) = words {
                let (ks, ps) = &wst[cd.parent as usize];
                next_wst.push((wt.push(ks, k), wt.push(ps, p)));
            }
            next.push(Hyp { kctx, khist, pctx: (h.pctx * 26 + p as u64) % modk, score: cd.score, diverged });
            back.parent.push(cd.parent);
            back.letter.push(k);
            if next.len() == beam {
                break;
            }
        }
        hyps = next;
        wst = next_wst;
        live_bytes += back.parent.len() * 5;
        backs.push(back);
        if !opts.keep_all_backpointers && backs.len() % RELEASE_EVERY == 0 {
            live_bytes -= release_unreachable(&mut backs);
        }
        if opts.max_back_bytes > 0 && live_bytes > opts.max_back_bytes && beam > MIN_BEAM {
            beam = (beam * 4 / 5).max(MIN_BEAM); // narrow rather than grow without bound
        }

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

/// Steps between releases of unreachable back-pointers.
const RELEASE_EVERY: usize = 16;
const MIN_BEAM: usize = 1000;

/// Drop every back-pointer entry that no hypothesis of the newest step descends from, and
/// renumber the parents that remain. The newest step is untouched, so the caller's hypothesis
/// indices stay valid. Returns the bytes released.
fn release_unreachable(backs: &mut [Back]) -> usize {
    let live = backs.len();
    if live < 2 {
        return 0;
    }
    // Mark, newest to oldest: which entries of each step are ancestors of a current hypothesis.
    let mut keep: Vec<Vec<bool>> = backs.iter().map(|b| vec![false; b.parent.len()]).collect();
    keep[live - 1].fill(true);
    for j in (1..live).rev() {
        let (older, newer) = keep.split_at_mut(j);
        for (i, _) in newer[0].iter().enumerate().filter(|(_, &k)| k) {
            older[j - 1][backs[j].parent[i] as usize] = true;
        }
    }
    // Compact, oldest to newest, remapping each step's parents into the compacted step before.
    let before: usize = backs.iter().map(|b| b.parent.len()).sum();
    let mut remap_prev: Vec<u32> = Vec::new();
    for j in 0..live {
        let b = &mut backs[j];
        let kept = keep[j].iter().filter(|&&k| k).count();
        let mut parent = Vec::with_capacity(kept);
        let mut letter = Vec::with_capacity(kept);
        let mut remap = vec![u32::MAX; b.parent.len()];
        for i in 0..b.parent.len() {
            if keep[j][i] {
                remap[i] = parent.len() as u32;
                parent.push(if j == 0 { b.parent[i] } else { remap_prev[b.parent[i] as usize] });
                letter.push(b.letter[i]);
            }
        }
        *b = Back { parent, letter };
        remap_prev = remap;
    }
    let after: usize = backs.iter().map(|b| b.parent.len()).sum();
    (before - after) * 5
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

/// Letters on each side of a candidate swap point that [`untangle`] rescores.
const UNTANGLE_WINDOW: usize = 24;
/// A point is a candidate swap point when swapping there costs less than this (nats) locally.
const UNTANGLE_MAX_COST: f32 = 6.0;
/// A stretch is only swapped when that beats leaving it by this much (nats): keeps the pass
/// from disturbing rows that are already coherent.
const UNTANGLE_MARGIN: f64 = 8.0;
/// Weight of the generic word frequencies in each row's word cache (pseudo-count).
const UNTANGLE_PRIOR: f64 = 20.0;
/// Shorter ciphers are left alone: they rarely swap, and there is too little text to judge.
pub const UNTANGLE_MIN_LEN: usize = 200;

/// Log-probability of `s[from..to]`, each letter scored with up to `order` letters of context from `s`.
fn lm_span(lm: &LangModel, s: &[u8], from: usize, to: usize) -> f32 {
    let deq = lm.deq();
    let order = lm.order();
    (from..to)
        .map(|j| {
            let start = j.saturating_sub(order);
            let ctx = s[start..j].iter().fold(0u64, |c, &x| c * 26 + x as u64);
            deq[lm.row(ctx, j - start)[s[j] as usize] as usize]
        })
        .sum()
}

fn word_score(words: &WordTrie, s: &[u8]) -> f32 {
    s.iter().fold(StreamState::new(), |st, &x| words.push(&st, x)).score()
}

/// Change in score from swapping the rows from position `i` on, judged on a window around
/// `i` by the same models the search uses (so it is almost never positive).
fn swap_gain(lm: &LangModel, words: Option<&WordTrie>, ww: f32, a: &[u8], b: &[u8], i: usize) -> f32 {
    let lo = i.saturating_sub(UNTANGLE_WINDOW);
    let hi = (i + UNTANGLE_WINDOW).min(a.len());
    let swapped = |x: &[u8], y: &[u8]| -> Vec<u8> { x[lo..i].iter().chain(&y[i..hi]).copied().collect() };
    let (sa, sb) = (swapped(a, b), swapped(b, a));
    let (ca, cb) = (&a[lo..hi], &b[lo..hi]);
    let (f, t) = (i - lo, (i - lo + lm.order()).min(hi - lo));
    let mut g = lm_span(lm, &sa, f, t) + lm_span(lm, &sb, f, t) - lm_span(lm, ca, f, t) - lm_span(lm, cb, f, t);
    if let Some(w) = words {
        g += ww * (word_score(w, &sa) + word_score(w, &sb) - word_score(w, ca) - word_score(w, cb));
    }
    g
}

type Bag = FxHashMap<u32, u32>;

/// Log-likelihood ratio of the words `bag` under a row's word cache (`row`, `total` words)
/// against generic word frequencies: positive when the row has used these words more than
/// chance, as a text repeating its own names and topic does.
fn cache_llr(bag: &Bag, row: &Bag, total: u32, uni: &FxHashMap<u32, f32>) -> f64 {
    let n = total as f64;
    bag.iter()
        .map(|(w, &c)| {
            let p = (uni[w] as f64).exp();
            let in_row = row.get(w).copied().unwrap_or(0);
            c as f64 * ((in_row as f64 + UNTANGLE_PRIOR * p) / (n + UNTANGLE_PRIOR) / p).ln()
        })
        .sum()
}

/// Key and plaintext score alike, so over a long cipher the search can hand a stretch of the
/// key to the plaintext row and back wherever both texts continue plausibly (often at a word
/// break): every letter pair is right, but each row reads as two texts spliced together.
/// The models the search uses cannot see this (they prefer the splice), so this pass looks at
/// the whole text instead: it cuts the rows at every point where swapping is locally cheap and
/// assigns each stretch to the row whose other stretches use the same words (names, topic,
/// narrative voice), paying the local cost of every swap it makes. Letter pairs never change,
/// only which row each letter is shown in. Returns the number of stretches swapped.
pub fn untangle(lm: &LangModel, wm: &WordModel, trie: Option<&WordTrie>, word_weight: f32, sol: &mut RkcSolution) -> usize {
    let n = sol.key.len();
    if n < UNTANGLE_MIN_LEN {
        return 0;
    }
    let (a, b) = (&sol.key, &sol.plain);
    // Candidate cut points: locally cheap swaps, at most one per 4 letters (the cheapest).
    let mut cuts: Vec<(usize, f32)> = Vec::new();
    for i in 1..n {
        let g = swap_gain(lm, trie, word_weight, a, b, i);
        if g <= -UNTANGLE_MAX_COST {
            continue;
        }
        match cuts.last_mut() {
            Some(last) if i - last.0 <= 3 => {
                if g > last.1 {
                    *last = (i, g);
                }
            }
            _ => cuts.push((i, g)),
        }
    }
    if cuts.is_empty() {
        return 0;
    }
    let bounds: Vec<usize> = std::iter::once(0).chain(cuts.iter().map(|c| c.0)).chain(std::iter::once(n)).collect();
    let m = bounds.len() - 1;
    let mut uni: FxHashMap<u32, f32> = FxHashMap::default();
    let bags: Vec<[Bag; 2]> = (0..m)
        .map(|j| {
            let (lo, hi) = (bounds[j], bounds[j + 1]);
            [&a[lo..hi], &b[lo..hi]].map(|s| {
                let mut bag = Bag::default();
                for (w, lp) in wm.known_words(s) {
                    *bag.entry(w).or_default() += 1;
                    uni.insert(w, lp);
                }
                bag
            })
        })
        .collect();
    // flip[j]: stretch j is shown with its rows swapped. Row r holds bags[j][r ^ flip[j]].
    let mut flip = vec![false; m];
    let mut rows: [Bag; 2] = [Bag::default(), Bag::default()];
    for bag in &bags {
        for r in 0..2 {
            for (&w, &c) in &bag[r] {
                *rows[r].entry(w).or_default() += c;
            }
        }
    }
    // Local cost of the swaps at the two ends of stretches lo..hi when they are shown flipped
    // by `toggle` relative to now (swaps inside the range cost the same either way).
    let ends_cost = |flip: &[bool], lo: usize, hi: usize, toggled: bool| -> f64 {
        let mut s = 0.0;
        let f = |j: usize| flip[j] ^ toggled;
        if lo > 0 && f(lo) != flip[lo - 1] {
            s += cuts[lo - 1].1 as f64;
        }
        if hi < m && f(hi - 1) != flip[hi] {
            s += cuts[hi - 1].1 as f64;
        }
        s
    };
    // Moves flip any run of stretches at once: a spliced block of many stretches cannot be
    // repaired one stretch at a time, since each single flip costs two swaps.
    const MAX_RUN: usize = 40;
    for _ in 0..20 {
        let mut best: Option<(f64, usize, usize)> = None;
        for lo in 0..m {
            for hi in lo + 1..=(lo + MAX_RUN).min(m) {
                // Rows without this run, then the run's words as shown now and as flipped.
                let mut without = [rows[0].clone(), rows[1].clone()];
                let mut now = [Bag::default(), Bag::default()];
                let mut flipped = [Bag::default(), Bag::default()];
                for j in lo..hi {
                    for r in 0..2 {
                        for (&w, &c) in &bags[j][r ^ flip[j] as usize] {
                            *without[r].get_mut(&w).unwrap() -= c;
                            *now[r].entry(w).or_default() += c;
                        }
                        for (&w, &c) in &bags[j][r ^ !flip[j] as usize] {
                            *flipped[r].entry(w).or_default() += c;
                        }
                    }
                }
                let total = |r: usize| without[r].values().sum::<u32>();
                let (t0, t1) = (total(0), total(1));
                let score = |bag: &[Bag; 2]| cache_llr(&bag[0], &without[0], t0, &uni) + cache_llr(&bag[1], &without[1], t1, &uni);
                let gain = score(&flipped) - score(&now) + ends_cost(&flip, lo, hi, true) - ends_cost(&flip, lo, hi, false);
                if gain > UNTANGLE_MARGIN && best.map_or(true, |b| gain > b.0) {
                    best = Some((gain, lo, hi));
                }
            }
        }
        let Some((_, lo, hi)) = best else { break };
        for j in lo..hi {
            for r in 0..2 {
                for (&w, &c) in &bags[j][r ^ flip[j] as usize] {
                    *rows[r].get_mut(&w).unwrap() -= c;
                }
                for (&w, &c) in &bags[j][r ^ !flip[j] as usize] {
                    *rows[r].entry(w).or_default() += c;
                }
            }
            flip[j] = !flip[j];
        }
    }
    for j in 0..m {
        if flip[j] {
            let (lo, hi) = (bounds[j], bounds[j + 1]);
            sol.key[lo..hi].swap_with_slice(&mut sol.plain[lo..hi]);
        }
    }
    flip.iter().filter(|&&f| f).count()
}

/// Run [`untangle`] on every solution of a search (unless it had letter hints, which fix
/// which row is the key), then drop solutions that became duplicates of a better one.
pub fn untangle_results(lm: &LangModel, wm: &WordModel, trie: Option<&WordTrie>, opts: &RkcOptions, sols: &mut Vec<RkcSolution>) {
    let hinted = opts.key_hints.iter().chain(&opts.plain_hints).any(|h| h.is_some());
    if hinted || sols.first().map_or(true, |s| s.key.len() < UNTANGLE_MIN_LEN) {
        return;
    }
    for s in sols.iter_mut() {
        untangle(lm, wm, trie, opts.word_weight, s);
    }
    let mut kept: Vec<RkcSolution> = Vec::with_capacity(sols.len());
    for s in sols.drain(..) {
        if !kept.iter().any(|k| (k.key == s.key && k.plain == s.plain) || (k.key == s.plain && k.plain == s.key)) {
            kept.push(s);
        }
    }
    *sols = kept;
}

/// Fraction of positions where the better-matching output row equals the true key or plaintext
/// (the stricter measure: a row that switches between the two texts loses the switched part).
pub fn row_accuracy(sol: &RkcSolution, key: &[u8], plain: &[u8]) -> f64 {
    let n = key.len().min(sol.key.len());
    if n == 0 {
        return 0.0;
    }
    let same = |x: &[u8], y: &[u8]| x[..n].iter().zip(&y[..n]).filter(|(a, b)| a == b).count();
    let best = same(&sol.key, key).max(same(&sol.key, plain)).max(same(&sol.plain, key)).max(same(&sol.plain, plain));
    best as f64 / key.len() as f64
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
    fn releasing_unreachable_backpointers_changes_nothing() {
        use crate::testutil::{model, sample};
        let lm = model();
        let p = sample(200_000, 400);
        let k = sample(300_000, 400);
        let c: Vec<u8> = p.iter().zip(&k).map(|(&a, &b)| enc(a, b)).collect();
        let base = RkcOptions { beam: 3000, results: 3, threads: 1, ..Default::default() };
        let full = solve(lm, &c, &RkcOptions { keep_all_backpointers: true, ..base.clone() }, None, None);
        let lean = solve(lm, &c, &base, None, None);
        assert_eq!(full, lean, "releasing unreachable back-pointers must not change any result");
        assert!(pair_accuracy(&lean[0], &k, &p) > 0.5);
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

    #[test]
    fn untangle_regroups_stretches_by_vocabulary() {
        use crate::testutil::model;
        // Two "topics" with disjoint vocabularies; the rows are spliced in the middle.
        let topic = |words: &[&str], seed: usize, n: usize| -> String { (0..n).map(|i| words[(i * 7 + seed * 13 + i * i) % words.len()]).collect::<Vec<_>>().join(" ") };
        let a_words = ["the", "whale", "harpoon", "captain", "ship", "ocean", "mast", "sailor", "wave", "deck", "storm", "anchor"];
        let b_words = ["the", "garden", "letter", "morning", "parlour", "carriage", "lady", "dinner", "walk", "sister", "ball", "visit"];
        let (ta, tb) = (topic(&a_words, 1, 400), topic(&b_words, 2, 400));
        let wm = WordModel::from_texts(&[ta.clone(), tb.clone()], 2, false);
        let (key, plain) = (crate::text::scrub(&ta)[..600].to_vec(), crate::text::scrub(&tb)[..600].to_vec());
        let truth = RkcSolution { key: key.clone(), plain: plain.clone(), score: 0.0 };
        let mut sol = truth.clone();
        // Swap two stretches at word boundaries.
        let edge = |s: &[u8], near: usize| (near..near + 40).find(|&i| wm.segment(&s[..i]).lengths.iter().sum::<usize>() == i && wm.segment(&s[..i + 1]).oov.last() == Some(&false)).unwrap_or(near);
        let (x, y) = (edge(&key, 150), edge(&key, 380));
        sol.key[x..y].swap_with_slice(&mut sol.plain[x..y]);
        let before = row_accuracy(&sol, &key, &plain);
        eprintln!("DBG x={x} y={y} gains {} {}", swap_gain(model(), None, 0.0, &sol.key, &sol.plain, x), swap_gain(model(), None, 0.0, &sol.key, &sol.plain, y));
        let nsw = untangle(model(), &wm, None, 0.0, &mut sol);
        eprintln!("DBG swaps {nsw}");
        assert_eq!(pair_accuracy(&sol, &key, &plain), 1.0, "letter pairs must not change");
        assert!(row_accuracy(&sol, &key, &plain) > before + 0.1, "{before} -> {}", row_accuracy(&sol, &key, &plain));
        let mut short = RkcSolution { key: key[..100].to_vec(), plain: plain[..100].to_vec(), score: 0.0 };
        assert_eq!(untangle(model(), &wm, None, 0.0, &mut short), 0, "short ciphers are left alone");
    }
}
