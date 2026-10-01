//! Known-source key search for running key ciphers.
//!
//! Running keys are usually passages from existing texts (books, lyrics, speeches).
//! Given candidate source texts, we slide each one along the ciphertext and score
//! the resulting plaintext. Every alignment is first screened cheaply with a dense
//! quadgram table (best window of `window` letters), then the best alignments are
//! rescored with the full language model.
//!
//! Partial matches still surface: if the key is a variant of the source text (a
//! different word here and there), the windows where it agrees still decrypt to
//! English, so the alignment ranks highly and the plaintext is mostly readable.

use crate::lm::{DenseNgram, LangModel};
use crate::text::{dec, scrub, strip_gutenberg};
use std::cmp::Ordering as CmpOrd;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct Source {
    pub name: String,
    pub letters: Vec<u8>,
}

/// Load source texts from files and/or directories (non-recursive). Gutenberg
/// boilerplate is stripped.
pub fn load_sources(paths: &[PathBuf]) -> std::io::Result<Vec<Source>> {
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file()).collect();
            v.sort();
            files.extend(v);
        } else if p.is_file() {
            files.push(p.clone());
        }
    }
    let mut out = Vec::new();
    for f in files {
        let bytes = std::fs::read(&f)?;
        let text = String::from_utf8_lossy(&bytes);
        let letters = scrub(strip_gutenberg(&text));
        if !letters.is_empty() {
            out.push(Source { name: display_name(&f), letters });
        }
    }
    Ok(out)
}

fn display_name(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
}

/// One candidate alignment of a source text against the ciphertext.
#[derive(Clone, Debug)]
pub struct KnownHit {
    pub source: String,
    /// Letter offset in the source aligned with cipher position 0 (may be negative).
    pub offset: i64,
    /// Cipher positions covered by the source (`start..end`).
    pub start: usize,
    pub end: usize,
    /// Source letters used as the stream for `start..end`.
    pub key: Vec<u8>,
    /// The other stream (`cipher - key`) for `start..end`.
    pub plain: Vec<u8>,
    /// Best window mean log-prob per letter under the full model.
    pub window_score: f32,
    /// Mean log-prob per letter of `plain` under the full model.
    pub score: f32,
    /// Fraction of covered positions lying in windows that read as English.
    pub coverage: f32,
}

#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    source: u32,
    offset: i64,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrd> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    // Reverse: BinaryHeap becomes a min-heap on score.
    fn cmp(&self, o: &Self) -> CmpOrd {
        o.score.total_cmp(&self.score)
    }
}

pub struct KnownOptions {
    /// Screening window (letters). Clamped to the cipher length.
    pub window: usize,
    /// Alignments kept after screening and rescored with the full model.
    pub rescore: usize,
    /// Hits returned.
    pub results: usize,
    pub threads: usize,
}

impl Default for KnownOptions {
    fn default() -> Self {
        KnownOptions { window: 24, rescore: 300, results: 20, threads: 0 }
    }
}

/// Search all alignments of all sources. `progress` receives (done, total) alignment counts.
pub fn search(
    lm: &LangModel,
    quad: &DenseNgram,
    cipher: &[u8],
    sources: &[Source],
    opts: &KnownOptions,
    progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    cancel: Option<&AtomicBool>,
) -> Vec<KnownHit> {
    let n = cipher.len();
    let qn = quad.n;
    if n < qn {
        return vec![];
    }
    let w = opts.window.clamp(qn, n);
    let min_cover = w as i64; // a source must cover at least one window
    let threads = if opts.threads == 0 { std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1) } else { opts.threads };

    // Work items: (source index, alignment range).
    let mut items: Vec<(usize, i64, i64)> = Vec::new();
    let mut total = 0usize;
    for (si, s) in sources.iter().enumerate() {
        let lo = -(n as i64 - min_cover);
        let hi = s.letters.len() as i64 - min_cover; // inclusive
        if hi < lo {
            continue;
        }
        let mut a = lo;
        while a <= hi {
            let b = (a + 200_000).min(hi + 1);
            items.push((si, a, b));
            total += (b - a) as usize;
            a = b;
        }
    }
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let keep = opts.rescore.max(opts.results);

    let heaps: Vec<BinaryHeap<Cand>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(keep + 1);
                    let mut ws = vec![0f32; n];
                    let mut plain = vec![0u8; n];
                    loop {
                        if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= items.len() {
                            break;
                        }
                        let (si, a0, a1) = items[i];
                        let src = &sources[si].letters;
                        let slen = src.len() as i64;
                        for a in a0..a1 {
                            let start = (-a).max(0) as usize;
                            let end = (slen - a).min(n as i64) as usize;
                            if end < start + w {
                                continue;
                            }
                            for j in start..end {
                                plain[j] = dec(cipher[j], src[(a + j as i64) as usize]);
                            }
                            // Quadgram score for each window end, then best sliding window.
                            for j in start + qn - 1..end {
                                ws[j] = quad.window(&plain, j);
                            }
                            let first = start + qn - 1;
                            let span = w - (qn - 1);
                            let mut sum: f32 = ws[first..first + span].iter().sum();
                            let mut best = sum;
                            for j in first + span..end {
                                sum += ws[j] - ws[j - span];
                                if sum > best {
                                    best = sum;
                                }
                            }
                            let score = best / span as f32;
                            if heap.len() < keep {
                                heap.push(Cand { score, source: si as u32, offset: a });
                            } else if score > heap.peek().unwrap().score {
                                heap.pop();
                                heap.push(Cand { score, source: si as u32, offset: a });
                            }
                        }
                        let d = done.fetch_add((a1 - a0) as usize, Ordering::Relaxed) + (a1 - a0) as usize;
                        if let Some(p) = progress {
                            p(d, total);
                        }
                    }
                    heap
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut cands: Vec<Cand> = heaps.into_iter().flat_map(|h| h.into_vec()).collect();
    cands.sort_by(|x, y| y.score.total_cmp(&x.score));
    cands.truncate(keep);

    // Rescore with the full model.
    let mut hits: Vec<KnownHit> = cands.iter().map(|c| rescore(lm, cipher, &sources[c.source as usize], c.offset, w)).collect();
    hits.sort_by(|x, y| y.window_score.total_cmp(&x.window_score).then(y.coverage.total_cmp(&x.coverage)));
    hits.truncate(opts.results);
    hits
}

/// Threshold (nats/letter, full model) above which a window "reads as English".
const GOOD_WINDOW: f32 = -2.4;
const COVER_WINDOW: usize = 12;

fn rescore(lm: &LangModel, cipher: &[u8], src: &Source, a: i64, w: usize) -> KnownHit {
    let n = cipher.len();
    let slen = src.letters.len() as i64;
    let start = (-a).max(0) as usize;
    let end = (slen - a).min(n as i64) as usize;
    let key: Vec<u8> = (start..end).map(|j| src.letters[(a + j as i64) as usize]).collect();
    let plain: Vec<u8> = (start..end).map(|j| dec(cipher[j], key[j - start])).collect();
    let len = plain.len();
    let window_score = (0..=len.saturating_sub(w)).map(|s| lm.score_per_letter(&plain[s..s + w.min(len)])).fold(f32::NEG_INFINITY, f32::max);
    let cw = COVER_WINDOW.min(len);
    let mut good = vec![false; len];
    for s in 0..=len - cw {
        if lm.score_per_letter(&plain[s..s + cw]) > GOOD_WINDOW {
            good[s..s + cw].iter_mut().for_each(|g| *g = true);
        }
    }
    let coverage = good.iter().filter(|&&g| g).count() as f32 / len.max(1) as f32;
    KnownHit {
        source: src.name.clone(),
        offset: a,
        start,
        end,
        score: lm.score_per_letter(&plain),
        key,
        plain,
        window_score,
        coverage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::enc;

    #[test]
    fn finds_embedded_key() {
        let book = "it was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness \
                    we hold these truths to be self evident that all men are created equal and endowed with rights ";
        let (lm, _) = LangModel::train(&[scrub(&book.repeat(30))], 4);
        let quad = lm.dense(4);
        let src = Source { name: "book".into(), letters: scrub(book) };
        let key = &src.letters[40..80];
        let plain = scrub("wehold these truths to be self evident th");
        let c: Vec<u8> = plain.iter().zip(key).map(|(&p, &k)| enc(p, k)).collect();
        let hits = search(&lm, &quad, &c, &[src], &KnownOptions { threads: 1, ..Default::default() }, None, None);
        // The plaintext also occurs in the book, so the mirrored alignment scores equally.
        let hit = hits.iter().take(2).find(|h| h.offset == 40).expect("key alignment not in top 2");
        assert_eq!(hit.plain, plain[..c.len()].to_vec());
    }
}
