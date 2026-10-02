//! Word-level model: unigram word frequencies and word-break discovery.
//!
//! The character model never sees spaces, so it cannot tell that `REDSHIRTENGINEER`
//! is three words. [`WordModel::segment`] finds the most likely split of a letter
//! stream (dynamic programming over the vocabulary) and its log-probability, which
//! can be used to display results with spaces and to re-rank candidate solutions.

use crate::map::FxHashMap;
use crate::text::{strip_gutenberg, unscrub};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MAX_WORD: usize = 24;
/// Cost (nats) of an out-of-vocabulary stretch: `OOV_BASE + OOV_PER_LETTER * len`.
const OOV_BASE: f32 = -4.0;
const OOV_PER_LETTER: f32 = -3.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Segmentation {
    /// Log-probability (nats) of the stream under the word model.
    pub score: f32,
    /// Length of each word, in order (sums to the stream length).
    pub lengths: Vec<usize>,
    /// Which words were out of vocabulary.
    pub oov: Vec<bool>,
}

impl Segmentation {
    /// Render the letters with spaces between words, e.g. `FORTUNATE IS THE REDSHIRT`.
    pub fn render(&self, letters: &[u8]) -> String {
        let mut out = String::with_capacity(letters.len() + self.lengths.len());
        let mut at = 0;
        for (i, &l) in self.lengths.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&unscrub(&letters[at..at + l]));
            at += l;
        }
        out
    }
}

pub struct WordModel {
    /// Letter sequence (0..26) -> natural-log probability.
    logp: FxHashMap<Vec<u8>, f32>,
    /// Kept vocabulary with raw counts (what [`WordModel::save`] writes).
    counts: Vec<(Vec<u8>, u32)>,
    max_len: usize,
}

impl WordModel {
    /// Build from raw text files in `dir` (Gutenberg boilerplate stripped), skipping
    /// the file names in `exclude`. Words seen fewer than `min_count` times are dropped.
    pub fn from_corpus(dirs: &[PathBuf], exclude: &[String], min_count: u32) -> io::Result<Self> {
        let mut counts: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let names = crate::text::corpus_files(dirs, exclude)?;
        for p in names {
            let bytes = fs::read(&p)?;
            let text = String::from_utf8_lossy(&bytes);
            count_words(strip_gutenberg(&text), &mut counts);
        }
        Ok(Self::from_counts(counts, min_count))
    }

    pub fn from_counts(counts: FxHashMap<Vec<u8>, u32>, min_count: u32) -> Self {
        let kept: Vec<(Vec<u8>, u32)> = counts
            .into_iter()
            .filter(|(w, c)| {
                *c >= min_count && w.len() <= MAX_WORD && (w.len() > 1 || matches!(w[0], 0 | 8)) // single letters: only A, I
            })
            .collect();
        let total: f64 = kept.iter().map(|(_, c)| *c as f64).sum::<f64>().max(1.0);
        let max_len = kept.iter().map(|(w, _)| w.len()).max().unwrap_or(1);
        let logp = kept.iter().map(|(w, c)| (w.clone(), ((*c as f64) / total).ln() as f32)).collect();
        WordModel { logp, counts: kept, max_len }
    }

    /// Write the vocabulary as `word<TAB>count` lines (plain text, ~1 MB).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut rows: Vec<&(Vec<u8>, u32)> = self.counts.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut out = String::from("#cryptok-words v1\n");
        for (w, c) in rows {
            out.push_str(&unscrub(w).to_lowercase());
            out.push('\t');
            out.push_str(&c.to_string());
            out.push('\n');
        }
        fs::write(path, out)
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();
        if lines.next() != Some("#cryptok-words v1") {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a cryptok word list"));
        }
        let mut counts: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for l in lines {
            if let Some((w, c)) = l.split_once('\t') {
                if let Ok(c) = c.parse::<u32>() {
                    counts.insert(crate::text::scrub(w), c);
                }
            }
        }
        Ok(Self::from_counts(counts, 1))
    }

    pub fn vocab_size(&self) -> usize {
        self.logp.len()
    }

    /// Most likely segmentation of `s` into words (out-of-vocabulary stretches allowed
    /// at a penalty, so names and typos do not break the parse).
    pub fn segment(&self, s: &[u8]) -> Segmentation {
        let n = s.len();
        // best[i]: best score for s[..i]; back[i]: (word length, was_oov).
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut back = vec![(0usize, false); n + 1];
        best[0] = 0.0;
        for i in 1..=n {
            for l in 1..=i.min(self.max_len) {
                let prev = best[i - l];
                if prev == f32::NEG_INFINITY {
                    continue;
                }
                if let Some(&lp) = self.logp.get(&s[i - l..i]) {
                    if prev + lp > best[i] {
                        best[i] = prev + lp;
                        back[i] = (l, false);
                    }
                }
            }
            // Out-of-vocabulary fallback: a single unknown letter extends the previous
            // unknown word or starts a new one (cost is per letter plus a start cost).
            let (pl, poov) = if i >= 1 { back[i - 1] } else { (0, false) };
            let extend = if i >= 2 && poov && best[i - 1] > f32::NEG_INFINITY { best[i - 1] + OOV_PER_LETTER } else { f32::NEG_INFINITY };
            let fresh = best[i - 1] + OOV_BASE + OOV_PER_LETTER;
            let _ = pl;
            let (cand, ext) = if extend > fresh { (extend, true) } else { (fresh, false) };
            if cand > best[i] {
                best[i] = cand;
                back[i] = (if ext { back[i - 1].0 + 1 } else { 1 }, true);
            }
        }
        let mut lengths = vec![];
        let mut oov = vec![];
        let mut i = n;
        while i > 0 {
            let (l, o) = back[i];
            lengths.push(l);
            oov.push(o);
            i -= l;
        }
        lengths.reverse();
        oov.reverse();
        Segmentation { score: best[n], lengths, oov }
    }
}

/// Reversed-word trie for incremental segmentation: walking backwards from the newest
/// letter finds every vocabulary word that ends there.
pub struct WordTrie {
    child: Vec<[u32; 26]>,
    logp: Vec<f32>,
    oov_base: f32,
    oov_per: f32,
}

/// Incremental segmentation state of one letter stream (one per hypothesis).
#[derive(Clone, Copy)]
pub struct StreamState {
    /// Last 24 letters, 5 bits each, newest in the low bits.
    hist: u128,
    /// `best[j]` = best segmentation score of the stream truncated `j` letters ago
    /// (`bh[0]` is the whole stream so far).
    bh: [f32; WINDOW],
    /// Whether the best segmentation of the stream so far ends in an unknown word.
    oov: bool,
}

const WINDOW: usize = 24;

impl StreamState {
    pub fn new() -> Self {
        let mut bh = [f32::NEG_INFINITY; WINDOW];
        bh[0] = 0.0;
        StreamState { hist: 0, bh, oov: false }
    }
    /// Best word-segmentation score of everything pushed so far.
    pub fn score(&self) -> f32 {
        self.bh[0]
    }
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

impl WordTrie {
    /// Override the unknown-word penalty (nats): `base` once per unknown word plus `per` per letter.
    pub fn with_oov(mut self, base: f32, per: f32) -> Self {
        self.oov_base = base;
        self.oov_per = per;
        self
    }

    fn advance(&self, st: &StreamState, x: u8) -> (f32, bool) {
        // Unknown-word fallback: extend an unknown word, or start one.
        let mut best = st.bh[0] + self.oov_per + if st.oov { 0.0 } else { self.oov_base };
        let mut oov = true;
        let mut node = self.child[0][x as usize] as usize;
        let mut l = 1;
        while node != 0 {
            let lp = self.logp[node];
            if lp.is_finite() {
                let v = st.bh[l - 1] + lp;
                if v > best {
                    best = v;
                    oov = false;
                }
            }
            if l == WINDOW {
                break;
            }
            let letter = ((st.hist >> (5 * (l - 1))) & 31) as usize;
            if letter >= 26 {
                break;
            }
            node = self.child[node][letter] as usize;
            l += 1;
        }
        (best, oov)
    }

    /// How much the best segmentation score changes if letter `x` is appended.
    #[inline]
    pub fn gain(&self, st: &StreamState, x: u8) -> f32 {
        self.advance(st, x).0 - st.bh[0]
    }

    pub fn push(&self, st: &StreamState, x: u8) -> StreamState {
        let (best, oov) = self.advance(st, x);
        let mut bh = [f32::NEG_INFINITY; WINDOW];
        bh[0] = best;
        bh[1..].copy_from_slice(&st.bh[..WINDOW - 1]);
        StreamState { hist: (st.hist << 5) | x as u128, bh, oov }
    }
}

impl WordModel {
    pub fn trie(&self) -> WordTrie {
        let mut child: Vec<[u32; 26]> = vec![[0; 26]];
        let mut logp: Vec<f32> = vec![f32::NEG_INFINITY];
        for (w, &lp) in &self.logp {
            let mut node = 0usize;
            for &c in w.iter().rev() {
                let nxt = child[node][c as usize] as usize;
                node = if nxt == 0 {
                    child.push([0; 26]);
                    logp.push(f32::NEG_INFINITY);
                    let id = child.len() - 1;
                    child[node][c as usize] = id as u32;
                    id
                } else {
                    nxt
                };
            }
            logp[node] = lp;
        }
        WordTrie { child, logp, oov_base: OOV_BASE, oov_per: OOV_PER_LETTER }
    }
}

fn count_words(text: &str, counts: &mut FxHashMap<Vec<u8>, u32>) {
    let mut cur: Vec<u8> = Vec::new();
    let mut clitic = false; // the word being read follows an apostrophe ("don't" -> don, t)
    fn flush(cur: &mut Vec<u8>, clitic: bool, counts: &mut FxHashMap<Vec<u8>, u32>) {
        if !cur.is_empty() && !(clitic && cur.len() <= 2) {
            *counts.entry(cur.clone()).or_insert(0) += 1;
        }
        cur.clear();
    }
    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            cur.push(c.to_ascii_lowercase() as u8 - b'a');
        } else {
            let apostrophe = c == '\'' || c == '\u{2019}';
            flush(&mut cur, clitic, counts);
            clitic = apostrophe;
        }
    }
    flush(&mut cur, clitic, counts);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::scrub;

    fn model() -> WordModel {
        let mut c: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for (w, n) in [("fortunate", 5), ("is", 900), ("the", 5000), ("red", 120), ("shirt", 40), ("redshirt", 3), ("engineer", 30), ("who", 400), ("a", 800), ("i", 700)] {
            c.insert(scrub(w), n);
        }
        WordModel::from_counts(c, 1)
    }

    #[test]
    fn save_and_load_round_trip() {
        let m = model();
        let p = std::env::temp_dir().join(format!("cryptok-words-test-{}.txt", std::process::id()));
        m.save(&p).unwrap();
        let l = WordModel::load(&p).unwrap();
        std::fs::remove_file(&p).ok();
        let s = scrub("FORTUNATEISTHEREDSHIRT");
        assert_eq!(l.vocab_size(), m.vocab_size());
        assert!((l.segment(&s).score - m.segment(&s).score).abs() < 1e-3);
    }

    #[test]
    fn finds_word_breaks() {
        let m = model();
        let s = scrub("FORTUNATEISTHEREDSHIRTENGINEERWHO");
        let seg = m.segment(&s);
        assert_eq!(seg.render(&s), "FORTUNATE IS THE REDSHIRT ENGINEER WHO");
        assert!(seg.oov.iter().all(|o| !o));
    }

    #[test]
    fn unknown_stretches_are_tolerated() {
        let m = model();
        let s = scrub("THEZQXWHO");
        let seg = m.segment(&s);
        assert_eq!(seg.lengths.iter().sum::<usize>(), s.len());
        assert!(seg.oov.iter().any(|o| *o));
        let good = m.segment(&scrub("THEREDWHO")).score;
        assert!(good > seg.score);
    }

    #[test]
    fn incremental_state_matches_segmentation() {
        let m = model();
        let t = m.trie();
        let s = scrub("FORTUNATEISTHEREDSHIRTENGINEERWHO");
        let mut st = StreamState::new();
        for &x in &s {
            st = t.push(&st, x);
        }
        assert!((st.score() - m.segment(&s).score).abs() < 1e-3, "{} vs {}", st.score(), m.segment(&s).score);
    }
}
