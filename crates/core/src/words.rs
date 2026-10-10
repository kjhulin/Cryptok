//! Word-level model: word and word-pair frequencies, and word-break discovery.
//!
//! The character model never sees spaces, so it cannot tell that `REDSHIRTENGINEER`
//! is three words. [`WordModel::segment`] finds the most likely split of a letter
//! stream (dynamic programming over the vocabulary) and its log-probability, which
//! can be used to display results with spaces and to re-rank candidate solutions.

use crate::map::{FxHashMap, U64Map};
use crate::text::{strip_gutenberg, unscrub};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MAX_WORD: usize = 24;
const NONE: u32 = u32::MAX;
/// Absolute discount for word-pair probabilities.
const DISCOUNT: f64 = 0.75;
/// Word pairs seen fewer times than this are not stored (they back off to the unigram).
const MIN_BIGRAM: u32 = 3;
/// Word triples seen fewer times than this are not stored (they back off to the bigram).
const MIN_TRIGRAM: u32 = 3;
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

/// Word and word-pair log-probabilities: `P(w|v)` is an absolute-discounted bigram that
/// backs off to the unigram `P(w)`.
#[derive(Clone)]
struct Lex {
    uni: Vec<f32>,
    ln_gamma: Vec<f32>,
    pairs: U64Map,
    /// ln P(w | a, b) for stored triples, keyed by [`tri_key`].
    triples: U64Map,
    /// ln back-off weight of each stored two-word context `(a, b)`, keyed by [`pair_key`].
    ln_gamma3: U64Map,
}

#[inline]
fn pair_key(v: u32, w: u32) -> u64 {
    ((v as u64) << 32) | w as u64
}

#[inline]
fn tri_key(a: u32, b: u32, w: u32) -> u64 {
    ((a as u64) << 42) | ((b as u64) << 21) | w as u64
}

impl Lex {
    /// ln P(w | previous word `v`), or ln P(w) when there is no (known) previous word.
    #[inline]
    fn lp2(&self, v: u32, w: u32) -> f32 {
        if v == NONE {
            return self.uni[w as usize];
        }
        match self.pairs.get(pair_key(v, w)) {
            Some(bits) => f32::from_bits(bits),
            None => self.ln_gamma[v as usize] + self.uni[w as usize],
        }
    }

    /// ln P(w | `a` `b`): trigram when stored, otherwise backed off to the bigram.
    #[inline]
    fn lp(&self, a: u32, b: u32, w: u32) -> f32 {
        if a == NONE || b == NONE {
            return self.lp2(b, w);
        }
        if let Some(bits) = self.triples.get(tri_key(a, b, w)) {
            return f32::from_bits(bits);
        }
        match self.ln_gamma3.get(pair_key(a, b)) {
            Some(bits) => f32::from_bits(bits) + self.lp2(b, w),
            None => self.lp2(b, w),
        }
    }
}

pub struct WordModel {
    /// Vocabulary by id (most frequent first) with unigram counts.
    words: Vec<(Vec<u8>, u32)>,
    ids: FxHashMap<Vec<u8>, u32>,
    /// Word pairs `(previous id, id, count)`.
    bigrams: Vec<(u32, u32, u32)>,
    /// Word triples `(id, id, id, count)`.
    trigrams: Vec<(u32, u32, u32, u32)>,
    lex: Lex,
    max_len: usize,
}

/// Call `f(Some(word))` for every word of `text` (letters as 0..26) and `f(None)` at
/// sentence breaks and dropped contraction fragments.
fn scan(text: &str, mut f: impl FnMut(Option<&[u8]>)) {
    let mut cur: Vec<u8> = Vec::new();
    let mut clitic = false; // the word being read follows an apostrophe ("don't" -> don, t)
    let mut prev_newline = false;
    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            cur.push(c.to_ascii_lowercase() as u8 - b'a');
            prev_newline = false;
            continue;
        }
        if !cur.is_empty() {
            if clitic && cur.len() <= 2 {
                f(None);
            } else {
                f(Some(&cur));
            }
            cur.clear();
        }
        clitic = c == '\'' || c == '\u{2019}';
        if matches!(c, '.' | '!' | '?') || (c == '\n' && prev_newline) {
            f(None);
        }
        prev_newline = c == '\n';
    }
    if !cur.is_empty() {
        f(Some(&cur));
    }
}

fn vocabulary(counts: FxHashMap<Vec<u8>, u32>, min_count: u32) -> Vec<(Vec<u8>, u32)> {
    let mut kept: Vec<(Vec<u8>, u32)> = counts
        .into_iter()
        .filter(|(w, c)| *c >= min_count && w.len() <= MAX_WORD && (w.len() > 1 || matches!(w[0], 0 | 8))) // single letters: only A, I
        .collect();
    kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    kept
}

/// How progress splits across the stages of building a word model: tokenising the texts
/// (reading the files too, when [`WordModel::from_corpus_progress`] does it), counting pairs,
/// then the remaining work (triples, tables).
const TOKEN_SHARE: f64 = 0.50;
const PAIR_SHARE: f64 = 0.12;

/// Turns words into small integer ids as the texts are scanned, so that later passes work on
/// integers instead of hashing the same words again.
#[derive(Default)]
struct Interner {
    ids: FxHashMap<Vec<u8>, u32>,
    words: Vec<Vec<u8>>,
    counts: Vec<u32>,
}

impl Interner {
    /// The id of `w`, counting one more occurrence. Only a new word allocates.
    fn count(&mut self, w: &[u8]) -> u32 {
        if let Some(&i) = self.ids.get(w) {
            self.counts[i as usize] += 1;
            return i;
        }
        let i = self.words.len() as u32;
        self.ids.insert(w.to_vec(), i);
        self.words.push(w.to_vec());
        self.counts.push(1);
        i
    }

    /// Scan one text into a stream of word ids, with `NONE` at sentence breaks.
    fn tokenize(&mut self, text: &str) -> Vec<u32> {
        let mut stream = Vec::with_capacity(text.len() / 6);
        scan(text, |w| stream.push(w.map_or(NONE, |w| self.count(w))));
        stream
    }
}

impl WordModel {
    /// Build from raw text files in the corpus directories (Gutenberg boilerplate stripped),
    /// skipping the file names in `exclude`. Words seen fewer than `min_count` times are dropped.
    /// Word triples are only collected when `trigrams` is set (they did not improve accuracy).
    pub fn from_corpus(dirs: &[PathBuf], exclude: &[String], min_count: u32, trigrams: bool) -> io::Result<Self> {
        Self::from_corpus_progress(dirs, exclude, min_count, trigrams, &|_| {})
    }

    /// [`from_corpus`](Self::from_corpus) that reports the fraction done (0.0 to 1.0) as it goes.
    /// Each file is reduced to word ids as soon as it is read, so the corpus text is never all
    /// in memory at once.
    pub fn from_corpus_progress(dirs: &[PathBuf], exclude: &[String], min_count: u32, trigrams: bool, progress: &dyn Fn(f64)) -> io::Result<Self> {
        let names = crate::text::corpus_files(dirs, exclude)?;
        let total = names.len().max(1) as f64;
        let mut interner = Interner::default();
        let mut streams = Vec::with_capacity(names.len());
        for (i, p) in names.iter().enumerate() {
            progress(TOKEN_SHARE * i as f64 / total);
            let bytes = fs::read(p)?;
            streams.push(interner.tokenize(strip_gutenberg(&String::from_utf8_lossy(&bytes))));
        }
        Ok(Self::from_streams(interner, streams, min_count, trigrams, progress))
    }

    /// Build from in-memory texts: word counts, then counts of adjacent in-vocabulary pairs.
    pub fn from_texts<S: AsRef<str>>(texts: &[S], min_count: u32, trigrams: bool) -> Self {
        Self::from_texts_progress(texts, min_count, trigrams, &|_| {})
    }

    /// [`from_texts`](Self::from_texts) that reports the fraction done (0.0 to 1.0) as it goes.
    pub fn from_texts_progress<S: AsRef<str>>(texts: &[S], min_count: u32, trigrams: bool, progress: &dyn Fn(f64)) -> Self {
        let n = texts.len().max(1) as f64;
        let mut interner = Interner::default();
        let mut streams = Vec::with_capacity(texts.len());
        for (i, t) in texts.iter().enumerate() {
            progress(TOKEN_SHARE * i as f64 / n);
            streams.push(interner.tokenize(t.as_ref()));
        }
        Self::from_streams(interner, streams, min_count, trigrams, progress)
    }

    /// The shared back half: pick the vocabulary, then count adjacent pairs (and triples) of
    /// in-vocabulary words in the id streams.
    fn from_streams(interner: Interner, streams: Vec<Vec<u32>>, min_count: u32, trigrams: bool, progress: &dyn Fn(f64)) -> Self {
        let Interner { ids, words: seen_words, counts } = interner;
        let counts: FxHashMap<Vec<u8>, u32> = seen_words.iter().cloned().zip(counts).collect();
        let words = vocabulary(counts, min_count);
        // Interner id -> vocabulary id (NONE for words that were dropped).
        let mut remap = vec![NONE; seen_words.len()];
        for (new, (w, _)) in words.iter().enumerate() {
            remap[ids[w] as usize] = new as u32;
        }
        drop((ids, seen_words));
        let vocab_id = |t: u32| if t == NONE { NONE } else { remap[t as usize] };
        let n = streams.len().max(1) as f64;
        let mut pairs: FxHashMap<u64, u32> = FxHashMap::default();
        for (i, stream) in streams.iter().enumerate() {
            progress(TOKEN_SHARE + PAIR_SHARE * i as f64 / n);
            let mut prev = NONE;
            for &t in stream {
                let id = vocab_id(t);
                if prev != NONE && id != NONE {
                    *pairs.entry(pair_key(prev, id)).or_insert(0) += 1;
                }
                prev = id;
            }
        }
        let bigrams: Vec<(u32, u32, u32)> = pairs.into_iter().filter(|&(_, c)| c >= MIN_BIGRAM).map(|(k, c)| ((k >> 32) as u32, k as u32, c)).collect();
        // Triples whose two-word prefix and suffix are stored pairs.
        let stored: std::collections::HashSet<u64> = bigrams.iter().map(|&(v, w, _)| pair_key(v, w)).collect();
        let mut triples: FxHashMap<u64, u32> = FxHashMap::default();
        for stream in streams.iter().filter(|_| trigrams) {
            let (mut a, mut b) = (NONE, NONE);
            for &t in stream {
                let id = vocab_id(t);
                if a != NONE && b != NONE && id != NONE && stored.contains(&pair_key(a, b)) && stored.contains(&pair_key(b, id)) {
                    *triples.entry(tri_key(a, b, id)).or_insert(0) += 1;
                }
                a = b;
                b = id;
            }
        }
        let trigrams = triples
            .into_iter()
            .filter(|&(_, c)| c >= MIN_TRIGRAM)
            .map(|(k, c)| ((k >> 42) as u32, ((k >> 21) & 0x1F_FFFF) as u32, (k & 0x1F_FFFF) as u32, c))
            .collect();
        Self::build(words, bigrams, trigrams)
    }

    /// Unigram-only model from word counts (no pair statistics).
    pub fn from_counts(counts: FxHashMap<Vec<u8>, u32>, min_count: u32) -> Self {
        Self::build(vocabulary(counts, min_count), vec![], vec![])
    }

    fn build(words: Vec<(Vec<u8>, u32)>, mut bigrams: Vec<(u32, u32, u32)>, mut trigrams: Vec<(u32, u32, u32, u32)>) -> Self {
        bigrams.sort_unstable();
        trigrams.sort_unstable();
        let total: f64 = words.iter().map(|(_, c)| *c as f64).sum::<f64>().max(1.0);
        let uni: Vec<f32> = words.iter().map(|(_, c)| ((*c as f64) / total).ln() as f32).collect();
        let ids = words.iter().enumerate().map(|(i, (w, _))| (w.clone(), i as u32)).collect();
        let max_len = words.iter().map(|(w, _)| w.len()).max().unwrap_or(1);
        // Seen mass of each previous word, for the back-off weight.
        let mut seen = vec![0.0f64; words.len()];
        for &(v, _, c) in &bigrams {
            seen[v as usize] += (c as f64 - DISCOUNT).max(0.0) / words[v as usize].1 as f64;
        }
        let gamma: Vec<f64> = seen.iter().map(|s| (1.0 - s).clamp(1e-4, 1.0)).collect();
        let ln_gamma: Vec<f32> = gamma.iter().map(|g| g.ln() as f32).collect();
        let mut pairs = U64Map::with_capacity(bigrams.len().max(16) * 2);
        for &(v, w, c) in &bigrams {
            let cv = words[v as usize].1 as f64;
            let p = (c as f64 - DISCOUNT).max(0.0) / cv + gamma[v as usize] * (uni[w as usize] as f64).exp();
            pairs.insert(pair_key(v, w), (p.ln() as f32).to_bits());
        }
        // Trigram level: absolute discounting of P(w|a,b) over the stored triples, backing off
        // to the (interpolated) bigram probability.
        let pair_count: FxHashMap<u64, u32> = bigrams.iter().map(|&(v, w, c)| (pair_key(v, w), c)).collect();
        let lp2 = |v: u32, w: u32| -> f64 {
            match pairs.get(pair_key(v, w)) {
                Some(bits) => (f32::from_bits(bits) as f64).exp(),
                None => (gamma[v as usize] * (uni[w as usize] as f64).exp()).max(1e-30),
            }
        };
        let mut seen3: FxHashMap<u64, f64> = FxHashMap::default();
        for &(a, b, _, c) in &trigrams {
            if let Some(&cab) = pair_count.get(&pair_key(a, b)) {
                *seen3.entry(pair_key(a, b)).or_insert(0.0) += (c as f64 - DISCOUNT).max(0.0) / cab as f64;
            }
        }
        let mut ln_gamma3 = U64Map::with_capacity(seen3.len().max(16) * 2);
        let mut gamma3: FxHashMap<u64, f64> = FxHashMap::default();
        for (&k, &sm) in &seen3 {
            let g = (1.0 - sm).clamp(1e-4, 1.0);
            gamma3.insert(k, g);
            ln_gamma3.insert(k, (g.ln() as f32).to_bits());
        }
        let mut triples = U64Map::with_capacity(trigrams.len().max(16) * 2);
        for &(a, b, w, c) in &trigrams {
            let Some(&cab) = pair_count.get(&pair_key(a, b)) else { continue };
            let g = gamma3.get(&pair_key(a, b)).copied().unwrap_or(1.0);
            let p = (c as f64 - DISCOUNT).max(0.0) / cab as f64 + g * lp2(b, w);
            triples.insert(tri_key(a, b, w), (p.ln() as f32).to_bits());
        }
        WordModel { words, ids, bigrams, trigrams, lex: Lex { uni, ln_gamma, pairs, triples, ln_gamma3 }, max_len }
    }

    pub fn vocab_size(&self) -> usize {
        self.words.len()
    }

    /// The same vocabulary without pair statistics (every word scored on its own).
    pub fn without_bigrams(self) -> Self {
        Self::build(self.words, vec![], vec![])
    }

    /// Keep word pairs but drop word triples.
    pub fn without_trigrams(self) -> Self {
        Self::build(self.words, self.bigrams, vec![])
    }

    pub fn trigram_count(&self) -> usize {
        self.trigrams.len()
    }

    pub fn bigram_count(&self) -> usize {
        self.bigrams.len()
    }

    /// Write the model as plain text: `word<TAB>count` lines, then `#bigrams` and
    /// `word word<TAB>count` lines.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut out = String::from("#cryptok-words v3\n");
        for (w, c) in &self.words {
            out.push_str(&unscrub(w).to_lowercase());
            out.push('\t');
            out.push_str(&c.to_string());
            out.push('\n');
        }
        out.push_str("#bigrams\n");
        for &(v, w, c) in &self.bigrams {
            out.push_str(&unscrub(&self.words[v as usize].0).to_lowercase());
            out.push(' ');
            out.push_str(&unscrub(&self.words[w as usize].0).to_lowercase());
            out.push('\t');
            out.push_str(&c.to_string());
            out.push('\n');
        }
        out.push_str("#trigrams\n");
        for &(a, b, w, c) in &self.trigrams {
            for (i, id) in [a, b, w].iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(&unscrub(&self.words[*id as usize].0).to_lowercase());
            }
            out.push('\t');
            out.push_str(&c.to_string());
            out.push('\n');
        }
        fs::write(path, out)
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        match lines.next() {
            Some("#cryptok-words v3") | Some("#cryptok-words v2") | Some("#cryptok-words v1") => {}
            _ => return Err(bad("not a cryptok word list")),
        }
        let mut counts: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let mut pair_lines: Vec<(&str, u32)> = vec![];
        let mut triple_lines: Vec<(&str, u32)> = vec![];
        let mut section = 0; // 0 = words, 1 = bigrams, 2 = trigrams
        for l in lines {
            if l == "#bigrams" {
                section = 1;
                continue;
            }
            if l == "#trigrams" {
                section = 2;
                continue;
            }
            if let Some((w, c)) = l.split_once('\t') {
                if let Ok(c) = c.parse::<u32>() {
                    if section == 1 {
                        pair_lines.push((w, c));
                    } else if section == 2 {
                        triple_lines.push((w, c));
                    } else {
                        counts.insert(crate::text::scrub(w), c);
                    }
                }
            }
        }
        let words = vocabulary(counts, 1);
        let ids: FxHashMap<Vec<u8>, u32> = words.iter().enumerate().map(|(i, (w, _))| (w.clone(), i as u32)).collect();
        let mut bigrams = vec![];
        for (pair, c) in pair_lines {
            let Some((a, b)) = pair.split_once(' ') else { continue };
            if let (Some(&v), Some(&w)) = (ids.get(&crate::text::scrub(a)), ids.get(&crate::text::scrub(b))) {
                bigrams.push((v, w, c));
            }
        }
        let mut trigrams = vec![];
        for (t, c) in triple_lines {
            let ids3: Vec<Option<u32>> = t.split(' ').map(|x| ids.get(&crate::text::scrub(x)).copied()).collect();
            if let [Some(a), Some(b), Some(w)] = ids3[..] {
                trigrams.push((a, b, w, c));
            }
        }
        Ok(Self::build(words, bigrams, trigrams))
    }

    /// Vocabulary ids of the known words in the best segmentation of `s`, with their
    /// unigram log-probabilities (unknown stretches are skipped).
    pub fn known_words(&self, s: &[u8]) -> Vec<(u32, f32)> {
        let seg = self.segment(s);
        let mut at = 0;
        let mut out = Vec::new();
        for (&l, &oov) in seg.lengths.iter().zip(&seg.oov) {
            if !oov {
                if let Some(&w) = self.ids.get(&s[at..at + l]) {
                    out.push((w, self.lex.uni[w as usize]));
                }
            }
            at += l;
        }
        out
    }

    /// Most likely segmentation of `s` into words (out-of-vocabulary stretches allowed
    /// at a penalty, so names and typos do not break the parse). Words are scored with
    /// the bigram model given the best segmentation of the text before them.
    pub fn segment(&self, s: &[u8]) -> Segmentation {
        let n = s.len();
        // best[i]: best score for s[..i]; back[i]: (word length, was_oov); wid[i]: last word id.
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut back = vec![(0usize, false); n + 1];
        let mut wid = vec![NONE; n + 1];
        let mut wid2 = vec![NONE; n + 1]; // the word before that
        best[0] = 0.0;
        for i in 1..=n {
            for l in 1..=i.min(self.max_len) {
                let prev = best[i - l];
                if prev == f32::NEG_INFINITY {
                    continue;
                }
                if let Some(&w) = self.ids.get(&s[i - l..i]) {
                    let v = prev + self.lex.lp(wid2[i - l], wid[i - l], w);
                    if v > best[i] {
                        best[i] = v;
                        back[i] = (l, false);
                        wid[i] = w;
                        wid2[i] = wid[i - l];
                    }
                }
            }
            // Out-of-vocabulary fallback: a single unknown letter extends the previous
            // unknown word or starts a new one (cost is per letter plus a start cost).
            let extend = if back[i - 1].1 { best[i - 1] + OOV_PER_LETTER } else { f32::NEG_INFINITY };
            let fresh = best[i - 1] + OOV_BASE + OOV_PER_LETTER;
            let (cand, ext) = if extend > fresh { (extend, true) } else { (fresh, false) };
            if cand > best[i] {
                best[i] = cand;
                back[i] = (if ext { back[i - 1].0 + 1 } else { 1 }, true);
                wid[i] = NONE;
                wid2[i] = NONE;
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
    /// Word id ending at each node (`NONE` if the node is not a word).
    term: Vec<u32>,
    lex: Lex,
    oov_base: f32,
    oov_per: f32,
}

const WINDOW: usize = 24;

/// Incremental segmentation state of one letter stream (one per hypothesis).
#[derive(Clone, Copy)]
pub struct StreamState {
    /// Last 24 letters, 5 bits each, newest in the low bits.
    hist: u128,
    /// `bh[j]` = best segmentation score of the stream truncated `j` letters ago
    /// (`bh[0]` is the whole stream so far).
    bh: [f32; WINDOW],
    /// Id of the last word of that best segmentation (`NONE` if unknown or empty).
    wid: [u32; WINDOW],
    /// Id of the word before it.
    wid2: [u32; WINDOW],
    /// Whether the best segmentation of the stream so far ends in an unknown word.
    oov: bool,
}

impl StreamState {
    pub fn new() -> Self {
        let mut bh = [f32::NEG_INFINITY; WINDOW];
        bh[0] = 0.0;
        StreamState { hist: 0, bh, wid: [NONE; WINDOW], wid2: [NONE; WINDOW], oov: false }
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

    /// Best score, unknown-word flag and last word id after appending letter `x`.
    fn advance(&self, st: &StreamState, x: u8) -> (f32, bool, u32, u32) {
        // Unknown-word fallback: extend an unknown word, or start one.
        let mut best = st.bh[0] + self.oov_per + if st.oov { 0.0 } else { self.oov_base };
        let mut oov = true;
        let mut last = NONE;
        let mut before = NONE;
        let mut node = self.child[0][x as usize] as usize;
        let mut l = 1;
        while node != 0 {
            let w = self.term[node];
            if w != NONE {
                let v = st.bh[l - 1] + self.lex.lp(st.wid2[l - 1], st.wid[l - 1], w);
                if v > best {
                    best = v;
                    oov = false;
                    last = w;
                    before = st.wid[l - 1];
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
        (best, oov, last, before)
    }

    /// How much the best segmentation score changes if letter `x` is appended.
    #[inline]
    pub fn gain(&self, st: &StreamState, x: u8) -> f32 {
        self.advance(st, x).0 - st.bh[0]
    }

    pub fn push(&self, st: &StreamState, x: u8) -> StreamState {
        let (best, oov, last, before) = self.advance(st, x);
        let mut bh = [f32::NEG_INFINITY; WINDOW];
        bh[0] = best;
        bh[1..].copy_from_slice(&st.bh[..WINDOW - 1]);
        let mut wid = [NONE; WINDOW];
        wid[0] = last;
        wid[1..].copy_from_slice(&st.wid[..WINDOW - 1]);
        let mut wid2 = [NONE; WINDOW];
        wid2[0] = before;
        wid2[1..].copy_from_slice(&st.wid2[..WINDOW - 1]);
        StreamState { hist: (st.hist << 5) | x as u128, bh, wid, wid2, oov }
    }
}

impl WordModel {
    pub fn trie(&self) -> WordTrie {
        let mut child: Vec<[u32; 26]> = vec![[0; 26]];
        let mut term: Vec<u32> = vec![NONE];
        for (id, (w, _)) in self.words.iter().enumerate() {
            let mut node = 0usize;
            for &c in w.iter().rev() {
                let nxt = child[node][c as usize] as usize;
                node = if nxt == 0 {
                    child.push([0; 26]);
                    term.push(NONE);
                    let nid = child.len() - 1;
                    child[node][c as usize] = nid as u32;
                    nid
                } else {
                    nxt
                };
            }
            term[node] = id as u32;
        }
        WordTrie { child, term, lex: self.lex.clone(), oov_base: OOV_BASE, oov_per: OOV_PER_LETTER }
    }
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
        assert_eq!(l.bigram_count(), m.bigram_count());
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

    #[test]
    fn bigrams_prefer_common_word_pairs() {
        // "red shirt" always occurs together; "red tree" is never seen. Both words are common.
        let text = "the red shirt. ".repeat(20) + &"a green tree. ".repeat(20) + &"the red hat. a tree. ".repeat(5);
        let m = WordModel::from_texts(&[text], 2, true);
        assert!(m.bigram_count() > 0);
        let a = m.segment(&scrub("REDSHIRT")).score;
        let b = m.segment(&scrub("REDTREE")).score;
        assert!(a > b, "{a} vs {b}");
    }

    #[test]
    fn incremental_state_matches_segmentation_with_bigrams() {
        let text = "the red shirt engineer who is a fortunate engineer. ".repeat(10);
        let m = WordModel::from_texts(&[text], 2, true);
        let t = m.trie();
        let s = scrub("FORTUNATEENGINEERWHOISTHEREDSHIRT");
        let mut st = StreamState::new();
        for &x in &s {
            st = t.push(&st, x);
        }
        assert!((st.score() - m.segment(&s).score).abs() < 1e-3, "{} vs {}", st.score(), m.segment(&s).score);
    }
}
