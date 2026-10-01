//! Character n-gram language model with interpolated Kneser–Ney smoothing.
//!
//! The model stores, for every context seen in training (up to `order` letters),
//! a full 26-entry row of quantised log-probabilities `ln P(next | context)`.
//! Scoring a letter is therefore a single hash probe for the longest stored
//! context (backing off to shorter contexts only when the context was never seen,
//! which is exactly what interpolated Kneser–Ney prescribes).
//!
//! Contexts are packed base-26 with the most recent letter least significant, so
//! the last `L` letters of a packed context `c` are simply `c % 26^L`.

use crate::map::{FxHashMap, U64Map};
use crate::text::{scrub, strip_gutenberg, ALPHABET};
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

/// Quantisation step for log-probabilities (nats). u8 covers 0 .. -25.5.
pub const STEP: f32 = 0.1;
const MAGIC: &[u8; 4] = b"CKLM";
const VERSION: u32 = 1;
pub const MAX_ORDER: usize = 12;

struct Level {
    map: U64Map,
    rows: Vec<u8>,
}

pub struct LangModel {
    order: usize,
    levels: Vec<Level>,
    deq: [f32; 256],
    pow: Vec<u64>,
}

fn pow26(n: usize) -> Vec<u64> {
    let mut v = vec![1u64; n + 1];
    for i in 1..=n {
        v[i] = v[i - 1] * 26;
    }
    v
}

fn deq_table() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, x) in t.iter_mut().enumerate() {
        *x = -(i as f32) * STEP;
    }
    t
}

/// Statistics reported after training.
#[derive(Debug, Clone)]
pub struct TrainStats {
    pub letters: u64,
    pub contexts_per_level: Vec<usize>,
    pub discounts: Vec<f64>,
}

impl LangModel {
    /// Context length (Markov order). Scoring uses up to `order` previous letters.
    pub fn order(&self) -> usize {
        self.order
    }

    #[inline]
    pub fn pow(&self, n: usize) -> u64 {
        self.pow[n]
    }

    /// Quantised log-prob row for the longest stored suffix of `ctx` (at most `len` letters).
    #[inline]
    pub fn row(&self, ctx: u64, len: usize) -> &[u8] {
        let mut l = len.min(self.order);
        loop {
            let lv = &self.levels[l];
            if let Some(i) = lv.map.get(ctx % self.pow[l]) {
                let i = i as usize * ALPHABET;
                return &lv.rows[i..i + ALPHABET];
            }
            l -= 1; // level 0 always contains the empty context
        }
    }

    /// Dequantisation table: `deq()[q]` is the log-probability for quantised value `q`.
    #[inline]
    pub fn deq(&self) -> &[f32; 256] {
        &self.deq
    }

    /// Joint log-probability (nats) of a letter sequence, starting with an empty context.
    pub fn score(&self, letters: &[u8]) -> f32 {
        let modk = self.pow[self.order];
        let mut ctx = 0u64;
        let mut s = 0f32;
        for (i, &c) in letters.iter().enumerate() {
            s += self.deq[self.row(ctx, i)[c as usize] as usize];
            ctx = (ctx * 26 + c as u64) % modk;
        }
        s
    }

    /// Mean log-probability per letter — comparable across strings of different lengths.
    pub fn score_per_letter(&self, letters: &[u8]) -> f32 {
        if letters.is_empty() {
            return f32::NEG_INFINITY;
        }
        self.score(letters) / letters.len() as f32
    }

    pub fn score_str(&self, s: &str) -> f32 {
        self.score(&scrub(s))
    }

    /// Train from a directory of text files (Gutenberg boilerplate is stripped).
    /// Files whose names appear in `exclude` are skipped.
    pub fn train_dir(dir: &Path, order: usize, exclude: &[String]) -> io::Result<(Self, TrainStats)> {
        let mut names: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| {
                let n = p.file_name().unwrap().to_string_lossy().to_string();
                !exclude.iter().any(|x| *x == n)
            })
            .collect();
        names.sort();
        let mut texts = Vec::with_capacity(names.len());
        for p in names {
            let mut buf = Vec::new();
            fs::File::open(&p)?.read_to_end(&mut buf)?;
            let s = String::from_utf8_lossy(&buf);
            texts.push(scrub(strip_gutenberg(&s)));
        }
        Ok(Self::train(&texts, order))
    }

    /// Train from pre-scrubbed letter sequences.
    pub fn train(texts: &[Vec<u8>], order: usize) -> (Self, TrainStats) {
        assert!((1..=MAX_ORDER).contains(&order), "order must be 1..={MAX_ORDER}");
        let k = order;
        let m = k + 1;
        let pow = pow26(m);

        // 1. Raw counts of (k+1)-grams.
        let mut letters = 0u64;
        let mut top: FxHashMap<u64, u32> = FxHashMap::default();
        for t in texts {
            letters += t.len() as u64;
            let mut r = 0u64;
            for (i, &c) in t.iter().enumerate() {
                r = (r * 26 + c as u64) % pow[m];
                if i + 1 >= m {
                    *top.entry(r).or_insert(0) += 1;
                }
            }
        }

        // 2. Kneser–Ney continuation counts for lower orders:
        //    cnt[L][g] = number of distinct letters x such that x·g was seen.
        let mut cnt: Vec<FxHashMap<u64, u32>> = (0..=k).map(|_| FxHashMap::default()).collect();
        cnt[k] = top;
        for l in (0..k).rev() {
            let mut lower: FxHashMap<u64, u32> = FxHashMap::default();
            for &g in cnt[l + 1].keys() {
                *lower.entry(g % pow[l + 1]).or_insert(0) += 1;
            }
            cnt[l] = lower;
        }

        // 3. Build probability rows level by level (low to high), interpolating with the level below.
        struct Build {
            map: U64Map,
            probs: Vec<f32>,
        }
        let mut built: Vec<Build> = Vec::with_capacity(k + 1);
        let mut discounts = Vec::with_capacity(k + 1);
        let mut contexts_per_level = Vec::with_capacity(k + 1);
        for l in 0..=k {
            let counts = &cnt[l];
            let (mut n1, mut n2) = (0u64, 0u64);
            for &c in counts.values() {
                if c == 1 {
                    n1 += 1;
                } else if c == 2 {
                    n2 += 1;
                }
            }
            let d = if n1 + n2 == 0 { 0.5 } else { (n1 as f64 / (n1 as f64 + 2.0 * n2 as f64)).clamp(0.1, 0.95) };
            discounts.push(d);

            // Gather counts per context.
            let mut map = U64Map::with_capacity(counts.len() / 4 + 1);
            let mut raw: Vec<u32> = Vec::new();
            let mut keys: Vec<u64> = counts.keys().copied().collect();
            keys.sort_unstable(); // deterministic layout
            for g in keys {
                let c = counts[&g];
                let h = g / 26;
                let w = (g % 26) as usize;
                let idx = match map.get(h) {
                    Some(i) => i as usize,
                    None => {
                        let i = raw.len() / ALPHABET;
                        map.insert(h, i as u32);
                        raw.resize(raw.len() + ALPHABET, 0);
                        i
                    }
                };
                raw[idx * ALPHABET + w] = c;
            }
            let nctx = raw.len() / ALPHABET;
            contexts_per_level.push(nctx);

            let mut probs = vec![0f32; raw.len()];
            // Context keys by index, needed to find the lower-order row.
            let mut ctx_of = vec![0u64; nctx];
            {
                // Rebuild index -> context mapping.
                for g in cnt[l].keys() {
                    let h = g / 26;
                    ctx_of[map.get(h).unwrap() as usize] = h;
                }
            }
            for i in 0..nctx {
                let row = &raw[i * ALPHABET..(i + 1) * ALPHABET];
                let s: u64 = row.iter().map(|&c| c as u64).sum();
                let t = row.iter().filter(|&&c| c > 0).count() as f64;
                let s = s as f64;
                let lower: Vec<f32> = if l == 0 {
                    vec![1.0 / 26.0; ALPHABET]
                } else {
                    let h = ctx_of[i];
                    // Longest stored suffix in lower levels.
                    let mut ll = l - 1;
                    loop {
                        if let Some(j) = built[ll].map.get(h % pow[ll]) {
                            let j = j as usize * ALPHABET;
                            break built[ll].probs[j..j + ALPHABET].to_vec();
                        }
                        ll -= 1;
                    }
                };
                let gamma = d * t / s;
                for w in 0..ALPHABET {
                    let c = row[w] as f64;
                    let p = ((c - d).max(0.0) / s) + gamma * lower[w] as f64;
                    probs[i * ALPHABET + w] = p as f32;
                }
            }
            built.push(Build { map, probs });
        }

        // 4. Quantise.
        let levels = built
            .into_iter()
            .map(|b| Level {
                map: b.map,
                rows: b
                    .probs
                    .iter()
                    .map(|&p| (-(p.max(1e-30)).ln() / STEP).round().clamp(0.0, 255.0) as u8)
                    .collect(),
            })
            .collect();

        let lm = LangModel { order: k, levels, deq: deq_table(), pow: pow26(k) };
        (lm, TrainStats { letters, contexts_per_level, discounts })
    }

    /// Serialise to a compact binary file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = io::BufWriter::new(fs::File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(self.order as u32).to_le_bytes())?;
        w.write_all(&STEP.to_le_bytes())?;
        for lv in &self.levels {
            let n = lv.rows.len() / ALPHABET;
            // Recover keys in row order.
            let mut keys = vec![u64::MAX; n];
            lv_keys(lv, &mut keys);
            w.write_all(&(n as u64).to_le_bytes())?;
            for k in &keys {
                w.write_all(&k.to_le_bytes())?;
            }
            w.write_all(&lv.rows)?;
        }
        w.flush()
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let data = fs::read(path)?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let mut pos = 0usize;
        let mut take = |n: usize| -> io::Result<&[u8]> {
            if pos + n > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated model file"));
            }
            let s = &data[pos..pos + n];
            pos += n;
            Ok(s)
        };
        if take(4)? != MAGIC {
            return Err(bad("not a cryptok language model file"));
        }
        let ver = u32::from_le_bytes(take(4)?.try_into().unwrap());
        if ver != VERSION {
            return Err(bad(&format!("unsupported model version {ver}")));
        }
        let order = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        if !(1..=MAX_ORDER).contains(&order) {
            return Err(bad("invalid order"));
        }
        let step = f32::from_le_bytes(take(4)?.try_into().unwrap());
        if (step - STEP).abs() > 1e-6 {
            return Err(bad("unsupported quantisation step"));
        }
        let mut levels = Vec::with_capacity(order + 1);
        for _ in 0..=order {
            let n = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
            let kb = take(n * 8)?;
            let mut map = U64Map::with_capacity(n);
            for (i, ch) in kb.chunks_exact(8).enumerate() {
                map.insert(u64::from_le_bytes(ch.try_into().unwrap()), i as u32);
            }
            let rows = take(n * ALPHABET)?.to_vec();
            levels.push(Level { map, rows });
        }
        if levels[0].map.get(0).is_none() {
            return Err(bad("model has no unigram row"));
        }
        Ok(LangModel { order, levels, deq: deq_table(), pow: pow26(order) })
    }

    /// Number of stored contexts at each level.
    pub fn contexts_per_level(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.rows.len() / ALPHABET).collect()
    }

    /// Dense table of `ln P(last | previous n-1 letters)` for every n-gram (n <= order+1).
    /// Used by hill-climbing solvers, where a flat array lookup beats hashing.
    pub fn dense(&self, n: usize) -> DenseNgram {
        assert!(n >= 1 && n <= self.order + 1 && n <= 6, "dense n-gram size out of range");
        let ctxs = 26usize.pow(n as u32 - 1);
        let mut t = vec![0f32; ctxs * ALPHABET];
        for ctx in 0..ctxs {
            let row = self.row(ctx as u64, n - 1);
            for w in 0..ALPHABET {
                t[ctx * ALPHABET + w] = self.deq[row[w] as usize];
            }
        }
        DenseNgram { n, t }
    }
}

/// Flat n-gram log-probability table (see [`LangModel::dense`]).
pub struct DenseNgram {
    pub n: usize,
    t: Vec<f32>,
}

impl DenseNgram {
    /// Log-prob of the n-gram ending at `end` (requires `end + 1 >= n`).
    #[inline]
    pub fn window(&self, s: &[u8], end: usize) -> f32 {
        let mut idx = 0usize;
        for &c in &s[end + 1 - self.n..=end] {
            idx = idx * 26 + c as usize;
        }
        unsafe { *self.t.get_unchecked(idx) }
    }

    /// Sum over all complete n-gram windows.
    pub fn score(&self, s: &[u8]) -> f32 {
        (self.n.saturating_sub(1)..s.len()).map(|e| self.window(s, e)).sum()
    }
}

fn lv_keys(lv: &Level, out: &mut [u64]) {
    // U64Map does not expose iteration; probe candidates is impractical, so we
    // keep a reverse index by scanning the map's internal storage via `for_each`.
    lv.map.for_each(|k, v| out[v as usize] = k);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> LangModel {
        let t = scrub(&"the quick brown fox jumps over the lazy dog and then the cat sat on the mat ".repeat(50));
        LangModel::train(&[t], 3).0
    }

    #[test]
    fn rows_are_distributions() {
        let lm = tiny();
        for len in 0..=3 {
            let row = lm.row(0, len);
            let total: f32 = row.iter().map(|&q| lm.deq()[q as usize].exp()).sum();
            assert!((total - 1.0).abs() < 0.08, "row sums to {total}");
        }
    }

    #[test]
    fn english_beats_noise() {
        let lm = tiny();
        let a = lm.score_per_letter(&scrub("thecatsatonthemat"));
        let b = lm.score_per_letter(&scrub("qzxjvkqwzpxqjvkzq"));
        assert!(a > b + 1.0, "{a} vs {b}");
    }

    #[test]
    fn save_load_roundtrip() {
        let lm = tiny();
        let p = std::env::temp_dir().join("cryptok_test_model.cklm");
        lm.save(&p).unwrap();
        let lm2 = LangModel::load(&p).unwrap();
        let s = scrub("thequickbrownfox");
        assert_eq!(lm.score(&s), lm2.score(&s));
        std::fs::remove_file(p).ok();
    }
}
