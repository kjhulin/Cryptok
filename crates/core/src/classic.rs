//! Classic ciphers. Currently: periodic Vigenère with an optional keyed alphabet
//! (Quagmire III — the system used for Kryptos K1 and K2).

use crate::lm::{DenseNgram, LangModel};

/// A mixed alphabet built from a keyword (duplicates dropped, remaining letters appended).
#[derive(Clone, Debug, PartialEq)]
pub struct Alphabet {
    /// `letters[i]` = letter (0..26) at position `i`.
    pub letters: [u8; 26],
    /// `index[letter]` = position of that letter.
    pub index: [u8; 26],
    pub keyword: String,
}

impl Alphabet {
    pub fn standard() -> Self {
        Self::from_keyword("")
    }

    pub fn from_keyword(kw: &str) -> Self {
        let mut used = [false; 26];
        let mut letters = [0u8; 26];
        let mut n = 0;
        for c in kw.bytes().filter(|c| c.is_ascii_alphabetic()).map(|c| c.to_ascii_uppercase() - b'A').chain(0..26) {
            if !used[c as usize] {
                used[c as usize] = true;
                letters[n] = c;
                n += 1;
            }
        }
        let mut index = [0u8; 26];
        for (i, &l) in letters.iter().enumerate() {
            index[l as usize] = i as u8;
        }
        Alphabet { letters, index, keyword: kw.to_ascii_uppercase() }
    }

    pub fn as_string(&self) -> String {
        crate::text::unscrub(&self.letters)
    }
}

#[derive(Clone, Debug)]
pub struct VigenereSolution {
    pub period: usize,
    /// Key letters (0..26), i.e. the alphabet letter at each column's shift.
    pub key: Vec<u8>,
    pub alphabet: String,
    pub plain: Vec<u8>,
    pub score: f32,
}

impl VigenereSolution {
    pub fn per_letter(&self) -> f32 {
        self.score / self.plain.len().max(1) as f32
    }
}

/// Decrypt with shifts given as alphabet positions per column.
pub fn vigenere_decrypt(cipher: &[u8], shifts: &[u8], a: &Alphabet) -> Vec<u8> {
    cipher
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let s = shifts[i % shifts.len()];
            a.letters[((a.index[c as usize] + 26 - s) % 26) as usize]
        })
        .collect()
}

/// Encrypt (for tests and tooling). `key` holds key letters.
pub fn vigenere_encrypt(plain: &[u8], key: &[u8], a: &Alphabet) -> Vec<u8> {
    plain
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let k = a.index[key[i % key.len()] as usize];
            a.letters[((a.index[p as usize] + k) % 26) as usize]
        })
        .collect()
}

const POLISH: usize = 3;
/// Below this many letters, likely periods also get simulated annealing.
const SHORT_TEXT: usize = 120;
const ANNEAL_RUNS: usize = 8;
const ANNEAL_ITERS: usize = 50_000;

/// N-gram size for hill climbing: short texts need 5-grams to separate English from
/// junk; long texts are well served by cache-friendly quadgrams.
pub fn climb_ngram_size(lm: &LangModel, letters: usize) -> usize {
    (if letters < 150 { 5 } else { 4 }).min(lm.order() + 1)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Hill-climb the column shifts for a fixed period using a dense n-gram table.
/// Changing one column only affects the n-gram windows touching that column, so
/// each trial is scored incrementally. Returns (shifts, n-gram score).
fn climb(q: &DenseNgram, cipher: &[u8], a: &Alphabet, shifts: Vec<u8>) -> (Vec<u8>, f32) {
    climb_n(q, cipher, a, shifts, usize::MAX)
}

fn climb_n(q: &DenseNgram, cipher: &[u8], a: &Alphabet, mut shifts: Vec<u8>, max_sweeps: usize) -> (Vec<u8>, f32) {
    let n = cipher.len();
    let period = shifts.len();
    let w = q.n;
    let mut buf = vigenere_decrypt(cipher, &shifts, a);
    let mut cur = q.score(&buf);
    let set_col = |buf: &mut [u8], col: usize, s: u8| {
        for i in (col..n).step_by(period) {
            buf[i] = a.letters[((a.index[cipher[i] as usize] + 26 - s) % 26) as usize];
        }
    };
    // Window end positions affected by each column (exact when period >= w; else full rescoring).
    let incremental = period >= w;
    let ends: Vec<Vec<usize>> = (0..period)
        .map(|col| {
            let mut v = Vec::new();
            for p in (col..n).step_by(period) {
                for e in p..(p + w).min(n) {
                    if e + 1 >= w {
                        v.push(e);
                    }
                }
            }
            v
        })
        .collect();
    let part = |buf: &[u8], col: usize| -> f32 { ends[col].iter().map(|&e| q.window(buf, e)).sum() };
    let mut sweeps = 0;
    loop {
        sweeps += 1;
        let mut improved = false;
        for col in 0..period {
            let orig = shifts[col];
            let base = if incremental { cur - part(&buf, col) } else { 0.0 };
            let mut col_best = (orig, cur);
            for s in 0..26u8 {
                if s == orig {
                    continue;
                }
                set_col(&mut buf, col, s);
                let sc = if incremental { base + part(&buf, col) } else { q.score(&buf) };
                if sc > col_best.1 + 1e-4 {
                    col_best = (s, sc);
                }
            }
            shifts[col] = col_best.0;
            set_col(&mut buf, col, col_best.0);
            if col_best.0 != orig {
                cur = col_best.1;
                improved = true;
            }
        }
        if !improved || sweeps >= max_sweeps {
            return (shifts, cur);
        }
    }
}

/// Coordinate ascent on the column shifts using the full language model (slower, sharper).
fn polish(lm: &LangModel, cipher: &[u8], a: &Alphabet, mut shifts: Vec<u8>) -> (Vec<u8>, f32) {
    let period = shifts.len();
    let mut buf = vigenere_decrypt(cipher, &shifts, a);
    let mut cur = lm.score(&buf);
    let set_col = |buf: &mut [u8], col: usize, s: u8| {
        for i in (col..cipher.len()).step_by(period) {
            buf[i] = a.letters[((a.index[cipher[i] as usize] + 26 - s) % 26) as usize];
        }
    };
    loop {
        let mut improved = false;
        for col in 0..period {
            let orig = shifts[col];
            let mut col_best = (orig, cur);
            for s in 0..26u8 {
                if s == orig {
                    continue;
                }
                set_col(&mut buf, col, s);
                let sc = lm.score(&buf);
                if sc > col_best.1 + 1e-4 {
                    col_best = (s, sc);
                }
            }
            shifts[col] = col_best.0;
            set_col(&mut buf, col, col_best.0);
            if col_best.0 != orig {
                cur = col_best.1;
                improved = true;
            }
        }
        if !improved {
            return (shifts, cur);
        }
    }
}

/// Simulated annealing on the column shifts with the full model (short texts).
pub fn anneal(lm: &LangModel, cipher: &[u8], a: &Alphabet, period: usize, iters: usize, seed: u64) -> (Vec<u8>, f32) {
    let mut rng = Rng(seed | 1);
    let mut shifts: Vec<u8> = (0..period).map(|_| (rng.next() % 26) as u8).collect();
    let mut buf = vigenere_decrypt(cipher, &shifts, a);
    let mut cur = lm.score(&buf);
    let mut best = (shifts.clone(), cur);
    let set_col = |buf: &mut [u8], col: usize, s: u8| {
        for i in (col..cipher.len()).step_by(period) {
            buf[i] = a.letters[((a.index[cipher[i] as usize] + 26 - s) % 26) as usize];
        }
    };
    let (t0, t1) = (4.0f32, 0.2f32);
    for it in 0..iters {
        let temp = t0 * (t1 / t0).powf(it as f32 / iters as f32);
        let col = (rng.next() % period as u64) as usize;
        let old = shifts[col];
        let s = ((old as u64 + 1 + rng.next() % 25) % 26) as u8;
        set_col(&mut buf, col, s);
        let sc = lm.score(&buf);
        let accept = sc >= cur || ((sc - cur) / temp).exp() > (rng.next() % 1_000_000) as f32 / 1e6;
        if accept {
            shifts[col] = s;
            cur = sc;
            if cur > best.1 {
                best = (shifts.clone(), cur);
            }
        } else {
            set_col(&mut buf, col, old);
        }
    }
    polish(lm, cipher, a, best.0)
}

/// Column shifts that exactly maximise the bigram log-probability of the decryption.
/// Adjacent letters fall in adjacent columns (wrapping to column 0 on the next row), so
/// the objective is a cycle of pairwise terms: fix column 0's shift, then dynamic
/// programming around the cycle. A far better starting point than per-column unigram
/// fits when columns hold only a few letters.
fn bigram_init(q2: &DenseNgram, cipher: &[u8], a: &Alphabet, p: usize) -> Vec<u8> {
    let n = cipher.len();
    let dec = |c: u8, s: u8| a.letters[((a.index[c as usize] + 26 - s) % 26) as usize] as usize;
    // pair[col][s][t]: bigram score of letters in `col` (shift s) followed by col+1 (shift t).
    let mut pair = vec![[[0f32; 26]; 26]; p];
    for j in 0..n.saturating_sub(1) {
        let col = j % p;
        for s in 0..26u8 {
            let x = dec(cipher[j], s);
            for t in 0..26u8 {
                let y = dec(cipher[j + 1], t);
                pair[col][s as usize][t as usize] += q2.value(x * 26 + y);
            }
        }
    }
    if p == 1 {
        let s = (0..26).max_by(|&x, &y| pair[0][x][x].total_cmp(&pair[0][y][y])).unwrap();
        return vec![s as u8];
    }
    let mut best = (f32::NEG_INFINITY, vec![0u8; p]);
    for s0 in 0..26usize {
        // dp[t] = best score with column `col`'s shift = t; back[col][t] = previous shift.
        let mut dp = [f32::NEG_INFINITY; 26];
        let mut back = vec![[0u8; 26]; p];
        for t in 0..26 {
            dp[t] = pair[0][s0][t];
        }
        for col in 1..p - 1 {
            let mut nd = [f32::NEG_INFINITY; 26];
            for t in 0..26 {
                for s in 0..26 {
                    let v = dp[s] + pair[col][s][t];
                    if v > nd[t] {
                        nd[t] = v;
                        back[col + 1][t] = s as u8;
                    }
                }
            }
            dp = nd;
        }
        // Close the cycle: last column back to column 0 (shift s0).
        let (mut bt, mut bv) = (0usize, f32::NEG_INFINITY);
        for t in 0..26 {
            let v = dp[t] + pair[p - 1][t][s0];
            if v > bv {
                bv = v;
                bt = t;
            }
        }
        if bv > best.0 {
            let mut sh = vec![0u8; p];
            sh[0] = s0 as u8;
            sh[p - 1] = bt as u8;
            for col in (2..p).rev() {
                sh[col - 1] = back[col][sh[col] as usize];
            }
            best = (bv, sh);
        }
    }
    best.1
}

/// Solve a periodic Vigenère cipher over the given alphabet, trying every period
/// up to `max_period`. Hill climbing uses a fast quadgram table with random restarts;
/// candidates are then rescored with the full language model and compared with a
/// penalty of ln(26) per key letter, so multiples of the true period do not win by
/// over-fitting.
pub fn solve_vigenere(lm: &LangModel, cipher: &[u8], a: &Alphabet, max_period: usize, restarts: usize) -> Vec<VigenereSolution> {
    let q = lm.dense(climb_ngram_size(lm, cipher.len()));
    solve_vigenere_with(lm, &q, cipher, a, max_period, restarts)
}

/// As [`solve_vigenere`] with a pre-built dense table (reuse it across alphabets).
pub fn solve_vigenere_with(lm: &LangModel, q: &DenseNgram, cipher: &[u8], a: &Alphabet, max_period: usize, restarts: usize) -> Vec<VigenereSolution> {
    let n = cipher.len();
    if n == 0 {
        return vec![];
    }
    let uni = lm.row(0, 0);
    let deq = lm.deq();
    let pen = (26f32).ln();
    let q2 = lm.dense(2);
    let ic_top: Vec<usize> = {
        let mut ic = period_ic(cipher, max_period);
        ic.sort_by(|a, b| b.1.total_cmp(&a.1));
        ic.into_iter().take(3).map(|x| x.0).collect()
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out: Vec<VigenereSolution> = Vec::new();
    for period in 1..=max_period.min(n) {
        // Initialise each column with its best shift under the unigram model.
        let init: Vec<u8> = (0..period)
            .map(|col| {
                let f = |s: u8| -> f32 {
                    cipher
                        .iter()
                        .skip(col)
                        .step_by(period)
                        .map(|&c| deq[uni[a.letters[((a.index[c as usize] + 26 - s) % 26) as usize] as usize] as usize])
                        .sum()
                };
                (0..26u8).max_by(|&x, &y| f(x).total_cmp(&f(y))).unwrap()
            })
            .collect();
        // Fast n-gram climbs from the unigram start, the exact bigram-chain optimum,
        // and random restarts...
        let mut cands = vec![climb(q, cipher, a, init), climb(q, cipher, a, bigram_init(&q2, cipher, a, period))];
        for _ in 0..restarts {
            let start: Vec<u8> = (0..period).map(|_| (rng.next() % 26) as u8).collect();
            cands.push(climb(q, cipher, a, start));
        }
        cands.sort_by(|x, y| y.1.total_cmp(&x.1));
        cands.dedup_by(|x, y| x.0 == y.0);
        // ...then polish the most promising with the full model (short texts need it).
        let mut best: (Vec<u8>, f32) = (vec![], f32::NEG_INFINITY);
        for c in cands.into_iter().take(if n < 150 { POLISH } else { 1 }) {
            let r = polish(lm, cipher, a, c.0);
            if r.1 > best.1 {
                best = r;
            }
        }
        // Very short texts: hill climbing gets trapped, so the most likely periods (by
        // index of coincidence) also get several simulated-annealing runs.
        if n < SHORT_TEXT && ic_top.contains(&period) {
            let runs: Vec<(Vec<u8>, f32)> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..ANNEAL_RUNS)
                    .map(|run| sc.spawn(move || anneal(lm, cipher, a, period, ANNEAL_ITERS, 0x5DEE_CE66_D ^ ((period as u64) << 32) ^ run as u64)))
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for r in runs {
                if r.1 > best.1 {
                    best = r;
                }
            }
        }
        let plain = vigenere_decrypt(cipher, &best.0, a);
        out.push(VigenereSolution {
            period,
            key: best.0.iter().map(|&s| a.letters[s as usize]).collect(),
            alphabet: a.as_string(),
            score: best.1,
            plain,
        });
    }
    // Drop solutions whose key just repeats a shorter key (e.g. ABSCISSAABSCISSA).
    out.retain(|s| !(1..s.period).any(|d| s.period % d == 0 && (d..s.period).all(|i| s.key[i] == s.key[i - d])));
    out.sort_by(|x, y| (y.score - y.period as f32 * pen).total_cmp(&(x.score - x.period as f32 * pen)));
    out
}

/// Mean normalised index of coincidence (x26) of the columns for each period; English ≈ 1.7,
/// random ≈ 1.0. Independent of the alphabet, so it ranks periods before any alphabet search.
pub fn period_ic(cipher: &[u8], max_period: usize) -> Vec<(usize, f64)> {
    (1..=max_period.min(cipher.len() / 2).max(1))
        .map(|p| {
            let mut total = 0.0;
            let mut cols = 0;
            for col in 0..p {
                let mut f = [0u32; 26];
                let mut n = 0u32;
                for &c in cipher.iter().skip(col).step_by(p) {
                    f[c as usize] += 1;
                    n += 1;
                }
                if n > 1 {
                    let s: u32 = f.iter().map(|&x| x * x.saturating_sub(1)).sum();
                    total += 26.0 * s as f64 / (n * (n - 1)) as f64;
                    cols += 1;
                }
            }
            (p, if cols > 0 { total / cols as f64 } else { 0.0 })
        })
        .collect()
}

/// Mixed alphabet ranked by how well a quick climb decrypts the cipher with it.
#[derive(Clone, Debug)]
pub struct AlphabetHit {
    pub keyword: String,
    pub alphabet: Alphabet,
    pub period: usize,
    /// Mean n-gram log-prob per letter after climbing, less ln(26)/n per key letter.
    pub score: f32,
}

/// Rank candidate alphabet keywords for a Quagmire III (keyed-alphabet Vigenère) cipher.
/// For each distinct alphabet and each candidate period: n-gram hill climbing from the
/// unigram start plus `restarts` random starts. Runs on all cores. Practical for lists of
/// up to a few thousand keywords; a full dictionary takes minutes.
pub fn rank_alphabets(lm: &LangModel, q: &DenseNgram, cipher: &[u8], keywords: &[String], periods: &[usize], restarts: usize, top: usize) -> Vec<AlphabetHit> {
    // Distinct alphabets only (keywords with the same de-duplicated letters coincide).
    let mut seen = std::collections::HashSet::new();
    let alphas: Vec<(String, Alphabet)> = keywords
        .iter()
        .filter_map(|k| {
            let a = Alphabet::from_keyword(k);
            seen.insert(a.letters).then(|| (k.to_ascii_uppercase(), a))
        })
        .collect();
    let uni: Vec<f32> = lm.row(0, 0).iter().map(|&x| lm.deq()[x as usize]).collect();
    // Per period, per column: (letter, count) pairs.
    let hists: Vec<Vec<Vec<(u8, f32)>>> = periods
        .iter()
        .map(|&p| {
            (0..p)
                .map(|col| {
                    let mut f = [0u32; 26];
                    cipher.iter().skip(col).step_by(p).for_each(|&c| f[c as usize] += 1);
                    (0..26u8).filter(|&c| f[c as usize] > 0).map(|c| (c, f[c as usize] as f32)).collect()
                })
                .collect()
        })
        .collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let chunk = alphas.len().div_ceil(threads).max(1);
    let n = cipher.len().max(1) as f32;
    let mut hits: Vec<AlphabetHit> = std::thread::scope(|sc| {
        let hs: Vec<_> = alphas
            .chunks(chunk)
            .map(|part| {
                let (uni, hists) = (&uni, &hists);
                sc.spawn(move || {
                    let mut out: Vec<AlphabetHit> = Vec::new();
                    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
                    for (kw, a) in part {
                        for (pi, &p) in periods.iter().enumerate() {
                            // Unigram start per column from the column's letter histogram.
                            let init: Vec<u8> = (0..p)
                                .map(|col| {
                                    let h = &hists[pi][col];
                                    let f = |s: u8| -> f32 {
                                        h.iter().map(|&(c, k)| k * uni[a.letters[((a.index[c as usize] + 26 - s) % 26) as usize] as usize]).sum()
                                    };
                                    (0..26u8).max_by(|&x, &y| f(x).total_cmp(&f(y))).unwrap()
                                })
                                .collect();
                            let mut best = climb(q, cipher, a, init);
                            for _ in 0..restarts {
                                let start: Vec<u8> = (0..p).map(|_| (rng.next() % 26) as u8).collect();
                                let r = climb(q, cipher, a, start);
                                if r.1 > best.1 {
                                    best = r;
                                }
                            }
                            let sc = best.1;
                            // Penalise long keys (ln 26 per key letter) so short texts don't over-fit.
                            out.push(AlphabetHit { keyword: kw.clone(), alphabet: a.clone(), period: p, score: (sc - p as f32 * 26f32.ln()) / n });
                        }
                        if out.len() > top * 8 {
                            out.sort_by(|x, y| y.score.total_cmp(&x.score));
                            out.truncate(top * 2);
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    hits.sort_by(|x, y| y.score.total_cmp(&x.score));
    // Keep the best period per alphabet.
    let mut kept = std::collections::HashSet::new();
    hits.retain(|h| kept.insert(h.alphabet.letters));
    hits.truncate(top);
    hits
}

/// Solve a keyed-alphabet Vigenère when the alphabet keyword is unknown but is one of
/// `keywords`. Periods come from the index of coincidence (alphabet-independent); every
/// candidate alphabet is ranked by a quick climb, then the best few get the full solver.
/// Short texts automatically get more random restarts (they need them).
pub fn solve_vigenere_keyword_search(lm: &LangModel, cipher: &[u8], keywords: &[String], max_period: usize, finalists: usize) -> (Vec<AlphabetHit>, Vec<VigenereSolution>) {
    let mut ic = period_ic(cipher, max_period);
    ic.sort_by(|a, b| b.1.total_cmp(&a.1));
    let periods: Vec<usize> = ic.iter().take(3).map(|x| x.0).collect();
    let q = lm.dense(climb_ngram_size(lm, cipher.len()));
    let restarts = if cipher.len() < 150 { 20 } else { 2 };
    let ranked = rank_alphabets(lm, &q, cipher, keywords, &periods, restarts, finalists.max(1));
    let pen = 26f32.ln();
    let mut sols: Vec<VigenereSolution> = ranked
        .iter()
        .flat_map(|h| solve_vigenere_with(lm, &q, cipher, &h.alphabet, max_period, 30).into_iter().take(1))
        .collect();
    sols.sort_by(|x, y| (y.score - y.period as f32 * pen).total_cmp(&(x.score - x.period as f32 * pen)));
    (ranked, sols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{scrub, unscrub};

    #[test]
    fn keyed_alphabet() {
        assert_eq!(Alphabet::from_keyword("KRYPTOS").as_string(), "KRYPTOSABCDEFGHIJLMNQUVWXZ");
        assert_eq!(Alphabet::standard().as_string(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }

    #[test]
    fn roundtrip() {
        let a = Alphabet::from_keyword("KRYPTOS");
        let p = scrub("betweensubtleshading");
        let k = scrub("palimpsest");
        let c = vigenere_encrypt(&p, &k, &a);
        let shifts: Vec<u8> = k.iter().map(|&x| a.index[x as usize]).collect();
        assert_eq!(unscrub(&vigenere_decrypt(&c, &shifts, &a)), "BETWEENSUBTLESHADING");
    }
}
