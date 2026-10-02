//! Digraph and fractionating ciphers on a 5×5 key square (I/J merged): Playfair and
//! Bifid, recovered by simulated annealing, plus the 2×2 Hill cipher (exhaustive).

use crate::lm::{DenseNgram, LangModel};
use crate::rng::Rng;

const I: u8 = 8;
const J: u8 = 9;

/// Merge J into I.
pub fn fold_j(letters: &[u8]) -> Vec<u8> {
    letters.iter().map(|&c| if c == J { I } else { c }).collect()
}

pub type Square = [u8; 25];

pub fn square_from_keyword(kw: &str) -> Square {
    let mut sq = [0u8; 25];
    let mut used = [false; 26];
    let mut n = 0;
    for c in kw.bytes().filter(|c| c.is_ascii_alphabetic()).map(|c| c.to_ascii_uppercase() - b'A').chain(0..26) {
        let c = if c == J { I } else { c };
        if !used[c as usize] {
            used[c as usize] = true;
            sq[n] = c;
            n += 1;
        }
    }
    sq
}

pub fn square_string(sq: &Square) -> String {
    sq.chunks(5).map(|r| r.iter().map(|&c| (b'A' + c) as char).collect::<String>()).collect::<Vec<_>>().join("/")
}

fn inverse(sq: &Square) -> [u8; 26] {
    let mut inv = [0u8; 26];
    for (p, &c) in sq.iter().enumerate() {
        inv[c as usize] = p as u8;
    }
    inv[J as usize] = inv[I as usize];
    inv
}

// ---------------------------------------------------------------------------------------
// Playfair

fn playfair_pair(sq: &Square, inv: &[u8; 26], a: u8, b: u8, dir: i8, out: &mut Vec<u8>) {
    let (pa, pb) = (inv[a as usize] as i8, inv[b as usize] as i8);
    let (ra, ca, rb, cb) = (pa / 5, pa % 5, pb / 5, pb % 5);
    let m = |v: i8| ((v + 5) % 5) as usize;
    let (x, y) = if ra == rb {
        (ra as usize * 5 + m(ca + dir), rb as usize * 5 + m(cb + dir))
    } else if ca == cb {
        (m(ra + dir) * 5 + ca as usize, m(rb + dir) * 5 + cb as usize)
    } else {
        (ra as usize * 5 + cb as usize, rb as usize * 5 + ca as usize)
    };
    out.push(sq[x]);
    out.push(sq[y]);
}

/// Decrypt digraph by digraph (a trailing odd letter is dropped).
pub fn playfair_decrypt(sq: &Square, cipher: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let inv = inverse(sq);
    for p in cipher.chunks_exact(2) {
        playfair_pair(sq, &inv, p[0], p[1], -1, out);
    }
}

/// Encrypt, splitting doubled letters with X and padding with X as usual.
pub fn playfair_encrypt(sq: &Square, plain: &[u8]) -> Vec<u8> {
    let inv = inverse(sq);
    let plain = fold_j(plain);
    let mut digraphs = Vec::new();
    let mut i = 0;
    while i < plain.len() {
        let a = plain[i];
        let b = plain.get(i + 1).copied().filter(|&b| b != a);
        match b {
            Some(b) => {
                digraphs.push((a, b));
                i += 2;
            }
            None => {
                digraphs.push((a, 23));
                i += 1;
            }
        }
    }
    let mut out = Vec::new();
    for (a, b) in digraphs {
        playfair_pair(sq, &inv, a, b, 1, &mut out);
    }
    out
}

// ---------------------------------------------------------------------------------------
// Bifid

/// Bifid decryption with block size `period` (0 = whole message).
pub fn bifid_decrypt(sq: &Square, cipher: &[u8], period: usize, out: &mut Vec<u8>) {
    out.clear();
    let inv = inverse(sq);
    let block = if period == 0 { cipher.len().max(1) } else { period };
    for chunk in cipher.chunks(block) {
        let m = chunk.len();
        let mut digits = Vec::with_capacity(2 * m);
        for &c in chunk {
            let p = inv[c as usize];
            digits.push(p / 5);
            digits.push(p % 5);
        }
        for i in 0..m {
            out.push(sq[(digits[i] * 5 + digits[m + i]) as usize]);
        }
    }
}

pub fn bifid_encrypt(sq: &Square, plain: &[u8], period: usize) -> Vec<u8> {
    let inv = inverse(sq);
    let block = if period == 0 { plain.len().max(1) } else { period };
    let mut out = Vec::new();
    for chunk in fold_j(plain).chunks(block) {
        let rows: Vec<u8> = chunk.iter().map(|&c| inv[c as usize] / 5).collect();
        let cols: Vec<u8> = chunk.iter().map(|&c| inv[c as usize] % 5).collect();
        let seq: Vec<u8> = rows.into_iter().chain(cols).collect();
        for pair in seq.chunks(2) {
            out.push(sq[(pair[0] * 5 + pair[1]) as usize]);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Square annealing

#[derive(Clone, Debug)]
pub struct SquareSolution {
    pub square: Square,
    pub plain: Vec<u8>,
    pub score: f32,
    /// Bifid period (0 for Playfair / whole message).
    pub period: usize,
}

fn mutate(sq: &mut Square, rng: &mut Rng) {
    match rng.below(50) {
        0 => {
            // swap two rows
            let (a, b) = (rng.below(5), rng.below(5));
            for c in 0..5 {
                sq.swap(a * 5 + c, b * 5 + c);
            }
        }
        1 => {
            let (a, b) = (rng.below(5), rng.below(5));
            for r in 0..5 {
                sq.swap(r * 5 + a, r * 5 + b);
            }
        }
        2 => sq.reverse(),
        _ => sq.swap(rng.below(25), rng.below(25)),
    }
}

fn anneal_square(q: &DenseNgram, decrypt: &dyn Fn(&Square, &mut Vec<u8>), iters: usize, rng: &mut Rng) -> (Square, f32) {
    let mut sq = square_from_keyword("");
    for i in (1..25).rev() {
        sq.swap(i, rng.below(i + 1));
    }
    let mut buf = Vec::new();
    decrypt(&sq, &mut buf);
    let mut cur = q.score(&buf);
    let mut best = (sq, cur);
    let (t0, t1) = (20.0f32, 1.0f32);
    for it in 0..iters {
        let temp = t0 * (t1 / t0).powf(it as f32 / iters as f32);
        let mut cand = sq;
        mutate(&mut cand, rng);
        decrypt(&cand, &mut buf);
        let sc = q.score(&buf);
        if sc >= cur || ((sc - cur) / temp).exp() > rng.unit() {
            sq = cand;
            cur = sc;
            if cur > best.1 {
                best = (sq, cur);
            }
        }
    }
    best
}

fn best_square(q: &DenseNgram, decrypt: &dyn Fn(&Square, &mut Vec<u8>), iters: usize, restarts: usize, seed: u64) -> (Square, f32) {
    let mut rng = Rng::new(seed);
    let mut best: Option<(Square, f32)> = None;
    for _ in 0..restarts.max(1) {
        let r = anneal_square(q, decrypt, iters, &mut rng);
        if best.as_ref().is_none_or(|b| r.1 > b.1) {
            best = Some(r);
        }
    }
    best.unwrap()
}

/// Recover a Playfair key square. `cipher` letters are folded J→I.
pub fn solve_playfair(lm: &LangModel, q: &DenseNgram, cipher: &[u8], iters: usize, restarts: usize, seed: u64) -> SquareSolution {
    let cipher = fold_j(cipher);
    let dec = |sq: &Square, out: &mut Vec<u8>| playfair_decrypt(sq, &cipher, out);
    let (square, _) = best_square(q, &dec, iters, restarts, seed);
    let mut plain = Vec::new();
    playfair_decrypt(&square, &cipher, &mut plain);
    SquareSolution { square, score: lm.score(&plain), plain, period: 0 }
}

/// Recover a Bifid key square for each candidate period (0 = whole message); best first.
pub fn solve_bifid(lm: &LangModel, q: &DenseNgram, cipher: &[u8], periods: &[usize], iters: usize, restarts: usize, seed: u64) -> Vec<SquareSolution> {
    let cipher = fold_j(cipher);
    let mut out: Vec<SquareSolution> = periods
        .iter()
        .map(|&period| {
            let dec = |sq: &Square, out: &mut Vec<u8>| bifid_decrypt(sq, &cipher, period, out);
            let (square, _) = best_square(q, &dec, iters, restarts, seed ^ period as u64);
            let mut plain = Vec::new();
            bifid_decrypt(&square, &cipher, period, &mut plain);
            SquareSolution { square, score: lm.score(&plain), plain, period }
        })
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

// ---------------------------------------------------------------------------------------
// Hill 2x2

#[derive(Clone, Debug)]
pub struct HillSolution {
    /// Encryption matrix `[a, b, c, d]`: `(c1, c2) = (a*p1 + b*p2, c*p1 + d*p2)`.
    pub matrix: [u8; 4],
    pub plain: Vec<u8>,
    pub score: f32,
}

fn modinv(a: i32) -> Option<i32> {
    (1..26).find(|&x| (a.rem_euclid(26) * x) % 26 == 1)
}

pub fn hill_encrypt(plain: &[u8], m: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in plain.chunks_exact(2) {
        let (x, y) = (p[0] as u32, p[1] as u32);
        out.push(((m[0] as u32 * x + m[1] as u32 * y) % 26) as u8);
        out.push(((m[2] as u32 * x + m[3] as u32 * y) % 26) as u8);
    }
    out
}

/// Decryption matrix for an encryption matrix, if invertible mod 26.
pub fn hill_inverse(m: [u8; 4]) -> Option<[u8; 4]> {
    let det = m[0] as i32 * m[3] as i32 - m[1] as i32 * m[2] as i32;
    let inv = modinv(det)?;
    let f = |v: i32| ((v * inv).rem_euclid(26)) as u8;
    Some([f(m[3] as i32), f(-(m[1] as i32)), f(-(m[2] as i32)), f(m[0] as i32)])
}

/// Try every invertible 2×2 key (157,248 of them); returns the best `top` by n-gram score.
pub fn solve_hill2(lm: &LangModel, q: &DenseNgram, cipher: &[u8], top: usize) -> Vec<HillSolution> {
    let mut cands: Vec<(f32, [u8; 4])> = Vec::new();
    let mut buf = Vec::with_capacity(cipher.len());
    for a in 0..26u8 {
        for b in 0..26u8 {
            for c in 0..26u8 {
                for d in 0..26u8 {
                    let Some(inv) = hill_inverse([a, b, c, d]) else { continue };
                    buf.clear();
                    buf.extend(hill_encrypt(cipher, inv));
                    cands.push((q.score(&buf), [a, b, c, d]));
                }
            }
        }
    }
    cands.sort_by(|x, y| y.0.total_cmp(&x.0));
    cands
        .into_iter()
        .take(top)
        .map(|(_, m)| {
            let plain = hill_encrypt(cipher, hill_inverse(m).unwrap());
            HillSolution { matrix: m, score: lm.score(&plain), plain }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{model, sample};
    use crate::text::{scrub, unscrub};

    fn agreement(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len().min(b.len());
        a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / n as f64
    }

    #[test]
    fn playfair_known_example() {
        // Wikipedia: key PLAYFAIR EXAMPLE, "HIDETHEGOLDINTHETREESTUMP" -> BMODZBXDNABEKUDMUIXMMOUVIF
        let sq = square_from_keyword("PLAYFAIREXAMPLE");
        assert_eq!(unscrub(&playfair_encrypt(&sq, &scrub("hidethegoldinthetreestump"))), "BMODZBXDNABEKUDMUIXMMOUVIF");
        let mut p = Vec::new();
        playfair_decrypt(&sq, &scrub("BMODZBXDNABEKUDMUIXMMOUVIF"), &mut p);
        assert_eq!(unscrub(&p), "HIDETHEGOLDINTHETREXESTUMP");
    }

    #[test]
    fn bifid_roundtrip() {
        let sq = square_from_keyword("BGWKZQPNDSIOAXEFCLUMTHYVR");
        let p = fold_j(&sample(100, 61));
        for period in [0, 5, 7] {
            let mut out = Vec::new();
            bifid_decrypt(&sq, &bifid_encrypt(&sq, &p, period), period, &mut out);
            assert_eq!(out, p, "period {period}");
        }
    }

    #[test]
    fn hill_roundtrip_and_solve() {
        let lm = model();
        let q = lm.dense(4);
        let p = sample(40_000, 200);
        let m = [3, 3, 2, 5];
        let c = hill_encrypt(&p, m);
        assert_eq!(hill_encrypt(&c, hill_inverse(m).unwrap()), p);
        assert_eq!(solve_hill2(lm, &q, &c, 1)[0].plain, p);
    }

    #[test]
    fn solves_playfair() {
        let lm = model();
        let q = lm.dense(4);
        let sq = square_from_keyword("MONARCHYBDEFGHIKLNPQSTUVWXZ");
        let p = sample(50_000, 500);
        let c = playfair_encrypt(&sq, &p);
        let mut truth = Vec::new();
        playfair_decrypt(&sq, &c, &mut truth);
        let sol = solve_playfair(lm, &q, &c, 200_000, 6, 1);
        let acc = agreement(&sol.plain, &truth);
        assert!(acc > 0.95, "playfair accuracy {acc}");
        let c = playfair_encrypt(&square_from_keyword("EXAMPLEKYBCDFGHILNOQRSTUVWZ"), &sample(120_000, 400));
        let sol = solve_playfair(lm, &q, &c, 200_000, 6, 7);
        let mut truth = Vec::new();
        playfair_decrypt(&square_from_keyword("EXAMPLEKYBCDFGHILNOQRSTUVWZ"), &c, &mut truth);
        let acc = agreement(&sol.plain, &truth);
        assert!(acc > 0.95, "playfair accuracy (2nd) {acc}");
    }

    #[test]
    fn solves_bifid() {
        let lm = model();
        let q = lm.dense(4);
        let sq = square_from_keyword("BGWKZQPNDSIOAXEFCLUMTHYVR");
        let p = fold_j(&sample(60_000, 400));
        let c = bifid_encrypt(&sq, &p, 7);
        let sol = &solve_bifid(lm, &q, &c, &[5, 7, 9], 60_000, 4, 1)[0];
        assert_eq!(sol.period, 7);
        assert!(agreement(&sol.plain, &p) > 0.95);
    }
}
