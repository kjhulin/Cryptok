//! Transposition ciphers.
//!
//! * **Route transpositions** — write the text into rows of width `w`, read it out by
//!   columns (left/right, top/bottom), or the inverse. One or two such steps are
//!   brute-forced over all widths and directions. Kryptos K3 is two of these.
//! * **Keyed columnar transposition** — columns read in a secret order (incomplete last
//!   row allowed). The column order is recovered by hill climbing for each key length.
//!
//! Non-letter characters (e.g. Kryptos' `?`) are transposed along with the letters; only
//! letters are scored.

use crate::lm::{DenseNgram, LangModel};
use crate::text::scrub;

/// One route step: rows of `width`, columns read in the given directions.
/// `inverse` applies the step backwards (fill by columns, read by rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Route {
    pub width: usize,
    pub right_to_left: bool,
    pub bottom_to_top: bool,
    pub inverse: bool,
}

impl Route {
    pub fn describe(&self) -> String {
        format!(
            "{} rows of {}, columns {} {}",
            if self.inverse { "un-read" } else { "read" },
            self.width,
            if self.right_to_left { "right→left" } else { "left→right" },
            if self.bottom_to_top { "bottom→top" } else { "top→bottom" }
        )
    }

    /// Order in which positions of the row-major grid are visited.
    fn order(&self, n: usize) -> Vec<usize> {
        let w = self.width;
        let rows = n.div_ceil(w);
        let mut out = Vec::with_capacity(n);
        for ci in 0..w {
            let c = if self.right_to_left { w - 1 - ci } else { ci };
            for ri in 0..rows {
                let r = if self.bottom_to_top { rows - 1 - ri } else { ri };
                let i = r * w + c;
                if i < n {
                    out.push(i);
                }
            }
        }
        out
    }

    pub fn apply<T: Copy>(&self, s: &[T]) -> Vec<T> {
        let ord = self.order(s.len());
        if self.inverse {
            let mut out = s.to_vec();
            for (k, &i) in ord.iter().enumerate() {
                out[i] = s[k];
            }
            out
        } else {
            ord.iter().map(|&i| s[i]).collect()
        }
    }

    fn all(n: usize, max_width: usize) -> Vec<Route> {
        let mut v = Vec::new();
        for width in 2..=max_width.min(n.saturating_sub(1)) {
            for bits in 0..8u8 {
                v.push(Route { width, right_to_left: bits & 1 != 0, bottom_to_top: bits & 2 != 0, inverse: bits & 4 != 0 });
            }
        }
        v
    }
}

#[derive(Clone, Debug)]
pub enum Method {
    Route(Vec<Route>),
    Columnar { order: Vec<usize> },
}

#[derive(Clone, Debug)]
pub struct TranspositionSolution {
    pub method: Method,
    /// The rearranged text, non-letters included.
    pub text: String,
    /// Mean full-model log-prob per letter.
    pub per_letter: f32,
}

impl TranspositionSolution {
    pub fn describe(&self) -> String {
        match &self.method {
            Method::Route(steps) => steps.iter().map(|r| r.describe()).collect::<Vec<_>>().join(", then "),
            Method::Columnar { order } => {
                let key: String = (0..order.len())
                    .map(|i| (b'A' + order.iter().position(|&c| c == i).unwrap() as u8) as char)
                    .collect();
                format!("columnar, {} columns, column order {}", order.len(), key)
            }
        }
    }
}

fn letters_of(chars: &[char]) -> Vec<u8> {
    scrub(&chars.iter().collect::<String>())
}

/// Brute-force one and two route steps. Returns the best `top` by full-model score.
pub fn solve_route(lm: &LangModel, q: &DenseNgram, text: &str, max_width: usize, top: usize) -> Vec<TranspositionSolution> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    let n = chars.len();
    if n < 4 {
        return vec![];
    }
    // Work on indices so one permutation serves letters and symbols alike.
    let idx: Vec<u32> = (0..n as u32).collect();
    let is_letter: Vec<bool> = chars.iter().map(|c| c.is_ascii_alphabetic()).collect();
    let letter: Vec<u8> = chars.iter().map(|c| (c.to_ascii_uppercase() as u8).wrapping_sub(b'A')).collect();
    let routes = Route::all(n, max_width);
    let firsts: Vec<(Route, Vec<u32>)> = routes.iter().map(|r| (*r, r.apply(&idx))).collect();
    let score = |perm: &[u32], buf: &mut Vec<u8>| -> f32 {
        buf.clear();
        buf.extend(perm.iter().filter(|&&i| is_letter[i as usize]).map(|&i| letter[i as usize]));
        q.score(buf) / buf.len().max(1) as f32
    };

    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let chunk = firsts.len().div_ceil(threads).max(1);
    let mut cands: Vec<(f32, Vec<Route>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = firsts
            .chunks(chunk)
            .map(|part| {
                let (routes, score) = (&routes, &score);
                sc.spawn(move || {
                    let mut buf = Vec::with_capacity(n);
                    let mut best: Vec<(f32, Vec<Route>)> = Vec::new();
                    let push = |s: f32, m: Vec<Route>, best: &mut Vec<(f32, Vec<Route>)>| {
                        best.push((s, m));
                        if best.len() > 64 {
                            best.sort_by(|a, b| b.0.total_cmp(&a.0));
                            best.truncate(16);
                        }
                    };
                    for (r1, p1) in part {
                        push(score(p1, &mut buf), vec![*r1], &mut best);
                        for r2 in routes.iter() {
                            let p2 = r2.apply(p1);
                            push(score(&p2, &mut buf), vec![*r1, *r2], &mut best);
                        }
                    }
                    best
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Rescore the leaders with the full model; drop duplicates (many routes coincide).
    let mut out: Vec<TranspositionSolution> = Vec::new();
    for (_, steps) in cands.into_iter().take(top * 8) {
        let mut perm = idx.clone();
        for r in &steps {
            perm = r.apply(&perm);
        }
        let t: Vec<char> = perm.iter().map(|&i| chars[i as usize]).collect();
        let text: String = t.iter().collect();
        if out.iter().any(|o| o.text == text) {
            continue;
        }
        let l = letters_of(&t);
        out.push(TranspositionSolution { method: Method::Route(steps), text, per_letter: lm.score_per_letter(&l) });
    }
    out.sort_by(|a, b| b.per_letter.total_cmp(&a.per_letter));
    out.truncate(top);
    out
}

/// Decrypt a keyed columnar transposition: `order[k]` is the grid column read k-th.
pub fn columnar_decrypt<T: Copy + Default>(c: &[T], order: &[usize]) -> Vec<T> {
    let k = order.len();
    let n = c.len();
    let rows = n.div_ceil(k);
    let full = if n % k == 0 { k } else { n % k }; // columns with `rows` entries
    let mut out = vec![T::default(); n];
    let mut pos = 0;
    for &col in order {
        let len = if col < full { rows } else { rows - 1 };
        for r in 0..len {
            out[r * k + col] = c[pos + r];
        }
        pos += len;
    }
    out
}

/// Recover a keyed columnar transposition by hill climbing the column order for each
/// key length in `min_cols..=max_cols`.
pub fn solve_columnar(lm: &LangModel, q: &DenseNgram, text: &str, min_cols: usize, max_cols: usize, restarts: usize, top: usize) -> Vec<TranspositionSolution> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    let n = chars.len();
    let idx: Vec<u32> = (0..n as u32).collect();
    let is_letter: Vec<bool> = chars.iter().map(|c| c.is_ascii_alphabetic()).collect();
    let letter: Vec<u8> = chars.iter().map(|c| (c.to_ascii_uppercase() as u8).wrapping_sub(b'A')).collect();
    let mut buf = Vec::with_capacity(n);
    let mut eval = |order: &[usize]| -> f32 {
        let p = columnar_decrypt(&idx, order);
        buf.clear();
        buf.extend(p.iter().filter(|&&i| is_letter[i as usize]).map(|&i| letter[i as usize]));
        q.score(&buf)
    };
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand = move |m: usize| -> usize {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % m as u64) as usize
    };
    let mut out = Vec::new();
    for k in min_cols.max(2)..=max_cols.min(n / 2) {
        let mut best: (Vec<usize>, f32) = (vec![], f32::NEG_INFINITY);
        for _ in 0..=restarts {
            let mut order: Vec<usize> = (0..k).collect();
            for i in (1..k).rev() {
                order.swap(i, rand(i + 1));
            }
            let mut cur = eval(&order);
            loop {
                let mut improved = false;
                // Swap two columns.
                for i in 0..k {
                    for j in i + 1..k {
                        order.swap(i, j);
                        let s = eval(&order);
                        if s > cur + 1e-3 {
                            cur = s;
                            improved = true;
                        } else {
                            order.swap(i, j);
                        }
                    }
                }
                // Move one column to another position.
                for i in 0..k {
                    for j in 0..k {
                        if i == j {
                            continue;
                        }
                        let c = order.remove(i);
                        order.insert(j, c);
                        let s = eval(&order);
                        if s > cur + 1e-3 {
                            cur = s;
                            improved = true;
                        } else {
                            let c = order.remove(j);
                            order.insert(i, c);
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
            if cur > best.1 {
                best = (order, cur);
            }
        }
        let p = columnar_decrypt(&idx, &best.0);
        let t: Vec<char> = p.iter().map(|&i| chars[i as usize]).collect();
        out.push(TranspositionSolution {
            method: Method::Columnar { order: best.0 },
            per_letter: lm.score_per_letter(&letters_of(&t)),
            text: t.into_iter().collect(),
        });
    }
    out.sort_by(|a, b| b.per_letter.total_cmp(&a.per_letter));
    out.truncate(top);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_inverse_roundtrip() {
        let s: Vec<u32> = (0..37).collect();
        for r in Route::all(37, 10) {
            let mut inv = r;
            inv.inverse = !r.inverse;
            assert_eq!(inv.apply(&r.apply(&s)), s, "{r:?}");
        }
    }

    #[test]
    fn columnar_roundtrip() {
        // Encrypt by reading grid columns in `order`, then decrypt.
        let p: Vec<u32> = (0..23).collect();
        let order = vec![2, 0, 3, 1];
        let k = order.len();
        let mut c = Vec::new();
        for &col in &order {
            let mut i = col;
            while i < p.len() {
                c.push(p[i]);
                i += k;
            }
        }
        assert_eq!(columnar_decrypt(&c, &order), p);
    }
}
