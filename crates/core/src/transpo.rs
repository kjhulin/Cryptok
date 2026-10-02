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
    /// Two keyed columnar transpositions in a row; `first` was applied first when encrypting.
    DoubleColumnar { first: Vec<usize>, second: Vec<usize> },
    RailFence { rails: usize, offset: usize },
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
            Method::DoubleColumnar { first, second } => {
                format!("double columnar, keys {} then {}", order_key(first), order_key(second))
            }
            Method::RailFence { rails, offset } => format!("rail fence, {rails} rails, offset {offset}"),
        }
    }
}

fn order_key(order: &[usize]) -> String {
    (0..order.len()).map(|i| (b'A' + order.iter().position(|&c| c == i).unwrap() as u8) as char).collect()
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

/// Rail index of each position in a zig-zag over `rails` rails, starting `offset` steps in.
fn rail_pattern(n: usize, rails: usize, offset: usize) -> Vec<usize> {
    let period = 2 * (rails - 1);
    (0..n)
        .map(|i| {
            let t = (i + offset) % period;
            if t < rails { t } else { period - t }
        })
        .collect()
}

/// Rail fence decryption (`rails >= 2`).
pub fn rail_fence_decrypt<T: Copy + Default>(c: &[T], rails: usize, offset: usize) -> Vec<T> {
    let n = c.len();
    let pat = rail_pattern(n, rails, offset);
    let mut pos: Vec<usize> = (0..n).collect();
    pos.sort_by_key(|&i| (pat[i], i)); // ciphertext is read rail by rail
    let mut out = vec![T::default(); n];
    for (k, &i) in pos.iter().enumerate() {
        out[i] = c[k];
    }
    out
}

pub fn rail_fence_encrypt<T: Copy>(p: &[T], rails: usize, offset: usize) -> Vec<T> {
    let pat = rail_pattern(p.len(), rails, offset);
    let mut pos: Vec<usize> = (0..p.len()).collect();
    pos.sort_by_key(|&i| (pat[i], i));
    pos.into_iter().map(|i| p[i]).collect()
}

/// Brute-force every rail count up to `max_rails` and every starting offset.
pub fn solve_rail_fence(lm: &LangModel, text: &str, max_rails: usize, top: usize) -> Vec<TranspositionSolution> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::new();
    for rails in 2..=max_rails.min(chars.len().saturating_sub(1)).max(2) {
        for offset in 0..2 * (rails - 1) {
            let t = rail_fence_decrypt(&chars, rails, offset);
            out.push(TranspositionSolution {
                method: Method::RailFence { rails, offset },
                per_letter: lm.score_per_letter(&letters_of(&t)),
                text: t.into_iter().collect(),
            });
        }
    }
    out.sort_by(|a, b| b.per_letter.total_cmp(&a.per_letter));
    out.truncate(top);
    out
}

/// Recover a double columnar transposition by simulated annealing over both column
/// orders, for every pair of key lengths in `min_cols..=max_cols`.
pub fn solve_double_columnar(lm: &LangModel, q: &DenseNgram, text: &str, min_cols: usize, max_cols: usize, iters: usize, restarts: usize, top: usize) -> Vec<TranspositionSolution> {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    let n = chars.len();
    let idx: Vec<u32> = (0..n as u32).collect();
    let is_letter: Vec<bool> = chars.iter().map(|c| c.is_ascii_alphabetic()).collect();
    let letter: Vec<u8> = chars.iter().map(|c| (c.to_ascii_uppercase() as u8).wrapping_sub(b'A')).collect();
    let mut buf = Vec::with_capacity(n);
    let mut eval = |o1: &[usize], o2: &[usize]| -> f32 {
        let p = columnar_decrypt(&columnar_decrypt(&idx, o2), o1);
        buf.clear();
        buf.extend(p.iter().filter(|&&i| is_letter[i as usize]).map(|&i| letter[i as usize]));
        q.score(&buf)
    };
    let mut rng = crate::rng::Rng::new(0xD0B1E);
    let mut out = Vec::new();
    for k1 in min_cols.max(2)..=max_cols.min(n / 2) {
        for k2 in min_cols.max(2)..=max_cols.min(n / 2) {
            let mut best: Option<(Vec<usize>, Vec<usize>, f32)> = None;
            for _ in 0..restarts.max(1) {
                let mut o1: Vec<usize> = (0..k1).collect();
                let mut o2: Vec<usize> = (0..k2).collect();
                rng.shuffle(&mut o1);
                rng.shuffle(&mut o2);
                let mut cur = eval(&o1, &o2);
                let mut top_here = (o1.clone(), o2.clone(), cur);
                let (t0, t1) = (4.0f32, 0.1f32);
                for it in 0..iters {
                    let temp = t0 * (t1 / t0).powf(it as f32 / iters as f32);
                    let which = rng.below(k1 + k2) < k1;
                    let order = if which { &mut o1 } else { &mut o2 };
                    let len = order.len();
                    let (i, j) = (rng.below(len), rng.below(len));
                    if i == j {
                        continue;
                    }
                    let saved = order.clone();
                    if rng.below(2) == 0 {
                        order.swap(i, j);
                    } else {
                        let c = order.remove(i);
                        order.insert(j, c);
                    }
                    let sc = eval(&o1, &o2);
                    if sc >= cur || ((sc - cur) / temp).exp() > rng.unit() {
                        cur = sc;
                        if cur > top_here.2 {
                            top_here = (o1.clone(), o2.clone(), cur);
                        }
                    } else if which {
                        o1 = saved;
                    } else {
                        o2 = saved;
                    }
                }
                if best.as_ref().is_none_or(|b| top_here.2 > b.2) {
                    best = Some(top_here);
                }
            }
            let (o1, o2, _) = best.unwrap();
            let p = columnar_decrypt(&columnar_decrypt(&idx, &o2), &o1);
            let t: Vec<char> = p.iter().map(|&i| chars[i as usize]).collect();
            out.push(TranspositionSolution {
                method: Method::DoubleColumnar { first: o1, second: o2 },
                per_letter: lm.score_per_letter(&letters_of(&t)),
                text: t.into_iter().collect(),
            });
        }
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
    fn rail_fence_known_example() {
        // Classic: WEAREDISCOVEREDFLEEATONCE on 3 rails.
        let p: Vec<char> = "WEAREDISCOVEREDFLEEATONCE".chars().collect();
        let c: String = rail_fence_encrypt(&p, 3, 0).into_iter().collect();
        assert_eq!(c, "WECRLTEERDSOEEFEAOCAIVDEN");
        for rails in 2..6 {
            for off in 0..2 * (rails - 1) {
                assert_eq!(rail_fence_decrypt(&rail_fence_encrypt(&p, rails, off), rails, off), p);
            }
        }
    }

    #[test]
    fn solves_rail_fence() {
        use crate::testutil::{model, sample};
        let p: String = crate::text::unscrub(&sample(70_000, 120));
        let chars: Vec<char> = p.chars().collect();
        let c: String = rail_fence_encrypt(&chars, 4, 2).into_iter().collect();
        let best = &solve_rail_fence(model(), &c, 8, 1)[0];
        assert_eq!(best.text, p);
    }

    #[test]
    fn solves_double_columnar() {
        use crate::testutil::{model, sample};
        let lm = model();
        let q = lm.dense(4);
        let p: String = crate::text::unscrub(&sample(80_000, 240));
        let chars: Vec<char> = p.chars().collect();
        let enc = |t: &[char], order: &[usize]| -> Vec<char> {
            let k = order.len();
            let mut out = Vec::new();
            for &col in order {
                let mut i = col;
                while i < t.len() {
                    out.push(t[i]);
                    i += k;
                }
            }
            out
        };
        let c: String = enc(&enc(&chars, &[2, 0, 3, 1]), &[1, 3, 0, 2]).into_iter().collect();
        let best = &solve_double_columnar(lm, &q, &c, 4, 4, 20_000, 4, 1)[0];
        assert_eq!(best.text, p);
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
