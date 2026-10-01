//! Classic ciphers. Currently: periodic Vigenère with an optional keyed alphabet
//! (Quagmire III — the system used for Kryptos K1 and K2).

use crate::lm::LangModel;

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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Hill-climb the column shifts for a fixed period. Returns (shifts, score).
fn climb(lm: &LangModel, cipher: &[u8], a: &Alphabet, mut shifts: Vec<u8>) -> (Vec<u8>, f32) {
    let mut best = lm.score(&vigenere_decrypt(cipher, &shifts, a));
    loop {
        let mut improved = false;
        for col in 0..shifts.len() {
            let orig = shifts[col];
            let mut col_best = (orig, best);
            for s in 0..26u8 {
                if s == orig {
                    continue;
                }
                shifts[col] = s;
                let sc = lm.score(&vigenere_decrypt(cipher, &shifts, a));
                if sc > col_best.1 {
                    col_best = (s, sc);
                }
            }
            shifts[col] = col_best.0;
            if col_best.1 > best + 1e-4 {
                best = col_best.1;
                improved = true;
            }
        }
        if !improved {
            return (shifts, best);
        }
    }
}

/// Solve a periodic Vigenère cipher over the given alphabet, trying every period
/// up to `max_period`. Periods are compared with a penalty of ln(26) per key letter
/// so that multiples of the true period do not win by over-fitting.
pub fn solve_vigenere(lm: &LangModel, cipher: &[u8], a: &Alphabet, max_period: usize, restarts: usize) -> Vec<VigenereSolution> {
    let n = cipher.len();
    if n == 0 {
        return vec![];
    }
    let uni = lm.row(0, 0);
    let deq = lm.deq();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::new();
    for period in 1..=max_period.min(n) {
        // Initialise each column with its best shift under the unigram model.
        let init: Vec<u8> = (0..period)
            .map(|col| {
                (0..26u8)
                    .max_by(|&x, &y| {
                        let f = |s: u8| -> f32 {
                            cipher
                                .iter()
                                .skip(col)
                                .step_by(period)
                                .map(|&c| deq[uni[a.letters[((a.index[c as usize] + 26 - s) % 26) as usize] as usize] as usize])
                                .sum()
                        };
                        f(x).total_cmp(&f(y))
                    })
                    .unwrap()
            })
            .collect();
        let mut best = climb(lm, cipher, a, init);
        for _ in 0..restarts {
            let start: Vec<u8> = (0..period).map(|_| (rng.next() % 26) as u8).collect();
            let r = climb(lm, cipher, a, start);
            if r.1 > best.1 {
                best = r;
            }
        }
        let plain = vigenere_decrypt(cipher, &best.0, a);
        let key = best.0.iter().map(|&s| a.letters[s as usize]).collect();
        out.push(VigenereSolution { period, key, alphabet: a.as_string(), plain, score: best.1 });
    }
    let pen = (26f32).ln();
    // Drop solutions whose key just repeats a shorter key (e.g. ABSCISSAABSCISSA).
    out.retain(|s| !(1..s.period).any(|d| s.period % d == 0 && (d..s.period).all(|i| s.key[i] == s.key[i - d])));
    out.sort_by(|x, y| (y.score - y.period as f32 * pen).total_cmp(&(x.score - x.period as f32 * pen)));
    out
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
