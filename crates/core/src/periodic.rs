//! Periodic polyalphabetic ciphers beyond plain Vigenère: Beaufort, Variant Beaufort,
//! Porta, Gronsfeld and the Quagmire I–IV family, plus autokey.
//!
//! Every periodic cipher here decrypts column `i % period` with a per-column "shift" `s`,
//! so one hill climber serves them all.

use crate::classic::Alphabet;
use crate::lm::{DenseNgram, LangModel};
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub enum Mode {
    Vigenere,
    /// `p = k - c` (self-reciprocal).
    Beaufort,
    /// `p = c + k`.
    VariantBeaufort,
    /// 13 reciprocal alphabets; key letters pair up (A/B, C/D, ...).
    Porta,
    /// Vigenère with a numeric key (digits 0-9).
    Gronsfeld,
    /// Quagmire I-IV: keyed plaintext alphabet and/or keyed cipher alphabet.
    /// I: plain keyed, cipher straight. II: plain straight, cipher keyed.
    /// III: both the same keyed alphabet. IV: both keyed, differently.
    Quagmire { plain: Alphabet, cipher: Alphabet },
}

impl Mode {
    pub fn name(&self) -> String {
        match self {
            Mode::Vigenere => "Vigenère".into(),
            Mode::Beaufort => "Beaufort".into(),
            Mode::VariantBeaufort => "Variant Beaufort".into(),
            Mode::Porta => "Porta".into(),
            Mode::Gronsfeld => "Gronsfeld".into(),
            Mode::Quagmire { plain, cipher } => {
                let (p, c) = (&plain.keyword, &cipher.keyword);
                let kind = if p.is_empty() && c.is_empty() {
                    "Vigenère"
                } else if c.is_empty() {
                    "Quagmire I"
                } else if p.is_empty() {
                    "Quagmire II"
                } else if p == c {
                    "Quagmire III"
                } else {
                    "Quagmire IV"
                };
                format!("{kind} (plain alphabet {}, cipher alphabet {})", plain.as_string(), cipher.as_string())
            }
        }
    }

    /// Number of distinct shift values per column.
    pub fn shifts(&self) -> usize {
        match self {
            Mode::Porta => 13,
            Mode::Gronsfeld => 10,
            _ => 26,
        }
    }

    #[inline]
    pub fn decrypt_letter(&self, c: u8, s: u8) -> u8 {
        match self {
            Mode::Vigenere | Mode::Gronsfeld => (c + 26 - s) % 26,
            Mode::Beaufort => (s + 26 - c) % 26,
            Mode::VariantBeaufort => (c + s) % 26,
            Mode::Porta => {
                if c < 13 {
                    13 + (c + s) % 13
                } else {
                    (c - 13 + 13 - s) % 13
                }
            }
            Mode::Quagmire { plain, cipher } => plain.letters[((cipher.index[c as usize] + 26 - s) % 26) as usize],
        }
    }

    /// Encrypt one letter under the shift `s` (inverse of `decrypt_letter`).
    pub fn encrypt_letter(&self, p: u8, s: u8) -> u8 {
        match self {
            Mode::Vigenere | Mode::Gronsfeld => (p + s) % 26,
            Mode::Beaufort => (s + 26 - p) % 26,
            Mode::VariantBeaufort => (p + 26 - s) % 26,
            Mode::Porta => self.decrypt_letter(p, s),
            Mode::Quagmire { plain, cipher } => cipher.letters[((plain.index[p as usize] + s) % 26) as usize],
        }
    }

    /// The key letter (or digit) that a shift value stands for.
    pub fn key_char(&self, s: u8) -> char {
        match self {
            Mode::Gronsfeld => (b'0' + s) as char,
            Mode::Porta => (b'A' + 2 * s) as char,
            Mode::Quagmire { cipher, .. } => (b'A' + cipher.letters[s as usize]) as char,
            _ => (b'A' + s) as char,
        }
    }

    /// Shift value for a key letter / digit (Porta: either letter of the pair).
    pub fn shift_of(&self, key: char) -> Option<u8> {
        match self {
            Mode::Gronsfeld => key.to_digit(10).map(|d| d as u8),
            Mode::Porta => key.is_ascii_alphabetic().then(|| (key.to_ascii_uppercase() as u8 - b'A') / 2),
            Mode::Quagmire { cipher, .. } => {
                key.is_ascii_alphabetic().then(|| cipher.index[(key.to_ascii_uppercase() as u8 - b'A') as usize])
            }
            _ => key.is_ascii_alphabetic().then(|| key.to_ascii_uppercase() as u8 - b'A'),
        }
    }
}

pub fn decrypt(mode: &Mode, cipher: &[u8], shifts: &[u8]) -> Vec<u8> {
    cipher.iter().enumerate().map(|(i, &c)| mode.decrypt_letter(c, shifts[i % shifts.len()])).collect()
}

pub fn encrypt(mode: &Mode, plain: &[u8], shifts: &[u8]) -> Vec<u8> {
    plain.iter().enumerate().map(|(i, &p)| mode.encrypt_letter(p, shifts[i % shifts.len()])).collect()
}

#[derive(Clone, Debug)]
pub struct PeriodicSolution {
    pub mode: String,
    pub period: usize,
    pub key: String,
    pub plain: Vec<u8>,
    pub score: f32,
}

impl PeriodicSolution {
    pub fn per_letter(&self) -> f32 {
        self.score / self.plain.len().max(1) as f32
    }
}

fn set_col(mode: &Mode, cipher: &[u8], buf: &mut [u8], col: usize, period: usize, s: u8) {
    for i in (col..cipher.len()).step_by(period) {
        buf[i] = mode.decrypt_letter(cipher[i], s);
    }
}

/// Coordinate ascent over the column shifts for one period.
fn climb(mode: &Mode, q: &DenseNgram, cipher: &[u8], shifts: &mut [u8]) -> f32 {
    let period = shifts.len();
    let ns = mode.shifts() as u8;
    let mut buf = decrypt(mode, cipher, shifts);
    let mut cur = q.score(&buf);
    loop {
        let mut improved = false;
        for col in 0..period {
            let keep = shifts[col];
            let mut best = (keep, cur);
            for s in 0..ns {
                if s == keep {
                    continue;
                }
                set_col(mode, cipher, &mut buf, col, period, s);
                let sc = q.score(&buf);
                if sc > best.1 + 1e-4 {
                    best = (s, sc);
                }
            }
            set_col(mode, cipher, &mut buf, col, period, best.0);
            if best.0 != keep {
                shifts[col] = best.0;
                cur = best.1;
                improved = true;
            }
        }
        if !improved {
            return cur;
        }
    }
}

/// Per-column starting shifts that maximise the unigram score of each column alone.
fn unigram_init(mode: &Mode, uni: &DenseNgram, cipher: &[u8], period: usize) -> Vec<u8> {
    (0..period)
        .map(|col| {
            (0..mode.shifts() as u8)
                .max_by(|&a, &b| {
                    let f = |s: u8| -> f32 { (col..cipher.len()).step_by(period).map(|i| uni.value(mode.decrypt_letter(cipher[i], s) as usize)).sum() };
                    f(a).total_cmp(&f(b))
                })
                .unwrap()
        })
        .collect()
}

/// Search periods `1..=max_period`; returns the best solution per period, ranked with a
/// penalty for longer keys (more freedom to overfit).
pub fn solve_periodic(lm: &LangModel, q: &DenseNgram, cipher: &[u8], mode: &Mode, max_period: usize, restarts: usize) -> Vec<PeriodicSolution> {
    let uni = lm.dense(1);
    let mut rng = Rng::new(0xC0FFEE);
    let mut out = Vec::new();
    for period in 1..=max_period.min(cipher.len() / 2).max(1) {
        let mut best: Option<(Vec<u8>, f32)> = None;
        for r in 0..=restarts {
            let mut shifts = if r == 0 {
                unigram_init(mode, &uni, cipher, period)
            } else {
                (0..period).map(|_| rng.below(mode.shifts()) as u8).collect()
            };
            let sc = climb(mode, q, cipher, &mut shifts);
            if best.as_ref().is_none_or(|b| sc > b.1) {
                best = Some((shifts, sc));
            }
        }
        let (shifts, _) = best.unwrap();
        let plain = decrypt(mode, cipher, &shifts);
        out.push(PeriodicSolution {
            mode: mode.name(),
            period,
            key: shifts.iter().map(|&s| mode.key_char(s)).collect(),
            score: lm.score(&plain),
            plain,
        });
    }
    let pen = (mode.shifts() as f32).ln();
    out.sort_by(|a, b| (b.score - b.period as f32 * pen).total_cmp(&(a.score - a.period as f32 * pen)));
    out
}

// ---------------------------------------------------------------------------------------
// Autokey

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Autokey {
    /// Key = primer followed by the plaintext.
    Plaintext,
    /// Key = primer followed by the ciphertext.
    Ciphertext,
}

pub fn autokey_decrypt(kind: Autokey, cipher: &[u8], primer: &[u8]) -> Vec<u8> {
    let l = primer.len();
    let mut p: Vec<u8> = Vec::with_capacity(cipher.len());
    for (i, &c) in cipher.iter().enumerate() {
        let k = if i < l {
            primer[i]
        } else {
            match kind {
                Autokey::Plaintext => p[i - l],
                Autokey::Ciphertext => cipher[i - l],
            }
        };
        p.push((c + 26 - k) % 26);
    }
    p
}

pub fn autokey_encrypt(kind: Autokey, plain: &[u8], primer: &[u8]) -> Vec<u8> {
    let l = primer.len();
    let mut c: Vec<u8> = Vec::with_capacity(plain.len());
    for (i, &p) in plain.iter().enumerate() {
        let k = if i < l {
            primer[i]
        } else {
            match kind {
                Autokey::Plaintext => plain[i - l],
                Autokey::Ciphertext => c[i - l],
            }
        };
        c.push((p + k) % 26);
    }
    c
}

#[derive(Clone, Debug)]
pub struct AutokeySolution {
    pub kind: Autokey,
    pub primer: Vec<u8>,
    pub plain: Vec<u8>,
    pub score: f32,
}

/// Recover the primer of a Vigenère autokey cipher by coordinate ascent for each primer
/// length `1..=max_primer`. Ranked with a per-letter penalty for longer primers.
pub fn solve_autokey(lm: &LangModel, q: &DenseNgram, cipher: &[u8], max_primer: usize, restarts: usize) -> Vec<AutokeySolution> {
    let mut rng = Rng::new(0xA070);
    let mut out = Vec::new();
    for kind in [Autokey::Plaintext, Autokey::Ciphertext] {
        for l in 1..=max_primer.min(cipher.len() / 2) {
            let mut best: Option<(Vec<u8>, f32)> = None;
            for _ in 0..=restarts {
                let mut primer: Vec<u8> = (0..l).map(|_| rng.below(26) as u8).collect();
                let mut cur = q.score(&autokey_decrypt(kind, cipher, &primer));
                loop {
                    let mut improved = false;
                    for j in 0..l {
                        let keep = primer[j];
                        let mut b = (keep, cur);
                        for s in 0..26u8 {
                            if s == keep {
                                continue;
                            }
                            primer[j] = s;
                            let sc = q.score(&autokey_decrypt(kind, cipher, &primer));
                            if sc > b.1 + 1e-4 {
                                b = (s, sc);
                            }
                        }
                        primer[j] = b.0;
                        if b.0 != keep {
                            cur = b.1;
                            improved = true;
                        }
                    }
                    if !improved {
                        break;
                    }
                }
                if best.as_ref().is_none_or(|x| cur > x.1) {
                    best = Some((primer, cur));
                }
            }
            let primer = best.unwrap().0;
            let plain = autokey_decrypt(kind, cipher, &primer);
            out.push(AutokeySolution { kind, score: lm.score(&plain), primer, plain });
        }
    }
    let pen = 26f32.ln();
    out.sort_by(|a, b| (b.score - b.primer.len() as f32 * pen).total_cmp(&(a.score - a.primer.len() as f32 * pen)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{model, sample};
    use crate::text::{scrub, unscrub};

    fn all_modes() -> Vec<Mode> {
        vec![
            Mode::Vigenere,
            Mode::Beaufort,
            Mode::VariantBeaufort,
            Mode::Porta,
            Mode::Gronsfeld,
            Mode::Quagmire { plain: Alphabet::from_keyword("KRYPTOS"), cipher: Alphabet::standard() },
            Mode::Quagmire { plain: Alphabet::from_keyword("KRYPTOS"), cipher: Alphabet::from_keyword("PALIMPSEST") },
        ]
    }

    #[test]
    fn encrypt_decrypt_inverse() {
        let p = sample(0, 60);
        for m in all_modes() {
            let shifts: Vec<u8> = (0..5).map(|i| (i * 3 + 1) as u8 % m.shifts() as u8).collect();
            assert_eq!(decrypt(&m, &encrypt(&m, &p, &shifts), &shifts), p, "{}", m.name());
        }
    }

    #[test]
    fn known_beaufort_and_porta() {
        // Beaufort with key "KEY" on "ATTACK": classic worked example.
        let k: Vec<u8> = scrub("KEY");
        // Beaufort: c = k - p. A->K, T->(E-T)=L, T->(Y-T)=F, A->K, C->(E-C)=C, K->(Y-K)=O.
        assert_eq!(unscrub(&encrypt(&Mode::Beaufort, &scrub("ATTACK"), &k)), "KLFKCO");
        // Porta is reciprocal.
        let c = encrypt(&Mode::Porta, &scrub("DEFENDTHEEASTWALL"), &[3, 1]);
        assert_eq!(unscrub(&decrypt(&Mode::Porta, &c, &[3, 1])), "DEFENDTHEEASTWALL");
    }

    #[test]
    fn solves_each_mode() {
        let lm = model();
        let q = lm.dense(4);
        let p = sample(20_000, 400);
        for (m, key) in [
            (Mode::Beaufort, "CIPHER"),
            (Mode::VariantBeaufort, "SECRET"),
            (Mode::Porta, "LEMON"),
            (Mode::Gronsfeld, "31415"),
            (Mode::Quagmire { plain: Alphabet::from_keyword("KRYPTOS"), cipher: Alphabet::standard() }, "ABSCISSA"),
        ] {
            let shifts: Vec<u8> = key.chars().map(|c| m.shift_of(c).unwrap()).collect();
            let c = encrypt(&m, &p, &shifts);
            let sol = &solve_periodic(lm, &q, &c, &m, 10, 8)[0];
            let right = sol.plain.iter().zip(&p).filter(|(a, b)| a == b).count();
            assert!(right as f64 / p.len() as f64 > 0.95, "{}: key {} got {right}/{}", m.name(), sol.key, p.len());
        }
    }

    #[test]
    fn autokey_roundtrip_and_solve() {
        let p = sample(30_000, 300);
        let primer = scrub("QUEENY");
        for kind in [Autokey::Plaintext, Autokey::Ciphertext] {
            assert_eq!(autokey_decrypt(kind, &autokey_encrypt(kind, &p, &primer), &primer), p);
        }
        let lm = model();
        let q = lm.dense(4);
        let c = autokey_encrypt(Autokey::Plaintext, &p, &primer);
        let sol = &solve_autokey(lm, &q, &c, 8, 6)[0];
        assert_eq!(sol.kind, Autokey::Plaintext);
        let right = sol.plain.iter().zip(&p).filter(|(a, b)| a == b).count();
        assert!(right as f64 / p.len() as f64 > 0.95, "primer {} got {right}", unscrub(&sol.primer));
    }
}
