//! Monoalphabetic ciphers: Caesar / Atbash / Affine (exhaustive) and general simple
//! substitution (hill climbing over the 26! keys).

use crate::lm::{DenseNgram, LangModel};
use crate::rng::Rng;
use crate::text::unscrub;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub description: String,
    pub plain: Vec<u8>,
    /// Mean full-model log-prob per letter.
    pub per_letter: f32,
}

const AFFINE_A: [u8; 12] = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25];

fn modinv(a: u8) -> u8 {
    (1..26).find(|&x| (a as u16 * x as u16) % 26 == 1).unwrap()
}

/// Decrypt `c = a*p + b (mod 26)`. Caesar is `a = 1`; Atbash is `a = b = 25`.
pub fn affine_decrypt(cipher: &[u8], a: u8, b: u8) -> Vec<u8> {
    let inv = modinv(a) as u16;
    cipher.iter().map(|&c| (((c as u16 + 26 - b as u16) * inv) % 26) as u8).collect()
}

pub fn affine_encrypt(plain: &[u8], a: u8, b: u8) -> Vec<u8> {
    plain.iter().map(|&p| ((a as u16 * p as u16 + b as u16) % 26) as u8).collect()
}

/// Try every Caesar shift, Atbash and every affine key; best `top` by language score.
pub fn solve_affine_family(lm: &LangModel, cipher: &[u8], top: usize) -> Vec<Candidate> {
    let mut out = Vec::new();
    for &a in &AFFINE_A {
        for b in 0..26u8 {
            let plain = affine_decrypt(cipher, a, b);
            let description = match (a, b) {
                (1, 0) => "identity (no encryption)".to_string(),
                (1, _) => format!("Caesar shift {b} (key letter {})", (b'A' + b) as char),
                (25, 25) => "Atbash".to_string(),
                _ => format!("Affine a={a} b={b}"),
            };
            out.push(Candidate { per_letter: lm.score_per_letter(&plain), description, plain });
        }
    }
    out.sort_by(|x, y| y.per_letter.total_cmp(&x.per_letter));
    out.truncate(top);
    out
}

/// English letters, most to least frequent.
const FREQ_ORDER: &[u8; 26] = b"ETAOINSHRDLCUMWFGYPBVKJXQZ";

fn decrypt_with(cipher: &[u8], map: &[u8; 26], out: &mut Vec<u8>) {
    out.clear();
    out.extend(cipher.iter().map(|&c| map[c as usize]));
}

fn climb(q: &DenseNgram, cipher: &[u8], map: &mut [u8; 26], buf: &mut Vec<u8>) -> f32 {
    decrypt_with(cipher, map, buf);
    let mut cur = q.score(buf);
    loop {
        let mut improved = false;
        for i in 0..26 {
            for j in i + 1..26 {
                map.swap(i, j);
                decrypt_with(cipher, map, buf);
                let s = q.score(buf);
                if s > cur + 1e-4 {
                    cur = s;
                    improved = true;
                } else {
                    map.swap(i, j);
                }
            }
        }
        if !improved {
            return cur;
        }
    }
}

/// Solve a simple substitution cipher by shotgun hill climbing: start from a frequency
/// match, climb with letter swaps, then repeatedly shake the best key and climb again.
/// `map[cipher_letter]` = plaintext letter.
pub fn solve_substitution(lm: &LangModel, q: &DenseNgram, cipher: &[u8], restarts: usize, seed: u64) -> Candidate {
    let mut rng = Rng::new(seed);
    let mut count = [0usize; 26];
    for &c in cipher {
        count[c as usize] += 1;
    }
    let mut by_freq: Vec<usize> = (0..26).collect();
    by_freq.sort_by_key(|&l| std::cmp::Reverse(count[l]));
    let mut map = [0u8; 26];
    for (rank, &l) in by_freq.iter().enumerate() {
        map[l] = FREQ_ORDER[rank] - b'A';
    }
    let mut buf = Vec::with_capacity(cipher.len());
    let mut best_map = map;
    let mut best = climb(q, cipher, &mut best_map, &mut buf);
    for r in 0..restarts {
        let mut m = best_map;
        if r % 4 == 3 {
            rng.shuffle(&mut m); // occasional full restart
        } else {
            for _ in 0..2 + rng.below(4) {
                m.swap(rng.below(26), rng.below(26));
            }
        }
        let s = climb(q, cipher, &mut m, &mut buf);
        if s > best {
            best = s;
            best_map = m;
        }
    }
    decrypt_with(cipher, &best_map, &mut buf);
    // Describe the key as the plaintext letter for each cipher letter A..Z.
    let key: String = best_map.iter().map(|&p| (b'A' + p) as char).collect();
    Candidate {
        description: format!("simple substitution; cipher A-Z -> plain {key}"),
        per_letter: lm.score_per_letter(&buf),
        plain: buf,
    }
}

pub fn substitution_encrypt(plain: &[u8], key_alphabet: &str) -> Vec<u8> {
    let k: Vec<u8> = key_alphabet.bytes().map(|c| c.to_ascii_uppercase() - b'A').collect();
    plain.iter().map(|&p| k[p as usize]).collect()
}

pub fn describe(c: &Candidate) -> String {
    format!("{} ({:.3}/letter)\n{}", c.description, c.per_letter, unscrub(&c.plain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{model, sample};

    #[test]
    fn affine_roundtrip() {
        let p = sample(0, 50);
        for &a in &AFFINE_A {
            for b in 0..26 {
                assert_eq!(affine_decrypt(&affine_encrypt(&p, a, b), a, b), p);
            }
        }
    }

    #[test]
    fn finds_caesar_and_atbash() {
        let lm = model();
        let p = sample(1000, 80);
        let c = affine_encrypt(&p, 1, 7);
        assert_eq!(solve_affine_family(lm, &c, 1)[0].plain, p);
        let c = affine_encrypt(&p, 25, 25);
        let best = &solve_affine_family(lm, &c, 1)[0];
        assert_eq!(best.plain, p);
        assert_eq!(best.description, "Atbash");
        let c = affine_encrypt(&p, 5, 8);
        assert_eq!(solve_affine_family(lm, &c, 1)[0].plain, p);
    }

    #[test]
    fn solves_simple_substitution() {
        let lm = model();
        let q = lm.dense(4);
        let p = sample(5000, 400);
        let c = substitution_encrypt(&p, "QWERTYUIOPASDFGHJKLZXCVBNM");
        let sol = solve_substitution(lm, &q, &c, 30, 1);
        let right = sol.plain.iter().zip(&p).filter(|(a, b)| a == b).count();
        assert!(right as f64 / p.len() as f64 > 0.95, "only {right}/{}", p.len());
    }
}
