//! Statistical triage: which family of cipher does this text look like?

use crate::classic::period_ic;
use crate::text::scrub;

#[derive(Clone, Debug)]
pub struct Analysis {
    pub letters: usize,
    pub distinct: usize,
    /// Index of coincidence (English ≈ 0.066, uniform random ≈ 0.038).
    pub ic: f64,
    /// Share of the text made of E T A O I N (English ≈ 0.52); survives transposition.
    pub etaoin: f64,
    /// Up to three best periods with their normalised column IC (English ≈ 1.7, random ≈ 1.0).
    pub periods: Vec<(usize, f64)>,
    /// Digraph-aligned pairs of identical letters (never occurs in Playfair).
    pub doubled_digraphs: usize,
    pub has_j: bool,
    /// Suggestions: `(cryptok command, why)`, most likely first.
    pub suggestions: Vec<(String, String)>,
}

pub fn index_of_coincidence(c: &[u8]) -> f64 {
    let mut f = [0u64; 26];
    for &x in c {
        f[x as usize] += 1;
    }
    let n = c.len() as u64;
    if n < 2 {
        return 0.0;
    }
    f.iter().map(|&x| x * x.saturating_sub(1)).sum::<u64>() as f64 / (n * (n - 1)) as f64
}

pub fn analyze(text: &str) -> Analysis {
    let c = scrub(text);
    let n = c.len();
    let mut f = [0usize; 26];
    for &x in &c {
        f[x as usize] += 1;
    }
    let distinct = f.iter().filter(|&&x| x > 0).count();
    let ic = index_of_coincidence(&c);
    let etaoin = if n == 0 { 0.0 } else { b"ETAOIN".iter().map(|&l| f[(l - b'A') as usize]).sum::<usize>() as f64 / n as f64 };
    let mut periods = period_ic(&c, 20.min(n / 10).max(1)); // keep >= 10 letters per column
    periods.retain(|p| p.0 > 1);
    periods.sort_by(|a, b| b.1.total_cmp(&a.1));
    periods.truncate(3);
    let doubled_digraphs = c.chunks_exact(2).filter(|p| p[0] == p[1]).count();
    let has_j = f[9] > 0;

    let mut s: Vec<(String, String)> = Vec::new();
    let mut add = |cmd: &str, why: String| s.push((cmd.to_string(), why));
    if n < 20 {
        add("", "too few letters for reliable statistics".into());
    } else if ic > if n < 400 { 0.054 } else { 0.060 } {
        if etaoin > 0.44 {
            add("transpose", format!("IC {ic:.3} and E/T/A/O/I/N make up {:.0}% of the text: letter frequencies look like English, so the letters are probably just rearranged", etaoin * 100.0));
            add("rail", "also try a rail fence if the text is short".into());
        } else {
            add("subst", format!("IC {ic:.3} is English-like but the common letters are not E/T/A/O/I/N: monoalphabetic substitution (Caesar/Atbash/Affine are tried first)"));
        }
    } else {
        let best = periods.first().copied();
        if let Some((p, v)) = best.filter(|&(_, v)| v > 1.45) {
            add("periodic", format!("columns at period {p} look like single-alphabet English (column IC {v:.2} x26): Vigenère/Beaufort/Porta/Quagmire family"));
            add("vigenere", "if you know a keyword for the alphabet, try `cryptok vigenere --alphabet KEYWORD`".into());
        }
        let playfair_like = n % 2 == 0 && !has_j && doubled_digraphs == 0 && distinct <= 25 && n >= 40;
        if playfair_like {
            add("playfair", format!("even length, no J, no doubled letter in any digraph over {} digraphs: classic Playfair signature", n / 2));
        }
        if !has_j && distinct <= 25 {
            add("bifid", "no J (25-letter alphabet) and flat statistics: Bifid or another square-based cipher".into());
        }
        add("autokey", "no clear period: plaintext/ciphertext autokey is possible".into());
        add("rkc", "or a running key cipher (key as long as the message)".into());
        add("hill", "Hill (2x2) also gives flat statistics; tries all keys quickly".into());
    }
    Analysis { letters: n, distinct, ic, etaoin, periods, doubled_digraphs, has_j, suggestions: s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::{encrypt, Mode};
    use crate::polygraphic::{playfair_encrypt, square_from_keyword};
    use crate::subst::{affine_encrypt, substitution_encrypt};
    use crate::testutil::sample;
    use crate::text::unscrub;

    fn first(a: &Analysis) -> &str {
        &a.suggestions[0].0
    }

    #[test]
    fn classifies_families() {
        // Short texts have noisy statistics; they must classify too.
        let short = sample(95_000, 175);
        assert_eq!(first(&analyze(&unscrub(&substitution_encrypt(&short, "QWERTYUIOPASDFGHJKLZXCVBNM")))), "subst");
        let mut rev = short.clone();
        rev.reverse();
        assert_eq!(first(&analyze(&unscrub(&rev))), "transpose");
        let p = sample(90_000, 600);
        let mono = substitution_encrypt(&p, "QWERTYUIOPASDFGHJKLZXCVBNM");
        assert_eq!(first(&analyze(&unscrub(&mono))), "subst");
        assert_eq!(first(&analyze(&unscrub(&affine_encrypt(&p, 1, 3)))), "subst");
        let mut shuffled = p.clone();
        shuffled.reverse();
        assert_eq!(first(&analyze(&unscrub(&shuffled))), "transpose");
        let vig = encrypt(&Mode::Vigenere, &p, &[2, 17, 24, 15, 19, 14, 18]);
        let a = analyze(&unscrub(&vig));
        assert_eq!(first(&a), "periodic");
        assert_eq!(a.periods[0].0 % 7, 0);
        let pf = playfair_encrypt(&square_from_keyword("MONARCHY"), &p);
        let a = analyze(&unscrub(&pf));
        assert!(a.suggestions.iter().any(|s| s.0 == "playfair"));
    }
}
