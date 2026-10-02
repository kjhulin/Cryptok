//! Keyless encodings that often appear in puzzles: Morse, A1Z26, Baconian, Polybius,
//! binary and hexadecimal ASCII.

fn morse_char(code: &str) -> Option<char> {
    const TABLE: &[(&str, char)] = &[
        (".-", 'A'), ("-...", 'B'), ("-.-.", 'C'), ("-..", 'D'), (".", 'E'), ("..-.", 'F'), ("--.", 'G'),
        ("....", 'H'), ("..", 'I'), (".---", 'J'), ("-.-", 'K'), (".-..", 'L'), ("--", 'M'), ("-.", 'N'),
        ("---", 'O'), (".--.", 'P'), ("--.-", 'Q'), (".-.", 'R'), ("...", 'S'), ("-", 'T'), ("..-", 'U'),
        ("...-", 'V'), (".--", 'W'), ("-..-", 'X'), ("-.--", 'Y'), ("--..", 'Z'),
        ("-----", '0'), (".----", '1'), ("..---", '2'), ("...--", '3'), ("....-", '4'), (".....", '5'),
        ("-....", '6'), ("--...", '7'), ("---..", '8'), ("----.", '9'),
    ];
    TABLE.iter().find(|(m, _)| *m == code).map(|&(_, c)| c)
}

/// Morse: letters separated by spaces, words by `/` or two or more spaces or `|`.
/// `·`/`•` count as dots and `–`/`—`/`_` as dashes.
pub fn morse(s: &str) -> String {
    let s: String = s
        .chars()
        .map(|c| match c {
            '·' | '•' | '*' => '.',
            '–' | '—' | '_' => '-',
            '|' => '/',
            c => c,
        })
        .collect();
    let s = s.replace("   ", " / ").replace("  ", " / ");
    let out: String = s.split_whitespace().map(|t| if t == "/" { ' ' } else { morse_char(t).unwrap_or('?') }).collect();
    out.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
}

/// A1Z26: numbers 1–26 separated by anything non-numeric.
pub fn a1z26(s: &str) -> String {
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .map(|t| match t.parse::<u32>() {
            Ok(n @ 1..=26) => (b'A' + (n - 1) as u8) as char,
            _ => '?',
        })
        .collect()
}

/// Baconian: groups of five A/B symbols (also 0/1, or lower/upper case after `case = true`).
/// `distinct` selects the 26-letter variant (I≠J, U≠V); the classic variant has 24 letters.
pub fn baconian(s: &str, distinct: bool, case: bool) -> String {
    let bits: Vec<u8> = s
        .chars()
        .filter_map(|c| match (case, c) {
            (false, 'a' | 'A' | '0') => Some(0),
            (false, 'b' | 'B' | '1') => Some(1),
            (true, c) if c.is_ascii_lowercase() => Some(0),
            (true, c) if c.is_ascii_uppercase() => Some(1),
            _ => None,
        })
        .collect();
    bits.chunks_exact(5)
        .map(|g| {
            let v = g.iter().fold(0usize, |a, &b| a * 2 + b as usize);
            if distinct {
                if v < 26 { (b'A' + v as u8) as char } else { '?' }
            } else {
                // 24-letter alphabet: I/J and U/V share a code.
                const L: &[u8; 24] = b"ABCDEFGHIKLMNOPQRSTUWXYZ";
                match v {
                    0..=23 => L[v] as char,
                    _ => '?',
                }
            }
        })
        .collect()
}

/// Polybius square given as digit pairs (11–55); `square` is 25 letters, row-major
/// (default alphabetical with J merged into I).
pub fn polybius(s: &str, square: &str) -> String {
    let sq: Vec<char> = if square.is_empty() { "ABCDEFGHIKLMNOPQRSTUVWXYZ".chars().collect() } else { square.to_ascii_uppercase().chars().collect() };
    let digits: Vec<usize> = s.chars().filter_map(|c| c.to_digit(10)).map(|d| d as usize).collect();
    digits
        .chunks_exact(2)
        .map(|p| if (1..=5).contains(&p[0]) && (1..=5).contains(&p[1]) { sq.get((p[0] - 1) * 5 + p[1] - 1).copied().unwrap_or('?') } else { '?' })
        .collect()
}

/// 8-bit binary groups to ASCII.
pub fn binary(s: &str) -> String {
    let bits: Vec<u8> = s.chars().filter_map(|c| match c { '0' => Some(0), '1' => Some(1), _ => None }).collect();
    bits.chunks_exact(8).map(|g| g.iter().fold(0u8, |a, &b| a * 2 + b) as char).collect()
}

/// Hex bytes to ASCII.
pub fn hex(s: &str) -> String {
    let h: Vec<u8> = s.chars().filter_map(|c| c.to_digit(16)).map(|d| d as u8).collect();
    h.chunks_exact(2).map(|g| (g[0] * 16 + g[1]) as char).collect()
}

pub const KINDS: &str = "morse, a1z26, baconian, baconian26, polybius, binary, hex";

pub fn decode(kind: &str, s: &str) -> Option<String> {
    Some(match kind {
        "morse" => morse(s),
        "a1z26" => a1z26(s),
        "baconian" => baconian(s, false, false),
        "baconian26" => baconian(s, true, false),
        "polybius" => polybius(s, ""),
        "binary" => binary(s),
        "hex" => hex(s),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes() {
        assert_eq!(morse("... --- ...  / .-- . -.. -.."), "SOS WEDD");
        assert_eq!(morse("... --- ... / -.-."), "SOS C");
        assert_eq!(a1z26("8-5-12-12-15 23 15-18-12-4"), "HELLOWORLD");
        assert_eq!(baconian("AABBB AABAA ABABA ABABA ABBAB", false, false), "HELLO");
        assert_eq!(baconian("aabbb aabaa ababa ababa abbab", false, false), "HELLO");
        assert_eq!(polybius("23 15 31 31 34", ""), "HELLO");
        assert_eq!(binary("01001000 01101001"), "Hi");
        assert_eq!(hex("48 65 6c 6c 6f"), "Hello");
    }
}
