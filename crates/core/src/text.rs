//! Text normalisation helpers. Internally every letter is a `u8` in `0..26` (A = 0).

pub const ALPHABET: usize = 26;

/// Keep only ASCII letters, mapped to `0..26`.
pub fn scrub(s: &str) -> Vec<u8> {
    scrub_bytes(s.as_bytes())
}

pub fn scrub_bytes(b: &[u8]) -> Vec<u8> {
    b.iter()
        .filter_map(|&c| match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a'),
            _ => None,
        })
        .collect()
}

/// Convert letters in `0..26` back to an uppercase string.
pub fn unscrub(v: &[u8]) -> String {
    v.iter().map(|&c| (b'A' + c) as char).collect()
}

/// Map each letter of `s` to its index among the letters of `s`.
/// Returns, for every byte position in `s`, `Some(letter_index)` if it is a letter.
pub fn letter_positions(s: &str) -> Vec<Option<usize>> {
    let mut n = 0;
    s.bytes()
        .map(|c| {
            if c.is_ascii_alphabetic() {
                n += 1;
                Some(n - 1)
            } else {
                None
            }
        })
        .collect()
}

/// Remove Project Gutenberg header and licence footer if the markers are present.
pub fn strip_gutenberg(text: &str) -> &str {
    let start = text
        .find("*** START OF")
        .or_else(|| text.find("***START OF"))
        .and_then(|i| text[i..].find('\n').map(|j| i + j + 1))
        .unwrap_or(0);
    let end = text[start..]
        .find("*** END OF")
        .or_else(|| text[start..].find("***END OF"))
        .or_else(|| text[start..].find("End of the Project Gutenberg"))
        .or_else(|| text[start..].find("End of Project Gutenberg"))
        .map(|j| start + j)
        .unwrap_or(text.len());
    &text[start..end]
}

#[inline]
pub fn enc(p: u8, k: u8) -> u8 {
    (p + k) % 26
}

#[inline]
pub fn dec(c: u8, k: u8) -> u8 {
    (c + 26 - k) % 26
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_roundtrip() {
        assert_eq!(unscrub(&scrub("Hello, World!")), "HELLOWORLD");
    }

    #[test]
    fn gutenberg_strip() {
        let t = "header\n*** START OF THIS BOOK ***\nbody text\n*** END OF THIS BOOK ***\nlicence";
        assert_eq!(strip_gutenberg(t).trim(), "body text");
    }

    #[test]
    fn enc_dec() {
        for p in 0..26 {
            for k in 0..26 {
                assert_eq!(dec(enc(p, k), k), p);
            }
        }
    }
}

/// All regular files in the given directories (sorted by path), skipping the file names in `exclude`.
pub fn corpus_files(dirs: &[std::path::PathBuf], exclude: &[String]) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut names = vec![];
    for dir in dirs {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_file() && !exclude.contains(&n) {
                names.push(p);
            }
        }
    }
    names.sort();
    Ok(names)
}
