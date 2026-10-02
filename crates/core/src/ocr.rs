//! Cleaning OCR output into cipher text.

/// Result of [`clean`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cleaned {
    pub text: String,
    /// Characters that were discarded (stray punctuation, accents, box-drawing noise).
    pub dropped: usize,
    /// Letters kept.
    pub letters: usize,
}

/// Turn raw OCR output into cipher text: letters are upper-cased, `?` is kept (Kryptos-style
/// ciphers contain it), digits are kept only with `allow_digits`, line breaks are kept, runs
/// of other whitespace become one space, and everything else is dropped and counted.
pub fn clean(raw: &str, allow_digits: bool) -> Cleaned {
    let mut text = String::with_capacity(raw.len());
    let (mut dropped, mut letters) = (0, 0);
    let mut pending_space = false;
    for ch in raw.chars() {
        match ch {
            '\n' => {
                while text.ends_with(' ') {
                    text.pop();
                }
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                pending_space = false;
            }
            c if c.is_whitespace() => pending_space = true,
            c if c.is_ascii_alphabetic() || c == '?' || (allow_digits && c.is_ascii_digit()) => {
                if pending_space && !text.is_empty() && !text.ends_with('\n') {
                    text.push(' ');
                }
                pending_space = false;
                if c.is_ascii_alphabetic() {
                    letters += 1;
                }
                text.push(c.to_ascii_uppercase());
            }
            _ => dropped += 1,
        }
    }
    Cleaned { text: text.trim_end().to_string(), dropped, letters }
}

/// File extension for the image formats Tesseract can read, from the file's first bytes.
pub fn image_extension(b: &[u8]) -> Option<&'static str> {
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("png")
    } else if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpg")
    } else if b.starts_with(b"GIF8") {
        Some("gif")
    } else if b.starts_with(b"BM") {
        Some("bmp")
    } else if b.len() > 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("webp")
    } else if b.starts_with(&[b'I', b'I', 42, 0]) || b.starts_with(&[b'M', b'M', 0, 42]) {
        Some("tif")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_ocr_noise() {
        let c = clean("  ob|kr  Qz—\n\n ?xy  \n12 ab", false);
        assert_eq!(c.text, "OBKR QZ\n?XY\nAB");
        assert_eq!(c.dropped, 4); // '|', '—', '1', '2'
        assert_eq!(c.letters, 10);
    }

    #[test]
    fn digits_are_optional() {
        assert_eq!(clean("12 ab 3", true).text, "12 AB 3");
        assert_eq!(clean("12 ab 3", false).text, "AB");
    }

    #[test]
    fn detects_image_types() {
        assert_eq!(image_extension(&[0x89, b'P', b'N', b'G', 13, 10]), Some("png"));
        assert_eq!(image_extension(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("jpg"));
        assert_eq!(image_extension(b"%PDF-1.7"), None);
    }
}
