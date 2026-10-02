//! OCR through the `tesseract` command-line program (https://github.com/tesseract-ocr/tesseract).
//!
//! Tesseract is not bundled: install it with your package manager (`apt install tesseract-ocr`,
//! `brew install tesseract`, or the Windows installer). The browser UI can also run OCR on its
//! own with Tesseract.js when the program is missing.

use cryptok_core::ocr::{clean, image_extension, Cleaned};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub fn tesseract_available() -> bool {
    Command::new("tesseract").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

pub struct OcrOptions {
    /// Tesseract page segmentation mode: 6 = block of text, 7 = one line, 11 = sparse text.
    pub psm: u32,
    pub allow_digits: bool,
    pub lang: String,
}

impl Default for OcrOptions {
    fn default() -> Self {
        OcrOptions { psm: 6, allow_digits: false, lang: "eng".into() }
    }
}

#[derive(Debug)]
pub struct OcrResult {
    pub raw: String,
    pub cleaned: Cleaned,
}

/// An OCR failure. `public` is safe to show to a remote user; `detail` (paths, Tesseract's
/// stderr) is for the server log only.
#[derive(Debug)]
pub struct OcrError {
    pub public: String,
    pub detail: Option<String>,
}

impl OcrError {
    fn public(msg: &str) -> Self {
        OcrError { public: msg.into(), detail: None }
    }
    fn internal(msg: &str, detail: String) -> Self {
        OcrError { public: msg.into(), detail: Some(detail) }
    }
}

impl std::fmt::Display for OcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.public)?;
        if let Some(d) = &self.detail {
            write!(f, " ({d})")?;
        }
        Ok(())
    }
}

/// Longest Tesseract run before it is killed.
const OCR_TIMEOUT: Duration = Duration::from_secs(40);
/// Output we are willing to buffer from Tesseract.
const OCR_MAX_OUTPUT: u64 = 1 << 20;
const MAX_SIDE: u32 = 12_000;
const MAX_PIXELS: u64 = 16_000_000;

/// Where Tesseract reads the image from.
enum Source<'a> {
    /// A file the caller owns (the `cryptok ocr` command).
    File(&'a Path),
    /// Bytes piped to Tesseract's standard input. Nothing is written to disk.
    Memory(&'a [u8]),
}

fn run_tesseract(source: Source, opt: &OcrOptions) -> Result<OcrResult, OcrError> {
    let mut whitelist = String::from("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz?");
    if opt.allow_digits {
        whitelist.push_str("0123456789");
    }
    if !opt.lang.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '+') {
        return Err(OcrError::public("invalid language"));
    }
    let mut cmd = Command::new("tesseract");
    match &source {
        Source::File(p) => cmd.arg(p),
        Source::Memory(_) => cmd.arg("stdin"),
    };
    cmd.arg("stdout")
        .args(["--psm", &opt.psm.to_string(), "-l", &opt.lang])
        .args(["-c", &format!("tessedit_char_whitelist={whitelist}")])
        .env("OMP_THREAD_LIMIT", "1")
        .stdin(if matches!(source, Source::Memory(_)) { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            OcrError::public("tesseract is not installed (apt install tesseract-ocr / brew install tesseract)")
        } else {
            OcrError::internal("could not start the OCR engine", e.to_string())
        }
    })?;
    let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let stdin = child.stdin.take();
    // Feed the image and drain both pipes on helper threads (capped), so a chatty, hung or
    // slow-reading child can never block us. Scoped threads borrow the image: no copy is made.
    let (status, stdout, stderr) = std::thread::scope(|sc| {
        if let (Some(mut pipe), Source::Memory(bytes)) = (stdin, &source) {
            sc.spawn(move || {
                let _ = pipe.write_all(bytes); // a broken pipe just means Tesseract stopped early
            });
        }
        let t_out = sc.spawn(move || {
            let mut b = Vec::new();
            let _ = (&mut out).take(OCR_MAX_OUTPUT).read_to_end(&mut b);
            b
        });
        let t_err = sc.spawn(move || {
            let mut b = Vec::new();
            let _ = (&mut err).take(64 * 1024).read_to_end(&mut b);
            b
        });
        let deadline = Instant::now() + OCR_TIMEOUT;
        let status = loop {
            match child.try_wait() {
                Ok(Some(s)) => break Ok(s),
                Ok(None) if Instant::now() > deadline => {
                    let _ = child.kill(); // closes its pipes, which ends the helper threads
                    let _ = child.wait();
                    break Err(OcrError::public("the image took too long to read; try a smaller or simpler picture"));
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(25)),
                Err(e) => break Err(OcrError::internal("OCR failed", e.to_string())),
            }
        };
        (status, t_out.join().unwrap_or_default(), t_err.join().unwrap_or_default())
    });
    let status = status?;
    if !status.success() {
        return Err(OcrError::internal("the image could not be read", String::from_utf8_lossy(&stderr).trim().to_string()));
    }
    let raw = String::from_utf8_lossy(&stdout).to_string();
    let cleaned = clean(&raw, opt.allow_digits);
    Ok(OcrResult { raw, cleaned })
}

/// Run Tesseract on an image file and clean the result into cipher text.
pub fn ocr_file(path: &Path, opt: &OcrOptions) -> Result<OcrResult, OcrError> {
    run_tesseract(Source::File(path), opt)
}

/// Pixel dimensions from an image header, without decoding it.
pub fn image_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    let be32 = |o: usize| b.get(o..o + 4).map(|x| u32::from_be_bytes([x[0], x[1], x[2], x[3]]));
    let le16 = |o: usize| b.get(o..o + 2).map(|x| u16::from_le_bytes([x[0], x[1]]) as u32);
    let le32 = |o: usize| b.get(o..o + 4).map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]).unsigned_abs());
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some((be32(16)?, be32(20)?))
    } else if b.starts_with(b"GIF8") {
        Some((le16(6)?, le16(8)?))
    } else if b.starts_with(b"BM") {
        if le32(14)? == 12 { Some((le16(18)?, le16(20)?)) } else { Some((le32(18)?, le32(22)?)) }
    } else if b.starts_with(&[0xFF, 0xD8]) {
        let mut i = 2;
        while i + 9 < b.len() {
            if b[i] != 0xFF {
                i += 1;
                continue;
            }
            let m = b[i + 1];
            if m == 0xFF || m == 0x00 || m == 0x01 || (0xD0..=0xD8).contains(&m) {
                i += if m == 0xFF { 1 } else { 2 };
                continue;
            }
            if (0xC0..=0xCF).contains(&m) && m != 0xC4 && m != 0xC8 && m != 0xCC {
                let h = u16::from_be_bytes([b[i + 5], b[i + 6]]) as u32;
                let w = u16::from_be_bytes([b[i + 7], b[i + 8]]) as u32;
                return Some((w, h));
            }
            i += 2 + u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        }
        None
    } else if b.len() > 30 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        match &b[12..16] {
            b"VP8 " => Some((le16(26)? & 0x3FFF, le16(28)? & 0x3FFF)),
            b"VP8L" => {
                let x = b.get(21..25)?;
                Some((((x[0] as u32) | ((x[1] as u32 & 0x3F) << 8)) + 1, ((x[1] as u32 >> 6) | ((x[2] as u32) << 2) | ((x[3] as u32 & 0xF) << 10)) + 1))
            }
            b"VP8X" => {
                let x = b.get(24..30)?;
                Some((((x[0] as u32) | ((x[1] as u32) << 8) | ((x[2] as u32) << 16)) + 1, ((x[3] as u32) | ((x[4] as u32) << 8) | ((x[5] as u32) << 16)) + 1))
            }
            _ => None,
        }
    } else {
        None
    }
}

/// OCR image bytes uploaded by a browser. **The image is never written to disk**: it goes to
/// Tesseract's standard input and exists only in memory. It must be a PNG, JPEG, GIF, BMP or
/// WebP whose header declares a sane size, so a small file cannot expand into gigabytes of pixels.
pub fn ocr_bytes(bytes: &[u8], opt: &OcrOptions) -> Result<OcrResult, OcrError> {
    match image_extension(bytes) {
        Some("png" | "jpg" | "gif" | "bmp" | "webp") => {}
        _ => return Err(OcrError::public("unsupported image type (use PNG, JPEG, GIF, BMP or WebP)")),
    }
    let (w, h) = image_dimensions(bytes).ok_or(OcrError::public("the image header is damaged or unsupported"))?;
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || (w as u64) * (h as u64) > MAX_PIXELS {
        return Err(OcrError::public("the image is too large (limit 12000 pixels a side, 16 megapixels)"));
    }
    run_tesseract(Source::Memory(bytes), opt)
}

/// Overwrite a buffer that held an upload before it is freed (best effort: keeps the picture
/// out of memory that is reused, swapped, or dumped later).
pub fn wipe(buf: &mut [u8]) {
    buf.fill(0);
    std::hint::black_box(&buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_dimensions() {
        // PNG signature + IHDR for 640x480.
        let mut png = vec![0x89, b'P', b'N', b'G', 13, 10, 26, 10, 0, 0, 0, 13, b'I', b'H', b'D', b'R'];
        png.extend(640u32.to_be_bytes());
        png.extend(480u32.to_be_bytes());
        assert_eq!(image_dimensions(&png), Some((640, 480)));
        // JPEG with SOF0: 300 high, 200 wide.
        let jpg = [0xFF, 0xD8, 0xFF, 0xC0, 0, 17, 8, 1, 44, 0, 200, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1];
        assert_eq!(image_dimensions(&jpg), Some((200, 300)));
        assert_eq!(image_dimensions(b"GIF89a\x10\x00\x20\x00"), Some((16, 32)));
        assert_eq!(image_dimensions(b"not an image at all"), None);
    }

    #[test]
    fn rejects_decompression_bombs_and_junk() {
        let opt = OcrOptions::default();
        let mut png = vec![0x89, b'P', b'N', b'G', 13, 10, 26, 10, 0, 0, 0, 13, b'I', b'H', b'D', b'R'];
        png.extend(60_000u32.to_be_bytes());
        png.extend(60_000u32.to_be_bytes());
        assert!(ocr_bytes(&png, &opt).unwrap_err().public.contains("too large"));
        assert!(ocr_bytes(b"%PDF-1.7", &opt).unwrap_err().public.contains("unsupported"));
        assert!(ocr_bytes(&[0x89, b'P', b'N', b'G'], &opt).unwrap_err().public.contains("header"));
    }
}
