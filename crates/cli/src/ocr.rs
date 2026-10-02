//! OCR through the `tesseract` command-line program (https://github.com/tesseract-ocr/tesseract).
//!
//! Tesseract is not bundled: install it with your package manager (`apt install tesseract-ocr`,
//! `brew install tesseract`, or the Windows installer). The browser UI can also run OCR on its
//! own with Tesseract.js when the program is missing.

use cryptok_core::ocr::{clean, image_extension, Cleaned};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

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

pub struct OcrResult {
    pub raw: String,
    pub cleaned: Cleaned,
}

/// Run Tesseract on an image file and clean the result into cipher text.
pub fn ocr_file(path: &std::path::Path, opt: &OcrOptions) -> Result<OcrResult, String> {
    let mut whitelist = String::from("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz?");
    if opt.allow_digits {
        whitelist.push_str("0123456789");
    }
    if !opt.lang.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '+') {
        return Err("invalid --lang".into());
    }
    let out = Command::new("tesseract")
        .arg(path)
        .arg("stdout")
        .args(["--psm", &opt.psm.to_string(), "-l", &opt.lang])
        .args(["-c", &format!("tessedit_char_whitelist={whitelist}")])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                "tesseract is not installed (apt install tesseract-ocr / brew install tesseract)".to_string()
            } else {
                format!("cannot run tesseract: {e}")
            }
        })?;
    if !out.status.success() {
        return Err(format!("tesseract failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    let raw = String::from_utf8_lossy(&out.stdout).to_string();
    let cleaned = clean(&raw, opt.allow_digits);
    Ok(OcrResult { raw, cleaned })
}

/// OCR image bytes (as uploaded by the browser). The bytes are written to a private
/// temporary file named by us, never by the sender.
pub fn ocr_bytes(bytes: &[u8], opt: &OcrOptions) -> Result<OcrResult, String> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let ext = image_extension(bytes).ok_or("unsupported image type (use PNG, JPEG, GIF, BMP, WebP or TIFF)")?;
    let path: PathBuf = std::env::temp_dir().join(format!("cryptok-ocr-{}-{}.{ext}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&path, bytes).map_err(|e| format!("cannot write temporary file: {e}"))?;
    let r = ocr_file(&path, opt);
    let _ = std::fs::remove_file(&path);
    r
}
