//! Shared helpers for unit tests: a small English model and sample plaintext.

use crate::lm::LangModel;
use crate::text::{scrub, strip_gutenberg};
use std::path::Path;
use std::sync::OnceLock;

fn book(name: &str) -> Vec<u8> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus").join(name);
    scrub(strip_gutenberg(&std::fs::read_to_string(p).expect("corpus file")))
}

/// Order-4 model trained on a few novels (not the one `sample` is taken from).
pub fn model() -> &'static LangModel {
    static M: OnceLock<LangModel> = OnceLock::new();
    M.get_or_init(|| {
        let texts: Vec<Vec<u8>> = ["1342.txt", "84.txt", "1661.txt"].iter().map(|n| book(n)).collect();
        LangModel::train(&texts, 4).0
    })
}

/// `len` letters of Moby-Dick, starting `skip` letters in (held out of `model`).
pub fn sample(skip: usize, len: usize) -> Vec<u8> {
    static B: OnceLock<Vec<u8>> = OnceLock::new();
    let b = B.get_or_init(|| book("2701.txt"));
    b[skip..skip + len].to_vec()
}
