//! Cryptok Code Cracker 2.0 core library.
//!
//! * [`text`]  — normalising text to the 26-letter alphabet and cleaning corpora.
//! * [`lm`]    — interpolated Kneser–Ney character n-gram language model.
//! * [`rkc`]   — running key cipher solver (Viterbi beam search with state merging).

pub mod lm;
pub mod map;
pub mod rkc;
pub mod text;

pub use lm::LangModel;
