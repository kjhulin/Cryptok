//! Cryptok Code Cracker 2.0 core library.
//!
//! * [`text`]  — normalising text to the 26-letter alphabet and cleaning corpora.
//! * [`lm`]    — interpolated Kneser–Ney character n-gram language model.
//! * [`rkc`]   — running key cipher solver (Viterbi beam search with state merging).
//! * [`classic`], [`periodic`], [`subst`], [`polygraphic`], [`transpo`] — other cipher families.
//! * [`analyze`] — statistical triage; [`decode`] — keyless encodings (Morse, A1Z26, ...).

pub mod analyze;
pub mod auto;
pub mod chain;
pub mod classic;
pub mod decode;
pub mod known;
pub mod lm;
pub mod map;
pub mod periodic;
pub mod polygraphic;
pub mod rkc;
pub mod rng;
pub mod subst;
pub mod text;
pub mod words;
#[cfg(test)]
mod testutil;
pub mod transpo;

pub use lm::LangModel;
