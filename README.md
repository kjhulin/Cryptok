# Cryptok Code Cracker 2.0

A rewrite of [CryptokCodeCracker](https://github.com/kjhulin/CryptokCodeCracker) in Rust, focused on fast and accurate running key cipher (RKC) solving, with a CLI and a local web UI.

**Status:** early development. Working today: language model, running key solver (Viterbi), known-text key search, crib search, keyed-alphabet Vigenère (solves Kryptos K1/K2), CLI, and a browser UI.

## Build

```
cargo build --release          # binary: target/release/cryptok
cargo test --release
```

No external crates are required.

## Quick start

```
cryptok train          # once: learns corpus/ -> cryptok.cklm (~5 s, ~95 MB)
cryptok serve          # opens the web UI at http://127.0.0.1:8077/
```

The web UI has three tabs:

- **Running key** — solve, pin letters on the worksheet (type in either stream; the other follows), place cribs, drag across a result to pin that stretch, solve again.
- **Known texts** — slide every source text along the cipher as a candidate key. Add your own sources (lyrics, speeches) with `--sources corpus,path/to/texts`.
- **Vigenère** — repeating-key Vigenère, optionally over keyword-mixed alphabets.

### Command line

```
cryptok rkc BVFBHGHXAWJEKEDMDZAPRMWGNMTVIRPWIKHGIPUU
cryptok rkc --plain-hint '____THE___' CIPHER   # fix known plaintext letters ('_' = unknown)
cryptok crib --word REDSHIRT CIPHER           # best positions for a crib
cryptok known --sources corpus,bench/private CIPHER   # known-text key search
cryptok vigenere --alphabet ",KRYPTOS" "$(cat bench/kryptos/k2.txt)"
cryptok score "some text"                      # language-model score
```

## Training text

`corpus/` (committed) holds Project Gutenberg books and US presidential speeches (public domain).
Gutenberg is mostly 19th-century prose, so `scripts/fetch-corpora.sh` can add modern text from the
NLTK data repository (Brown corpus, Reuters newswire, movie reviews, web text) into `corpus-extra/`.
That text is research-licensed, so it is git-ignored and never committed; `cryptok train` picks it up
automatically when present (`--corpus corpus,corpus-extra`). Training also writes `cryptok.words`,
the word list used by the word model, next to the model.

On held-out text, adding the extra corpora lifts running-key accuracy on modern text from 58% to 74% and
on the DEF CON 20 cipher from 68% to 74%, at a small cost on 19th-century books (83% to 82%).
Release binaries ship a model trained with all of it.

## Releases and CI

GitHub Actions runs build + tests on Linux, macOS and Windows for every PR, plus an accuracy gate
(`cryptok bench run --min-acc 70`) that fails if running-key accuracy regresses. Pushing a `v*` tag
builds per-platform archives (binary, corpus, word list) and a pre-trained `cryptok.cklm` model and
attaches them to a GitHub release.

## Tests

`cargo test --release` runs unit tests plus regression tests on real puzzles (Kryptos K1/K2, DEF CON 23) when a trained model is present.

Key and plaintext are interchangeable in a running key cipher, so results are shown as two streams, A and B.

## Benchmark

```
cryptok train --exclude 1342.txt,2701.txt,84.txt --out bench/holdout.cklm
cryptok bench gen --holdout 1342.txt,2701.txt,84.txt
cryptok bench run --model bench/holdout.cklm --beam 10000
```

Test ciphers are generated from books held out of training. Accuracy counts a position as correct if the (key, plaintext) pair matches in either order.

## How it works

- **Language model** (`crates/core/src/lm.rs`): order-6 character model with interpolated Kneser–Ney smoothing, trained on Project Gutenberg texts with licence boilerplate stripped. Every stored context has a full row of quantised log-probabilities, so scoring a letter is one hash lookup.
- **Word model** (`crates/core/src/words.rs`): word and word-pair (bigram) statistics from the corpus, saved as `cryptok.words` next to the model. During the beam search every hypothesis tracks the best word segmentation of its key and plaintext so far, scoring each word given the one before it (absolute-discounted bigram, backing off to the word frequency), and each new letter is scored by how much it improves that segmentation (weight `--word-weight`, default 0.4; 0 turns it off). The character model cannot see spaces; this adds the information that `REDSHIRTENGINEER` is words. It lifts mean accuracy on held-out text by several points (DEF CON 23 blind: 45% to 77%) at about 3.5x the time per cipher, and results are displayed with the discovered word breaks. Unigram-only is `--unigram`; word triples (`--trigram`) are supported but did not help.
- **RKC solver** (`crates/core/src/rkc.rs`): Viterbi beam search over (key, plaintext) pairs that merges hypotheses sharing the same last 6 key letters, prunes key/plaintext mirror duplicates, uses back-pointers, and expands candidates on all cores.

## Layout

```
crates/core   language model, solvers (library)
crates/cli    `cryptok` command
corpus/       training texts (Project Gutenberg, public domain)
bench/        benchmark cases; bench/private/ is git-ignored for copyrighted test keys
```
