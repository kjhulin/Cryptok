# Cryptok Code Cracker 2.0

A rewrite of [CryptokCodeCracker](https://github.com/kjhulin/CryptokCodeCracker) in Rust, focused on fast and accurate running key cipher (RKC) solving, with a CLI and (coming) a local web UI.

**Status:** early development. The language model, RKC solver, CLI and benchmark harness work; the web UI and classic-cipher tools are next.

## Build

```
cargo build --release          # binary: target/release/cryptok
cargo test --release
```

No external crates are required.

## Quick start

```
cryptok train                                  # learns corpus/ -> cryptok.cklm (~5 s, ~95 MB)
cryptok rkc BVFBHGHXAWJEKEDMDZAPRMWGNMTVIRPWIKHGIPUU
cryptok rkc --plain-hint '____THE___' CIPHER   # fix known plaintext letters ('_' = unknown)
cryptok score "some text"                      # language-model score
```

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
- **RKC solver** (`crates/core/src/rkc.rs`): Viterbi beam search over (key, plaintext) pairs that merges hypotheses sharing the same last 6 key letters, prunes key/plaintext mirror duplicates, uses back-pointers, and expands candidates on all cores.

## Layout

```
crates/core   language model, solvers (library)
crates/cli    `cryptok` command
corpus/       training texts (Project Gutenberg, public domain)
bench/        benchmark cases; bench/private/ is git-ignored for copyrighted test keys
```
