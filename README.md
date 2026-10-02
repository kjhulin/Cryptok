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

### Other ciphers (command line)

Not sure what you have? `cryptok analyze TEXT` prints the index of coincidence, likely periods and which of these to try:

| Command | Ciphers |
|---|---|
| `subst` | Caesar, Atbash, Affine (exhaustive), general monoalphabetic substitution (hill climbing) |
| `periodic --mode …` | Vigenère, Beaufort, Variant Beaufort, Porta, Gronsfeld, Quagmire I–IV (`--plain-alphabet`/`--cipher-alphabet`) |
| `autokey` | Vigenère autokey, plaintext and ciphertext key |
| `transpose [--double]` | routes, keyed columnar, double columnar |
| `rail` | rail fence (all rail counts and offsets); scytale is a one-step route in `transpose` |
| `playfair`, `bifid` | 5×5 key square recovered by simulated annealing (Bifid over a list of periods) |
| `hill` | 2×2 Hill cipher, all 157,248 keys |
| `chain --steps a,b,…` | several layers at once, outermost first (e.g. `rail,subst`, `vigenere,columnar`); see below |
| `decode --kind …` | Morse, A1Z26, Baconian, Polybius, binary, hex (no key or model needed) |

```
cryptok subst "$(cat mono.txt)"
cryptok periodic --mode beaufort --max-period 12 CIPHER
cryptok playfair CIPHER
```

**Chaining.** `cryptok chain --steps rail,subst CIPHER` undoes layers in the order given (outermost first). An outer layer is solved while the inner ones still scramble the text, so each stage is ranked by something the inner layers leave intact: unigram likelihood for a substitution/periodic layer over transpositions, bigram repetition for a transposition over monoalphabetic substitutions, and the full language model for the last step. Combinations with no such statistic (two transpositions in a row, a Playfair under anything) are rejected with an explanation; `autokey`, `hill`, `playfair` and `bifid` work only as the last step.

### Command line

```
cryptok rkc BVFBHGHXAWJEKEDMDZAPRMWGNMTVIRPWIKHGIPUU
cryptok rkc --plain-hint '____THE___' CIPHER   # fix known plaintext letters ('_' = unknown)
cryptok crib --word REDSHIRT CIPHER           # best positions for a crib
cryptok known --sources corpus,bench/private CIPHER   # known-text key search
cryptok vigenere --alphabet ",KRYPTOS" "$(cat bench/kryptos/k2.txt)"
cryptok score "some text"                      # language-model score
```

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
- **Word model** (`crates/core/src/words.rs`): word frequencies from the corpus. During the beam search every hypothesis tracks the best word segmentation of its key and plaintext so far, and new letters are scored by how much they improve it (weight `--word-weight`, default 0.3; 0 turns it off). The character model cannot see spaces; this adds the information that `REDSHIRTENGINEER` is words. On the 60-case benchmark it lifts mean accuracy from 80.1% to ~84% (DEF CON 23 blind: 45% to 77%), at about 3.5x the time per cipher. Results are displayed with the discovered word breaks.
- **RKC solver** (`crates/core/src/rkc.rs`): Viterbi beam search over (key, plaintext) pairs that merges hypotheses sharing the same last 6 key letters, prunes key/plaintext mirror duplicates, uses back-pointers, and expands candidates on all cores.

## Layout

```
crates/core   language model, solvers (library)
crates/cli    `cryptok` command
corpus/       training texts (Project Gutenberg, public domain)
bench/        benchmark cases; bench/private/ is git-ignored for copyrighted test keys
```
