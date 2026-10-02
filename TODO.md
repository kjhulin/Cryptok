# Todo

- [ ] (in progress) See how many of the ciphers from past DEF CON badge challenges we can break automatically (collect the ciphertexts with known answers, run `analyze` and the solvers/`chain` on each, record which ones fall and why the rest don't; `bench/real.tsv` already holds DC20 and DC23).
  - Done: `cryptok contest run` and `bench/contests.tsv` (DC20, DC23, Kryptos K1-K3, 10 synthetic). Baseline: 13 of 15 solved; the two running-key cases are partial (DC20 58%, DC23 78%) and limited by the language model, not the beam.
  - Open: add the other badge challenges (DC21, DC22, DC24 onward) once their ciphertexts and published solutions can be fetched; the sandbox's network proxy currently blocks elegin.com, forum.defcon.org and defcon.org. Candidates seen in search results: DC23 lanyards (Nyctograph, Gold-Bug, QWERTY Playfair), DC24 (one-time pad, Caesar). Unverified, so none are in the file yet.
  - Open: improve running-key accuracy on short texts (DC20, DC23).

## Ciphers not yet implemented

Homophonic substitution, ADFGX/ADFGVX, Trifid, four-square/two-square, 3×3 Hill, Myszkowski and disrupted columnar, book ciphers, Enigma, repeating-key XOR, and web UI tabs for the newer solvers (they are command-line only).
