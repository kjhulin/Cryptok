#!/usr/bin/env python3
"""Generate DEF CON 23-style running-key cases: short ciphers (default 40 letters) where the
key is a passage starting at a sentence boundary in a held-out book and the plaintext is a
sentence-initial passage of modern prose (NLTK Brown corpus, not used for training).

Usage: python3 bench/gen_short.py BROWN_DIR [--n 100] [--len 40] [--seed 23] > bench/cases-short40.tsv
BROWN_DIR is the unpacked `brown/` folder of nltk_data's corpora/brown.zip.
Keep Brown out of any model you evaluate with these cases (corpus-extra/ includes it).
"""
import argparse, os, random, re

HOLDOUT = ["1342.txt", "2701.txt", "84.txt"]

def letters(s):
    return re.sub(r"[^A-Z]", "", s.upper())

def book_sentences(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); b = t.find("*** END")
    if a >= 0: t = t[t.find("\n", a) + 1:]
    if b >= 0: t = t[:b]
    t = re.sub(r"\s+", " ", t)
    # Sentence starts: after . ! ? (optionally a closing quote) and a space, before a capital.
    starts = [m.end() for m in re.finditer(r"[.!?][\"'”’]?\s+(?=[\"“]?[A-Z])", t)]
    return t, starts

def brown_sentences(d):
    out = []
    for f in sorted(os.listdir(d)):
        if not re.fullmatch(r"c[a-r]\d\d", f): continue
        for line in open(os.path.join(d, f), encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line: continue
            words = [w.rsplit("/", 1)[0] for w in line.split()]
            out.append(" ".join(words))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brown")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--len", type=int, default=40)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    books = [book_sentences(os.path.join(a.corpus, h)) for h in HOLDOUT]
    sents = brown_sentences(a.brown)
    # Plaintext: a sentence start, continuing into following sentences until long enough.
    print(f"# id\tlength\tcipher\tkey\tplain  (DC23-style: key from holdout {','.join(HOLDOUT)} at a sentence start; plaintext from Brown at a sentence start; seed {a.seed})")
    i = 0
    while i < a.n:
        j = rnd.randrange(len(sents) - 20)
        plain = letters(" ".join(sents[j:j + 20]))[: a.len]
        t, starts = books[i % len(books)]
        s = rnd.choice(starts[len(starts) // 20: -len(starts) // 20])
        key = letters(t[s: s + 400])[: a.len]
        if len(plain) < a.len or len(key) < a.len:
            continue
        c = "".join(chr(65 + (ord(p) + ord(k) - 130) % 26) for p, k in zip(plain, key))
        i += 1
        print(f"{i}\t{a.len}\t{c}\t{key}\t{plain}")

main()
