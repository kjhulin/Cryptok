#!/usr/bin/env python3
"""Generate the `synthetic` rows of bench/contests.tsv.

Contest-style puzzles built from original prose (not in corpus/, so the language model has
not seen it). Usage: python3 bench/gen_synthetic.py >> bench/contests.tsv
"""
import random, re

P1 = ("THEBADGEARRIVEDINAPLAINBROWNENVELOPEWITHNOINSTRUCTIONSJUSTASMALLCARDTHATSAIDTRUSTNOTHINGTHATBLINKS"
      "ANDTHATWASALLTHETEAMNEEDEDTOSTAYUPTHREENIGHTSARGUINGOVERWHETHERTHEPATTERNONTHELEDSWASAMESSAGEORAJOKE")
P2 = ("ITWASRAININGINTHEHARBOURTOWNTHENIGHTTHEOLDLIGHTHOUSEKEEPERLEFTAFOLDEDLETTERUNDERTHELAMPANDWALKEDOUTINTOTHESTORM"
      "NOBODYSAWHIMAGAINBUTTHELETTERSTAYEDTHEREFORYEARSEACHVISITORREADITANDSHRUGGEDBECAUSETHEWORDSMADENOSENSEUNTILTHEGIRL"
      "WHOLOVEDPUZZLESNOTICEDTHATEVERYFIFTHLETTERSPELLEDSOMETHINGELSE")
A = lambda s: [ord(c) - 65 for c in s]
S = lambda v: "".join(chr(65 + x % 26) for x in v)

def caesar(p, k): return S([c + k for c in A(p)])
def subst(p, al): return "".join(al[ord(c) - 65] for c in p)
def vig(p, key): k = A(key); return S([c + k[i % len(k)] for i, c in enumerate(A(p))])
def beaufort(p, key): k = A(key); return S([k[i % len(k)] - c for i, c in enumerate(A(p))])
def autokey(p, primer):
    pa, ka, out = A(p), A(primer), []
    for i, c in enumerate(pa):
        k = ka[i] if i < len(ka) else pa[i - len(ka)]
        out.append(c + k)
    return S(out)
def rail(p, r, off=0):
    per = 2 * (r - 1)
    pat = [(lambda t: t if t < r else per - t)((i + off) % per) for i in range(len(p))]
    return "".join(p[i] for i in sorted(range(len(p)), key=lambda i: (pat[i], i)))
def columnar(p, order):
    k = len(order)
    return "".join(p[i] for col in order for i in range(col, len(p), k))
def square(key):
    seen = []
    for c in (key + "ABCDEFGHIKLMNOPQRSTUVWXYZ").replace("J", "I"):
        if c not in seen: seen.append(c)
    return seen
def playfair(p, sq):
    p = p.replace("J", "I"); pairs = []; i = 0
    while i < len(p):
        a = p[i]; b = p[i + 1] if i + 1 < len(p) and p[i + 1] != a else "X"
        pairs.append((a, b)); i += 2 if b != "X" or (i + 1 < len(p) and p[i + 1] == "X") else 1
    prepared, out = "", ""
    for a, b in pairs:
        prepared += a + b
        ia, ib = sq.index(a), sq.index(b); ra, ca, rb, cb = ia // 5, ia % 5, ib // 5, ib % 5
        if ra == rb: x, y = sq[ra * 5 + (ca + 1) % 5], sq[rb * 5 + (cb + 1) % 5]
        elif ca == cb: x, y = sq[((ra + 1) % 5) * 5 + ca], sq[((rb + 1) % 5) * 5 + cb]
        else: x, y = sq[ra * 5 + cb], sq[rb * 5 + ca]
        out += x + y
    return out, prepared
def bifid(p, sq, period):
    p = p.replace("J", "I"); out = ""
    for s in range(0, len(p), period):
        ch = p[s:s + period]
        rows = [sq.index(c) // 5 for c in ch]; cols = [sq.index(c) % 5 for c in ch]
        seq = rows + cols
        out += "".join(sq[seq[i] * 5 + seq[i + 1]] for i in range(0, len(seq), 2))
    return out, p
def hill(p, m):
    p = p if len(p) % 2 == 0 else p[:-1]; a = A(p); out = []
    for i in range(0, len(a), 2):
        out += [(m[0] * a[i] + m[1] * a[i + 1]) % 26, (m[2] * a[i] + m[3] * a[i + 1]) % 26]
    return S(out)

rng = random.Random(2026)
gold = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"); rng.shuffle(gold)
gold = "".join(gold)
rows = []
def row(id, typ, cipher, plain, note, hint="-"):
    rows.append("\t".join([id, "synthetic", typ, cipher, plain[:60], hint, note]))

row("syn-caesar", "affine", caesar(P1, 11), P1, "Caesar shift 11")
row("syn-goldbug", "subst", subst(P2, gold), P2, "random monoalphabetic key (Gold-Bug style)")
c, prep = playfair(P2, square("QWERTYUIOPASDFGHKLZXCVBNM"))
row("syn-qwerty-playfair", "playfair", c, prep, "Playfair on a QWERTY keyboard square (DC23-style)")
row("syn-beaufort", "beaufort", beaufort(P2, "BADGE"), P2, "Beaufort, key BADGE")
row("syn-vigenere", "vigenere", vig(P2, "BADGER"), P2, "Vigenere, key BADGER")
row("syn-autokey", "autokey", autokey(P2, "DEFCON"), P2, "plaintext autokey, primer DEFCON")
row("syn-rail", "rail", rail(P2, 5), P2, "rail fence, 5 rails")
row("syn-columnar-subst", "chain", columnar(subst(P2, gold), [3, 0, 5, 1, 4, 2]), P2, "6-column columnar over a substitution (chain)")
c, prep = bifid(P2, square("BADGEPUZLSCTFHKMNOQRVWXY"), 8)
row("syn-bifid", "bifid", c, prep, "Bifid, period 8")
row("syn-hill", "hill", hill(P2, [3, 3, 2, 5]), P2, "Hill 2x2, key [3 3; 2 5]")
print("\n".join(rows))
