#!/usr/bin/env python3
"""Convert the NLTK corpora fetched by fetch-corpora.sh into plain-text training files.

usage: prepare_corpora.py RAW_DIR PUBLIC_OUT EXTRA_OUT

PUBLIC_OUT (committed): US presidential speeches (public domain).
EXTRA_OUT  (git-ignored): Brown, Reuters, movie reviews and web text, which are free to
           use for research but are not redistributed with the repository.
"""
import os, re, sys, glob, collections

raw, public, extra = sys.argv[1:4]
os.makedirs(public, exist_ok=True)
os.makedirs(extra, exist_ok=True)

def read(p):
    with open(p, encoding="latin-1") as f:
        return f.read()

def write(d, name, parts):
    text = "\n\n".join(parts)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{d}/{name}: {len(text)/1e6:.2f} MB")

# --- public domain speeches
write(public, "speeches-inaugural.txt", [read(p) for p in sorted(glob.glob(f"{raw}/inaugural/*.txt"))])
write(public, "speeches-state-of-the-union.txt", [read(p) for p in sorted(glob.glob(f"{raw}/state_union/*.txt"))])

# --- Brown corpus: strip /TAG, group by genre prefix (ca = news, cb = editorial, ...)
by_genre = collections.defaultdict(list)
for p in sorted(glob.glob(f"{raw}/brown/c[a-r]*")):
    name = os.path.basename(p)
    if not re.fullmatch(r"c[a-r]\d\d", name):
        continue
    text = re.sub(r"/[^\s/]+", "", read(p))  # word/tag -> word
    by_genre[name[:2]].append(text)
for g, parts in by_genre.items():
    write(extra, f"brown-{g}.txt", parts)

# --- movie reviews (informal modern prose)
for kind in ("pos", "neg"):
    write(extra, f"movie-reviews-{kind}.txt", [read(p) for p in sorted(glob.glob(f"{raw}/movie_reviews/{kind}/*.txt"))])

# --- Reuters newswire, split in three so single files stay a manageable size
docs = [read(p) for p in sorted(glob.glob(f"{raw}/reuters/training/*")) + sorted(glob.glob(f"{raw}/reuters/test/*"))]
third = len(docs) // 3 + 1
for i in range(3):
    write(extra, f"reuters-{i+1}.txt", docs[i * third:(i + 1) * third])

# --- web text: forum posts, overheard conversation, personals, wine notes, film scripts
for name in ("firefox", "overheard", "singles", "wine", "pirates", "grail"):
    write(extra, f"webtext-{name}.txt", [read(f"{raw}/webtext/{name}.txt")])
