#!/usr/bin/env bash
# Fetch extra training text (modern prose, news, speeches) from the NLTK data repository.
# Only needs git, unzip and python3. Writes:
#   corpus/speeches-*.txt   (public domain; committed)
#   corpus-extra/*.txt      (research-licensed; git-ignored, never committed)
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone --depth 1 --branch gh-pages --filter=blob:none --sparse https://github.com/nltk/nltk_data.git "$tmp/nltk_data"
git -C "$tmp/nltk_data" sparse-checkout set --no-cone \
  /packages/corpora/brown.zip /packages/corpora/inaugural.zip /packages/corpora/state_union.zip \
  /packages/corpora/webtext.zip /packages/corpora/movie_reviews.zip /packages/corpora/reuters.zip
mkdir -p "$tmp/raw"
for z in "$tmp"/nltk_data/packages/corpora/*.zip; do unzip -qo "$z" -d "$tmp/raw"; done
python3 scripts/prepare_corpora.py "$tmp/raw" corpus corpus-extra
echo "Done. Train with: cryptok train --corpus corpus,corpus-extra"
