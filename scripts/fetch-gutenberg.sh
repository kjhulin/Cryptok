#!/usr/bin/env bash
# Fetch the extra Project Gutenberg books listed in data/gutenberg-extra.tsv (~1,800 English
# books, ~720 MB, public domain in the US) into corpus-gutenberg/ (git-ignored). Training uses
# corpus-gutenberg/ automatically when it exists.
# Books come from the GITenberg mirror on GitHub (raw.githubusercontent.com), which is often
# reachable where gutenberg.org is not. Needs curl and xargs. Safe to re-run: existing files
# are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
out=corpus-gutenberg
mkdir -p "$out"
fetch() {
  id=$1; name=$2; dest="$out/pg$id.txt"
  [ -s "$dest" ] && return 0
  for f in "$id.txt" "$id-0.txt" "$id-8.txt"; do
    if curl -sSfL --retry 3 -m 120 -o "$dest.tmp" "https://raw.githubusercontent.com/GITenberg/$name/master/$f" 2>/dev/null; then
      mv "$dest.tmp" "$dest"
      return 0
    fi
  done
  rm -f "$dest.tmp"
  echo "missing: $id $name" >&2
}
export -f fetch
export out
grep -v '^#' data/gutenberg-extra.tsv | tr -d '\r' | xargs -P "${JOBS:-16}" -L 1 bash -c 'fetch "$0" "$1"'
echo "$(ls "$out" | wc -l) books in $out/. Train with: cryptok train"
