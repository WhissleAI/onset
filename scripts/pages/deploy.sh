#!/usr/bin/env bash
# Publish the site to the gh-pages branch (GitHub Pages "legacy" source).
# Used because GitHub Actions can't run on this account right now, so .github/workflows/pages.yml can't deploy.
# Usage: scripts/pages/deploy.sh   (builds _site from the current checkout, force-pushes it as gh-pages)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 scripts/pages/build.py
if grep -rlE "sk-ant|tvly-[A-Za-z0-9]{6}|tlm_[A-Za-z0-9]{8}|/Users/" _site >/dev/null; then
  echo "refusing: _site contains a key or a home path"; exit 1
fi
rev=$(git rev-parse --short HEAD); remote=$(git remote get-url origin)
tmp=$(mktemp -d); cp -R _site/. "$tmp"; touch "$tmp/.nojekyll"
git -C "$tmp" init -q -b gh-pages
git -C "$tmp" add -A
git -C "$tmp" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" \
  commit -q -m "Onset site built from main $rev"
git -C "$tmp" push -q -f "$remote" gh-pages
rm -rf "$tmp"
echo "pushed gh-pages ($rev); live in ~1 min at https://whissleai.github.io/headstart/"
