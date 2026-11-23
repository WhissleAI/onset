#!/usr/bin/env bash
# Build the native macOS app: dist/<APP_NAME>.app (identity in macapp/app.conf). See MACAPP.md.
#
#   scripts/build_macapp.sh               full build (reuses the bundled Python if requirements + version are unchanged)
#   scripts/build_macapp.sh --clean       rebuild the Python runtime from scratch too
#   scripts/build_macapp.sh --dev-env     the app also reads THIS repo's .env for keys missing from the Keychain
#                                         (developer builds only: bakes this repo's path into Info.plist)
#   scripts/build_macapp.sh --dmg         also make dist/<APP_NAME>.dmg (a second copy: ~1 GB more disk)
#   scripts/build_macapp.sh --no-swap     leave the new build in dist/.staging (test it there first)
#   scripts/build_macapp.sh --release     public release (MACAPP.md "Release"): a slim app whose models download on
#                                         first launch from the public archives in models/manifest.release.json
#                                         (scripts/publish_models.py), Developer ID + timestamp signing, DMG, notarized and
#                                         stapled. Add --bundle-models for an all-in-one app instead (models read-only in
#                                         Contents/Resources/models). Goes to dist/release/ (never touches dist/<APP_NAME>.app). Needs
#                                         ONSET_SIGN_ID="Developer ID Application: ..." and NOTARY_PROFILE=<notarytool
#                                         keychain profile>. Refuses --dev-env: a public build never reads a .env.
#   scripts/build_macapp.sh --bundle-models   bundle the models into the app (release profile) without the rest
#   env: MODELS_MANIFEST=models/manifest.onnx.json ONNX_RUNTIME=1   the native (ONNX) runtime: its manifest, and
#        ONNX_RUNTIME=1 passed to the server (Info.plist HSServerEnv); REQUIREMENTS=macapp/requirements.onnx.txt
#        a slimmer Python; MODELS_SRC=DIR where the models to bundle come from (default MODELS_DIR);
#        PY_FROM=path/to/Other.app reuse that app's bundled Python (APFS clone) when its requirements match
#   ONSET_SIGN_ID="<identity>" scripts/build_macapp.sh
#                                         sign with a real identity (`security find-identity -v -p codesigning`) instead
#                                         of ad-hoc: macOS then keeps the microphone + Keychain grants across rebuilds
#
# The build goes to dist/.staging/<APP_NAME>.app and replaces dist/<APP_NAME>.app only when no copy of the app (or its
# server) is running from dist/: a running app is never rebuilt over or quit. Otherwise it stays staged and the
# script says so; `scripts/build_macapp.sh --swap` swaps later. The bundled Python is cloned between builds (APFS
# copy-on-write: no extra disk). Intermediates go to build/macapp and are deleted at the end.
# Needs: Xcode command line tools (swiftc, cc, codesign, iconutil), uv (python-build-standalone + pip installs).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
# shellcheck disable=SC1091
source macapp/app.conf
CLEAN=0; DEV_ENV=0; DMG=0; PY_ONLY=0; SWAP=1; SWAP_ONLY=0; RELEASE=0; BUNDLE_MODELS=0; NOTARIZE=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;; --dev-env) DEV_ENV=1 ;; --dmg) DMG=1 ;; --python-only) PY_ONLY=1 ;; --no-swap) SWAP=0 ;; --swap) SWAP_ONLY=1 ;;
    --release) RELEASE=1; DMG=1; NOTARIZE=1; SWAP=0 ;; --bundle-models) BUNDLE_MODELS=1 ;;
    --no-notarize) NOTARIZE=0 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
REQUIREMENTS="${REQUIREMENTS:-macapp/requirements.txt}"
# Release builds run the embedder on ONNX Runtime (embeddinggemma-2-onnx, 8-bit: the release profile ships that copy, not
# the 1.5 GB torch one). ONNX_RUNTIME=0 scripts/build_macapp.sh --release forces the torch embedder.
if [ "$RELEASE" = 1 ] && [ -z "${ONNX_RUNTIME:-}" ]; then ONNX_RUNTIME=1; fi
# The app's manifest: the published release models (first-run download from public links) when present.
if [ -z "${MODELS_MANIFEST:-}" ]; then
  MODELS_MANIFEST=models/manifest.json; [ -f models/manifest.release.json ] && MODELS_MANIFEST=models/manifest.release.json
fi
if [ "$RELEASE" = 1 ]; then
  [ "$DEV_ENV" = 1 ] && { echo "--release never reads a .env: drop --dev-env"; exit 2; }
  case "${ONSET_SIGN_ID:-}" in "Developer ID Application:"*) ;; *) echo "--release needs ONSET_SIGN_ID=\"Developer ID Application: ...\" (security find-identity -v -p codesigning)"; exit 2 ;; esac
  if [ "$NOTARIZE" = 1 ] && [ -z "${NOTARY_PROFILE:-}" ]; then
    echo "--release needs NOTARY_PROFILE=<profile> (xcrun notarytool store-credentials <profile> --apple-id <id> --team-id <team>), or --no-notarize"; exit 2
  fi
fi
EXEC="$(echo "$APP_NAME" | tr -cd '[:alnum:]')"         # executable name: the app name without spaces
FINAL="dist/$APP_NAME.app"
APP="dist/.staging/$APP_NAME.app"
if [ "$RELEASE" = 1 ]; then APP="dist/release/$APP_NAME.app"; fi
C="$APP/Contents"
RES="$C/Resources"
PY="$RES/python"
WORK="build/macapp"
REQ_STAMP="$(cat "$REQUIREMENTS" <(echo "$PYTHON_VERSION") | shasum -a 256 | cut -c1-16)"
step() { printf '\n== %s\n' "$1"; }

running_from() { pgrep -f "$REPO/$1/Contents/" >/dev/null 2>&1; }
swap() {
  if running_from "$FINAL"; then
    echo "  $FINAL is running: NOT replaced. The new build is in $APP; quit $APP_NAME, then: scripts/build_macapp.sh --swap"
    return 1
  fi
  if running_from "$APP"; then echo "  a test instance is running from $APP: quit it first"; return 1; fi
  rm -rf "$FINAL"
  mv "$APP" "$FINAL"
  rmdir dist/.staging 2>/dev/null || true
  echo "  installed $FINAL"
}
if [ "$SWAP_ONLY" = 1 ]; then [ -d "$APP" ] || { echo "nothing staged in $APP"; exit 1; }; swap; exit $?; fi

command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/): it fetches python-build-standalone"; exit 1; }
mkdir -p "$(dirname "$APP")" "$WORK"
if running_from "$APP"; then echo "a test instance is running from $APP: quit it first"; exit 1; fi
# keep the bundled Python across rebuilds (it is most of the size and time); everything else is rebuilt.
# Source: the previous staged build, else the installed app (cloned: APFS copy-on-write, the installed app untouched).
if [ "$CLEAN" = 0 ]; then
  for src in "$APP" "$FINAL" ${PY_FROM:+"$PY_FROM"}; do
    sp="$src/Contents/Resources/python"
    if [ -d "$sp" ] && [ "$(cat "$sp/.req_stamp" 2>/dev/null)" = "$REQ_STAMP" ]; then
      if [ "$src" = "$APP" ]; then mv "$sp" "$WORK/python.keep"; else cp -cR "$sp" "$WORK/python.keep"; fi
      break
    fi
  done
fi
rm -rf "$APP"
mkdir -p "$C/MacOS" "$RES"

step "Python $PYTHON_VERSION (python-build-standalone) + server dependencies"
if [ -d "$WORK/python.keep" ]; then
  mv "$WORK/python.keep" "$PY"
  echo "  reused (requirements unchanged: $REQ_STAMP)"
else
  export UV_CACHE_DIR="$REPO/$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$REPO/$WORK/uv-python"
  uv python install --no-bin "$PYTHON_VERSION" >/dev/null
  src="$(ls -d "$UV_PYTHON_INSTALL_DIR"/cpython-"$PYTHON_VERSION"-macos-*/ | head -1)"
  mv "$src" "$PY"
  rm -f "$PY"/lib/python3.*/EXTERNALLY-MANAGED
  uv pip install --python "$PY/bin/python3" --link-mode copy --compile-bytecode -q -r "$REQUIREMENTS"
  # trim what a headless server never uses
  P="$(ls -d "$PY"/lib/python3.*)"
  rm -rf "$P/test" "$P/idlelib" "$P/tkinter" "$P/turtledemo" "$P/ensurepip" "$P/lib-dynload/_tkinter"* \
         "$PY/lib/tcl"* "$PY/lib/tk"* "$PY/lib/itcl"* "$PY/lib/thread"* "$PY/share" "$PY/include"
  find "$PY" -type d -name tests -path '*site-packages/*' -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$P/site-packages/torch/include" "$P/site-packages/torch/share" "$P/site-packages/pip" "$P/site-packages"/pip-*.dist-info
  find "$P/site-packages" -name '*.pyi' -delete
  echo "$REQ_STAMP" > "$PY/.req_stamp"
  rm -rf "$WORK/uv-cache" "$WORK/uv-python"
fi
if [ "${ONNX_RUNTIME:-0}" = 1 ]; then
  "$PY/bin/python3" -c "import sys, onnxruntime, fastapi; print('  python', sys.version.split()[0], 'onnxruntime', onnxruntime.__version__)"
else
  "$PY/bin/python3" -c "import sys, torch, transformers, sentence_transformers, fastapi; print('  python', sys.version.split()[0], 'torch', torch.__version__)"
fi

if [ "$PY_ONLY" = 1 ]; then rm -rf "$WORK"; du -sh "$PY"; exit 0; fi

step "Server code, web UI, small data"
A="$RES/app"
mkdir -p "$A"
rsync -a --exclude __pycache__ --exclude '*.pyc' server web "$A/"
mkdir -p "$A/scripts" "$A/models/arctag" "$A/vendor/arctag" "$A/eval/multipart" "$A/index" "$A/runs" "$A/macapp"
cp scripts/fetch_models.py "$A/scripts/"
cp "$MODELS_MANIFEST" "$A/models/manifest.json"
cp models/arctag/LICENSE models/arctag/NOTICE "$A/models/arctag/" 2>/dev/null || true
cp vendor/arctag/LICENSE vendor/arctag/NOTICE "$A/vendor/arctag/"
cp eval/questions.jsonl eval/conversations.jsonl "$A/eval/"
cp eval/multipart/questions.jsonl "$A/eval/multipart/"
cp index/questions_local.jsonl "$A/index/"
cp runs/web_cache.json "$A/runs/" 2>/dev/null || true
cp macapp/app.conf macapp/server_main.py "$A/macapp/"
python3 scripts/fetch_models.py build-tagger --out "$A/vendor/arctag/libarctag.dylib" >/dev/null
"$PY/bin/python3" -m compileall -q "$A/server" "$A/macapp" "$A/scripts" >/dev/null
if [ ! -f "$PY/.pruned" ]; then        # once per Python build: drop bytecode of modules the server never imports
  "$PY/bin/python3" -I macapp/prune_bytecode.py "$A" "$(ls -d "$PY"/lib/python3.*/site-packages)"
  touch "$PY/.pruned"
fi

if [ "$BUNDLE_MODELS" = 1 ]; then
  step "Models, bundled read-only (models/release_profile.json)"
  python3 scripts/fetch_models.py --manifest "$MODELS_MANIFEST" bundle --out "$RES/models" ${MODELS_SRC:+--src "$MODELS_SRC"}
fi

step "Licences (Resources/Licenses)"
L="$RES/Licenses"; mkdir -p "$L"
cp LICENSE "$L/Onset-LICENSE.txt" 2>/dev/null || true
cp macapp/licenses/THIRD_PARTY.md "$L/"
cp macapp/licenses/Apache-2.0.txt "$L/"
cp vendor/arctag/LICENSE "$L/ArcTagger-LICENSE.txt"; cp vendor/arctag/NOTICE "$L/ArcTagger-NOTICE.txt"
MD="$RES/models"; [ -d "$MD" ] || MD="$(python3 -c 'from server import modelpaths; print(modelpaths.models_dir())')"
cp models/arctag/LICENSE "$L/ArcTaggerModel-LICENSE.txt" 2>/dev/null || true
cp models/arctag/NOTICE "$L/ArcTaggerModel-NOTICE.txt" 2>/dev/null || true
cp vendor/arctag/third_party/cjson/LICENSE "$L/cJSON-LICENSE.txt"
cp macapp/licenses/EmbeddingGemma-NOTICE.txt "$L/"
for f in arc-v1/LICENSE:ArcV1-LICENSE.txt arc-v1/NOTICE:ArcV1-NOTICE.txt \
         arc-v1/moonshine-streaming-tiny/LICENSE:Moonshine-LICENSE.txt \
         arc-v1/moonshine-streaming-tiny/ORIGIN.md:Moonshine-ORIGIN.md \
         embeddinggemma-2-onnx/LICENSE:EmbeddingGemma2-ONNX-LICENSE.txt \
         embeddinggemma-2-onnx/NOTICE:EmbeddingGemma2-ONNX-NOTICE.txt \
         answer-lm/LICENSE:Qwen2.5-0.5B-Instruct-LICENSE.txt; do
  src="$MD/${f%%:*}"; if [ -f "$src" ]; then cp "$src" "$L/${f##*:}"; fi
done
SP="$(ls -d "$PY"/lib/python3.*/site-packages)"
cp "$(ls -d "$PY"/lib/python3.*)/LICENSE.txt" "$L/Python-LICENSE.txt" 2>/dev/null || true
cp "$SP/_soundfile_data/COPYING" "$L/libsndfile-LGPL-2.1.txt" 2>/dev/null || true    # LGPL: a separate, replaceable dylib
for f in "$SP"/onnxruntime/ThirdPartyNotices.txt "$SP"/onnxruntime/LICENSE; do
  if [ -f "$f" ]; then cp "$f" "$L/onnxruntime-$(basename "$f")"; fi
done
echo "  $(ls "$L" | wc -l | tr -d ' ') files; Python packages keep theirs in Resources/python/lib/*/site-packages/*.dist-info"

step "Native shell (Swift)"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
xcrun swiftc -O -sdk "$SDK" -target "$(uname -m)-apple-macos13.0" -module-name AppShell \
  -o "$C/MacOS/$EXEC" macapp/Sources/*.swift
cat > "$C/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>$EXEC</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$APP_VERSION</string>
  <key>CFBundleVersion</key><string>$APP_VERSION</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>$APP_NAME listens while you hold Space (or after you click the mic) so it can search while you speak. Audio stays on this Mac unless you add a Deepgram key.</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
  <key>HSDevEnvPath</key><string>$( [ "$DEV_ENV" = 1 ] && echo "$REPO/.env" )</string>
  <key>HSServerEnv</key><dict>$( [ "${ONNX_RUNTIME:-0}" = 1 ] && echo "<key>ONNX_RUNTIME</key><string>1</string>" )</dict>
</dict></plist>
PLIST

step "Icon"
if [ -f macapp/AppIcon.icns ]; then
  cp macapp/AppIcon.icns "$RES/AppIcon.icns"
else
  IS="$WORK/AppIcon.iconset"; mkdir -p "$IS"
  xcrun swiftc -O -sdk "$SDK" -o "$WORK/make_icon" macapp/make_icon.swift
  "$WORK/make_icon" "$IS" "$ICON_TEXT" "$ICON_COLORS"
  iconutil -c icns "$IS" -o "$RES/AppIcon.icns"
fi

SIGN="${ONSET_SIGN_ID:--}"   # "-" = ad-hoc: a new identity every build, so macOS asks for the mic again (MACAPP.md)
TS="--timestamp=none"; case "$SIGN" in "Developer ID Application:"*) TS="--timestamp" ;; esac   # notarization needs a timestamp
PY_ENT="${PY_ENTITLEMENTS:-$( [ "$RELEASE" = 1 ] && echo macapp/python.release.entitlements || echo macapp/python.entitlements )}"
step "Codesign ($([ "$SIGN" = - ] && echo "ad-hoc" || echo "$SIGN"), hardened runtime, $TS)"
# every Mach-O in the bundle, found by its header (not its name), inside-out: libraries + helper executables, then the
# Python executable with its own entitlements, then the app. Models (Resources/models) hold no code.
MACHO="$WORK/macho.txt"
python3 - "$PY" "$A" > "$MACHO" <<'PYEOF'
import os, sys
MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            if os.path.islink(p) or f.endswith(".class"):
                continue
            try:
                with open(p, "rb") as h:
                    if h.read(4) in MAGIC:
                        print(p)
            except OSError:
                pass
PYEOF
grep -v "^$PY/bin/" "$MACHO" | tr '\n' '\0' | xargs -0 -n 100 -P 4 codesign -f -s "$SIGN" $TS --options runtime
grep "^$PY/bin/" "$MACHO" | while IFS= read -r f; do
  codesign -f -s "$SIGN" $TS --options runtime --entitlements "$PY_ENT" "$f"
done
echo "  $(wc -l < "$MACHO" | tr -d ' ') Mach-O files signed"
codesign -f -s "$SIGN" $TS --options runtime --entitlements macapp/app.entitlements "$APP"
codesign --verify --strict --deep "$APP" && echo "  signature OK"

rm -rf "$WORK"
rmdir build 2>/dev/null || true
step "Done"
du -sh "$APP" | sed 's/^/  /'
echo "  python runtime: $(du -sh "$PY" | cut -f1)   app code: $(du -sh "$A" | cut -f1)"
if [ "$RELEASE" = 1 ]; then APP_OUT="$APP"
elif [ "$SWAP" = 1 ]; then swap || true; APP_OUT="$( [ -d "$FINAL" ] && [ ! -d "$APP" ] && echo "$FINAL" || echo "$APP")"
else APP_OUT="$APP"; echo "  staged (--no-swap): $APP"; fi
echo "  open \"$APP_OUT\"   (models: python3 scripts/fetch_models.py status)"
APP="$APP_OUT"
OUT="$(dirname "$APP")"
DMG_PATH="$OUT/$APP_NAME.dmg"
[ "$RELEASE" = 1 ] && DMG_PATH="$OUT/$APP_NAME-$APP_VERSION-arm64.dmg"

notarize() {   # notarize + staple one file (dmg); prints the submission id and status
  local f="$1" log="$OUT/notarize-$(basename "$1").json"
  xcrun notarytool submit "$f" --keychain-profile "$NOTARY_PROFILE" --wait --output-format json > "$log"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('  notarization', d.get('id'), d.get('status'))" "$log"
  grep -q '"status":"Accepted"' "$log" || grep -q '"status" : "Accepted"' "$log" || {
    echo "  NOT accepted: xcrun notarytool log <id> --keychain-profile $NOTARY_PROFILE"; return 1; }
}

make_dmg() {   # the app + an Applications link, lzma-compressed, signed
  local src="$WORK/dmg"; rm -rf "$src"; mkdir -p "$src"
  cp -cR "$APP" "$src/" ; ln -s /Applications "$src/Applications"
  rm -f "$DMG_PATH"
  hdiutil create -volname "$APP_NAME" -srcfolder "$src" -ov -format ULMO "$DMG_PATH" >/dev/null
  rm -rf "$src"
  [ "$SIGN" != - ] && codesign -f -s "$SIGN" $TS "$DMG_PATH"
  du -sh "$DMG_PATH" | sed 's/^/  /'
}

if [ "$DMG" = 1 ]; then
  mkdir -p "$WORK"
  step "DMG"
  make_dmg
  if [ "$NOTARIZE" = 1 ]; then
    step "Notarize (notarytool profile $NOTARY_PROFILE)"
    notarize "$DMG_PATH"
    # the dmg's ticket covers the app inside: staple the app too, then rebuild the dmg around the stapled app (so it
    # opens offline after being copied out) and notarize + staple that final dmg
    xcrun stapler staple "$APP" >/dev/null && echo "  app stapled"
    make_dmg
    notarize "$DMG_PATH"
    xcrun stapler staple "$DMG_PATH" >/dev/null && echo "  dmg stapled"
    xcrun stapler validate "$DMG_PATH"
  fi
  (cd "$OUT" && shasum -a 256 "$(basename "$DMG_PATH")" > "$(basename "$DMG_PATH").sha256" && cat "$(basename "$DMG_PATH").sha256" | sed 's/^/  sha256 /')
  rm -rf "$WORK"; rmdir build 2>/dev/null || true
fi
