#!/usr/bin/env bash
# Build the native macOS app: dist/<APP_NAME>.app (identity in macapp/app.conf). See MACAPP.md.
#
#   scripts/build_macapp.sh               full build (reuses the bundled Python if requirements + version are unchanged)
#   scripts/build_macapp.sh --clean       rebuild the Python runtime from scratch too
#   scripts/build_macapp.sh --dev-env     the app also reads THIS repo's .env for keys missing from the Keychain
#                                         (developer builds only: bakes this repo's path into Info.plist)
#   scripts/build_macapp.sh --dmg         also make dist/<APP_NAME>.dmg (a second copy: ~1 GB more disk)
#
# Output: dist/<APP_NAME>.app only. Intermediates go to build/macapp and are deleted at the end.
# Needs: Xcode command line tools (swiftc, cc, codesign, iconutil), uv (python-build-standalone + pip installs).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
# shellcheck disable=SC1091
source macapp/app.conf
CLEAN=0; DEV_ENV=0; DMG=0; PY_ONLY=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;; --dev-env) DEV_ENV=1 ;; --dmg) DMG=1 ;; --python-only) PY_ONLY=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
EXEC="$(echo "$APP_NAME" | tr -cd '[:alnum:]')"         # executable name: the app name without spaces
APP="dist/$APP_NAME.app"
C="$APP/Contents"
RES="$C/Resources"
PY="$RES/python"
WORK="build/macapp"
REQ_STAMP="$(cat macapp/requirements.txt <(echo "$PYTHON_VERSION") | shasum -a 256 | cut -c1-16)"
step() { printf '\n== %s\n' "$1"; }

command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/): it fetches python-build-standalone"; exit 1; }
mkdir -p dist "$WORK"
if [ "$CLEAN" = 1 ]; then rm -rf "$APP"; fi
# keep the bundled Python across rebuilds (it is most of the size and time); everything else is rebuilt
if [ -d "$APP" ]; then
  if [ -d "$PY" ] && [ "$(cat "$PY/.req_stamp" 2>/dev/null)" = "$REQ_STAMP" ]; then
    mv "$PY" "$WORK/python.keep"
  fi
  rm -rf "$APP"
fi
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
  uv pip install --python "$PY/bin/python3" --link-mode copy --compile-bytecode -q -r macapp/requirements.txt
  # trim what a headless server never uses
  P="$(ls -d "$PY"/lib/python3.*)"
  rm -rf "$P/test" "$P/idlelib" "$P/tkinter" "$P/turtledemo" "$P/ensurepip" "$P/lib-dynload/_tkinter"* \
         "$PY/lib/tcl"* "$PY/lib/tk"* "$PY/lib/itcl"* "$PY/lib/thread"* "$PY/share" "$PY/include"
  find "$PY" -type d -name tests -path '*site-packages/*' -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$P/site-packages/torch/include" "$P/site-packages/torch/share"
  echo "$REQ_STAMP" > "$PY/.req_stamp"
  rm -rf "$WORK/uv-cache" "$WORK/uv-python"
fi
"$PY/bin/python3" -c "import sys, torch, transformers, sentence_transformers, fastapi; print('  python', sys.version.split()[0], 'torch', torch.__version__)"

if [ "$PY_ONLY" = 1 ]; then rm -rf "$WORK"; du -sh "$PY"; exit 0; fi

step "Server code, web UI, small data (no models: they live in Application Support)"
A="$RES/app"
mkdir -p "$A"
rsync -a --exclude __pycache__ --exclude '*.pyc' server web "$A/"
mkdir -p "$A/scripts" "$A/models/arctag" "$A/vendor/arctag" "$A/eval/multipart" "$A/index" "$A/runs" "$A/macapp"
cp scripts/fetch_models.py "$A/scripts/"
cp models/manifest.json "$A/models/"
cp models/arctag/LICENSE models/arctag/NOTICE "$A/models/arctag/" 2>/dev/null || true
cp vendor/arctag/LICENSE vendor/arctag/NOTICE "$A/vendor/arctag/"
cp eval/questions.jsonl eval/conversations.jsonl "$A/eval/"
cp eval/multipart/questions.jsonl "$A/eval/multipart/"
cp index/questions_local.jsonl "$A/index/"
cp runs/web_cache.json "$A/runs/" 2>/dev/null || true
cp macapp/app.conf macapp/server_main.py "$A/macapp/"
python3 scripts/fetch_models.py build-tagger --out "$A/vendor/arctag/libarctag.dylib" >/dev/null
"$PY/bin/python3" -m compileall -q "$A/server" "$A/macapp" "$A/scripts" >/dev/null

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

step "Codesign (ad-hoc, hardened runtime)"
# every Mach-O in the bundled Python first (inside-out), then the Python executable with its own entitlements, then the app
find "$PY" "$A/vendor" -type f \( -name '*.so' -o -name '*.dylib' \) -print0 | xargs -0 -n 200 codesign -f -s - --timestamp=none 2>/dev/null
find "$PY/bin" -type f -perm -u+x -print0 | while IFS= read -r -d '' f; do
  if file "$f" | grep -q Mach-O; then
    codesign -f -s - --timestamp=none --options runtime --entitlements macapp/python.entitlements "$f"
  fi
done
codesign -f -s - --timestamp=none --options runtime --entitlements macapp/app.entitlements "$APP"
codesign --verify --strict "$APP" && echo "  signature OK"

rm -rf "$WORK"
rmdir build 2>/dev/null || true
step "Done"
du -sh "$APP" | sed 's/^/  /'
echo "  python runtime: $(du -sh "$PY" | cut -f1)   app code: $(du -sh "$A" | cut -f1)"
echo "  open \"$APP\"   (models: python3 scripts/fetch_models.py status)"

if [ "$DMG" = 1 ]; then
  step "DMG"
  rm -f "dist/$APP_NAME.dmg"
  hdiutil create -volname "$APP_NAME" -srcfolder "$APP" -ov -format UDZO "dist/$APP_NAME.dmg" >/dev/null
  du -sh "dist/$APP_NAME.dmg" | sed 's/^/  /'
fi
