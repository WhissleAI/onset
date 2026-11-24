#!/usr/bin/env bash
# Cross-build the two C libraries for Windows x64 (arc.dll = Arc Mini, the ear; arctag.dll = Arc Tagger) with mingw-w64,
# on macOS or Linux. The DLLs are committed in vendor/arc and vendor/arctag, so the Windows build machine needs no C
# compiler (docs/BUILD_WINDOWS.md). Rebuild them whenever vendor/*/src changes.
#   brew install mingw-w64        (macOS)     apt install gcc-mingw-w64-x86-64   (Linux)
#   scripts/build_windows_libs.sh
# Only the Windows UCRT + KERNEL32 are imported (static libgcc): nothing else to ship.
set -euo pipefail
cd "$(dirname "$0")/.."
CC="${CC:-x86_64-w64-mingw32-gcc}"
command -v "$CC" >/dev/null || { echo "need $CC (brew install mingw-w64)"; exit 2; }
v=vendor/arc
"$CC" -O2 -std=c11 -shared -pthread -static \
  -I$v/include -I$v/src $v/src/*.c -lm -o $v/arc.dll
v=vendor/arctag
"$CC" -O3 -shared -static -DARCTAG_BUILD_SHARED -I$v/include -I$v/src -I$v/third_party/cjson \
  $v/src/*.c $v/third_party/cjson/cJSON.c -o $v/arctag.dll
for f in vendor/arc/arc.dll vendor/arctag/arctag.dll; do
  bad=$(x86_64-w64-mingw32-objdump -p "$f" | awk '/DLL Name/ {print $3}' | grep -viE '^(KERNEL32|api-ms-win-crt-.*|msvcrt)\.dll$' || true)
  [ -z "$bad" ] || { echo "$f imports $bad (would need shipping)"; exit 1; }
  echo "built $f ($(du -h "$f" | cut -f1))"
done
