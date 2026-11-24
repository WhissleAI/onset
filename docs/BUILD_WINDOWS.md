# Building Onset for Windows

Onset for Windows is the same engine and the same page as the Mac app, inside a small Windows shell:

| | Mac | Windows |
|---|---|---|
| Window | Swift + WKWebView (`macapp/Sources`) | Python + Edge WebView2 via pywebview (`winapp/onset_win.py`) |
| Engine | `macapp/server_main.py` (FastAPI) | the same file |
| Page | `web/index.html` + `onset.js` | the same files |
| Ear (Arc Mini) / Arc Tagger | `libarc.dylib` / `libarctag.dylib` | `vendor/arc/arc.dll` / `vendor/arctag/arctag.dll` (committed) |
| Python packages | `macapp/requirements.txt` | `macapp/requirements-native.txt` (ONNX, **no torch**) + `winapp/requirements-win.txt` |
| Models | first-run download, `models/manifest.release.json` | the same download |
| Keys | Keychain | Windows Credential Manager |
| Data | `~/Library/Application Support/Onset` | `%LOCALAPPDATA%\Onset` (`models`, `data`, `logs`, `settings.json`) |
| Installer | DMG, notarized | Inno Setup `.exe`, per-user, no admin prompt |
| Updates | GitHub releases (`-arm64.dmg`) | GitHub releases (`-win-x64-setup.exe`), once a day |

Build from **main**. You need no C compiler and no Mac.

## One-time setup (Windows 10 1809+ or 11, x64)

```powershell
winget install Git.Git JRSoftware.InnoSetup
irm https://astral.sh/uv/install.ps1 | iex          # uv: fetches the Python runtime and the packages
# open a NEW PowerShell so uv and iscc are on PATH
git clone https://github.com/WhissleAI/onset.git
cd onset
```

The Edge WebView2 Runtime is already on Windows 11 and on most Windows 10 PCs. The installer checks for it and points to
Microsoft's download if it is missing.

## Build

```powershell
git pull
powershell -ExecutionPolicy Bypass -File winapp\build_windows.ps1
```

That makes:

- `dist\windows\Onset\`: the app folder (Python runtime, engine, page, DLLs). You can run it in place:
  `dist\windows\Onset\python\pythonw.exe -I -B dist\windows\Onset\winapp\onset_win.py`
- `dist\windows\Onset-<version>-win-x64-setup.exe` and its `.sha256`. The version comes from `macapp/app.conf`
  (`APP_VERSION`), so Mac and Windows share one version.

The first build takes about 5 minutes. Later builds reuse the Python runtime unless the requirements change. Add
`-Clean` to rebuild everything, or `-NoInstaller` to skip Inno Setup.

## Test before you hand it out

1. **Engine smoke.** This downloads the models once (about 1.2 GB to `%LOCALAPPDATA%\Onset\models`) and asks a
   six-question conversation, including follow-ups:
   ```powershell
   powershell -ExecutionPolicy Bypass -File winapp\build_windows.ps1 -NoInstaller -Smoke
   ```
   Expect `6/6 passed`.
2. **Install and use it like a customer.** Run the setup `.exe`, open Onset from the Start menu, then:
   - [ ] First run shows "One-time setup", then Download with progress, then "Getting ready…", then the page "Ask anything."
     (Skipped if the smoke test already downloaded the models.)
   - [ ] Type `who is the prime minister of India`, then `how old is he`. The second answer is about Modi.
   - [ ] Click the mic, allow the microphone, and ask a question out loud. The words appear as you speak and the answer
     comes when you stop. Hold **Space** to talk works too.
   - [ ] The gear opens Settings. Save restarts the engine (splash, then the page again).
   - [ ] Close the window. Within about 2 s, Task Manager shows no `python.exe` left from the Onset folder.
   - [ ] Reopen it. It starts in about 10 s with no download.
   - [ ] Uninstall from Settings > Apps. It asks whether to remove the models too.
3. Send back `%LOCALAPPDATA%\Onset\logs\` (`app.log`, `server.log`) with anything that failed.

## What is different on Windows (for now)

- **"Find my files", Music and opening apps** are Mac-only today (`server/device.py`, Spotlight and AppleScript). On
  Windows those questions go to the web like any other question. The Windows versions are next (Windows Search, media
  keys, `start`).
- **Microphone.** The shell grants Onset's own local page the microphone (`allow_microphone` in `onset_win.py`). If
  that hook fails, WebView2 shows its own prompt once. Windows' Settings > Privacy > Microphone still applies.
- **Speed.** Arc Mini's fast paths are tuned for Apple Silicon (NEON). On x64 the plain C path runs. Check that live
  words keep up on a typical laptop, and say so if they don't.
- **Signing.** Unsigned installers trigger SmartScreen ("Windows protected your PC" > More info > Run anyway). With a
  code-signing certificate, set `$env:ONSET_WIN_CERT` (the `.pfx` path) and `$env:ONSET_WIN_CERT_PASSWORD` before
  building. `signtool.exe` from the Windows SDK must be on PATH.

## Releasing

Upload `Onset-<version>-win-x64-setup.exe` and its `.sha256` to the **same** GitHub release as the Mac DMG
(`v<version>`). The Windows app's update check looks for the `-win-x64-setup.exe` asset on the latest release.

## When something changes on main

- **Python code, the page, or models:** just rebuild. Nothing Windows-specific to do.
- **New Python packages:** add them to `macapp/requirements-native.txt` (the engine) or `winapp/requirements-win.txt`
  (the shell only). Don't add torch: the Windows app runs everything on ONNX Runtime and C.
- **`vendor/arc/src` or `vendor/arctag/src` (the C libraries):** someone with a Mac or Linux box runs
  `scripts/build_windows_libs.sh` (mingw-w64) and commits the two DLLs.
- **POSIX-only Python** (`resource`, `fcntl`, `os.kill(pid, 0)`, `/usr/bin/...`, `osascript`): guard it with
  `sys.platform`. `os.kill(pid, 0)` on Windows *terminates* the process. `macapp/server_main.py` shows the
  Windows-safe check.

## Files

- `winapp/onset_win.py`: the shell (window, model check and download, engine supervisor, Settings, bridge, updates)
- `winapp/build_windows.ps1`: the one-command build
- `winapp/onset.iss`: the installer
- `winapp/requirements-win.txt`: the shell's packages
- `winapp/Onset.ico`: the icon
- `scripts/build_windows_libs.sh`: rebuild `arc.dll` and `arctag.dll` (Mac or Linux, mingw-w64)
- `scripts/smoke_typed.py`: the conversation smoke test (any OS)
