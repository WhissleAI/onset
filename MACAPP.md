# Onset for macOS

A native app: a small window (Swift, AppKit + SwiftUI, WKWebView) around the same web UI. It contains its own Python
runtime and the server. The models live in `~/Library/Application Support/<APP_NAME>/models`, not in the app or the repo.

```
Onset.app/Contents/
  MacOS/Onset                  Swift shell (macapp/Sources/*.swift)
  Resources/python/            python-build-standalone 3.11 + the server's pinned dependencies (macapp/requirements.txt)
  Resources/app/               server/, web/, scripts/fetch_models.py, models/manifest.json, libarctag.dylib (built from vendor/arctag/src)
~/Library/Application Support/Onset/
  models/                      arc-v1, arctag, embeddinggemma-2, index-data, demo-audio (models/manifest.json)
  data/                        memory (MEM_DIR=data/memory), search cache (data/search_cache.json)
  logs/server.log              server output + request log (server.log.1 … .5 = earlier runs). Keys are never logged.
```

## Name and identity
`macapp/app.conf` is the only place to rename the app: `APP_NAME`, `BUNDLE_ID`, version, placeholder icon text and colours
(or drop a real `macapp/AppIcon.icns`). The build script, the Info.plist, the Application Support folder and
`server/modelpaths.py` all read it.

## Build
```sh
scripts/build_macapp.sh              # -> dist/Onset.app (745 MB). About 30 s when the Python runtime is reused.
scripts/build_macapp.sh --no-swap    # leave the new build in dist/.staging/Onset.app (test it there)
scripts/build_macapp.sh --swap       # promote the staged build to dist/ (refused while Onset runs from dist/)
scripts/build_macapp.sh --clean      # also rebuild the Python runtime (downloads ~350 MB of wheels; ~3 min)
scripts/build_macapp.sh --dev-env    # developer build: keys missing from the Keychain come from this repo's .env
scripts/build_macapp.sh --dmg        # also dist/Onset.dmg (a second compressed copy)
```
Needs the Xcode command line tools (`swiftc`, `cc`, `codesign`, `iconutil`) and [uv](https://docs.astral.sh/uv/).
**A running app is never rebuilt over or quit.** Every build goes to `dist/.staging/<APP_NAME>.app` first. It
replaces `dist/<APP_NAME>.app` only when nothing (app or server) is running from there; otherwise it stays staged
and the script says so. Intermediates (`build/macapp`, the uv cache, the downloaded interpreter) are deleted.
The bundled Python is kept between builds while `macapp/requirements.txt` and `PYTHON_VERSION` don't change
(`Resources/python/.req_stamp`).

Python runtime: `uv python install` gets a relocatable python-build-standalone CPython, and `uv pip install` installs
the pinned requirements into it. Then tests, Tk, headers, pip, `.pyi` stubs and `torch/include` are removed.
`macapp/prune_bytecode.py` keeps precompiled `.pyc` only for the modules the server imports; it removes 6,200 files
(123 MB), mostly transformers' ~500 unused model families. Warm-up is unchanged at about 16 s.

Size, 745 MB in total:
- Python runtime: 735 MB, of which torch is 344 MB (`libtorch_cpu.dylib` alone is 215 MB), transformers 116 MB,
  scipy 88 MB, numpy 45 MB, sympy 44 MB (needed by torch), scikit-learn 35 MB (needed by sentence-transformers),
  PIL + torchvision 24 MB (EmbeddingGemma 2's processor imports them), and about 100 MB of kept bytecode.
- App code: 3 MB.

Further cuts would need a different runtime, not trimming: an ONNX/CoreML EmbeddingGemma and Arc would drop torch,
transformers, scipy and sympy, about 600 MB.

Signing: ad-hoc (`codesign -s -`) with the hardened runtime. The shell has `macapp/app.entitlements` (audio-input,
network client/server). The bundled `python3.11` has `macapp/python.entitlements`: it loads its own ad-hoc-signed
extension modules and libarctag, so library validation is off, and ctypes/torch need writable-executable memory.
The app is not sandboxed: it runs a child process and reads models from Application Support.

**Keep the mic grant across rebuilds: a stable signing identity.** macOS remembers the microphone (and Keychain)
grant per code identity. An ad-hoc signature is a new identity on every build, so each rebuild asks again (and a
stale grant can leave the mic half-working until it is reset). Sign with a real identity instead:
```sh
security find-identity -v -p codesigning           # lists usable identities
ONSET_SIGN_ID="Developer ID Application: <Team> (<TEAMID>)" scripts/build_macapp.sh
```
Every Mach-O in the bundle is then signed with that identity (still hardened runtime, no timestamp: local builds).
The first codesign may ask for the login keychain password ("codesign wants to access key"): click *Always Allow*.
No identity? Make a local self-signed one (free, this Mac only): Keychain Access > Keychain Access menu > Certificate
Assistant > Create a Certificate… Name `Onset Local Signing`, Identity Type *Self Signed Root*, Certificate Type
*Code Signing*, Create. Then `ONSET_SIGN_ID="Onset Local Signing" scripts/build_macapp.sh`. (If
`find-identity -v` doesn't list it, open the certificate in Keychain Access > Trust > Code Signing: *Always Trust*.)
The first launch of the newly signed build asks for the microphone once more; later rebuilds keep the grant.
Without `ONSET_SIGN_ID` the build stays ad-hoc.

## Models
`models/manifest.json` lists every model: source, pinned revision, size, sha256 of each file, licence.
`scripts/fetch_models.py` uses only the standard library. Both the repo setup and the app's first run use it.

| dir | what | source | size |
|---|---|---|---|
| `arc-v1` | Arc v1 ear (arc-audio 1.0.1 code + weights + Moonshine tiny) | **no public source**: `WhissleAI/arc-v1` is not anonymously reachable; `WhissleAI/arc-v1.1` (public, gated) is a different package, not a drop-in. Import a copy. | 575 MB |
| `arctag` | Arc Tagger int8 ONNX + tokenizer + meta | HF `WhissleAI/arc-tagger@221e43d` `onnx/` (v0.4.1). The v0.3.0 that this repo was tested on is no longer on the Hub; a local v0.3.0 copy is accepted. | 10 MB |
| (bundled) | `libarctag.dylib` | built from `vendor/arctag/src` (`fetch_models.py build-tagger`; the app build does it) | 0.2 MB |
| `embeddinggemma-2` | EmbeddingGemma 2 | HF `google/embeddinggemma-2@f9972ac` (Apache 2.0, not gated) | 1.5 GB |
| (repo) | search director | `server/director/` in git | 0.1 MB |
| `index-data` | local moments index + 11k Wikipedia leads | **no public source** (built from unpublished recordings). Import a copy. | 103 MB |
| `demo-audio` | spoken eval questions for the Replay buttons (optional) | **no public source**. Import a copy. | 18 MB |

```sh
python3 scripts/fetch_models.py status            # what is there
python3 scripts/fetch_models.py fetch             # download the HF models, sha256-checked, resumable
python3 scripts/fetch_models.py import arc-v1=/path/to/arc-v1 index-data=/path/to/index/data demo-audio=/path/to/eval/audio
python3 scripts/fetch_models.py verify            # sha256 everything
# developer machine: link existing copies instead of copying (no disk used)
python3 scripts/fetch_models.py link --dev arc-v1=/path/to/arc-v1   # --dev: HF cache + this repo's models/arctag, index/data, eval/audio
```
The server resolves each path in this order (`server/modelpaths.py`): its own env var (`ARC_DIR`, `EMBED_MODEL`,
`ARCTAG_MODEL_DIR`, `ARCTAG_LIB`, `INDEX_DIR`, `AUDIO_DIR`), then `MODELS_DIR/<dir>` (`MODELS_DIR` defaults to
`~/Library/Application Support/<APP_NAME>/models`), then the in-repo locations (`models/arctag`, `index/data`,
`eval/audio`). No path in the code names a particular user's home folder.

## Run
`open dist/Onset.app`. On launch the app:
1. checks the models (`fetch_models.py status`). If any are missing, a first-run screen lists them with **Download**
   (shows a progress bar) and **Import Folder…** (for the models with no public download);
2. picks a free localhost port, reusing last launch's port when it is free so the page origin, and the memory id kept
   in localStorage, stay the same. It never takes 8770 (the demo server's port);
3. starts `Resources/python/bin/python3 -I app/macapp/server_main.py --port P --parent-pid <app>`. The environment is
   clean: `MODELS_DIR`, `HEADSTART_DATA_DIR`, `HF_HUB_OFFLINE=1`, keys from the Keychain and the chosen ear;
4. shows a splash with elapsed time until `/api/health` reports `warm` (about 15–20 s on an M3 Pro once the files are
   in the page cache; the first launch after a build adds about 45 s while macOS scans the new binaries), then shows
   the UI;
5. restarts the server if it exits unexpectedly. After more than 3 crashes in 2 minutes it stops retrying and shows
   the error with **Open Logs**. If the app itself dies, the server's parent watchdog exits within about 1 s.
   **Quit** sends SIGTERM, then SIGKILL after 4 s.

Menu: About, Settings… (⌘,), Forget Memory…, Open Logs, Open Models Folder, Restart Server; View > Show Details and
Question List (⌘D), Reload (⌘R); **Replay** (every question from `/api/questions`; ⌥⌘1–9 for the first nine).
The page opens in its compact "minimal" view, which hides the question list; ⌘D, or D in the page, shows it.
A dropped websocket shows a "Reconnecting to the local server…" banner, and the page reconnects every 1.5 s.
When the server is restarted after a crash or a Settings change, the page reloads against it.

Settings stores the Anthropic, Tavily and Deepgram keys in the login Keychain (generic passwords, service = bundle id).
A saved key is shown only as "saved". Saving restarts the server with the new environment.
- **Ear**: Automatic uses Deepgram Flux when a Deepgram key is set, else Arc on this Mac. You can also pick Arc or
  Flux directly. With no Deepgram key everything runs on the local Arc ear (`EAR=arc`), which is verified below.
- **Dev fallback**: an app built with `--dev-env` reads keys missing from the Keychain, plus the non-secret settings
  (`EAR`, `FLUX_SOFT`, `DIRECTOR`), from the repo's `.env`. Release builds leave `HSDevEnvPath` empty.

Microphone: `NSMicrophoneUsageDescription` is in Info.plist and the `audio-input` entitlement is set. The
WKUIDelegate grants `getUserMedia` to the page's own origin only (127.0.0.1:P). macOS asks the user once. Both
mic modes work in the window: hold Space for push-to-talk (the web view is made first responder when the page is
shown), or click Ask for the open mic.

## Verify (unattended)
Test instances run from staging, hidden, with their own support folder, so they never touch the user's window, logs,
port or memory:
```sh
dist/.staging/Onset.app/Contents/MacOS/Onset -hidden YES -supportDir /tmp/onset_test \
  -modelsDir "$HOME/Library/Application Support/Onset/models" -snapshotDir /tmp/shots \
  -autoReplay q01.wav,q02.wav,q03.wav,q04.wav,q05.wav,q06.wav,q07.wav,q08.wav -autoQuit YES &
.venv/bin/python scripts/soak_ws.py --port <P> --n 10   # replay + push-to-talk + open-mic runs on ONE websocket
```
`-hidden YES` gives no Dock icon, no activation and an off-screen window (it still renders). `-autoReplay` replays
each file through the page in turn, waiting for the page's run to end, the way a person clicks through the list.
The app saves `01_splash_warming.png`, `02_ui_loaded.png`, `03_replay_speaking.png` and `04_replay_answered.png`
using its own snapshot APIs. It also writes `page_after_replay.txt` and `automation.log` (state changes and the mic
preconditions: secure context, getUserMedia, AudioWorklet). `scripts/preflight.py --port <P>` also works against the
app's server; P is in the first line of server.log.

Mic capture: `-autoMic 4,0.3,6,14,3` (instead of `-autoReplay`) holds push-to-talk for those seconds through the
page's own mic path, 1 s apart, and logs the page's capture counters at mid-hold and release to `automation.log`
(frames from the capture node, seconds captured and sent, AudioContext state, track state). The server logs the same
counters on every mic stop: `run N stop (client) | page: why=release held_s=6.51 frames=2440 in_s=6.51 sent_s=6.51
…`, next to `run N stopping: 6.5s of audio`. `held_s` and `sent_s` should match (the room's sound is enough). A test
instance needs its own bundle id for a microphone grant of its own (edit `BUNDLE_ID` in `macapp/app.conf` for that
build only), so it never touches the installed app's grant.

## Needs a human
- **Microphone prompt**: the first Space/Ask shows macOS's "Onset would like to access the microphone". The
  automation checks everything up to that prompt.
- **Keychain prompt**: saving keys in Settings writes Keychain items owned by the app. An ad-hoc signature changes
  with every build, so after a rebuild macOS may ask "Onset wants to use your confidential information…". Click
  *Always Allow*. A stable signature (`ONSET_SIGN_ID`, above) removes the prompt.
- Gatekeeper: a downloaded ad-hoc build is quarantined. Right-click > Open, or `xattr -dr com.apple.quarantine Onset.app`.

## Later: notarization (needs a Developer ID)
1. Sign inside-out with `codesign --options runtime --timestamp -s "Developer ID Application: …"`: every `.so`/`.dylib`
   in `Resources/python` and `Resources/app/vendor`, then `python3.11` with `python.entitlements`, then the app with
   `app.entitlements`. Notarization rejects ad-hoc and untimestamped signatures.
2. `ditto -c -k --keepParent dist/Onset.app Onset.zip && xcrun notarytool submit Onset.zip --keychain-profile <profile> --wait`
3. `xcrun stapler staple dist/Onset.app`, then build the dmg (`--dmg`) and staple that too.
With a team ID, library validation could be turned back on for python (`disable-library-validation` removed), since
every library would then carry the same team signature.

## Search cache across launches
The app sets `CACHE_PERSIST=1` and `CACHE_PATH=<data>/search_cache.json`, so a search made in an earlier session is
reused (exact meaning, or embedding nearest neighbour ≥ 0.92) while it is fresh. The TTLs are unchanged: live data
10 min, general 6 h, stable 3 days. The file holds web docs only (never cards / facts), is bounded by `CACHE_MAX` (400)
and is written atomically (tmp file + rename). A hit is always labelled "cached · as of <time>", with the date when it
is from an earlier day, and is never shown as live. **Clear Search Cache…** (app menu) calls `POST /api/cache/clear`,
which also clears the router's in-memory card / fact lookups.

The router's card / fact caches stay in memory per launch. They use monotonic-clock expiries and tuple-keyed values,
and cards carry their own "as of": persisting them is not a small change.

## Request log (server.log)
Each line starts with `[req HH:MM:SS.mmm]`. It records websocket open and close (duration, number of runs), each run's
start (replay file / push-to-talk / open mic, ear) and stop (seconds of audio), and each question's endpoint
(semantic / ear) and completion: ready ms, path, lane, fallback, error, and the first 80 characters of the question.
It also logs open-mic "heard" (not a request), answer errors, ear fallback/errors, and any crash of the websocket
handler with its traceback. If the user reports "stuck after N questions", the last lines show whether the question
never reached an endpoint (the ear), never finished (the lanes/answer) or the socket closed. `REQUEST_LOG=0` turns
it off.

## Training rows (Settings > Improve Onset, `TRAIN_COLLECT`)
**On by default in this dev build.** Settings > Improve Onset > "Use my questions to improve the models" (sets
`TRAIN_COLLECT=1`/`0` for the server; a dev `.env` can set it too). When on, each answered question adds one line to
`~/Library/Application Support/Onset/data/train/rows.jsonl` (server/trainlog.py): timings, route and tiers, paid /
free calls, verify / re-check outcome, calibration features and confidence, endpoint trigger, ear, and the question
and query with personal data masked (5+ digit numbers, emails, handles, birth years, the tagger's redacted words).
No audio, no answer text (its word count only), no keys; the browser's client id is stored only as a salted hash
(salt in `data/train/.salt`, per install). Nothing is uploaded. The training loop (`scripts/train_cycle.sh`,
PROGRESS.md "Continuous training") reads the file locally. To delete the rows: remove `data/train/`.

## Troubleshooting
- **Stuck on "Models needed"**: `python3 scripts/fetch_models.py status`. Import the local-only models; a link
  pointing at an incomplete copy has to be removed before Download can fill the folder.
- **Server keeps restarting**: Open Logs. The usual cause is a model folder that is present but incomplete (run
  `fetch_models.py verify`).
- **Mic does nothing**: System Settings > Privacy & Security > Microphone > enable Onset. To ask again:
  `tccutil reset Microphone ai.whissle.onset`.
- **Answers say "extractive"**: no Anthropic key (Settings), or the API is unreachable.
- **Web inspector**: right-click > Inspect Element (enabled in this build).
- **Memory gone after relaunch**: the page origin includes the port. The app reuses the last port; if another
  program has taken it, the browser-side id changes. The server-side store in `data/memory` is kept.
