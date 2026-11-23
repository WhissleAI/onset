# Onset for macOS

A native app: a small window (Swift, AppKit + SwiftUI, WKWebView) around the same web UI. It contains its own Python
runtime and the server. The models live in `~/Library/Application Support/<APP_NAME>/models`, not in the app or the repo.

```
Onset.app/Contents/
  MacOS/Onset                  Swift shell (macapp/Sources/*.swift)
  Resources/python/            python-build-standalone 3.11 + the server's pinned dependencies (macapp/requirements.txt)
  Resources/app/               server/, web/, scripts/fetch_models.py, models/manifest.json, libarctag.dylib (built from vendor/arctag/src)
~/Library/Application Support/Onset/
  models/                      arc-v1, arctag, embeddinggemma-2, answer-lm, index-data, demo-audio (models/manifest.json)
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
| `answer-lm` | Qwen2.5-0.5B-Instruct, 4-bit ONNX: writes the answer on this Mac | built from `onnx-community/Qwen2.5-0.5B-Instruct@22942cb` by `scripts/native/quantize_answer_lm.py` (Apache 2.0) | 338 MB |
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

## Native runtime (ONNX_RUNTIME, branch feat/native)
Goal: every model runs on ONNX Runtime or C inside the notarized app, with no PyTorch at runtime and the models shipped
in the bundle instead of downloaded. Status: **EmbeddingGemma 2 is done** (ONNX, 8-bit weights, parity below). The Arc
encoder and trunk are exported with exact parity. **Arc's heads, Engine and decoder still run on torch**, so the app
still needs torch and transformers. `ONNX_RUNTIME=1` (default off) switches the embedder now and is where the Arc
port will plug in.

### Every model the server loads at runtime
| model | framework today | files / size | loaded by | called from |
|---|---|---|---|---|
| EmbeddingGemma 2 (text 270M + audio 300M; vision unused) | torch + transformers + sentence-transformers. Docs + audio on MPS bf16, queries on CPU f32 (a second text-only copy) | `embeddinggemma-2/model.safetensors` 1.49 GB + tokenizer 32 MB | `server/embed.py` | query vectors: speculator (rank), lanes.local_text, knowledge (11k leads), sitebrain, keystone/hook, persona (categories). Doc vectors: speculator (fresh web docs, rerank). Audio vectors: lanes.local_audio (the audio lane, local-routed questions) |
| **→ ONNX**: same model | onnxruntime + tokenizers + numpy | `embeddinggemma-2-onnx/` 669 MB (text.onnx 155, audio.onnx 346, embed_tokens.npz 135, tokenizer 32) | `server/embed_onnx.py` (ONNX_RUNTIME=1) | same calls |
| Qwen2.5-0.5B-Instruct (answer writer) | onnxruntime + tokenizers (no torch): 4-bit weights incl. a 4-bit tied embedding/LM head, logits for the last position only (`scripts/native/quantize_answer_lm.py`), greedy, ≤ 96 new tokens, one generation at a time on a worker thread | `answer-lm/model.onnx` 327 MB + tokenizer 11 MB | `server/answer_lm.py` (warmed by `answer.warm()`) | `server/answer.py` Stream (drafts, soft and final answers). Measured 2026-11-23 on an M3 Pro (CPU, ANSWER_LM_THREADS=0): ~620-token prompt (5 short sources) first token ~0.5 s, ~1,460 tokens ~1.6 s; decode 65-110 tok/s. The earlier torch/MPS bf16 path was faster to first token (0.18 s at ~540 tokens) but a 988 MB download |
| Arc v1 Mini (Moonshine Streaming tiny encoder 6 layers + Arc trunk 5 SeqLayers + tag/entity/diar/bio/word heads) | torch + transformers (MoonshineStreaming*) via the `arc_audio` package | `arc-v1/model.safetensors` 117 MB, `moonshine-streaming-tiny/model.safetensors` 176 MB (encoder + tagged decoder), `decoder_v1e` 138 MB, `decoder_release` 138 MB | `server/ear.py` arc() | EAR=arc: the local ear (builtin transcript = Moonshine decoder beam search). EAR=flux: `server/arcmeta.py` ARC_META=mini runs Arc's heads on Deepgram's words (command_complete, act/intent, speaker, bio events). `server/arcstate.py`: Arc's voice embedding per frame (diarization="voice") |
| Arc Transcriber (Arc v1.1 full: Parakeet-TDT 0.6B tap + adapters) | torch (MPS), `arc_transcriber` code | not bundled (~3.1 GB RSS) | `server/arcmeta.py` transcriber() | ARC_META=transcriber only (opt-in read-out, never default). Must stay off in the public app |
| Arc Tagger | **C** (`libarctag.dylib`: its own ONNX reader + Accelerate, no onnxruntime) via ctypes | `arctag/arctag_int8.onnx` 9 MB + tokenizer + meta; lib 0.2 MB | `server/tagger.py` | session (endpoint gate, frame read-out: entity/number/pii/events/intent/act/state), heard, memory, persona, confidence, trainlog; `server/safety.py` (TAGGER_SAFETY=1): profanity/toxicity span heads + utterance flags on the question handle, and a **second streaming handle** for the answer text (AnswerMasker) |
| Search director (Mamba-3, int8) | **numpy** (int8 rows, f32 scale, dequantised at load) | `server/director/director.npz` 56 KB | `server/director.py` | DIRECTOR=on (Flux updates: end of turn, fire/draft/keep/addressed/route/pii heads) |
| Keystone director, VOI, TTL, answer hash, verifier | **numpy / JSON** | `server/keystone/{kdir_model,voi_model,ttl_model,verify_eq}.json`, `ahash_model.npz` (<0.3 MB) | `server/keystone/*` | KDIR=1 / Keystone store |
| Answer calibration, ASR confidence | **numpy / JSON** (logistic) | `server/calib_model.json`, `server/conf_model.json` | `server/calibrate.py`, `server/confidence.py` | every answer / every ear update |
| ArcState voice prior | JSON | `server/arcstate_voice_mu.json` | `server/arcstate.py` | ARC_STATE |

Data (not models, but in the same folder): `index-data` 102 MB (moments + their text/audio vectors, 11k Wikipedia leads
+ vectors), `demo-audio` 17 MB (replay questions).

### EmbeddingGemma 2 on ONNX Runtime
Export (`scripts/native/export_embed.py`, developer machine): the text tower takes embeddings in and returns the
masked-mean pooled 768-d vector, so audio soft tokens go through the same graph. The audio tower returns soft tokens
plus a mask. The token table is int8 per row, gathered in numpy. The log-mel front end is numpy (window + mel bank
precomputed). The tokenizer is `tokenizers`. Two export fixes: the audio tower's sliding-window mask is rebuilt
without vmap, and `unfold` becomes an index gather. The mask must be **bool**: in eager mode Transformers passes an
additive float mask that `Gemma4AudioAttention.masked_fill` inverts, so torch eager is wrong (cos 0.38 per token)
and sdpa (production) is right.
Weights (`scripts/native/quantize_embed.py`): 8-bit weight-only MatMulNBits, block 32, accuracy_level 4. Measured
alternatives, query/doc cosine against torch and rerank top-5 agreement:

| graph | query cos | doc cos | rerank top-5 | query p50 | 8 docs p50 | text size |
|---|---|---|---|---|---|---|
| f32 | 1.00000 | 0.99991 | 0.974 | 24 ms | 1061 ms | 548 MB |
| **8-bit weights, acc 4 (shipped)** | **0.99988** | **0.99959** | **0.958** | **13-16 ms** | **547-605 ms** | **155 MB** |
| 8-bit weights, acc 0 | 0.99995 | 0.99978 | 0.968 | 103 ms | 1201 ms | 155 MB |
| 4-bit weights | 0.98803 | 0.97402 | 0.728 | 29 ms | 1078 ms | 87 MB |
| dynamic int8 (activations too) | 0.98922 | 0.96697 | 0.694 | 15 ms | 482 ms | 140 MB |

The rerank ceiling is below 1.0 even at f32 because the torch reference embeds docs in bf16 on MPS. CoreML EP: fails
on these graphs (NeuralNetwork and MLProgram), so it runs on CPU only.

Parity, shipped graphs vs the torch path the server runs (`scripts/native/parity_embed.py`, `runs/native/embed_parity.json`;
100 eval questions, 300 Wikipedia leads, 222 local moments, 20 spoken local questions; run in a venv with no torch):
- cosine: queries 0.99988 (min 0.99972), leads 0.99959 (min 0.99836), moments 0.99974, audio 0.99984 (min 0.99934);
- shipped 11k-lead index (torch-built vectors) with each side's query: top-1 agreement 1.00, top-5 0.978;
- rerank (20 nearest leads, each side embeds them itself): top-5 agreement 0.958;
- local moments eval (`index/eval_local.py`, 20 spoken questions): every mode identical (text r@1 0.80, audio 0.65,
  fused text+audio+lexical 0.85, MRR equal), except fused_tal on the ASR text: r@1 0.80 → 0.75 (one question);
- latency p50: query 13 ms vs 55 ms (CPU f32 torch); 8 docs 605 ms vs 265 ms (**MPS GPU**); 3 s audio 329 ms vs
  123 ms (MPS); warm-up 2.3 s vs 16 s; peak RSS 2.6 GB vs 3.0 GB.
The doc and audio batches are 2-2.5x slower on CPU than torch on the GPU. End to end (58-set real-audio replay,
`eval/run_ab.py --modes headstart`, WEB_BACKEND=wikipedia, EAR=flux; torch, ONNX, torch run one after another):
58/58 correct in all three runs. p50 ready: torch 1,632 and 1,436 ms, ONNX 1,618 ms. Paired median ONNX − torch: +30
and +52 ms, against torch − torch −74 ms. So ONNX is within run-to-run noise, not shown faster or slower, and
`ONNX_RUNTIME` stays **default off** (the "no slower p50" bar is not proven).

### Arc v1 Mini: what is exported
`scripts/native/export_arc.py` → `server/arc_onnx.py`:
- `MSTStream`: `arc_audio.mst.MSTStreamEncoder` line for line (same buffers, windows, 240 ms look-ahead, trimming).
  The frame embedder, the 6 sliding-window layers and the final norm are ONNX; the two stride-2 convs are numpy.
- `TrunkStep`: `Trunk.step` for one 10 Hz row with the recurrent state explicit: `h_in/h_out`, `bx_in/bx_out`
  `[5,B,8,64,64]` (zeros = the first step). `Pooler` is the same numpy code as arc_audio's.
- Parity (`scripts/native/parity_arc.py`, 12 eval questions, live 100 ms chunks + 1.2 s silence, `runs/native/arc_parity.json`):
  encoder frames max |diff| 1.3e-5 (cos ≥ 0.9999997), trunk outputs max |diff| 1.0e-4 (cos ≥ 0.9999997), frame-tag
  probabilities max |diff| 2e-5, **0 tag flips** in 26k frame×tag cells. Compute per 100 ms of audio: 25.7 ms vs 20.9
  ms p50 (p90 61.5 vs 34.9): many small ORT calls. f32: 66 MB trunk + 14 MB encoder.

Not exported yet (Arc still needs torch + transformers): the heads and Engine (`arc_audio/engine.py`, 1,392 lines that
call the tag/entity/diar/bio/word heads on torch tensors), the builtin transcript (Moonshine decoder beam search through
Transformers `generate`, logits processors for name bias, the tagged-decoder read-out) and word alignment. Plan: (1) the
heads as numpy/ONNX functions with the same signatures (all Linear/LayerNorm/GELU/einsum, plus the pointer and
pooling maths); (2) the Engine's ~20 torch calls replaced with numpy; (3) the decoder as two ONNX graphs (first step,
then one step with a KV cache) plus a numpy beam search with the same loop-fix and bias; (4) parity on tags/events,
command_complete timing, names/numbers and transcript WER on eval/audio; (5) drop torch. Smallest fallback until then:
keep torch + transformers for Arc only, and drop sentence-transformers, scikit-learn, scipy, torchvision, pillow and
accelerate (about 150 MB). The ONNX embedder no longer needs them; scipy is only Arc's resampler for non-16 kHz input,
which the server never sends.

### Switch
`ONNX_RUNTIME=1` (server/config.py) makes `server/embed.py` use `server/embed_onnx.py` for every query, doc and audio
embedding. torch, transformers and sentence-transformers are then never imported for embeddings. The model folder is
`MODELS_DIR/embeddinggemma-2-onnx` (or `EMBED_ONNX_DIR`), listed in `models/manifest.json` with sha256 (optional
entry: only needed when the flag is on; when it is on, `embeddinggemma-2` is not needed). `ONNX_THREADS` (default 4)
sets intra-op threads. `macapp/requirements-native.txt` is the dependency set for when Arc is ported too.

### Sizes (runtime + models)
| | Python runtime | models | total |
|---|---|---|---|
| today (`requirements.txt`, safetensors) | 736 MB (torch 314, scipy 85, transformers 64, numpy 43, sympy 33, sklearn 28, PIL 14, torchvision 10) | arc-v1 574 + arctag 10 + embeddinggemma-2 1,525 + index-data 102 (+ demo-audio 17) = 2,211 (2,228) MB | 2.95 GB |
| ONNX_RUNTIME=1 now (Arc on torch; ST, sklearn, scipy, PIL, torchvision, accelerate dropped, onnxruntime added) | est. ~670 MB | 574 + 10 + **669** + 102 = 1,355 MB | est. ~2.0 GB |
| all native (`requirements-native.txt`, measured install: 172 MB site-packages + ~39 MB interpreter, before bytecode) | ~215 MB | Arc est. ~300 MB f32 (~90 MB at 8 bit) + 10 + 669 + 102 | est. ~1.1-1.3 GB |
The "est." rows are not built yet: they are package sums, not a measured bundle. Arc trimming that needs no port:
`decoder_v1e.safetensors` (138 MB) only serves `decoder="v1e"`, which the server never asks for, and `README.md`,
`examples/`, `results/`, `tests/` are not needed at runtime. The licence files stay.

### Licences and redistribution (the app is a public download)
| component | licence | redistribution | the app must include |
|---|---|---|---|
| EmbeddingGemma 2 (and our ONNX build) | Apache 2.0 (model card → ai.google.dev/gemma/docs/gemma_4_license); deployments must follow the Gemma Prohibited Use Policy | allowed, modified files must say so | `embeddinggemma-2-onnx/LICENSE` (Apache 2.0 text) + `NOTICE` (what we changed); link to the Prohibited Use Policy in the app's About/licences |
| Arc v1 (code + weights) | **CC BY-NC 4.0**, Whissle; "commercial use requires a separate licence from WhissleAI" | free non-commercial redistribution with attribution is allowed; WhissleAI is the rights holder, so shipping its own app is its decision, but **a paid / ad-supported / for-profit-business use of the downloaded app is outside the licence users receive** | `arc-v1/LICENSE`, `arc-v1/NOTICE` |
| Qwen2.5-0.5B-Instruct (answer-lm, modified: 4-bit ONNX) | Apache 2.0 (Qwen team, Alibaba Cloud) | allowed; keep the licence | `answer-lm/LICENSE` (copied to Resources/Licenses as `Qwen2.5-0.5B-Instruct-LICENSE.txt`) |
| Moonshine Streaming tiny (modified by Whissle, inside arc-v1) | MIT (Useful Sensors / Moonshine AI) | allowed; keep the copyright + licence | `arc-v1/moonshine-streaming-tiny/LICENSE`, `ORIGIN.md` |
| jellyfish Metaphone fallback (in arc_audio/rescorer.py) | MIT | allowed | covered by arc-v1/NOTICE |
| Arc Tagger (weights + libarctag C code) | **CC BY-NC 4.0 + Whissle terms** (the terms exclude use "in or with a product or service" and "by or for a for-profit organisation in its business") | the same NC flag as Arc; Whissle Inc. owns it | `models/arctag/LICENSE`, `models/arctag/NOTICE`, `vendor/arctag/LICENSE`, `vendor/arctag/NOTICE` |
| cJSON (in libarctag) | MIT | allowed | `vendor/arctag/third_party/cjson/LICENSE` |
| ONNX Runtime | MIT | allowed | its `ThirdPartyNotices.txt` + LICENSE (inside the wheel; keep them when trimming) |
| tokenizers (Apache 2.0), numpy (BSD), soundfile (BSD) + libsndfile (**LGPL-2.1**, shipped as a dylib in `_soundfile_data`) | as listed | allowed; LGPL: keep it a separate replaceable dylib and ship its licence | each package's licence; libsndfile LGPL text |
| Search director, Keystone, calibration models | this repo (trained by us) | ours | — |
| index-data | built from third-party videos: NBA/cricket/Olympics broadcasts, court audio, meeting role-plays; Wikipedia leads CC BY-SA 4.0 | **NOT redistributable**: third-party broadcast and video clips, so it must not go in a public download. The Wikipedia leads can ship with attribution + CC BY-SA | ship only a rebuilt index without the private / broadcast moments, or none |
| demo-audio | synthetic TTS (macOS `say` voices, Kokoro-82M Apache 2.0) + recorded voices | **flag**: macOS system-voice output and the recorded voices need clearing before public redistribution | optional: leave out of the public build |

Nothing here forbids redistributing the app as a free, non-commercial download. The two blockers are the NC terms
of Arc and the Arc Tagger, which Whissle has to decide on for any commercial or business use, and index-data /
demo-audio, which must stay out of the public bundle.

### Hand-off to the release build (feat/release)
- Bundle `MODELS_DIR` contents: `arc-v1`, `arctag`, `embeddinggemma-2-onnx` (with ONNX_RUNTIME=1; then
  `embeddinggemma-2` is not needed) and, for a public build, no `index-data` / `demo-audio` (see licences).
  Paths, sizes and sha256 are in `models/manifest.json` (`fetch_models.py verify`).
- Set `ONNX_RUNTIME=1` in the server environment (macapp/Sources/Server.swift) once the release build switches. Point
  `MODELS_DIR` (or the per-model env vars) at the bundled `Resources/models`.
- Add `onnxruntime==1.32.0` to `macapp/requirements.txt`. While Arc needs torch, drop sentence-transformers, torchvision,
  pillow, accelerate (and scikit-learn/scipy that come with them) only after the build is checked with `ONNX_RUNTIME=1`.
  `macapp/prune_bytecode.py` must keep `onnxruntime/capi`.
- Signing: `onnxruntime/capi/*.so` and `libonnxruntime*.dylib` must be signed like the other Mach-Os.

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

Settings stores the Deepgram, Tavily and Telem keys in the login Keychain (generic passwords, service = bundle
id; `<bundle id>.test` for a `-supportDir` test instance).
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

## Release
A public build anyone can download: the models are inside the app, nothing is fetched at first run, no `.env` is read,
every Mach-O is signed with the Developer ID and a timestamp, and the DMG is notarized and stapled.

```sh
ONSET_SIGN_ID="Developer ID Application: Whissle LLC (XXK95K57U5)" NOTARY_PROFILE=<profile> \
  nice -n 10 scripts/build_macapp.sh --release
# -> dist/release/Onset.app, dist/release/Onset-<version>-arm64.dmg (+ .sha256), notarize-*.json (submission ids)
```
- `--release` refuses `--dev-env`, needs a `Developer ID Application:` identity and a notarytool keychain profile
  (`xcrun notarytool store-credentials <profile> --apple-id <apple id> --team-id XXK95K57U5`, with an app-specific
  password; the profile lives in the login Keychain). `--no-notarize` builds and signs only. It writes to
  `dist/release/` and never touches `dist/Onset.app` or a running Onset.
- **Models** (`--bundle-models`, implied): `scripts/fetch_models.py bundle` copies what `models/release_profile.json`
  allows from `MODELS_DIR` (or `MODELS_SRC`) into `Contents/Resources/models` (APFS clones, sha256-checked) and writes
  the bundle's own `manifest.json` (`"bundled": true`, exactly the files shipped). The app sees that manifest and
  uses the bundle read-only as `MODELS_DIR`; its model check runs `fetch_models.py status --manifest <bundle>`.
  The profile ships Arc v1, Arc Tagger, EmbeddingGemma 2, the Wikipedia leads (index-data/knowledge) and the Replay
  questions voiced with Kokoro. It leaves out the local-moments index (third-party recordings, not ours to
  redistribute) and questions voiced with macOS system voices; `/api/questions` lists only what is installed.
- **Native runtime** (feat/native): `MODELS_MANIFEST=<onnx manifest> ONNX_RUNTIME=1 REQUIREMENTS=<slim
  requirements> scripts/build_macapp.sh --release`. A model the profile does not name is bundled whole when required,
  so the ONNX manifest needs no profile change. `ONNX_RUNTIME=1` reaches the server through Info.plist `HSServerEnv`.
- **Signing**: every Mach-O is found by its header (not its extension) under `Resources/python` and `Resources/app`,
  signed inside-out with `--options runtime --timestamp`; the Python executables get
  `macapp/python.release.entitlements` (network only: library validation stays on since every library has our team
  signature, and no JIT / unsigned-executable-memory: the release smoke test runs torch, ctypes and libarctag without
  them); the app gets `macapp/app.entitlements` (audio-input, network client/server). `codesign --verify --strict --deep`.
- **DMG**: the app plus an Applications link, lzma (`ULMO`), signed. Notarized with `notarytool submit --wait`; the
  ticket covers the app inside, so the app is stapled, the DMG rebuilt around the stapled app, notarized again and
  stapled (an app copied out of the DMG then opens offline too). `shasum -a 256` next to it.
- **Licences**: `Contents/Resources/Licenses` (`macapp/licenses/THIRD_PARTY.md` is the index; Arc v1, Moonshine,
  Arc Tagger licence + NOTICE, Apache 2.0, Python). Python packages keep theirs in their `.dist-info`.

Verify:
```sh
spctl -a -vvv -t install dist/release/Onset-*.dmg     # accepted, source=Notarized Developer ID
spctl -a -vvv dist/release/Onset.app
xcrun stapler validate dist/release/Onset-*.dmg
# clean account: a fresh support dir and the test Keychain service (<bundle id>.test, used by every -supportDir
# instance), so the user's keys and defaults are never read or changed. Run it from the mounted (read-only) DMG.
hdiutil attach -nobrowse -readonly dist/release/Onset-*.dmg
"/Volumes/Onset/Onset.app/Contents/MacOS/Onset" -hidden YES -supportDir /tmp/onset_clean -snapshotDir /tmp/onset_shots \
  -autoOnboarding skip -autoReplay q02.wav -autoQuit YES
```
`00_onboarding.png` shows the first-run sheet; `automation.log` says "onboarding shown", then "onboarding skipped
(free mode)"; `page_after_replay.txt` holds the answer from the free path (Arc ear, Wikipedia, extractive answer).

Publish: `gh release create v<version> --draft --title ... --notes-file ... <dmg> <dmg>.sha256` makes a DRAFT; a
person publishes it. The README's "Download for Mac" links `releases/latest`.

## First run: keys
A release build (no `--dev-env`) opens a sheet before the server starts (so the models load once, with the keys).
It explains each key, what it powers, what works without it and where to get one; each has **Test** (one free call:
Tavily `GET /usage`, Deepgram `GET /v1/projects`, Telem an empty `POST /v1/search` that
is refused 400 after the key is accepted, never billed) showing "Works" or a plain error. Keys go only to the
Keychain. **Continue** saves them, **Skip** runs the free mode. *Set Up Keys…* (app menu) reopens it; Settings has
the same rows. The server environment follows the keys: web = Tavily with a Tavily key, else Telem, else DuckDuckGo
(`WEB_FAILOVER=telem` when both are set, `VERIFY_FETCH=telem` with Telem alone); ear = Deepgram Flux with a Deepgram
key, else Arc; answers are always written on this Mac by Qwen2.5-0.5B-Instruct (`answer-lm`; no key; a developer
`.env` can still set `ANSWER_BACKEND=claude` + `ANTHROPIC_API_KEY`). `KNOWLEDGE_DIR` points at the
models' `index-data/knowledge` (the app never loaded the Wikipedia leads before: the default path is in-repo).

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
- **Answers say "extractive"**: the on-device answer model is missing (`fetch_models.py fetch answer-lm`) or failed to
  load (Open Logs), or it gave no first token within `ANSWER_DEADLINE_MS` (4 s).
- **Web inspector**: right-click > Inspect Element (enabled in this build).
- **Memory gone after relaunch**: the page origin includes the port. The app reuses the last port; if another
  program has taken it, the browser-side id changes. The server-side store in `data/memory` is kept.
