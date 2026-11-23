## Onset 0.2.0 for Mac

Onset is voice search that starts searching while you're still asking. Hold Space and speak, or type: Onset hears
the question as it forms (Arc), works out what you mean after a few words (Arc Tagger), searches the right place
(local Wikipedia facts, Wikidata, or the live web) and has the answer ready when you stop, with its sources.

**Download:** `Onset-0.2.0-arm64.dmg` below. Open it, drag Onset to Applications, open Onset. Signed with Whissle
LLC's Developer ID and notarized by Apple. Check the download: `shasum -a 256 Onset-0.2.0-arm64.dmg` should match
`Onset-0.2.0-arm64.dmg.sha256` (`8951055c3873a085ae29d85017017a1ee062b8d9a8efbc1cd6605e2c3a28070a`).

### What's new
- **Answers written on your Mac, no key needed.** A 4-bit Qwen2.5-0.5B writes the short answer from the sources
  (first words in about 0.5 s on an M3 Pro). Add an Anthropic key to use Claude instead.
- **A lighter, faster ear: Arc Mini.** Arc Mini runs in C with no PyTorch. It loads in under a second instead of
  4.5 s, uses about 400 MB of memory instead of 1.1 GB, and makes fewer mistakes in noise (35% vs 43.5% word errors
  at 10 dB SNR) with the same accuracy on clean speech.
- **Better words for questions.** A 32 MB spoken-question language model re-ranks Arc's final words: held-out
  word errors 6.84% → 6.23%, names and numbers 85.5% → 86.4%. Words still stream as fast, and Onset still knows
  you've stopped at the same moment.
- **A smaller app; models download on first launch.** The app no longer carries its models. On first launch it
  downloads them once (1.16 GB), checks each file's checksum, and works offline from then on.
- **Updates from GitHub.** Onset checks this page about once a day (or Onset > Check for Updates…), and installs a
  new version only if it is signed by Whissle LLC and its checksum matches.
- **Smaller search model.** EmbeddingGemma 2 now runs as an 8-bit ONNX model (669 MB instead of 1.5 GB; rankings
  match the full model 96% of the time in the top five).

### Keys (all optional)
Onset works with no keys: Arc listens on your Mac, answers come from Wikipedia and Wikidata, and Qwen on your Mac
writes the answer. On first launch a setup sheet explains each key and has a **Test** button for it (one free
call). Change keys any time in Settings (⌘,).

| Key | What it adds | Without it | Get one |
|---|---|---|---|
| Deepgram | Deepgram Flux hears you in the cloud: the quickest turn-taking | Arc on your Mac (private) | https://console.deepgram.com/signup |
| Tavily | Live web search: fresh, ranked results for news, sports, prices | Telem if you add its key, else Wikipedia | https://app.tavily.com |
| Telem | Web search router across providers; used without Tavily, and as Tavily's backup | Tavily alone, or Wikipedia | https://app.telem.ai |
| Anthropic | Claude writes the answer from the sources | Qwen on your Mac writes it | https://console.anthropic.com/settings/keys |

Usage on these services is billed to your own account.

### Privacy
- Keys are stored only in your Mac's Keychain and sent only to their own service.
- Your questions, memory, search cache and history stay on your Mac (~/Library/Application Support/Onset).
- Audio leaves your Mac only if you add a Deepgram key (to Deepgram). Search queries go to the search service in
  use (Wikipedia, Wikidata, Tavily or Telem); with an Anthropic key, the question and sources go to Anthropic.
- The first-launch model download comes from Whissle's Dropbox; the update check reads this GitHub releases page.

### System requirements
- A Mac with Apple Silicon (M1 or later), macOS 13 Ventura or later.
- Disk: 270 MB download, plus 1.16 GB of models fetched once on first launch.
- Memory: 8 GB or more.

### What's inside, and licences
Onset, Arc Mini, libarc and Arc Tagger (Whissle, CC BY-NC 4.0 plus Whissle terms: **free for non-commercial use**;
any commercial or business use needs a licence from Whissle, https://whissle.ai). Moonshine Streaming tiny inside
Arc Mini (MIT). Arc's question language model (CC BY-SA 4.0, built from public question datasets; sources in its
NOTICE). EmbeddingGemma 2 (Google, Apache 2.0; Gemma Prohibited Use Policy applies), converted to 8-bit ONNX by
Whissle. Qwen2.5-0.5B-Instruct (Qwen, Apache 2.0), converted to 4-bit ONNX by Whissle. About 11,000 Wikipedia lead
paragraphs (CC BY-SA 4.0). Replay demo questions in synthetic speech (Kokoro-82M, Apache 2.0). Python 3.11 and its
packages. The full texts are in Onset.app/Contents/Resources/Licenses (THIRD_PARTY.md is the index).

### Known limits
- English only.
- Not in this build: search over local recordings ("local moments").
- On the local ear, voice enrollment (telling your voice from others) is off.
