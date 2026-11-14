## Onset 0.1.0 for Mac

Onset is voice search that starts searching while you're still asking. Hold Space and speak, or type: Onset hears
the question as it forms (Arc), works out what you mean after a few words (Arc Tagger), searches the right place
(local Wikipedia facts, Wikidata, or the live web) and has the answer ready when you stop, with its sources.

**Download:** `Onset-0.1.0-arm64.dmg` below. Open it, drag Onset to Applications, open Onset. Signed with Whissle
LLC's Developer ID and notarized by Apple. Check the download: `shasum -a 256 Onset-0.1.0-arm64.dmg` should match
`Onset-0.1.0-arm64.dmg.sha256`.

### Keys (all optional)
Onset works with no keys: Arc listens on your Mac, answers come from Wikipedia and Wikidata, and the answer is the
best sentence from the top source. On first launch a setup sheet explains each key and has a **Test** button for it
(one free call). Change keys any time in Settings (⌘,).

| Key | What it adds | Without it | Get one |
|---|---|---|---|
| Deepgram | Deepgram Flux hears you in the cloud: the quickest turn-taking | Arc on your Mac (private, a little slower) | https://console.deepgram.com/signup |
| Tavily | Live web search: fresh, ranked results for news, sports, prices | Telem if you add its key, else Wikipedia | https://app.tavily.com |
| Telem | Web search router across providers; used without Tavily, and as Tavily's backup | Tavily alone, or Wikipedia | https://app.telem.ai |
| Anthropic | Claude writes a short answer from the sources | The best source sentence, with its link | https://console.anthropic.com/settings/keys |

Web search uses Tavily when its key is set, else Telem, else free Wikipedia. Usage on these services is billed to
your own account.

### Privacy
- Keys are stored only in your Mac's Keychain and sent only to their own service. They are never written to disk
  or logs.
- Your questions, memory, search cache and history stay on your Mac (~/Library/Application Support/Onset). Nothing
  is uploaded. Settings > Improve Onset keeps redacted timing rows on your Mac only; turn it off there.
- Audio leaves your Mac only if you add a Deepgram key (to Deepgram). Search queries go to the search service in
  use (Wikipedia, Wikidata, Tavily or Telem); with an Anthropic key, the question and sources go to Anthropic.

### System requirements
- A Mac with Apple Silicon (M1 or later). Intel Macs are not supported.
- macOS 13 Ventura or later.
- Disk: DMG_SIZE download; the app takes APP_SIZE (all models are inside: nothing is downloaded at first run).
- Memory: 8 GB or more (the local server measured 1.3 GB resident after a few questions).
- Start-up: about 20 s while the models load (measured 17-21 s on an M3 Pro); the very first launch can take longer while macOS checks the app.
- Microphone permission (asked once, the first time you hold Space or click Ask).

### What's inside, and licences
Arc v1 and Arc Tagger (Whissle, CC BY-NC 4.0 plus Whissle terms: **free for non-commercial use**; any commercial or
business use needs a licence from Whissle, https://whissle.ai), Moonshine Streaming tiny inside Arc (MIT), EmbeddingGemma 2 (Google, Apache 2.0; use follows the Gemma Prohibited Use Policy), libsndfile (LGPL-2.1, a separate
replaceable library), about 11,000 Wikipedia
lead paragraphs (CC BY-SA 4.0), Replay demo questions in synthetic speech (Kokoro-82M, Apache 2.0), Python 3.11
(PSF) and its packages. The full texts are in Onset.app/Contents/Resources/Licenses (THIRD_PARTY.md is the index).

Not in this build: search over local recordings ("local moments"); that index is built from recordings we can't
redistribute, so the feature is simply absent.

### Known limits
- English only.
- The answer model needs an Anthropic key; without one, answers are extractive (labelled "extractive · not generated").
