# Onset: models, data and third-party licences

Everything below ships inside Onset.app. The licence texts are in this folder (Contents/Resources/Licenses).

| Component | What it does in Onset | Licence | File here |
|---|---|---|---|
| Arc v1 (WhissleAI) | on-device ear: words + audio metadata | CC BY-NC 4.0; commercial use needs a licence from WhissleAI | ArcV1-LICENSE.txt, ArcV1-NOTICE.txt |
| Moonshine Streaming tiny (Useful Sensors / Moonshine AI), modified by WhissleAI inside Arc v1 | Arc's encoder + built-in transcript | MIT | Moonshine-LICENSE.txt |
| Arc Tagger model + libarctag (Whissle) | on-device tagging, safety | CC BY-NC 4.0 + Whissle terms | ArcTagger-LICENSE.txt, ArcTagger-NOTICE.txt |
| cJSON (Dave Gamble and contributors), inside libarctag | JSON parsing | MIT | ArcTagger-NOTICE.txt |
| EmbeddingGemma 2 (Google), `google/embeddinggemma-2` | meaning search, rerank | Apache 2.0 (model card) | Apache-2.0.txt |
| Wikipedia lead paragraphs (about 11,000 articles) | the free local knowledge tier | CC BY-SA 4.0, © Wikipedia contributors; each answer links its article | (text: https://creativecommons.org/licenses/by-sa/4.0/) |
| Replay questions (synthetic speech, Kokoro-82M by hexgrad) | Replay menu demo questions | Apache 2.0 (model); the clips are synthetic, no real person's voice | Apache-2.0.txt |
| Python 3.11 (python-build-standalone) | runs the local server | PSF License | Python-LICENSE.txt |
| Python packages (torch, transformers, sentence-transformers, numpy, scipy, fastapi, uvicorn, httpx, anthropic, ...) | the local server | each package's own licence (BSD / MIT / Apache 2.0) | Contents/Resources/python/lib/python3.11/site-packages/*.dist-info |

Not included: the local-moments index (built from third-party recordings that are not ours to redistribute) and
replay questions voiced with macOS system voices.

Cloud services are used only with your own keys: Deepgram (ear), Tavily and Telem (web search), Anthropic (answers).
Their terms are between you and each provider.
