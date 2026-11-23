# Onset: models, data and third-party licences

Everything below ships inside Onset.app. The licence texts are in this folder (Contents/Resources/Licenses).

| Component | What it does in Onset | Licence | File here |
|---|---|---|---|
| Arc v1 (WhissleAI) | on-device ear: words + audio metadata | CC BY-NC 4.0: free for non-commercial use; commercial use needs a licence from WhissleAI | ArcV1-LICENSE.txt, ArcV1-NOTICE.txt (also covers the MIT jellyfish Metaphone fallback) |
| Moonshine Streaming tiny (Useful Sensors / Moonshine AI), modified by WhissleAI inside Arc v1 | Arc's encoder + built-in transcript | MIT | Moonshine-LICENSE.txt, Moonshine-ORIGIN.md |
| Arc Tagger model + libarctag (Whissle) | on-device tagging, safety | CC BY-NC 4.0 + Whissle terms: free for non-commercial use | ArcTaggerModel-LICENSE.txt, ArcTaggerModel-NOTICE.txt, ArcTagger-LICENSE.txt, ArcTagger-NOTICE.txt |
| cJSON (Dave Gamble and contributors), inside libarctag | JSON parsing | MIT | cJSON-LICENSE.txt |
| EmbeddingGemma 2 (Google), `google/embeddinggemma-2` | meaning search, rerank | Apache 2.0 (model card); use must follow the Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy) | Apache-2.0.txt, EmbeddingGemma-NOTICE.txt (EmbeddingGemma2-ONNX-* in a native-runtime build) |
| Qwen2.5-0.5B-Instruct (Qwen team, Alibaba Cloud), ONNX export `onnx-community/Qwen2.5-0.5B-Instruct` rev 22942cb, modified by Whissle (4-bit embedding/LM head, last-position logits) | writes the short answer from the sources, on this Mac (no key) | Apache 2.0 | Qwen2.5-0.5B-Instruct-LICENSE.txt, Apache-2.0.txt |
| Wikipedia lead paragraphs (about 11,000 articles) | the free local knowledge tier | CC BY-SA 4.0, © Wikipedia contributors; each answer links its article | (text: https://creativecommons.org/licenses/by-sa/4.0/) |
| Replay questions (synthetic speech, Kokoro-82M by hexgrad) | Replay menu demo questions | Apache 2.0 (model); the clips are synthetic, no real person's voice | Apache-2.0.txt |
| Python 3.11 (python-build-standalone) | runs the local server | PSF License | Python-LICENSE.txt |
| libsndfile (inside the soundfile package, a separate replaceable dylib: python/lib/python3.11/site-packages/_soundfile_data) | reads audio files | LGPL-2.1 | libsndfile-LGPL-2.1.txt |
| ONNX Runtime (native-runtime builds only) | runs the ONNX models | MIT | onnxruntime-LICENSE, onnxruntime-ThirdPartyNotices.txt |
| Python packages (torch, transformers, sentence-transformers, numpy, scipy, fastapi, uvicorn, httpx, anthropic, ...) | the local server | each package's own licence (BSD / MIT / Apache 2.0) | Contents/Resources/python/lib/python3.11/site-packages/*.dist-info |

Not included: the local-moments index (built from broadcast clips and private recordings: not redistributable), replay
questions voiced with macOS system voices or recorded voices, and Arc's alternative v1e decoder.

Web search without a key uses DuckDuckGo's public results page. Cloud services are used only with your own keys:
Deepgram (ear), Tavily and Telem (web search); a developer can also point answers at Anthropic's API
(ANSWER_BACKEND=claude + ANTHROPIC_API_KEY in a .env). Their terms are between you and each provider.
