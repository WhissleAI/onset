# libarc (vendored)

Source: https://github.com/WhissleAI/arc_transcriber `c/include` and `c/src` at commit
`955218abe66eeabec011363cdf6ee5afb9b49823` (2026-11-22), libarc 1.4.1, ABI 2. `src/arc_nft2_ort.c` (the optional
ONNX Runtime backend) is left out; `src/arc_nft2_none.c` stands in for it. LICENSE and NOTICE are the repository's.

Built by `scripts/fetch_models.py build-arc` into `vendor/arc/libarc.dylib` (git-ignored); loaded by `server/arc_c.py`.
Weights: `MODELS_DIR/arc-mini` = `WhissleAI/arc-v1.1` `mini/` at v1.2.0 (`models/manifest.json`).

One change, marked "Onset patch" in the code (`include/arc.h`, `src/arc_stream.c`): a new function
`arc_stream_set_words(stream, 1)` makes FULL-mode PARTIAL and FINAL records carry word times. Finals align their text as
`arc_options.words` does (`arc_decode_segment(..., words = 1)`); partials align their greedy text over the streamed
encoder states they were decoded from (`arc_align_words`), without inline tags. Upstream streams give word times only
in EXTERNAL mode. A new function only, so the ABI stays 2.
