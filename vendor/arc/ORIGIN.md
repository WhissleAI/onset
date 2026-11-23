# libarc (vendored)

Source: https://github.com/WhissleAI/arc_transcriber `c/include` and `c/src` at commit
`a512de7066275a84cbc17f23662e41fe71e699ea` (merge of PR #28, 2026-11-23; n-gram LM fusion: `arc_set_lm`, `src/arc_lm.c`), libarc 1.5.0, ABI 2.
(Previously 955218abe66eeabec011363cdf6ee5afb9b49823, libarc 1.4.1; `src/arc_stream.c` is unchanged upstream between the two.) `src/arc_nft2_ort.c` (the optional
ONNX Runtime backend) is left out; `src/arc_nft2_none.c` stands in for it. LICENSE and NOTICE are the repository's.

Built by `scripts/fetch_models.py build-arc` into `vendor/arc/libarc.dylib` (git-ignored); loaded by `server/arc_c.py`.
Weights: `MODELS_DIR/arc-mini` = `WhissleAI/arc-v1.1` `mini/` at v1.2.0 (`models/manifest.json`).

One change, marked "Onset patch" in the code (`include/arc.h`, `src/arc_stream.c`): a new function
`arc_stream_set_words(stream, 1)` makes FULL-mode PARTIAL and FINAL records carry word times. Finals align their text as
`arc_options.words` does (`arc_decode_segment(..., words = 1)`); partials align their greedy text over the streamed
encoder states they were decoded from (`arc_align_words`), without inline tags. Upstream streams give word times only
in EXTERNAL mode. A new function only, so the ABI stays 2.

Bit 4 of `arc_stream_set_words` (also Onset's): FULL-mode partials decode without the n-gram LM set by `arc_set_lm`
(its weight is zeroed around the partial decode only); finals keep the fusion. Used when the LM changes partials more
than finals (MACAPP.md "Arc n-gram LM").
