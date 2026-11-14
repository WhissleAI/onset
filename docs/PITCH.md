# Onset: pitch

**One line:** Onset searches while you are still talking, so the answer is ready when you stop.

Every number below comes from a file in this repo. Source key:
- **[AB]** `runs/ab_full5.json`, identical to `runs/ab_latest.json` (chart: `runs/ab_latest.png`). This is the final
  real-audio A/B: 58 spoken questions, 13 voices, all branches merged, git `b16efd9`, run 2026-11-12. The question mix
  is 22 factoid, 8 filler, 8 self-correction and 20 local-moments questions.
- **[ROB]** `runs/robustness.md`: room noise (`ab_snr20_post.json`, `ab_snr10_post.json`) and speaking rate
  (`ab_rate_fast.json`, `ab_rate_slow.json`).
- **[CONV]** `runs/conv.json` (with memory) and `runs/conv_nomem.json` (without): 6 two-turn spoken conversations.
- **[ERR]** `runs/error_analysis.md`: root causes of every wrong or ungrounded answer.
- **[IDX]** `index/README.md`: the local moments index, audio retrieval and the second-stage ranker.
- **[PROG]** `PROGRESS.md`: component timings, the policy simulator, the Tavily lane and the Deepgram ear.
- **[DEMO]** `DEMO_CHECKLIST.md`: demo fallbacks.
- **[TAV]** `docs/RETRIEVAL.md` section 10 and `eval/results/tavily_ab_2026-11-14.json`: the A/B on real Tavily with
  today's stack (Flux ear, director, router, Keystone, Parity budget), 100 spoken questions, run 2026-11-14.

---

## Slide 1: The problem
**Voice search waits for silence.**

- Today's assistants work in sequence: wait for the speaker to stop, detect the pause, search, then think. All of
  that waiting happens after the last word.
- In our own baseline (same ear, same search, same answer model) that wait is **1.54 s at p50 and 2.13 s at p90**,
  measured from the end of the last spoken word to the first answer token [AB].
- **0.51 s** of it is spent just deciding the speaker has stopped (baseline endpoint lag, p50 513 ms) [AB].
- Text search became instant years ago. Voice search still has a second and a half of dead air.

## Slide 2: The insight
**Most of the meaning arrives before the speaker finishes.**

- "Who founded the company…": by word 3 the intent is known, and only the entity is missing.
- **Arc Tagger**, a 10M-parameter on-device model, reads the partial sentence in **0.7 ms per word**. It returns
  the intent, typed slots, any self-correction and whether the request is complete [PROG]. Routing costs almost
  nothing, so we can re-route on every word.
- So we treat search the way a CPU treats a branch: **speculate early, cancel when wrong, and only pay latency we
  would have paid anyway.**

## Slide 3: How it works
```
mic → ear: Arc (on-device streaming ASR, self-revising words) · or Deepgram (pluggable)
    → Arc Tagger (intent · slots · corrections · complete?)    ~1 ms/word
    → Speculator: fire / cancel stale / draft / reuse           ← the search intelligence
    → lanes: web (Tavily; Wikipedia when there is no key) · local moments · spoken-audio lane (EmbeddingGemma 2)
    → the answer model streams a 1–2 sentence answer with citations
```
- **Fire:** free lanes (cards, Wikidata facts, the local index) run mid-sentence. Under the default Parity budget
  (`COST_MODE=parity`, `server/cost.py`) the one paid web search goes out when the meaning is final, plus one more
  only if the meaning changes after it. (The older speed budget allowed up to 6 speculative calls plus 1 final.)
- **Cancel stale:** when the ASR revises a word ("acronis" becomes "nvidia") or the speaker corrects themselves
  ("Canada, actually make that Turkey"), the in-flight search is cancelled and shown as cancelled on the timeline.
- **Draft early:** once every lane for the current meaning has returned, the answer is drafted in the background. At
  the endpoint, the draft is reused if the meaning still matches.
- **Semantic endpoint:** answer when the tagger says the request is complete, the speaker has been silent, and the
  words have held through one ASR refresh. The speculator stays warm in case the speaker keeps going.
- **Searching on sound:** EmbeddingGemma 2 (Google, Apache 2.0) puts speech and text in one vector space. The audio
  of the whole question finds the right local moment in the top 4 for 10 of 10 questions, with no ASR [IDX].
- **Local moments ranker:** a second-stage ranker adds text cosine, lexical and tag matching, and audio. On the
  20-question set it raises recall@1 on clean text from **0.80 to 0.90**, and on what Arc actually heard from **0.65
  to 0.80** [IDX]. (On the earlier 10-question set, the text-only baseline was 0.70.)
- **Conversation memory:** "When was *it* founded?" resolves to Nvidia, so the follow-up search is
  "nvidia founded" rather than just "founded" [CONV].

## Slide 4: Evidence (real audio, controlled A/B) [AB]
"Same ear, same search, same reranker, same answer model. The only difference is *when* we start."

| 58 spoken questions, 13 voices | Baseline | **Onset** |
|---|---|---|
| p50 time to answer (from last spoken word) | 1538 ms | **1160 ms** |
| mean | 1622 ms | **1247 ms** |
| p90 | 2126 ms | **1763 ms** |
| sources ready, p50 | 880 ms | **650 ms** |
| answer correct (keyword-scored) | 96.6 % | 96.6 % |
| web calls per question | 0.66 | 2.38 |

- **Paired:** Onset was faster on **44 of 58** questions. Median saving **348 ms**, mean **375 ms**.
  Accuracy is equal: 55 questions were correct in both modes, and each mode got 1 right that the other missed.
  The 38 web questions are 100 % correct in both modes; the 20 local questions are 90 % in both.
- **Where it shines:** questions with fillers ("um, I forget…"), p50 **1094 vs 1854 ms**. Self-corrections,
  **1228 vs 1599 ms**. Factoids, **1173 vs 1562 ms**.
- **How the answers were produced:** 31 reused a draft, 11 were speculative hits and 16 were speculative misses.
- **It holds in a noisy room [ROB]:** with 20 dB babble (other people's voices), Onset was faster on **35 of
  48** questions and saved **440 ms** at the median. Accuracy was the same in both modes (77.1 %). Noise hurts the
  ear, not the speculation.
- **Speaking rate doesn't matter [ROB]:** at 1.21× speed Onset won 16 of 20 and saved 388 ms. At 0.85× it won
  15 of 20 and saved 486 ms. Accuracy was equal in both modes each time.
- **Follow-ups with memory [CONV]:** Onset answered the second turn in **0.82–1.34 s**, the baseline in
  **1.15–3.01 s**.
- **The policy is simulated before we spend real calls [PROG]:** a virtual-clock simulator replays 51 recorded Arc
  traces through the real speculator. It predicted the real p50 to within **3.9 %** (45 ms), and its paired saving
  was 412 ms against 398 ms measured.
- Demo capture (final code, live Race replays): `docs/media/headstart_race.gif`, which runs q01, then q27, then
  conv01, then local_05. Stills:
  - `q01_race_nvidia.jpg`: 1.14 vs 1.58 s.
  - `q27_correction_canada_turkey.jpg`: 1.54 vs 1.80 s.
  - `conv01_memory_chip.jpg`: memory "it → Nvidia", 1.13 vs 1.48 s.
  - `local05_moment_chips.jpg`: 0.70 vs 1.42 s, with ▶ mm:ss moment chips.

  These are single live replays, not the benchmark. Repeated replays varied: across 8 replays, Onset was
  behind twice and roughly tied twice. See the limitations below.

## Slide 5: Why now, and the impact
- **Why now:** three things only recently became true at the same time.
  1. Streaming on-device ASR that revises its partials every ~100 ms. Arc v1 uses 9 ms of compute per 100 ms chunk [PROG].
  2. Language understanding cheap enough to run on every partial: Arc Tagger, 0.7 ms per word on a laptop [PROG].
  3. One embedding space for speech and text (EmbeddingGemma 2), so retrieval can start from the sound itself.
- **Impact:** about a third of a second saved at the median on every spoken query, and up to **2.69 s** on the
  worst baseline case (q38: 960 vs 3650 ms; the baseline request there was a slow, hedged answer request) [AB]. No extra hardware is needed: the tagger and the ASR run on the device, and
  speculation only costs budgeted search calls. The layer sits in front of any search API, Tavily included. It fits
  voice assistants, in-car systems, contact-center agent assist, and live search over meetings and broadcasts (the
  local "moments" lane).
- **The bigger idea:** search intelligence moves *upstream* of the query. The system decides when, what and
  whether to search while the query is still being formed.

---

## Three likely judge questions

**1. "Speculation must be expensive. What does it cost?"**
Measured on real Tavily [TAV], it costs less than plain search: **0.46 vs 0.63 credits per question** over 100 spoken
questions (showcase 0.70 vs 0.73, 58 set 0.22 vs 0.52, current events 1.00 vs 0.92), with answers 486 ms sooner at
the paired median and accuracy equal (80/82 vs 79/82). Mid-sentence only free lanes run; the paid search waits until
the meaning is final (Parity budget). Plain search pays more because its router hedges a slow free Wikidata lookup
with a paid search; Onset waits for the free answer.

The numbers in the table above are from an earlier build ([AB], speed budget: 2.38 vs 0.66 web calls per question).
A session budget and a process cap stop speculation 2 credits early, so the final query always has credit left
[PROG].

The tagger and the ASR run on the device and cost nothing per call. Drafts are capped at 160 output tokens
(`server/answer.py`) and are cancelled as soon as the meaning moves on. In return, the median answer arrives 348 ms
sooner.

**2. "What if the ASR is wrong?"**
That case is the design, not an exception. Arc revises its own words, and every revision either keeps a search or
marks it stale.

On q01 the ASR first heard "founded condo", then "founded company acronis". Both searches were cancelled, and the
draft built on "founded company nvidia" was reused: 715 vs 1390 ms [AB].

When the key word is lost for good, both modes fail together. ASR errors are the largest root cause of wrong
answers: 107 of 225 wrong or ungrounded rows across all runs [ERR]. A pluggable Deepgram ear is ready, so a
stronger ASR can be swapped in [PROG].

**3. "How is this different from a streaming answer model or voice-to-voice?"**
Those stream the *output*. We start the *input side* early: retrieval and grounding. A voice-to-voice model still
waits for the end of the turn before it can search, or it answers from memory without sources. Onset answers
with citations, and its sources are ready at 650 vs 880 ms (p50) [AB]. It sits *in front of* any answer model or
voice stack, so it adds to them rather than replacing them.

**(Bonus) "What's defensible?"**
- The speculator policy (when to fire, cancel, draft, reuse and endpoint), tuned on real audio with an A/B harness
  and a simulator that predicts the outcome to within 4 %.
- A 10M-parameter tagger that gives intent, slots, corrections and completeness at about 1 ms per word on the
  device. An answer-model router takes about 500 ms (README) and cannot do this in time.
- A spoken-audio retrieval lane over our own moments index.

---

## Demo safety net [DEMO]
- `./run_demo.sh` runs a preflight and prints GO, GO (DEGRADED) or NO-GO.
- **If the venue Wi-Fi fails:**
  - The web lane serves a recorded cache: 57 recorded Wikipedia responses in `runs/web_cache.json`.
  - If the answer model is unreachable, the answer is extracted from the top source instead.
  - Both fallbacks are labelled in the UI (a one-line notice such as "Live web unavailable · using saved results", and "cached" / "extractive" badges).
    Nothing cached is ever shown as live.
- Local-moments questions need no web at all.

## Honest limitations (say these before the judges do)
- **Most numbers above ran on the Wikipedia lane.** The [AB] tables and most PROGRESS results use the keyless
  Wikipedia fallback. The headline cost and speed on real Tavily are in [TAV] (one paired run per set, on a shared
  machine at load average 31-41, so absolute times are inflated and only the paired gain should be quoted).
- **Deepgram ear:** it is built and tested against a fake websocket server, but has never run against real Deepgram
  [PROG].
- **Onset is slower on 14 of 58 questions [AB].** 12 of the 14 are speculative misses (the other two are a
  reused draft, q06, and a speculative hit, q29), and 8 of them are local-moments questions. On local questions the p50 gain is small: 1316 vs 1348 ms.
- **Loud hall (10 dB SNR):** the gain shrinks to 173 ms at the median (32 of 48 faster), because the energy VAD never
  hears silence, so the semantic endpoint never fires [ROB]. A fix is in the code (ASR word-gap silence and a chatter
  guard), but it has **not been re-measured** [PROG]. In noise, Onset also answered neighbours' chatter more
  often (at 20 dB, inside continuous noise: 9 of 48 vs 2 of 48) [ROB]. The chatter guard targets this, also not yet re-measured.
  For the demo we use a close mic.
- **Accuracy is keyword-scored** (`eval/run_ab.py`), with stems and number words allowed. 19 of the 225 wrong rows in
  the error analysis were scoring artefacts, where the answer was right but the keyword check was too strict [ERR].
  The claim is accuracy *parity*, not an accuracy gain.
- **Small, synthetic eval:** 58 questions spoken by TTS voices (Kokoro and macOS `say`), plus 6 two-turn
  conversations. The ASR heard 39 of 58 exactly; mean WER was 5.3 % [AB].
- **Memory follow-ups:** with keyword scoring, Onset got 3 of 5 scored follow-ups and the baseline 2 of 5
  [CONV]. "There" in conv04 resolved to Australia, not Canberra, which is a known resolution gap [ERR].
- **Browser mic path:** a human has not yet tested it. Replays and the server-side mic path are verified [PROG].
- **Licences:**
  - Arc and Arc Tagger are Whissle's on-device models, released under CC BY-NC 4.0, a research licence that does
    not cover commercial use yet.
  - EmbeddingGemma 2 is Google's, under Apache 2.0.
  - The moments index uses third-party YouTube audio, which needs rights clearance before any public showing [IDX].
- **One machine:** every timing comes from a single M3 Pro laptop, with the two modes run in alternating order for
  each question.
- **Live replays vary more than the benchmark.** On the final code (20:20), I ran 8 Race replays:
  - Onset won 4 clearly.
  - 2 were dead heats: q01 at 1.98 vs 1.93 s, and q27 at 1.57 vs 1.63 s.
  - Onset lost 2: q27 at 2.23 vs 1.88 s, and conv01 at 1.49 vs 1.33 s.

  In both losses, the answer model's first token took longer on the Onset side (+1583 ms and +880 ms). In those
  replays the answer started only at Arc's endpoint, because the semantic endpoint did not fire. If a replay ties on
  stage, replay it again, or use local_05, which won by 0.71 s.
