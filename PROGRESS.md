# Head Start: progress log

Speculative voice search: retrieval and answer drafting start from the third spoken word, so the answer is ready
when the speaker stops. Built for SearchHack (Sunnyvale, Telem AI × Tavily). Architecture: README.md.

## Layout
| path | what |
|---|---|
| `server/` | tagger binding (ctypes → vendor/arctag native C), frames, speculator, lanes, answer, session, ear (Arc), app (FastAPI WS) |
| `web/` | browser UI: mic + replay, live transcript, frame card, speculation timeline, A/B race panes, results |
| `eval/` | 42 spoken questions (`questions.jsonl`, `audio/q*.wav`, Kokoro + macOS voices), `run_ab.py` harness |
| `index/` | local moments corpus (35 min real audio, 222 moments, text+audio vectors), local questions, retrieval eval |
| `tests/` | deterministic speculation tests (fake lanes + LLM) + Tavily lane tests against `fake_tavily.py` |

Run: `.venv/bin/python -m uvicorn server.app:app --port 8770` → http://localhost:8770.
Keys in `.env` (git-ignored): `ANTHROPIC_API_KEY` (set), `TAVILY_API_KEY` (missing → Wikipedia web lane).

## Measured (2026-11-12, M3 Pro)
- Arc Tagger native int8: 0.7 ms/word, revision 3 ms; weather_query at word 3.
- Arc v1 live: 9 ms per 100 ms chunk; words 0.1–0.4 s behind speech; own pause endpoint ~470 ms after last word.
- EmbeddingGemma 2: MPS bf16 docs ~30–160 ms; CPU text-only copy (271M) for queries 43 ms, identical vectors.
- Haiku 5.5: default config thinks (can burn max_tokens with no text). thinking=disabled + effort=low → TTFT ~620 ms.
- Local retrieval (10 q): text r@1 0.7 / r@4 0.9; whole spoken-question AUDIO → index r@1 0.7 (text vecs) / r@4 1.0
  (max) — no ASR needed; first 1.5 s of audio useless (r@1 ≤ 0.1).
- Real-audio spot checks (quiet machine): HS 0.70–1.07 s vs BL 1.5–2.4 s on Nvidia/Eiffel/Apollo; local q05 0.75 vs 1.12 s.

## Real-audio A/B (eval/run_ab.py, 48 spoken questions, 13 voices, Arc ear, Wikipedia web lane, Haiku 5.5)
| run | baseline p50 / mean / p90 | Head Start p50 / mean / p90 | acc BL / HS | HS faster | notes |
|---|---|---|---|---|---|
| full1 (17:00) | 1600 / 1677 / 2104 | 1473 / 1663 / 2663 | 85.4 / 81.2 % | 26/48 | semantic endpoint threw away the speculator when the speaker paused → 13 re-answers from cold |
| full2 (17:25) | 1546 / 1586 / 1866 | **1254 / 1216 / 1831** | 83.3 / **85.4 %** | **37/48** | soft endpoint keeps speculator warm; paired median saved **435 ms**; drafts reused 52 %; sources p50 568 vs 926 ms; 2.9 vs 0.8 web calls |
Chart: runs/ab_latest.png. Remaining slow cases: API TTFT outliers (q26 1.3 s) → hedged requests added (both modes).

## Fixed along the way (from tests, traces and agent reports)
- Draft built from stale or local-only results → drafts wait for all lanes of their own query, fresh docs preferred.
- Fire marked done before its docs were embedded → done only after vectors.
- Semantic endpoint on misheard partial → words must hold one ASR refresh or be refreshed after silence began.
- Noise floor learned speech as "silence" → floor clamped ≤ 0.01.
- Same utterance answered twice after semantic endpoint → partials of an answered utterance ignored.
- Every web question also searched local moments → route is web | local only.
- Filler words in queries ("wondering") → stopwords extended. Final arriving while busy → queued.
- Audio lane hogged the GPU queue → only for local-routed questions, ≥1.8 s of speech.

## Conversation memory (17:45) — eval/run_conv.py, 6 two-turn spoken conversations, follow-up uses a pronoun
- Follow-up "it/its/they/there" → the previous question's thing; "he/she" → a person, usually named only in the
  previous ANSWER (tagged with a second Arc Tagger handle). Query, ranking text and answer prompt use the resolved
  question; the previous turn's docs (already embedded) seed the next speculator.
- With memory, HS follow-ups: 3/5 scored correct (runs/conv.json; baseline 2/5; conv06 expects Tokyo, answer named Sony Interactive) and grounded (queries "nvidia founded", "albert einstein born"),
  ready 0.82–1.34 s; baseline 1.15–3.0 s. Without memory the searches are bare ("founded", "museum") and the
  model falls back to unsourced knowledge ("the sources don't say...") at 1.4–1.7 s.
- conv03 failed in both: Arc heard turn 1 as "Coron pride and prejudice" (ASR, not memory).

## Verified end to end (17:40)
- Live server, Race mode, WS replay: q01 HS 1139 vs BL 1755 ms; q27 (correction) 1110 vs 1409.
- Mic path (server side; Float32 frames over /ws, 3 questions in one session): 1021/457/626 vs 1361/1114/1259 ms.
- Browser mic capture itself (AudioWorklet) NOT yet exercised: needs a human to grant mic permission.

## Agent wave 2 (launched 17:40)
- policy-sim (branch policy-sim): Arc trace recorder + virtual-time simulator + parameter sweep (wikipedia/tavily profiles)
- ear-deepgram (branch ear-deepgram): pluggable ear, Deepgram live ear ready for a key, fake-server tests
- tavily-lane (branch tavily-lane): API verification, intent→topic routing, credits, cache, fake server
- robustness (main, eval/ + runs/): noise 20/10 dB, speaking rate, current-events set; owns CPU via runs/.run_ab.lock
- reviewer (read-only): independent correctness review of server/
- pitch (docs/): PITCH.md, JUDGE_QA.md, demo GIF + screenshots

## Pluggable ear (ear-deepgram branch)
- `server/ear.py`: `make_ear()` picks `EAR=arc|deepgram`. DeepgramEar = raw websocket to `/v1/listen` (nova-3,
  interim_results, endpointing 300 ms, utterance_end_ms 1000, smart_format, punctuate), KeepAlive every 3 s, Finalize on
  flush, CloseStream on close; fallback to Arc on no key / connect failure / drop (re-feeds audio since the last final).
- `tests/test_ear_deepgram.py`: fake Deepgram websocket server (documented message shapes) -> DeepgramEar -> two race
  Sessions; revisions, finals, timestamps, identical words, keepalive, Finalize, fallback. Not yet run against real Deepgram.

## Tavily web lane (tavily-lane branch) — built against the docs + a local fake, no key yet
Docs verified 2026-11-12: docs.tavily.com/documentation/api-reference/endpoint/search, /rate-limits, /api-credits,
/best-practices/best-practices-search.
- Request: `search_depth` speculative `fast` (1 credit, reranked ~500-char chunks, `chunks_per_source` 2), final `fast`
  too (on the critical path; baseline uses the same final depth, same lanes). `advanced` = 2 credits + highest latency:
  never by default. `auto_parameters` false (it can silently raise depth to advanced), `include_usage` true (billed
  credits come back in `usage.credits` and reconcile our estimate), `max_results` 5, query cut to 400 chars.
- Topic by intent: news_query → `news` (+ `time_range` day for today/tonight/breaking, week for latest/yesterday/this
  week); qa_stock/qa_currency → `finance` + `max_age_hours` 1; weather → `general` + `max_age_hours` 6 (docs: time_range
  drops weather/price pages because they keep old dates). 434 (flex capacity, only with max_age_hours) → retried once
  without it.
- Credits: per fire (meter), per question in the metric (`credits`, `credits_spec`, `credits_final`, `web_cached`,
  `web_backend`, `session_credits`), per mode in run_ab summaries (`mean_credits*`, `total_credits`). Session budget
  120 (speculation stops 2 credits early so the final query always has credit), process cap 800 (free plan = 1000/mo).
  Out of budget / breaker → final query falls back to Wikipedia, speculative fire skipped.
- Errors: 429 (Retry-After) / 432 / 433 / 401 trip a process breaker → Wikipedia until it clears; 5xx/400/timeout → that
  fire is a lane error, a failed final falls back to Wikipedia. Failed calls are refunded in the ledger.
- Cache: TTL (900 s; 120 s for news/finance/weather), key = sorted content words + params, per session by default
  (A/B-honest: baseline never sees Head Start's results; TAVILY_CACHE_SCOPE=global for a demo booth). In-flight
  requests are shared; a stale fire's request (already paid) is not aborted, it lands in the cache.
- HTTP/2 (h2 installed) + 120 s keep-alive on the shared httpx client; hard total deadline TAVILY_TIMEOUT_S=4.
- Fake: tests/fake_tavily.py (MockTransport in-process, or `uvicorn --app-dir tests fake_tavily:app` +
  TAVILY_BASE_URL); lognormal latency per depth (placeholder medians: ultra-fast 320, fast 480, basic 800,
  advanced 1600 ms), paired per query so A/B modes see the same latency.
- Slow-lane sim (fake Tavily 600-1500 ms, 6 questions x 3 seeds, FakeStream TTFT 300 ms): Head Start faster 18/18,
  ~300-340 ms saved (vs ~435 ms real with Wikipedia): the final query needs the last word, so a slow lane can't be
  hidden fully; HS wins by firing it when the word is visible instead of at the endpoint. 2.8 credits/q vs 1.
- include_answer race (TAVILY_ANSWER=1, off): on the after-endpoint path (miss + baseline, both modes for fairness) a
  parallel `include_answer: basic` call races Haiku; first text wins. Wins only if Tavily's answer overhead < our
  embed + Haiku TTFT (~620 ms) — unmeasured; costs +1 credit per miss. Answer has no [n] citations.
- When a key lands: `eval/tavily_probe.py --yes` (~54 credits) → p50 per depth, include_answer overhead, response keys;
  then `eval/run_ab.py` with the Tavily lane.

## Post-fix real A/B (full3, 18:55, all merges + review fixes, Wikipedia lane, 48 spoken questions)
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1478 / 1523 / 1830 ms | **1152 / 1144 / 1662 ms** |
| sources p50 | 880 ms | **644 ms** |
| accuracy | 91.7 % | 91.7 % |
Paired: median saved **398 ms**, mean 379 ms, faster on **37/48**; drafts reused 52 %; 0 answered twice; endpoint
semantic 16 / ear 32 (stricter gate). runs/ab_full3.json.

## HEADLINE ON TAVILY (21:55; real key; 70 spoken questions = 58 scored + 12 current events) — runs/ab_tavily.json = runs/ab_latest.*
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1726 / 1780 / 2399 ms | **1320 / 1357 / 2036 ms** |
| sources p50 | 926 ms | **703 ms** |
| accuracy (58 scored) | 93.1 % | 93.1 % |
| Tavily credits / question | 0.69 | 2.43 |
Paired: median saved 288 ms, faster on 51/70. By kind (paired median saved, faster): current events 744 ms 10/12
(HS 1074 vs BL 2178 p50), filler 618 ms 7/8, correction 556 ms 5/8, factoid 372 ms 20/22, local −110 ms 9/20 (local
regressed with the Tavily-tuned knobs — DRAFT_BUDGET 2 / SEM_STABLE 550 — not yet investigated).
Probe (eval/tavily_probe.py): fast p50 210 ms / p90 514, 1 credit; include_answer +760 ms → kept off. Tavily caches
repeat queries server-side (~0.1 s): the harness alternates mode order, which balances it on average.

## FINAL HEADLINE (Wikipedia lane): full5 (20:00; all 8 agent branches merged; 108 tests; 58 spoken questions = 38 web (22 factoid, 8 filler, 8 correction) + 20 local) — runs/ab_latest.*
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1538 / 1622 / 2126 ms | **1160 / 1247 / 1763 ms** |
| sources p50 | 880 ms | **650 ms** |
| accuracy | 96.6 % | 96.6 % (web 100/100 %, local 90/90 %) |
HS slower on 14/58 (8 local: local p50 1316 vs 1348 — little gain on that lane).
Paired: median saved **348 ms**, mean 375 ms, faster on **44/58**; fillers 1094 vs 1854 ms; corrections 1228 vs 1599 ms.
run_demo.sh preflight: GO (q01 1189 vs 1646 ms; local_05 686 vs 1351 ms).

## Earlier headline: full4 (19:40; review fixes + answer quality; Wikipedia lane; 48 spoken questions) — runs/ab_latest.*
| | baseline | Head Start |
|---|---|---|
| p50 / mean / p90 ready | 1558 / 1561 / 1806 ms | **1176 / 1171 / 1582 ms** |
| sources p50 | 918 ms | **626 ms** |
| accuracy | 93.8 % | **97.9 %** |
Paired: median saved **381 ms**, mean 390 ms, faster on **38/48**. 83 tests pass.

## Robustness (post-fix code; runs/robustness.md)
| condition | BL p50 | HS p50 | acc BL/HS | Arc WER | HS faster (median saved) |
|---|---|---|---|---|---|
| clean | 1478 | 1152 | 93.8/93.8 % | 4.1 % | 37/48 (398 ms) |
| 20 dB babble | 1546 | 1174 | 77.1/77.1 % | 15 % | 35/48 (440 ms) |
| 10 dB babble | 1567 | 1505 | 54.2/60.4 % | 41 % | 32/48 (173 ms) |
| speech 1.21× (20 q) | 1440 | 1106 | 85/85 % | 5.5 % | 16/20 (388 ms) |
| speech 0.85× (20 q) | 1559 | 1148 | 95/95 % | 5.8 % | 15/20 (486 ms) |
- 10 dB loses the semantic endpoint (energy VAD never hears silence) → fixed 19:30: ASR word-gap silence when the room
  is noisy; chatter guard (no early answer for utterances 12 dB quieter than the last question). Not yet re-measured.
- 10 dB re-run after the fix (runs/ab_snr10_fix.json): HS 1437 vs BL 1618 ms p50, saved 212 ms (was 173), faster
  32/48; accuracy 72.9 / 77.1 %. Semantic endpoint still 2/48: the noise is BABBLE (other voices), Arc transcribes
  stray words from it, so the last-word time keeps advancing and no pause is seen. Speech-vs-speech separation at
  10 dB is out of scope; the demo uses a close mic.
- Demo advice: close mic (headset/phone) keeps the room ≥ 20 dB; push-to-talk avoids neighbours' chatter.

## Answer quality (merged 19:20; runs/error_analysis.md)
- 225 wrong/ungrounded rows across all runs: ASR 107, query construction 48, answer model 30, retrieval 21, scoring 19.
- Fixes: fillers/contractions, mangled correction cues, entity phrase queries + Wikipedia entity lane, zero-hit retry,
  no-disclaimer prompt. Offline replay: previously-wrong 9→29/48 right+grounded; regression 20/20.

## Policy simulator (merged; sim/)
- Replays 51 recorded Arc traces into the real Session/Speculator on a virtual clock (~6 ms/question).
- Post-fix validation vs full3: HS p50 err 45 ms (3.9 %), per-question corr 0.82, path agreement 85 %, paired
  saved sim 412 vs real 398 ms.
- Sweep → Wikipedia: current config is within 10 ms of best. Tavily: T2 (DRAFT 2, SEM_SILENCE 200, SEM_STABLE 550,
  SEM_COVER 0.4, HEDGE 1000) → −20 ms p50, ungrounded 4.0→3.3 %, −17 % answer-model calls; now the default when
  WEB_BACKEND=tavily. Aggressive W1-on-Tavily would be −100 ms but 17 % ungrounded → rejected.

## Merged (18:00)
- ear-deepgram → main (21 tests then). tavily-lane → main (42 tests). UI glitch round committed (paired-median tile,
  label collision avoidance, settled pipeline strip). Pitch docs + demo media in docs/.
- Deferred: tavily-lane's eval/run_ab.py credit columns (runs/deferred/run_ab.tavily.py) until the robustness agent
  commits its run_ab.py edits.
- In flight: review-fixes (10 confirmed bugs incl. hedge busy-spin, cancel-before-start hang, session wedge on
  exception, overlapping soft answers, open semantic gate), policy-sim, robustness (snr20/snr10/rate).

## Local moments: second-stage ranker + moment citations (branch local-moments)
Diagnosis of the local rows in ab_full2/ab_full3 (0.6–0.7 accuracy): mostly **scoring**, then retrieval, then ASR.
- Scoring (6 of the misses): right answer, strict `expect` — "nailed" vs `nails`, "fifteen" vs `15`, "clearly the
  winner" vs `won`, "prepping" vs `prep`. Gold amended with alternatives (`nails|nailed`, `15|fifteen`, `winner|won`,
  `prep|prepping|preparation`); numbers below give both.
- Ranking: gold at cosine rank 3–5 (local01 Hunter: four adjacent Hunter moments; local04 Lonzo Ball: the ASR wrote
  "Ponto ball"; local02 objection: rank 3). ASR: local06 "office's wedding" (baseline missed scotus-002 entirely),
  local09 "Debate with Trump" (Arc dropped "prep": gold not retrievable from the words), "Samuel one".
- Answer: right moment, model said "the transcript is garbled"/"sources don't say".
Changes: `server/rerank.py` (text cosine + 0.05·lexical/tags/role/event + 0.2·audio-lane cosine over the whole index,
0.2 ms), used for the final ranking and drafts of local-routed questions in both modes (`RERANK_LOCAL`, `RERANK_AUDIO`);
the audio lane keeps its spoken-question vector for it. Answer prompt: moment lines read
`[1] NBA broadcast at 00:26, play-by-play announcer: …before [moment] after…`, and the system prompt asks for when +
who + a verbatim quote (drafts use the same prompt, so draft reuse keeps it). 10 more questions (local11–20), traces.
Retrieval (index/eval_local.py, 20 q, Arc ASR text): r@1 0.65 → 0.80, r@4 0.90 → 1.00, MRR 0.78 → 0.88 (Head Start);
clean text r@1 0.80 → 0.90 (baseline ranker). E2E replay (index/replay_local.py: real traces + real audio lane in the
simulator, real Haiku, one sample each, 100 calls): gold moment at source #1 HS 13 → 16 / 20, BL 11 → 15 / 20, gold
in sources BL 18 → 20; answers citing a mm:ss HS 4 → 13, BL 3 → 14 / 20 (q11–20 measured with the first prompt wording); accuracy (amended expect) HS 20 → 19, BL 17 → 19 of 20;
strict expect q01–10 HS 8 → 9, BL 7 → 9. Single-sample LLM noise is ±1–2 here: accuracy is at ceiling on the
amended gold; the ranker's win shows in which moment is cited first. Remaining: local09 (ASR drops "prep").

## Open / next
- [ ] Tavily key from user (sponsor); optional Deepgram key (Arc Mini mishears some voices)
- [ ] Human check of the browser mic path
- [x] Correction cues as ASR hears them ("Sari, I mean", "no weight", "I met")
- [x] Demo script (DEMO.md)

## Arc Tagger depth (branch tagger-depth)
The frame now reads the rest of Arc Tagger's per-word read-out (acts, device-directed, typed number/time spans, entity
types, span-level PII, jailbreak) and the UI shows it in a "read-out" row of the frame card.
- **Finance bug fixed (main):** Tavily answers 400 "Finance topic is not supported for fast or ultra-fast" — every
  qa_stock / qa_currency question failed over to Wikipedia. web_plan now sends topic=finance at `TAVILY_FINANCE_DEPTH`
  (basic, same 1 credit). runs/replay_finance.json (5 q): grounded 4/5 → 5/5, S&P/Bitcoin/Nvidia answers dated today;
  cost ~+1.3 s per finance call (speculation hides it on the Arc side; the baseline pays it).
- **Freshness from typed time spans:** a result question (game/race/score/won) said with a relative time ("last night",
  "this past weekend", "right now") → topic news + time_range day|week; money/crypto/market questions → finance;
  absolute years stay general. Arc's weather intent with no weather word (ibuprofen, Japan visa) no longer gets the
  6 h max_age. runs/replay_tagger.json: 11/42 questions change plan; sources published ≤7 days 0/40 → 25/30 (news
  ones); grounded 10/11 → 6/11 (3 finance were the 400 above; 3 news answers now say "sources don't give the score");
  blind judge after 4 / before 7 — mixed, no gold for current events. `TAGGER_HINTS=0` turns it off.
- **Privacy:** phone / email / street address / card / gov id / dob / handle spans never reach the web query; the
  answer model sees "[private]". private_name is NOT used (fires on Nvidia, Mount, Toyota, Mozart); utterance-level
  pii.request / pii.leak fire on most ordinary questions.
- **Jailbreak:** safety.jailbreak.attempt (1.0 on the injection probe, off on all 80 others) → no speculative search.
- **Not used (measured):** device_directed is 0.82–0.92 on chatter vs 0.95–1.0 on questions (no margin); entity
  types are noisy (Mona Lisa / Jaws = person, penicillin = org). Gate: six act/intent-completeness rules in
  eval/gate_eval.py; none dominates the shipped gate (each +complete costs +early), so the gate is unchanged.
- Sim (tavily profile, 58 traces × 5 seeds): identical to main (p50 1598 / p90 2295 ms, paths hit 53 % / miss 26 % /
  draft 21 %, grounded 98.6 %): the eval traces contain no PII / money / injection, and the sim's web lane ignores the
  Tavily plan, so freshness/topic effects are measured only by the replay.

## Search narrowing from the Arc read-out (branch search-narrowing; server/narrow.py, eval/narrow_eval.py)
Offline, real Tavily (fast, 1 credit), 80 questions (30 showcase + 12 current + 38 web), heard text replayed word by
word through the native tagger; the speculator's fire policy replayed on the partial frames. Blind judge (Haiku 5.5,
fixed 0/1/2 rubric, every distinct result graded once, shuffled, no variant labels): runs/narrow_judge.json.
283 Tavily credits, judge calls est. 150-190 (one per question per 25-result chunk, 3 passes; not counted), 400 answer calls. runs/narrow_eval.json (+ runs/narrow_cache.json: re-analysis
costs 0 credits). Families: Arc intent where reliable (weather / news / stock), else content words — Arc's fine
intents misfire here ("screenshot on a Mac" = cooking_recipe, "ramen places" = general_joke, ibuprofen = weather).

Top-5 relevance, paired on the questions whose final results changed (g@5 = mean grade of the top 5, 0-2):
| variant | n changed | P@5 | g@5 | top-1 graded 2 | nDCG@5 | a "2" in top 5 | Tavily ms (paired p50) |
|---|---|---|---|---|---|---|---|
| include_domains **prefer**, all families | 32 | .76 → .76 | 1.28 → 1.18 | **.41 → .63** | .82 → .90 | .88 → .78 | **−88 ms** (n=74) |
| include_domains **restrict** | 36 | .73 → **.59** | 1.21 → **.91** | .39 → .39 | .81 → .80 | .86 → **.58** | **−219 ms** (n=36) |
| query shaping (generic words out, family noun, explicit dates) | 23 | .67 → **.82** | 1.08 → 1.20 | .35 → .39 | .78 → .80 | .83 → .83 | — |
| shaping + prefer, all families | 36 | .74 → .79 | 1.22 → 1.21 | .39 → .61 | .80 → .89 | .86 → .78 | |
| **shaping + prefer on finance/weather/travel/reviews/local/health/recipe/home (default)** | 33 | .73 → **.80** | 1.19 → **1.26** | .39 → **.58** | .80 → **.86** | .85 → .79 | |
All 80: base P@5 .797 / top-1 .519 / nDCG .830 → default .828 / .595 / .854.
- Domains per family (prefer): helped finance (+0.24 g@5), weather w/ shaping (+0.17), travel, reviews; hurt sports
  (−0.9: ESPN/nba.com pages are schedules/hubs, not the result), convert (−0.5), tech (−0.5), definition (−0.3) →
  those are off by default. Family selection is on this same data (n = 1-6 per family): treat as a prior, not proven.
- restrict: fastest (Tavily's own response_time p50 0.44 → 0.20 s) but drops relevant non-listed sources: rejected.
- Answers (Haiku on each variant's final top 5): keyword-correct 50/53 in every variant, grounded 76/80 base vs
  75/80 default — narrowing changes which page is first, not whether the answer model finds the answer.
- Read-out rerank (entity phrase / content coverage / family domain / freshness on Tavily's order): top-1 +1 of 79,
  nDCG −0.001 → no effect; kept behind NARROW_RERANK=0 (Tavily's order already uses the query; the live final
  ranking is EmbeddingGemma cosine over the pool, not re-measured here).
- Speculative fires (showcase sc01–19, real policy): base 58 fires = 3.05 credits/q, 23 useful (a result in the final
  top 5, or the final query itself), 35 wasted. Gate "≥2 content words" (default) drops the lone-entity-prefix fires
  ("kansas", "manchester", "apple", "$100"): 4–6 fires, 0 useful lost. "≥3" would cut 38 % of speculative credits
  (−1.16/q) but loses 2 useful fires and 1 of 9 early top-1 hits (NARROW_GATE_MIN_CONTENT=3).
- Results while you speak (earliest word whose speculative results already contain the final top-1): base 9/19
  questions, words before the end [1,1,1,2,3,5,5,6,6], p50 at 73 % of the utterance; default 8/19,
  [1,1,2,3,3,5,5,5], 70 %. Narrowing does not make the top-1 appear earlier (partial queries rarely have the family).
- Frame fix (all modes): "what does the word ubiquitous actually mean exactly" — "actually" + a continuation verb was
  treated as a self-correction and dropped "ubiquitous" (query "word mean exactly"); now kept.
- Not done: policy simulator for the −88 ms prefer latency (its Tavily lane is a fixed latency model; the shift is
  small next to the 550 ms + lognormal model) and EmbeddingGemma pool ranking on these results.
Flags (server/config.py): NARROW_DOMAINS off|prefer|restrict, NARROW_DOMAIN_FAMILIES, NARROW_QUERY, NARROW_DATES,
NARROW_GATE / NARROW_GATE_MIN_CONTENT, NARROW_RERANK (+ weights). Re-run: `nice -n 19 .venv/bin/python
eval/narrow_eval.py --variants base,query_domsel --spec base,query_domsel --spec-ids sc0,sc1 --judge --answers base,query_domsel`.

## Deepgram Flux ear + Arc metadata (branch flux-ear; server/ear.py FluxEar, server/arcmeta.py)
Flux API facts, checked against the docs AND the live service (developers.deepgram.com/reference/speech-to-text/listen-flux,
/docs/flux/state, /docs/flux/configuration, /docs/flux/voice-agent-eager-eot): `wss://api.deepgram.com/v2/listen`,
`Authorization: Token`, `model=flux-general-en&encoding=linear16&sample_rate=16000`, `eot_threshold` 0.5–1.0 (0.7),
`eager_eot_threshold` 0.3–0.9 (unset = no eager events; must be <= eot), `eot_timeout_ms` 500–60000 (5000), `keyterm`.
Server sends `Connected`, then `TurnInfo` {event Update|StartOfTurn|EagerEndOfTurn|TurnResumed|EndOfTurn, turn_index,
audio_window_*, transcript, words[{word (punctuated, cased), confidence, start, end}] — the WHOLE turn each time,
end_of_turn_confidence, trigger on EndOfTurn: model|manual|timeout}; Update every ~240 ms of audio. Client: binary
audio (80 ms chunks recommended), `ForceEndTurn`, `CloseStream`, `Configure`; no KeepAlive message (we send silence).
EndOfTurn transcript == the preceding eager transcript unless a TurnResumed came between (docs; observed).

Mapping: Update/StartOfTurn/TurnResumed → on_partial; EagerEndOfTurn → Session.on_eager → soft answer
(FLUX_SOFT=eager, default); EndOfTurn → on_final (confirms the soft answer, or answers the longer question); flush →
ForceEndTurn. Fallback to Arc on connect failure / drop (shared with the nova-3 ear). Defaults = lowest latency:
eager 0.3 (docs min), eot 0.7, eot_timeout 1000 ms.

Real Flux traces (eval/record_flux_traces.py → runs/flux_traces.jsonl, 80 questions = eval set + showcase, real time):
- WER 0.020 (15/742 words) vs Arc Mini builtin 0.123 on the same questions.
- EndOfTurn after the last word: p50 652 / mean 671 / p90 1091 ms (78 model, 5 timeout triggers). Arc's own pause
  endpoint is ~470–600 ms: Flux's FINAL is not faster; its EAGER is.
- 3/80 questions split into two turns by EndOfTurn (trigger=model, conf 0.78–0.80): "weather in London, sorry, I mean
  Tokyo", "Boston sorry / I mean Denver", "which one's better for photos / the iPhone 17 Pro or the Pixel 10 Pro".
  FLUX_EOT=0.85 (docs' "complex pipeline" setting) would likely avoid 2 of 3 at some confirmation latency; not default.

Soft-endpoint gate on eager events (eval/flux_gate_eval.py --arc; trigger = answer start − last word end):
| policy | p50 | mean | p90 | premature soft answers /100 q |
|---|---|---|---|---|
| no soft answer (EndOfTurn) | 652 | 671 | 1091 | 0 |
| **eager (default, FLUX_SOFT=eager)** | **212** | **242** | 599 | 26.2 |
| eager, then wait for Arc command_complete (FLUX_SOFT=eager+arc) | 230 | 280 | 627 | 11.2 |
| eager + tagger gate (FLUX_SOFT=eager+gate) | 354 | 430 | 936 | 8.8 |
| eager + Arc cc event within 0.6 s (no waiting) | 315 | 384 | 815 | 3.8 |
| eager + conf>=0.5 (≈ eager_eot 0.5) | 452 | 490 | 1018 | 8.8 |
| eager + Arc for_me>=0.5 | 234 | 264 | 613 | 25.0 (for-me doesn't help) |
Premature = mostly self-corrections and trailing qualifiers; a premature soft answer costs an answer call and is
superseded (the speculator stays warm), not latency. Arc's command_complete measurably helps (premature 26→11 for
+18 ms p50) → shipped as the opt-in FLUX_SOFT=eager+arc; the tagger gate costs +140 ms p50.

Arc metadata without its decoder (ARC_META=mini|transcriber|off; external-transcript mode on Flux's words,
fallback=None so Arc's builtin decoder never runs; RTF = compute / audio, M3 Pro, real Flux words):
| variant | load | RAM (peak RSS) | CPU per 100 ms chunk | RTF | command_complete after last word |
|---|---|---|---|---|---|
| Arc Mini v1 (1.0.1, already loaded for the Arc ear) | 2.4 s | 0.7 GB (no extra) | 10 ms p50 / 15 max | 0.09–0.10 (0.26 live, contended) | +160–300 ms |
| Arc v1.1 Mini (arc_transcriber 1.2.1) | 3.2 s | 1.0 GB | 10.5 / 14 ms | 0.10 | +160–200 ms; adds audio events |
| Arc Transcriber v1.1 full, CPU, model's 10 s stream chunks | 5 s | 3.3 GB | 0 until flush, flush 0.6–1.0 s | — | only at flush |
| Arc Transcriber, CPU, 1 s chunk / 1 s ctx | 5 s | 3.3 GB | 600–870 ms spikes | 0.39 | +1.3 s |
| Arc Transcriber, MPS, 0.5 s chunk / 1 s ctx | 6 s | 3.1 GB | p95 81 ms, cold 1.1 s | 0.26 | +1.2–1.4 s |
Transcriber (Parakeet-TDT 0.6B tap 12 + adapters) needs arc_transcriber ≥1.2 (local 1.0.1 can't load v1.1 configs;
loaded as `arc_audio_t`). It "runs" in real time on MPS but its metadata lands >1 s after the words → read-out only,
useless for endpointing. Default ARC_META=mini. Read-out: `arc` events + `frame.arc` {for_me, cc_p, cc_events, act,
intent, speaker, role, bio, audio_events, rtf}.

Real A/B EAR=flux vs EAR=arc (eval/run_ab.py, showcase sc01–sc10, WEB_BACKEND=wikipedia because the Tavily key returns
432 "usage limit"; sc06 'current' skipped on Wikipedia → 9 q; times from the VAD speech end so both ears are comparable):
| ear | mode | ready p50 / mean | endpoint after speech end p50 | WER | endpoint |
|---|---|---|---|---|---|
| arc | baseline | 2366 / 2763 | 728 | 0.037 | ear 9 |
| arc | headstart | 2033 / 1878 | 632 | 0.037 | tagger 5, ear 4 |
| flux | baseline | 2174 / 2063 | 688 | 0.000 | ear 9 |
| **flux** | **headstart** | **1197 / 1167** | **244** | **0.000** | eager 9 |
(Same 10 with the 432 Tavily key, i.e. no speculation: flux HS 1640/1830 vs arc HS 2676/2901; endpoint 261 vs 630.)
runs/ab_ear_{arc,flux}{,_wiki}.json.

## Confidence fusion + learned end of turn + search director (branch confidence-fusion; server/confidence.py, server/director.py)
**Fusion layer** (`server/confidence.py`): per-word confidence = logistic over Flux word conf, revision history
(`WordTrack`: changes / updates held), last-word position, the tagger span covering the word (score margin, ver).
Out-of-fold on 2,295 real train traces (director dataset): AUC 0.917 (Flux conf alone) → **0.973**, Brier .0194 → .0150
(2.3 % of words get revised). Per-slot conf = min word conf × type conf (score vs alt); key conf = weakest content word.
Frames carry `wconf`, `key_conf`, `slot_conf`; words events `fconf`. Arc audio names/numbers: only at flush (after the
turn), state always `not_in_list`, span_p .03–.18 → no mid-utterance use (`arc_name_boost` kept for a linked catalog).

**Fused end of turn (FLUX_SOFT=fused, default on this branch)**: logistic on Flux end_of_turn_confidence, pause since the
last word, tagger command_complete at the last word (+ score), gate / act / slot-end / content, Arc cc_p / cc event /
for-me, word count, last-word Flux conf; scored on EVERY Flux update (not only Eager), one threshold on Eager, one on
updates (eval/fit_confidence.py, nested CV; eval/conf_features.py builds the features from real traces).
- 80 real Flux traces (nested CV): eager 212 / 242 / 599 ms (p50 / mean / p90), 26.2 premature /100 q; eager+arc
  237 / 288 / 651, 10.0; **fused 174 / 223 / 591, 8.8**. Ablations: dropping Arc, tagger, last-word conf or pause each hurts.
- Live-path bug found + fixed: the ear forwarded words only when the TEXT changed, so pause / last-word-conf were stale
  (+15 ms p50, +49 ms mean, 17/80 questions off); `on_turn_tick(ev, words)` now refreshes timings + confidences.
- Shipped model: refit on the director dataset's 2,295-trace train split, thresholds 0.725 / 0.725 chosen on val
  (≤ 30 premature /100 q on val): original 80 213 ms / 12.5 (out-of-sample), synthetic test 279 / 29.2 (eager 251 / 54).
- **Real A/B** (EAR=flux, Wikipedia lane, 29 showcase q; runs/ab_cfw_{default,fused}.json): HS p50 1339 → **1239** ms,
  p90 2562 → **2166**, mean 1534 → 1373, accuracy 13/15 → 14/15, paired median −112 ms (fused faster 19/29), web
  calls 4.93 → 4.86 /q; baseline 2087 ms. Tavily A/B of the old default (runs/ab_cf_default.json): BL 1692 / HS 1152 ms
  p50, 1.0 vs 4.5 credits /q, accuracy 15/15 vs 13/15 (sc07: Haiku echoed a flour-weights table from the speculative
  pool; sc15: Flux's own EndOfTurn split the question — both correct under default and fused on Wikipedia).

**Speculation defaults from the Flux-trace simulator** (`sim/flux_traces.py`, `--flux`; new measured Tavily latency
profile `tavily_fast`; 80 traces × 3 seeds; runs/sim_flux_grid.json): SPEC_CONF=1 (hold a fire one update when the key
word's fused conf < 0.6): wasted fires 1.03 → 0.82 /q, fires 3.85 → 3.49, p50 unchanged; FLUX_TICK=1 (speculator ticks on
unchanged Flux updates): draft reuse .03 → .13; DRAFT_ON_LAND=0 (on: +0.9 drafts /q for the same reuse).

**Confidence-weighted queries** (eval/confq_eval.py; 12 Tavily credits, 6 judge calls): quoting confident multi-word
entities changed 10/80 final queries, P@5 .64 → .62, top-1 .60 → .60, nDCG .841 → .842 (no effect); dropping
low-confidence non-entity words changed 0 (no final Flux word is < 0.5 fused conf). Not shipped.

**Arc Transcriber mid-utterance** (eval/transcriber_eval.py, 33 questions > 3.5 s): command_complete AUC for "these words
are the whole question" 0.68 (Transcriber) vs 0.84 (Mini) over all updates, 0.71 vs 0.79 in the decision window; its
read-out is 1.32 s stale (Mini 0.35 s). Not used.

**Search director** (server/director.py runtime, numpy int8 Mamba-3; trained in PromptingNemo-private
recipes/headstart_director): see the section below.

## Search director (opt-in: DIRECTOR=on, FLUX_SOFT=director; server/director.py + server/director/; recipe: PromptingNemo-private branch headstart-director, recipes/headstart_director)
A tiny streaming Mamba-3 (Director v3 family) over the per-update event stream (every Flux TurnInfo + tagger poll +
Arc audio state; 54 step features defined once in server/director.py and imported by the recipe, + hashed tokens of the
newest 8 words). Heads: eot (answer now), fire (a search now = the final search), draft (meaning already final), keep
(word kept in the final query). Runtime: numpy int8, **29 K params, 52 KB, 0.16 ms / update**, numpy-streaming vs
torch-parallel parity exact; any failure → FLUX_SOFT=fused.

**Data (all real Flux traces at 1×; faster-than-real-time changes Flux's turn events: 2× 11/16 vs 1× 14/16 identical
to the original traces)**: ~15 K spoken-search texts written by Claude Code agents (runs/director_queries; plain,
filler, correction, late_flip, pause, long, short, numbers, followup, not_search, multi_part …), voiced with 52 voices
(28 Kokoro EN, 6 accented Kokoro, 18 macOS say), rate .85–1.2, and a realistic acoustic chain (rooms, phone / laptop /
AGC / mp3 channels, babble / café / car / TV / keyboard / wind / pink / hum at 5–30 dB, overlapping talkers, hesitations;
test-only held-out street / music / bluetooth / halls). **13,301 traces streamed (~1,390 Deepgram audio-min ≈ $10.7),
9,775 featurized** (tagger + Arc Mini replayed as the Session sees them) → train 7,905 / val 491 / test 1,000 (split by
query root text), + the original 80 as a second test set. Index: runs/director_trace_index.jsonl.

**Offline (thresholds chosen on val only; bootstrap 95 % CIs)**, s model vs the fused logistic (FLUX_SOFT=fused):
| set | policy | p50 | mean | p90 | premature /100 q | not_search answered |
|---|---|---|---|---|---|---|
| test (1,000) | eager | 291 | 318 | 663 | 60.6 | 100 % |
| | fused logistic | 295 [290–320] | 366 | 771 | 32.3 [28.9–36.3] | 81 % |
| | **director** | **254 [249–272]** | **310** | **695** | **30.5 [26.7–34.4]** | **44 %** |
| original 80 | fused logistic | 213 | | | 12.5 | |
| | **director** | **134–146** | | | 11.8–12.5 | |
Pareto (both tuned on val for the same premature target): test @20 director 314 ms / 18.4 vs logistic 450 / 16.3;
original 80 @20 199 / 5.6 vs 410 / 3.8. Per condition (test, director vs logistic vs eager, p50 / premature):
clean 249/28 · 294/27 · 281/51; noise 306/39 · 374/47 · 284/92; reverb 335/27 · 430/37 · 351/75; held-out
conditions 296/36 · 331/46 · 310/70. Fire head: wasted fires 3.63 → 1.97 /q at the same 90 % useful coverage; draft
head: a final-meaning draft for 94 % of questions (hand rule 19 %); keep head AUC .972 vs .965 (content-word rule; off).
Sizes (val eot loss): s 29 K .263, m 82 K .258, l 306 K overfits → s shipped. Learning curve (s): 1.5 K .288 → 3 K .275
→ 6 K .266 → 7.9 K .263 (flattening). No-token ablation ≈ equal; text-only simulated streams: no gain.

**Real A/Bs (EAR=flux, Wikipedia lane, 29 showcase q, both modes)** — runs/ab_dir_*.json, runs/ab_dirn_*.json:
| audio | policy | HS p50 / p90 / mean | paired vs fused | accuracy | web calls /q | BL p50 |
|---|---|---|---|---|---|---|
| clean | fused | 1195 / 1775 / 1342 | | 12/15 | 4.97 | 1626 |
| clean | director | 1247 / 1839 / 1376 | +8 ms, faster 14/29 | 13/15 | 5.00 | 1699 |
| babble 10 dB + room | fused | 1307 / 2204 / 1439 | | 9/15 | 5.62 | 2337 |
| babble 10 dB + room | director | 1192 / 1988 / 1368 | −7 ms, faster 15/29 | 9/15 | 5.62 | 2374 |
The director endpoints ~95 ms earlier (endpoint lag p50 292 vs 387 ms), but ready time follows the in-flight search +
answer TTFT, so the end-to-end difference is within run noise (baselines alone differ 37–73 ms). **Defaults unchanged:
FLUX_SOFT=fused; the director ships opt-in** (largest offline gains: noisy / reverberant audio, not_search refusals).

## Director v2 (branch director/v2; recipe: PromptingNemo-private headstart-director, recipes/headstart_director, sweep5)
More data, four more input streams, three more heads. **Defaults unchanged**: FLUX_SOFT=fused, and every v2 hook is
OFF (DIRECTOR_STREAMS, DIRECTOR_ARC_EAR, DIRECTOR_ADDRESSED, DIRECTOR_ROUTE, DIRECTOR_PII; tests/test_director_v2.py).
The numbers below come from runs/director_v2/ (copies of the recipe's results/: sweep5/<run>/eval_*.json, eval2.json,
sweep5_eot_loss.json, sweep4_r1/, pilot4_7905/) and runs/ab_v2_*.json.

**Data.**
- The 3,526 streamed but never featurized traces: 3,308 featurized with the v2 featurizer (`eval/conf_features.py
  --v2`); 217 have no Flux EndOfTurn. All are extra voicings of TRAIN texts, so test / val did not change. Train grows
  7,905 -> 10,853 (after the jitter filter).
- New, written by Claude Code agents (texts in the recipe's texts/), voiced (voice_extra.py), streamed through real
  Flux and featurized:
  - 450 open-mic scenes: 6-10 utterances each, from the owner U, other people P / P2 and a TV / radio / podcast chain
    (band-limited, roomy, 6-16 dB under). 40 % addressed. Statements, side questions to people, backchannels,
    thinking aloud, reported speech, TV questions. Includes "i like to eat pancakes". Split by scene: test 86 / val 48.
  - 1,210 PII / entity questions: places, public people, companies, decoy numbers, and real PII (house numbers,
    phones, cards, emails, gov ids, birth years, handles). Gold `hide` spans are aligned to Flux's words (TTS reads
    "415" as "four fifteen", so number runs are collapsed on both sides). Test 271 / val 119.
  - test_v2 / val_v2: 1,100 fresh questions (none in the old set; the 14 old styles), phase-3 acoustics incl. the
    test-only conditions -> 760 / 321 usable. The old test's audio was pruned in round 1, so this is the only test
    that carries the Arc streams.
  - Caveat: the agents built the PII set, the test_v2 questions and part of the scenes' addressed lines from
    templates or pools, so phrasings repeat. The addressed metric is also reported on test utterances whose text
    never occurs in train (below).
- Flux: 454 audio min for the new data (scenes 191, PII 154, test_v2 109) ≈ **$3.50** at $0.0077/min (billing logs
  data/flux_*.jsonl.billing.jsonl). A/B ears are not included (below).

**Inputs** (server/director.py GROUPS; one featurizer for training and serving; a missing stream reads `*_ok` = 0,
and training zeroes the arc / ear streams on 30 % of traces so the model runs without them):
| stream | features | source |
|---|---|---|
| pros (8) | last word duration, phrase-final lengthening, last / max gap, speaking rate, time in turn, pause / median gap | Flux word start / end |
| tag (39) | PII kinds, profanity, typed number kinds, mention kinds, router.plan lane / vertical / route conf / skip_web / plan changed | tagger spans + server/router.py on the live frame |
| arc (30) | speech, silence, overlap, degraded audio, music, laughter, non-English, speaker change, interruption, turn yield, #speakers, other speaker, level vs utterance peak; utterance statement / question / opinion / command / backchannel / answer / device-directed, emotion x5, arousal | Arc Mini heads the assistant domain doesn't ship (server/arcmeta.ArcExtra) |
| ear (9) | Arc ear word count vs Flux, Jaccard, Flux's last word heard by Arc, Arc ahead / behind in time, Arc words after Flux's last, Arc's own endpoint near Flux's last word | a 2nd Arc stream with its own decoder |

**Learning curve** (s model, v1 recipe, same regularisation; val eot BCE; test p50 at a val-matched premature target):
| train traces | val eot | test p50 @10 / @20 premature per 100 q | file |
|---|---|---|---|
| 7,905 | .2497 | 441 / 336 | pilot4_7905/v1in_s0 |
| 10,853 (+ the 3.3 K) | .2478 / .2473 (2 seeds) | 424 / 336, 420 / 334 | sweep4_r1/v1in_s{0,1} |
| 11,957 (+ PII + scenes) | .2476 / .2471 | 389 / 334, 413 / 333 | sweep5/v1in_s{0,1} |
Still flattening: +50 % data buys ~.0025 val BCE and ~20-50 ms at the strict target.

**End of turn, offline** (thresholds on the matching val split only; bootstrap 95 % CIs):
| set | policy | p50 | mean | p90 | premature /100 q | not_search answered |
|---|---|---|---|---|---|---|
| test (1,000) | eager | 291 | 318 | 663 | 60.6 | 100 % |
| | fused logistic | 295 [290-320] | 366 | 771 | 32.3 | 81 % |
| | director v1 inputs (v1in_s0 / s1) | 255 [249-279] / 253 | 312 / 307 | 711 / 706 | 32.4 / 34.7 | 75 / 44 % |
| | **director v2 (v2_s0 / s1)** | 255 [245-279] / 276 | 313 / 330 | 734 / 734 | 33.5 / 30.3 | 44 / 38 % |
| test_v2 (760, all streams) | eager | 293 | 355 | 738 | 57.9 | 100 % |
| | fused logistic | 337 [315-381] | 426 | 868 | 34.4 | 70 % |
| | director v1 inputs | 303 / 293 | 377 / 354 | 785 / 773 | 31.5 / 32.9 | 56 / 51 % |
| | **director v2** | 293 [259-330] / 294 | 379 / 379 | 815 / 845 | 32.2 / 33.9 | 51 / 40 % |
| heldout80 | fused logistic | 213 | 272 | 628 | 12.5 | |
| | director v1 inputs / v2 | 133 / 128 ; 133 / 174 | | | 12.5 ; 12.5 / 8.8 | |
Pareto (director and logistic each tuned on val for the target), test_v2 p50 @10 / @20 premature per 100 q: logistic
632 / 534, director v1 inputs 496 / 411 (s0), director v2 526 / 413 (s0). The director's lead over the fused logistic
holds on the fresh, harder set (most of it in noise / held-out conditions).

Threshold-free (eot BCE over every labelled update, difference vs v1in_s0 with a trace-bootstrap CI;
sweep5_eot_loss.json):
| run | test | test_v2 | heldout80 |
|---|---|---|---|
| v1 inputs, seed 1 | +.0017 [-.0023, .0049] | -.0050 [-.0101, .0003] | +.0049 [-.0028, .0135] |
| v1 inputs + v2 heads | +.0043 [.0009, .0075] | -.0014 | +.0106 [.0033, .0184] |
| v2 inputs, v1 heads | +.0001 | -.0024 | +.0082 [.0005, .0179] |
| v2 (s0 / s1) | +.0031 / +.0013 | +.0011 / -.0029 | +.0137 / +.0098 |
| v2 without arc / ear / pros / tag | +.0033 / +.0027 / +.0044 / +.0023 | +.0070 / +.0049 / -.0038 / -.0053 | +.023 / +.014 / +.011 / +.016 |
- **The new input streams do not improve end of turn.** Every v2 variant is within seed noise (~.005) of the v1 inputs
  on test and test_v2.
- The multi-task heads cost a little: +.004 BCE on test, +.01 on the 80 clean originals. The thresholded latency
  doesn't move beyond the CIs.
- Fire / draft / keep are unchanged: wasted fires 2.0-2.1 /q at ~90 % coverage, keep AUC .972-.976.

**Addressed / should-respond (open mic)** (test_om: 86 held-out scenes, 716 utterances Flux transcribed, 292
addressed). The head is read at the last update whose newest word is in the utterance. Each policy's threshold keeps
>= 95 % of addressed utterances on val. "False responds" = share of the 424 NOT-addressed utterances that would be
answered:
| gate | AUC | addressed answered | false responds [95 % CI] |
|---|---|---|---|
| always answer (open mic before fix/ambient-strip) | | 100 % | 100 % |
| **rule gate (session.open_mic_skip, fix/ambient-strip)** | .58 | 97.9 % | **82.5 %** [78-87] |
| Arc device-directed (arc_for_me) | .55 | 94.5 % | 94.1 % |
| tagger device_directed act | .81 | 90.4 % | 71.5 % |
| **director v2 addressed head (v2_s0 / s1)** | **.90 / .89** | 95.2 / 93.8 % | **39.9 %** [35-45] / 46.0 % |
| director AND rule gate | .90 | 95.9 % | 44.6 % |
| same, utterances never seen in train (527) | .88 | 93.1 % | 41.1 % |
| same, Flux decision points only (656) | | 95.6 % | 37.4 % |
| director, v1 inputs + heads (no new streams) | .84 | 89.7 % | 44.1 % |
- By kind, answered: statements 17 % (rule 77 %), chit-chat 33 % (76 %), answers to a person 35 % (88 %), self-talk
  27 % (87 %), backchannels 52 % (96 %). Still hard: side questions to people 58 % (100 %), thinking aloud 57 %,
  TV / radio questions 67 % (78 %).
- In scene A-001: "i like to eat pancakes" p .14 (heard), "how many tablespoons in a quarter cup" .99,
  "you always burn the first one" .05; the threshold is .24.
- Ablations (AUC): all streams .90; without ear .89, arc .88, pros .88, tag .87; v1 inputs .84. The audio streams
  help this head, unlike end of turn.

**Route** (card / facts / web / local / none, the router's lane on the FINAL words, called from partial words;
director argmax vs router.py's rules on the same partial words, `wait` = no call):
| set | at 50 % / 75 % / 100 % of the words: director | router rules | stays right from (ms before the last word) |
|---|---|---|---|
| test (1,000) | 90.6 [88.7-92.5] / 92.9 / 97.7 | 89.9 [88.0-91.9] / 92.2 / 96.3 | 1709 vs 1754 |
| test_v2 (760) | 80.9 [78.2-83.8] / 86.1 / 90.7 | 80.3 [77.4-83.2] / 85.1 / 90.1 | 1339 vs 1387 |
| test_pii (271) | 92.6 / 95.2 / 98.9 | 91.5 / 94.8 / 98.2 | 3223 vs 3266 |
- Equal to the rules within the CI. The head learns the rules' own call (the router plan is one of its inputs:
  without the tag stream it drops to 71-73 % at 50 %).
- It never beats the rules early by a margin that would start a free lookup sooner. DIRECTOR_ROUTE (hold / release
  paid speculation on a confident call) stays off and is not worth an A/B.

**PII-hide** (test_pii: 271 lines, 3,575 Flux words, 1,000 to hide; the head's threshold maximizes F1 on val_pii):
| policy | precision | recall | F1 [95 % CI] | entity-mention words hidden | words hidden on lines without PII |
|---|---|---|---|---|---|
| rule frame._pii_hide | .951 | .645 | .769 [.70-.82] | 1.8 % | 0.18 % |
| head alone | .857 | .753 | .801 [.76-.84] | 10.1 % | 0.71 % |
| **head + entity guard (shipped in DIRECTOR_PII)** | .889 | .737 | **.806** [.76-.84] | **0.0 %** | 0.71 % |
- Per category (head + guard vs rule):
  - phone .92 vs .76 F1;
  - birth year .67 vs 0 (Flux spells "nineteen seventy nine", which the rule's 4-digit regex never sees);
  - card .86 vs .83; gov id .88 vs .86;
  - email .59 vs .77 (worse: misheard names); handle .33 vs .30.
- Places, public people and companies on non-PII lines: 0 % / 0 % / 0.4 % hidden.
- Without the guard the head also hides city / company names that sit next to a number ("oakland california",
  "starbucks"). With it, a word inside a tagged entity mention without digits is never hidden.
- **Matches or beats the rule on F1 with zero over-hiding of entity names**, but the CIs overlap. DIRECTOR_PII stays OFF.

**Runtime.**
- v2 s model: 32,449 params, **57 KB** int8 (server/director/director.npz). numpy-streaming vs torch-parallel parity is
  exact. int8 vs float max |dp| .010.
- Per Flux update, step features + v2 streams + router.plan + model: **0.35 ms p50 / 0.43 p90** (model alone 0.17).
- Live check (run_ab replay, warm Arc): every update carries the Arc extra state and the Arc ear words; the director
  endpointed (trigger=director).
- DIRECTOR_ARC_EAR runs a second Arc stream with its decoder: ~2x Arc CPU per session. The ear stream adds the least
  (above), so the A/B ran without it.

**End to end** (eval/ab_director_v2.sh: EAR=flux, Wikipedia lane, never Tavily; FLUX_SOFT=director with the v2 model
and DIRECTOR_STREAMS=1 vs FLUX_SOFT=fused; showcase 29 + the 58 set; two reps, order alternated): runs/ab_v2b_*.json, summary runs/ab_v2b_summary.json:
| set (questions x reps) | policy | HS p50 / p90 / mean | endpoint lag p50 | paired director - fused, median [95 % CI] (mean) | director faster | accuracy | BL p50 per run |
|---|---|---|---|---|---|---|---|
| showcase (29 x 2) | fused | 1230 / 2111 / 1285 | 336 | | | 28/30 | 1843, 1708 |
| | director v2 | **1100** / 1823 / 1175 | **261** | **-58 [-120, +16]** (-110 [-244, +6]) | 35/58 | 29/30 | 2041, 1599 |
| 58 set (58 x 2) | fused | 1472 / 1906 / 1494 | 429 | | | 115/116 | 2014, 2020 |
| | director v2 | **1386** / 1891 / 1449 | 460 | **-34 [-102, +13]** (-46 [-129, +50]) | 66/116 | 116/116 | 1930, 1996 |
- Directionally faster on both sets, but both paired CIs include 0. The baseline alone moves 442 ms between showcase
  runs (1599-2041) and 6-84 ms on the 58 set: **within run noise**.
- The first try (runs/ab_v2_showcase_*_r1.json, kept for the record) found a bug that predates v2. FLUX_SOFT=director
  never passed the end-of-turn score to the speculator, so search efficiency's SPEC_EOT_FIRE release never fired:
  - 0.83 vs 1.38 speculative fires /q;
  - sources 792 vs 572 ms;
  - **ready +265 ms mean** (paired median +240 [116, 392]).
  Fixed in 14ce3c7: director mode feeds the speculator the same fused score (tests/test_director_v2.py). The v1
  director A/Bs above in this file ran before search efficiency merged, so they were not affected.
- Why an earlier endpoint barely moves ready time (showcase r1): the director endpoints ~95 ms earlier (235 vs 328),
  but under fused 12/29 answers reuse a speculative draft against 3/29 under the director. Endpointing before the draft
  exists turns a draft reuse into a fresh answer after the endpoint (ready - endpoint 964 vs 691 ms).
- Variant with DIRECTOR_SPEC=1: the director's fire / draft heads gate speculation (runs/ab_v2spec_*.json, 1 rep,
  eval/ab_director_v2_spec.sh):
  - showcase: paired director - fused **+90 [-73, +207]** (12/29 faster).
  - 58 set: -207 [-370, -139], but that fused run was slow throughout. Its baseline is 230 ms slower than the director
    run's (2246 vs 2016), and its p90 is 2566 against 1906 in the v2b reps. Run noise, not an effect.
  - No reliable gain: DIRECTOR_SPEC stays off.
- Flux for the A/B ears: 93 audio min ≈ $0.71.

**Recommendation.**
- **Do not make the director the default.** End to end it is directionally faster than fused (-58 / -34 ms paired
  medians) but within run noise on both sets, as in round 1. Offline it still beats the fused logistic at every
  premature target, including on the fresh test_v2.
- FLUX_SOFT=director should be used only with 14ce3c7: without the fix it is ~240 ms slower than fused.
- **Ship candidate (OFF, opt-in):**
  1. DIRECTOR_ADDRESSED=1 for open mic. The head halves the rule gate's false responds (82.5 % -> 40 %) at 95 %
     recall: statements 77 % -> 17 % answered, "i like to eat pancakes" heard. Needs DIRECTOR=on and
     DIRECTOR_STREAMS=1, since the audio streams carry ~.06 AUC for this head. Next step: a live open-mic check with
     real people / TV, then on by default for open mic only (push-to-talk never uses the gate).
  2. DIRECTOR_PII=1 matches / beats frame._pii_hide on F1 (.81 vs .77) and catches spoken years and phone numbers the
     rule misses, with no over-hiding of names. CIs overlap; keep off until a real-speech PII set confirms it.
- **Don't ship:** DIRECTOR_ROUTE (no early gain over the rules), DIRECTOR_ARC_EAR (CPU for the weakest stream), and v2
  inputs for end of turn (no gain; multi-task costs .004-.01 BCE).
- Data is no longer the bottleneck for end of turn: +50 % data, ~.0025 BCE.

Sizes / cost / disk:
- Model 57 KB int8.
- Recipe data/ 2.1 GB (1.9 GB at the start): ds4 135 MB; audio 79 MB after pruning 4,800+ featurized wavs (≥ 790 MB).
- Flux $3.50 (data) + $0.71 (A/B ears) = $4.21.
- Free disk swung from 5 to 14 GB with system swap during the run.

## Search efficiency: search the subject, not the sentence (branch search/spec-efficiency; agent S)
Problem: Head Start spent ~4.5 paid web calls per question (1.0 for the baseline); most fires repeated the same
subject with one more word. SPEC_EFF=1 (default; SPEC_EFF=0 = the old per-word policy):
- `server/searchcache.py` `qkey(frame)`: canonical key = search topic (Tavily topic / time range; news-intent flicker
  ignored), subject (entity mentions + phrases + carried referent), stemmed content terms (Flux's half-heard last
  word "celt" matches "celtics" by prefix), typed numbers / dates. `relation(new, old)`: same | subsumed | extends
  (same subject, more words, or a date added) | grows (entity phrase got longer: skipped only if the earlier results
  already name it) | None (meaning changed: correction, new entity, changed number/date/topic -> search).
- Fire policy (speculator `_spec_web`): a new meaning is searched once it has held SPEC_SETTLE_MS=350, or at once
  when the fused end-of-turn score >= SPEC_EOT_FIRE=0.3 (Flux; other ears: the tagger gate releases a first search
  that names its subject). Hard SPEC_PAID_MAX=2 paid calls per question incl. the final; the last call waits
  SPEC_LAST_SETTLE_MS=650 (or the end-of-turn score) so a half-heard replacement ("pride" -> "pride and prejudice")
  can't spend it. Skips are events (`spec_skip`) and counted in the metric (`web_skipped`).
- Pool reuse: a partial that extends an earlier search drafts from that search's pool, re-ranked with EmbeddingGemma,
  only if `coverage()` passes (best cosine >= SPEC_COVER_SIM 0.5, every subject token named in the top 5, >= 50 % of
  terms, and for live families a search that asked for the same topic + dates); a coverage failure spends the next
  call on that meaning. At the endpoint: same meaning searched -> wait for it; covered -> `pool-reuse`; else one
  final search (`coverage-miss` / `speculative-miss`) unless the budget is spent (`budget-pool`). Nothing landed
  yet -> wait PENDING_EMPTY_S=3 for in-flight searches rather than answer with no sources (all paths).
- Cross-question search cache (CACHE_SEM=1): exact key, then an embedding nearest neighbour among entries with the
  same topic + subject + numbers (CACHE_NN_SIM 0.92). TTL: live (news/sports/weather/prices) 10 min, general 6 h,
  stable (definitions, conversions, recipes, reference) 3 days. Both modes share it, but only entries fetched before
  the asking question started count (a Head Start search never answers the same question's baseline). Optional
  persistence CACHE_PERSIST=1 -> runs/search_cache.json (docs only, <= CACHE_MAX 400 entries). Hits are labelled:
  meter `semcache` + `as_of`, docs `cached` + `as_of`, metric `search_cached` / `search_cached_as_of`, UI badge
  "cache HH:MM". eval/run_ab.py `--search-cache per-run` (default: cleared before every run) | shared | off.
- Router hook for agent R: `lanes.plan(frame) -> [lane names]` if it exists; web speculation / the final web call
  happen only when the plan includes "web".

Simulator (sim/spec_eff.py; 80 real Flux traces x 3 seeds; accuracy proxy = expected keyword in the answer's
sources; fake build_prompt fixed: it rejected the answer hint, so every simulated answer had been the extractive
fallback):
| profile | policy | p50 | p90 | paid calls/q | wasted fires/q | draft reuse | src keyword ok | lead of 1st useful fire |
|---|---|---|---|---|---|---|---|---|
| wikipedia | old | 1182 | 1578 | 3.52 | 0.86 | .267 | .755 | 782 |
| wikipedia | new | 1221 | 1611 | 1.27 | 0.15 | .312 | .736 | -84 |
| wikipedia | new, SPEC_PAID_MAX=3 | 1226 | 1611 | 1.31 | 0.17 | .308 | .755 | -84 |
| wikipedia | baseline | 1739 | 2331 | 1.00 | - | - | .717 | - |
| tavily latency | old | 1804 | 2173 | 3.52 | 0.93 | .046 | .704 | 865 |
| tavily latency | new | 1857 | 2422 | 1.29 | 0.17 | .071 | .767 | -56 |
| tavily latency | baseline | 2424 | 3025 | 1.00 | - | - | .774 | - |
Search cache: off = per-pass shared = identical (no two of the 80 questions share a key / subject); a cache kept
across repeated passes (a demo booth replaying the set) -> 0.03 paid calls/q, p50 965 ms (that is the "repeated
runs flatter the numbers" effect: the A/B clears it per run). Arc ear (no fused end-of-turn score; old sim traces):
old 1186 ms / 2.24 calls -> new 1314 ms / 1.16 calls (fallback ear: a known regression; SPEC_EFF=0 restores).
The "lead" is now negative: the first useful search goes out after the last word (on the end-of-turn score) but
before the endpoint; the head start comes from beating the endpoint, not from mid-sentence results.

Real A/B (Wikipedia lane, EAR=flux, both modes, search cache cleared per run; commit cccb165; runs/s/ab_*.json,
not committed). Old = SPEC_EFF=0 at the same commit. Paid calls on Wikipedia = calls that would be paid on Tavily.
| set | policy | mode | p50 | p90 | accuracy | paid calls/q (web-routed) | wasted fires/q | draft reuse | p50 lead 1st useful fire |
|---|---|---|---|---|---|---|---|---|---|
| showcase (29) | old | HS | 1272 | 2619 | 13/15 | 4.93 | 1.34 | 3/29 | 1138 |
| showcase (29) | new | HS | 1243 | 1758 | 14/15 | 1.48 | 0.14 | 12/29 | 406 |
| showcase (29) | old / new | BL | 2002 / 1797 | 3314 / 2262 | 13/15 / 13/15 | 1 | - | - | - |
| eval (58: 38 web + 20 local) | old | HS | 1272 | 1923 | 58/58 | 2.19 (3.32) | 1.33 | 26/58 | 480 |
| eval (58) | new | HS | 1475 | 2225 | 55/58 | 0.84 (1.29) | 0.79 | 22/58 | 98 |
| eval (58) | old / new | BL | 1994 / 2060 | 2542 / 2707 | 57/58 / 55/58 | 0.66 (1) | - | - | - |
The eval-new run caught a slow answer-model window: ready - sources (answer-model time) rose ~200-260 ms in EVERY
group of that run, including the baseline and the 20 local questions the web policy never touches (local HS
1129 -> 1404 ms). The search-side number, endpoint -> sources, moved +78 ms on web questions (580 -> 658). Of the 3
new HS misses, q05 and q08 had the right sources (answer-model wording; q08 was also missed by BL in that run); q07 exposed two
bugs fixed after the run: (1) the "nothing landed" wait ignored searches that went stale while on the wire,
(2) a budget spent on ASR-garbled partials whose results were empty left no sources -> now one final search over the
budget, flagged `budget_override: empty-pool` (and an empty speculative hit retries the final query).

## Integration: tiered router + efficient speculation (branch search/integrate; merges search/spec-efficiency then search/router)
Both interfaces hold:
- **R -> S:** speculator `_web` passes `frame=f` to `lanes.web` (the router runs inside it, both modes).
  `_meter_ev` passes the search-cache keys and `ROUTER_KEYS`. `_credits` adds `router_counts` (web_paid / free_calls / tiers).
  `_fire.run` handles `lanes.Skip` (state "skipped", `web_used` refunded) and refunds free fires (meter paid == 0).
- **S -> R:** `lanes.plan(frame, final=False)` returns a `RoutePlan` (lane names, conf, sig). "web" is listed only when the
  router may need the paid lane: card / facts plans with conf >= `ROUTER_SKIP_CONF` list the cheap lane alone, "wait"
  is "web" at the endpoint. Paid web speculation, the final web call and the empty-pool retry run only when the plan
  includes "web".
- **Cheap-only plans:**
  - One free speculative lookup per distinct plan (vertical + args; `ROUTER_SPEC_CHEAP_MAX`=4), marked `no_paid`: an
    empty / weak lookup mid-sentence is a free Skip, never a paid fallback. Later partials of the same plan borrow it
    for their draft.
  - At the endpoint: a fitting card in the pool answers (`card-hit`), the same fact lookup is reused (`route-hit`),
    else one cheap lookup (`route-card` / `route-facts`), which may fall back to the web.
  - The fitting card leads the sources; partial cards are dropped.
- **`_paid()`** ignores skipped fires, router-free fires and `no_paid` lookups (also while in flight).
- **Search cache:**
  - Consulted only for web-only plans.
  - Stores paid web docs only: card / facts docs (meta tier / card / snapshot) are never cached, so they can't come
    back as plain docs.
  - `_brief` never marks a card / fact "cached"; the UI badge requires `d.cached && !d.card && !d.tier`.
- **Hold (R's recommendation):**
  - No paid speculative web fire while the plan conf < `ROUTER_SPEC_HOLD_CONF`=0.3, or while the plan is the generic
    wh- "leads" plan (`ROUTER_SPEC_HOLD_LEADS`=1; it was the no-signal default: "what's a hundred US dollars" became a
    currency card two words later).
  - The hold applies while the turn is still going (end-of-turn score < `SPEC_EOT_FIRE`).
  - Corrections are never held (q27 / sc27 answer the replacement).
- **Facts hedge (`ROUTER_FACTS_HEDGE_MS`=350):**
  - A Wikidata lookup still out after 350 ms races the paid web lane; the first good result wins, and the hedge
    counts as paid.
  - At the endpoint, a speculative fact lookup still in flight 350 ms after it was sent triggers one final call that
    races fact vs web.
- **UI:**
  - Lane chip + card under the answer.
  - "paid N · free M" replaces "web N" when routed. The router's "card answer" flag and paid/free stat were pushed
    after the stats rendered, so they never showed; fixed.
  - Timeline rows and the pipeline lane count name card / facts tiers (classes t-card / t-facts; `.card` is the panel
    class).
  - Details view: the card shrinks to title + source (it pushed every source out of the fixed-height list).
  - Screenshots: sc01 / sc08 / q01 / sc17, minimal + details at 1280x720.
- **Tests:** 238 (main 180 + S + R + tests/test_integrate.py 22 + 1 router).

A/B, Wikipedia lane, EAR=flux, search cache cleared per run, both modes, main = 720f297.
"paid" = web calls that reach the paid backend (Tavily credits on Tavily). "free" = card API / Wikidata / Wiktionary /
local-lead calls. "wasted" = speculative fires that fed nothing. "answered twice" = more than one answer for the
question: sc22 splits into two turns on both branches.

Showcase (29 on Wikipedia), commit a4bab66 (run 1) and 4d18a10 (run 2):

| run | mode | p50 | p90 | accuracy | paid/q | free/q | wasted fires/q | answered twice |
|---|---|---|---|---|---|---|---|---|
| main | HS | 1143 | 1749 | 13/15 | 4.97 | 0 | 1.34 | 1 |
| main | BL | 1731 | 2172 | 13/15 | 1.00 | 0 | - | 1 |
| integrated (1) | HS | 1033 | 1584 | 14/15 | 0.90 | 0.62 | 0.28 | 1 |
| integrated (1) | BL | 1578 | 2161 | 13/15 | 0.72 | 0.24 | - | 1 |
| integrated (2) | HS | 1172 | 1772 | 14/15 | 0.97 | 0.48 | 0.28 | 1 |
| integrated (2) | BL | 1671 | 2208 | 14/15 | 0.72 | 0.24 | - | 1 |

Per lane (integrated plan at the endpoint; run 2):

| lane | n | HS p50 main -> int | HS paid/q | BL p50 main -> int | BL paid/q |
|---|---|---|---|---|---|
| card | 5 | 1199 -> 453 | 5 -> 0.4 | 1641 -> 751 | 1 -> 0 |
| facts | 4 | 1274 -> 1302 | 4.25 -> 0.25 | 1740 -> 1728 | 1 -> 0.25 |
| web | 20 | 1136 -> 1197 | 5.1 -> 1.25 | 1754 -> 1705 | 1 -> 1 |

In run 1 every card question cost 0 paid calls. In run 2 two of them paid 1:
- sc27: a free lookup still in flight was miscounted; fixed in 51c4b33.
- sc28: a correction partial, exempt from the hold.

58 set (38 web-set + 20 local), commit 4d18a10 (before the facts hedge):

| run | mode | p50 | p90 | accuracy | paid/q | free/q | wasted fires/q | answered twice |
|---|---|---|---|---|---|---|---|---|
| main | HS | 1196 | 1970 | 57/58 | 2.09 | 0 | 1.31 | 0 |
| main | BL | 1826 | 2424 | 55/58 | 0.62 | 0 | - | 0 |
| integrated | HS | 1364 | 2108 | 58/58 | 0.14 | 2.03 | 0.91 | 0 |
| integrated | BL | 1955 | 2766 | 58/58 | 0.09 | 0.86 | - | 0 |

Per lane, 58 set:

| lane | n | HS p50 / p90 main -> int | HS paid/q | BL p50 main -> int |
|---|---|---|---|---|
| facts | 35 | 1208 / 2282 -> 1423 / 2075 | 3.14 -> 0.14 | 1853 -> 2017 |
| web | 23 | 1149 / 1628 -> 1137 / 2025 | 0.48 -> 0.13 | 1790 -> 1728 |

Local questions were unchanged (HS 1182 -> 1175).

The 58-set latency regression comes from the facts lane, in both modes. On the 38 web-set questions, sources landed
+126 ms later (median) while answer-model time was flat. Speculative fact lookups took p50 236 / p75 440 / p90 706 ms,
against 205 / 235 / 263 for main's per-word Wikipedia searches, and the endpoint waited up to PENDING_CAP_S for them.
51c4b33 adds the facts hedge (above). Its A/B and the single Tavily showcase run (estimated ~1.7 paid calls/q x 30 ≈ 51
credits; hard cap `TAVILY_PROCESS_CREDITS=60`) are still to run: the demo held runs/.run_ab.lock past this session.

## Cost-and-speed meter (branch feat/meter)
On-screen meter for "faster, cheaper retrieval, fewer wasted tokens". Always on (display only: no extra calls, no effect
on the answer path).
- **Token capture (server/answer.py):** every answer-model request records its usage, read live from the SDK stream's
  message snapshot (`message_start` input, `message_delta` final output). `Stream.usage()` gives in / out over all
  requests plus the winner's share.
  - A request stopped before the API's final usage arrived: output = streamed chars / 4, flagged `est`.
  - A request cancelled before its first event: input = prompt chars / 4, counted (it may be billed), flagged `est`.
  - A connection failure before the API accepted the request counts 0.
  - Card answers make no request: 0 tokens.
- **Waste (server/meter.py + speculator):** the speculator keeps every stream of the question (drafts, soft and final
  answers). The stream that answered is `ans_*`. Everything else is `waste_*`: drafts thrown away, a superseded soft
  answer, the hedge loser of the answering stream.
  - At `close()` the speculator sends one `usage` event with the final tally (drafts still streaming when the metric
    went out are included) and updates the metric dict in place, so `session.history` / eval rows carry it.
  - A question replaced by a correction (no metric) still reports its spend (`answered: false`). The client adds it to
    the session totals.
- **Metric fields:** `ans_in ans_out waste_in waste_out llm_requests waste_streams tokens_est usd usd_web usd_llm
  usd_llm_waste usd_est price_verified wasted_fires` (+ the existing `paid_calls free_calls`). `wasted_fires` =
  spec_fires - spec_used (same as run_ab's mean_wasted_fires).
- **Prices (config.py METER_*, checked 2026-11-13, all $ shown as "est."):**
  - Tavily: $0.008 per credit, pay-as-you-go (tavily.com/pricing). A basic / fast search is 1 credit. When the backend
    reports actual credits they are used.
  - claude-haiku-5-5: $0.10 in / $0.50 out per MTok (platform.claude.com pricing page, prompts up to 100K).
  - Any other model has no verified price: tokens are not priced until METER_ANSWER_IN/OUT_USD_MTOK are set, and the
    meter says "configured, not verified".
  - On the Wikipedia lane, "paid" calls cost nothing and are priced as Tavily calls (labelled in the UI).
  - `GET /api/meter` returns the prices.
- **UI (web/meter.js + meter.css; app.js hooks: event, metric, start, results):** a bar pinned to the viewport bottom,
  in both minimal and details views, light + dark.
  - Hero: session average over questions both modes answered, e.g. "0.53 s faster · 2.0× cheaper". It flips to
    "2.0× the cost" / "same cost" (within 15 %) when that is what was measured.
  - This question: HS vs BL paid / free / wasted / tokens (+waste, ~ = estimated) / $ est.
  - Session: questions, total time saved, $ HS vs BL, paid calls and the "N× fewer (more) paid calls" ratio.
  - `details` opens the per-question table, per-mode totals, price notes and a reset.
  - Single mode: "$X per question · Head Start only"; nothing about the other mode is imputed.
  - Below 1100 px the per-question row moves into details; below 700 px it shows only the hero + button.
- **Results view:** "Cost & speed, same terms as the live meter" table from the latest A/B file. It shows p50 ready,
  paid / free / wasted per question, tokens, $ est. per question and per 1,000 questions.
  - Files older than the meter have no tokens: they say "not recorded", and their $ is web only.
  - eval/run_ab.py now copies the meter fields into rows and adds mean_tokens_in/out, mean_waste_in and mean_usd to
    the summary.
  - On the current runs/ab_latest.json (70 q, 2026-11-12, before the router integration): HS 2.57 paid/q vs BL 0.71,
    about 3.4× the est. cost. The meter shows that as it is.
- **Live check:** Race on WEB_BACKEND=wikipedia, port 8793, sc01 + q01 + q27 in one session, light run.
  - HS faster on all 3, avg 0.41-0.53 s.
  - HS made 2 paid calls vs 1 (q01: the facts hedge's paid web call; BL's fact lookup answered free), so the hero read
    "2.0× the cost".
  - sc01 = card, 0 tokens in both modes.
  - q27: HS 1.6k answer input + 1.2-2.4k waste (a draft for the corrected-away "Canada" partial / a hedge duplicate);
    BL 1.2k.
- **Tests:** tests/test_meter.py, 11 tests: exact usage, cancelled draft = estimated waste, hedge loser = waste, card =
  0 tokens, connection failure not billed, $ arithmetic, unverified price, metric fields in both modes, unused drafts
  counted at close, a superseded question still reports.

## Memory v2 + follow-up latency (branch feat/memory; server/memory.py, web/graph.js)
**Why Head Start follow-ups were slow** (event dumps from eval/run_conv.py --events):
- The router planned the words as spoken. "when was it founded" matched no fact pattern, so it got the generic
  "leads" plan, which holds paid speculation (route-hold). Nothing useful fired mid-sentence, and at the endpoint
  58/72 follow-ups went "speculative-miss" (search after the endpoint).
- "which museum is it in" ended on "in". The dangling-end rule held the final for REPAIR_HOLD_S, about 1.5 s of
  endpoint lag in both modes.
- The turn models (director / fused) read pronoun questions as unfinished. This mattered less than the two causes
  above: Flux's EndOfTurn usually came at about the same time as the eager event.

**Fixes** (all default on, each with its own flag):
- `MEM_INLINE`: memory resolves the reference IN PLACE (`Frame.inline`), and `router.repaired_text` plans on it.
  - "when was it founded" -> "when was nvidia founded"
  - "where is its headquarters" -> "where is the headquarters of sony interactive entertainment"
  - "what is the population there" -> "... in canberra" (the knowledge population pattern also accepts "in")
  - "and AMD?" -> "who founded amd"
  - "what about tomorrow?" -> today swapped for tomorrow
  
  Follow-ups now get the free Wikidata / card lanes, and speculation fires on the resolved meaning.
- `MEM_PREWARM`: a free pre-warm after each answer. One SPARQL query fetches every PROPS property for the likely
  next subjects (the answer entity and the question subject) into the knowledge cache. A follow-up's fact lookup
  then needs no network call, and a lookup that arrives while the pre-warm is still in flight waits for it instead
  of sending a second query.
- `MEM_HOLD`: no speculation while a pronoun's referent is still being answered, or while the user is stating a
  fact. The frame is rebuilt and speculated on as soon as that answer lands (`Session._refresh`).
- Pronoun + stranded preposition ("is it in", "is he from ... for") no longer counts as a dangling end. On the 80
  single eval questions, the only frame change is that "Mount Kilimanjaro, which country is it in" is no longer
  held. All other frames and plans are unchanged (scratchpad frames_dump.py).
- `MEM_EOT` (thresholds 0.4 / 0.7): a lower end-of-turn bar for frames that memory resolved in place. It only
  touches follow-ups.

**Memory v2** (`server/memory.py`):
- Typed entities across turns: question entities, the answer's new entities (spans grown over capitalised words,
  question repeats and demonyms dropped), and kind votes (question 2, answer 1, "what the question asked for" 3,
  Wikidata values 3).
- QIDs and property edges come from the facts docs that were used. Also kept: "said" edges (answer -> subject),
  co-mention edges, and gender cues ("her husband" -> she / he).
- References it resolves: it / he / she / there / possessives, "that company / city / person", "the first one",
  "the other one", "going back to the first one". The asked-for answer entity leads for kind-naming questions
  ("which company", "capital of", superlatives), and "where ..." questions keep "it" on the subject.
- User facts ("I live in Denver", "I'm vegetarian / vegan", "I don't eat meat", "allergic to X"):
  - A pure statement is acknowledged locally: path "memory", no search, no model call.
  - A fact fills in what a later question leaves out: no place in a weather / time / sunset / "near me" question ->
    home; a food question -> diet.
  - The frame read-out and the UI say "using: you live in Denver".
  - Facts never leave the machine except as the resolved query word.
- Persistence: the browser sends a random id ("hello"). Memory is stored in `runs/memory/<sha256(id)[:24]>.json`
  (git-ignored) with TTLs `MEM_CONV_TTL_S`=30 min (conversation) and `MEM_USER_TTL_S`=30 days (user facts).
  "forget" clears it on screen, in the session and on disk. A scripted conversation starts fresh but keeps the user's
  facts. Checked end to end in headless Chromium: restored after a reload, file deleted by forget.

**Graph panel** (`web/graph.js`, `web/graph.css`):
- Live SVG knowledge graph with a tiny force layout and no libraries. Colours by kind, Wikidata edges labelled,
  user facts hang off "you".
- The entity used to resolve the current follow-up gets a gold halo.
- Placement: in the left column in the details view; a collapsible dock bottom right in the minimal view, above the
  cost meter.
- The note under the transcript reads "memory: going back to the first one -> Nvidia (turn 1) · asked as 'nvidia
  when was it founded'".

**Eval**: 21 spoken conversations (6 two-turn + 15 three-turn with pronouns, references, ellipsis and user facts;
eval/make_conversations.py, `say` voices). eval/run_conv.py changes:
- Scores every turn and supports repeats.
- Clears the search cache and the router caches per run (A/B-honest).
- Writes event dumps and incremental results.

Follow-up turns, Wikipedia lane, EAR=flux, FLUX_SOFT=director. The machine was shared with other agents' training
and evals, at load average 20-115. Absolute latencies moved 2-3x with load, so the same-run HS vs BL comparison is
the number to read.

| run | load | mode | follow-up p50 | p90 | correct | paid/q | HS paths |
|---|---|---|---|---|---|---|---|
| before (main 53b7fac, 2 reps) | ~30 | BL | 2134 | 3158 | 58/68 | 0.57 | |
| | | HS | 1830 | 2557 | 54/68 | 0.51 | spec-miss 58/72, route-facts 6, spec-hit 4 |
| before3 (main 66c017a, 1 rep) | ~40 | BL | 3440 | 4512 | 22/34 | 0.94 | |
| | | HS | 3768 | 6094 | 27/34 | 1.25 | spec-hit 20, spec-miss 12, route-facts 4 |
| after3 (feat/memory, 1 rep) | ~20 | BL | 1986 | 2322 | 28/34 | 0.58 | |
| | | HS | **1548** | 2324 | **32/34** | 0.61 | spec-hit 23, draft-reuse 7, route-facts 5, spec-miss 1 |

- Within run, the HS/BL follow-up p50 ratio went from 1.10 (before3) and 0.86 (before) to 0.78 (after3). On
  turn 1 the ratios were 1.13 and 0.66: before3 ran loaded enough that HS lost on turn 1 too.
- HS follow-ups are no longer slower than HS first questions: 1548 vs 1664 ms turn-1 p50 in after3.
- The load-independent signals:
  - Follow-ups searched after the endpoint: 12/36 -> 1/36.
  - Correct follow-ups: 27 -> 32/34.
  - Paid calls per follow-up: 1.25 -> 0.61.
- Baseline also improved (22 -> 28/34), because the same memory resolution runs in both modes.
- Remaining misses are ASR: "what other movies did he make" heard as "one other move deremy", and "Marie Curie"
  heard as "mari guri" on turn 1, which then made "she" resolve to Pierre.
- Runs: runs/conv_before*.json, runs/conv_after3.json (+ _events). Rerun both arms on a quiet machine before
  quoting absolute ms.

**Tests**: tests/test_memory.py has 27 tests:
- Multi-turn resolution, kinds, gender, ordinals and "that company"
- Ellipsis and time swap; Wikidata QIDs and edges
- User facts: learned, used, a place said in the question wins, never added to unrelated queries
- Store round trip / TTLs / forget / bad ids
- The statement ack with no search; the hold and refresh; the end-of-turn bar
- Prefetch filling the cache (no network on the follow-up)

The full suite passes: 299.

Screenshots (1280x720, light + dark, minimal + details, no console errors, no horizontal scroll at rest):
scratchpad feature_shots/memory/ (conv07 three-turn "going back to the first one", conv12, conv19 user facts).
## Plan while you speak + verify (branch feat/plan; server/planner.py, server/verify.py, web/plan.js)
**Planner (PLAN=1, default ON).** A multi-part spoken question is split into sub-questions while it is spoken; each
part becomes a synthetic Frame (its own words + the parent's tagged entity spans) and goes through the router on its
own, so one part can be a free Wikidata fact or weather card while its sibling goes to the web.
- Rules only, no model call (decompose() p50 0.02-0.04 ms, p99 < 0.6 ms per frame, measured on every frame of the
  paired runs below): comparisons ("which is taller, Everest or K2", "is X taller than Y", "compare Nvidia and AMD
  revenue", "Python versus Java, which came out first"), conjunctions ("flights to Denver and the weather there" ->
  pronoun resolved to Denver; "population of France and Germany"; "... and also ..."; up to PLAN_MAX_PARTS=3), multi-hop
  ("how old was the founder of Amazon when it was founded" -> who founded amazon | when was {founder} born | when was
  amazon founded). Input is router.repaired_text (corrections spliced in place); Arc Tagger entity phrases guard
  "pride and prejudice"-style splits.
- Speculator hook (on_frame / on_eot -> _plan_step): a CLOSED part (the speaker moved past it: left of "and", the first
  option before "or") fires at once; the open part after SPEC_SETTLE_MS or a likely end of turn. Whole-sentence
  speculation and drafts are off for planned frames. A search of the same query is reused, never paid twice (also a
  whole-sentence search made before the plan formed). Budget: SPEC_PAID_MAX + PLAN_EXTRA_PAID (1) per extra part,
  WEB_BUDGET, the Tavily ledger; cheap-only part plans are marked no_paid mid-sentence.
- Corrections re-plan: "who was born first, Mozart or Beethoven, actually make that Bach" -> the Beethoven part's search
  goes stale, the Bach part fires (test).
- Multi-hop bridge: the hop part's entity comes from its Wikidata fact value, else from role wording in its top results
  ("founded by X", "written by X", "CEO X", "X is the capital"), as soon as its docs land (before their embeddings);
  the bridge part then fires (mid-sentence when possible).
- Endpoint (both modes, _final_plan): every part searched in parallel (speculative ones reused; a speculative lookup
  still out after PLAN_PART_CAP_S=0.5 s is raced by a final call), sources ranked per part by lane order (no query
  embedding on the critical path: ~0.1-0.5 s on this loaded laptop), up to 3 per part (2 for 3 parts); the answer
  prompt gets a part-aware hint ("(1) how tall is everest: [1], [2]; (2) ...; answer each part with its citation, then
  say which is taller"); the card one-liner shortcut is off for plan answers. Metric `plan` (parts, fire, tier, paid).
- Single-part questions: decompose() returns None for every word prefix of the 58 set, the showcase and the local
  set (960 prefix frames, eval/multipart/plan_textcheck.py), so their path is unchanged by construction.
- Independent recall / precision check on the director's ~15 k spoken-search texts written by other agents
  (plan_textcheck.py --corpus): style multi_part split 74/100; false-split rate on the other styles 0.0-1.9 %
  (plain 0.7 %, filler 0.4 %, correction 0.4 %, followup 0.1 %; late_flip 1.9 %, mostly genuine "X, and Y?" questions).

**Verify (VERIFY=1, default ON).** After the last answer token (ready / done already measured), the answer's numbers
(with scale: "60.9 billion" = "60,922 million"), years and names are looked up in the cited sources, then in all
sources: badge "✓ N sources agree" (distinct URLs) / "✓ matches 1 source" / "◐ k of n claims found" /
"⚠ sources disagree: 1993 vs 1991" / "unverified". A disagreement needs a different year right after the claim's own
predicate, in a sentence naming the claim's subject, in a source whose title is that subject, and not another year
the answer states (Harvard 1636 vs Yale 1701 is two facts). Arithmetic on the answer's own years ("30 years old") is
labelled "computed". ~0.5-2 ms, no network, no model. It checks that the sources say it, not that it is true
(tooltip says so). Event `verify`, metric `verify`.

**UI.** Plan chips under "What Arc hears" (visible in the minimal demo view): P1 / P2 / P3 with the sub-question, its
lane (FACTS / CARD / WEB) and status (dashed = waiting, pulsing = searching, green = done, struck = stale after a
correction); timeline rows of part searches are labelled "P1 · query"; verify badge under each pane's answer; a
"Multi" replay tab (eval/multipart; audio copied to eval/audio, git-ignored). Screenshots (1280×720, light + dark,
mid-speech + after, minimal + details; no console errors, no horizontal scroll):
scratchpad/feature_shots/plan/.

**Eval set.** eval/multipart/questions.jsonl: 26 spoken questions (13 comparisons, 7 conjunctions, 6 multi-hop),
keyword answers per part, Kokoro voices (eval/multipart/make_audio.py -> eval/audio_multipart/, git-ignored).
Scorer: eval/multipart/score.py. Paired runner: eval/multipart/ab_toggle.py (each question run with PLAN=1 and PLAN=0
back to back in one process, order alternating): the laptop's load average was 46-62 during this session (other
agents + director training), and separate runs minutes apart moved BOTH modes by seconds, planner or not
(runs/plan/sc_on vs sc_off: baseline p50 4787 vs 3122 ms with no question planned), so only paired numbers are quoted.

Paired A/B, Wikipedia lane, EAR=flux, search cache per run, commit 2d3fb98 (runs/plan/tg_*.json, not committed).
"paid" = calls that would be Tavily credits; "free" = Wikidata / cards / local leads.

| set | mode | planner | p50 ready | p90 | per-part acc | all parts right | paid/q | free/q |
|---|---|---|---|---|---|---|---|---|
| multi-part (26) | Head Start | on | **1889** | **2166** | **47/54 (.870)** | 21/26 | 1.81 | 4.38 |
| multi-part (26) | Head Start | off | 3343 | 5428 | 46/54 (.852) | 20/26 | 0.85 | 1.88 |
| multi-part (26) | Baseline | on | 2294 | 3037 | 43/54 (.796) | 18/26 | 1.73 | 3.31 |
| multi-part (26) | Baseline | off | 2925 | 4354 | 45/54 (.833) | 19/26 | 0.81 | 0.77 |
Head Start planner on vs off, paired: faster on 19/26, p50 1657 ms saved (sources p50 1106 vs 2544 ms: the parts land
in parallel, mostly on free lanes). Head Start vs Baseline with the planner on: p50 1889 vs 2294. Cost: ~+1 paid call
per multi-part question (a part whose Wikidata lookup is slow is hedged to the web) and ~+2.5 free calls.
Single-part regression (Head Start, paired, no question planned in either arm):

| set | p50 on / off | p90 on / off | paired p50 on faster | accuracy on / off | paid/q on / off |
|---|---|---|---|---|---|
| showcase (30) | 1322 / 1288 | 2635 / 2402 | -8 ms (14/30 faster) | 13/15 / 14/15 | 0.87 / 0.93 |
| 58 set | 1290 / 1390 | 1898 / 2106 | +14 ms (31/58) | 58/58 / 58/58 | 0.28 / 0.40 |
The one showcase accuracy flip (sc11) took the draft-reuse path in one arm and pool-reuse in the other with the same
non-answer wording; neither arm planned it. Verdict: no single-part regression -> PLAN defaults ON.

Verify on the paired runs (planner-on arm, before the last two false-alarm fixes): 58 set supported 38 + agree 10,
partial 7, unverified 3; showcase supported 15 + agree 1, unverified 11 (live / how-to answers without checkable
numbers), partial 3. On multi-part answers 9/52 were "disagree", nearly all false alarms where the "other" year was
another fact of the same answer (Harvard 1636 vs Yale 1701, born 1964 vs founded 1994): fixed (own-year, nearest
predicate, same-subject sentence rules; tests), not re-measured end to end.

Known gaps: ASR spellings break sub-questions ("K2" heard "k two" -> no Wikidata hit); the hop's first Wikidata
value is used when there are several founders (Microsoft -> Bill Gates or Paul Allen); "compare X and Y" without an
attribute searches the bare names; per-part keyword accuracy rewards the model's own knowledge as well as sources
(verify shows which). Tests: tests/test_planner.py (20 cases: decomposition, single-part non-splits, Head Start fires parts
mid-sentence, baseline after the endpoint, correction re-plan, bridge, reused whole-sentence search, verify);
tests/test_integrate.py's paid-budget test now allows PLAN_EXTRA_PAID per extra part (its question is two-part).
After the last verify fixes, a re-shot of mp21 shows "✓ 2 sources agree · 30 computed" (was a false "⚠ 1964 vs 1994");
one remaining false alarm seen: mp02 baseline "1756 vs 1979" (another page titled Mozart).

## Keystone: Arc-keyed answer graph + VOI gate (branch feat/index; server/keystone/, eval/keystone/, docs/RETRIEVAL.md)
One structure for the search cache, the local index and memory. Every retrieved passage / fact / answer becomes a memory
cell (the reranker's EmbeddingGemma vector, 256-dim first pass + fp16 rescore); every answered question writes a typed
edge (subject) -[relation]-> (value cell) keyed by the Arc read-out (intent + entity span/type + wh- answer type ->
`founded_by(nvidia)`, TTL by volatility); an answer-equivalence verifier (embedding NN + Arc agreement) and a calibrated
value-of-information gate decide local vs paid web. Memory = activation on graph nodes; memory.py / searchcache.py stay
as fallbacks. Local answers are labelled "local index hit · 0 paid · Onset index · as of …".
- Workload replay (800 questions, 30 users, Zipf + sessions + paraphrases + near-misses; Wikipedia as the paid stand-in,
  text-level): paid calls / question no reuse 0.555 -> today's cache 0.464 -> **Keystone + VOI 0.369** at the same regret
  (0.75 % vs 0.9 %), keyword accuracy 0.658 vs 0.646, p50 310 vs 375 ms; last 200 questions 0.265 vs 0.385 paid/q.
  Dev stream: 0.350 vs 0.359. Oracle with the same store: 0.284 (headroom for a learned director).
- Hand-gated components (cells, typed keys, hash, activation) reach 0.28-0.32 paid/q but at 6-10 % regret: the VOI gate
  (ECE 0.046 test) is what keeps accuracy. The learned 64-bit hash loses to plain embedding NN (top-1 0.82 vs 0.93); the
  Arc verifier keeps NN coverage and cuts near-miss false accepts ~30-50 % (19.5 % -> 13.5 %, 12.6 % -> 6.5 %).
- Streaming: relation known 3 words before the end; the key resolves on the last word (subject last) and a typed hit
  costs us, no embedding; dense top-1 converges before the end for 69 % of spoken eval questions (p50 lead 717 ms).
- First pass: brute fp32-256 + rescore beats HNSW below ~50k cells (0.35 ms at 10k); HNSW wins at 100k (1.3 ms).
- Real-audio A/B (58 set, Onset mode, paired back-to-back): cold store = no cost (+9 ms paired median, same sources);
  warm store after two fixes (endpoint reuses the speculator's query embedding; persistence off the event loop):
  paid 0.64 -> 0.36 /q, ready -100 ms paired median, 58/58 correct. The 2 cold misses (local04, local11) are
  local-moments questions the index never touches (same moments, answer wording vs strict keywords; 6/6 on re-run in
  both conditions). INDEX is now ON by default.
- Shared Keystone (server/keystone/shared.py, SQLite behind a SharedKeystone API): only Wikidata / verify-supported
  public typed edges pass the allowlist (user facts, "you", memory, activation, unverified answers refused; tested).
  Lookup: personal overlay -> shared -> VOI. Network replay: per-user 0.489 paid/q, + shared verified edges 0.470
  (shared answers 96 % correct), fully pooled 0.369. Labels "verified by Onset users · as of ...".
- Flags INDEX (default 1), INDEX_SERVE, INDEX_PERSIST, INDEX_SAVE_S, INDEX_SHARED (1), KEYSTONE_SHARED, GRAPH_KEY,
  GRAPH_ACT, VOI, VOI_THETA (fitted 0.95), KEYSTONE_DIR (data dir). Tests: tests/test_keystone.py (25). Pipeline:
  eval/keystone/run_all.sh; A/B: eval/keystone/ab_index.sh.

## Cost parity (branch feat/cost; server/cost.py, COST_MODE)
Tavily judges "faster and cheaper". Before this branch Onset paid for more searches than plain search: showcase 0.97
paid calls/q vs 0.74, 58 set 0.59 vs 0.28, and 1.7-2.0x the baseline's answer-model $. Goal: no more paid calls than
the baseline, answer-model $ <= 1.1x, >= 70 % of the speed gain kept.

**Where the extra cost came from** (traces of the main-branch runs):
- A first paid search on a prefix ("kansas city chiefs win game"), then a second when the speaker added the date or
  the subject ("... sunday", "when's best time year" -> "... iceland northern lights"): 7-8 of 29 showcase questions.
- The facts hedge at the endpoint (`_wait_or_hedge`, hedge_ms=0): a free speculative Wikidata lookup still out after
  350 ms is raced with a final call whose web leg is paid at once. On the 58 set this was 11 of the 12 paid calls
  beyond the baseline's (the baseline's own lookup is free; the router's 350 ms hedge rarely fires there).
- Answer-model waste: a draft fired for the meaning a soft endpoint was already answering (pure duplicate: sc03-05,
  sc10, sc14, sc19, sc22, sc30), up to 6 drafts per question on Wikipedia (DRAFT_BUDGET=6), hedge losers.
- A meter bug hid part of the baseline's waste: a hedge loser was counted only once its cancellation landed, and the
  baseline closes its question right after the answer, so its losers usually went uncounted. Fixed (answer.Stream
  marks the loser at the end of the answer); the "base" arm below still has the bug, so its model-$ ratio flatters
  the baseline slightly.

**What changed:**
- `COST_MODE=parity` (server/cost.py; config COST_*), now the default; `COST_MODE=speed` = the previous policy.
  - One paid web call per question, sent when the meaning is final: end-of-turn score >= COST_PARITY_EOT 0.3 (the
    fused score, fed in director mode too; Arc ear: the tagger gate; no score at all: SPEC_LAST_SETTLE_MS of stable
    words). Mid-sentence, only free lanes run (cards, Wikidata facts, local index, search cache). Plan parts
    (multi-part questions) follow the same rule.
  - That call is the endpoint's search (speculative-hit / pool reuse). The endpoint adds one call
    (COST_PARITY_FINAL_EXTRA=1) only when the pool doesn't cover the final words: a real meaning change (correction,
    new subject, ASR revision). Measured: 3 of 58 showcase answers and 0 of 116 58-set answers made 2 paid calls.
  - A free fact lookup still out at the endpoint is waited for (PENDING_CAP_S) instead of hedged with a paid call
    (COST_PARITY_FACTS_HEDGE_MS=-1).
  - At most one draft (COST_PARITY_DRAFTS), only when the end-of-turn score is >= 0.6 (COST_PARITY_DRAFT_EOT; at 0.3
    "how do I take a screenshot on a mac" was drafted before "... just part of the screen").
- Both modes: no draft for a meaning an answer is already streaming for (COST_DRAFT_DEDUP); Anthropic prompt caching
  of the answer model's system prompt (COST_CACHE): 727 tokens (count_tokens), above Haiku 5.5's 512-token minimum;
  cache hits $0.01/MTok vs $0.10 base, 5-minute writes $0.125 (platform.claude.com prompt-caching page, 2026-11-13).
  Verified live: the second request reads 728 tokens from the cache. The meter prices hits / writes at those rates.
- Meter (web/meter.js): the hero leads with the absolute difference, "same paid calls · 0.40 s sooner" when Onset
  made no more paid calls than plain search, else "0.23 s faster · +$0.0027 per question"; the ratio ("1.3x the cost")
  stays in the line under it; waste stays in every row. A Speed | Parity toggle (sent as `cost` with each run; the
  server applies it per session: Session(cost_mode=...)) and the price notes name the prompt cache.
- Tavily (docs.tavily.com, checked 2026-11-13): search_depth ultra-fast / fast / basic = 1 credit, advanced = 2.
  Onset already uses fast (finance: basic, fast is refused), the cheapest. include_answer has no credit cost listed
  (cost is by depth), but the probe measured +760 ms per call, and the existing TAVILY_ANSWER race is a second,
  parallel call (+1 credit). Not used: on the same call it would delay the sources by ~0.76 s; it saves at most the
  model's share (~$0.0002 of ~$0.008 per question). Parity's framing is the same call answered sooner.

**A/B** (Wikipedia lane, EAR=flux, FLUX_SOFT=director, DIRECTOR=on = the demo .env; search cache per run; "paid" =
calls that would be Tavily credits; $ est.: $0.008/credit + Haiku 5.5 tokens). base = main a3a76fb (separate
worktree); speed / parity = this branch. Alternating arm order, runs/.run_ab.lock per replay. The machine was shared
(load 5-58; another agent's A/Bs interleaved), so p50 is the mean of per-run p50s and "gain" is paired within each
run (BL p50 - HS p50). Raw files: scratchpad cost/*.json (not committed).

| set | arm | runs | HS p50 / p90 | BL p50 / p90 | gain ms | HS acc | BL acc | HS paid/q | BL paid/q | HS $/q | BL $/q | model $/q x1e6 HS / BL | HS waste tok/q | answers with 2 paid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| showcase (29) | base | 2 | 1220 / 1960 | 1834 / 2642 | 614 | 25/30 | 25/30 | 0.97 | 0.74 | 0.0081 | 0.0061 | 396 / 203 (1.95x)* | 2054 | 14/58 |
| showcase | speed | 2 | 1172 / 1830 | 1892 / 2554 | 720 | 28/30 | 27/30 | 0.98 | 0.72 | 0.0081 | 0.0060 | 243 / 170 (1.43x)** | 1134** | 16/58 |
| showcase | **parity** | 2 | 1725 / 3267 | 2335 / 3770 | **610** | 27/30 | 28/30 | **0.78** | **0.78** | **0.0064** | 0.0064 | 189 / 174 (**1.08x**) | 596 | 3/58 |
| 58 set | base | 2 | 1613 / 2542 | 2165 / 2956 | 552 | 113/116 | 115/116 | 0.59 | 0.28 | 0.0050 | 0.0025 | 320 / 187 (1.71x)* | 1352 | 6/116 |
| 58 set | speed | 2 | 1480 / 2080 | 2009 / 2760 | 530 | 115/116 | 116/116 | 0.50 | 0.21 | 0.0043 | 0.0018 | 272 / 134 (2.03x)** | 1561** | 6/116 |
| 58 set | **parity** | 2 | 1402 / 2048 | 1885 / 2708 | **483** | 116/116 | 115/116 | **0.14** | **0.22** | **0.0013** | 0.0019 | 154 / 139 (**1.11x**) | 461 | 0/116 |

p90 = pooled over both runs, except speed (mean of the two runs' p90). \* the base arm has the hedge-loser meter bug (the baseline's waste undercounted). \** speed: one of its two runs predates
the meter fix, so model $ / waste are from one run; latency, accuracy and paid calls from both. The parity showcase r1
ran under load 26+ (both modes ~0.6 s slower than r2: HS 2015 / BL 2729 p50); its gain is paired within the run.

Verdict vs the goal:
- Paid calls: parity <= baseline on both sets (showcase 0.78 = 0.78; 58 set 0.14 vs 0.22: the baseline's own
  router hedge pays on slow Wikidata lookups, parity waits for the free one). Meaning-change second calls: 3 of 58
  showcase answers (sc26 "butter" -> "buttermilk pancakes" in both runs, sc11 ASR revision "yolk sauce" -> "yolk
  soft"), none on the 58 set.
- Answer-model $: 1.08x (showcase) and 1.11x (58 set) the baseline's, from 1.95x / 1.71x. Half of what is left is
  hedge losers (HEDGE_MS duplicates on a slow API window, both modes hedge alike); the rest is a superseded soft
  answer or a draft whose speaker kept talking. Total $ per question: parity <= baseline on both sets.
- Speed: paired gain 610 ms vs base 614 (showcase, 99 %) and 483 vs 552 (58 set, 87 %). Accuracy unchanged.
- So COST_MODE defaults to parity. The 58-set model $ is 0.01 over the 1.1x line in both runs; say so if asked.
  COST_MODE=speed (or the meter's Speed toggle) brings back the earlier budget (~0.2 s more gain on the showcase in
  some runs, 1.3-2.3x the paid calls).

Tests: tests/test_cost.py (13: presets, parity holds the paid call until the meaning is final / speed fires on settle,
end-to-end parity <= baseline, meaning change gets one more call, multi-part within the plan cap, correction, no draft
for a meaning being answered, parity drafts only a final meaning, parity waits for a slow free fact lookup, cached
system prompt, meter prices cache hits / writes and splits them by answer / waste). tests/conftest.py pins
COST_MODE=speed for the older speculation tests. Full suite 333 passed.
Tools: eval/run_ab.py rows carry cost_mode + cache tokens, summaries mean_usd_llm / mean_usd_web, traces keep the
end-of-turn score (COST_TRACE_EOT=1 emits it per update, eval only).
