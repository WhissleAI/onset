# Onset audit: agentic voice search against the SearchHack brief

Audit of `main` at 411a69f (2026-11-14). The pipeline is mapped from the code, not from the docs. Every number cites
the file it came from. "est." marks an estimate. Read-only: no Tavily credits were spent. Tavily request fields and
prices were checked against docs.tavily.com (search, extract, map, api-credits) on 2026-11-14.

## Top 10 refinements (ranked by judge impact / effort)

| # | refinement | why a judge cares | effort | cost / risk |
|---|---|---|---|---|
| 1 | **Re-run the headline on real Tavily with today's stack** (Flux + director + router + Keystone + `COST_MODE=parity`): showcase (30) and the 58 set, both modes, plus the 12 current-events questions | The only real-Tavily A/B (`runs/ab_tavily.json` = `runs/ab_latest.json`) is git 21be397, from before the router, parity and Keystone: Arc ear, **2.43 credits/q vs 0.69**. Every parity number (paid calls ≤ baseline) is Wikipedia-lane only. A Tavily judge will ask for Tavily numbers. | 0.5 day (harness exists: `eval/run_ab.py`) | est. 60-110 credits: 2 modes × ~100 questions × 0.3-0.8 paid/q (parity, PROGRESS "Cost parity"). **Needs a credit OK.** |
| 2 | **Fix stale claims and the stale Results panel** (see §6) | `docs/PITCH.md` / `docs/JUDGE_QA.md` still say "no Tavily key", "≤ 6 web calls", "2.38 vs 0.66 web calls", "Deepgram never run", "Head Start". `/api/results` serves the pre-parity Tavily run (2.57 vs 0.71 web calls/q). In the Mac app the panel is empty, with developer copy ("The eval harness writes runs/ab_latest.json"). | 2-3 h | none |
| 3 | **Use what Tavily already returns**: blend the result `score` into pool ranking and coverage, and send `include_published_date: true` (free) for date chips and a freshness check in verify | `score` is parsed (`server/lanes.py:386`) and then ignored: the pool is ranked by EmbeddingGemma cosine only (`server/speculator.py:397-415`). `published_date` comes back only for `topic=news`, so verify cannot catch a stale page on a general question. | 0.5 day, flag `TAVILY_SCORE_FUSE` | 0 credits. Re-rank on the recorded `runs/narrow_cache.json` first. |
| 4 | **`/extract` as the verify escalation** (design in §3.2) | Covers the "verify" step of the agentic track with Tavily itself. Today a re-check runs on free lanes only (`VERIFY_RESEARCH_PAID=0`) and it confirmed 1 claim in 93 questions (PROGRESS "Phase 2"). | 1 day | ≤ 1 credit per escalation (1 credit per 5 URLs). It runs after the answer, so ready time is unchanged. |
| 5 | **Crawl → Keystone "site brain"** (§3.4): crawl one site offline (e.g. docs.tavily.com), load it into Keystone as a labelled local tier, then answer spoken questions about it at 0 paid calls | Shows Tavily `/crawl` + memory + routing in one demo beat. "Ask Onset about the Tavily API" lands with this judge. | 1 day | est. 10-20 credits, once |
| 6 | **Spoken answer (TTS) + barge-in** | There is no TTS anywhere (`web/`, `macapp/`: no speechSynthesis or AVSpeech). A voice-search demo with no voice reply looks unfinished on "new interfaces". On-device speech is free: `speechSynthesis` in the browser, `AVSpeechSynthesizer` in the Mac app. Barge-in = stop speaking when the VAD hears new speech. | 1 day | 0 $. Measure time to first audio. |
| 7 | **`/map` on site-shaped questions** (§3.3): "what does X sell", "X pricing", "X careers" → the official site from Wikidata P856 (free) → map with instructions → a link card | Discovery track: the answer arrives as a navigable site card, not just snippets. It can start speculatively once Arc tags the org mid-sentence. | 1 day | est. 2 credits per 10 pages. 10 s minimum timeout, so it runs in the background, never on the answer path. |
| 8 | **Personalization into the search call**: home city → Tavily `country`; diet / home into query shaping (done today); per-user domain preferences learned from cited sources | Problem 04. Today, user facts fill in missing slots (`server/memory.py`) and no Tavily parameter reflects who is asking. | 0.5 day | 0 credits (parameters only) |
| 9 | **Fix the copy a judge reads first** | The tagline "Search starts at the third word" (`web/index.html:23`) is no longer true for paid search under parity: the paid call goes out at end of turn and only free lanes run mid-sentence. Also the pipe ("Ear: Arc", "Lanes: web · local · audio") when Flux is the ear and cards / facts / Keystone are lanes; the banner tag "Degraded mode" (`web/app.js:204`); "ready · degraded"; "ready · wikipedia · haiku-5-5" (`web/app.js:187-190`). | 1-2 h | none |
| 10 | **Close the evidence gaps** a skeptical judge will probe: a human test of the browser mic (open since PROGRESS:56/206), a small real-voice set (all evals are TTS voices), and re-quoting the absolute ms measured under load average 20-115 | Credibility of every latency number | 0.5 day | none |

Also worth doing, lower on the list: `include_images` for entity questions (multimodal track, unmeasured
latency); `start_date` / `end_date` from Arc's typed date spans instead of dates written into the query text
(`NARROW_DATES`); a per-month credit guard (`TAVILY_PROCESS_CREDITS=800` is per process and resets on restart).

## Bugs and outright defects found

No code bug on the answer path. Product and data defects:

1. **The Results panel shows a superseded run.** `server/app.py:141-143` serves `runs/ab_latest.json`, which is
   byte-identical to `runs/ab_tavily.json` (git 21be397, pre-router, pre-parity). The panel and the meter's eval
   summary show "web calls/question 2.57 vs 0.71", about 3.4× the baseline's est. cost. That contradicts the parity
   story the live meter tells. The file is untracked, so a fresh clone or the Mac app (`scripts/build_macapp.sh:107`
   copies only `web_cache.json`) shows "No results yet. The eval harness writes `runs/ab_latest.json`."
   (`web/app.js:1557`).
2. **PITCH / JUDGE_QA say `runs/ab_full5.json` is identical to `runs/ab_latest.json`.** It is not (`cmp`):
   ab_latest is the Tavily run.
3. **Endpoint knobs change with the backend** (`server/config.py:93-110`: `DRAFT_BUDGET`, `SEM_SILENCE_MS`,
   `SEM_STABLE_MS`, `SEM_COVER_S` and `HEDGE_MS` all branch on `WEB_BACKEND == "tavily"`). `.env` has a Tavily key,
   so the demo runs the Tavily knobs (550 ms stable, 1000 ms hedge), while every recent A/B ran
   `WEB_BACKEND=wikipedia` with 420 / 800. The measured endpoint behaviour is therefore not the demo's. PROGRESS
   ("HEADLINE ON TAVILY") already saw local questions regress under those knobs (−110 ms, 9/20). This is a
   methodology risk, not a crash. Fix: pin the knobs explicitly in the eval and the demo, or measure item 1.

## 1. The real pipeline (from code)

```
mic (AudioWorklet, Float32 over /ws) or replay wav
 └─ ear  server/ear.py make_ear(): EAR=flux in .env (Deepgram Flux v2, eager EOT 0.3 / EOT 0.7);
          falls back to Arc on no key / connect / send failure (re-feeds the open utterance, DG_FALLBACK_REPLAY_S 10)
 └─ Arc audio metadata  server/arcmeta.py (ARC_META=mini): cc events, speaker, device-directed, affect...
 └─ Arc Tagger  server/tagger.py (native int8, 0.7 ms/word): intent, typed spans, dialogue act, command_complete
 └─ frame / heard  server/frame.py (corrections, repairs, PII redaction incl. spoken years), server/heard.py
          (LOOP_ASR_FIX=0: misheard-name repair, off), server/memory.py (pronoun / ellipsis resolved in place)
 └─ turn-taking  server/session.py + confidence.py + director.py (FLUX_SOFT=director, DIRECTOR=on in .env);
          open-mic gate server/openmic.py (OPENMIC_GATE=2, OPENMIC_SEARCH=1; push-to-talk never gated)
 └─ planner  server/planner.py:560 decompose() (rules, 0.02-0.04 ms): comparison / conjunction / multi-hop, ≤ 3 parts
 └─ speculator  server/speculator.py: settle 350 ms or EOT ≥ 0.3 → fire; COST_MODE=parity: mid-sentence only free
          lanes, one paid call when the meaning is final, +1 only on a meaning change; ≤ 1 draft at EOT ≥ 0.6
 └─ router  server/router.py: card (weather / fx / clock / units / sun) | facts (Wikidata, leads index, Wiktionary)
          | Keystone local tier (server/keystone/hook.py, VOI gate θ 0.95, shared verified edges) | web | wait
 └─ web lane  server/lanes.py: Tavily /search (else Wikipedia, else recorded cache); session ledger + TTL cache
          + semantic search cache (server/searchcache.py)
 └─ rank  EmbeddingGemma 2 cosine over the pool (fresh-only); local moments: RRF of text / lexical / tags / audio
 └─ answer  server/answer.py: Haiku 5.5, thinking off, effort low, max_tokens 160, prompt-cached system,
          inline [n] citations, hedge at HEDGE_MS, extractive fallback after ANSWER_DEADLINE_MS 4000
 └─ verify  server/verify.py:162 (numbers / years / names vs cited sources, ~1 ms) → recheck server/recheck.py
          (free lanes; VERIFY_RESEARCH_PAID=0)
 └─ UI  web/*.js: race panes, timeline, plan chips, verify badge, meter, memory graph. No TTS.
```

### Tavily calls actually made

Only `POST /search`. No `/extract`, `/map` or `/crawl` anywhere in `server/`. The body comes from
`server/lanes.py:224-288`:

| param | value | default in Tavily | note |
|---|---|---|---|
| `search_depth` | `fast` (spec + final); `basic` for finance (fast refused) | basic | 1 credit each |
| `topic` | general / news / finance from intent + Arc hints | general | |
| `time_range` | day / week for news; typed Arc date spans on general questions | null | |
| `max_age_hours` | weather 6, finance 1; retried without it on 434 | null | |
| `max_results` | 5 | 10 | |
| `chunks_per_source` | 2 | 3 | |
| `include_domains` + `include_domains_mode` | `prefer` for finance / weather / travel / reviews / local / health / recipe / home | [] / restrict | +.19 top-1 graded 2 (PROGRESS narrowing) |
| `include_answer` | false (`TAVILY_ANSWER=0`) | false | measured +760 ms (probe) |
| `include_raw_content`, `include_images` | false | false | |
| `include_usage` | true | false | credits reconciled |
| `auto_parameters` | false | false | |
| **not used** | `include_published_date`, `start_date` / `end_date`, `country`, `language`, `exact_match`, `exclude_domains`, `include_favicon` | | see refinements 3, 8 |

Response use: `title` (+ `published_date` for news), `content`, `url`, `score` (stored, then unused), `answer`
(unused), `response_time`, `request_id`, `usage.credits`.

### Flags that are off by default

`TAVILY_ANSWER`, `LOOP_ASR_FIX`, `LOOP_KEEP_EXACT`, `VERIFY_RESEARCH_PAID`, `NARROW_RERANK`, `STALE_CANCEL`,
`SPEC_DRIFT`, `DRAFT_ON_LAND`, `INDEX_SPEC_EMBED`, `CACHE_PERSIST`, `DIRECTOR_SPEC / KEEP / STREAMS / ARC_EAR /
ADDRESSED / ROUTE / PII`, `ARC_OFF`, `WEB_CACHE_RECORD`. `DIRECTOR` is off in code and on in `.env`. `EAR` is `arc`
in code and `flux` in `.env`. The Mac app picks Flux when a Deepgram key exists, else Arc
(`macapp/Sources/Server.swift:64-69`).

## 2. The six problems and nine tracks

Score = my read of what a judge can be shown today (0-5).

| problem / track | what Onset does (evidence) | score | honest gap | best refinement |
|---|---|---|---|---|
| 01 too much information | 5 results × 2 chunks; 1-2 sentence cited answer, < 45 words (`answer.py:13`); cards answer with no model call | 4 | no "show more", and no spoken answer | #6 TTS |
| 02 fragmented context | memory v2 across turns: follow-ups 32/34 vs Arc-off 2/34 (ARC_VALUE); planner merges parts | 4 | context stays in one device's store | #5 |
| 03 expensive at scale | parity: paid/q 0.14 vs 0.22 (58 set), 0.78 = 0.78 (showcase); model $ 1.08-1.11× (PROGRESS "Cost parity"); Keystone warm 0.64 → 0.36 paid/q; replay 0.369 vs 0.555 paid/q | 4 | all on Wikipedia; the only Tavily run says 2.43 vs 0.69 credits | **#1** |
| 04 weak personalization | user facts ("I live in Denver", diet) fill slots; activation bias in Keystone | 2 | nothing about the user reaches Tavily; no learned preferences | #8 |
| 05 limited memory | conversation 30 min, user facts 30 days (`MEM_*_TTL_S`), Keystone persistent + shared verified edges (96 % correct) | 4 | the shared store is SQLite on one machine | #5 |
| 06 wrong source / need a router | tiered router card / facts / Keystone / web / wait; card questions p50 1199 → 453 ms, paid 5 → 0.4/q (PROGRESS "Integration") | 5 | — | #3 (use `score` as a confidence) |
| T intelligent retrieval | narrowing (P@5 .797 → .828, top-1 .519 → .595 on real Tavily); EmbeddingGemma pool rank; local moments r@1 .80 → .90 | 4 | Tavily `score` thrown away | #3 |
| T long context & memory | as 02 / 05 | 4 | | #5 |
| T search routing | as 06 | 5 | | |
| T agentic search | plan (rules; 26-question multi-part set: p50 1889 vs 3343 ms planner off, parts 47/54) → search per part → verify (false "disagree" 15/104 → 0/52) → recheck (free lanes) → part-aware synthesis with citations | 3 | the planner is rules-only; verify only checks that sources say it; the re-check never uses Tavily; no "read the page" step | **#4**, #7 |
| T efficient search | ready p50 paired gain 483 ms (58 set) / 610 ms (showcase) at parity; 160 output tokens, prompt cache, ≤ 1 draft | 5 | absolute ms measured under load 5-58 | #1 |
| T knowledge & discovery | memory graph panel; moments index with timestamps | 2 | no site / topic exploration | #7, #5 |
| T personalized search | as 04 | 2 | | #8 |
| T multimodal | spoken audio → moments via EmbeddingGemma speech vectors (no ASR; r@4 1.0 on 10 questions); Arc paralinguistics | 3 | no images / video; audio lane only for local questions | `include_images` |
| T new interfaces | speculative race UI, timeline of fired / cancelled searches, open mic with a request gate (86.8 % of turns handled right) | 4 | no spoken reply; push-to-talk / open mic not explained on screen | #6, #9 |

## 3. Agentic search: what exists, and where /extract, /map and /crawl fit

### 3.1 Today
- **Plan:** `planner.decompose` (rules) → parts routed one by one; multi-hop bridge from a Wikidata value or role
  wording in the top results. Known gaps (PROGRESS "Plan"): "k two" vs K2, several founders, "compare X and Y"
  with no attribute.
- **Search:** speculative + final; coverage check (`SPEC_COVER_*`) decides pool reuse vs a second paid call.
- **Verify:** `verify.check` is string-level: numbers with scale, years, names against the cited sources, then all
  sources. It does not judge truth (the tooltip says so).
- **Re-search:** `recheck` runs one free lookup (Wikidata, leads) on the key claim. It can patch and label
  ("1775 (corrected from 1776)"). Paid web only with `VERIFY_RESEARCH_PAID=1`.
- **Synthesize + cite:** inline [n]; plan answers get a per-part hint; local moments are cited with mm:ss.

### 3.2 /extract (verify escalation + thin snippets)
- **Trigger A (post-answer, default candidate):** verify is `unverified` / `partial` / `disagree` on a web-routed
  answer, and the free re-check came back "still unconfirmed".
- **Action:** `POST /extract {urls: [top cited URL (+ the runner-up if it shares the subject)], query: <claim
  sentence>, chunks_per_source: 3, extract_depth: basic, format: text, timeout: 3}`, then re-run `verify.check` on
  the chunks and emit `re-checked ✓ (page)` / `corrected` / `still unconfirmed`.
- **Trigger B (pre-answer, behind its own flag):** the final pool's best cosine < `SPEC_COVER_SIM` and Tavily's
  top `score` is high. Extract the top URL with a 700 ms cap; fall back to the snippets on timeout.
- **Budget:** 1 credit per 5 successful URLs (failures are free). At the measured re-check rate (14-15 of 93
  questions), est. ≤ 0.2 credits/q. Latency: unmeasured. Probe 10 URLs first (est. 2 credits).
- **Voice value:** "read me what the page says" follow-ups ("what does that article say about…") hit the same
  extract with no new search.

### 3.3 /map (site-shaped questions)
- **Trigger:** an Arc org / product entity + a site-intent cue (sell, products, pricing, careers, docs, contact,
  menu, hours).
- **Domain:** Wikidata P856 (official website). It is free, and memory pre-warm can already fetch it.
- **Call:** `POST /map {url, instructions: "<cue> pages", max_depth: 1, max_breadth: 20, limit: 20, timeout: 10}`.
- **Cost:** 2 credits per 10 pages with instructions, so est. ≤ 4 credits per map. Cache it per domain in Keystone
  with the "stable" TTL (3 days).
- **Path:** never on the answer path, because the 10 s minimum timeout is far above the voice budget. The spoken
  answer comes from /search as today, and a "site" card with the mapped URLs follows. Onset-specific: fire it
  speculatively mid-sentence when the org is tagged, so it is usually back by the time the answer ends.

### 3.4 /crawl (offline knowledge for a vertical)
- **Real time:** not for real time (map + extract latency).
- **Use:** crawl one site at build time into Keystone cells with the label "Onset index · <site> · as of <date>".
  Example: `docs.tavily.com`, `limit: 50`, `extract_depth: basic`.
- **Cost:** est. 5 (map) + 10 (extract) = 15 credits, once.
- **Payoff:** spoken questions about that site answer from the local tier (0 paid, ~0.35 ms lookup at 10k cells,
  RETRIEVAL.md), with the VOI gate deciding when to go live instead.
- **Demo line:** "the docs live in Onset's memory, and Tavily keeps them fresh".

## 4. Voice specifics

| topic | today (evidence) | gap |
|---|---|---|
| turn-taking | Flux eager / EOT + fused score + director; soft endpoint stays warm and re-answers if the speaker continues (`session.py:576-595`) | 10 dB rooms: the fix (word-gap silence, chatter guard) is still not re-measured (PITCH limitations) |
| corrections | cue list + Arc spans; after-the-fact correction supersedes the answered question (`session.py:277-310`); corrections 64/66 vs Arc-off 59/66 | — |
| follow-ups | in-place resolution, pre-warm, hold; follow-ups searched after the endpoint 12/36 → 1/36 | ASR name errors ("mari guri") still lose turn 3 |
| open mic | gate v2 + search shapes: test scenes 86.8 % right, 21.5 % non-requests answered, 4.7 % missed; gold misses 11.5 % | 1 in 5 room remarks still gets answered |
| barge-in | new speech while answering is queued FIFO (`session.py:870`); only a correction supersedes | with no TTS, barge-in only matters once item 6 lands |
| spoken answer | **none** | item 6 |
| privacy | PII redaction from tagger spans (`frame.py:93`, spoken years 65 → 6 of 69 DOB turns); facts stay on device; shared store allowlist | 65.7 % of PII turns still leak some PII word (ARC_VALUE text level); the street + city stay by design |

### Latency budget per stage (measured; source in brackets)

| stage | p50 | source |
|---|---|---|
| Arc Tagger per word / revision | 0.7 ms / 3 ms | PROGRESS "Measured" |
| endpoint lag (baseline, ear's own pause rule) | 513-533 ms | ab_full5 / ab_tavily |
| planner decompose | 0.02-0.04 ms | PROGRESS "Plan" |
| Keystone lookup (10k cells) | 0.35 ms | PROGRESS "Keystone" |
| query embedding (CPU copy) | 43 ms | PROGRESS "Measured" |
| Wikidata speculative lookup | 236 ms (p90 706) | PROGRESS "Integration" |
| Tavily fast search | 210 ms (p90 514) probe; 440 ms response_time on narrow eval | PROGRESS Tavily / narrowing |
| sources ready after the last word | HS 650-703 vs BL 880-926 ms | ab_full5 / ab_tavily |
| Haiku first token | ~620 ms | PROGRESS "Measured" |
| verify | 0.5-2 ms | PROGRESS "Plan" |

### Failure modes

| failure | behaviour | evidence |
|---|---|---|
| answer model down | no token in 4 s, or breaker open (30 s) → extractive sentence from the top source, labelled "· not generated" / "model offline" | `answer.py:388-443`, `web/app.js:709,810` |
| Tavily 429 / 432 / 433 / 401 | process breaker → Wikipedia; 5xx / timeout (4 s) → that fire errors, a failed final → Wikipedia | PROGRESS Tavily lane |
| web down | 2 failures → recorded cache (`runs/web_cache.json`, 57 entries), labelled "cached" | config `WEB_BREAKER_*` |
| Flux down | connect / send failure → Arc ear, re-fed audio | `ear.py:303-340` |
| over budget | session 120 / process 800 credits; speculation stops 2 early | config `TAVILY_*` |
| mic | browser mic path still not tested by a human | PROGRESS:56, 206 |

## 5. Product and demo

- **Mac app vs web:** the same UI (WKWebView over the same server), so features are at parity. Keys go in the
  Keychain from Settings; the ear is chosen there. Differences:
  - Results panel: empty in the app (bug 1).
  - Warm-up: ~16 s in the app vs ~40 s noted in DEMO.md.
  - First run downloads the models, with a progress view.
  - 745 MB app.
- **First 30 seconds:** the header (tagline, pipe, Race / Onset / Baseline), "warming models…", then
  "ready · wikipedia · haiku-5-5". A judge sees engineer labels before any answer. Recommended order: ask one live
  question in Race; let the meter hero line ("same paid calls · 0.40 s sooner") carry the cost story; keep
  Results for Q&A only once it shows a parity run.
- **Error copy to change:** "Degraded mode" tag; "ready · degraded"; "Answer failed: <raw error>"
  (`web/app.js:700`); "No results yet. The eval harness writes runs/ab_latest.json".

## 6. Claims in docs/PITCH.md and docs/JUDGE_QA.md

| claim | status | current fact |
|---|---|---|
| Name "Head Start" (16 + 12 mentions) | stale | Onset |
| "TAVILY_API_KEY is not set; every number uses Wikipedia" (PITCH limitations, JUDGE_QA §9) | stale | a key exists; one 70-question Tavily A/B (pre-router): p50 1320 vs 1726 ms, paired 288 ms, 51/70, accuracy 93.1 % both, 2.43 vs 0.69 credits/q |
| "≤ 6 speculative + 1 final web calls" (`WEB_BUDGET`) | stale as the default | parity: 1 paid when the meaning is final, +1 on a meaning change (`COST_PARITY_*`); `SPEC_PAID_MAX=2` in speed mode |
| "2.38 vs 0.66 web calls/q, about 2.4 credits/q" | stale | parity (Wikipedia lane): 0.14 vs 0.22 (58 set), 0.78 vs 0.78 (showcase) |
| "Deepgram ear only tested against a fake" | stale | Flux is the `.env` ear and the director was trained on Flux traces; the Flux A/Bs are in PROGRESS |
| headline 1160 vs 1538 ms, 44/58, 348 ms (ab_full5, Arc ear, pre-router) | true for that file, not today's default config | parity re-run: paired gain 483 ms (58 set) / 610 ms (showcase), measured under load |
| "ab_full5 identical to ab_latest" | false | ab_latest = ab_tavily |
| "Degraded mode banner" | true, but the copy breaks the brief's plain-copy rule | rename (item 9) |
| "Browser mic not tested by a human" | still open | — |
| 20 dB babble 35/48, 440 ms; rate ±; simulator 3.9 % | supported (runs/robustness.md, PROGRESS); pre-router code | re-measure, or label "earlier build" |
| "Search starts at the third word" (UI tagline) | half true | free lanes and planning start mid-sentence; paid search waits for the end of turn under parity (PROGRESS "Search efficiency": the lead of the first useful fire is now ~0 or negative) |

Suggested rewrite of the cost line: "Onset makes no more paid searches than plain voice search (0.14 vs 0.22 per
question on 58 spoken questions, Wikipedia lane) and answers ~0.5 s sooner. On Tavily: <item 1 numbers>."
