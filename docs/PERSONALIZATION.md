# Personalization: answers for who is asking (PERSONA=1)

The organizers' problem 04: "weak personalization: results ignore who's asking and why". Onset already hears more
than the words. This feature uses that, plus what the user asked before, in two places: how the answer is written and
which sources it is written from. Every change is explained on screen in plain words ("Answered for you").

Code: `server/persona.py` (new), with one-line hooks in `frame.py`, `speculator.py`, `session.py`, `lanes.py` and
`app.py`; UI in `web/graph.js` / `web/graph.css`. Flag `PERSONA=1` (default **off**, see "Default" below).

## 1. How to answer this question (style)

Read per question, mid-utterance too (drafts get the same note), from:

| Signal | Source | Effect |
|---|---|---|
| speaking pace, syllables / s over the question's word times | ear word timings (Flux) | ≥ 7.0 syl/s on ≥ 3 words: "shorter: you asked quickly" |
| pauses ≥ 0.45 s inside the question, spoken fillers, hedges ("I think", "what's it called") | ear words + timings | "a little fuller: you sounded unsure" (answer + one plain sentence defining the key term) |
| a definition ask | Arc Tagger intent `qa_definition` or "what is X" | counts toward "unsure" with one other cue; lowers expertise |
| "no, that's not what I meant", "that's wrong" after an answer | words + the previous turn | "fixing it: you said the last answer missed" (starts with "Sorry,") |
| "just the …", "short answer" / "tell me more", "explain" | words | shorter / more detail, and learned for later turns |
| "in feet", "in km" | words | units, learned for later turns |
| throat clear | Arc audio bio events (`arcmeta`) | one hesitation cue |
| angry / high arousal | Arc audio emotion (`arcmeta` x2 stream) | with a follow-up, the "fix" tone. **Only when `DIRECTOR_STREAMS=1`; not measured here.** |

Arc does not produce age or gender buckets in this build (checked: `arc-v1` config has bio cough / breath /
throat_clear only). We would not store them anyway.

The style becomes one note appended to the answer prompt. Exact `count_tokens` cost on `claude-haiku-5-5`:

| note | tokens |
|---|---|
| short | 50 |
| fuller | 49 |
| fix | 34 |
| more | 32 |
| sense line ("Answer about Jaguar Cars …") | 28 |
| units | 15 |

Most questions get no note (see the measured per-question average below).

## 2. Who is asking (profile)

Per client id, next to conversation memory, in the app data dir (`HEADSTART_DATA_DIR`, i.e. ~/Library/Application
Support/Onset/persona/ in the Mac app; nothing is written into the bundle), 30-day TTL:

- **Interests**: the titles of the public articles an answer actually used (cited, or, with no citation, a source
  whose title shares a word with the question), with that article's embedding and a coarse category (car, planet,
  river, …; EmbeddingGemma label match). Recency-weighted (0.6 per turn, 7-day half-life). Never a Keystone cell
  (its title can be the user's question), a recording moment, a card, or anything from a turn with redacted personal
  data (`PII_YEARS` years included).
- **Preferences**: answer length (-1 short … 1 detailed), expertise, units. No acoustic baselines: per-speaker
  pace / hesitation norms belong to Arc state (feat/arcstate); `persona.relative()` folds its relative signals in
  (`server.arcstate.relative(session)` → `{pace_rel, hes_rel}`, used when that module exists; else absolute bars).
- **Not stored**: audio, transcripts, the user's words, demographics.
- The memory panel's "For you" row shows it; **export** downloads it as JSON (no vectors), **clear** (and "forget")
  deletes it here and on disk.

## 3. Retrieval

- **Interest rerank** (final + draft ranking): + `PERSONA_BIAS` (0.25) × cos(source, interests), only for sources of
  the user's recent category (e.g. "car" after two car questions), never for an earlier turn's own sources. When this
  changes the top source, the answer gets the sense line and the UI says why ("Mercury (planet): you asked about
  Venus earlier").
- **Sense search**: with a recent category, one extra free Wikipedia search "<subject> <category>" ("mercury planet")
  runs in parallel with the lane call. Its articles are kept only when they are named by the subject and Wikipedia
  shows the name has several senses (a disambiguation page or a "Subject (sense)" title) or the sense isn't in the
  results yet. Never on `WEB_BACKEND=cache`; free on every backend (it never calls Tavily).

## 4. Measured

Set: `eval/persona/` — 15 scripted spoken sessions, 41 turns (macOS `say` voices; quick turns at 270 wpm,
hesitant turns at 150 wpm with fillers and 0.45 s pauses), fed live through the Flux ear + an Onset session,
`WEB_BACKEND=wikipedia` (no Tavily spend), `COST_MODE` default. Each session is a new user. Arms alternate per
session in one process (`eval/persona/run_persona.py`). One repetition per run; machine shared and loaded (load
average 7-15), so treat ms as noisy.

- 9 **sense** turns: an ambiguous name after two questions on one topic ("How fast is a Jaguar?" after Porsche /
  Mustang, or after cheetah / lions; "How hot is Mercury?" after Jupiter / Venus; Mercury after gold / lead; python
  after anaconda / boa; Amazon after Nile / Congo, or after Microsoft / Google; Mustang after Tesla / Corvette;
  apples after mangoes / bananas). Scored on the answer's lead sentence (names the meant sense, not the other) or a
  top source that is that sense's article.
- 12 **style** turns: quick terse (short ≤ 20 words), hesitant (fuller ≥ 28), "tell me more" (≥ 45), "no, that's not
  what I meant" (apologises), units carried ("in feet" then "how tall is K2"), "just the short answer" carried to the
  next two questions.
- **content**: gold keys on every turn that has one (31).
- **judge**: `claude-haiku-5-5`, blind to the arm, scores 1-5 "fits this user" from the earlier questions, how it was
  spoken, the question and the answer. 82 calls per run: 28.4 k in + 3.4 k out tokens = **$0.0045 per run**.

| run (code) | arm | content | sense | style | judge (all) | judge (targeted turns) | p50 / p90 ms | note tokens / q (est.) |
|---|---|---|---|---|---|---|---|---|
| ab3 (before the other-sense rule) | off | 29/31 | 7/9 | 9/12 | 4.05 | | 1,961 / 2,688 | 0 |
| | on | 29/31 | 8/9 | **12/12** | 4.02 | | 1,995 / 2,815 | 12.8 |
| **ab4** (= this branch but the subject-word interest filter) | off | 30/31 | 6/9 | 7/12 | 3.78 | 3.86 | 2,505 / 3,625 | 0 |
| | **on** | 30/31 | **8/9** | **11/12** | **3.98** | **4.00** | 2,373 / 3,709 | 13.5 |

What changed, ab4, turn by turn:
- Sense fixed by persona: **Jaguar → the XJ220 car** after Porsche / Mustang (off: the big cat; chip "Jaguar XJ220:
  you asked about Ford Mustang earlier"), **apples → the fruit** after mangoes / bananas. "Mercury (planet): you asked
  about Jupiter earlier" flipped the top source; both arms' answers already led with the planet. One miss in both
  arms: Jaguar after cheetah / lions (the Wikipedia intros have no jaguar speed; both answers lead with "the sources
  don't give …"). The other sense turns were already right without persona (local Wikipedia picks the common sense).
- Style: quick terse 9 / 17 / 6 words (off 23 / 17 / 6), "just the short answer" carried 8 / 12 / 21 words (off 27 /
  28 / 33), the correction starts with "Sorry," (off: no acknowledgement), "tell me more" 78 words (off 53), hesitant
  50 / 40 (off 42 / 30). The one style miss: the third "short" turn (21 words; the learned preference had decayed to
  the threshold).
- Content: no turn lost (30/31 both; the miss is the same "how fast is a Mustang" ASR turn in both arms).
- One questionable flip: "where is it displayed" (Mona Lisa) led with "Mona Lisa replicas and reinterpretations"; the
  answer still said the Louvre.
- In ab3 (without the other-sense rule) the bias also overturned "Eiffel Tower" for "Eiffel Tower with Trees" (a
  painting) after Mona Lisa questions: why the rule exists.

Cost: the answer notes average 13.5 est. tokens per question (most questions get none; exact sizes in §1), about
0.6 % of the ~2,100-token prompt; answer input tokens per question were 2,057 on vs 2,111 off (source sets differ).
The sense search is one free Wikipedia request, in parallel; it never calls Tavily.

Latency: on the persona set, p50 2,373 ms on vs 2,505 off (ab4) and 1,995 vs 1,961 (ab3): no regression visible
within that noise. Per question, persona adds no network wait (the sense search runs in parallel with the lane call
and is only issued with a recent category) and microseconds of CPU (the category of each pool source is one 42 x 768
dot product, cached). **The showcase replay (`eval/showcase/questions.jsonl`) was not measured**: it was queued
behind the shared run_ab lock and stopped under the round's memory budget. To run it:
`PERSONA=0|1 nice -n 10 .venv/bin/python eval/run_ab.py --questions eval/showcase/questions.jsonl --modes headstart
--no-latest --wait-lock 5400 --out runs/ab_persona_sc_<arm>.json` (alternate the arms, RUN_AB_LOCK = main's lock).

## Default

**Off.** The persona-set A/B is positive (sense 6 → 8 / 9, style 7 → 11 / 12, judge 3.78 → 3.98, content equal) but
it is one repetition on 41 turns, and the showcase p50 check the brief requires before a default-on was not run.
Turn it on with `PERSONA=1` (sub-switches: `PERSONA_STYLE`, `PERSONA_SENSE`, `PERSONA_BIAS`).

## Open issues

- One repetition per arm on 41 turns: the sense / style gains are a handful of turns each. Re-run with `--repeat`-style
  reps on a quiet machine before quoting them hard.
- The style signals were measured on `say` voices. Real speakers' pace varies more; the 7 syl/s bar and the pause
  bar (0.45 s) need real-voice calibration; relative bars come from Arc state once feat/arcstate lands.
- Under Tavily the rerank applies to Tavily's results and the sense search still adds free Wikipedia articles; not
  measured (no Tavily spend in this round).
- Arc audio emotion feeds the "fix" tone only with `DIRECTOR_STREAMS=1`; unmeasured. `device_directed`, speaker and
  role are not used (they say who is talking, not what they want).
- Keystone is not touched (its files are being changed on feat/kdir). Natural next step: key the user's personal
  Keystone cells by the profile's categories so a remembered answer of the user's sense wins over a shared one.
- Memory can rewrite an ambiguous follow-up through the previous subject (seen once: "how fast is a jaguar" also
  searched "ford mustang gt fast"); that is memory.py's ellipsis swap, not persona.
- Labels in the profile are Wikipedia titles; a title can still be odd ("Mona Lisa replicas and reinterpretations").
