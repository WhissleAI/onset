# Arc integration audit

This audit covers every field the Arc models produce in Onset (code id `headstart`). For each field it records where the field is consumed and whether a test proves that consuming it changes behaviour. It was written on branch `feat/loop` (based on `feat/cost`). The Keystone rows are read from `feat/index` (`docs/RETRIEVAL.md`, `server/keystone/`).

Status legend:
- **verified**: a test fails if the field stops changing behaviour.
- **untested**: it is consumed, but no test pins that.
- **UI-only**: it is shown and nothing else.
- **default-off**: it is consumed only behind a flag that is off.
- **unused**: it is produced and read by nothing.

New tests from this audit are in `tests/test_loop.py` and are named in the Test column.

## 1. Arc Tagger read-out

Sources: `server/tagger.py` poll, `frame.build`, `session._apply`.

| Field | Consumers (effect) | Test | Status |
|---|---|---|---|
| fine / coarse intent + p | `frame.query` (hint word), `frame.web_hints` (topic), `router` (fx / weather card / facts-leads), `narrow` (family), `speculator` (topic news), `session.open_mic_skip`, Keystone relkey / ahash | test_tagger_depth (several), test_tavily::test_session_sends_intent_params, test_open_mic::test_open_mic_gate, test_narrow::test_family_from_readout_and_words | verified |
| dialogue acts (`act_question`, `acts`) | `session.gate_open`, `open_mic_skip`, `user_statement`; confidence eot feature `act_q`; director features; Keystone write gate | test_open_mic::test_open_mic_gate, test_review_fixes::test_gate_tracks_latest_state, test_memory::test_user_fact_statement… | verified |
| `device_directed` | UI read-out; director feature; Keystone `ingest` stops below 0.3 (feat/index only) | Keystone: test_keystone::test_side_talk_is_never_written | **not used for gating on this branch** (gap G2) |
| entity mention spans + types (`slots`, `ent_types`) | memory kinds (`memory._ent` via KIND), one-turn carry person / thing, `router.place_of`, `searchcache` subject, speculator fires on first sight of a typed span, planner part types, `web_hints` (team / event_name → news), `narrow` (team → sports, work_product → reviews), **heard.py Arc type conditions the misheard-name match (new)** | test_memory::*, test_speculation::test_conversation_memory_resolves_pronouns, **test_loop::test_entity_type_team_makes_a_relative_time_question_news**, **test_loop::test_arc_type_conditions_the_match** | verified (now) |
| typed number spans (`num_types`; date / time / duration / money / percent) | money → finance topic; date / duration / time_of_day → `when` → Tavily `time_range`; `convert` | **test_loop::test_money_span_makes_a_price_question_finance**, **test_loop::test_typed_time_span_sets_freshness_on_a_general_question**, **test_loop::test_typed_month_and_year_spans_reach_tavily**, **test_loop::test_month_words_without_a_typed_span_do_not_set_freshness** | verified (now). **Fixed:** see below |
| PII spans (`redact_tok`, `redacted`) | `clean_text` masks them; query, phrases and convert exclude them; meter; director pii head can replace them | test_tagger_depth::test_personal_data_never_reaches_the_web_query (+2) | verified |
| safety.jailbreak | speculator: no speculative search or draft; planner: no decompose; UI | **test_loop::test_jailbreak_flag_stops_speculation** | verified (now) |
| events: command_complete (`cc_at`, `cc_score`) | `session.gate_open`, eot features, director features, UI | test_review_fixes::test_gate_tracks_latest_state / test_gate_on_eval_set, test_confidence::test_eot_score_rises… | verified |
| other events, `state` emit group | requested in `tagger.py` `emit=…state`; only CC is read | — | **unused** |
| self-correction (`corrected`, `dropped`, `cue`, `phrases`) | `router.repaired_text`, `lanes` key, speculator never holds corrections, planner replans, Keystone `live_slots` / relkey; **memory, carry and context now use the kept words (new)** | test_router::test_repaired_text…, test_integrate::test_correction_is_not_held, test_planner::test_correction_replans, test_keystone::test_a_corrected_subject_writes_nothing…, **test_loop::test_correction_keeps_the_abandoned_subject_out_of_memory** | verified. **Fixed:** see below |
| span score / thr / ver / alt | confidence word features, `slot_conf` type confidence, director features | test_confidence::test_fused_word_conf… | verified |
| `phrases` / `entity_query` | second entity-only web query, planner, searchcache, narrow rerank, memory subject; **heard.py rewrites it (new)** | test_answer_quality::test_entity_query, test_narrow::test_rerank_prefers_entity_and_domain, **test_loop::test_low_confidence_name_is_rewritten_and_the_search_rekeyed** | verified |
| `convert` | `router.fx_args`, `answer_hint`, planner skip | test_router::test_fx_args, test_repair::test_hint_reaches_the_answer_model | verified (planner skip untested) |

## 2. Arc audio metadata

Source: `server/arcmeta.py`.

| Field | Consumers | Test | Status |
|---|---|---|---|
| for_me (addressedness) | confidence eot feature (fitted weight −0.033: close to nothing), director feature, UI | test_ear_flux::test_arc_meta_rides_along… (display only) | effectively unused (gap G2) |
| cc_p | eot feature, director | test_confidence::test_fused_soft_endpoint_fires… | verified (weakly) |
| cc_events | eager + arc release (`session._arc_cc_since`), director cc_t, UI | test_ear_flux::test_eager_waits_for_arc_command_complete | verified |
| audio act / intent, speaker / role, bio, audio_events | UI (`hears.js`) | — | UI-only |
| overlap, emotion, arousal, speaker change, interruption, yield, n speakers, level, music, laugh, degradation, dd (x2 stream) | director v2 "arc" feature group only | test_director_v2::test_step_feats2_reads_arc_streams (features only) | default-off (DIRECTOR=on + DIRECTOR_STREAMS=1) |
| utt_open, rtf, at, model | nothing | — | unused |

## 3. Arc ear words

- ArcEar words carry no confidence, so `wconf` / `sconf` stay empty and the key-word confidence gate is off.
- heard.py then only re-scores Arc entity mentions, at a stricter threshold (+0.04). Tested by test_loop::test_no_ear_confidence_only_tagger_mentions_and_stricter.
- The final `reason` and the `arc_command_complete` ear event are UI-only.
- On Flux, the ear word confidence now feeds the misheard-name suspect signal `heard.suspect_conf`.

## 4. Director heads

Source: `server/director.py`.

| Head | Consumer | Test | Status |
|---|---|---|---|
| eot | soft endpoint (FLUX_SOFT=director) | partly (threshold forced to 0.99 in the test) | partly |
| fire / draft | speculator (DIRECTOR_SPEC=1) | none | untested, default-off |
| addr | open-mic (DIRECTOR_ADDRESSED) | test_director_v2::test_open_mic_gate_uses_addressed_head… | verified |
| route | holds paid speculation (DIRECTOR_ROUTE) | test_director_v2::test_route_hook_holds_or_releases… | verified |
| pii | `director.apply_pii` | test_director_v2::test_apply_pii_replaces_rule… | verified |
| keep | **never read**: DIRECTOR_KEEP is a dead flag | — | unused (gap G4) |

## 5. Confidence fusion

Arc inputs to `server/confidence.py`:
- **Word model:** tagger span in_span, margin and verification.
- **slot_conf:** span score vs alt.
- **eot model:** cc_last, cc_score, act_q, gate, slot_end from the tagger; cc_p, cc_recent and for_me from the audio model.

`arc_name_boost` has no caller, and arcmeta emits no names / numbers for it (dead).

**New finding (Phase 1):** the fused per-word confidence predicts whether a word *survives into the final transcript*. On final words it saturates: 97 % of misheard name words score ≈ 1.0. Flux's raw word confidence still separates them: 39 % of misheard name words are below 0.9, against 6.8 % of correctly heard names (6,000 real traces). heard.py therefore uses `min(ear, fused)`.

## Fixed in this branch (small wiring gaps)

1. **Typed time spans set Tavily freshness on their own.**
   - Before: `when` came from day / week regexes over all words. The Arc span type added nothing, month / year ranges were never produced, and a general question ("what did Apple announce yesterday") got no time_range unless it was a news intent or a sports result.
   - Now: an Arc `entity.number.date` / `duration` / `time_of_day` span is read together with the two words before it (Arc often tags just "month" of "this month"). It can set `month` and `year`, and these come only from typed spans: "the year of the dragon" sets nothing.
   - A typed time on a general question keeps topic general and sends `time_range` (`lanes.web_plan`).
   - The narrow family logic is unchanged: it still keys on news + range.
2. **A corrected-away subject no longer leaks.**
   - "who founded Microsoft, sorry, I meant Amazon": the memory turn text, the follow-up's previous question (`carry.prev.question`, used for "what about X" swaps) and the next turn's answer-model context used the raw words, Microsoft included.
   - They now use `Frame.kept_text` / `clean_text`.
   - `memory.observe` parses only the kept words for its fallback subject, asked kind and gender.
3. **`narrow`: "product" vs `work_product`.** Arc emits `work_product`, so the reviews-family branch on the entity type never fired.
4. **Arc entity type → heard.py.** A misheard span's Arc type (person / place / org) now conditions which name it may become.

## Bigger gaps (reported, not fixed here)

- **G1, Keystone docs overstate gating** (feat/index):
  - RETRIEVAL.md says side talk never writes a cell, edge or activation, but `note_answer`, `note_qvec` and `absorb_memory` don't check `addressed()`.
  - Typed number spans don't set the live-relation argument (relkey takes it from a word regex).
  - Not this branch's to edit.
- **G2, device_directed / for_me don't gate open-mic.** The obvious side-talk signal is ignored unless the director is on. The handoff measured `device_directed` ≥ 0.6 on 94 % of side talk, so the raw signal is too weak to gate on. It needs the fine-tuned head (arc_tagger_handoff docs/03) and an open-mic A/B, not a one-line rule.
- **G3, emotion / arousal / overlap / speaker** reach only the default-off director features. A cheap rule such as "overlap + speaker change → hold the speculative paid call" is plausible, but it needs the open-mic scenes to measure.
- **G4, dead director `keep` head / DIRECTOR_KEEP flag and dead `arc_name_boost`.** Delete them or implement them.
- **G5, untested consumption:**
  - director fire / draft (DIRECTOR_SPEC)
  - the key_conf delay in `speculator.on_frame`
  - the planner's `convert` skip
- **G6, produced but unread:**
  - tagger `state` group and non-CC events
  - Arc audio act / intent: could feed the open-mic gate
  - `fconf` / `key_conf` are emitted but not rendered
