"""Settings: environment (+ headstart/.env) with defaults tuned for a laptop demo."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Model + data locations (server/modelpaths.py; models/manifest.json; scripts/fetch_models.py fills MODELS_DIR)
from . import modelpaths as _mp  # noqa: E402

APP_NAME = _mp.app_name()
MODELS_DIR = _mp.models_dir()
DATA_DIR = _mp.data_dir()
ARC_DIR = _mp.resolve("arc-v1", "ARC_DIR")
ARC_THREADS = int(os.environ.get("ARC_THREADS", "6"))
EMBED_MODEL = _mp.embed_model()
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "mps")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-haiku-5-5")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
WEB_BACKEND = os.environ.get("WEB_BACKEND", "tavily" if TAVILY_API_KEY else "wikipedia")
INDEX_DIR = _mp.resolve("index-data", "INDEX_DIR", legacy=ROOT / "index/data")
AUDIO_DIR = _mp.resolve("demo-audio", "AUDIO_DIR", legacy=ROOT / "eval/audio")   # replay questions (eval/audio/*.wav)

# Tavily web lane (docs.tavily.com/documentation/api-reference/endpoint/search). Credits: ultra-fast/fast/basic = 1,
# advanced = 2. Speculative fires are hidden behind speech, so they take a fast depth; the final (post-endpoint) query
# is on the user's critical path, so it defaults to the same fast depth (and baseline uses it too: same lanes).
TAVILY_BASE_URL = os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/")
TAVILY_SPEC_DEPTH = os.environ.get("TAVILY_SPEC_DEPTH", "fast")
TAVILY_FINAL_DEPTH = os.environ.get("TAVILY_FINAL_DEPTH", "fast")
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))
TAVILY_CHUNKS = int(os.environ.get("TAVILY_CHUNKS", "2"))              # 1-3; ignored by ultra-fast
TAVILY_TIMEOUT_S = float(os.environ.get("TAVILY_TIMEOUT_S", "4.0"))
TAVILY_WEATHER_MAX_AGE_H = int(os.environ.get("TAVILY_WEATHER_MAX_AGE_H", "6"))   # 0 = don't send max_age_hours
TAVILY_FINANCE_MAX_AGE_H = int(os.environ.get("TAVILY_FINANCE_MAX_AGE_H", "1"))
TAVILY_SESSION_CREDITS = float(os.environ.get("TAVILY_SESSION_CREDITS", "120"))   # per Session; 0 = unlimited
TAVILY_FINAL_RESERVE = float(os.environ.get("TAVILY_FINAL_RESERVE", "2"))        # speculation can't touch these
TAVILY_PROCESS_CREDITS = float(os.environ.get("TAVILY_PROCESS_CREDITS", "800"))   # whole server; 0 = unlimited
TAVILY_CACHE_TTL_S = float(os.environ.get("TAVILY_CACHE_TTL_S", "900"))
TAVILY_FINANCE_DEPTH = os.environ.get("TAVILY_FINANCE_DEPTH", "basic")   # topic=finance is refused on fast / ultra-fast
TAGGER_HINTS = os.environ.get("TAGGER_HINTS", "1") == "1"   # Tavily topic / freshness from the full Arc read-out
TAVILY_FRESH_TTL_S = float(os.environ.get("TAVILY_FRESH_TTL_S", "120"))         # news / finance / weather
# Search narrowing from the Arc read-out (server/narrow.py; measured by eval/narrow_eval.py -> runs/narrow_eval.json)
# Defaults = the measured best (PROGRESS.md "Search narrowing"): shaped query + include_domains (prefer) on the
# families where it helped; restrict OFF (top-5 grade -0.30, answering-doc-in-top-5 86 -> 58 %, though -219 ms);
# rerank OFF (no measurable effect on Tavily's own order).
NARROW_DOMAINS = os.environ.get("NARROW_DOMAINS", "prefer")       # off | prefer | restrict : Tavily include_domains per family
NARROW_DOMAIN_FAMILIES = os.environ.get(                           # comma list; "" = every family in narrow.FAMILY_DOMAINS
    "NARROW_DOMAIN_FAMILIES", "finance,weather,travel,reviews,local,health,recipe,home")
NARROW_PREFER_ONLY = os.environ.get("NARROW_PREFER_ONLY", "reference,news")   # families never restricted
NARROW_QUERY = os.environ.get("NARROW_QUERY", "1") == "1"          # generic words out, family noun in
NARROW_DATES = os.environ.get("NARROW_DATES", "1") == "1"          # (with NARROW_QUERY) "tomorrow" -> "November 13 2026"
NARROW_GATE = os.environ.get("NARROW_GATE", "1") == "1"            # no speculative credit on a low-information partial
NARROW_GATE_MIN_CONTENT = int(os.environ.get("NARROW_GATE_MIN_CONTENT", "2"))
NARROW_RERANK = os.environ.get("NARROW_RERANK", "0") == "1"        # read-out rerank of web results
NARROW_W_ENTITY = float(os.environ.get("NARROW_W_ENTITY", "0.5"))
NARROW_W_COVER = float(os.environ.get("NARROW_W_COVER", "0.5"))
NARROW_W_DOMAIN = float(os.environ.get("NARROW_W_DOMAIN", "0.3"))
NARROW_W_FRESH = float(os.environ.get("NARROW_W_FRESH", "0.3"))
NARROW_TODAY = os.environ.get("NARROW_TODAY", "")                  # tests / eval: fixed "today" (YYYY-MM-DD)
# session: each Session has its own cache (A/B-honest: baseline never reads Head Start's results). global: one cache
# for the server (saves credits on a demo booth replaying the same questions; do NOT use for A/B timing).
TAVILY_CACHE_SCOPE = os.environ.get("TAVILY_CACHE_SCOPE", "session")
# Race Tavily's own include_answer against Haiku on the after-endpoint path (miss + baseline). Off by default.
TAVILY_ANSWER = os.environ.get("TAVILY_ANSWER", "0") == "1"
TAVILY_ANSWER_DEPTH = os.environ.get("TAVILY_ANSWER_DEPTH", "basic")              # include_answer: basic | advanced

# Speculation policy (see speculator.py)
MIN_WORDS = int(os.environ.get("SPEC_MIN_WORDS", "3"))
STABLE_UPDATES = int(os.environ.get("SPEC_STABLE_UPDATES", "1"))
WEB_BUDGET = int(os.environ.get("SPEC_WEB_BUDGET", "6"))
LOCAL_BUDGET = int(os.environ.get("SPEC_LOCAL_BUDGET", "8"))
# Per-backend defaults from the policy simulator (sim/, validated ±4% vs the real A/B). Env vars always win.
# Wikipedia (~230 ms lane): current settings are within 10 ms of the best found. Tavily (~0.9 s lane): "T2" is
# 20 ms faster at p50, cuts ungrounded answers 4.0% -> 3.3% and answer-model calls by 17% (runs: sim_*; PROGRESS.md).
_TAVILY = WEB_BACKEND == "tavily"
DRAFT_BUDGET = int(os.environ.get("SPEC_DRAFT_BUDGET", "2" if _TAVILY else "6"))
DRAFT_STABLE_MS = int(os.environ.get("SPEC_DRAFT_STABLE_MS", "0"))
AUDIO_LANE = os.environ.get("SPEC_AUDIO_LANE", "1") == "1"
COMMIT_TAU = float(os.environ.get("SPEC_COMMIT_TAU", "0.55"))
# Local-routed questions: rank moments with server/rerank.py (text cosine + the audio lane's spoken-question
# vector + lexical/tags) instead of pool cosine alone. Same ranker in both modes (baseline has no audio vector).
RERANK_LOCAL = os.environ.get("RERANK_LOCAL", "1") == "1"
RERANK_AUDIO = os.environ.get("RERANK_AUDIO", "1") == "1"

# Semantic endpointing (Head Start only): answer when the tagger says the command is complete and the speaker has been
# silent this long, instead of waiting for the ear's own pause endpoint (~450 ms on Arc).
SEMANTIC_ENDPOINT = os.environ.get("SEMANTIC_ENDPOINT", "1") == "1"
SEM_SILENCE_MS = int(os.environ.get("SEM_SILENCE_MS", "200" if _TAVILY else "250"))
SEM_STABLE_MS = int(os.environ.get("SEM_STABLE_MS", "550" if _TAVILY else "420"))   # one Arc partial refresh (~0.4 s) must confirm the words
SEM_COVER_S = float(os.environ.get("SEM_COVER_S", "0.4" if _TAVILY else "0.3"))     # an ASR refresh this long after silence began covers the speech
PENDING_CAP_S = float(os.environ.get("PENDING_CAP_S", "1.0"))   # max wait at the endpoint for an in-flight speculative search
PENDING_EMPTY_S = float(os.environ.get("PENDING_EMPTY_S", "3.0"))  # ... when nothing at all has landed (no sources yet)
HEDGE_MS = int(os.environ.get("HEDGE_MS", "1000" if _TAVILY else "800"))   # duplicate the answer request if no token by then (both modes)

# Ear (ASR words for the sessions): EAR=arc (Arc v1 Mini built-in transcript, local) | flux (Deepgram Flux) | deepgram (Deepgram live websocket;
# needs DEEPGRAM_API_KEY, falls back to Arc without it or when the connection drops). See server/ear.py.
EAR = os.environ.get("EAR", "arc").strip().lower()
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DG_URL = os.environ.get("DG_URL", "wss://api.deepgram.com/v1/listen")
DG_MODEL = os.environ.get("DG_MODEL", "nova-3")
DG_LANGUAGE = os.environ.get("DG_LANGUAGE", "en")
DG_ENDPOINTING_MS = int(os.environ.get("DG_ENDPOINTING_MS", "300"))      # pause that closes an utterance (speech_final)
DG_UTTERANCE_END_MS = int(os.environ.get("DG_UTTERANCE_END_MS", "1000"))  # word-gap backstop (UtteranceEnd); min 1000; 0 = off
DG_WORD_FIELD = os.environ.get("DG_WORD_FIELD", "word")   # "word" (lowercase, like Arc's words) | "punctuated_word"
DG_KEEPALIVE_S = float(os.environ.get("DG_KEEPALIVE_S", "3"))            # Deepgram closes after 10 s without audio/KeepAlive
DG_CONNECT_TIMEOUT_S = float(os.environ.get("DG_CONNECT_TIMEOUT_S", "4"))
DG_FLUSH_TIMEOUT_S = float(os.environ.get("DG_FLUSH_TIMEOUT_S", "1.5"))  # wait for Finalize's result at the end of a run
DG_FALLBACK_REPLAY_S = float(os.environ.get("DG_FALLBACK_REPLAY_S", "10"))   # audio of the open utterance re-fed to Arc
DG_ARC_EVENTS = os.environ.get("DG_ARC_EVENTS", "0") == "1"  # also run Arc on Deepgram's words for its own events (CPU)
# Deepgram Flux (EAR=flux): conversational STT on /v2/listen with model-based turn detection
# (developers.deepgram.com/reference/speech-to-text/listen-flux, /docs/flux/configuration). Lowest-latency defaults:
# eager_eot_threshold 0.3 = the docs' minimum (EagerEndOfTurn as early as Flux allows; it drives our soft answer and a
# TurnResumed just means the longer question is answered again, which the session already handles); eot_threshold 0.7
# = docs' default (a premature EndOfTurn would SPLIT a question in two, which costs more than ~100 ms); eot_timeout_ms
# 1000 (docs range 500-60000, default 5000): a voice-search question is over after 1 s of silence.
FLUX_URL = os.environ.get("FLUX_URL", "wss://api.deepgram.com/v2/listen")
FLUX_MODEL = os.environ.get("FLUX_MODEL", "flux-general-en")
FLUX_EAGER_EOT = os.environ.get("FLUX_EAGER_EOT", "0.3")          # 0.3-0.9; "" or "off" disables eager events
FLUX_EOT = float(os.environ.get("FLUX_EOT", "0.7"))               # 0.5-1.0 (must be >= eager)
FLUX_EOT_TIMEOUT_MS = int(os.environ.get("FLUX_EOT_TIMEOUT_MS", "1000"))   # 500-60000
FLUX_KEYTERMS = [t.strip() for t in os.environ.get("FLUX_KEYTERMS", "").split(",") if t.strip()]
# What an EagerEndOfTurn does in the Head Start session: "eager" = soft answer now (no silence / settle wait: Flux's
# turn model already judged the pause); "eager+gate" = only if our tagger gate (session.gate_open) also agrees;
# "eager+arc" = now if Arc's command_complete (ARC_META) fired near the last word, else when it does while the words
# hold; "off" = ignore eager events (the tagger+silence semantic endpoint still runs on Flux's words either way).
FLUX_SOFT = os.environ.get("FLUX_SOFT", "fused").strip().lower()
# Arc audio metadata on the cloud ear's words (server/arcmeta.py): mini | transcriber | off. Arc runs its heads in
# external-transcript mode (no decoder): for-me (device-directed), command_complete, act/intent, speaker role, events.
ARC_META = os.environ.get("ARC_META", "mini").strip().lower()
ARC_CC_WINDOW_S = float(os.environ.get("ARC_CC_WINDOW_S", "0.6"))   # Arc cc event this close to the last word counts
ARC_META_MAX_PENDING = int(os.environ.get("ARC_META_MAX_PENDING", "20"))    # chunks queued before audio is dropped
# ARC_META=transcriber only (opt-in, not bundled): the arc_transcriber 1.2.x code + WhissleAI/arc-v1.1 weights
ARC_T_CODE = _mp.resolve("arc-transcriber", "ARC_T_CODE")
ARC_T_WEIGHTS = _mp.resolve("arc-v1.1", "ARC_T_WEIGHTS")
ARC_T_DEVICE = os.environ.get("ARC_T_DEVICE", "mps")
ARC_T_CHUNK_S = float(os.environ.get("ARC_T_CHUNK_S", "0.5"))   # Parakeet backbone stream chunk (model default 10 s)
ARC_T_CTX_S = float(os.environ.get("ARC_T_CTX_S", "1.0"))       # context each side (model default 2 s)
SEM_WORDGAP_MS = int(os.environ.get("SEM_WORDGAP_MS", "600"))   # noisy rooms: ASR word gap that counts as a pause
CHATTER_RATIO = float(os.environ.get("CHATTER_RATIO", "0.25"))   # -12 dB vs the last answered question = chatter

# Degraded / offline demo mode (see DEMO_CHECKLIST.md). WEB_BACKEND=cache serves the web lane only from the recorded
# cache; with a live backend the cache is the fallback when a live call fails (flagged "cached" in events, metric, UI).
WEB_CACHE_PATH = Path(os.environ.get("WEB_CACHE_PATH", ROOT / "runs/web_cache.json"))
WEB_CACHE_FALLBACK = os.environ.get("WEB_CACHE_FALLBACK", "1") == "1"   # live failure -> recorded cache
WEB_CACHE_RECORD = os.environ.get("WEB_CACHE_RECORD", "0") == "1"       # write every live result into the cache file
WEB_CACHE_MIN_COVER = float(os.environ.get("WEB_CACHE_MIN_COVER", "0.75"))   # fuzzy hit: share of query words covered
WEB_LIVE_TIMEOUT_S = float(os.environ.get("WEB_LIVE_TIMEOUT_S", "3.0"))  # hard deadline on one Wikipedia call
WEB_BREAKER_S = float(os.environ.get("WEB_BREAKER_S", "30"))     # after a live failure, go straight to the cache this long
# Answer model unreachable -> extractive answer (best-matching sentence of the top source, flagged), both modes.
ANSWER_FALLBACK = os.environ.get("ANSWER_FALLBACK", "1") == "1"
ANSWER_DEADLINE_MS = int(os.environ.get("ANSWER_DEADLINE_MS", "4000"))   # no first token by then -> extractive
ANSWER_BREAKER_S = float(os.environ.get("ANSWER_BREAKER_S", "30"))
ANSWER_MAX_RETRIES = int(os.environ.get("ANSWER_MAX_RETRIES", "2"))     # SDK retries per request (SDK default; the hedge is another)
# Confidence-aware speculation (server/confidence.py fused word confidence; only when the ear gives word confidences):
# a query whose weakest content word is below SPEC_CONF_LO waits for one more ear update before it spends a web call;
# drafts start only when the key words are >= SPEC_CONF_DRAFT. Measured in the simulator on real Flux traces.
SPEC_CONF = os.environ.get("SPEC_CONF", "1") == "1"
SPEC_CONF_LO = float(os.environ.get("SPEC_CONF_LO", "0.6"))
SPEC_CONF_DRAFT = float(os.environ.get("SPEC_CONF_DRAFT", "0.0"))
# Flux sends a TurnInfo every ~240 ms but the session only sees CHANGED words: FLUX_TICK=1 also ticks the speculator on
# unchanged updates (stability timers, drafts whose searches landed after the last word). DRAFT_ON_LAND=1 checks for
# a draft the moment a search lands instead of at the next ear update.
FLUX_TICK = os.environ.get("FLUX_TICK", "1") == "1"
DRAFT_ON_LAND = os.environ.get("DRAFT_ON_LAND", "0") == "1"
# Search director (server/director.py: tiny streaming Mamba-3, numpy int8): DIRECTOR=on steps it on every Flux update;
# FLUX_SOFT=director answers on its end-of-turn head; DIRECTOR_SPEC=1 lets its fire / draft heads gate speculation;
# DIRECTOR_KEEP=1 builds the final query from its keep head. Any failure falls back to FLUX_SOFT=fused.
DIRECTOR = os.environ.get("DIRECTOR", "off").strip().lower() in ("1", "on", "true")
DIRECTOR_SPEC = os.environ.get("DIRECTOR_SPEC", "0") == "1"
DIRECTOR_KEEP = os.environ.get("DIRECTOR_KEEP", "0") == "1"
# Director v2 (a v2 model in server/director/; PROGRESS "Director v2"). Every hook defaults OFF.
#   DIRECTOR_STREAMS=1   ArcMeta also pushes the Arc heads the assistant domain doesn't ship + the audio level
#                        (arcmeta.ArcExtra) to the director on every 100 ms chunk (without it those inputs read "missing")
#   DIRECTOR_ARC_EAR=1   ArcMeta also runs Arc's own decoder on the same audio: the Arc ear's words next to Flux's (CPU)
#   DIRECTOR_ADDRESSED=1 open mic: the addressed head decides "heard, not a request" (session.open_mic_skip's rule
#                        is the fallback when there is no reading); DIRECTOR_ADDR_THR overrides the shipped threshold
#   DIRECTOR_ROUTE=1     the route head releases / holds paid speculation before the router's rules can tell the lane
#   DIRECTOR_PII=1       the pii head picks the words kept out of the search (instead of frame._pii_hide)
DIRECTOR_STREAMS = os.environ.get("DIRECTOR_STREAMS", "0") == "1"
DIRECTOR_ARC_EAR = os.environ.get("DIRECTOR_ARC_EAR", "0") == "1"
DIRECTOR_ADDRESSED = os.environ.get("DIRECTOR_ADDRESSED", "0") == "1"
DIRECTOR_ADDR_THR = float(os.environ.get("DIRECTOR_ADDR_THR", "0") or 0)       # 0 = the model's own (director.json)
DIRECTOR_ROUTE = os.environ.get("DIRECTOR_ROUTE", "0") == "1"
DIRECTOR_ROUTE_THR = float(os.environ.get("DIRECTOR_ROUTE_THR", "0.8"))
DIRECTOR_PII = os.environ.get("DIRECTOR_PII", "0") == "1"
STALE_CANCEL = os.environ.get("STALE_CANCEL", "0") == "1"   # cancel in-flight searches the speaker revised away
WEB_BREAKER_FAILS = int(os.environ.get("WEB_BREAKER_FAILS", "2"))   # consecutive primary-backend failures before the cache takes over
MEMORY_TTL_S = float(os.environ.get("MEMORY_TTL_S", "180"))   # push-to-talk follow-ups keep context this long
# Spoken repairs the ear split into two turns ("what's the weather in Boston, sorry," | "I mean Denver"): a turn that
# starts with a repair marker (or follows one that ended on it) within REPAIR_GAP_S of audio after the previous turn
# is merged into it as a correction (session.py); so is any turn right after one that ended on a dangling function
# word ("is it bad to take a" | "ibuprofen on an empty stomach", sc15), superseding any answer already given. A final that ENDS on a
# marker is held up to REPAIR_HOLD_S (wall) for that next turn before it is answered. Both modes alike.
REPAIR_GAP_S = float(os.environ.get("REPAIR_GAP_S", "1.5"))
REPAIR_HOLD_S = float(os.environ.get("REPAIR_HOLD_S", "1.2"))
# Search efficiency (agent S; server/searchcache.py, speculator.py). SPEC_EFF=1: "search the subject, not the sentence":
# a speculative web search fires only when the query's MEANING changes (canonical key from the frame: topic, subject
# entities, terms, numbers/dates); a partial that only adds words re-ranks the docs already fetched, and the final
# question searches again only when the pool doesn't cover it. SPEC_EFF=0 = the old policy (a search per new word).
SPEC_EFF = os.environ.get("SPEC_EFF", "1") == "1"
SPEC_PAID_MAX = int(os.environ.get("SPEC_PAID_MAX", "2"))          # paid web calls per question, final included
SPEC_COVER_SIM = float(os.environ.get("SPEC_COVER_SIM", "0.5"))     # final query vs the pool's best passage (cosine)
SPEC_COVER_ENT = float(os.environ.get("SPEC_COVER_ENT", "1.0"))     # share of subject tokens named in the top docs
SPEC_COVER_SIM_ANY = float(os.environ.get("SPEC_COVER_SIM_ANY", "0.8"))  # pool reuse with no related search
SPEC_COVER_TERMS = float(os.environ.get("SPEC_COVER_TERMS", "0.5"))  # share of content terms found in the top docs
# A new meaning is searched once its words have held this long (Flux sends an update every ~240 ms: ~two
# updates later), the budget's last call (and a short no-entity prefix) after SPEC_LAST_SETTLE_MS, or at once when
# the fused end-of-turn score reaches SPEC_EOT_FIRE. Measured: sim/spec_eff.py (PROGRESS.md "Search efficiency").
SPEC_SETTLE_MS = float(os.environ.get("SPEC_SETTLE_MS", "350"))
SPEC_NOSUBJ_MIN = int(os.environ.get("SPEC_NOSUBJ_MIN", "3"))  # fewer words and no entity: wait SPEC_LAST_SETTLE_MS
SPEC_TAIL_SETTLE_MS = float(os.environ.get("SPEC_TAIL_SETTLE_MS", "0"))  # when the frame ENDS on its entity
SPEC_EOT_FIRE = float(os.environ.get("SPEC_EOT_FIRE", "0.3"))   # fused end-of-turn score that counts as settled
SPEC_LAST_SETTLE_MS = float(os.environ.get("SPEC_LAST_SETTLE_MS", "650"))
SPEC_DRIFT = os.environ.get("SPEC_DRIFT", "0") == "1"       # added words absent from the results = meaning drifted
SPEC_GROW_CHECK = os.environ.get("SPEC_GROW_CHECK", "1") == "1"   # "eiffel" -> "eiffel tower": skip if the docs name it
# Cross-question semantic cache of web results (both modes share it, like the recorded web cache). Hits are labelled
# cached + "as of" in events and the UI. CACHE_SEM=0 disables it (measurement); the A/B clears it per run.
CACHE_SEM = os.environ.get("CACHE_SEM", "1") == "1"
CACHE_NN_SIM = float(os.environ.get("CACHE_NN_SIM", "0.92"))       # embedding nearest neighbour (same subject/topic)
CACHE_TTL_LIVE_S = float(os.environ.get("CACHE_TTL_LIVE_S", "600"))         # news / sports / weather / prices
CACHE_TTL_GENERAL_S = float(os.environ.get("CACHE_TTL_GENERAL_S", str(6 * 3600)))
CACHE_TTL_STABLE_S = float(os.environ.get("CACHE_TTL_STABLE_S", str(3 * 86400)))   # definitions, conversions, ...
CACHE_MAX = int(os.environ.get("CACHE_MAX", "400"))
CACHE_PERSIST = os.environ.get("CACHE_PERSIST", "0") == "1"       # write runs/search_cache.json (docs only)
CACHE_PATH = Path(os.environ.get("CACHE_PATH", DATA_DIR / "search_cache.json"))

# ---------------------------------------------------------------------------------------------------- tiered router
# server/router.py: each question goes to the cheapest source that can answer it. card = live-data verticals
# (server/verticals/: weather, FX, time, units, sun), facts = local knowledge (server/knowledge.py: Wikidata properties,
# local Wikipedia lead index, Wiktionary), web = the paid lane (Tavily | Wikipedia). Both race modes route the same way
# (the router sits inside lanes.web), so the A/B still differs only in WHEN searches start.
ROUTER = os.environ.get("ROUTER", "on").strip().lower() in ("1", "on", "true")
ROUTER_SKIP_CONF = float(os.environ.get("ROUTER_SKIP_CONF", "0.75"))   # plan confidence at/above which paid web is skipped
ROUTER_TIMEOUT_S = float(os.environ.get("ROUTER_TIMEOUT_S", "0.6"))    # one free-API request
ROUTER_LANE_TIMEOUT_S = float(os.environ.get("ROUTER_LANE_TIMEOUT_S", "2.0"))   # a whole cheap lane (geocode + forecast)
ROUTER_SLOW_TIMEOUT_S = float(os.environ.get("ROUTER_SLOW_TIMEOUT_S", "1.5"))   # one request to a slower host (SPARQL, met.no)
ROUTER_CARD_ANSWER = os.environ.get("ROUTER_CARD_ANSWER", "1") == "1"  # a fitting card answers itself (no model call)
ROUTER_LEADS = os.environ.get("ROUTER_LEADS", "1") == "1"              # local Wikipedia lead index (if built)
ROUTER_LEADS_STRONG = float(os.environ.get("ROUTER_LEADS_STRONG", "0.76"))   # lead cosine that counts as an answer (calibrated on the eval factoids)
ROUTER_UA = os.environ.get("ROUTER_UA", "HeadStartDemo/0.1 (https://github.com/; voice search research prototype)")
# Router x speculation (integration): a PAID speculative web call waits while the router's plan confidence is below
# ROUTER_SPEC_HOLD_CONF ("unsure" what kind of question it is) and the turn isn't likely over; cheap card / fact
# lookups fire speculatively, at most ROUTER_SPEC_CHEAP_MAX distinct plans per question.
# Facts lane hedge: a fact lookup (Wikidata search + SPARQL) still out after this many ms starts the paid web lane too;
# the first good result wins (both modes; not for the speculator's free mid-sentence lookups). <0 disables.
ROUTER_FACTS_HEDGE_MS = float(os.environ.get("ROUTER_FACTS_HEDGE_MS", "350"))
ROUTER_SPEC_HOLD_CONF = float(os.environ.get("ROUTER_SPEC_HOLD_CONF", "0.3"))
ROUTER_SPEC_HOLD_LEADS = os.environ.get("ROUTER_SPEC_HOLD_LEADS", "1") == "1"   # ... also the generic wh- (leads) plan
ROUTER_SPEC_CHEAP_MAX = int(os.environ.get("ROUTER_SPEC_CHEAP_MAX", "4"))
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", ROOT / "index/data/knowledge"))

# ---------------------------------------------------------------------------------------------------- cost meter
# Prices for the on-screen cost-and-speed meter (server/meter.py, web/meter.js). Every $ shown is an ESTIMATE ("est."):
# measured calls / tokens x these list prices. Checked 2026-11-13.
# Tavily: pay-as-you-go $0.008 per API credit (https://www.tavily.com/pricing); a basic / fast search costs 1 credit
# (lanes.COST; advanced = 2). On the Wikipedia lane "paid" calls cost nothing; the meter prices them as Tavily calls.
METER_TAVILY_CREDIT_USD = float(os.environ.get("METER_TAVILY_CREDIT_USD", "0.008"))
METER_CREDITS_PER_CALL = float(os.environ.get("METER_CREDITS_PER_CALL", "1"))
# Answer model, USD per million tokens (input, output), base rates for prompts up to 100K tokens, from
# https://platform.claude.com/docs/en/about-claude/pricing. A model not listed here has no verified price: set
# METER_ANSWER_IN_USD_MTOK / METER_ANSWER_OUT_USD_MTOK; the meter flags the price as unverified until you do.
METER_MODEL_PRICES = {"claude-haiku-5-5": (0.10, 0.50)}
METER_ANSWER_IN_USD_MTOK = os.environ.get("METER_ANSWER_IN_USD_MTOK")
METER_ANSWER_OUT_USD_MTOK = os.environ.get("METER_ANSWER_OUT_USD_MTOK")

# ---------------------------------------------------------------------------------------------------- session memory
# server/memory.py: entity memory across a conversation's turns, user facts, a local per-browser store, the
# knowledge-graph panel. Measured with eval/run_conv.py (PROGRESS.md "Memory v2").
MEM_V2 = os.environ.get("MEM_V2", "1") == "1"                       # multi-turn entity memory (else one turn back)
MEM_INLINE = os.environ.get("MEM_INLINE", "1") == "1"              # resolve the pronoun IN the words for the router
MEM_HOLD = os.environ.get("MEM_HOLD", "1") == "1"                  # no speculation on an unresolved pronoun
MEM_PREWARM = os.environ.get("MEM_PREWARM", "1") == "1"            # free Wikidata pre-warm of likely follow-ups
MEM_PREWARM_TIMEOUT_S = float(os.environ.get("MEM_PREWARM_TIMEOUT_S", "4.0"))
# end of turn for a follow-up whose pronoun memory resolved: the director / fused score was trained on standalone
# questions and reads "when was it founded" as unfinished; with the referent known the bar is lower (eager, update)
MEM_EOT = os.environ.get("MEM_EOT", "1") == "1"
MEM_EOT_THR = [float(x) for x in os.environ.get("MEM_EOT_THR", "0.4,0.7").split(",")]
MEM_MAX_TURNS = int(os.environ.get("MEM_MAX_TURNS", "24"))
MEM_LOOKBACK = int(os.environ.get("MEM_LOOKBACK", "8"))            # turns a reference may reach back
MEM_PERSIST = os.environ.get("MEM_PERSIST", "1") == "1"            # runs/memory/<client hash>.json (git-ignored)
MEM_DIR = Path(os.environ.get("MEM_DIR", DATA_DIR / "memory"))
MEM_CONV_TTL_S = float(os.environ.get("MEM_CONV_TTL_S", "1800"))    # conversation entities survive a reconnect this long
MEM_USER_TTL_S = float(os.environ.get("MEM_USER_TTL_S", str(30 * 86400)))   # user facts ("I live in Denver")
MEM_USER_FACTS = os.environ.get("MEM_USER_FACTS", "1") == "1"      # learn + use simple facts the user says
# ---------------------------------------------------------------------------------------------------- planner + verify
# server/planner.py: a multi-part spoken question ("which is taller, Everest or K2", "flights to Denver and the weather
# there", "how old was the founder of Amazon when it was founded") is split into sub-questions while it is spoken; each
# part is routed and searched on its own (in parallel, as soon as it is stable) and the answer gets the sources grouped
# by part. Single-part questions never take this path (decompose() -> None: nothing changes).
PLAN = os.environ.get("PLAN", "1") == "1"
PLAN_MAX_PARTS = int(os.environ.get("PLAN_MAX_PARTS", "3"))
PLAN_EXTRA_PAID = int(os.environ.get("PLAN_EXTRA_PAID", "1"))    # paid speculative calls allowed per part beyond the first
PLAN_WAIT_S = float(os.environ.get("PLAN_WAIT_S", "3.0"))        # endpoint: max wait for the parts' searches
PLAN_PART_CAP_S = float(os.environ.get("PLAN_PART_CAP_S", "0.5"))  # endpoint: a part's speculative lookup still out
#                                                                   after this -> a final search races it
PLAN_K_PART = int(os.environ.get("PLAN_K_PART", "0"))            # sources per part (0 = 3 for 2 parts, 2 for 3)
# server/verify.py: after the answer, its numbers / years / names are checked against the cited sources (string and
# number matching, cross-source agreement); a badge shows the result. Runs after the last answer token: never delays it.
VERIFY = os.environ.get("VERIFY", "1") == "1"
VERIFY_TOL = float(os.environ.get("VERIFY_TOL", "0.01"))         # relative tolerance for non-year numbers
# ---------------------------------------------------------------------------------------------------- Keystone index
# server/keystone/: one structure for the index, the search cache and memory. Every passage / fact / answer Onset
# retrieves becomes a memory cell (the reranker's EmbeddingGemma vector, never embedded twice); every answered question
# writes a typed edge (subject) -[relation]-> (value cell), keyed by the Arc read-out; a learned answer-equivalence
# verifier and a calibrated value-of-information gate decide when a question can be answered locally instead of paying
# for the web. Write-up + measurements: docs/RETRIEVAL.md (runs/index/).
INDEX = os.environ.get("INDEX", "1") == "1"                        # ingest + local tier (ON: real-audio A/B, RETRIEVAL.md §7)
INDEX_SERVE = os.environ.get("INDEX_SERVE", "1") == "1"            # 0 = shadow: ingest + decide, never serve
INDEX_PERSIST = os.environ.get("INDEX_PERSIST", "1") == "1"
INDEX_SAVE_S = float(os.environ.get("INDEX_SAVE_S", "60"))         # persist at most this often (background thread)
KEYSTONE_DIR = Path(os.environ.get("KEYSTONE_DIR", DATA_DIR / "keystone"))
INDEX_MAX_CELLS = int(os.environ.get("INDEX_MAX_CELLS", "20000"))
INDEX_FP_DIM = int(os.environ.get("INDEX_FP_DIM", "256"))          # first-pass Matryoshka dims (runs/index/ann_bench.json)
INDEX_TIMEOUT_MS = float(os.environ.get("INDEX_TIMEOUT_MS", "80")) # local lookup budget (embedding included)
INDEX_SPEC_EMBED = os.environ.get("INDEX_SPEC_EMBED", "0") == "1"   # embedding lookups on speculative calls too
INDEX_HASH = os.environ.get("INDEX_HASH", "1") == "1"              # learned answer-equivalence hash (candidate source)
GRAPH_KEY = os.environ.get("GRAPH_KEY", "1") == "1"                # Arc-keyed typed edges (subject, relation)
GRAPH_ACT = os.environ.get("GRAPH_ACT", "1") == "1"                # activation (session / user memory) ranking bias
INDEX_SHARED = os.environ.get("INDEX_SHARED", "1") == "1"          # cross-user store of verified public typed edges
KEYSTONE_SHARED = Path(os.environ.get("KEYSTONE_SHARED", DATA_DIR / "keystone" / "shared.sqlite"))
VOI = os.environ.get("VOI", "1") == "1"                            # calibrated gate (else the hand gate)
VOI_THETA = float(os.environ["VOI_THETA"]) if os.environ.get("VOI_THETA") else None   # override the fitted theta
# ---------------------------------------------------------------------------------------------------- cost parity
# server/cost.py: COST_MODE picks the speculation budget per question (PROGRESS.md "Cost parity").
#   speed  = the policy above (SPEC_PAID_MAX paid calls, DRAFT_BUDGET drafts, settle-time release)
#   parity = one paid web call per question, fired when the meaning is final (end-of-turn score >= COST_PARITY_EOT),
#            reused at the endpoint; one more only if the endpoint's words aren't covered by it (a real meaning
#            change: correction, new subject); at most COST_PARITY_DRAFTS drafts, only on a final meaning.
# A browser session can pick its own mode ("cost" in the start / replay message; the meter's toggle).
# Default parity (measured, PROGRESS.md "Cost parity": paid calls <= the baseline's on the showcase and the 58 set,
# >= 80 % of the speed gain kept). COST_MODE=speed restores the earlier policy.
COST_MODE = os.environ.get("COST_MODE", "parity").strip().lower()
COST_PARITY_PAID_MAX = int(os.environ.get("COST_PARITY_PAID_MAX", "1"))       # speculative paid calls per question
COST_PARITY_FINAL_EXTRA = int(os.environ.get("COST_PARITY_FINAL_EXTRA", "1"))  # + calls the endpoint may add (miss)
COST_PARITY_EOT = float(os.environ.get("COST_PARITY_EOT", "0.3"))             # "meaning is final" (fused score)
# parity: a free speculative fact lookup still out at the endpoint is waited for (up to PENDING_CAP_S) instead of
# hedged with an immediate paid web call (ROUTER_FACTS_HEDGE_MS): on the 58 set that hedge was 11 of the 12 paid
# calls parity made beyond the baseline's. <0 = no hedge.
COST_PARITY_FACTS_HEDGE_MS = float(os.environ.get("COST_PARITY_FACTS_HEDGE_MS", "-1"))
COST_PARITY_DRAFTS = int(os.environ.get("COST_PARITY_DRAFTS", "1"))
COST_PARITY_DRAFT_EOT = float(os.environ.get("COST_PARITY_DRAFT_EOT", "0.6"))
COST_TRACE_EOT = os.environ.get("COST_TRACE_EOT", "0") == "1"   # eval: emit every end-of-turn score (spec_eot events)
# Both modes: no draft for a meaning an answer (soft or final) is already being generated for (it was pure waste:
# the draft started after the soft endpoint's own answer and was never read).
COST_DRAFT_DEDUP = os.environ.get("COST_DRAFT_DEDUP", "1") == "1"
# Anthropic prompt caching of the answer model's fixed system prompt (727 tokens on claude-haiku-5-5, counted with
# /v1/messages/count_tokens; the minimum cacheable prefix for Haiku 5.5 is 512 tokens). Prices from
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching (checked 2026-11-13), Haiku 5.5, prompts up
# to 100K: base input $0.10/MTok, 5-minute cache write $0.125 (1.25x), cache hit $0.01 (0.1x).
COST_CACHE = os.environ.get("COST_CACHE", "1") == "1"
COST_CACHE_WRITE_MULT = float(os.environ.get("COST_CACHE_WRITE_MULT", "1.25"))
COST_CACHE_READ_MULT = float(os.environ.get("COST_CACHE_READ_MULT", "0.1"))
# ---------------------------------------------------------------------------------------------------- Arc-off arm
# "Plain voice search" (mode="plain"; docs/ARC_VALUE.md): what a typical voice assistant does, to measure what Arc's
# understanding adds. The raw final transcript (punctuation stripped) is the web query; no tagger read-out at all (no
# correction handling, entity carry / memory, router lanes / cards, PII redaction, freshness / topic, open-mic gate,
# planner). Same ear, same web lane, same reranker, same answer model and prompt minus the frame hints.
# ARC_OFF=1 makes every Session plain (e.g. the live demo); eval/run_ab.py / run_conv.py use --modes ...,plain.
ARC_OFF = os.environ.get("ARC_OFF", "0") == "1"
PLAIN_CONTEXT = os.environ.get("PLAIN_CONTEXT", "0") == "1"   # plain arm: give the answer model the previous Q + A
# ---------------------------------------------------------------------------------------------------- loop (heard)
# server/heard.py: retrieval-in-the-loop speech understanding. A low-confidence span or an Arc entity mention that no
# known name spells is re-scored (phonetic key + letters, Arc type) against the names the session already holds
# (speculative results, memory, Wikidata labels looked up, the local leads index); a confident match rewrites the frame
# ("mari guri" -> Marie Curie) and re-keys the search. A word with fused confidence >= LOOP_KEEP_CONF is never changed.
LOOP_ASR_FIX = os.environ.get("LOOP_ASR_FIX", "0") == "1"
LOOP_KEEP_CONF = float(os.environ.get("LOOP_KEEP_CONF", "0.9"))      # never override a word at / above this
LOOP_SUSPECT_CONF = float(os.environ.get("LOOP_SUSPECT_CONF", "0.7"))  # a word below this makes its span suspect
# Off by default (the "never override a high-confidence word" rule): an Arc mention whose words are ALL high-confidence
# may still change, but only to a name with the identical phonetic key that the context (results / memory / answer /
# Wikidata) holds. Flux heard "mari guri" at 0.905 / 1.0 (conv21): only this path can fix it.
LOOP_KEEP_EXACT = os.environ.get("LOOP_KEEP_EXACT", "0") == "1"
LOOP_LOW_CONF = float(os.environ.get("LOOP_LOW_CONF", "0.5"))       # outside an Arc mention, the leads index needs this
LOOP_FIX_MIN = float(os.environ.get("LOOP_FIX_MIN", "0.84"))        # match score to rewrite
LOOP_FIX_MARGIN = float(os.environ.get("LOOP_FIX_MARGIN", "0.05"))  # over the runner-up candidate
LOOP_ANSWER_FIX = os.environ.get("LOOP_ANSWER_FIX", "1") == "1"     # memory: the answer's names re-resolve the question's
LOOP_LEADS_PENALTY = float(os.environ.get("LOOP_LEADS_PENALTY", "0.05"))  # the offline index is big: stricter
# server/recheck.py: verify-triggered re-search. A "disagree" / "unverified" / partial badge on the answer's key claim
# -> one targeted lookup AFTER the answer (free Wikidata, then the local leads index; the paid web only with
# VERIFY_RESEARCH_PAID=1, counted in the question's paid calls). Never delays the first answer token.
VERIFY_RESEARCH = os.environ.get("VERIFY_RESEARCH", "0") == "1"
VERIFY_RESEARCH_PAID = os.environ.get("VERIFY_RESEARCH_PAID", "0") == "1"
VERIFY_RESEARCH_TIMEOUT_S = float(os.environ.get("VERIFY_RESEARCH_TIMEOUT_S", "3.0"))
