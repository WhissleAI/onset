"""One voice-search session: ear events in, tagger + speculator in the middle, events out.

Ear events (any ear: Arc on live audio, or a scripted word stream in tests):
  on_audio(pcm)                       16 kHz float32, appended to the current question's audio
  on_partial(words)                   the current utterance's words so far [{text,start,end}] (may revise earlier ones)
  on_final(words)                     the utterance is over (ear endpoint); words are final
Each partial (about every 100 ms) also counts as a tick for the speculator's stability timers.
"""
from __future__ import annotations

import asyncio
import collections
import re
import time

import numpy as np

from . import frame as framemod
from . import config, confidence, director, heard, lanes
from . import memory as memmod
from .speculator import Speculator
from .tagger import Tagger

_END = re.compile(r"[.?!]+$")
CC = "discourse.assistant.command_complete"


def gate_open(f, cc_at: int | None, spans: list[dict]) -> bool:
    """Semantic-endpoint gate on the tagger's LATEST state (eval/gate_eval.py measures the rules on the eval set).

    Open when Arc's command_complete event fired at the newest word, or when the utterance is a question whose newest
    word ends a typed slot (entity / number span): "who founded the company Nvidia", "how tall is the Eiffel Tower".
    Not sticky: a command_complete at an earlier word no longer counts once more words arrive. (Arc dedups the event
    within 3 words, so it often cannot re-fire on the true last word; the slot-end alternative covers those.) The old
    gate, (any command_complete so far) OR act_question, was open on 90% of mid-question prefixes."""
    n = f.n
    if cc_at is not None and cc_at == n - 1:
        return True
    slot_end = any(s.get("family") in ("entity", "number") and s["type"] != "entity.numeral" and s["end"] == n
                   for s in spans)
    return bool(f.act_question and slot_end)


# open mic (click the mic, it keeps listening): what the room hears is not always a request. A plain statement ("i like
# to eat pancakes") is heard, labelled, and not searched / answered. Push-to-talk and replays never use this gate.
_REQUEST_START = re.compile(r"^(what|whats|what's|who|whos|who's|whom|whose|when|where|wheres|where's|why|which|how|hows|"
                            r"how's|is|are|was|were|do|does|did|can|could|will|would|should|shall|may|might|has|have|had|"
                            r"tell|show|find|search|look|give|get|play|open|define|explain|list|compare|check|any|anything)$")


def open_mic_skip(f, ctx: dict | None = None) -> str | None:
    """The open-mic gate: OPENMIC_GATE=2 (default) is server/openmic.py (follow-up rescue, device_directed, request
    evidence, with _rule_skip inside); 1 is the statement rule alone. ctx: raw ear words, seconds since the last answered
    request, Arc speaker ids (openmic.followup)."""
    if config.OPENMIC_GATE >= 2:
        from . import openmic
        return openmic.skip(f, ctx, _rule_skip)
    return _rule_skip(f)


def _rule_skip(f) -> str | None:
    """Open-mic gate on the frame at the endpoint. Returns why the utterance is not answered ("statement"), or None to
    answer it. Conservative: anything that looks like a request (a question or command act, a web intent the tagger is
    confident in, a question / command word up front) is answered; only an utterance the tagger reads as a statement
    or a report ("i like to eat pancakes", "we went there last week") is held back."""
    acts = set(f.acts or ())
    if f.act_question or "question" in acts or "command_request" in acts:
        return None
    if f.intent in framemod.WEB_INTENTS and f.intent_p >= 0.5:
        return None
    first = framemod.norm(f.words[0]) if f.words else ""
    if _REQUEST_START.match(first):
        return None
    if "statement" in acts or f.intent in ("intent.report_inform", "discourse.intent.report_inform"):
        return "statement"
    return None


def meaning(f) -> tuple:
    """What an answer depends on: slots + content words (incl. a carried referent). Fillers ("um"), word order and
    intent flicker don't change it."""
    return (tuple(f.slots), tuple(sorted(set(f.content))))


def clean(w: str) -> str:
    """Strip sentence-final punctuation so a partial "Paris." doesn't close the tagger's utterance early."""
    return _END.sub("", w.strip()) or w.strip()


class Session:
    def __init__(self, emit, mode: str = "headstart", domain: str = "assistant", t0: float | None = None,
                 open_mic: bool = False, cost_mode: str | None = None):
        # every event carries "q": the id of the question (utterance) it belongs to, so a client can tell this turn's
        # events from a previous turn's late ones (its answer streaming while the next question is heard)
        self._out = emit
        self.qid = 0
        if config.ARC_OFF:
            mode = "plain"
        self.mode = mode
        self.cost_mode = cost_mode               # COST_MODE override for this session (server/cost.py); None = config
        self.plain = mode == "plain"           # Arc-off arm: no tagger read-out anywhere (frame.plain)
        self.open_mic = open_mic                 # open mic: statements are heard, not answered (open_mic_skip)
        self.tagger = Tagger(domain=domain)
        self.t0 = t0 or time.perf_counter()        # wall time of audio sample 0
        self.history: list[dict] = []
        self.semantic = mode == "headstart" and config.SEMANTIC_ENDPOINT
        self.silence_ms = 0.0
        self.noise = []
        self.noisy = False                     # background too loud for the energy VAD (≈ <13 dB SNR)
        self.answered_level = None             # median voiced RMS of the last answered question
        self.last_answered_end = -1.0           # audio time of the last word of the last closed utterance
        self.last_request_end = -1.0            # ... of the last utterance that was answered (open-mic follow-ups)
        self.last_request_speaker = None        # Arc's speaker id when it was asked (None: not known)
        self.last_answered_words: list[str] = []
        self.pending_finals: collections.deque = collections.deque()   # closed utterances waiting for their answer
        self.busy = False                       # an answer is being produced (finals are answered one at a time, FIFO)
        self.final_task: asyncio.Task | None = None   # the current question's soft (semantic-endpoint) answer
        self.soft: dict | None = None        # a semantic-endpoint answer awaiting the ear's confirmation
        self.carry: dict = {}                # {"person": [...], "thing": [...]} from the last question + answer
        self.context = ""                    # last question + answer text, for the answer model
        self.ans_tagger = Tagger(domain="talk")
        self.bg_tasks: list = []
        self.mem = memmod.Memory()           # entity memory across turns + user facts (server/memory.py)
        self.prewarm_task: asyncio.Task | None = None
        self.carry_pool: dict = {}           # its documents + vectors, seeded into the next speculator
        self.ledger = lanes.Ledger()         # Tavily credits + response cache, across this session's questions
        self.closed = False
        self._audio_end = 0.0
        self.arc: dict | None = None           # latest Arc audio metadata (ARC_META, cloud ears): for-me, cc, role, ...
        # spoken repairs split across two ear turns ("... Boston, sorry," | "I mean Denver"): see _join()
        self.held: dict | None = None          # a final that ended on a repair marker, waiting for its correction
        self.last_closed: dict | None = None   # the last closed question (a correction right after it reopens it)
        self.superseded: list[dict] = []       # metrics of answers replaced by a spoken correction (not in history)
        self.audio_ended = False               # the speaker stopped (push-to-talk release / end of replay): no holds
        self._new_question()

    def set_t0(self, t0: float):
        """Align the session clock with audio sample 0 (wall time)."""
        self.t0 = t0
        self.spec.t0 = t0

    def _new_question(self):
        """Fresh listening state (tagger, words, speculator) for the next utterance. Doesn't touch busy / the queue:
        an earlier question may still be answering while this one is heard."""
        self.qid += 1
        self.tagger.reset()
        self.words: list[str] = []                   # what the tagger has
        self.times: list[tuple] = []
        self.spans: list[dict] = []
        self.current: dict | None = None
        self.cc_at: int | None = None                # word index of the newest command_complete event
        self.prev_intent = None
        self.frame = None
        self.audio: list[np.ndarray] = []
        self.q_audio_start: float | None = None      # audio time (s) of the first word of this question
        self.q_levels: list[float] = []              # voiced RMS while this question's words were arriving
        self.spec = Speculator(self._q_emit(self.qid), self.mode, self.t0, ledger=self.ledger, cost_mode=self.cost_mode)
        if self.carry_pool:
            self.spec.seed(self.carry_pool)
        self.spec.context = self.context
        self.spec.bg = self.bg_tasks                 # after-answer re-checks of every question (VERIFY_RESEARCH)
        self.soft = None
        self.final_task = None
        self.eager_wait: dict | None = None          # FLUX_SOFT=eager+arc: an eager end-of-turn waiting for Arc's cc
        self.track = confidence.WordTrack()          # per-word revision history (server/confidence.py)
        self.wconf: list[float] = []                 # fused per-word confidence
        self.cc_score: float | None = None           # score of the newest command_complete event (at cc_at)
        self.turn_ev: str | None = None              # latest Flux TurnInfo event + end_of_turn_confidence
        self.eot_p: float | None = None
        self._tick_synced = False
        self.last_ws: list[dict] = []
        self.eot_last: float | None = None           # latest fused end-of-turn score (read-out)
        self.dstream = director.Stream() if config.DIRECTOR and not self.plain else None
        self.director_out: dict | None = None
        self.words_changed_at = time.perf_counter()
        self.prefix_ws: list[dict] = []              # a corrected earlier turn, prepended to this one's words
        self.repair_checked = False                  # this utterance was already checked for being a correction
        self.held_note = False                       # a "spec_hold" (unresolved pronoun) was announced
        self.mem_used: str | None = None             # the entity memory resolved this question's reference to
        self.loop_pool = heard.Pool()                # names this question may have meant (LOOP_ASR_FIX, heard.py)
        self.loop_fixed: set = set()                 # (said, as) already announced

    def emit(self, e: dict):
        """Emit an event of the question being heard now (unless it names its own question)."""
        self._out({"q": self.qid, **e})

    def _q_emit(self, q: int):
        """An emitter pinned to question q (its speculator keeps emitting after the next question started)."""
        return lambda e: self._out({"q": q, **e})

    @property
    def events(self) -> set:
        """Tagger events that hold for the CURRENT words (command_complete only if it fired at the newest word)."""
        return {CC} if (self.cc_at is not None and self.cc_at == len(self.words) - 1) else set()

    # ------------------------------------------------------------------ ear events
    def on_audio(self, pcm: np.ndarray, at_s: float):
        """pcm ends at audio time at_s."""
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        self.noise.append(rms)
        self.noise = self.noise[-60:]
        # noise floor from the quietest chunks, clamped so a stretch of continuous speech can't become the "floor"
        raw_floor = float(np.percentile(self.noise, 10)) if len(self.noise) >= 5 else 0.0
        floor = min(raw_floor, 0.01)
        self.noisy = raw_floor > 0.01        # steady background above the clamp: energy silence is unreliable
        if rms > max(0.006, 2.5 * floor):
            self.silence_ms = 0.0
            if getattr(self, "words", None):
                self.q_levels.append(rms)
                self.q_levels = self.q_levels[-80:]
        else:
            self.silence_ms += 1000.0 * len(pcm) / 16000
        self.audio.append(pcm)
        # keep ~12 s
        tot = sum(len(a) for a in self.audio)
        while tot > 16000 * 12 and len(self.audio) > 1:
            tot -= len(self.audio[0])
            self.audio.pop(0)
        self._audio_end = at_s

    def question_audio(self) -> np.ndarray | None:
        if self.q_audio_start is None or not self.audio:
            return None
        buf = np.concatenate(self.audio)
        start_s = self._audio_end - len(buf) / 16000
        a = max(0, int((self.q_audio_start - 0.15 - start_s) * 16000))
        return buf[a:]

    def _fresh(self, ws: list[dict]) -> list[dict]:
        """Drop words of utterances already closed (a late or repeated partial must not reopen an answered question)."""
        if self.last_answered_end < 0:
            return ws
        return [w for w in ws if w.get("start") is None or w["start"] >= self.last_answered_end - 1e-3]

    # ------------------------------------------------------------------ spoken repairs across ear turns
    def _join(self, ws: list[dict], final: bool = False) -> list[dict] | None:
        """This turn's own (fresh) words -> the words of the question being heard. The ear can end a turn inside a
        self-correction (Flux fires EndOfTurn at "what's the weather in Boston, sorry," and hears "I mean Denver" as
        a new turn). A turn that starts within REPAIR_GAP_S of the previous one and either opens with a repair marker
        or follows a turn that ended on one (or on a dangling function word: "is it bad to take a" | "ibuprofen on an
        empty stomach") is the same question: the previous turn's words are prepended, so the
        frame's correction logic sees "Boston, sorry, I mean Denver" and asks about Denver; any answer already given
        to the previous turn is superseded. None = the words so far might start a marker ("I" of "I mean"): wait."""
        if not ws or self.plain:
            return ws                          # plain: every ear turn is its own question (no repair merging)
        start = ws[0].get("start") or 0.0
        h = self.held
        if h is not None and not h["merged"]:
            self.held = None
            if start - (h["ws"][-1].get("end") or 0.0) <= config.REPAIR_GAP_S:
                # the held turn (still the open question) continues with this one
                h["merged"] = True
                self.prefix_ws = list(h["ws"])
                self.repair_checked = True
                self.emit({"t": "repair", "ms": self.spec.ms(), "how": "held",
                           "words": [framemod.norm(w["text"]) for w in h["ws"]]})
            else:
                h["job"] = self._close_question(h["ws"])     # too late to be its correction: close it now
            h["ev"].set()
        lc = self.last_closed
        if not self.repair_checked and not self.words and lc is not None:
            if start - lc["end"] <= config.REPAIR_GAP_S:
                r = framemod.repair_start([framemod.norm(clean(w["text"])) for w in ws])
                if r is None and not final:
                    return None
                if r or lc["trailing"]:
                    self._reopen(lc)
            self.repair_checked = True
        return self._prefixed(ws)

    def _prefixed(self, ws: list[dict]) -> list[dict]:
        if not self.prefix_ws or not ws:
            return ws
        end = self.prefix_ws[-1].get("end") or 0.0
        return self.prefix_ws + [w for w in ws if w.get("start") is None or w["start"] >= end - 1e-3]

    def _reopen(self, lc: dict):
        """A correction of the question closed just before: supersede that question's answer (cancel it if it is still
        queued or being answered; withdraw it from history and conversation memory if it is done) and hear this turn
        as its continuation."""
        job = lc["job"]
        self.last_closed = None
        job["superseded"] = True
        if any(j is job for j in self.pending_finals):
            self.pending_finals = collections.deque(j for j in self.pending_finals if j is not job)
        m = job.get("metric")
        if m is not None:
            if any(x is m for x in self.history):
                self.history = [x for x in self.history if x is not m]
                m["superseded"] = True
                self.superseded.append(m)
            mem = job.get("mem_before")
            if mem is not None:
                # the replaced answer must not be what "it" / "and when?" refer to next
                self.carry, self.context, self.carry_pool = mem[:3]
                if len(mem) > 3:
                    self.mem.restore(mem[3])
                    self.emit_memory(why="correction")
                self.spec.close()
                self.spec = Speculator(self._q_emit(self.qid), self.mode, self.t0, ledger=self.ledger, cost_mode=self.cost_mode)
                if self.carry_pool:
                    self.spec.seed(self.carry_pool)
                self.spec.context = self.context
        else:
            ft = job.get("final_task")
            if ft is not None and not ft.done():
                ft.cancel()
            job["spec"].close()                     # a finalize in flight stops at its next step (superseded)
        self.prefix_ws = list(lc["ws"])
        self.emit({"t": "extended", "correction": True, "ms": self.spec.ms(),
                   "words": [framemod.norm(w["text"]) for w in lc["ws"]]})

    def _sync_words(self, ws: list[dict]):
        new = [clean(w["text"]) for w in ws]
        new = [w for w in new if w]
        k = 0
        while k < min(len(new), len(self.words)) and new[k] == self.words[k]:
            k += 1
        changed = False
        revised = k < len(self.words)
        if revised:                                  # revision of words[k:]
            if not self.plain:
                self.tagger.push(new[k:], from_=k)
            changed = True
        elif len(new) > len(self.words):
            if not self.plain:
                self.tagger.push(new[k:])
            changed = True
        self.words = new
        self.times = [(w.get("start"), w.get("end")) for w in ws if clean(w["text"])]
        self.confs = [w.get("conf") for w in ws if clean(w["text"])]
        if changed:
            h = self.track.update(new)
            self._tick_synced = True
            self.words_changed_at = time.perf_counter()
            if self.cc_at is not None and self.cc_at >= k:
                self.cc_at = None                    # the word it fired on was revised away
            if not self.plain:
                r = self.tagger.poll()
                self._apply(r, revised_from=k if revised else None)
            self._confs(h)
        if ws and self.q_audio_start is None:
            self.q_audio_start = ws[0].get("start") or 0.0
        return changed, k

    def _confs(self, h: dict | None = None):
        h = h or {"revs": self.track.revs, "age": self.track.age}
        try:
            self.wconf = confidence.word_confs(self.words, self.confs, h["revs"], h["age"], self.spans)
        except Exception:                            # a confidence read-out must never break the words path
            self.wconf = []

    def _apply(self, r: dict, revised_from=None):
        a0 = r.get("a0", 0)
        self.spans = [s for s in self.spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            self.current = r["current"]
        for u in r.get("utterances") or []:
            self.current = u
        for e in r.get("events") or []:
            if e["type"] == CC:
                w = e.get("word")
                self.cc_at = w if w is not None else len(self.words) - 1
                self.cc_score = float(e.get("score") or 0.0)

    def _words_json(self) -> list[dict]:
        confs = getattr(self, "confs", None) or []
        out = []
        for i, (w, t) in enumerate(zip(self.words, self.times)):
            d = {"w": w, "t0": t[0], "t1": t[1]}
            c = confs[i] if i < len(confs) else None
            if c is not None:
                d["conf"] = round(float(c), 3)          # ear word confidence (Deepgram Flux / nova)
            if i < len(self.wconf) and c is not None:
                d["fconf"] = round(self.wconf[i], 3)    # fused: ear + revision history + tagger span (confidence.py)
            out.append(d)
        return out

    def _carry(self, words: list[str]) -> dict:
        """The frame's conversation memory for these words: the one-turn carry, re-ranked / extended by the
        multi-turn entity memory (MEM_V2), plus the user's facts."""
        if not config.MEM_V2:
            return self.carry
        try:
            return self.mem.carry_for(words, self.carry)
        except Exception as e:                       # memory must never break the words path
            self.emit({"t": "error", "where": "memory", "error": f"{type(e).__name__}: {e}"[:300]})
            return self.carry

    def _build(self, carry=None):
        if self.plain:
            return framemod.plain(self.words)
        f = framemod.build(self.words, self.spans, self.current, self.events, self.prev_intent,
                           local_ok=bool(len(lanes.index())), carry=self._carry(self.words) if carry is None else carry)
        if self.wconf and len(self.wconf) == f.n and any(c is not None for c in getattr(self, "confs", []) or []):
            f.wconf = tuple(self.wconf)                # only when the ear gives word confidences (Flux / nova)
        if config.DIRECTOR_PII:
            director.apply_pii(f, self.director_out)
        ear = list(getattr(self, "confs", []) or [])
        if len(ear) == f.n:
            f.sconf = tuple(heard.suspect_conf(ear, self.wconf if len(self.wconf) == f.n else None))
        if config.LOOP_ASR_FIX:
            self._loop_fix(f)
        return f

    def _loop_candidates(self) -> heard.Pool:
        """The names this question may have meant: speculative results so far, conversation memory, Wikidata labels
        already looked up (the local leads index is added inside heard.propose)."""
        pool = self.loop_pool
        try:
            pool.add_docs(list(self.spec.pool.values()))
            for k, e in list(getattr(self.mem, "ents", {}).items())[-40:]:
                pool.add(memmod._title(e.get("name") or k), "memory", heard.TYPE_KIND.get(e.get("kind", ""), ""))
            for lab in heard.facts_labels()[-60:]:
                pool.add(lab, "facts")
            for name, kind in heard.keystone_names():   # Keystone: active graph nodes + their edge values
                pool.add(name, "keystone", heard.TYPE_KIND.get(kind, ""))
        except Exception:
            pass
        return pool

    def _sticky(self, f) -> list:
        """A name fixed while the ear was unsure keeps its fix when a later revision of the same span sounds the same:
        Flux heard "mari gure" (unsure) -> Marie Curie, then settled on "mari guri" at 0.905 / 1.0. The span's revision
        history is the evidence (the same reason the fused confidence uses revisions); only a same-phonetic-key
        revision of an Arc mention qualifies."""
        out = []
        if not self.loop_fixed:
            return out
        toks = [framemod.norm(w) for w in f.words]
        ments = heard._mention_spans(f)
        skip = set(f.dropped) | set(f.redact_tok) | set(f.cue)
        # any 1-3 word window (the tagger may not have re-marked the revised span yet when the endpoint comes)
        for n in (3, 2, 1):
            for aa in range(0, len(toks) - n + 1):
                bb = aa + n
                if any(i in skip or not toks[i] for i in range(aa, bb)) or toks[aa] in framemod.STOP:
                    continue
                said = " ".join(toks[aa:bb])
                k = heard.pkey(said)
                for old_said, name, kind in self.loop_fixed:
                    if len(k) >= 3 and heard.pkey(old_said) == k == heard.pkey(name) and \
                            heard.letters(said) != heard.letters(name.lower()):
                        t = next((tt for ma, mb, tt in ments if ma <= aa and bb <= mb), "")
                        return [heard.Fix(aa, bb, said, name, 1.0, "heard-earlier", t, 0.0, kind)]
        return out

    def _loop_fix(self, f):
        """Retrieval in the loop: re-resolve misheard names against what the session already knows (heard.py)."""
        try:
            fixes = heard.propose(f, list(f.sconf), self._loop_candidates()) or self._sticky(f)
            if not fixes:
                return
            heard.apply(f, fixes)
            for fx in fixes:
                k = (fx.said, fx.name, fx.ckind)
                if k not in self.loop_fixed:
                    self.loop_fixed.add(k)
                    self.emit({"t": "heard_fix", "ms": self.spec.ms(), **fx.to_json()})
        except Exception as e:                       # never break the words path
            self.emit({"t": "error", "where": "heard", "error": f"{type(e).__name__}: {e}"[:300]})

    def on_partial(self, ws: list[dict], now: float | None = None):
        """Never raises: a bad partial is reported as an error event and the ear keeps going."""
        if self.closed:
            return
        try:
            self._on_partial(ws, now)
        except Exception as e:
            self.emit({"t": "error", "where": "on_partial", "error": f"{type(e).__name__}: {e}"[:300]})

    def _on_partial(self, ws: list[dict], now: float | None = None):
        # words of the utterance being answered are done; anything after it is the NEXT question, heard (and
        # speculated on) even while the previous answer is still streaming
        ws = self._join(self._fresh(ws))
        if not ws:
            return
        self.last_ws = ws
        changed, k = self._sync_words(ws)
        f = self._build()
        if f.intent.startswith("discourse.intent."):
            self.prev_intent = (f.intent, f.intent_p)
        if changed:
            self.emit({"t": "words", "words": self._words_json(),
                       "from": k, "ms": self.spec.ms()})
            self.emit({"t": "frame", **f.to_json(), "spans": [
                {"type": s["type"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in self.spans],
                "arc": self.arc, "cc_word": self.cc_at, "ms": self.spec.ms(), "director": self._director_json()})
        self.frame = f
        self._note_used(f)
        if self._hold(f):
            return
        self.spec.on_frame(f, now)
        if self.turn_ev is None and gate_open(f, self.cc_at, self.spans):
            self.spec.on_eot(1.0, "gate")   # no Flux end-of-turn score (Arc / nova-3 ear): the tagger's gate stands in
        wall = time.perf_counter()
        stable_ms = (wall - self.words_changed_at) * 1000
        silence_ms = self.silence_ms
        if self.noisy and ws:
            # in steady noise the energy VAD never hears silence: use the gap since the ASR's last word ended
            # (needs a longer threshold: a word being spoken isn't emitted yet)
            gap = (getattr(self, "_audio_end", 0.0) - (ws[-1].get("end") or 0.0)) * 1000
            if gap >= config.SEM_WORDGAP_MS:
                silence_ms = max(silence_ms, gap - (config.SEM_WORDGAP_MS - config.SEM_SILENCE_MS))
        speech_stop = wall - silence_ms / 1000
        # chatter guard: an utterance much quieter than the last question we answered (a neighbour, the room) gets no
        # early answer; the ear's own endpoint still answers it if it really is a question
        quiet = self._quiet()
        # the words are trustworthy if they held through one ASR refresh, or if they last changed in a refresh made
        # after the speaker went quiet (that refresh already saw the whole question)
        settled = stable_ms >= config.SEM_STABLE_MS or self.words_changed_at >= speech_stop + config.SEM_COVER_S
        if (self.semantic and not quiet and f.n >= 2 and silence_ms >= config.SEM_SILENCE_MS
                and settled and gate_open(f, self.cc_at, self.spans) and self._soft(ws, f, by="tagger")):
            return
        qa = self.question_audio()
        if qa is not None:
            self.spec.on_audio(qa, f.route, f.n)

    def _hold(self, f) -> bool:
        """No speculation on these words (MEM_HOLD):
          - a pronoun whose referent is still being answered ("when was it ..." heard while the previous answer
            streams: memory is updated when it ends): a search now would be for "founded", wasted;
          - a statement of a user fact ("I live in Denver"): remembered, not searched.
        The frame is re-evaluated when the previous answer lands (_answer)."""
        if not config.MEM_HOLD or self.mode != "headstart":
            return False
        why = None
        if self.user_statement(f):
            why = "user fact"
        elif (self.busy or self.pending_finals) and self._unresolved(f):
            why = "pronoun: referent still being answered"
        if why and not self.held_note:
            self.held_note = True
            self.emit({"t": "spec_hold", "why": why, "ms": self.spec.ms(), "words": list(self.words)})
        return why is not None

    @staticmethod
    def _unresolved(f) -> bool:
        toks = [framemod.norm(w) for w in f.words]
        strong = any(t in framemod.PRONOUNS - framemod.WEAK_PRONOUNS for t in toks)
        mention = any(t not in framemod.NUMBER_TYPES for t, _ in f.slots)
        return strong and not mention and not f.carried

    def user_statement(self, f) -> bool:
        """"I live in Denver", "I'm vegetarian": a fact about the user, not a request (MEM_USER_FACTS)."""
        if not config.MEM_USER_FACTS:
            return False
        toks = [framemod.norm(w) for w in f.words if framemod.norm(w)]
        if not toks or not memmod.learn_facts(toks):
            return False
        return not (f.act_question or _REQUEST_START.match(toks[0]) or "command_request" in set(f.acts or ()))

    def _note_used(self, f):
        """Tell the graph panel which remembered entity this question's reference resolved to (it lights up)."""
        k = framemod.norm(f.carried) if f.carried else None
        if k and k != self.mem_used and self.mem.entity(k) is not None:
            self.mem_used = k
            e = self.mem.entity(k)
            self.emit({"t": "memory", "used": k, "label": e["name"], "turns": e["turns"], "ref": f.mem_ref,
                       "inline": f.inline, "ms": self.spec.ms()})

    def emit_memory(self, learned: list | None = None, why: str = "", q: int | None = None):
        try:
            e = {"t": "memory", "graph": self.mem.graph(self.mem_used), "learned": learned or [], "why": why,
                 "ms": self.spec.ms()}
            if q is not None:
                e["q"] = q                       # the turn that changed memory, not the question being heard
            self.emit(e)
        except Exception as e:
            self.emit({"t": "error", "where": "memory", "error": f"{type(e).__name__}: {e}"[:300]})

    def _quiet(self) -> bool:
        return (self.answered_level is not None and len(self.q_levels) >= 3
                and float(np.median(self.q_levels)) < self.answered_level * config.CHATTER_RATIO)

    def _soft(self, ws: list[dict], f, by: str) -> bool:
        """Semantic endpoint: the request looks complete and the speaker paused. Answer now (soft), but keep listening
        and keep the speculator warm: the ear's final confirms or extends it. by = what fired it (tagger: our gate +
        silence; eager: the ear's own early end-of-turn, Deepgram Flux EagerEndOfTurn)."""
        sig = meaning(f)
        if framemod.dangling_end([framemod.norm(w) for w in self.words]):
            return False                           # "is it bad to take a": mid-phrase, never complete
        if self.busy or not f.has_content() or (self.soft is not None and self.soft["sig"] == sig):
            return False
        if self.open_mic and self._not_request(f):
            return False                           # open mic, not a request: the ear's final labels it, no answer
        if self.user_statement(f):
            return False                           # "I live in Denver": remembered at the final, not searched
        if self.final_task is not None and not self.final_task.done():
            self.final_task.cancel()               # superseded by this longer question: one soft answer at a time
        prev = self.soft
        self.soft = {"words": [framemod.norm(w) for w in self.words], "sig": sig, "metric": None,
                     "end": ws[-1].get("end") or 0.0, "failed": False, "by": by, "q": self.qid}
        self.final_task = asyncio.create_task(self._soft_answer(self.spec, self.soft, f, extended=prev is not None))
        return True

    def on_meta(self, m: dict):
        """Arc audio metadata from the ear (server/arcmeta.py). Shown in the read-out; never raises. With
        FLUX_SOFT=eager+arc it can also release an eager end-of-turn that was waiting for Arc's command_complete."""
        if self.closed:
            return
        if m.get("x2_only"):                      # director v2 streams (every 100 ms chunk): kept, not shown
            self.arc_x2, self.arc_aw, self.arc_af = m.get("x2"), m.get("aw"), m.get("af")
            return
        self.arc = m
        self.emit({"t": "arc", **m, "ms": self.spec.ms()})
        w = self.eager_wait
        if w is not None and self.words == w["words"] and self._arc_cc_since(w["end"]):
            self.eager_wait = None
            try:
                f = self.frame or self._build()
                if self._soft(w["ws"], f, by="eager+arc"):
                    return
            except Exception as e:
                self.emit({"t": "error", "where": "on_meta", "error": f"{type(e).__name__}: {e}"[:300]})

    def _arc_cc_since(self, word_end: float) -> bool:
        """Arc's command_complete event fired near the newest word (eval/flux_gate_eval.py: within 0.6 s)."""
        ev = (self.arc or {}).get("cc_events") or []
        return any(e.get("t") is not None and e["t"] >= word_end - config.ARC_CC_WINDOW_S for e in ev)

    def on_eager(self, ws: list[dict], now: float | None = None):
        """The ear's own early end-of-turn (Flux EagerEndOfTurn: Flux's turn model is moderately confident the
        speaker is done; the EndOfTurn that follows carries the same words unless a TurnResumed comes first).
        config.FLUX_SOFT (measured by eval/flux_gate_eval.py on real Flux traces):
          eager       soft answer now (lowest latency; ~26 premature soft answers / 100 questions, mostly
                      self-corrections "X, sorry, I mean Y" and trailing qualifiers)
          eager+arc   now if Arc's command_complete already fired near the last word, else as soon as it does while
                      the words hold (+18 ms p50, premature halved)
          eager+gate  only if our tagger gate agrees (+140 ms p50, premature / 3)
          off         ignore eager events"""
        mode = config.FLUX_SOFT
        if self.closed or not self.semantic or mode in ("off", "fused", "director"):
            return                                     # fused: on_turn_tick already scored this EagerEndOfTurn
        try:
            own = self._fresh(ws)
            if not own:
                return
            self._on_partial(own, now)                # same words the partial just carried: normally a no-op sync
            ws = self._join(own)
            if not ws:
                return
            f = self.frame or self._build()
            if self.soft is not None and self.soft["sig"] == meaning(f):
                return                                 # the tagger gate already answered exactly this
            end = ws[-1].get("end") or 0.0
            ok = f.n >= 1 and not self._quiet()
            if ok and mode == "eager+gate":
                ok = gate_open(f, self.cc_at, self.spans)
            elif ok and mode == "eager+arc" and not self._arc_cc_since(end):
                self.eager_wait = {"words": list(self.words), "ws": ws, "end": end}
                self.emit({"t": "eager_skip", "ms": self.spec.ms(), "words": list(self.words), "waiting": "arc"})
                return
            if not ok:
                self.emit({"t": "eager_skip", "ms": self.spec.ms(), "words": list(self.words)})
                return
            self._soft(ws, f, by="eager+arc" if mode == "eager+arc" else "eager")
        except Exception as e:
            self.emit({"t": "error", "where": "on_eager", "error": f"{type(e).__name__}: {e}"[:300]})

    # ------------------------------------------------------------------ fused end of turn (FLUX_SOFT=fused)
    def on_turn_info(self, ev: str, eot_p: float | None):
        """Flux TurnInfo header, before its words reach on_partial."""
        self.turn_ev, self.eot_p = ev, eot_p
        self._tick_synced = False

    def eot_features(self, f, ws: list[dict]) -> dict:
        n = len(self.words)
        toks = [framemod.norm(w) for w in self.words]
        end = ws[-1].get("end") or 0.0
        slot_end = any(s.get("family") in ("entity", "number") and s["type"] != "entity.numeral" and s["end"] == n
                       for s in self.spans)
        a = self.arc or {}
        cc_last = self.cc_at is not None and self.cc_at == n - 1
        confs = getattr(self, "confs", None) or []
        return confidence.eot_features(
            eot=float(self.eot_p or 0.0), pause_s=max(0.0, self._audio_end - end), n_words=n, cc_last=cc_last,
            cc_score=(self.cc_score or 0.0) if cc_last else 0.0,
            gate=bool(gate_open(f, self.cc_at, self.spans) and f.has_content()), act_question=f.act_question,
            slot_end=slot_end, has_content=f.has_content(), last_content=bool(toks) and toks[-1] not in framemod.STOP,
            arc_cc_p=a.get("cc_p"), arc_cc_recent=self._arc_cc_since(end), arc_for_me=a.get("for_me"),
            is_eager=self.turn_ev == "EagerEndOfTurn", last_conf=confs[-1] if confs else None)

    def on_turn_tick(self, ev: str, ws: list[dict] | None = None):
        """After each Flux TurnInfo (its words already synced): age the revision history and, with FLUX_SOFT=fused,
        answer (soft) when the fused end-of-turn score clears the fitted threshold (eval/fit_confidence.py: one
        threshold on EagerEndOfTurn, a higher one on plain updates). Never raises."""
        if self.closed or ev == "EndOfTurn" or not self.words:
            return
        try:
            if ws:
                # Flux re-sends the whole turn every update: same text, but fresher word timings and confidences
                # (the ear only forwards a partial when the TEXT changed). The end-of-turn features need both: the
                # pause since the last word and the last word's confidence (measured: stale values delayed the fused
                # endpoint by ~80 ms p50 in the Flux-trace replay)
                ws = self._fresh(ws)
                ws = self._prefixed(ws)
                if ws and [clean(w["text"]) for w in ws if clean(w["text"])] == self.words:
                    self.last_ws = ws
                    self.times = [(w.get("start"), w.get("end")) for w in ws if clean(w["text"])]
                    self.confs = [w.get("conf") for w in ws if clean(w["text"])]
            if not self._tick_synced:
                self.track.update(self.words)
                self._confs()
                if config.FLUX_TICK and self.frame is not None and self.frame.n == len(self.words):
                    f0 = self._build()                 # same words, fresher confidences: a speculator tick
                    self.frame = f0
                    self.spec.on_frame(f0)
            self._tick_synced = False
            mode = config.FLUX_SOFT
            if self.dstream is not None and self.last_ws:
                f = self.frame or self._build()
                out = None
                try:
                    out = self.dstream.step(director.session_record(self, f, self.last_ws, ev), self.wconf, f)
                except Exception as e:                 # the director must never break the session: fall back
                    self.emit({"t": "error", "where": "director", "error": f"{type(e).__name__}: {e}"[:300]})
                    self.dstream = None
                if out is not None:
                    self.director_out = out
                    f.director = out
                    self.emit({"t": "director", **self._director_json(), "ev": ev, "ms": self.spec.ms()})
                    if config.DIRECTOR_SPEC:
                        self.spec.on_frame(f)          # its fire / draft heads just changed: let the speculator act
                    if mode == "director" and self.semantic and not self.busy:
                        # the speculator's search-efficiency release (SPEC_EOT_FIRE, tuned on the fused score) gets the
                        # same fused end-of-turn score as under FLUX_SOFT=fused: only the soft endpoint changes owner.
                        # (Without this, director mode never fed the speculator: fewer early searches, later
                        # sources, +240 ms ready on the showcase A/B, runs/ab_v2_showcase_*_r1.json.)
                        p = confidence.eot_score(self.eot_features(f, self.last_ws))
                        if p is not None:
                            self.eot_last = p
                            self.spec.on_eot(p)
                        th = (director.model().thr or {}).get("director_eot") or [0.65, 0.9]
                        thr = self._mem_thr(f, ev, th[0] if ev == "EagerEndOfTurn" else th[1])
                        if out["eot"] >= thr and not self._quiet():
                            self._soft(self.last_ws, f, by="director")
                        return
            if mode == "director":
                mode = "fused"                         # no director model / it failed: the fused logistic decides
            if mode != "fused" or not self.semantic or self.busy or not self.last_ws:
                return
            f = self.frame or self._build()
            p = confidence.eot_score(self.eot_features(f, self.last_ws))
            if p is None:
                return
            self.eot_last = p
            self.spec.on_eot(p)                        # search efficiency: a likely-final meaning may search now
            thr = self._mem_thr(f, ev, confidence.eot_threshold("eager" if ev == "EagerEndOfTurn" else "update"))
            if p >= thr and not self._quiet() and f.n >= 1:
                if self.soft is None or self.soft["sig"] != meaning(f):
                    self.emit({"t": "eot_score", "p": round(p, 3), "thr": thr, "ev": ev, "ms": self.spec.ms(),
                               "words": list(self.words)})
                self._soft(self.last_ws, f, by="fused")
        except Exception as e:
            self.emit({"t": "error", "where": "on_turn_tick", "error": f"{type(e).__name__}: {e}"[:300]})

    def _not_request(self, f, out: dict | None = None, ws: list | None = None) -> str | None:
        """Open-mic gate: why this utterance is heard but not answered, or None to answer it. The rule
        (open_mic_skip) unless DIRECTOR_ADDRESSED=1 and the director has read these words (its addressed head)."""
        ws = ws if ws is not None else self.last_ws
        start = next((w.get("start") for w in ws or [] if w.get("start") is not None), None)
        ctx = {"raw": [w.get("text", "") for w in ws or []] or None, "speaker": (self.arc or {}).get("speaker"),
               "request_speaker": self.last_request_speaker,
               "since_request_s": None if start is None or self.last_request_end < 0 else start - self.last_request_end}
        why = open_mic_skip(f, ctx)
        if config.DIRECTOR_ADDRESSED and why is None:
            # the director's addressed head on top of the rule (both must say "request"; offline the AND beats either)
            o = out if out is not None else self.director_out
            if o and "addr" in o and [framemod.norm(w) for w in o.get("words") or []] == \
                    [framemod.norm(w) for w in f.words][-len(o.get("words") or []):]:
                thr = config.DIRECTOR_ADDR_THR or float((director.model().thr or {}).get("director_addr", 0.5))
                return "not_addressed" if o["addr"] < thr else None
        return why

    @staticmethod
    def _mem_thr(f, ev: str, thr: float) -> float:
        """End-of-turn bar for a follow-up whose reference memory resolved in place (MEM_EOT): the turn models were
        fit on standalone questions and read "when was it founded" as unfinished; with the referent known it is a
        complete question ("when was nvidia founded")."""
        if config.MEM_EOT and f is not None and f.carried and f.inline:
            return min(thr, config.MEM_EOT_THR[0 if ev == "EagerEndOfTurn" else 1])
        return thr

    def _director_json(self):
        o = self.director_out
        if not o:
            return None
        j = {"eot": round(o["eot"], 3), "fire": round(o["fire"], 3), "draft": round(o["draft"], 3)}
        if "addr" in o:
            r = director.ROUTES[int(np.argmax(o["route"]))]
            j.update(addr=round(o["addr"], 3), route=r, route_p=round(float(np.max(o["route"])), 3))
        return j

    async def _soft_answer(self, spec: Speculator, soft: dict, f, extended: bool):
        last_end = soft["end"]
        q = soft.get("q", self.qid)
        if extended:
            self.emit({"t": "extended", "ms": spec.ms(), "words": soft["words"], "correction": f.corrected, "q": q})
        self.emit({"t": "endpoint", "ms": spec.ms(), "speech_end_ms": round(last_end * 1000),
                   "frame": f.to_json(), "by": "semantic", "trigger": soft.get("by"), "soft": True, "q": q})
        try:
            m = await spec.finalize(f, self.t0 + last_end, close=False, by="semantic")
        except Exception as e:
            # a failed soft answer is "no soft answer": the ear's final answers the question normally
            soft["failed"] = True
            self.emit({"t": "answer_error", "error": f"{type(e).__name__}: {e}"[:300], "soft": True, "q": q})
            return None
        if m.get("superseded"):
            return None
        m["endpoint_by"] = "semantic"
        m["endpoint_trigger"] = soft.get("by")
        soft["metric"] = m
        return m

    # ------------------------------------------------------------------ endpoint
    def _close_question(self, ws: list[dict]) -> dict:
        """The ear closed an utterance: freeze what the answer needs and start listening for the next question at once
        (its partials may arrive while this one is still being answered)."""
        job = {"ws": ws, "spec": self.spec, "soft": self.soft, "final_task": self.final_task, "error": None,
               "director": self.director_out,
               "levels": list(self.q_levels), "q": self.qid}
        try:
            self._sync_words(ws)
            if not self.plain:
                self.tagger.endpoint()
                self._apply(self.tagger.poll())
            f = self._build()
            job.update(words=list(self.words), spans=list(self.spans), current=self.current, events=self.events,
                       prev_intent=self.prev_intent)
            self.emit({"t": "words", "words": self._words_json(),
                       "from": 0, "final": True, "ms": self.spec.ms()})
            self.emit({"t": "frame", **f.to_json(), "final": True, "spans": [
                {"type": s["type"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in self.spans],
                "cc_word": self.cc_at, "ms": self.spec.ms()})
        except Exception as e:
            job["error"] = e
        self.last_answered_end = max(self.last_answered_end, ws[-1].get("end") or 0.0)
        self.last_answered_words = [framemod.norm(w) for w in self.words]
        self.last_closed = {"ws": list(ws), "end": ws[-1].get("end") or 0.0, "job": job,
                            "trailing": framemod.unfinished([framemod.norm(clean(w["text"])) for w in ws])}
        self._new_question()
        return job

    async def on_final(self, ws: list[dict], early: bool = False) -> dict | None:
        """Answer a closed utterance. Finals are answered one at a time in arrival order; a final that arrives while
        another is answering is queued (returns None) and answered right after it by the call already running."""
        if self.closed:
            return None
        ws = self._join(self._fresh(ws or []), final=True)
        if not ws:
            return None
        if config.REPAIR_HOLD_S > 0 and not self.audio_ended and not self.plain and framemod.unfinished([framemod.norm(clean(w["text"])) for w in ws]):
            # the turn ended mid-correction ("... in Boston, sorry,") or mid-phrase ("is it bad to take a"): give the
            # rest a moment to arrive as the next turn before answering (it would be answered for Boston / for the
            # fragment). _join merges or releases it.
            h = self.held = {"ws": ws, "ev": asyncio.Event(), "merged": False, "job": None}
            self.emit({"t": "repair_hold", "ms": self.spec.ms(), "words": [framemod.norm(w["text"]) for w in ws]})
            try:
                await asyncio.wait_for(h["ev"].wait(), config.REPAIR_HOLD_S)
            except asyncio.TimeoutError:
                pass
            if self.held is h:
                self.held = None
            if h["merged"] or self.closed:
                return None                       # continued by the next turn: its final answers the whole question
            job = h["job"] or self._close_question(ws)
        else:
            job = self._close_question(ws)
        if self.busy:
            self.pending_finals.append(job)
            return None
        self.busy = True
        try:
            m = await self._answer(job)
            while self.pending_finals and not self.closed:
                await self._answer(self.pending_finals.popleft())
        finally:
            self.busy = False
        return m

    async def _answer(self, job: dict) -> dict:
        """Answer one closed question. Never leaves the session wedged: any failure becomes an answer_error event and a
        failure metric, and the question's speculator is always closed."""
        spec: Speculator = job["spec"]
        soft, final_task = job["soft"], job["final_task"]
        ws = job["ws"]
        last_end = ws[-1].get("end") or 0.0
        q = job.get("q", self.qid)
        m = None
        f = None
        if job.get("superseded"):                 # corrected before its turn came (_reopen)
            spec.close()
            return {"mode": self.mode, "question": " ".join(job.get("words") or []), "superseded": True}
        if self.plain:
            return await self._answer_plain(job, spec, last_end, q)
        try:
            if job["error"] is not None:
                raise job["error"]
            # build at answer time: an earlier question that finished meanwhile may have updated the carried entity
            learned = []
            if config.MEM_USER_FACTS:
                learned = self.mem.learn_user([framemod.norm(w) for w in job["words"]])
                if learned:
                    self.emit_memory(learned, why="user fact", q=q)
            f = framemod.build(job["words"], job["spans"], job["current"], job["events"], job["prev_intent"],
                               local_ok=bool(len(lanes.index())), carry=self._carry(job["words"]))
            if config.DIRECTOR_PII:
                director.apply_pii(f, job.get("director"))
            spec.context = self.context
            if learned and self.user_statement(f) and soft is None:
                return self._ack_user(job, spec, f, learned, last_end, q)
            why = self._not_request(f, job.get("director"), ws) if (self.open_mic and soft is None) else None
            if why:
                # open mic, not a request: heard and labelled, no search and no answer (not part of the conversation)
                self.emit({"t": "heard", "why": why, "ms": spec.ms(), "speech_end_ms": round(last_end * 1000),
                           "words": [framemod.norm(w) for w in job["words"]], "frame": f.to_json(), "q": q})
                return {"mode": self.mode, "question": " ".join(job["words"]), "skipped": why}
            if soft is not None and soft["sig"] == meaning(f):
                # the ear confirms what we already answered at the semantic endpoint
                if final_task is not None and not final_task.done():
                    await asyncio.wait({final_task})      # (a cancelled soft task must not look like OUR cancel)
                m = soft["metric"]                        # None if the soft answer failed or was superseded
            if m is None:
                if soft is not None:
                    # the speaker kept going after the semantic endpoint: answer the longer question from the same,
                    # still-warm speculator (its searches/drafts kept running on the later words)
                    self.emit({"t": "extended", "ms": spec.ms(), "words": [framemod.norm(w) for w in job["words"]],
                               "correction": f.corrected, "q": q})
                    if final_task is not None and not final_task.done():
                        final_task.cancel()
                self.emit({"t": "endpoint", "ms": spec.ms(), "speech_end_ms": round(last_end * 1000),
                           "frame": f.to_json(), "by": "ear", "q": q})
                m = await spec.finalize(f, self.t0 + last_end)
                m["endpoint_by"] = "ear"
                m["superseded_soft"] = soft is not None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            self.emit({"t": "answer_error", "error": err, "q": q})
            m = {"mode": self.mode, "question": " ".join(job.get("words") or [w["text"] for w in ws]),
                 "endpoint_by": "ear", "path": "failed", "error": err, "failed": True}
            self.emit({"t": "metric", **m, "q": q})
        finally:
            if final_task is not None and not final_task.done():
                final_task.cancel()
            spec.close()
        if job.get("superseded") or m.get("superseded"):
            # the speaker corrected this question while it was being answered: the corrected question answers instead
            m["superseded"] = True
            self.superseded.append(m)
            return m
        job["metric"] = m
        if not m.get("failed") and f is not None:
            self.last_request_end = max(self.last_request_end, last_end)
            self.last_request_speaker = (self.arc or {}).get("speaker")
            job["mem_before"] = (self.carry, self.context, self.carry_pool, self.mem.snapshot())
            self._remember(f, m, spec, q)
            if job.get("levels"):
                self.answered_level = float(np.median(job["levels"]))
            self.carry_pool = spec.export_pool()
            # the question already being heard started with the older pool / context: bring it up to date
            for s in [self.spec] + [j["spec"] for j in self.pending_finals]:
                s.seed(self.carry_pool)
                s.context = self.context
            self._refresh()
        self.history.append(m)
        return m

    async def _answer_plain(self, job: dict, spec, last_end: float, q: int) -> dict:
        """Arc-off arm (mode="plain"): the closed turn's words as they are. Every turn is answered (no open-mic gate,
        no user-fact acks, no repair merging); no conversation memory: the answer model gets no previous turn (unless
        PLAIN_CONTEXT=1) and nothing is carried into the next question's frame or pool. Never raises."""
        m = None
        try:
            if job["error"] is not None:
                raise job["error"]
            f = framemod.plain(job["words"])
            spec.context = self.context if config.PLAIN_CONTEXT else ""
            self.emit({"t": "endpoint", "ms": spec.ms(), "speech_end_ms": round(last_end * 1000),
                       "frame": f.to_json(), "by": "ear", "q": q})
            m = await spec.finalize(f, self.t0 + last_end)
            m["endpoint_by"] = "ear"
            m["superseded_soft"] = False
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            self.emit({"t": "answer_error", "error": err, "q": q})
            m = {"mode": self.mode, "question": " ".join(job.get("words") or [w["text"] for w in job["ws"]]),
                 "endpoint_by": "ear", "path": "failed", "error": err, "failed": True}
            self.emit({"t": "metric", **m, "q": q})
        finally:
            spec.close()
        job["metric"] = m
        if config.PLAIN_CONTEXT and not m.get("failed"):
            self.context = f"Q: {' '.join(job['words'])}\nA: {(m.get('answer') or '')[:300]}"
            for s_ in [self.spec] + [j["spec"] for j in self.pending_finals]:
                s_.context = self.context
        self.history.append(m)
        return m

    def _refresh(self):
        """Memory just changed (an answer landed): the question being heard may have a pronoun that only now
        resolves ("when was it ..." said while that answer streamed). Rebuild its frame and let the speculator act."""
        if self.closed or not self.words or self.mode != "headstart":
            return
        try:
            f = self._build()
            self.frame = f
            self._note_used(f)
            if not self._hold(f):
                self.spec.on_frame(f)
        except Exception as e:
            self.emit({"t": "error", "where": "memory refresh", "error": f"{type(e).__name__}: {e}"[:300]})

    def _ack_user(self, job: dict, spec, f, learned: list, last_end: float, q: int) -> dict:
        """A statement of a user fact ("I live in Denver"): remembered and acknowledged locally. No search, no model
        call; the fact stays on this machine and is used as a query word when a later question needs it."""
        t_end = self.t0 + last_end
        self.emit({"t": "endpoint", "ms": spec.ms(), "speech_end_ms": round(last_end * 1000), "frame": f.to_json(),
                   "by": "ear", "q": q})
        labels = [x["label"] for x in learned]
        text = "Noted: " + "; ".join(labels) + ". I'll use that when a question needs it (kept on this device)."
        self.emit({"t": "sources", "ms": spec.ms(), "docs": [], "q": q})
        self.emit({"t": "answer", "delta": text, "ms": spec.ms(), "q": q})
        now = time.perf_counter()
        m = {"mode": self.mode, "question": f.text, "query": "", "intent": f.intent, "endpoint_by": "ear",
             "path": "memory", "answer_by": "memory", "answer": text, "sources": [], "n_sources": 0,
             "ready_ms": round((now - t_end) * 1000), "sources_ms": round((now - t_end) * 1000),
             "done_ms": round((now - t_end) * 1000), "web_calls": 0, "paid_calls": 0, "credits": 0.0,
             "memory": {"learned": labels}}
        self.emit({"t": "metric", **{k: v for k, v in m.items() if k != "answer"}, "q": q})
        spec.close()
        job["metric"] = m
        self.history.append(m)
        return m

    def _remember(self, f, m, spec=None, q: int | None = None):
        """Conversation memory: entities of the question and of the answer, typed person vs thing (the one-turn
        carry), and the multi-turn entity memory (server/memory.py): typed entities, Wikidata QIDs / facts of the
        sources used, relations; then a free pre-warm of the likely follow-up subjects' facts."""
        person, thing = [], []
        ans_spans = []
        for t, txt in f.slots:
            if t in framemod.NUMBER_TYPES:
                continue
            (person if t == "person" else thing).append((t, txt))
        try:
            if self.ans_tagger.h is None:
                raise RuntimeError("session closed")     # never call into a destroyed native handle
            r = self.ans_tagger.tag_text(re.sub(r"\[\d+\]", "", m.get("answer") or ""))
            ans_spans = memmod.answer_entities(r.get("words") or [], r.get("spans", []),
                                               [framemod.norm(w) for w in f.words])
            if m.get("answer_by") == "card":
                ans_spans = []                 # a live-data card ("Low 2°C ...") names no entities worth keeping
            for sp in r.get("spans", []):
                if sp.get("family") != "entity" or not sp["type"].startswith("entity.mention"):
                    continue
                t = sp["type"].split(".")[-1]
                item = (t, framemod.norm(sp["text"]))
                if t == "person":
                    person.append(item)        # answer's people come after the question's
                elif not thing:
                    thing.append(item)
        except Exception:
            pass
        if config.LOOP_ASR_FIX and config.LOOP_ANSWER_FIX and ans_spans:
            f2 = self._answer_fix(f, ans_spans)
            if f2 is not f:
                old = {(t, x) for t, x in f.slots} - set(f2.slots)
                person = [p for p in person if p not in old]
                thing = [p for p in thing if p not in old]
                for t, x in set(f2.slots) - set(f.slots):
                    if t not in framemod.NUMBER_TYPES:
                        (person if t == "person" else thing).insert(0, (t, x))
                f = f2
        q_ents = [txt for t, txt in f.slots if t not in framemod.NUMBER_TYPES]
        if person or thing:
            self.carry = {"person": person, "thing": thing}
        # for elliptical follow-ups ("and when?", "what about Nvidia?"): the previous question's meaning
        self.carry = {**(self.carry or {}), "prev": {"content": list(f.content), "entities": q_ents,
                                                     "intent": f.intent, "question": f.kept_text, "inline": f.inline}}
        # the corrected-away subject must not reach the next turn's prompt or its follow-up swap (ARC_INTEGRATION.md)
        self.context = f"Q: {f.clean_text}\nA: {(m.get('answer') or '')[:300]}"
        if not config.MEM_V2:
            return
        try:
            docs = [spec.pool[i] for i in spec.used_ids if i in spec.pool] if spec is not None else []
            t = self.mem.observe(f, m.get("answer") or "", ans_spans, docs)
            self.mem.note_used(framemod.norm(f.carried) if f.carried else None)
            if config.INDEX:                     # Keystone: memory + the answer into the one structure (server/keystone)
                from .keystone import hook as _ks
                _ks.note_answer(f, f.query(), m.get("answer") or "", (m.get("verify") or {}).get("status"))
                _ks.absorb_memory(self.mem)
            m["memory"] = {"turn": t, "used": f.carried or None, "ref": f.mem_ref or None, "inline": f.inline or None,
                           "user": [k for k, _ in f.user_used]}
            self.emit_memory(why="turn", q=q)
            if config.MEM_PREWARM and self.mode == "headstart":
                self.prewarm_task = asyncio.create_task(self._prewarm())
        except Exception as e:
            self.emit({"t": "error", "where": "memory", "error": f"{type(e).__name__}: {e}"[:300]})

    def _answer_fix(self, f, ans_spans):
        """The loop after the answer: the answer's names re-resolve the question's misheard ones ("who was mari guri"
        answered "Marie Curie was ...": memory keeps Marie Curie, so "where was she born" finds her). Returns a fixed
        copy of the frame, or f itself."""
        try:
            pool = heard.Pool()
            for t, txt in ans_spans:
                pool.add(memmod._title(txt), "answer", heard.TYPE_KIND.get(t, ""))
            fixes = heard.propose(f, list(f.sconf), pool, use_leads=False)
            if not fixes:
                return f
            import copy
            f2 = heard.apply(copy.copy(f), fixes)
            for fx in fixes:
                self.emit({"t": "heard_fix", "ms": self.spec.ms(), "after_answer": True, **fx.to_json()})
            return f2
        except Exception as e:
            self.emit({"t": "error", "where": "heard", "error": f"{type(e).__name__}: {e}"[:300]})
            return f

    async def _prewarm(self):
        t = time.perf_counter()
        subs = await memmod.prewarm(self.mem)
        if subs and config.INDEX:                # Keystone: the warmed 1-hop Wikidata neighbourhoods -> typed edges
            from .keystone import hook as _ks
            for x in subs:
                _ks.absorb_wikidata(x)
        if subs and not self.closed:
            self.emit({"t": "prewarm", "subjects": subs, "took": round((time.perf_counter() - t) * 1000),
                       "ms": self.spec.ms()})

    def export_memory(self) -> dict:
        """Conversation memory to carry into the next run on the same connection (push-to-talk turns)."""
        return {"carry": self.carry, "context": self.context, "carry_pool": self.carry_pool, "at": time.time(),
                "mem": self.mem.to_json()}

    def import_memory(self, mem: dict | None):
        if mem and mem.get("mem"):
            self.mem = memmod.Memory.from_json(mem["mem"])     # (its own TTLs: conversation vs user facts)
        if not mem or time.time() - mem.get("at", 0) > config.MEMORY_TTL_S:
            return
        self.carry, self.context, self.carry_pool = mem["carry"], mem["context"], mem["carry_pool"]
        if self.carry_pool:
            self.spec.seed(self.carry_pool)
        self.spec.context = self.context

    def end_of_audio(self):
        """No more audio is coming (the user released push-to-talk, the replay ended): a held final has no
        continuation to wait for, answer it now."""
        self.audio_ended = True
        if self.held is not None:
            self.held["ev"].set()
            self.held = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.prewarm_task is not None and not self.prewarm_task.done():
            self.prewarm_task.cancel()
        if self.held is not None:
            self.held["ev"].set()                         # (its on_final sees closed and returns)
            self.held = None
        if (self.soft is not None and self.soft.get("metric") is not None and not self.busy
                and not self.history):
            # the ear never confirmed (stream ended): keep it only if it's the only answer — an unconfirmed soft
            # answer after a confirmed one is usually trailing chatter
            self.history.append(self.soft["metric"])
            self.soft = None
        if self.final_task is not None and not self.final_task.done():
            self.final_task.cancel()                      # an unconfirmed soft answer must not outlive the session
        self.spec.close()
        for j in self.pending_finals:
            j["spec"].close()
        self.pending_finals.clear()
        self.tagger.close()
        self.ans_tagger.close()
