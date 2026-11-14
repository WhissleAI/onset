"""Arc audio metadata on a cloud ear's words (ARC_META=mini|transcriber|off).

The words come from Deepgram (Flux or nova-3); Arc listens to the same audio in external-transcript mode and runs its
heads only (no transcription decoder, no builtin fallback decoder): device-directed ("for me"), command_complete
(continuous probability and its event), act / intent of the open utterance, speaker id + role, bio events (breath,
throat clear, ...), and, with Arc v1.1, audio events.

  mini         Arc v1 Mini (arc-v1 1.0.1, the model already loaded for the Arc ear / fallback: no extra RAM).
               Measured on this M3 Pro with real Flux words: ~10 ms of CPU per 100 ms chunk (RTF ~0.1),
               command_complete ~0.2 s after the last word.
  transcriber  Arc Transcriber (Arc v1.1 "full": NVIDIA Parakeet-TDT 0.6B encoder tap 12 + Arc adapters), code from the
               arc_transcriber package (1.2.1, loaded under another module name so it doesn't clash with arc-v1). Its
               backbone streams in chunks (ARC_T_CHUNK_S + ARC_T_CTX_S of context each side): on MPS with 0.5 s chunks
               RTF ~0.25 but metadata lands ~1.2-1.4 s after the last word and ~3.1 GB RSS (PROGRESS.md). Too late to
               steer endpointing; read-out only.
  off          nothing.

Everything runs on one dedicated thread per ArcMeta (audio + words are applied in order); results come back to the
asyncio loop as summaries via on_meta(meta)."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config

CC = "discourse.assistant.command_complete"
_t_arc = None
_t_lock = threading.Lock()


def transcriber():
    """Arc Transcriber (v1.1 full) via the arc_transcriber package, imported as `arc_audio_t`."""
    global _t_arc
    with _t_lock:
        if _t_arc is None:
            name = "arc_audio_t"
            if name not in sys.modules:
                pkg = config.ARC_T_CODE / "arc_audio"
                spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py",
                                                              submodule_search_locations=[str(pkg)])
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            Arc = sys.modules[name].Arc
            a = Arc.from_pretrained(str(config.ARC_T_WEIGHTS), backbone="full", device=config.ARC_T_DEVICE,
                                    threads=config.ARC_THREADS, load_decoder=False)
            a.cfg["backbone"]["stream_chunk_s"] = config.ARC_T_CHUNK_S
            a.cfg["backbone"]["stream_context_s"] = config.ARC_T_CTX_S
            _t_arc = a
        return _t_arc


def ready() -> bool:
    """The model ArcMeta needs is already loaded (building a stream is then instant)."""
    if variant() == "transcriber":
        return _t_arc is not None
    from . import ear
    return ear._arc is not None


def variant() -> str:
    v = (config.ARC_META or "off").strip().lower()
    return v if v in ("mini", "transcriber") else "off"


def warmup():
    if variant() == "transcriber":
        t = time.perf_counter()
        m = ArcMeta(lambda e: None)
        for _ in range(15):
            m._feed(np.zeros(1600, np.float32))
        m._flush()
        return time.perf_counter() - t
    return 0.0


def _top(xs):
    if not xs:
        return None
    x = xs[0]
    return {"label": x.get("label"), "p": x.get("p")}


class ArcMeta:
    """Feed it the ear's audio and words; it calls on_meta(summary) whenever the summary changes."""

    def __init__(self, on_meta, domain: str = "assistant", offset_s: float = 0.0, state=None):
        self.on_meta = on_meta
        from . import arcstate
        self.state = state if (state is not None and arcstate.ENABLED) else None
        voice = self.state is not None and arcstate.VOICE
        self.v = variant()
        if self.v == "transcriber":
            a = transcriber()
        else:
            from .ear import arc
            a = arc()
        # fallback=None: external mode must not quietly run Arc's builtin decoder as a name fallback
        kw = {"diarization": "voice"} if voice else {}     # ArcState: Arc's voice embedding per frame (arcstate.Tap)
        try:
            self.stream = a.stream(domain=domain, transcript="external", name_source="word", interim_source="deepgram",
                                   partials=False, fallback=None, **kw)
        except Exception:  # noqa: BLE001 (a model without the voice head: metadata as before, no voice state)
            if not kw:
                raise
            voice = False
            self.stream = a.stream(domain=domain, transcript="external", name_source="word",
                                   interim_source="deepgram", partials=False, fallback=None)
        self.tap = None
        if voice:
            try:
                self.tap = arcstate.Tap(self.stream, offset_s)
                if not self.tap.ok:
                    self.tap = None
                self.state.new_stream()
            except Exception:  # noqa: BLE001 (state is optional)
                self.tap = None
        self.n_feed = 0
        self.offset = float(offset_s)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arcmeta")
        self.loop = None
        self.closed = False
        self.last_key = None
        self.cc_events: list[dict] = []
        self.compute_s = 0.0
        self.audio_s = 0.0
        self.pending = 0
        self.lag_drops = 0
        # director v2 streams (config.DIRECTOR_STREAMS / DIRECTOR_ARC_EAR; off by default)
        self.extra = ArcExtra() if (config.DIRECTOR_STREAMS or config.DIRECTOR_ARC_EAR) else None
        self.ear_stream = None
        if config.DIRECTOR_ARC_EAR and self.v != "transcriber":
            self.ear_stream = a.stream(domain=domain, transcript="builtin", partials=True)
        self.aw: list = []
        self.af: list = []

    # ------------------------------------------------------------------ in (from the ear, on the event loop)
    def feed(self, pcm: np.ndarray):
        if self.closed:
            return
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        if self.pending > config.ARC_META_MAX_PENDING:     # the model can't keep up: drop audio, never block the ear
            self.lag_drops += 1
            return
        self.pending += 1
        self.pool.submit(self._run, self._feed, pcm)

    def words(self, ws: list[dict], final: bool):
        """ws: [{text,start,end}] on the audio clock (Flux's punctuated words are fine: Arc reads them as text)."""
        if self.closed or not ws:
            return
        ev = [{"text": w["text"], "start": float(w["start"]) - self.offset, "end": float(w["end"]) - self.offset,
               "is_final": final} for w in ws if w.get("start") is not None and float(w["start"]) >= self.offset]
        if not ev:
            return
        self.pool.submit(self._run, self.stream.add_words, ev)

    async def flush(self):
        if self.closed:
            return
        fut = asyncio.get_running_loop().run_in_executor(self.pool, self._run, self._flush)
        try:
            await asyncio.wait_for(fut, timeout=3.0)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ on the Arc thread
    def _run(self, fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001 (a metadata failure must never touch the words path)
            self._emit({"error": f"{type(e).__name__}: {e}"[:200]})

    def _feed(self, pcm):
        t = time.perf_counter()
        out = self.stream.feed(pcm)
        if self.tap is not None:
            self.n_feed += 1
            self.state.add_frames(self.tap.drain(self.offset))
            if self.n_feed % 5 == 0:
                self.state.set_emotion(self.tap.emotion())
        if self.extra is not None:
            self._feed_extra(pcm)
        self.compute_s += time.perf_counter() - t
        self.audio_s += len(pcm) / 16000
        self.pending = max(0, self.pending - 1)
        self._summarize(out)

    def _feed_extra(self, pcm):
        """Director v2: Arc's unshipped heads + level (and the Arc ear's own words) -> on_meta({"x2_only": ...})."""
        o = self.offset
        if config.DIRECTOR_STREAMS:
            self.extra.step(self.stream, pcm)
        if self.ear_stream is not None:
            ob = self.ear_stream.feed(pcm)
            for fu in ob.get("transcript") or []:
                self.af.append([fu.get("text", ""), None if fu.get("start") is None else fu["start"] + o,
                                None if fu.get("end") is None else fu["end"] + o])
                self.af = self.af[-3:]
                self.aw = []
            p = ob.get("transcript_partial")
            if p is not None:
                self.aw = [] if p.get("final") else [
                    (w["text"], None if w.get("start") is None else w["start"] + o,
                     None if w.get("end") is None else w["end"] + o) for w in (p.get("words") or [])]
        self._emit({"x2_only": True, "x2": self.extra.state() if config.DIRECTOR_STREAMS else None,
                    "aw": [list(w) for w in self.aw] if self.ear_stream is not None else None,
                    "af": list(self.af) if self.ear_stream is not None else None})

    def _flush(self):
        out = self.stream.flush()
        if self.tap is not None:
            self.state.add_frames(self.tap.drain(self.offset))
            self.state.set_emotion(self.tap.emotion())
        self._summarize(out, final=True)

    def _summarize(self, out: dict, final: bool = False):
        o = self.offset
        for e in out.get("events") or []:
            if e.get("tag") == CC:
                self.cc_events.append({"t": None if e.get("t") is None else round(e["t"] + o, 2), "p": e.get("p")})
        lv = out.get("live") or {}
        u = lv.get("utt") or {}
        if u:
            self.last_utt = u                  # the read-out keeps the last utterance's values after it closes
        else:
            u = getattr(self, "last_utt", None) or {}
        sp = lv.get("speaker") or {}
        if sp.get("id"):
            self.last_sp = sp
        else:
            sp = getattr(self, "last_sp", None) or {}
        role = _top(sp.get("role"))
        m = {
            "model": self.v, "at": None if lv.get("t") is None else round(lv["t"] + o, 2),
            "for_me": None if u.get("device_directed") is None else round(float(u["device_directed"]), 2),
            "cc_p": None if u.get("command_complete") is None else round(float(u["command_complete"]), 2),
            "cc_events": self.cc_events[-3:],
            "act": _top(u.get("act")), "intent": _top(u.get("intent")),
            "speaker": sp.get("id"), "role": (role or {}).get("label"),
            "bio": [{"tag": b.get("tag"), "t": b.get("t")} for b in (out.get("bio_events") or [])],
            "audio_events": [{"tag": a.get("tag") or a.get("label"), "p": a.get("p")} for a in
                             (out.get("audio_events") or [])],
            "utt_open": bool(lv.get("utt")),
            "rtf": round(self.compute_s / self.audio_s, 3) if self.audio_s else None,
        }
        if final:
            m["final"] = True
        key = (m["for_me"], m["cc_p"], len(self.cc_events), (m["act"] or {}).get("label"), m["speaker"], m["role"],
               bool(m["bio"]), bool(m["audio_events"]), m["utt_open"], final)
        if key == self.last_key:
            return
        self.last_key = key
        self._emit(m)

    def _emit(self, m):
        if self.loop is not None and not self.closed:
            self.loop.call_soon_threadsafe(self.on_meta, m)


# ---------------------------------------------------------------------------------------------- director v2 read-out
# Arc Mini computes every head on every frame, also the ones the assistant domain doesn't ship (overlap, emotion,
# arousal, speaker change, statement / backchannel acts): uncalibrated for this domain, but a learned reader can use
# them. ArcExtra keeps a short ring of those per-frame values + the audio level, and state() is what the search
# director (server/director.py, DIRECTOR_V2) reads at each Flux update. Training replays it the same way
# (eval/conf_features.py --v2), so serving and training compute the same numbers.
X_FRAME = {"speech": "meta.speech", "sil": "scene.silence", "ovl": "meta.overlap", "degr": "meta.degraded_audio",
           "music": "scene.music", "laugh": "scene.laughter", "noneng": "meta.lang.non_en"}
X_UTT = {"st": "discourse.act.statement", "q": "discourse.act.question", "op": "discourse.act.opinion_assessment",
         "cr": "discourse.act.command_request", "bc": "discourse.backchannel", "ans": "discourse.answer",
         "dd": "discourse.assistant.device_directed", "e_neu": "affect.emotion.neutral",
         "e_hap": "affect.emotion.happy", "e_ang": "affect.emotion.angry", "e_sad": "affect.emotion.sad",
         "e_sur": "affect.emotion.surprised", "a_lo": "affect.arousal.low", "a_hi": "affect.arousal.high"}
X_EV = {"chg": "speaker.change", "intr": "discourse.interruption", "yld": "discourse.turn_yield"}
X_RING = 10                       # frames (100 ms each) for the "recent max" values


class ArcExtra:
    """Per-frame Arc state for the director. step(stream, pcm) after every stream.feed(pcm) (one 100 ms chunk)."""

    def __init__(self):
        self.cols = None
        self.ring: list[dict] = []
        self.levels: list[float] = []
        self.utt = None
        self.nspk = 0
        self.spk = -1
        self.spk_turn = None

    def _cols(self, eng):
        R, tab = eng.R, eng.tab
        out = {}
        for grp, tags in (("frame", X_FRAME), ("utt", X_UTT), ("ev_at", X_EV)):
            cs = R.cols.get(grp) or []
            for k, tag in tags.items():
                i = tab.idx.get(tag)
                out[k] = (grp, cs.index(i)) if i is not None and i in cs else None
        return out

    def step(self, stream, pcm):
        eng = getattr(stream, "eng", None)
        x = np.asarray(pcm, np.float32)
        self.levels.append(float(10 * np.log10(np.mean(x * x) + 1e-10)) if len(x) else -100.0)
        if len(self.levels) > 600:
            self.levels = self.levels[-600:]
        last = getattr(eng, "last", None)
        if eng is None or not last:
            return
        if self.cols is None:
            self.cols = self._cols(eng)
        fr = {}
        for k, c in self.cols.items():
            if c is None:
                continue
            g, j = c
            if g != "utt" and g in last:
                fr[k] = float(last[g][j])
        self.ring.append(fr)
        if len(self.ring) > X_RING:
            self.ring.pop(0)
        up = getattr(eng, "utt_p", None)
        if up is not None:
            self.utt = {k: float(up[c[1]]) for k, c in self.cols.items() if c is not None and c[0] == "utt"}
        self.nspk = len(getattr(eng, "speakers", {}) or {})
        self.spk = int(getattr(eng, "last_spk", -1))

    def state(self) -> dict:
        r = self.ring[-1] if self.ring else {}
        mx = lambda k: max((f.get(k, 0.0) for f in self.ring), default=0.0)  # noqa: E731
        lv = self.levels
        ref = float(np.percentile(lv, 90)) if len(lv) >= 3 else (lv[-1] if lv else -100.0)
        s = {"speech": r.get("speech"), "sil": r.get("sil"), "ovl": mx("ovl"), "degr": r.get("degr"),
             "music": mx("music"), "laugh": mx("laugh"), "noneng": r.get("noneng"), "chg": mx("chg"),
             "intr": mx("intr"), "yld": r.get("yld"), "nspk": self.nspk, "spk": self.spk,
             "lvl": lv[-1] if lv else None, "lvl_ref": ref, "lvl3": float(np.mean(lv[-3:])) if lv else None}
        s["utt"] = dict(self.utt) if self.utt else None
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} | (
            {"utt": {k: round(v, 3) for k, v in self.utt.items()}} if self.utt else {})
