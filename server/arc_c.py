"""ctypes binding to libarc (Arc Mini in C11: no torch, no onnxruntime), vendored in vendor/arc.

libarc runs Arc Mini (WhissleAI/arc-v1.1 `mini/`, 4-bit weights, memory-mapped) from the release folder as published:
transcript with word times, utterance read-out (device-directed, command_complete, intent, act), sound events and
per-frame probabilities, offline (arc_transcribe) or streaming (arc_stream: PARTIAL / FINAL / TAGS / FRAME records).
vendor/arc carries one small patch (arc_stream_set_words): word times on streamed partials and finals, which the ear
needs (every session speaks words with start / end on the audio clock), and partials decoded from a fresh encode of
their window as arc_audio 1.0.1 did (upstream decodes the streamed encoder states, which show words ~0.25 s later).

  Engine(model_dir, threads)               one per thread: an engine runs one call at a time
  Engine.stream(domain, mode="full")       Stream: feed(pcm) / flush() -> {"transcript", "transcript_partial",
                                           "events", "frames", "live"}: the keys server/ear.py and server/arcmeta.py
                                           read from arc_audio's Stream, in the same shapes
  Engine.transcribe(pcm, domain)           a whole clip (server/userindex.py): utterances with words, names, numbers,
                                           events, speakers

Loading costs ~40-90 ms (weights are mmap'd), so streams get their own engine and never share one across threads.
"""
from __future__ import annotations

import ctypes as C
import json
import threading
from pathlib import Path

import numpy as np

from . import modelpaths

ABI = 2
CC = "discourse.assistant.command_complete"

c_float_p = C.POINTER(C.c_float)
c_str_p = C.POINTER(C.c_char_p)


class Tag(C.Structure):
    _fields_ = [("tag", C.c_char_p), ("word_begin", C.c_int), ("word_end", C.c_int), ("text", C.c_char_p)]


class Word(C.Structure):
    _fields_ = [("text", C.c_char_p), ("start", C.c_float), ("end", C.c_float), ("tag", C.c_char_p),
                ("tag_begin", C.c_int), ("conf", C.c_float), ("disfl", C.c_char_p), ("n_alts", C.c_int),
                ("alts", c_str_p), ("alt_p", c_float_p)]


class Label(C.Structure):
    _fields_ = [("label", C.c_char_p), ("p", C.c_float)]


class Utt(C.Structure):
    _fields_ = [("start", C.c_float), ("end", C.c_float), ("text", C.c_char_p), ("n_tags", C.c_int),
                ("tags", C.POINTER(Tag)), ("device_directed", C.c_float), ("command_complete", C.c_float),
                ("intent", Label), ("act", Label), ("directed_head", C.c_float), ("n_words", C.c_int),
                ("words", C.POINTER(Word)), ("n_alts", C.c_int), ("alts", c_str_p), ("alt_scores", c_float_p),
                ("decoder_text", C.c_char_p), ("directed", C.c_int), ("conf", C.c_float), ("clean_text", C.c_char_p),
                ("n_quantities", C.c_int), ("quantities", C.c_void_p), ("n_dropped", C.c_int),
                ("playback_leak", C.c_float), ("playback", C.c_int), ("tagger_json", C.c_char_p)]


class Name(C.Structure):
    _fields_ = [("start", C.c_float), ("end", C.c_float), ("tag", C.c_char_p), ("heard", C.c_char_p),
                ("entity", C.c_char_p), ("kb_index", C.c_int), ("state", C.c_char_p), ("p", C.c_float),
                ("model_entity", C.c_char_p), ("model_p", C.c_float), ("src", C.c_char_p), ("via", C.c_char_p),
                ("snap_score", C.c_float), ("span_p", C.c_float), ("show", C.c_int), ("n_words", C.c_int),
                ("words", C.POINTER(Word)), ("candidate", C.c_char_p), ("candidate_kb_index", C.c_int),
                ("candidate_p", C.c_float)]


class Number(C.Structure):
    _fields_ = [("start", C.c_float), ("end", C.c_float), ("kind", C.c_char_p), ("unit", C.c_char_p),
                ("heard", C.c_char_p), ("has_value", C.c_int), ("has_value2", C.c_int), ("value", C.c_double),
                ("value2", C.c_double), ("value_int", C.c_int), ("value2_int", C.c_int), ("state", C.c_char_p),
                ("kind_head", C.c_char_p), ("has_value_model", C.c_int), ("value_model", C.c_double)]


class Event(C.Structure):
    _fields_ = [("label", C.c_char_p), ("start", C.c_float), ("end", C.c_float), ("p", C.c_float)]


class Result(C.Structure):
    _fields_ = [("n_utts", C.c_int), ("utts", C.POINTER(Utt)), ("n_events", C.c_int), ("events", C.POINTER(Event)),
                ("n_names", C.c_int), ("names", C.POINTER(Name)), ("n_numbers", C.c_int),
                ("numbers", C.POINTER(Number)), ("audio_s", C.c_float), ("wall_ms", C.c_float)]


class Options(C.Structure):
    _fields_ = [("domain", C.c_char_p), ("one_utterance", C.c_int), ("beams", C.c_int), ("trunk", C.c_int),
                ("events", C.c_int), ("bias_names", C.c_int), ("words", C.c_int), ("names", C.c_int),
                ("numbers", C.c_int)]


class SpeakerTurn(C.Structure):
    _fields_ = [("speaker", C.c_char_p), ("start", C.c_float), ("end", C.c_float)]


class SpeakerGroup(C.Structure):
    _fields_ = [("group", C.c_char_p), ("label", C.c_char_p), ("p", C.c_float)]


class Speaker(C.Structure):
    _fields_ = [("id", C.c_char_p), ("last_seen", C.c_float), ("n_groups", C.c_int),
                ("groups", C.POINTER(SpeakerGroup)), ("n_tags", C.c_int), ("tags", C.POINTER(Label))]


class Frame(C.Structure):
    _fields_ = [("t", C.c_float), ("speech", C.c_float), ("in_utterance", C.c_int), ("device_directed", C.c_float),
                ("command_complete", C.c_float), ("n_labels", C.c_int), ("labels", c_str_p), ("p", c_float_p)]


class StreamOut(C.Structure):
    _fields_ = [("kind", C.c_int), ("id", C.c_int), ("utt", C.POINTER(Utt)), ("tags_final", C.c_int),
                ("event", C.POINTER(Event)), ("frame", C.POINTER(Frame)), ("audio_s", C.c_double)]


STREAM_FN = C.CFUNCTYPE(None, C.POINTER(StreamOut), C.c_void_p)


class StreamConfig(C.Structure):
    _fields_ = [("mode", C.c_int), ("domain", C.c_char_p), ("on_output", STREAM_FN), ("user", C.c_void_p),
                ("frames", C.c_int), ("events", C.c_int), ("partial_ms", C.c_int), ("partial_window_s", C.c_float),
                ("beams", C.c_int), ("bias_names", C.c_int), ("word_wait_s", C.c_float), ("delay_ms", C.c_int)]


class ExtWord(C.Structure):
    _fields_ = [("text", C.c_char_p), ("start", C.c_float), ("end", C.c_float)]


FULL, HEADS, EXTERNAL = 0, 1, 2
PARTIAL, FINAL, TAGS, EVENT, FRAME, ENDPOINT = 1, 2, 3, 4, 5, 6
NO_DECODER = 1

_lib = None
_lib_lock = threading.Lock()


def lib_path() -> Path:
    return modelpaths.arc_lib()


def lib():
    global _lib
    with _lib_lock:
        if _lib is None:
            L = C.CDLL(str(lib_path()))
            L.arc_abi_version.restype = C.c_int
            L.arc_version.restype = C.c_char_p
            if L.arc_abi_version() != ABI:
                raise RuntimeError(f"libarc ABI {L.arc_abi_version()} at {lib_path()}, this binding speaks {ABI}")
            if not hasattr(L, "arc_stream_set_words"):
                raise RuntimeError(f"{lib_path()} lacks arc_stream_set_words: build it from vendor/arc "
                                   "(scripts/fetch_models.py build-arc)")
            vp, sz = C.c_void_p, C.c_size_t
            L.arc_create_ex.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_char_p, sz]
            L.arc_create_ex.restype = vp
            L.arc_destroy.argtypes = [vp]
            if hasattr(L, "arc_set_lm"):                     # n-gram LM fusion (libarc with arc_lm.c)
                L.arc_set_lm.argtypes = [vp, C.c_char_p, C.c_float, C.c_int]
                L.arc_set_lm.restype = C.c_int
                L.arc_set_lm_weight.argtypes = [vp, C.c_float]
                L.arc_set_lm_weight.restype = C.c_int
                L.arc_lm_weight.argtypes = [vp]
                L.arc_lm_weight.restype = C.c_float
            L.arc_set_threads.argtypes = [vp, C.c_int]
            L.arc_last_error.restype = C.c_char_p
            L.arc_transcribe.argtypes = [vp, c_float_p, sz, C.POINTER(Options)]
            L.arc_transcribe.restype = C.POINTER(Result)
            L.arc_result_free.argtypes = [C.POINTER(Result)]
            L.arc_speakers.argtypes = [C.POINTER(Result), C.POINTER(C.POINTER(Speaker))]
            L.arc_diarization.argtypes = [C.POINTER(Result), C.POINTER(C.POINTER(SpeakerTurn))]
            L.arc_utt_speaker.argtypes = [C.POINTER(Result), C.c_int]
            L.arc_utt_speaker.restype = C.c_char_p
            L.arc_stream_open.argtypes = [vp, C.POINTER(StreamConfig)]
            L.arc_stream_open.restype = vp
            L.arc_stream_push.argtypes = [vp, c_float_p, sz]
            L.arc_stream_words.argtypes = [vp, C.POINTER(ExtWord), C.c_int, C.c_int]
            L.arc_stream_covered.argtypes = [vp, C.c_float]
            L.arc_stream_end_turn.argtypes = [vp]
            L.arc_stream_finish.argtypes = [vp]
            L.arc_stream_close.argtypes = [vp]
            L.arc_stream_set_words.argtypes = [vp, C.c_int]
            L.arc_stream_speakers.argtypes = [vp, C.POINTER(C.POINTER(Speaker))]
            L.arc_stream_speaker.argtypes = [vp, C.c_float, C.c_float]
            L.arc_stream_speaker.restype = C.c_char_p
            _lib = L
        return _lib


def version() -> str:
    return lib().arc_version().decode()


def _s(b) -> str | None:
    return None if b is None else b.decode("utf-8", "replace")


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def _p(x: float):
    return None if x < 0 else _r(x)


def _words(ws, n: int) -> list[dict]:
    out = []
    for i in range(n):
        w = ws[i]
        d = {"text": _s(w.text), "start": _r(w.start), "end": _r(w.end)}
        if w.tag:
            d["tag"] = _s(w.tag)
            d["tag_begin"] = bool(w.tag_begin)
        out.append(d)
    return out


def _utt(u: Utt) -> dict:
    d = {"start": _r(u.start, 2), "end": _r(u.end, 2), "text": _s(u.text) or "",
         "words": _words(u.words, u.n_words),
         "tags": [{"tag": _s(u.tags[i].tag), "word_begin": u.tags[i].word_begin, "word_end": u.tags[i].word_end,
                   "text": _s(u.tags[i].text)} for i in range(u.n_tags)]}
    for k in ("device_directed", "command_complete", "directed_head"):
        d[k] = _p(getattr(u, k))
    for k in ("intent", "act"):
        lab = getattr(u, k)
        d[k] = [{"label": _s(lab.label), "p": _r(lab.p)}] if lab.label else []
    return d


def _speakers(L, getter, h) -> list[dict]:
    sp = C.POINTER(Speaker)()
    n = getter(h, C.byref(sp))
    out = []
    for i in range(max(0, n)):
        s = sp[i]
        out.append({"id": _s(s.id), "last_seen": _r(s.last_seen, 2),
                    "groups": {_s(s.groups[j].group): {"label": _s(s.groups[j].label), "p": _r(s.groups[j].p)}
                               for j in range(s.n_groups)},
                    "tags": [{"label": _s(s.tags[j].label), "p": _r(s.tags[j].p)} for j in range(s.n_tags)]})
    return out


def _f32(pcm) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(pcm, np.float32).reshape(-1))


def model_config(model_dir) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text())


class Engine:
    """One libarc engine. Not thread-safe: one call at a time (streams get their own engine, see Engine.stream)."""

    def __init__(self, model_dir, threads: int = 2, decoder: bool = True):
        self.L = lib()
        self.dir = str(model_dir)
        self.threads = int(threads)
        self.decoder = bool(decoder)
        err = C.create_string_buffer(512)
        h = self.L.arc_create_ex(self.dir.encode(), self.threads, 0 if decoder else NO_DECODER, err, 512)
        if not h:
            raise RuntimeError(f"arc_create failed for {self.dir}: {err.value.decode(errors='replace')}")
        self.h = h
        self._cfg = None

    @property
    def cfg(self) -> dict:
        if self._cfg is None:
            self._cfg = model_config(self.dir)
        return self._cfg

    def threshold(self, tag: str, q: str, domain: str, default: float = 0.5) -> float:
        return float(((self.cfg.get("thresholds") or {}).get(f"{tag}@{q}") or {}).get(domain, default))

    def set_lm(self, path, weight: float, topk: int = 0) -> bool:
        """Shallow-fuse a token n-gram LM (ARCLM file) into this engine's decoder; path None / weight <= 0 clears it.
        False when this libarc has no LM support or the file does not load."""
        if not hasattr(self.L, "arc_set_lm"):
            return False
        ok = self.L.arc_set_lm(self.h, str(path).encode() if path else None, float(weight), int(topk)) == 0
        if not ok:
            import logging
            logging.getLogger(__name__).warning("arc_set_lm(%s): %s", path, _s(self.L.arc_last_error()))
        return ok

    def lm_weight(self) -> float:
        return float(self.L.arc_lm_weight(self.h)) if hasattr(self.L, "arc_lm_weight") else 0.0

    def close(self):
        if getattr(self, "h", None):
            self.L.arc_destroy(self.h)
            self.h = None

    __del__ = close

    def stream(self, domain: str = "assistant", mode: str = "full", partial_ms: int = 400, words: bool = True,
               frames: bool = True, events: bool = True, speakers: bool = True, partial_encode: bool = True,
               partial_lm: bool = True,
               own: bool = False) -> "Stream":
        """A stream on THIS engine (the stream holds the engine's call slot until it is closed). own=True: closing the
        stream also destroys the engine (an engine made for this one stream)."""
        s = Stream(self, domain=domain, mode=mode, partial_ms=partial_ms, words=words, frames=frames, events=events,
                   speakers=speakers, partial_encode=partial_encode, partial_lm=partial_lm)
        s.own = own
        return s

    def transcribe(self, pcm, domain: str = "unknown", words: bool = True, numbers: bool = True, names: bool = True,
                   events: bool = True) -> dict:
        """A whole clip -> {"transcript": [utt...], "names", "numbers", "events", "diarization", "speakers"} with times
        in seconds of the clip (the keys server/moments.py reads from arc_audio's predict)."""
        x = _f32(pcm)
        o = Options(domain=domain.encode(), trunk=1, events=int(events), words=int(words), names=int(names),
                    numbers=int(numbers))
        r = self.L.arc_transcribe(self.h, x.ctypes.data_as(c_float_p), len(x), C.byref(o))
        if not r:
            raise RuntimeError(f"arc_transcribe: {_s(self.L.arc_last_error())}")
        try:
            R = r.contents
            utts = []
            for i in range(R.n_utts):
                u = _utt(R.utts[i])
                u["speaker"] = _s(self.L.arc_utt_speaker(r, i))
                utts.append(u)
            names_ = [{"start": _r(n.start, 2), "end": _r(n.end, 2), "tag": _s(n.tag), "heard": _s(n.heard),
                       "entity": _s(n.entity), "state": _s(n.state), "p": _p(n.p), "show": bool(n.show)}
                      for n in (R.names[i] for i in range(R.n_names))]
            nums = [{"start": _r(n.start, 2), "end": _r(n.end, 2), "kind": _s(n.kind), "unit": _s(n.unit),
                     "heard": _s(n.heard), "value": n.value if n.has_value else None,
                     "value2": n.value2 if n.has_value2 else None, "state": _s(n.state), "kind_head": _s(n.kind_head)}
                    for n in (R.numbers[i] for i in range(R.n_numbers))]
            evs = [{"label": _s(e.label), "start": _r(e.start, 2), "end": _r(e.end, 2), "p": _r(e.p)}
                   for e in (R.events[i] for i in range(R.n_events))]
            tp = C.POINTER(SpeakerTurn)()
            nt = self.L.arc_diarization(r, C.byref(tp))
            diar = [{"speaker": _s(tp[i].speaker), "start": _r(tp[i].start, 2), "end": _r(tp[i].end, 2)}
                    for i in range(max(0, nt))]
            return {"transcript": utts, "names": names_, "numbers": nums, "events": evs, "diarization": diar,
                    "speakers": _speakers(self.L, self.L.arc_speakers, r), "audio_s": _r(R.audio_s, 2),
                    "wall_ms": _r(R.wall_ms, 1)}
        finally:
            self.L.arc_result_free(r)


class Stream:
    """arc_stream with arc_audio Stream's output shape, for server/ear.py (mode "full") and server/arcmeta.py
    ("external").

    feed(pcm) / flush() return {"transcript": [closed utterances: text, words (start/end in s from the stream's
    first sample), start, end, reason], "transcript_partial": the open utterance's words so far (kept on every output
    until its FINAL, like arc_audio's) or None, "events": [{"tag": command_complete, "t", "p"}] (the ev_at
    probability crossing the domain's threshold, 1 s refractory, as arc_audio's engine), "frames": [FRAME records
    as dicts] (frames=True), "live": the read-out now ({"t", "utt": {device_directed, command_complete, act, intent}
    while an utterance is open, "speaker": {"id", "role"}})}."""

    def __init__(self, engine: Engine, domain: str = "assistant", mode: str = "full", partial_ms: int = 400,
                 words: bool = True, frames: bool = True, events: bool = True, speakers: bool = True,
                 partial_encode: bool = True, partial_lm: bool = True):
        self.e, self.L = engine, engine.L
        self.domain = domain
        self.mode = {"full": FULL, "heads": HEADS, "external": EXTERNAL}[mode]
        if self.mode == FULL and not engine.decoder:
            raise ValueError("a full stream needs an engine with the decoder")
        self._cb = STREAM_FN(self._on)               # kept alive as long as the stream
        self._dom = domain.encode()
        self.cfg = StreamConfig(mode=self.mode, domain=self._dom, on_output=self._cb, frames=int(frames),
                                events=int(events), partial_ms=int(partial_ms))
        self.h = self.L.arc_stream_open(engine.h, C.byref(self.cfg))
        if not self.h:
            raise RuntimeError(f"arc_stream_open: {_s(self.L.arc_last_error())}")
        if self.mode == FULL and (words or partial_encode):
            # vendor/arc patch: 1 = word times on partials + finals; 2 = partials decode a fresh encode of their window
            # (as arc_audio 1.0.1), not the streamed states that lag by the encoder's 0.24 s look-ahead
            self.L.arc_stream_set_words(self.h, (1 if words else 0) | (2 if partial_encode else 0)
                                        | (0 if partial_lm else 4))
        self.cc_thr = engine.threshold(CC, "ev_at", domain, 0.5)
        self.cc_last = -10.0
        self.total = 0
        self.partial = None                         # the open utterance's partial (dict) and its id
        self.partial_id = None
        self.final_ids: set[int] = set()
        self.utt_tags: dict[int, dict] = {}         # read-out of utterances whose FINAL came before their tags settled
        self.live_utt = None
        self.live_sp = {}
        self.speakers = speakers and self.mode != FULL       # the speaker read-out (ArcMeta), every 0.5 s of audio
        self.n_push = 0
        self.labels = None
        self._reset_out()

    def _reset_out(self):
        self.out = {"transcript": [], "events": [], "frames": [], "audio_events": [], "bio_events": [], "endpoints": [],
                    "tags": []}

    # ------------------------------------------------------------------ records (inside push, on the caller's thread)
    def _on(self, op, _user):
        try:
            o = op.contents
            k = o.kind
            if k == PARTIAL:
                if o.id in self.final_ids:
                    return
                u = _utt(o.utt.contents)
                self.partial = {"start": u["start"], "end": _r(o.audio_s, 2), "text": u["text"],
                                "words": u["words"], "final": False}
                self.partial_id = o.id
            elif k == FINAL:
                u = _utt(o.utt.contents)
                u["id"] = o.id
                u["reason"] = "pause"
                u["tags_final"] = bool(o.tags_final)
                self.final_ids.add(o.id)
                if self.partial_id is not None and self.partial_id <= o.id:
                    self.partial, self.partial_id = None, None
                self.out["transcript"].append(u)
            elif k == TAGS:
                u = _utt(o.utt.contents)
                self.out["tags"].append({"id": o.id, "final": bool(o.tags_final), **{
                    x: u[x] for x in ("start", "end", "device_directed", "command_complete", "intent", "act")}})
            elif k == ENDPOINT:
                u = o.utt.contents
                self.out["endpoints"].append({"id": o.id, "start": _r(u.start, 2), "end": _r(u.end, 2)})
            elif k == EVENT:
                e = o.event.contents
                ev = {"tag": _s(e.label), "t": _r(e.start, 2), "start": _r(e.start, 2), "end": _r(e.end, 2),
                      "p": _r(e.p)}
                self.out["bio_events" if ev["tag"].startswith("bio_ev.") else "audio_events"].append(ev)
            elif k == FRAME:
                self._frame(o.frame.contents)
        except Exception:  # noqa: BLE001 (never raise through C)
            pass

    def _frame(self, f: Frame):
        if self.labels is None:
            self.labels = [_s(f.labels[i]) for i in range(f.n_labels)]
        p = np.ctypeslib.as_array(f.p, shape=(f.n_labels,)).copy()
        fr = {"t": _r(f.t, 2), "speech": _r(f.speech), "in_utterance": bool(f.in_utterance),
              "device_directed": _p(f.device_directed), "command_complete": _p(f.command_complete), "p": p}
        self.out["frames"].append(fr)
        cc = f.command_complete
        if f.in_utterance and cc >= self.cc_thr and f.t - self.cc_last > 1.0 - 1e-6:
            self.cc_last = f.t
            self.out["events"].append({"tag": CC, "t": _r(f.t, 2), "p": _r(cc, 4), "window": "at"})
        if f.in_utterance:
            self.live_utt = {"device_directed": fr["device_directed"], "command_complete": fr["command_complete"],
                             "act": self._best(p, "discourse.act."), "intent": self._best(p, "discourse.intent.")}
        else:
            self.live_utt = None
        self.live_t = fr["t"]

    def _best(self, p, prefix):
        idx = getattr(self, "_idx", None)
        if idx is None:
            # the utterance read-out columns (frame tags come first; the instant queries / events are prefixed)
            idx = self._idx = {pre: [i for i, l in enumerate(self.labels) if l and l.startswith(pre)]
                               for pre in ("discourse.act.", "discourse.intent.")}
        cols = idx.get(prefix) or []
        if not cols:
            return []
        j = max(cols, key=lambda i: p[i])
        return [{"label": self.labels[j], "p": _r(p[j])}]

    def _collect(self) -> dict:
        out = self.out
        self._reset_out()
        out["transcript_partial"] = self.partial
        out["time_s"] = _r(self.total / 16000)
        out["live"] = {"t": getattr(self, "live_t", None), "utt": self.live_utt, "speaker": self.live_sp}
        return out

    # ------------------------------------------------------------------ in
    def feed(self, pcm) -> dict:
        x = _f32(pcm)
        self.total += len(x)
        if self.L.arc_stream_push(self.h, x.ctypes.data_as(c_float_p), len(x)) != 0:
            raise RuntimeError(f"arc_stream_push: {_s(self.L.arc_last_error())}")
        self.n_push += 1
        if self.speakers and self.n_push % 5 == 0:
            self.live_sp = self.speaker() or self.live_sp
        return self._collect()

    def add_words(self, ws: list[dict], final: bool | None = None):
        """EXTERNAL: another recogniser's words [{"text", "start", "end", "is_final"}] (s from the first sample)."""
        if not ws:
            return
        fin = bool(ws[0].get("is_final")) if final is None else bool(final)
        texts = [str(w.get("text") or "").encode() for w in ws]
        arr = (ExtWord * len(ws))(*[ExtWord(t, float(w.get("start", -1) if w.get("start") is not None else -1),
                                            float(w.get("end", -1) if w.get("end") is not None else -1))
                                    for t, w in zip(texts, ws)])
        self.L.arc_stream_words(self.h, arr, len(ws), int(fin))
        if fin:
            self.L.arc_stream_covered(self.h, float(ws[-1].get("end") or 0.0))

    def end_turn(self) -> dict:
        self.L.arc_stream_end_turn(self.h)
        return self._collect()

    def flush(self) -> dict:
        self.L.arc_stream_finish(self.h)
        out = self._collect()
        for u in out["transcript"]:
            u["reason"] = "flush" if u is out["transcript"][-1] else u["reason"]
        out["transcript_partial"] = None
        self.partial = None
        return out

    def speaker(self) -> dict:
        sp = _speakers(self.L, self.L.arc_stream_speakers, self.h)
        if not sp:
            return {}
        s = max(sp, key=lambda x: x["last_seen"])
        role = (s["groups"].get("speaker.role") or {})
        return {"id": s["id"], "role": [role] if role else []}

    def close(self):
        """Frees the stream (and its engine when it owns it). Always call it: the C callback keeps a reference cycle,
        so garbage collection alone frees a stream late."""
        if getattr(self, "h", None):
            self.L.arc_stream_close(self.h)
            self.h = None
            self._cb = None
        if getattr(self, "own", False) and self.e is not None:
            self.e.close()
            self.e = None

    __del__ = close


def predict_shape(r: dict) -> dict:
    """Engine.transcribe() -> the keys of arc_audio's predict() that server/moments.py reads: "speakers" as a dict by
    id with "groups", "events" as {"t", "tag", "p"} (sound / channel events, at their start)."""
    out = dict(r)
    out["model_name"] = "Arc Mini"
    out["speakers"] = {s["id"]: {"groups": s["groups"], "tags": s["tags"], "last_seen": s["last_seen"]}
                       for s in r.get("speakers") or []}
    out["events"] = [{"t": e["start"], "end": e["end"], "tag": e["label"], "p": e["p"]} for e in r.get("events") or []]
    return out


def open_stream(model_dir, threads: int = 2, domain: str = "assistant", mode: str = "full", **kw) -> Stream:
    """A stream on its own engine (one per thread; an engine loads in ~40-90 ms). close() frees both.
    lm=(path, weight, topk) fuses an n-gram LM into the engine's decoder (FULL mode; libarc 1.5+ memory-maps the
    file, so every engine on the same file shares it)."""
    lm = kw.pop("lm", None)
    e = Engine(model_dir, threads, decoder=mode == "full")
    if lm and mode == "full":
        e.set_lm(*lm)
    return e.stream(domain, mode=mode, own=True, **kw)
