"""The user's own recordings, indexed on this Mac so the `local` lane can answer from them ("Team sync · 12:40").

Same pipeline as the offline index (index/build_index.py), through the same code (server/moments.py):

  file (wav/mp3/flac/ogg/aiff/caf via soundfile; m4a/aac and the audio track of mp4/mov/m4v via macOS afconvert)
    -> 16 kHz mono -> Arc v1 offline transcription with word timings, in ~60 s chunks cut at the quietest 100 ms
    -> utterance-level moments (moments.segment) -> Arc Tagger read-out per moment (entities, numbers, events, intents:
    meta.tags for the ranker's lexical/tag score, meta.chips for display) -> EmbeddingGemma 2 text vector per moment
    (+ audio vector of the moment's own audio, USER_INDEX_AUDIO) -> RECORDINGS_DIR/<id>/{docs.jsonl, vectors.npz}

Storage (config.RECORDINGS_DIR: the app's support folder, never the repo or the bundle):
  manifest.json                 every recording: id, name, status, progress, duration, moments, stats, error
  <id>/docs.jsonl, vectors.npz  that recording's moments, in the LocalIndex format (server/lanes.py)
  _incoming/<id>.<ext>          an uploaded file until it is indexed (then deleted); a file added by path is read in
                                place and never copied

Incremental: each recording is its own folder; lanes.index() = installed index-data + every ready folder. Adding or
removing one rebuilds that in-memory concatenation (milliseconds) and swaps it in; no vectors are recomputed.
One job at a time on one background thread; it pauses while a live question is running (busy probe from app.py), so
ingest never competes with the ear or the embedder on the critical path. Everything stays on this Mac.
"""
from __future__ import annotations

import json
import os
import queue
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config, moments

SR = moments.SR
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".aifc", ".caf"}
VIDEO_EXT = {".mp4", ".mov", ".m4v"}           # the audio track, decoded by macOS afconvert (Core Audio)
ALLOWED_EXT = AUDIO_EXT | VIDEO_EXT
ID_RE = re.compile(r"^r[0-9a-f]{12}$")
ACTIVE = ("queued", "decoding", "transcribing", "tagging", "embedding")
GENERIC_CUES = ["my recording", "my recordings", "in my meeting", "in my call", "my voice memo", "my voice note",
                "in my notes"]
_BASE_CUES: list[str] | None = None
MAX_PAUSE_S = 600.0     # a live run held open longer than this (an open mic left on) no longer holds indexing back


class Cancelled(Exception):
    pass


class Unsupported(ValueError):
    pass


def log(msg: str):
    print(f"[index {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ path safety
def check_ext(name: str) -> str:
    ext = Path(str(name)).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise Unsupported(f"not an audio or video file this app can index ({ext or 'no extension'}); "
                          f"allowed: {', '.join(sorted(ALLOWED_EXT))}")
    return ext


def safe_source(path: str) -> Path:
    """A file the user asked to index by path: absolute, an existing regular file (symlinks resolved), an allowed
    extension, within the size limit, and not inside the recordings folder itself."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise Unsupported("no path")
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise Unsupported("the path must be absolute")
    p = p.resolve()
    check_ext(p.name)
    if not p.is_file():
        raise Unsupported("no such file")
    root = Path(config.RECORDINGS_DIR).resolve()
    if root == p or root in p.parents:
        raise Unsupported("that file is inside the recordings folder")
    if p.stat().st_size > config.USER_INDEX_MAX_MB * 1e6:
        raise Unsupported(f"the file is larger than {config.USER_INDEX_MAX_MB:.0f} MB")
    return p


def valid_id(rid: str) -> bool:
    return isinstance(rid, str) and bool(ID_RE.match(rid))


def display_name(filename: str) -> str:
    """'team_sync-2026-11-20.m4a' -> 'Team sync 2026-11-20'."""
    stem = Path(str(filename)).name
    stem = stem[: -len(Path(stem).suffix)] if Path(stem).suffix else stem
    s = re.sub(r"[_]+", " ", stem)
    s = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", s)
    s = re.sub(r"[\x00-\x1f]", "", " ".join(s.split()))[:80]
    return (s[:1].upper() + s[1:]) if s else "Recording"


def clean_name(name) -> str:
    s = re.sub(r"[\x00-\x1f<>]", "", " ".join(str(name or "").split()))[:80]
    return s or "Recording"


# ------------------------------------------------------------------------------------------------ decoding
def _afconvert(src: Path, out: Path):
    exe = shutil.which("afconvert") or ("/usr/bin/afconvert" if Path("/usr/bin/afconvert").exists() else None)
    if exe:
        r = subprocess.run([exe, "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(src), str(out)],
                           capture_output=True, timeout=3600)
        if r.returncode == 0 and out.exists():
            return
        err = (r.stderr or b"").decode(errors="replace").strip()[:200]
    else:
        err = "afconvert not found"
    ff = shutil.which("ffmpeg")                     # developer machines only: the app never ships ffmpeg
    if ff:
        r = subprocess.run([ff, "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR),
                            "-c:a", "pcm_s16le", str(out)], capture_output=True, timeout=3600)
        if r.returncode == 0:
            return
        err = (r.stderr or b"").decode(errors="replace").strip()[:200]
    raise Unsupported(f"could not decode {src.suffix} audio: {err}")


def decode_to_wav(src: Path, out: Path, cancel=lambda: False) -> float:
    """Any supported file -> 16 kHz mono 16-bit WAV at `out`, streamed in blocks (a long file never sits in memory
    whole). Returns the duration in seconds."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    done = False
    if ext not in VIDEO_EXT and ext not in (".m4a", ".aac", ".caf"):
        try:
            from math import gcd
            from scipy.signal import resample_poly
            with sf.SoundFile(str(src)) as f, sf.SoundFile(str(out), "w", SR, 1, "PCM_16") as o:
                sr_in = f.samplerate
                g = gcd(int(sr_in), SR)
                up, down = SR // g, int(sr_in) // g
                block = int(sr_in) * 30
                while True:
                    if cancel():
                        raise Cancelled()
                    x = f.read(block, dtype="float32", always_2d=True)
                    if not len(x):
                        break
                    x = x.mean(1)
                    if sr_in != SR:
                        x = resample_poly(x, up, down).astype(np.float32)
                    o.write(np.clip(x, -1, 1))
            done = True
        except Cancelled:
            raise
        except Exception:                           # not a libsndfile format after all (e.g. an mp3 it can't read)
            done = False
    if not done:
        _afconvert(src, out)
    info = sf.info(str(out))
    if info.samplerate != SR or info.channels != 1:
        raise Unsupported("decoder produced an unexpected format")
    return float(info.frames) / SR


def chunk_bounds(wav_path: Path, chunk_s: float) -> list[tuple[int, int]]:
    """Sample ranges of ~chunk_s seconds, each cut at the quietest 100 ms in the last 8 s before the boundary (so a
    word is rarely split between two Arc calls)."""
    n = sf.info(str(wav_path)).frames
    ch = int(chunk_s * SR)
    if n <= ch * 1.25:
        return [(0, n)]
    out, a = [], 0
    win, fr = 8 * SR, SR // 10
    while a < n:
        b = a + ch
        if b >= n or n - b < ch // 4:
            out.append((a, n))
            break
        lo = max(a + ch // 2, b - win)
        x, _ = sf.read(str(wav_path), start=lo, stop=b, dtype="float32")
        k = len(x) // fr
        if k:
            e = (x[: k * fr].reshape(k, fr) ** 2).mean(1)
            b = lo + int(np.argmin(e)) * fr + fr // 2
        out.append((a, b))
        a = b
    return out


# ------------------------------------------------------------------------------------------------ model backend
class Backend:
    """The models the ingest uses. Tests swap in fakes (set_backend)."""

    def transcribe(self, wav: np.ndarray, domain: str) -> dict:
        from . import ear
        return ear.arc().predict(wav, domain=domain, transcript="builtin", return_frames=False)

    def tagger(self, domain: str):
        from .tagger import Tagger
        return Tagger(domain=domain)

    def _on_embed_thread(self, fn, *args):
        from . import embed
        return embed._pool.submit(fn, *args).result()   # the embedder's own worker (MPS is single-user)

    def embed_docs(self, texts: list[str], titles: list[str]) -> np.ndarray:
        from . import embed
        return self._on_embed_thread(embed.encode_docs, texts, titles)

    def embed_audio(self, wavs: list[np.ndarray]) -> np.ndarray:
        from . import embed
        return self._on_embed_thread(embed.encode_audio, wavs)


_backend: Backend = Backend()
_busy = lambda: False   # noqa: E731  set by app.py: True while a live question runs


def set_backend(b: Backend | None):
    global _backend
    _backend = b or Backend()


def set_busy_probe(fn):
    global _busy
    _busy = fn or (lambda: False)


# ------------------------------------------------------------------------------------------------ store + worker
class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.recs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self.cancel: set[str] = set()
        self.current: str | None = None
        self.thread: threading.Thread | None = None
        self.stopped = False
        self._load()

    # ---- persistence
    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def _load(self):
        try:
            rows = json.loads(self.manifest.read_text()).get("recordings", [])
        except (OSError, ValueError):
            rows = []
        for r in rows:
            if not valid_id(r.get("id")):
                continue
            if r.get("status") == "ready" and not (self.root / r["id"] / "docs.jsonl").exists():
                r.update(status="error", error="index files missing")
            self.recs[r["id"]] = r
        pending = [r for r in self.recs.values() if r.get("status") in ACTIVE]
        for r in pending:                       # interrupted by a quit: start over (source permitting)
            if r.get("src") and Path(r["src"]).exists():
                r.update(status="queued", progress=0.0, stage="queued")
                self.q.put(r["id"])
            else:
                r.update(status="error", error="interrupted, and the file is gone")
        if pending:
            self._save()
            self._ensure_worker()

    def _save(self):
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.manifest.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "recordings": list(self.recs.values())}, indent=1))
            os.replace(tmp, self.manifest)

    def _set(self, rid: str, save: bool = True, **kw):
        with self.lock:
            r = self.recs.get(rid)
            if r is None:
                return
            r.update(kw)
            if save:
                self._save()

    # ---- public
    def list(self) -> list[dict]:
        with self.lock:
            out = []
            for r in self.recs.values():
                out.append({k: r.get(k) for k in ("id", "name", "status", "stage", "progress", "duration_s",
                                                   "moments", "added_at", "finished_at", "error", "paused",
                                                   "kind", "stats", "file")})
            return out

    def get(self, rid: str) -> dict | None:
        with self.lock:
            r = self.recs.get(rid)
            return dict(r) if r else None

    def ready_dirs(self) -> list[Path]:
        with self.lock:
            return [self.root / r["id"] for r in self.recs.values() if r.get("status") == "ready"]

    def ready_names(self) -> list[str]:
        with self.lock:
            return [r["name"] for r in self.recs.values() if r.get("status") == "ready"]

    def add(self, src: Path, name: str | None = None, owned: bool = False, domain: str | None = None,
            kind: str = "path") -> dict:
        """Queue a file. owned: an upload in _incoming/ (deleted once indexed). A path already indexed (same file,
        size and mtime) returns the existing entry instead of indexing twice."""
        st = src.stat()
        sig = f"{src}|{st.st_size}|{int(st.st_mtime)}"
        with self.lock:
            if not owned:
                for r in self.recs.values():
                    if r.get("sig") == sig and r.get("status") not in ("error",):
                        return {**self.get(r["id"]), "duplicate": True}
            rid = "r" + uuid.uuid4().hex[:12]
            self.recs[rid] = {"id": rid, "name": clean_name(name) if name else display_name(src.name),
                              "file": src.name, "src": str(src), "owned": owned, "sig": None if owned else sig,
                              "kind": kind, "domain": domain or config.USER_INDEX_DOMAIN, "status": "queued",
                              "stage": "queued", "progress": 0.0, "duration_s": None, "moments": 0,
                              "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished_at": None, "error": None,
                              "paused": False, "stats": {}}
            self._save()
        self.q.put(rid)
        self._ensure_worker()
        log(f"queued {rid} {src.suffix} ({st.st_size / 1e6:.1f} MB)")
        return self.get(rid)

    def remove(self, rid: str) -> bool:
        with self.lock:
            r = self.recs.get(rid)
            if r is None:
                return False
            if self.current == rid:
                self.cancel.add(rid)            # the worker stops at its next checkpoint and cleans up
            del self.recs[rid]
            self._save()
            was_ready = r.get("status") == "ready"
        self._cleanup(r)
        if was_ready:
            refresh_index()
        log(f"removed {rid}")
        return True

    def _cleanup(self, r: dict):
        shutil.rmtree(self.root / r["id"], ignore_errors=True)
        if r.get("owned") and r.get("src"):
            try:
                p = Path(r["src"]).resolve()
                if (self.root / "_incoming").resolve() in p.parents:
                    p.unlink(missing_ok=True)
            except OSError:
                pass

    def idle(self) -> bool:
        return self.current is None and self.q.empty()

    def _ensure_worker(self):
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._work, name="user-index", daemon=True)
                self.thread.start()

    def _work(self):
        while not self.stopped:
            try:
                rid = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.lock:
                if rid not in self.recs:
                    continue
                self.current = rid
            try:
                ingest(self, rid)
            except Cancelled:
                log(f"{rid} cancelled")
                shutil.rmtree(self.root / rid, ignore_errors=True)
            except Exception as e:  # noqa: BLE001
                log(f"{rid} failed: {type(e).__name__}: {e}")
                self._set(rid, status="error", stage="error", error=f"{type(e).__name__}: {e}"[:300], paused=False)
                gone = rid not in self.recs
                shutil.rmtree(self.root / rid / ("" if gone else "work"), ignore_errors=True)
            finally:
                with self.lock:
                    self.current = None
                    self.cancel.discard(rid)


# ------------------------------------------------------------------------------------------------ the pipeline
def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1e6 if sys.platform == "darwin" else 1e3), 1)


def ingest(store: Store, rid: str):
    r = store.get(rid)
    if r is None:
        return
    d = store.root / rid
    work = d / "work"
    work.mkdir(parents=True, exist_ok=True)
    domain = r.get("domain") or config.USER_INDEX_DOMAIN
    t_all = time.perf_counter()
    stats: dict = {}

    def check():
        if rid in store.cancel or rid not in store.recs:
            raise Cancelled()

    def wait_idle():
        """Never compete with a live question: wait while one runs."""
        check()
        if _busy():
            store._set(rid, paused=True)
            t0 = time.monotonic()
            while _busy() and time.monotonic() - t0 < MAX_PAUSE_S:
                time.sleep(0.2)
                check()
            store._set(rid, paused=False)

    def progress(stage, p, save=False, **kw):
        store._set(rid, save=save, stage=stage, status=stage, progress=round(float(p), 3), **kw)

    # 1. decode -> 16 kHz mono wav (streamed)
    progress("decoding", 0.01, save=True)
    wav = work / "audio16k.wav"
    t = time.perf_counter()
    dur = decode_to_wav(Path(r["src"]), wav, cancel=lambda: rid in store.cancel)
    stats["decode_s"] = round(time.perf_counter() - t, 2)
    if dur < 0.5:
        raise Unsupported("the recording is shorter than half a second")
    progress("transcribing", 0.05, save=True, duration_s=round(dur, 1))

    # 2. Arc v1, chunk by chunk (progress 5-75 %)
    t = time.perf_counter()
    all_words, labs = [], []
    bounds = chunk_bounds(wav, config.USER_INDEX_CHUNK_S)
    for k, (a, b) in enumerate(bounds):
        wait_idle()
        x, _ = sf.read(str(wav), start=a, stop=b, dtype="float32")
        out = _backend.transcribe(x, domain)
        w, lab = moments.arc_labels(out, domain)
        w, lab = moments.shift_labels(w, lab, a / SR, spk_prefix=f"c{k}." if len(bounds) > 1 else "")
        all_words += w
        labs.append(lab)
        progress("transcribing", 0.05 + 0.70 * b / max(1, bounds[-1][1]), processed_s=round(b / SR, 1))
    stats["arc_s"] = round(time.perf_counter() - t, 2)
    stats["chunks"] = len(bounds)
    lab = moments.merge_labels(labs)
    if not all_words:
        raise Unsupported("no speech found")

    # 3. moments + Arc Tagger read-out (75-80 %)
    progress("tagging", 0.75, save=True)
    name = r["name"]
    src = {"id": rid, "label": name, "domain": domain, "about": "a recording you added"}
    docs = moments.build_docs(src, all_words, lab, id_prefix=f"u{rid[1:]}")
    t = time.perf_counter()
    tg = _backend.tagger(domain)
    try:
        for i, doc in enumerate(docs):
            if i % 20 == 0:
                check()
            moments.tag_doc(tg, doc)
            doc["meta"].update(recording=rid, user=True, file=r.get("file", ""))
    finally:
        try:
            tg.close()
        except Exception:  # noqa: BLE001
            pass
    stats["tag_s"] = round(time.perf_counter() - t, 3)

    # 4. EmbeddingGemma 2: text vectors (80-88 %), then audio vectors (88-99 %)
    progress("embedding", 0.80, save=True)
    t = time.perf_counter()
    tvs = []
    for i in range(0, len(docs), 16):
        wait_idle()
        part = docs[i:i + 16]
        tvs.append(np.asarray(_backend.embed_docs([moments.embed_text_of(x) for x in part],
                                                  [moments.embed_title_of(x) for x in part]), np.float32))
        progress("embedding", 0.80 + 0.08 * min(len(docs), i + 16) / len(docs))
    tv = np.concatenate(tvs)[:, : config.EMBED_DIM]
    stats["embed_text_s"] = round(time.perf_counter() - t, 2)
    av, rows = np.zeros((0, config.EMBED_DIM), np.float32), []
    if config.USER_INDEX_AUDIO:
        t = time.perf_counter()
        chunks = []
        for i, doc in enumerate(docs):
            a, b = doc["t"] - 0.15, doc["meta"]["end"] + 0.15
            if not (moments.AUDIO_MIN <= b - a <= moments.AUDIO_MAX):
                continue
            wait_idle()
            x, _ = sf.read(str(wav), start=max(0, int(a * SR)), stop=int(b * SR), dtype="float32")
            if len(x) < moments.AUDIO_MIN * SR:
                continue
            try:
                chunks.append(np.asarray(_backend.embed_audio([x]), np.float32)[:, : config.EMBED_DIM])
                rows.append(i)
            except Exception as e:  # noqa: BLE001  (one bad slice never fails the recording)
                log(f"{rid} audio vector for moment {i} failed: {str(e)[:120]}")
            progress("embedding", 0.88 + 0.11 * (i + 1) / len(docs))
        if chunks:
            av = np.concatenate(chunks)
        stats["embed_audio_s"] = round(time.perf_counter() - t, 2)
    check()

    # 5. write (tmp + rename), drop the work files, swap the index in
    with open(d / "docs.tmp", "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with open(d / "vectors.tmp", "wb") as f:
        np.savez(f, text=tv.astype(np.float32), audio=av.astype(np.float32), audio_rows=np.asarray(rows, np.int64))
    os.replace(d / "vectors.tmp", d / "vectors.npz")
    os.replace(d / "docs.tmp", d / "docs.jsonl")
    shutil.rmtree(work, ignore_errors=True)
    total = time.perf_counter() - t_all
    stats.update(total_s=round(total, 2), x_realtime=round(dur / max(total, 1e-6), 1), audio_vectors=len(rows),
                 peak_rss_mb=_rss_mb())
    with store.lock:
        if rid not in store.recs or rid in store.cancel:
            raise Cancelled()
        store._set(rid, status="ready", stage="ready", progress=1.0, moments=len(docs), paused=False,
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"), stats=stats, error=None)
    if r.get("owned"):
        store._cleanup({"id": "_none_", "owned": True, "src": r["src"]})
    refresh_index()
    log(f"{rid} ready: {dur:.0f}s audio, {len(docs)} moments in {total:.1f}s ({stats['x_realtime']}x real time)")


# ------------------------------------------------------------------------------------------------ module API
_store: Store | None = None
_store_lock = threading.Lock()


def store() -> Store:
    global _store
    with _store_lock:
        if _store is None or _store.root != Path(config.RECORDINGS_DIR):
            if _store is not None:
                _store.stopped = True
            _store = Store(Path(config.RECORDINGS_DIR))
        return _store


def reset():
    """Tests: forget the store (the next call re-reads config.RECORDINGS_DIR)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.stopped = True
        _store = None


def ready_dirs() -> list[Path]:
    return store().ready_dirs()


def refresh_index():
    """Rebuild the merged local index (installed index-data + ready recordings) and swap it in."""
    from . import lanes
    ix = lanes.LocalIndex(extra=ready_dirs())
    lanes.set_index(ix)
    sync_cues()
    return ix


def name_cues(name: str) -> list[str]:
    """Phrases of a question that point at this recording ("in the team sync")."""
    from .frame import norm
    words = norm(re.sub(r"[_\-/]+", " ", name)).split()
    while words and not re.search(r"[a-z]", words[-1]):
        words.pop()                                   # trailing dates / numbers: "team sync 2026 11 20"
    if not words or not any(len(w) >= 3 and w.isalpha() for w in words):
        return []
    words = words[:6]
    cues = []
    # the whole name, and its trailing 2+ word phrases ("Orion team sync" -> "team sync"): people say the short form
    for k in range(len(words)):
        core = " ".join(words[k:])
        if k and (len(words) - k < 2 or not any(len(w) >= 3 and w.isalpha() and w not in frame_stop() for w in words[k:])):
            continue
        cues += [f"in the {core}", f"in {core}", f"from the {core}", f"during the {core}", f"on the {core}",
                 f"at the {core}"]
        if len(words) - k >= 2:
            cues.append(core)
    return list(dict.fromkeys(cues))


def frame_stop() -> set:
    from .frame import STOP
    return STOP


def sync_cues():
    """frame.LOCAL_CUES = the built-in cues + generic "my recording(s)" + one set per ready recording's name (edited in
    place: server/frame.py and server/rerank.py share the list)."""
    global _BASE_CUES
    from . import frame
    if _BASE_CUES is None:
        _BASE_CUES = list(frame.LOCAL_CUES)
    extra = []
    names = store().ready_names()
    if names:
        extra += GENERIC_CUES
        for n in names:
            extra += name_cues(n)
    frame.LOCAL_CUES[:] = list(dict.fromkeys(_BASE_CUES + extra))


def write_upload_tmp(ext: str) -> tuple[Path, object]:
    d = Path(config.RECORDINGS_DIR) / "_incoming"
    d.mkdir(parents=True, exist_ok=True)
    fd, p = tempfile.mkstemp(prefix="up-", suffix=ext, dir=str(d))
    return Path(p), os.fdopen(fd, "wb")
