"""The user's own recordings (server/userindex.py): ingest pipeline with fake Arc / Arc Tagger / EmbeddingGemma,
chunked long files, incremental add + remove, persistence, the local lane returning user moments, question routing by
recording name, path safety and the HTTP API. Fast and model-free, except test_real_models_end_to_end (opt-in:
USER_INDEX_REAL=1, needs the models in MODELS_DIR and macOS `say`).

Run: .venv/bin/python -m pytest -q tests/test_user_index.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config, embed, frame, lanes, moments, rerank, userindex  # noqa: E402

SR = 16000
DIM = 768

SENTENCES = [
    "Good morning everyone, welcome to the sync.",
    "Priya shipped the beta to 42 testers on Tuesday.",
    "the vendor in Toronto has not sent the keys.",
    "the budget approved is 18000 dollars for cloud.",
    "Elena from Berlin gets an offer on Friday.",
    "the database migration is 70% done now.",
    "we run the load test on December eighth.",
    "we meet again on Monday at ten.",
]
# vocabulary: each word is a tone burst at 200 + 20*k Hz; the fake Arc hears the tone and says the word
VOCAB = list(dict.fromkeys(w for s in SENTENCES for w in s.split()))
assert len(VOCAB) < 95
PEOPLE = {"Priya", "Elena"}
PLACES = {"Toronto", "Berlin"}


def _vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    for w in frame.norm(text).split():
        if w not in frame.STOP:
            v[hash(w) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v + 1e-3


def synth(path: Path, sentences=SENTENCES, repeat: int = 1, gap_s: float = 2.0, sr: int = SR, stereo=False):
    """A wav whose 'speech' is tone bursts (one per word). Returns the script [(word, start, end)]."""
    out, script, t = [], [], 0.3
    out.append(np.zeros(int(0.3 * sr), np.float32))
    for _ in range(repeat):
        for s in sentences:
            for w in s.split():
                k = VOCAB.index(w)
                n = int(0.3 * sr)
                x = 0.3 * np.sin(2 * np.pi * (200 + 20 * k) * np.arange(n) / sr).astype(np.float32)
                out += [x, np.zeros(int(0.1 * sr), np.float32)]
                script.append((w, t, t + 0.3))
                t += 0.4
            out.append(np.zeros(int(gap_s * sr), np.float32))
            t += gap_s
    y = np.concatenate(out)
    if stereo:
        y = np.stack([y, y], 1)
    sf.write(str(path), y, sr)
    return script


class FakeBackend(userindex.Backend):
    """Arc: tone bursts -> words with timings (+ names / numbers like Arc's read-out). Tagger: places / people /
    numbers by word. Embedder: bag-of-words hashing (768-d)."""

    def __init__(self):
        self.calls = {"transcribe": 0, "embed_docs": 0, "embed_audio": 0, "audio_s": 0.0}

    def transcribe(self, wav, domain):
        self.calls["transcribe"] += 1
        self.calls["audio_s"] += len(wav) / SR
        fr = SR // 50
        k = len(wav) // fr
        on = (wav[: k * fr].reshape(k, fr) ** 2).mean(1) > 1e-3
        words, names, numbers, i = [], [], [], 0
        while i < k:
            if not on[i]:
                i += 1
                continue
            j = i
            while j < k and on[j]:
                j += 1
            seg = wav[i * fr: j * fr]
            if len(seg) >= SR // 10:
                f = np.argmax(np.abs(np.fft.rfft(seg))) * SR / len(seg)
                w = VOCAB[int(round((f - 200) / 20))]
                a, b = round(i * fr / SR, 2), round(j * fr / SR, 2)
                words.append({"text": w, "start": a, "end": b})
                if w in PEOPLE:
                    names.append({"start": a, "end": b, "tag": "entity.mention.person", "heard": w})
                if w[0].isdigit():
                    numbers.append({"start": a, "end": b, "heard": w, "kind": "count"})
            i = j
        return {"transcript": [{"words": words}] if words else [], "names": names, "numbers": numbers,
                "events": [{"t": 0.1, "tag": "speaker.change"}], "speakers": {},
                "diarization": [{"speaker": "T1", "start": 0.0, "end": len(wav) / SR}]}

    def tagger(self, domain):
        class T:
            def tag_text(self, text):
                spans = []
                for w in text.split():
                    c = w.strip(".")
                    if c in PLACES:
                        spans.append({"family": "entity", "type": "entity.mention.place", "text": c, "score": .9, "thr": .5})
                    if c in PEOPLE:
                        spans.append({"family": "entity", "type": "entity.mention.person", "text": c, "score": .9, "thr": .5})
                    if c[:1].isdigit():
                        spans.append({"family": "number", "type": "entity.number.money", "text": c, "score": .9, "thr": .3})
                return {"spans": spans, "events": [],
                        "utterances": [{"intent": [{"tag": "intent.report_inform", "on": True}]}]}

            def close(self):
                pass
        return T()

    def embed_docs(self, texts, titles):
        self.calls["embed_docs"] += len(texts)
        return np.stack([_vec(x) for x in texts])

    def embed_audio(self, wavs):
        self.calls["embed_audio"] += len(wavs)
        return np.stack([_vec(f"audio {len(w)}") for w in wavs])


@pytest.fixture
def fake(monkeypatch, tmp_path):
    b = FakeBackend()
    userindex.set_backend(b)
    base = tmp_path / "base_index"           # no installed index-data unless a test writes one
    base.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", base)
    monkeypatch.setattr(config, "USER_INDEX_AUDIO", True)
    monkeypatch.setattr(embed, "encode_queries", lambda ts: np.stack([_vec(t) for t in ts]))
    lanes.set_index(None)
    yield b
    lanes.set_index(None)


def wait_idle(st=None, timeout=20.0):
    st = st or userindex.store()
    t0 = time.time()
    while time.time() - t0 < timeout:
        if st.idle() and all(r["status"] in ("ready", "error") for r in st.list()):
            return st.list()
        time.sleep(0.02)
    raise AssertionError(f"ingest did not finish: {st.list()}")


def add(path, name=None):
    return userindex.store().add(userindex.safe_source(str(path)), name=name)


# ------------------------------------------------------------------------------------------------ pipeline
def test_ingest_builds_timestamped_tagged_moments(fake, tmp_path):
    script = synth(tmp_path / "team_sync.wav")
    rec = add(tmp_path / "team_sync.wav")
    assert rec["name"] == "Team sync" and rec["status"] == "queued"
    (r,) = wait_idle()
    assert r["status"] == "ready", r
    assert r["progress"] == 1.0 and r["moments"] == len(SENTENCES)          # one moment per sentence (pauses)
    assert abs(r["duration_s"] - (script[-1][2] + 2.0)) < 0.3
    st = r["stats"]
    assert st["x_realtime"] > 0 and st["chunks"] == 1 and st["audio_vectors"] == len(SENTENCES) and st["peak_rss_mb"] > 0
    d = config.RECORDINGS_DIR / r["id"]
    assert not (d / "work").exists()                                         # the 16 kHz work copy is gone
    docs = [json.loads(x) for x in (d / "docs.jsonl").read_text().splitlines()]
    z = np.load(d / "vectors.npz")
    assert z["text"].shape == (len(docs), DIM) and len(z["audio"]) == len(z["audio_rows"]) == len(docs)
    toronto = next(x for x in docs if "Toronto" in x["text"])
    t0 = next(s for w, s, e in script if w == "Toronto") - 0.4 * 3           # sentence starts 3 words earlier
    assert abs(toronto["t"] - t0) < 0.1
    assert toronto["title"] == f"Team sync · {moments.mmss(toronto['t'])}"
    assert {"text": "Toronto", "kind": "place"} in toronto["meta"]["chips"]   # Arc Tagger read-out -> chips
    assert "Toronto" in toronto["meta"]["tags"] and "intent.report_inform" in toronto["meta"]["tags"]
    priya = next(x for x in docs if "Priya" in x["text"])
    kinds = {(c["text"], c["kind"]) for c in priya["meta"]["chips"]}
    assert ("Priya", "person") in kinds and ("42", "number") in kinds         # Arc names / numbers + tagger
    assert priya["meta"]["recording"] == r["id"] and priya["meta"]["user"] is True
    assert priya["meta"]["context_after"].startswith("the vendor")


def test_long_file_is_chunked_and_keeps_its_clock(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USER_INDEX_CHUNK_S", 20.0)
    script = synth(tmp_path / "long.wav", repeat=4)                          # ~110 s -> several Arc chunks
    add(tmp_path / "long.wav")
    (r,) = wait_idle()
    assert r["status"] == "ready" and r["stats"]["chunks"] >= 4
    assert abs(fake.calls["audio_s"] - r["duration_s"]) < 0.1               # every sample transcribed exactly once
    docs = [json.loads(x) for x in (config.RECORDINGS_DIR / r["id"] / "docs.jsonl").read_text().splitlines()]
    assert len(docs) == 4 * len(SENTENCES)                                   # no sentence split by a chunk cut
    starts = sorted(s for (w, s, e) in script if w in ("Good", "Priya"))
    assert [round(d["t"], 1) for d in docs if d["text"].split()[0] in ("Good", "Priya")] == [round(s, 1) for s in starts]
    assert [d["t"] for d in docs] == sorted(d["t"] for d in docs)


def test_decode_resamples_stereo_and_other_rates(tmp_path):
    synth(tmp_path / "a.flac", sr=44100, stereo=True)
    out = tmp_path / "o.wav"
    dur = userindex.decode_to_wav(tmp_path / "a.flac", out)
    info = sf.info(str(out))
    assert info.samplerate == SR and info.channels == 1 and abs(dur - info.frames / SR) < 1e-6 and dur > 20


@pytest.mark.skipif(not shutil.which("afconvert"), reason="macOS afconvert")
def test_decode_m4a_and_video_audio_track_via_afconvert(tmp_path):
    synth(tmp_path / "a.wav")
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(tmp_path / "a.wav"), str(tmp_path / "a.m4a")], check=True)
    dur = userindex.decode_to_wav(tmp_path / "a.m4a", tmp_path / "o.wav")
    assert abs(dur - sf.info(str(tmp_path / "a.wav")).duration) < 0.2
    if shutil.which("ffmpeg"):                                               # make a real .mp4 with a video track
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=3",
                        "-i", str(tmp_path / "a.m4a"), "-t", "3", "-c:v", "libx264", "-c:a", "aac", "-shortest",
                        str(tmp_path / "v.mp4")], check=True)
        assert abs(userindex.decode_to_wav(tmp_path / "v.mp4", tmp_path / "v.wav") - 3.0) < 0.2


# ------------------------------------------------------------------------------------------------ the local lane
def test_local_lane_searches_user_moments_merged_with_installed_index(fake, tmp_path):
    base = config.INDEX_DIR                                                   # an "installed" index-data: 2 moments
    bdocs = [{"id": f"nba-{i}", "title": f"NBA broadcast · 00:0{i}", "text": t, "url": "", "t": float(i), "meta": {}}
             for i, t in enumerate(["Hunter hits the three", "Ball with the steal"])]
    (base / "docs.jsonl").write_text("\n".join(json.dumps(d) for d in bdocs) + "\n")
    np.savez(base / "vectors.npz", text=np.stack([_vec(d["text"]) for d in bdocs]),
             audio=np.stack([_vec("x")]), audio_rows=np.array([1]))
    synth(tmp_path / "standup.wav")
    add(tmp_path / "standup.wav", name="Standup")
    wait_idle()
    ix = lanes.index()
    assert len(ix) == 2 + len(SENTENCES) and ix.docs[0].id == "nba-0"
    assert list(ix.av_rows[:1]) == [1] and all(r >= 2 for r in ix.av_rows[1:])   # user audio rows offset past base
    docs = asyncio.run(lanes.local_text("which vendor in Toronto has not sent the keys"))
    top = docs[0]
    assert "Toronto" in top.text and top.source == "local" and top.title.startswith("Standup · 00:")
    b = top.brief()
    assert b["t"] == top.t and b["recording"] and b["label"] == "Standup"
    assert {"text": "Toronto", "kind": "place"} in b["chips"]
    # the ranker's lexicon sees the new index (gen), and the tags are searchable terms
    assert "toronto" in rerank.lexicon(ix).terms[top.id]


def test_question_naming_the_recording_routes_local(fake, tmp_path):
    q = "what did priya say in the team sync about testers".split()
    assert frame.build(q, [], None).route == "web"
    synth(tmp_path / "team_sync_2026-11-20.wav")
    add(tmp_path / "team_sync_2026-11-20.wav")
    (r,) = wait_idle()
    assert r["name"] == "Team sync 2026-11-20"
    lanes.index()
    assert frame.build(q, [], None).route == "local"
    assert frame.build("what did i say in my recordings about the budget".split(), [], None).route == "local"
    assert "team sync" not in rerank.strip_cues("what did priya say in the team sync")
    assert "in the team sync" in userindex.name_cues("Orion team sync")              # the short form people say
    assert "sync" not in userindex.name_cues("Orion team sync") and userindex.name_cues("2026 11 20") == []
    userindex.store().remove(r["id"])
    assert frame.build(q, [], None).route == "web"                           # the cue goes with the recording


# ------------------------------------------------------------------------------------------------ incremental
def test_incremental_add_and_remove(fake, tmp_path):
    synth(tmp_path / "one.wav")
    synth(tmp_path / "two.wav", sentences=SENTENCES[:3])
    a = add(tmp_path / "one.wav")
    wait_idle()
    n_embed = fake.calls["embed_docs"]
    ix1 = lanes.index()
    b = add(tmp_path / "two.wav")
    wait_idle()
    assert fake.calls["embed_docs"] - n_embed == 3                          # only the new recording was embedded
    ix2 = lanes.index()
    assert ix2 is not ix1 and ix2.gen != ix1.gen and len(ix2) == len(SENTENCES) + 3
    assert len({d.id for d in ix2.docs}) == len(ix2)                         # ids unique across recordings
    assert userindex.store().remove(a["id"])
    ix3 = lanes.index()
    assert len(ix3) == 3 and all(d.meta["recording"] == b["id"] for d in ix3.docs)
    assert not (config.RECORDINGS_DIR / a["id"]).exists()
    assert [r["id"] for r in userindex.store().list()] == [b["id"]]
    assert not userindex.store().remove(a["id"])                             # already gone
    again = add(tmp_path / "two.wav")                                        # same file again: not indexed twice
    assert again["duplicate"] and again["id"] == b["id"]


def test_remove_while_indexing_cancels_the_job(fake, tmp_path, monkeypatch):
    busy = {"on": True}
    userindex.set_busy_probe(lambda: busy["on"])                             # a live question holds the job
    synth(tmp_path / "x.wav")
    r = add(tmp_path / "x.wav")
    t0 = time.time()
    while not (userindex.store().get(r["id"]) or {}).get("paused"):
        assert time.time() - t0 < 5
        time.sleep(0.02)
    assert fake.calls["transcribe"] == 0                                     # paused before Arc ran
    userindex.store().remove(r["id"])
    busy["on"] = False
    t0 = time.time()
    while not userindex.store().idle():
        assert time.time() - t0 < 5
        time.sleep(0.02)
    assert userindex.store().list() == [] and not (config.RECORDINGS_DIR / r["id"]).exists()


def test_paused_while_live_then_finishes(fake, tmp_path):
    busy = {"on": True}
    userindex.set_busy_probe(lambda: busy["on"])
    synth(tmp_path / "x.wav")
    r = add(tmp_path / "x.wav")
    time.sleep(0.3)
    assert userindex.store().get(r["id"])["paused"] and fake.calls["transcribe"] == 0
    busy["on"] = False
    (done,) = wait_idle()
    assert done["status"] == "ready" and not done["paused"]


# ------------------------------------------------------------------------------------------------ persistence
def test_index_persists_across_restart_and_resumes_interrupted_jobs(fake, tmp_path):
    synth(tmp_path / "keep.wav")
    a = add(tmp_path / "keep.wav")
    wait_idle()
    userindex.reset()
    lanes.set_index(None)                                                    # "next launch"
    assert [r["id"] for r in userindex.store().list()] == [a["id"]]
    assert len(lanes.index()) == len(SENTENCES)
    # a job that was running when the app quit starts over on the next launch
    m = config.RECORDINGS_DIR / "manifest.json"
    rows = json.loads(m.read_text())
    synth(tmp_path / "half.wav")
    rows["recordings"].append({"id": "r0123456789ab", "name": "Half", "src": str(tmp_path / "half.wav"),
                               "status": "transcribing", "progress": 0.4, "domain": "meeting", "stats": {}})
    rows["recordings"].append({"id": "r0123456789ac", "name": "Gone", "src": str(tmp_path / "nope.wav"),
                               "status": "embedding", "progress": 0.9, "stats": {}})
    m.write_text(json.dumps(rows))
    userindex.reset()
    out = {r["id"]: r for r in wait_idle()}
    assert out["r0123456789ab"]["status"] == "ready"
    assert out["r0123456789ac"]["status"] == "error"
    assert len(lanes.index()) == 2 * len(SENTENCES)


def test_damaged_recording_folder_never_breaks_the_index(fake, tmp_path):
    synth(tmp_path / "a.wav")
    a = add(tmp_path / "a.wav")
    synth(tmp_path / "b.wav", sentences=SENTENCES[:2])
    add(tmp_path / "b.wav")
    wait_idle()
    (config.RECORDINGS_DIR / a["id"] / "vectors.npz").write_bytes(b"garbage")
    ix = userindex.refresh_index()
    assert len(ix) == 2                                                      # the other recording still loads


def test_failed_file_reports_an_error(fake, tmp_path):
    sf.write(str(tmp_path / "silence.wav"), np.zeros(SR * 3, np.float32), SR)
    add(tmp_path / "silence.wav")
    (r,) = wait_idle()
    assert r["status"] == "error" and "no speech" in r["error"]


# ------------------------------------------------------------------------------------------------ path safety
def test_path_safety(tmp_path, monkeypatch):
    synth(tmp_path / "ok.wav")
    assert userindex.safe_source(str(tmp_path / "ok.wav")) == (tmp_path / "ok.wav").resolve()
    (tmp_path / "notes.txt").write_text("secret")
    (tmp_path / "noext").write_text("x")
    os.symlink(tmp_path / "notes.txt", tmp_path / "sneaky.wav")             # an allowed name pointing at a text file
    os.symlink(tmp_path / "ok.wav", tmp_path / "fine_link.wav")
    inside = config.RECORDINGS_DIR / "x"
    inside.mkdir(parents=True)
    shutil.copy(tmp_path / "ok.wav", inside / "copy.wav")
    bad = ["", "ok.wav", "../ok.wav", str(tmp_path / "notes.txt"), str(tmp_path / "noext"), str(tmp_path / "sneaky.wav"),
           str(tmp_path / "missing.wav"), str(inside / "copy.wav"), str(tmp_path / "sub/../../etc/passwd"),
           str(tmp_path / "ok.wav") + "\x00.txt", None, 42]
    for p in bad:
        with pytest.raises(userindex.Unsupported):
            userindex.safe_source(p)
    assert userindex.safe_source(str(tmp_path / "fine_link.wav")) == (tmp_path / "ok.wav").resolve()
    monkeypatch.setattr(config, "USER_INDEX_MAX_MB", 0.001)
    with pytest.raises(userindex.Unsupported, match="larger"):
        userindex.safe_source(str(tmp_path / "ok.wav"))
    for good in ("a.WAV", "b.mp3", "c.m4a", "d.aac", "e.flac", "f.ogg", "g.mp4", "h.mov"):
        userindex.check_ext(good)
    for nope in ("a.exe", "a.wav.sh", "a.json", "a", ".wav/..", "a.py"):
        with pytest.raises(userindex.Unsupported):
            userindex.check_ext(nope)
    assert userindex.valid_id("r0123456789ab")
    for rid in ("../r0123456789ab", "r0123456789ab/..", "R0123456789AB", "manifest.json", "_incoming", ""):
        assert not userindex.valid_id(rid)
    assert userindex.display_name("../../etc/team_sync-notes.m4a") == "Team sync notes"


# ------------------------------------------------------------------------------------------------ HTTP API
def test_api_add_list_upload_remove(fake, tmp_path):
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    synth(tmp_path / "weekly.wav")
    r = c.post("/api/recordings", json={"path": str(tmp_path / "weekly.wav"), "name": "Weekly sync"})
    assert r.status_code == 200 and r.json()["added"][0]["name"] == "Weekly sync"
    bad = c.post("/api/recordings", json={"paths": [str(tmp_path / "x.txt"), "relative.wav"]})
    assert bad.status_code == 400 and len(bad.json()["errors"]) == 2
    assert c.post("/api/recordings", content=json.dumps({"path": str(tmp_path / "weekly.wav")}),
                  headers={"content-type": "text/plain"}).status_code == 415          # JSON only (no simple CORS post)
    assert c.post("/api/recordings", json={"path": str(tmp_path / "weekly.wav")},
                  headers={"origin": "https://evil.example"}).status_code == 403
    # upload: raw bytes + ?name=; a traversal-looking name is only a label
    synth(tmp_path / "memo.wav", sentences=SENTENCES[:2])
    up = c.post("/api/recordings/upload?name=../../memo.wav", content=(tmp_path / "memo.wav").read_bytes())
    assert up.status_code == 200, up.text
    rec = up.json()["added"][0]
    assert rec["name"] == "Memo" and Path(userindex.store().get(rec["id"])["src"]).parent == config.RECORDINGS_DIR / "_incoming"
    assert c.post("/api/recordings/upload?name=evil.sh", content=b"#!/bin/sh").status_code == 415
    assert c.post("/api/recordings/upload?name=x.wav", content=b"x",
                  headers={"origin": "http://localhost:1"}).status_code == 403
    wait_idle()
    lst = c.get("/api/recordings").json()
    names = {x["name"]: x for x in lst["recordings"]}
    assert names["Weekly sync"]["status"] == "ready" and names["Weekly sync"]["moments"] == len(SENTENCES)
    assert names["Memo"]["status"] == "ready" and ".m4a" in lst["allowed"] and not lst["busy"]
    assert not list((config.RECORDINGS_DIR / "_incoming").iterdir())        # the upload is deleted once indexed
    assert c.delete(f"/api/recordings/{names['Memo']['id']}").json()["ok"]
    assert c.delete("/api/recordings/..%2Fmanifest.json").status_code in (400, 404, 405)   # never a file op
    assert c.delete("/api/recordings/_incoming").status_code == 400
    assert c.delete("/api/recordings/r000000000000").status_code == 404
    assert c.delete(f"/api/recordings/{names['Weekly sync']['id']}", headers={"origin": "https://evil.example"}).status_code == 403
    assert [x["name"] for x in c.get("/api/recordings").json()["recordings"]] == ["Weekly sync"]


def test_build_index_uses_the_shared_moment_code():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "index"))
    import build_index
    assert build_index.segment is moments.segment and build_index.embed_title_of is moments.embed_title_of
    assert moments.mmss(3725) == "1:02:05" and moments.mmss(65) == "01:05"


# ------------------------------------------------------------------------------------------------ real models
@pytest.mark.skipif(os.environ.get("USER_INDEX_REAL") != "1" or not shutil.which("say"),
                    reason="opt-in: USER_INDEX_REAL=1 (real Arc v1 + Arc Tagger + EmbeddingGemma 2, macOS say)")
def test_real_models_end_to_end(tmp_path, monkeypatch):
    """A short spoken clip through the real pipeline; the local lane finds the right moment by a text question."""
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "no_base")
    userindex.set_backend(None)
    lanes.set_index(None)
    aiff = tmp_path / "clip.aiff"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff),
                    "Good morning everyone. This is the planning call. [[slnc 900]] "
                    "The vendor in Toronto has not sent the updated API keys, so we cannot test payments. [[slnc 900]] "
                    "Elena from Berlin will get an offer on Friday for the designer role."], check=True)
    add(aiff, name="Planning call")
    (r,) = wait_idle(timeout=600)
    assert r["status"] == "ready", r
    assert r["moments"] >= 2
    docs = asyncio.run(lanes.local_text("who is getting an offer for the designer role"))
    assert docs and "offer" in docs[0].text.lower() and docs[0].title.startswith("Planning call · 00:")
