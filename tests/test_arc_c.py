"""Arc Mini in C (server/arc_c.py over vendor/arc's libarc) as the ear, Arc metadata and recordings backend.

The pure parts (manifest / profile, the predict-shape adapter, the Deepgram word adapter, backend selection) always run.
The rest needs libarc.dylib (scripts/fetch_models.py build-arc) and MODELS_DIR/arc-mini; it is skipped without them,
and uses a spoken eval question (AUDIO_DIR/q05.wav, "how tall is Mount Everest") when one is there.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import arc_c, config, ear, modelpaths  # noqa: E402

MAN = json.loads((ROOT / "models" / "manifest.json").read_text())
PROFILE = json.loads((ROOT / "models" / "release_profile.json").read_text())
HAVE_LIB = modelpaths.arc_lib().exists()
HAVE_MINI = (config.ARC_MINI_DIR / "model.q4.safetensors").exists()
needs_model = pytest.mark.skipif(not (HAVE_LIB and HAVE_MINI), reason="libarc.dylib or MODELS_DIR/arc-mini missing")


def _speech():
    p = Path(config.AUDIO_DIR) / "q05.wav"
    if not p.exists():
        pytest.skip("no spoken eval audio (AUDIO_DIR/q05.wav)")
    import soundfile as sf
    x, sr = sf.read(str(p), dtype="float32")
    assert sr == 16000
    return x


# ------------------------------------------------------------------------------------------------ no model needed
def test_manifest_ships_arc_mini_and_release_drops_arc_v1():
    m = next(x for x in MAN["models"] if x["dir"] == "arc-mini")
    assert m["required"] and m["source"]["type"] == "hf" and m["source"]["repo"] == "WhissleAI/arc-v1.1"
    paths = {f["path"]: f for f in m["files"]}
    for need in ("config.json", "tags.json", "heads.safetensors", "model.q4.safetensors", "LICENSE", "NOTICE",
                 "moonshine-streaming-tiny/model.q4.safetensors", "moonshine-streaming-tiny/tokenizer.json",
                 "moonshine-streaming-tiny/LICENSE"):
        assert need in paths, need
    # the repo keeps the model under mini/; the licence files sit at its root
    assert all(f.get("remote", f["path"]) == ("mini/" + f["path"] if f["path"] not in ("LICENSE", "NOTICE")
                                              else f["path"]) for f in m["files"])
    assert m["size"] < 80e6
    v1 = next(x for x in MAN["models"] if x["dir"] == "arc-v1")
    assert v1["required"] is False
    assert PROFILE["models"]["arc-v1"].get("skip") and PROFILE["models"]["arc-mini"]["files"] == "all"
    assert any(x.get("source", {}).get("command") == "scripts/fetch_models.py build-arc" for x in MAN["models"])


def test_release_bundle_plan_has_arc_mini_not_arc_v1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
    fm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm)
    plan = {m["dir"]: files for m, files, _ in fm.bundle_plan(MAN, PROFILE)}
    assert "arc-mini" in plan and "arc-v1" not in plan
    assert {"LICENSE", "NOTICE"} <= {f["path"] for f in plan["arc-mini"]}


def test_vendored_libarc_keeps_licence_and_marks_the_patch():
    v = ROOT / "vendor" / "arc"
    assert (v / "LICENSE").exists() and (v / "NOTICE").exists() and (v / "ORIGIN.md").exists()
    h = (v / "include" / "arc.h").read_text()
    assert "#define ARC_ABI_VERSION 2" in h and "int arc_stream_set_words(arc_stream* s, int on);" in h
    assert "Onset patch" in (v / "src" / "arc_stream.c").read_text()


def test_arc_lib_path(tmp_path):
    root = tmp_path / "repo"
    assert modelpaths.arc_lib({}, root) == root / "vendor" / "arc" / "libarc.dylib"
    assert modelpaths.arc_lib({"ARC_LIB": "/x/libarc.dylib"}, root) == Path("/x/libarc.dylib")


def test_backend_selection(monkeypatch):
    monkeypatch.setattr(config, "ARC_BACKEND", "mini")
    assert ear.backend() == "mini"
    with pytest.raises(RuntimeError):
        ear.arc()                                  # no arc_audio object without ARC_BACKEND=v1
    monkeypatch.setattr(config, "ARC_BACKEND", "v1")
    assert ear.backend() == "v1"
    monkeypatch.setattr(config, "ARC_BACKEND", "anything")
    assert ear.backend() == "mini"


def test_predict_shape_matches_what_moments_reads():
    from server import moments
    r = {"transcript": [{"start": 0.4, "end": 2.0, "text": "How tall is Mount Everest?",
                         "words": [{"text": "How", "start": 0.27, "end": 0.71},
                                   {"text": "Everest?", "start": 1.27, "end": 2.15}]}],
         "names": [{"start": 1.15, "end": 2.15, "tag": "entity.mention.place", "heard": "Mount Everest",
                    "entity": None, "state": "not_in_list"}],
         "numbers": [{"start": 3.0, "end": 3.5, "heard": "8,848", "kind": None, "kind_head": "distance"}],
         "events": [{"label": "scene.laughter", "start": 5.0, "end": 5.6, "p": 0.9},
                    {"label": "meta.clipping", "start": 0.7, "end": 1.8, "p": 0.98}],
         "speakers": [{"id": "S1", "last_seen": 2.0, "groups": {"speaker.role": {"label": "speaker.role.host",
                                                                                  "p": 0.9}}, "tags": []}],
         "diarization": [{"speaker": "S1", "start": 0.5, "end": 2.1}]}
    words, lab = moments.arc_labels(arc_c.predict_shape(r), "meeting")
    assert words == [("How", 0.27, 0.71), ("Everest?", 1.27, 2.15)]
    assert lab["names"] == [(1.15, 2.15, "Mount Everest", "place")]
    assert lab["numbers"] == [(3.0, 3.5, "8,848", "distance")]
    assert lab["events"] == [(5.0, "scene.laughter")]               # meta.* dropped as before
    assert lab["roles"] == [(0.5, 2.1, "host")] and lab["spk"] == [(0.5, 2.1, "S1")]
    assert lab["labels_source"] == "Arc Mini"


def test_dg_words_adapter_forwards_deepgram_words():
    got = []

    class S:
        def add_words(self, ws, final=None):
            got.append((ws, final))

    a = ear.DGWords(S())
    a.on_message({"type": "Results", "is_final": False, "channel": {"alternatives": [{"words": [
        {"word": "how", "start": 0.3, "end": 0.6}, {"word": "tall", "start": 0.6, "end": 0.9}]}]}})
    a.on_message({"type": "Results", "is_final": True, "channel": {"alternatives": [{"words": []}]}})
    assert got == [([{"text": "how", "start": 0.3, "end": 0.6}, {"text": "tall", "start": 0.6, "end": 0.9}], False)]


# ------------------------------------------------------------------------------------------------ libarc + weights
@needs_model
def test_library_version_and_abi():
    assert arc_c.lib().arc_abi_version() == arc_c.ABI
    assert arc_c.version().startswith("1.4")


@needs_model
def test_bad_model_dir_raises(tmp_path):
    with pytest.raises(RuntimeError):
        arc_c.Engine(tmp_path, 1)


@needs_model
def test_full_stream_partials_and_final_carry_word_times():
    x = np.concatenate([_speech(), np.zeros(16000, np.float32)])
    s = arc_c.open_stream(config.ARC_MINI_DIR, 2, "assistant", partial_ms=400)
    partials, finals, events = [], [], []
    try:
        for i in range(0, len(x), 1600):
            o = s.feed(x[i:i + 1600])
            if o["transcript_partial"]:
                partials.append(o["transcript_partial"])
            finals += o["transcript"]
            events += o["events"]
        o = s.flush()
        finals += o["transcript"]
    finally:
        s.close()
    assert partials and all(not p["final"] for p in partials)
    timed = [p for p in partials if p["words"]]
    assert timed and all(w["start"] is not None and w["end"] >= w["start"] for p in timed for w in p["words"])
    assert len(finals) == 1
    f = finals[0]
    assert "everest" in f["text"].lower()
    ws = f["words"]
    assert [w["text"] for w in ws] == f["text"].split()
    assert all(0.0 <= a["start"] <= b["start"] for a, b in zip(ws, ws[1:]))
    assert 0.1 < ws[0]["start"] < 0.6 and 1.8 < ws[-1]["end"] < 2.4            # speech is ~0.4-2.1 s
    assert any(e["tag"] == arc_c.CC for e in events)                            # command_complete fired
    assert s.h is None and s.e is None                                          # close() freed stream + engine


@needs_model
def test_offline_transcribe_matches_stream_final():
    x = _speech()
    e = arc_c.Engine(config.ARC_MINI_DIR, 2)
    try:
        r = e.transcribe(x, "assistant")
    finally:
        e.close()
    assert len(r["transcript"]) == 1 and "everest" in r["transcript"][0]["text"].lower()
    assert r["transcript"][0]["words"] and r["diarization"]
    assert r["transcript"][0]["act"][0]["label"].startswith("discourse.act.")


@needs_model
def test_external_stream_gives_metadata_without_a_decoder():
    x = np.concatenate([_speech(), np.zeros(12000, np.float32)])
    s = arc_c.open_stream(config.ARC_MINI_DIR, 1, "assistant", mode="external")
    lives, finals = [], []
    try:
        assert not s.e.decoder
        s.add_words([{"text": "How", "start": 0.27, "end": 0.71, "is_final": False}])
        for i in range(0, len(x), 1600):
            o = s.feed(x[i:i + 1600])
            lives.append(o["live"])
            if i == 24000:
                s.add_words([{"text": t, "start": a, "end": b} for t, a, b in [
                    ("How", 0.27, 0.71), ("tall", 0.69, 0.95), ("is", 0.93, 1.17), ("Mount", 1.15, 1.29),
                    ("Everest?", 1.27, 2.15)]], final=True)
            finals += o["transcript"]
        finals += s.flush()["transcript"]
    finally:
        s.close()
    utts = [lv["utt"] for lv in lives if lv.get("utt")]
    assert utts and all(0 <= u["device_directed"] <= 1 for u in utts)
    assert any(u["act"] for u in utts) and any(u["intent"] for u in utts)
    assert any(lv.get("speaker", {}).get("id") for lv in lives)
    assert finals and "Everest?" in finals[-1]["text"]


@needs_model
def test_arc_ear_on_arc_mini_feeds_sessions_timed_words(monkeypatch):
    """server/ear.py ArcEar with ARC_BACKEND=mini: sinks get timed partial words every chunk and the final."""
    monkeypatch.setattr(config, "ARC_BACKEND", "mini")
    x = np.concatenate([_speech(), np.zeros(16000, np.float32)])

    class Sink:
        def __init__(self):
            self.partials, self.finals, self.audio = [], [], 0

        def on_audio(self, pcm, at):
            self.audio += len(pcm)

        def on_partial(self, ws, now=None):
            self.partials.append(ws)

        async def on_final(self, ws):
            self.finals.append(ws)

    sink, events = Sink(), []

    async def run():
        e = ear.ArcEar([sink], on_event=events.append)
        for i in range(0, len(x), 1600):
            await e.feed(x[i:i + 1600])
        await e.flush()
        e.close()
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert sink.audio == len(x)
    assert sink.partials and all(w["end"] is not None for ws in sink.partials for w in ws)
    assert len(sink.finals) == 1 and "everest" in " ".join(w["text"] for w in sink.finals[0]).lower()
    kinds = {ev["kind"] for ev in events}
    assert {"partial", "final", "arc_command_complete"} <= kinds
