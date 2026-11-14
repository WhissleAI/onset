"""Mamba search director step of a training cycle (server/director.py; recipe: PromptingNemo-private
headstart-director, recipes/headstart_director, linked as RECIPE). Reuses the recipe's own scripts end to end:

  1. stream   new synthetic clips (synth.py wavs) through REAL Deepgram Flux at 1x (recipe flux_record.py), capped at
              TRAIN_FLUX_MIN audio minutes per cycle ($0.0077 / min, logged)
  2. feats    eval/conf_features.py --v2 (tagger + Arc replay, as a live Session sees it); audio deleted afterwards
  3. arrays   recipe build_dataset.py --v2 --split train -> runs/train/director/ds_c<N> (TRAIN only: the fixed held-out
              splits test / val / test_v2 / val_v2 / heldout80 / test_om never change)
  4. train    recipe train.py with the shipped v2 recipe (sweep5 v2_s0: s model, same regularisation) on data/ds4 +
              --extra-train every ds_c* so far; seed = cycle
  5. measure  recipe evaluate.py on test / heldout80 / test_v2 (val-chosen thresholds), evaluate2.py (addressed),
              eot_loss.py (threshold-free eot BCE vs the current run), for the CURRENT model's run too (cached)
  6. export   recipe export.py -> <cycle dir>/director/{director.npz, director.json} (int8, parity checked)
Skipped when fewer than MIN_NEW new traces arrived since the last director training. Gate: eval/train/gates.director.

  .venv/bin/python eval/train/director_step.py --cycle N --out runs/train/cycles/cN/director [--no-stream]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RECIPE = Path(os.environ.get("DIRECTOR_RECIPE", C.MAIN.parent / "pnp-headstart-director/recipes/headstart_director"))
D = C.TRAIN / "director"
TRACES = D / "flux_traces.jsonl"
FLUX_MIN = float(os.environ.get("TRAIN_FLUX_MIN", "10"))
FLUX_USD_MIN = 0.0077
MIN_NEW = int(os.environ.get("TRAIN_DIRECTOR_MIN_NEW", "40"))
SHIPPED_RUN = "runs/sweep5/v2_s0"
REG = ["--drop", "0.3", "--tok-drop", "0.5", "--wd", "0.3", "--lr", "1e-3", "--steps", "3000", "--threads", "3"]


def sh(cmd: list[str], log: Path, cwd: Path = RECIPE, timeout: float = 3600, env: dict | None = None) -> int:
    with log.open("a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        e = dict(os.environ, HEADSTART=str(C.ROOT), OMP_NUM_THREADS="3", ARC_THREADS="2", **(env or {}))
        try:
            return subprocess.run(["nice", "-n", "10", *map(str, cmd)], cwd=str(cwd), stdout=f,
                                  stderr=subprocess.STDOUT, timeout=timeout, env=e).returncode
        except subprocess.TimeoutExpired:
            return 124


def feat_files() -> str:
    fs = [RECIPE / "data/features_all.jsonl"] + sorted(RECIPE.glob("data/featV2.9.*.jsonl")) + \
        sorted(RECIPE.glob("data/featV2.q2*.jsonl")) + [C.MAIN / "runs/conf_features.jsonl"]
    return ",".join(str(p) for p in fs if p.exists())


def pending_clips() -> list[tuple[dict, Path]]:
    done = {r["id"] for r in C.read_jsonl(TRACES)}
    out = []
    for cl in sorted(C.TRAIN.glob("synth/c*/clips.jsonl")):
        for c in C.read_jsonl(cl):
            wav = cl.parent / "audio" / c["file"]
            if c["id"] not in done and wav.exists():
                out.append((c, wav))
    return out


def stream(cycle: int, out: Path) -> dict:
    import soundfile as sf
    todo, mins = [], 0.0
    for c, wav in pending_clips():
        dur = sf.info(str(wav)).duration + 1.6
        if mins + dur / 60 > FLUX_MIN:
            break
        todo.append((c, wav))
        mins += dur / 60
    if not todo:
        return {"streamed": 0}
    man = D / f"stream_c{cycle}.jsonl"
    man.write_text("".join(json.dumps(c) + "\n" for c, _ in todo))
    audio = D / f"audio_c{cycle}"
    audio.mkdir(parents=True, exist_ok=True)
    for c, wav in todo:
        dst = audio / c["file"]
        if not dst.exists():
            dst.symlink_to(wav)
    rc = sh([C.PY, RECIPE / "flux_record.py", "--manifest", man, "--audio-dir", audio, "--out", TRACES,
             "--workers", "4"], out / "stream.log")
    got = {r["id"] for r in C.read_jsonl(TRACES)} & {c["id"] for c, _ in todo}
    usd = round(mins * FLUX_USD_MIN, 4)
    C.append_jsonl(D / "flux_cost.jsonl", [{"cycle": cycle, "clips": len(got), "audio_min": round(mins, 2), "usd_est": usd}])
    return {"streamed": len(got), "audio_min": round(mins, 2), "flux_usd_est": usd, "rc": rc}


def featurize(cycle: int, out: Path) -> dict:
    """conf_features --v2 over streamed traces not featurized yet; then delete their audio (synth wavs)."""
    have = set()
    for p in D.glob("featV2.c*.jsonl"):
        have |= {r["id"] for r in C.read_jsonl(p)}
    todo = [r for r in C.read_jsonl(TRACES) if r["id"] not in have]
    if not todo:
        return {"featurized": 0}
    tp = D / f"todo_c{cycle}.jsonl"
    tp.write_text("".join(json.dumps(r) + "\n" for r in todo))
    audio = D / "audio_all"
    audio.mkdir(parents=True, exist_ok=True)
    for cl in C.TRAIN.glob("synth/c*/audio/*.wav"):
        dst = audio / cl.name
        if not dst.exists():
            dst.symlink_to(cl)
    fo = D / f"featV2.c{cycle}.jsonl"
    rc = sh([C.PY, C.ROOT / "eval/conf_features.py", "--traces", tp, "--audio-dir", audio, "--out", fo, "--v2",
             "--no-ear-words"], out / "feats.log", cwd=C.ROOT)
    ids = {r["id"] for r in C.read_jsonl(fo)}
    pruned = 0
    for cl in C.TRAIN.glob("synth/c*/audio/*.wav"):
        if cl.stem in ids:
            cl.unlink()
            pruned += 1
    for p in audio.iterdir():
        if p.is_symlink() and not p.exists():
            p.unlink()
    return {"featurized": len(ids), "rc": rc, "audio_pruned": pruned}


def build(cycle: int, out: Path) -> dict:
    fo = D / f"featV2.c{cycle}.jsonl"
    if not fo.exists() or not C.read_jsonl(fo):
        return {"built": 0}
    man = ",".join(str(p) for p in sorted(C.TRAIN.glob("synth/c*/clips.jsonl")))
    ds = D / f"ds_c{cycle}"
    rc = sh([C.PY, RECIPE / "build_dataset.py", "--v2", "--feat", fo, "--manifest", man, "--out", ds, "--split", "train",
             "--traces", TRACES], out / "build.log")
    return {"built": len(list((ds / "train").glob("*.npz"))) if (ds / "train").exists() else 0, "rc": rc}


def evaluate(run: Path, out: Path, log: Path) -> dict:
    ds, lr = RECIPE / "data/ds4", RECIPE / "data/conf_model_final.json"
    feat = feat_files()
    m = {}
    for split, val, man in (("test", "val", "data/manifest_all.jsonl"), ("heldout80", "val", ""),
                            ("test_v2", "val_v2", "data/q2_clips.jsonl")):
        ej = run / f"eval_{split}.json"
        if not ej.exists():
            cmd = [C.PY, RECIPE / "evaluate.py", "--data", ds, "--run", run, "--feat", feat, "--split", split, "--val", val,
                   "--lr-model", lr, "--out", ej] + (["--manifest", man] if man else [])
            sh(cmd, log)
        e = C.read_json(ej, {})
        dd = (e.get("eot") or {}).get("director") or {}
        fr = (e.get("eot") or {}).get("frontier") or {}
        m[split] = {"p50": dd.get("p50"), "premature": dd.get("premature"),
                    "not_search_answered_pct": dd.get("not_search_answered_pct"),
                    "p50_at10": ((fr.get("10.0") or {}).get("director") or {}).get("p50"),
                    "p50_at20": ((fr.get("20.0") or {}).get("director") or {}).get("p50")}
    e2 = run / "eval2.json"
    if not e2.exists():
        sh([C.PY, RECIPE / "evaluate2.py", "--data", ds, "--run", run, "--manifest",
            "data/scenes.jsonl,data/pii_clips.jsonl", "--route-split", "test,test_v2", "--out", e2], log)
    a = ((C.read_json(e2, {}).get("addressed") or {}).get("director") or {})
    m["addr"] = {"auc": a.get("auc"), "false_respond_pct": a.get("false_respond_pct")}
    return m


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--current-run", default="", help="the current model's recipe run (default: registry meta / shipped)")
    ap.add_argument("--force", action="store_true", help="train even with < MIN_NEW new traces")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    if not (RECIPE / "train.py").exists() or not (RECIPE / "data/ds4").exists():
        rep["skip"] = "director recipe / data not on this machine"
        C.write_json(out / "metrics.json", rep)
        return rep
    if not a.no_stream:
        rep["stream"] = stream(a.cycle, out)
    rep["feats"] = featurize(a.cycle, out)
    rep["build"] = build(a.cycle, out)
    st = C.read_json(D / "state.json", {})
    extra = sorted(str(p / "train") for p in D.glob("ds_c*") if (p / "train").exists())
    n_extra = sum(len(list(Path(p).glob("*.npz"))) for p in extra)
    new = n_extra - st.get("trained_on_extra", 0)
    rep["extra_traces"], rep["new_since_last"] = n_extra, new
    if new < MIN_NEW and not a.force:
        rep["skip"] = f"{new} new traces since the last director training (< {MIN_NEW})"
        C.write_json(out / "metrics.json", rep)
        return rep
    run = (D / f"run_c{a.cycle}").resolve()
    run.mkdir(parents=True, exist_ok=True)
    rc = sh([C.PY, RECIPE / "train.py", "--data", RECIPE / "data/ds4", "--out", run, "--size", "s", "--seed",
             str(a.cycle), "--v2", *REG] + (["--extra-train", ",".join(extra)] if extra else []), out / "train.log")
    if rc != 0 or not (run / "model.pt").exists():
        rep["error"] = f"train.py exit {rc}"
        C.write_json(out / "metrics.json", rep)
        return rep
    cur_run = Path(a.current_run) if a.current_run else RECIPE / SHIPPED_RUN
    if not (cur_run / "model.pt").exists():
        rep["error"] = f"current run {cur_run.name} has no model.pt"
        C.write_json(out / "metrics.json", rep)
        return rep
    # the current run's eval files live in the recipe (results/ copies for the shipped run): reuse when present
    cur_dir = cur_run
    if not (cur_dir / "eval_test_v2.json").exists() and (RECIPE / "results" / cur_run.relative_to(RECIPE / "runs")).exists():
        cur_dir = RECIPE / "results" / cur_run.relative_to(RECIPE / "runs")
    m = {"current": evaluate(cur_dir if (cur_dir / "eval_test_v2.json").exists() else cur_run, out, out / "eval.log"),
         "candidate": evaluate(run, out, out / "eval.log")}
    el = out / "eot_loss.json"
    sh([C.PY, RECIPE / "eot_loss.py", "--data", RECIPE / "data/ds4", "--ref", cur_run, "--splits", "test,test_v2,heldout80",
        "--out", el, run], out / "eval.log")
    eo = C.read_json(el, {})
    for sp, rr in eo.items():
        for path, v in rr.items():
            who = "current" if Path(path).resolve() == cur_run.resolve() else "candidate"
            m[who].setdefault(sp, {})["eot_bce"] = v["bce"]
    ex = out / "export"
    rc = sh([C.PY, RECIPE / "export.py", "--run", run, "--data", RECIPE / "data/ds4", "--split", "heldout80", "--out", ex,
             "--thr", run / "eval_test.json"], out / "export.log")
    for n in ("director.npz", "director.json"):
        if (ex / n).exists():
            (out / n).write_bytes((ex / n).read_bytes())
    if (out / "director.json").exists():
        # thresholds of the v2 heads (addr / pii / route) come from evaluate2 (candidate) or else the current model
        dj = C.read_json(out / "director.json", {})
        cur_thr = (C.read_json(C.ROOT / "server/director/director.json", {}) or {}).get("thr") or {}
        e2 = C.read_json(run / "eval2.json", {})
        cand_thr = {"director_addr": ((e2.get("addressed") or {}).get("director") or {}).get("thr")}
        for k, v in cur_thr.items():
            dj.setdefault("thr", {})
            if k not in dj["thr"]:
                dj["thr"][k] = cand_thr.get(k) if cand_thr.get(k) is not None else v
        if not isinstance(dj.get("trained"), dict):
            dj["trained"] = {}
        dj["trained"]["run"] = f"eval/train/director_step.py cycle {a.cycle} ({run.name})"
        C.write_json(out / "director.json", dj)
    st["trained_on_extra"] = n_extra
    C.write_json(D / "state.json", st)
    rep.update(m, run=str(run), current_run=str(cur_run), export_rc=rc)
    C.write_json(out / "metrics.json", rep)
    print(json.dumps(rep, default=str)[:3000])
    return rep


if __name__ == "__main__":
    main()
