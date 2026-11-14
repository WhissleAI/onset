"""Calibrator step of a training cycle: label new answers, refit server/calibrate.py's model, measure current vs
candidate on the FIXED held-out sets. Run by cycle.py as one niced process (it loads the tagger / embedder / answer
model); writes <cycle dir>/calib/{metrics.json, calib_model.json}.

Data
  base      eval/calib_data.py's 651 labelled text-path answers (feat/calib), frozen into runs/train/calib/ once.
            Its test split (186 answers, 30 % of question families by hash) is the fixed held-out TEST set.
  synth     this cycle's synthetic standalone turns with Wikidata gold (synth.py) labelled by the same pipeline
            (eval/calib_data.py --families ...: Wikipedia lane, real answer model, measured $). Always TRAIN.
  real      harvested eval replays / app rows that carry conf_feats and a correct label: TRAIN, except the rows of
            the fixed live replays (LIVE_FILES), which form the "live" held-out set when there are >= 30 of them.
Gate: eval/train/gates.calib.

  .venv/bin/python eval/train/calib_step.py --cycle N --out runs/train/cycles/cN/calib [--no-label]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("WEB_BACKEND", "wikipedia")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.sys_path()

D = C.TRAIN / "calib"
ROWS = D / "label_rows.jsonl"                 # base + every cycle's labelled synthetic rows (resumable by text)
FEATS = D / "feats_cache.json"                # text -> features (the tagger replay is the slow part)
BASE_SRC = Path(os.environ.get("CALIB_BASE_ROWS", C.MAIN.parent / "headstart-calib/runs/calib/rows.jsonl"))
LIVE_FILES = ("ab_q58_", "ab_showcase_")      # fixed live replays: held out
MIN_LIVE = 30


def ensure_base() -> int:
    D.mkdir(parents=True, exist_ok=True)
    if not ROWS.exists():
        if not BASE_SRC.exists():
            return 0
        shutil.copy2(BASE_SRC, ROWS)
    return len(C.read_jsonl(ROWS))


def synth_families(cycle: int) -> Path | None:
    """This cycle's standalone synthetic turns with gold, as calib_data.py --families rows."""
    p = C.TRAIN / "synth" / f"c{cycle}" / "sessions.json"
    if not p.exists():
        return None
    out = []
    for s in C.read_json(p, []):
        for t in s["turns"]:
            g = t["gold"]
            if g.get("followup") or not g.get("expect") or not g.get("addressed", True) or s["cls"] == "multi_part":
                continue
            out.append({"fid": f"syn-c{cycle}-{t['tid']}", "canon": t["text"], "kind": s["cls"], "expect": g["expect"],
                        "expect_src": "wikidata", "paraphrases": []})
    if not out:
        return None
    fp = D / f"families_c{cycle}.jsonl"
    fp.write_text("".join(json.dumps(r) + "\n" for r in out))
    return fp


def label(cycle: int, log: Path) -> dict:
    fp = synth_families(cycle)
    if fp is None:
        return {"labelled": 0}
    n0 = len(C.read_jsonl(ROWS))
    cmd = [C.PY, str(C.ROOT / "eval/calib_data.py"), "--families", str(fp), "--out", str(ROWS), "--conc", "3"]
    with log.open("a") as f:
        p = subprocess.run(cmd, cwd=str(C.ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=3600,
                           env=dict(os.environ, WEB_BACKEND="wikipedia", TAVILY_API_KEY=""))
    txt = log.read_text().splitlines()
    usd = None
    for ln in reversed(txt):
        if "est. $" in ln:
            try:
                usd = float(ln.rsplit("est. $", 1)[1].split()[0])
            except ValueError:
                pass
            break
    return {"labelled": len(C.read_jsonl(ROWS)) - n0, "rc": p.returncode, "answer_usd_est": usd}


def split_rows(rows: list[dict], real: list[dict], base_texts: set | None = None) -> tuple[list, list, list]:
    """(train, test, live). Base test split = fixed TEST. Synthetic rows (group syn-*, incl. the Tavily slice) are
    train. Rows calib_data.py adds on its own (the 58 / showcase sets) are not base rows: dropped (they are the
    end-to-end sets). Real rows from the fixed live replay files = LIVE, other real labelled rows = train."""
    tr, te, live = [], [], []
    for r in rows:
        if r.get("correct") is None or r.get("error"):
            continue
        if str(r.get("group", "")).startswith("syn-"):
            tr.append({**r, "split": "train"})
        elif base_texts is not None and r.get("text") not in base_texts:
            continue
        elif r.get("split") == "test":
            te.append(r)
        else:
            tr.append(r)
    for r in real:
        if r.get("correct") is None or not r.get("conf_feats") or r.get("mode") != "headstart":
            continue
        rr = {"text": r.get("text") or r.get("question"), "conf_feats": r["conf_feats"], "correct": r["correct"],
              "group": f"real:{r.get('id') or r.get('h')}", "set": "real:" + r.get("src", "")}
        if any(Path(str(r.get("file", ""))).name.startswith(x) for x in LIVE_FILES):
            live.append(rr)
        else:
            tr.append(rr)
    return tr, te, live


def features(rows, cache: dict) -> list[dict]:
    import calib_fit as CF
    frames: dict = {}
    out = []
    for r in rows:
        if "conf_feats" in r:
            out.append(dict(r["conf_feats"]))
            continue
        k = r["text"]
        if k not in cache:
            cache[k] = CF.row_features(r, frames)
        out.append(cache[k])
    return out


def measure(model: dict, F: list[dict], y) -> dict:
    import numpy as np

    import calib_fit as CF
    from server import calibrate
    calibrate.reload(model)
    if not len(y):
        return {}
    p = [calibrate.raw(f, "post") for f in F]
    p = [0.5 if v is None else v for v in p]
    r = CF.report("post", p, np.asarray(y, float))
    return {k: r[k] for k in ("n", "ece", "brier", "auc")}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-label", action="store_true")
    ap.add_argument("--C", type=float, default=0.3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"base_rows": ensure_base()}
    if not rep["base_rows"]:
        rep["skip"] = "no base calibration rows"
        C.write_json(out / "metrics.json", rep)
        return rep
    if not a.no_label:
        rep["label"] = label(a.cycle, out / "label.log")
    import numpy as np

    import calib_fit as CF
    from server import calibrate
    rows = C.read_jsonl(ROWS)
    real = C.read_jsonl(C.TRAIN / "real" / "rows.jsonl")
    base_texts = {r.get("text") for r in C.read_jsonl(BASE_SRC)} if BASE_SRC.exists() else None
    tr, te, live = split_rows(rows, real, base_texts)
    cache = C.read_json(FEATS, {})
    Ftr, Fte = features(tr, cache), features(te, cache)
    Flive = features(live, cache)
    C.write_json(FEATS, cache)
    ytr = np.array([1.0 if r["correct"] else 0.0 for r in tr])
    yte = np.array([1.0 if r["correct"] else 0.0 for r in te])
    ylive = np.array([1.0 if r["correct"] else 0.0 for r in live])
    gtr = np.array([str(r.get("group") or r["text"]) for r in tr])
    cur = json.loads(calibrate.MODEL_PATH.read_text())
    cand = {"version": 1, "bands": cur.get("bands") or {"low": calibrate.CALIB_LOW, "high": calibrate.CALIB_HIGH},
            "trained_on": {"rows": int(len(tr)), "backend": "wikipedia", "label": "keyword scorer vs gold",
                           "cycle": a.cycle, "synthetic": int(sum(str(r.get("group", "")).startswith("syn-") for r in tr)),
                           "real": int(sum(str(r.get("group", "")).startswith("real:") for r in tr))}}
    for stage, names in (("pre", CF.PRE), ("post", CF.POST)):
        cand[stage] = CF.fit_stage(Ftr, ytr, gtr, names, a.C)
    m = {}
    for name, model in (("current", cur), ("candidate", cand)):
        m[name] = {"test": measure(model, Fte, yte)}
        if len(live) >= MIN_LIVE:
            m[name]["live"] = measure(model, Flive, ylive)
    calibrate.reload(cur)
    rep.update(n_train=int(len(tr)), n_test=int(len(te)), n_live=int(len(live)), **m)
    C.write_json(out / "calib_model.json", cand)
    C.write_json(out / "metrics.json", rep)
    print(json.dumps({k: v for k, v in rep.items()}, default=str)[:2000])
    return rep


if __name__ == "__main__":
    main()
