"""Arc + director signals for the KDIR workload's director-pool questions: the shipped search director (server/director.py,
Mamba-3 int8, numpy) replayed over each question's REAL Flux update stream (the director dataset's featurized
voicings, $DIRECTOR_DS = recipes/headstart_director/data/ds4/<split>/<id>.npz).

Per update: director eot / fire / draft / addressed / route (card, facts, web, local, none) + Flux word confidence
(last, min) and fused confidence (last, min) + tagger intent top-p, question act, device-directed. Saved per question
(all updates, for the per-word policy) to runs/index/kdir_dirfeats.npz.

    DIRECTOR_DS=.../data/ds4 .venv/bin/python eval/keystone/kdir_director_feats.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from server import director  # noqa: E402

DS = Path(os.environ.get("DIRECTOR_DS", ROOT.parent / "pnp-headstart-director/recipes/headstart_director/data/ds4"))
NAMES = ["d_eot", "d_fire", "d_draft", "d_addr", "d_r_card", "d_r_facts", "d_r_web", "d_r_local", "d_r_none",
         "conf_last", "conf_min", "fconf_last", "fconf_min", "intent_top_p", "act_question", "act_device_directed", "nw"]
FI = {n: i for i, n in enumerate(director.FEATS)}


def run_one(m, z) -> np.ndarray:
    feats = np.concatenate([z["feats"], z["feats2"]], 1) if m.v2 else z["feats"]
    state, out = None, []
    for t in range(len(feats)):
        r, state = m.step(state, feats[t].astype(np.float32), z["toks"][t], z["wtok"][t], z["wfeat"][t],
                          z["wfeat2"][t] if "wfeat2" in z.files else None)
        f = z["feats"][t]
        out.append([r["eot"], r["fire"], r["draft"], r.get("addr", 0.0), *list(r.get("route", np.zeros(5))),
                    f[FI["conf_last"]], f[FI["conf_min"]], f[FI["fconf_last"]], f[FI["fconf_min"]],
                    f[FI["intent_top_p"]], f[FI["act_question"]], f[FI["act_device_directed"]], f[FI["nw"]]])
    return np.array(out, np.float32)


def main():
    m = director.model()
    if m is None:
        raise SystemExit(f"no director model: {director._failed}")
    ids, arrs = [], []
    for sub in sorted(p for p in DS.iterdir() if p.is_dir()):
        for p in sorted(sub.glob("*.npz")):
            try:
                z = np.load(p, allow_pickle=True)
                a = run_one(m, z)
            except Exception as e:  # noqa: BLE001
                print("skip", p.name, type(e).__name__, str(e)[:80])
                continue
            ids.append(p.stem)
            arrs.append(a)
        print(sub.name, len(ids), flush=True)
    lens = np.array([len(a) for a in arrs])
    np.savez_compressed(ROOT / "runs/index/kdir_dirfeats.npz", ids=np.array(ids), lens=lens,
                        steps=np.concatenate(arrs), names=np.array(NAMES))
    print("saved", len(ids), "questions,", int(lens.sum()), "updates")


if __name__ == "__main__":
    main()
