"""KDIR's 'wait one more word' action, per Flux update: commit a local typed-key answer NOW, or wait.

For logged decisions (kdir_train.py logs) of director-pool questions that have a typed-key candidate, the shipped
director (Mamba-3 int8) is replayed over the question's REAL Flux update stream (kdir_director_feats.py) and, at every
update, the graph state of the words heard so far is added (typed key complete / equal to the final key / the typed
lookup already returns the final answer's cell / the subject has any edge). Label at update t: committing the local
answer now is no worse than waiting for the end of turn and paying (the prefix lookup returns the final cell and the
final key candidate is no-regret).

Two policies over the same per-update inputs (director outputs + Flux / fused confidence + graph prefix state):
  gbdt    gradient-boosted trees per update (no memory)
  gru     the director's per-update outputs + graph state fed through a tiny GRU (16 units): the 'director with
          graph-state inputs per word' variant (the director itself stays frozen: its numpy runtime has no training
          path, and re-training it needs the recipe's audio features)
Threshold chosen on DEV (precision of early commits >= the end-of-turn key path's), frozen for test. Reported: share
of key-path questions committed before end of turn, the speech-time lead (ms from the commit update to Flux's end of
turn), and early-commit errors.

    .venv/bin/python eval/keystone/kdir_wait.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT = ROOT / "runs/index"
DS = Path(os.environ.get("DIRECTOR_DS", ROOT.parent / "pnp-headstart-director/recipes/headstart_director/data/ds4"))


def director_steps():
    z = np.load(OUT / "kdir_dirfeats.npz")
    steps = z["steps"]
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    return {str(q): steps[off[i]:off[i + 1]] for i, q in enumerate(z["ids"])}


_TIMES: dict = {}


def times_of(qid):
    if not _TIMES:
        for sub in DS.iterdir():
            if sub.is_dir():
                for p in sub.glob("*.npz"):
                    _TIMES[p.stem] = p
    p = _TIMES.get(qid)
    if p is None:
        return None
    return np.load(p, allow_pickle=True)["times"]


def rows_of(logs, dsteps):
    seqs = []
    for L in logs:
        kc = L.get("kc")
        did = L.get("director_id")
        if not kc or not did or did not in dsteps:
            continue
        j = next((i for i, p in enumerate(L["paths"]) if p == "key"), None)
        if j is None or L["cand"][j]["y"] is None:
            continue
        ok_final = bool(L["cand"][j]["y"])
        st = dsteps[did]
        tm = times_of(did)
        if tm is None or len(tm) != len(st):
            continue
        X, y, lead = [], [], []
        n_words = len(kc)
        for t in range(len(st)):
            nw = int(round(st[t][-1] * 30))
            nw = max(1, min(nw, n_words))
            comp, same, hit, has = kc[nw - 1]
            X.append(list(st[t][:-1]) + [nw / 30.0, comp, hit, has])
            y.append(float(hit and same and ok_final))
            lead.append(float(tm[t][2] - tm[t][0]))
        seqs.append({"X": np.array(X, np.float32), "y": np.array(y), "lead": np.array(lead), "ok_final": ok_final,
                     "pos": L["pos"]})
    return seqs


def fit_gbdt(tr):
    import xgboost as xgb
    X = np.concatenate([s["X"] for s in tr])
    y = np.concatenate([s["y"] for s in tr])
    m = xgb.XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.1, n_jobs=2, tree_method="hist")
    m.fit(X, y)
    return lambda s: m.predict_proba(s["X"])[:, 1]


def fit_gru(tr, epochs=60, seed=0):
    import torch
    torch.manual_seed(seed)
    d = tr[0]["X"].shape[1]
    mu = np.concatenate([s["X"] for s in tr]).mean(0)
    sd = np.concatenate([s["X"] for s in tr]).std(0) + 1e-6

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.g = torch.nn.GRU(d, 16, batch_first=True)
            self.o = torch.nn.Linear(16, 1)

        def forward(self, x):
            h, _ = self.g(x)
            return self.o(h)[..., 0]
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    data = [(torch.tensor((s["X"] - mu) / sd)[None], torch.tensor(s["y"], dtype=torch.float32)[None]) for s in tr]
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
            loss.backward()
            opt.step()
    n_params = sum(p.numel() for p in net.parameters())

    def pred(s):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor((s["X"] - mu) / sd)[None]))[0].numpy()
    pred.n_params = n_params
    return pred


def evaluate(seqs, pred, thr):
    """Commit at the first update with p >= thr (before the last update); else end of turn (as today)."""
    early, errs, leads, n = 0, 0, [], 0
    for s in seqs:
        n += 1
        p = pred(s)
        for t in range(len(p) - 1):
            if p[t] >= thr:
                early += 1
                if not s["y"][t]:
                    errs += 1
                else:
                    leads.append(s["lead"][t] * 1000)
                break
    return {"n": n, "early_share": round(early / max(1, n), 3), "early_errors": errs,
            "early_precision": round((early - errs) / early, 3) if early else None,
            "lead_ms_p50": round(float(np.median(leads)), 0) if leads else None}


def main():
    dsteps = director_steps()
    tr_logs = []
    for tag in ("train_r0", "train_r1"):
        p = OUT / f"kdir_log_{tag}.pkl"
        if p.exists():
            tr_logs += pickle.loads(p.read_bytes())["logs"]
    dv_logs = pickle.loads((OUT / "kdir_log_dev_voi.pkl").read_bytes())["logs"]
    te_logs = pickle.loads((OUT / "kdir_log_test_voi.pkl").read_bytes())["logs"] if \
        (OUT / "kdir_log_test_voi.pkl").exists() else []
    tr, dv, te = rows_of(tr_logs, dsteps), rows_of(dv_logs, dsteps), rows_of(te_logs, dsteps)
    rep = {"sequences": {"train": len(tr), "dev": len(dv), "test": len(te)}}
    print(rep, flush=True)
    if len(tr) < 10 or len(dv) < 5:
        rep["note"] = "too few key-path director-pool questions"
        (OUT / "kdir_wait.json").write_text(json.dumps(rep, indent=1))
        return
    # the end-of-turn key path's own precision on dev: early commits must be at least as precise
    base_prec = float(np.mean([s["ok_final"] for s in dv]))
    rep["eot_key_precision_dev"] = round(base_prec, 3)
    for name, fitter in (("gbdt", fit_gbdt), ("gru", fit_gru)):
        pred = fitter(tr)
        pts = [(th, evaluate(dv, pred, th)) for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
        ok = [(th, e) for th, e in pts if e["early_precision"] is None or e["early_precision"] >= base_prec]
        th = min(ok, key=lambda x: x[0])[0] if ok else 0.95
        rep[name] = {"theta": th, "dev": dict(pts)[th], "test": evaluate(te, pred, th) if te else None,
                     "params": getattr(pred, "n_params", None)}
        print(name, rep[name], flush=True)
    (OUT / "kdir_wait.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
