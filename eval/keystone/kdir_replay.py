"""KDIR workload replay: every action's outcome for every question (counterfactual), and policies over them.

Same text-level method as replay.py (simulated clock, recorded fixture, no audio / answer model / Tavily), extended
with the KDIR action space (server/keystone/kdir.py):

  pay       today's path (the router's plan): paid when the fixture's tier is web, else the free lane it planned
  local     the best local candidate (typed key / hash / question NN / dense passages / a shared cell)
  local_rc  local + a background re-check on the free lanes. The re-check CATCHES a wrong local answer when the free
            lanes return a strong result that is right for the question (recorded counterfactually); a caught error
            is shown, then corrected ("corrected", counted separately, not regret); an uncaught one is regret
  free      free lanes first (leads index + Wikipedia summary / the planned card or Wikidata lane), paid web only when
            they come back weak. Never for live / fresh questions (a stale encyclopedia page is not news)

Per question (non-card): outcomes of every action, the lookup / graph / Arc / director features of every candidate
(rows for kdir_train.py), and the policy's choice. Correctness and regret are replay.py's (answer keywords where they
exist, else same evidence as today's path); regret = the served answer is wrong where today's path would be right.

Store modes (network effect): pooled (one store), user (per-user stores), user_shared (+ other users' verified typed
edges, keystone/shared.py), user_shared_cells (+ verified public question -> answer cells, the wider allowlist).

    .venv/bin/python eval/keystone/kdir_replay.py --stream runs/index/kdir_test.jsonl --policies fresh,a,voi,oracle
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval/keystone"))

import numpy as np  # noqa: E402

import replay as R  # noqa: E402
from server.keystone import kdir, voi  # noqa: E402
from server.keystone.core import Cand, Flags, Keystone  # noqa: E402

OUT = ROOT / "runs/index"
FX = OUT / "kdir_fixture.json"
VEC = OUT / "kdir_vecs.npz"
FULL = dict(cells=True, key=True, hash=True, act=True, qnn=True)
COST = {"R": 10.0, "C": 1.0, "eps": 0.02}          # oracle / labels: a wrong answer = 10 paid calls; corrected = 1
LAT_ORDER = ["local", "local_rc", "free", "pay"]
_FX: dict = {}
_VEC: dict = {}


def _load_fx(path):
    if path not in _FX:
        _FX[path] = json.loads(Path(path).read_text())
    return _FX[path]


def _load_vec(path):
    if path not in _VEC:
        z = np.load(path)
        _VEC[path] = ({k: v.astype(np.float32) for k, v in zip(z["doc_ids"], z["doc_vecs"])},
                      {k: v.astype(np.float32) for k, v in zip(z["q_texts"], z["q_vecs"])})
    return _VEC[path]


class KData:
    """replay.Data's interface (replay.run / make_q work on it) over the KDIR fixture + counterfactual lanes."""

    def __init__(self, stream_path, fx_path=None, vec_path=None):
        fx_path = fx_path or (FX if FX.exists() else OUT / "fixture.json")
        vec_path = vec_path or (VEC if VEC.exists() else OUT / "fixture_vecs.npz")
        self.stream = [json.loads(x) for x in Path(stream_path).read_text().splitlines() if x.strip()]
        self.fx = _load_fx(str(fx_path))
        self.stream = [it for it in self.stream if it["query"] in self.fx and not self.fx[it["query"]].get("error")]
        self.dvec, self.qvec = _load_vec(str(vec_path))
        self.fam_ids, self.text_fid = {}, {}
        for it in self.stream:
            self.text_fid.setdefault(it["query"], it["fid"])
            top = self.fx[it["query"]]["docs"][:3]
            self.fam_ids.setdefault(it["fid"], set()).update([d["id"] for d in top] + [d["url"] for d in top if d["url"]])
        self.frames = frames_for({it["query"] for it in self.stream})
        self.dsig = director_signals()
        self._leads: dict = {}

    def docs(self, text, which="route"):
        r = self.fx[text]
        src = r["docs"] if which == "route" else ((r.get(which) or {}).get("docs") or [])
        return [SimpleNamespace(id=d["id"], title=d["title"], text=d["text"], url=d["url"], source=d["source"],
                                meta=d.get("meta") or {}, score=d.get("score") or 0.0) for d in src]

    def free_docs(self, text):
        """The free-first lanes for a web-tier question: what was recorded (the Wikipedia summary of the frame's entity,
        or the planned card / Wikidata lane) + the offline leads index top 3 for the question vector (as
        knowledge.leads returns them: strong at cos >= ROUTER_LEADS_STRONG)."""
        docs = self.docs(text, "free")
        r = self.fx[text].get("free") or {}
        if r.get("lane") not in ("card", "facts") and self.qvec.get(text) is not None:
            from server import config, knowledge
            ix = knowledge.leads_index()
            seen = {d.url for d in docs}
            for i, sc in ix.search(np.asarray(self.qvec[text], np.float32)[: ix.vecs.shape[1]], 3):
                d = ix.docs[i]
                if d.get("url") in seen:
                    continue
                did = d["url"] or f"lead:{d['title']}"
                if did not in self.dvec:
                    self.dvec[did] = ix.vecs[i].astype(np.float32)
                docs.append(SimpleNamespace(id=did, title=f"{d['title']} · local Wikipedia",
                                            text=d["text"], url=d.get("url", ""), source="local Wikipedia", score=sc,
                                            meta={"tier": "facts", "strong": sc >= config.ROUTER_LEADS_STRONG}))
        return docs

    def leads(self, text):
        if text not in self._leads:
            self._leads[text] = kdir.leads_signal(self.qvec.get(text))
        return self._leads[text]

    def director(self, it):
        if not it.get("director_id") or it["form"] != "canon" or it["query"] != it["text"]:
            return None
        return self.dsig.get(it["director_id"])


_FRAMES: dict = {}


def frames_for(texts) -> dict:
    """Arc frames for every text (eval/replay_frames.py, word by word), cached on disk across runs."""
    from replay_frames import frame_for
    p = OUT / "kdir_frames.pkl"
    if not _FRAMES and p.exists():
        try:
            _FRAMES.update(pickle.loads(p.read_bytes()))
        except Exception:  # noqa: BLE001
            pass
    new = [t for t in texts if t not in _FRAMES]
    for t in new:
        f = frame_for(t)
        f.route = "web"
        _FRAMES[t] = f
    if new:
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(_FRAMES))
        tmp.replace(p)
    return {t: _FRAMES[t] for t in texts}


_DS: dict = {}


def director_signals() -> dict:
    """director id -> the director's outputs + Flux confidences at the question's last update (kdir_director_feats)."""
    if _DS:
        return _DS
    p = OUT / "kdir_dirfeats.npz"
    if not p.exists():
        return _DS
    z = np.load(p)
    names = list(z["names"])
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    steps = z["steps"]
    for i, qid in enumerate(z["ids"]):
        last = steps[off[i + 1] - 1]
        _DS[str(qid)] = {n: float(v) for n, v in zip(names, last)}
    return _DS


def _strong(docs) -> bool:
    return any((d.meta or {}).get("strong") for d in docs)


def _text(docs) -> str:
    return " ".join(f"{d.title} {d.text}" for d in docs[:5])


def _ids(docs) -> set:
    return {d.id for d in docs} | {d.url for d in docs if d.url}


# ------------------------------------------------------------------------------------------------ shared cells
# ------------------------------------------------------------------------------------------------ the loop
def simulate(data: KData, policy, emb_ms: float = 58.7, store: str = "pooled", feedback: bool = True,
             log: bool = False, seed: int = 0, correction_rate: float = 0.3, online=None, voi_model=None,
             theta: float = 0.95, allow=("pay", "local", "local_rc", "free")):
    """policy: "fresh" | "voi" | "oracle" | "oracle_local" | "behav" | a kdir.Policy. Returns (rows, logs, stats)."""
    rng = random.Random(seed)
    users: dict = {}
    sh = shc = None
    if store in ("user_shared", "user_shared_cells"):
        from server.keystone.shared import SQLiteShared
        sh = SQLiteShared()
    if store == "user_shared_cells":
        from server.keystone.shared import SharedCells
        shc = SharedCells()
    ks = Keystone(Flags(**FULL), shared=sh) if store == "pooled" and policy != "fresh" else None
    rows, logs = [], []
    fb_rows = []
    for it in data.stream:
        now = R.T0 + it["t"]
        if store != "pooled" and policy != "fresh":
            ks = users.get(it["user"])
            if ks is None:
                ks = users[it["user"]] = Keystone(Flags(**FULL), shared=sh)
        fx = data.fx[it["query"]]
        q = R.make_q(data, it)
        f = data.frames[it["query"]]
        fresh_docs = data.docs(it["query"])
        fresh_ids = data.fam_ids.get(it["fid"], set())
        fresh_ok = R.correct(it, _text(fresh_docs), {d.id for d in fresh_docs}, fresh_ids, {it["fid"]})
        tier = fx["tier"] or ("web" if fx["plan"]["lane"] != "card" else "card")
        web_q = tier == "web"
        row = {"pos": it["pos"], "fid": it["fid"], "user": it["user"], "tier": tier, "live": q.live,
               "has_expect": bool(it["expect"]), "pool": it.get("pool", ""), "form": it.get("form", "")}
        if tier == "card" or ks is None or q.qvec is None:
            ok = fresh_ok
            row.update(action="pay", path=tier, paid=int(web_q), lat_ms=float(fx["ms"]), correct=ok,
                       fresh_correct=fresh_ok, regret=False, corrected=False, rc=False)
            rows.append(row)
            if ks is not None and q.qvec is not None:
                ks.write(q, fresh_docs, data.dvec, tier=tier, now=now)
            continue
        plaus = ks.plausible(q)
        cands, feats = ks.lookup(q, now)
        if not plaus:
            cands = [c for c in cands if c.path in ("key", "shared")]
        if shc is not None and plaus and not any(c.path in ("key", "shared") for c in cands):
            sc = shc.lookup(q, now)
            if sc is not None:
                cands.append(sc)
        cost = feats["lookup_ms"] + (0.0 if (q.key and feats["key_hit"]) or not plaus else emb_ms)
        # ---- counterfactual outcomes
        free_docs = data.free_docs(it["query"]) if web_q else []
        free_ms = float(((fx.get("free") or {}).get("ms")) or 0.0)
        free_strong = _strong(free_docs)
        free_ok = R.correct(it, _text(free_docs), _ids(free_docs), fresh_ids, set()) if free_docs else False
        live_or_fresh = bool(q.live or getattr(f, "when", "") or fx["plan"].get("why", "").startswith("fresh"))
        catch = bool(free_strong and free_ok and not live_or_fresh)
        plan_paid = int(web_q)
        oc = {"pay": {"paid": plan_paid, "lat": float(fx["ms"]) + cost, "ok": fresh_ok, "regret": False,
                      "corrected": False}}
        if web_q and not live_or_fresh and fx.get("free") is not None:
            if free_strong:
                oc["free"] = {"paid": 0, "lat": free_ms + cost, "ok": free_ok,
                              "regret": bool(fresh_ok) and free_ok is False, "corrected": False}
            else:
                oc["free"] = {"paid": 1, "lat": free_ms + float(fx["ms"]) + cost, "ok": fresh_ok, "regret": False,
                              "corrected": False}
        cand_oc = []
        for c in cands:
            okc = R.correct(it, R._served_text(c), {x.doc_id for x, _ in c.cells} | {x.url for x, _ in c.cells if x.url},
                            fresh_ids, R._origins(data, c) if c.path != "shared_q" else _shared_origin(data, c))
            reg = bool(fresh_ok) and okc is False
            cand_oc.append({"ok": okc, "regret": reg, "y": R._noregret(okc, fresh_ok),
                            "rc_regret": reg and not catch, "rc_corrected": bool(okc is False and catch)})
        qx = cx = None
        if log or isinstance(policy, kdir.Policy) or policy == "behav":
            qx = kdir.question_vec(q, f, fx["plan"], ks, feats, cands, data.leads(it["query"]), data.director(it))
            cx = [kdir.cand_vec(c, feats, ks, cands, now, q) for c in cands]
        # ---- decide
        t_dec = time.perf_counter()
        act, cand, info = _decide(policy, cands, cand_oc, oc, feats, qx, cx, plan_paid, voi_model, theta, allow,
                                  web_q, live_or_fresh, rng)
        dec_ms = (time.perf_counter() - t_dec) * 1000
        kc = None
        if log and data.director(it) is not None and q.key is not None and q.key.complete and \
                any(c.path == "key" for c in cands):
            kc = prefix_keys(ks, it, q, cands, now)
        if log:
            logs.append({"kc": kc, "director_id": it.get("director_id") if data.director(it) else "",
                         "pos": it["pos"], "fid": it["fid"], "user": it["user"], "qx": qx, "cx": cx,
                         "paths": [c.path for c in cands], "cand": cand_oc, "oc": oc, "catch": catch,
                         "free_strong": bool(free_strong), "free_ok": free_ok if free_docs else None,
                         "has_free": fx.get("free") is not None and web_q, "lof": live_or_fresh,
                         "fresh_ok": fresh_ok, "plan_paid": plan_paid, "tier": tier, "has_expect": bool(it["expect"])})
        # ---- apply
        if act in ("local", "local_rc") and cand is not None:
            j = cands.index(cand)
            o = cand_oc[j]
            ok = o["ok"]
            rc = act == "local_rc"
            regret = o["rc_regret"] if rc else o["regret"]
            corrected = bool(rc and catch and ok is False)
            row.update(action=act, path=cand.path, paid=0, lat_ms=round(cost, 2), correct=ok if not corrected else
                       free_ok, fresh_correct=fresh_ok, regret=bool(regret), corrected=corrected, rc=rc,
                       served_ok=ok, p=round(getattr(cand, "p", 0.0) or 0.0, 4))
            if cand.path != "shared_q":
                ks.served(cand, now)
            ks.touch(q, now)
            top = cand.cells[0][0]
            if feedback:
                if rc and free_strong and not live_or_fresh:
                    kdir.feedback(ks, top.id, "caught" if corrected else ("confirm" if free_ok and ok else ""), top.url)
                if ok is False and not corrected and rng.random() < correction_rate:
                    kdir.feedback(ks, top.id, "correction", top.url)
                    if cx is not None:
                        fb_rows.append((qx, cx[j], 0.0))
                elif rc and free_strong and not live_or_fresh and cx is not None:
                    fb_rows.append((qx, cx[j], 0.0 if corrected else 1.0))
            if corrected:                              # the re-check's free answer is remembered
                ks.write(q, free_docs, data.dvec, tier="facts", now=now)
        elif act == "free" and "free" in oc:
            o = oc["free"]
            row.update(action="free", path="free" if o["paid"] == 0 else "web", paid=o["paid"], lat_ms=round(o["lat"], 2),
                       correct=o["ok"], fresh_correct=fresh_ok, regret=o["regret"], corrected=False, rc=False)
            ks.write(q, free_docs if o["paid"] == 0 else fresh_docs, data.dvec, tier="facts" if o["paid"] == 0 else tier,
                     now=now)
        else:
            o = oc["pay"]
            row.update(action="pay", path=tier, paid=plan_paid, lat_ms=round(o["lat"], 2), correct=fresh_ok,
                       fresh_correct=fresh_ok, regret=False, corrected=False, rc=False)
            ks.write(q, fresh_docs, data.dvec, tier=tier, now=now)
            if sh is not None and q.key is not None and q.key.complete and not q.key.subject.startswith("@"):
                R._publish_verified(sh, q, fresh_docs, now)
            if shc is not None:
                shc.publish_verified(q, fresh_docs, now, fid=it["fid"])
        row["dec_ms"] = round(dec_ms, 4)
        row.update({k: v for k, v in (info or {}).items() if k in ("p_ok", "p_catch")})
        rows.append(row)
        if online is not None and len(fb_rows) and it["pos"] % online["every"] == online["every"] - 1:
            online["refit"](policy, fb_rows)
    stats = ks.stats() if ks is not None and store == "pooled" else {"users": len(users)}
    if sh is not None:
        stats["shared"] = sh.stats()
    if shc is not None:
        stats["shared_cells"] = shc.stats()
    stats["feedback_rows"] = len(fb_rows)
    return rows, logs, stats


def prefix_keys(ks, it, q, cands, now) -> list:
    """Per word of the question: is the typed key complete, equal to the final key, and does the typed lookup already
    return the cell the final key candidate serves (the 'wait one more word' decision's label inputs)."""
    from server.keystone.relkey import key_of_frame
    target = next(c for c in cands if c.path == "key").cells[0][0].id
    out = []
    for pf in R.prefix_frames(it["query"]):
        k = key_of_frame(pf)
        comp = bool(k is not None and k.complete)
        same = bool(comp and k.text() == q.key.text())
        hit = False
        if comp and not k.subject.startswith("@"):
            _, vals = ks.g.lookup(k.subject, k.relation, now)
            hit = any(v.get("cell") == target for v in vals)
        elif same:
            hit = True
        has = False
        if comp and not k.subject.startswith("@"):
            node = ks.g.resolve(k.subject)
            has = bool(node and ks.g.edges.get((node, k.relation)))
        out.append((int(comp), int(same), int(hit), int(has)))
    return out


def _shared_origin(data, c) -> set:
    f = getattr(c, "fid", "")
    return {f} if f else set()


def _decide(policy, cands, cand_oc, oc, feats, qx, cx, plan_paid, voi_model, theta, allow, web_q, lof, rng):
    if policy in ("fresh",):
        return "pay", None, {}
    if policy == "voi":
        best = voi.choose(voi_model, [c if c.path != "shared_q" else _as_qnn(c) for c in cands], feats, theta)
        if best is not None:
            best = next((c for c in cands if c is best or getattr(best, "_orig", None) is c), best)
        return ("local", best, {}) if best is not None else ("pay", None, {})
    if policy in ("oracle", "oracle_local"):
        acts = allow if policy == "oracle" else ("pay", "local")
        best = None
        for a in LAT_ORDER:
            if a not in acts:
                continue
            for c, o in (list(zip(cands, cand_oc)) if a in ("local", "local_rc") else [(None, None)]):
                if a in ("local", "local_rc"):
                    if o["ok"] is None:
                        continue
                    J = (COST["eps"] if a == "local_rc" else 0.0) + COST["R"] * (o["rc_regret"] if a == "local_rc" else
                                                                               o["regret"]) + \
                        (COST["C"] * o["rc_corrected"] if a == "local_rc" else 0.0)
                    if a == "local" and o["ok"] is False and o["y"]:
                        J += 0.001                   # wrong but no regret (paying is wrong too): allowed, tie-break
                elif a == "free":
                    if "free" not in oc:
                        continue
                    J = oc["free"]["paid"] + COST["R"] * oc["free"]["regret"] + COST["eps"]
                else:
                    J = oc["pay"]["paid"]
                if best is None or J < best[0] - 1e-9:
                    best = (J, a, c)
        return best[1], best[2], {}
    if isinstance(policy, kdir.Policy):
        free_ok_q = web_q and not lof and "free" in oc
        a, c, info = policy.decide(qx, cands, cx, plan_paid, free_ok_q)
        if a not in allow:
            a = "local" if a == "local_rc" and "local" in allow else "pay"
        return a, c, info
    if policy == "behav":                        # exploration for logging: VOI-like with noise
        best = voi.choose(voi_model, cands, feats, theta) if voi_model else None
        r = rng.random()
        if best is not None:
            return ("local_rc" if r < 0.5 else "local"), best, {}
        if cands and r < 0.25:
            return "local", max(cands, key=lambda c: c.score), {}
        if "free" in oc and r < 0.5:
            return "free", None, {}
        return "pay", None, {}
    raise ValueError(policy)


def _as_qnn(c):
    d = Cand("qnn", c.cells, c.score, c.as_of, c.eq_p)
    d._orig = c
    return d


# ------------------------------------------------------------------------------------------------ summary
def summarize(rows, window=None):
    def agg(rs):
        n = len(rs)
        acc_e = [r["correct"] for r in rs if r["has_expect"] and r["correct"] is not None]
        acc_all = [r["correct"] for r in rs if r["correct"] is not None]
        lat = [r["lat_ms"] for r in rs]
        dec = [r["dec_ms"] for r in rs if "dec_ms" in r]
        acts = {}
        for r in rs:
            acts[r["action"]] = acts.get(r["action"], 0) + 1
        return {"n": n, "paid_per_q": round(sum(r["paid"] for r in rs) / n, 4),
                "local_share": round(sum(1 for r in rs if r["action"] in ("local", "local_rc")) / n, 4),
                "free_share": round(sum(1 for r in rs if r["action"] == "free" and r["paid"] == 0) / n, 4),
                "regret_rate": round(sum(1 for r in rs if r.get("regret")) / n, 4),
                "corrected_rate": round(sum(1 for r in rs if r.get("corrected")) / n, 4),
                "acc_expect": round(sum(acc_e) / len(acc_e), 4) if acc_e else None, "n_expect": len(acc_e),
                "acc_all": round(sum(acc_all) / len(acc_all), 4) if acc_all else None,
                "p50_ms": round(statistics.median(lat), 1), "p90_ms": round(sorted(lat)[int(0.9 * (n - 1))], 1),
                "dec_ms_p50": round(statistics.median(dec), 4) if dec else None,
                "dec_ms_p99": round(sorted(dec)[int(0.99 * (len(dec) - 1))], 4) if dec else None, "actions": acts}
    out = {"all": agg(rows)}
    if window:
        out["windows"] = [{"start": s, **agg(rows[s:s + window])} for s in range(0, len(rows), window)]
    return out


def baseline(data, policy, emb_ms):
    """replay.py's own policies (fresh / a / f) on this data, for comparability."""
    rows, _, st = R.run(data, policy, emb_ms, voi_model=voi.load(), theta=voi.load().theta)
    s = R.summarize(rows)["all"]
    return {"paid_per_q": s["paid_per_q"], "regret_rate": s["regret_rate"], "acc_expect": s["acc_expect"],
            "p50_ms": s["p50_ms"], "local_hit_rate": s["local_hit_rate"]}


def main(a):
    data = KData(a.stream)
    print(f"{a.stream}: {len(data.stream)} questions", flush=True)
    res = {"stream": a.stream, "policies": {}}
    vm = voi.load()
    for p in a.policies.split(","):
        t = time.time()
        if p == "a":
            res["policies"][p] = {"summary": {"all": baseline(data, "a", a.embed_ms)}}
            print(p, res["policies"][p]["summary"]["all"], flush=True)
            continue
        pol = kdir.load() if p == "kdir" else p
        rows, _, st = simulate(data, pol, a.embed_ms, store=a.store, voi_model=vm, theta=vm.theta)
        s = summarize(rows, 500)
        res["policies"][p] = {"summary": s, "stats": st}
        al = s["all"]
        print(f"{p:12s} paid/q {al['paid_per_q']:.3f} local {al['local_share']:.2f} free {al['free_share']:.2f} "
              f"regret {al['regret_rate']:.4f} corrected {al['corrected_rate']:.4f} acc(exp) {al['acc_expect']} "
              f"acc {al['acc_all']} p50 {al['p50_ms']} {al['actions']} [{time.time() - t:.0f}s]", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=str(OUT / "stream.jsonl"))
    ap.add_argument("--policies", default="fresh,voi,oracle_local,oracle")
    ap.add_argument("--store", default="pooled")
    ap.add_argument("--embed-ms", type=float, default=58.7)
    ap.add_argument("--out", default="")
    main(ap.parse_args())
