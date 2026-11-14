"""Keystone director (server/keystone/kdir.py) and learned TTL steps of a training cycle. Text-level replays only
(WEB_BACKEND=wikipedia, the recorded counterfactual fixture: no web or answer-model calls).

kdir:  one more DAgger round per cycle. The train stream is re-logged under the CURRENT policy (its own store states:
       new training rows every cycle), aggregated with every earlier round, the trees refit (eval/keystone/kdir_train
       .train, the shipped feature set), R re-picked by real dev replays (cheapest R whose dev regret stays <= today:
       the search cache / VOI gate, kdir_eval.part_pick's rule), then current and candidate are replayed on the FIXED
       held-out test streams (kdir_test 2,000 q, the old 800-q stream) with the same store / feedback settings.
ttl:   Wikidata statement histories (start / end qualifiers) per relation, split by statement hash: 4/5 fit the
       censored-exponential change rate, 1/5 held out. Held-out log-likelihood under the current vs candidate TTL
       table, plus the kdir test-stream replay with the learned TTLs on (KDIR_TTL) for both tables.
Inputs (read-only) come from the kdir recording (KDIR_DATA, default ../headstart-kdir/runs/index), linked into
runs/index; outputs go to the cycle dir. Gates: eval/train/gates.kdir / gates.ttl.

  .venv/bin/python eval/train/kdir_step.py kdir|ttl --cycle N --out runs/train/cycles/cN/kdir
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import zlib
from pathlib import Path

os.environ["WEB_BACKEND"] = "wikipedia"
os.environ["TAVILY_API_KEY"] = ""
os.environ.setdefault("INDEX", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.sys_path()

KDATA = Path(os.environ.get("KDIR_DATA", C.MAIN.parent / "headstart-kdir/runs/index"))
IDX = C.ROOT / "runs/index"
LINK = ["kdir_fixture.json", "kdir_vecs.npz", "fixture.json", "fixture_vecs.npz", "kdir_frames.pkl", "kdir_dirfeats.npz",
        "kdir_train.jsonl", "kdir_dev.jsonl", "kdir_test.jsonl", "stream.jsonl", "stream_s2.jsonl", "kdir_families.jsonl",
        "kdir_log_train_r0.pkl", "kdir_log_train_r1.pkl", "kdir_log_dev_voi.pkl", "kdir_log_old_dev_voi.pkl",
        "kdir_log_test_voi.pkl", "kdir_log_old_test_voi.pkl", "families.jsonl"]
TEST_STREAMS = ("test", "old_test")
SUMMARY_KEYS = ("n", "paid_per_q", "regret_rate", "corrected_rate", "acc_expect", "acc_all", "p50_ms", "p90_ms",
                "local_share")


def link_inputs() -> list[str]:
    """Symlink the recorded inputs into runs/index (never copied, never written). Returns what is missing."""
    IDX.mkdir(parents=True, exist_ok=True)
    missing = []
    for n in LINK:
        dst, src = IDX / n, KDATA / n
        if dst.exists() or dst.is_symlink():
            continue
        if src.exists():
            dst.symlink_to(src)
        else:
            missing.append(n)
    res = IDX / "kdir_results.json"
    if not res.exists() and (KDATA / "kdir_results.json").exists():
        res.write_text((KDATA / "kdir_results.json").read_text())      # a copy: kdir_eval may write to it
    return missing


def _summ(s: dict) -> dict:
    a = s["all"]
    return {k: a.get(k) for k in SUMMARY_KEYS}


def replay(name: str, md: dict, stream: str, ttl: bool = False) -> dict:
    import kdir_eval as KE
    s, _ = KE.run_policy(stream, name, {name: md}, ttl=ttl)
    return _summ(s)


def kdir_main(cycle: int, out: Path, r_grid=None) -> dict:
    import kdir_eval as KE
    import kdir_train as T
    from server.keystone import kdir
    missing = link_inputs()
    need = {"kdir_fixture.json", "kdir_train.jsonl", "kdir_test.jsonl", "kdir_dev.jsonl"}
    if need & set(missing):
        return {"skip": f"kdir recording missing: {sorted(need & set(missing))}"}
    cur = json.loads(kdir.MODEL_PATH.read_text())
    train_path = KE.STREAMS["dev"].with_name("kdir_train.jsonl")
    # DAgger: a round under the current policy (tagged by the policy's hash: an unchanged policy reuses its round)
    import hashlib
    tag = "train_p" + hashlib.sha1(kdir.MODEL_PATH.read_bytes()).hexdigest()[:10]
    sig = [tag] + sorted(p.name for p in IDX.glob("kdir_log_train_*.pkl"))      # what a refit would train on
    st_p = C.TRAIN / "kdir_state.json"
    if (IDX / f"kdir_log_{tag}.pkl").exists() and C.read_json(st_p, {}).get("sig") == sig:
        return {"skip": "no new kdir training rows since the last refit (same policy, same DAgger rounds)"}
    new = T.log_stream(train_path, kdir.Policy({k: cur[k] for k in ("ok", "catch", "free_strong", "free_ok", "R", "C",
                                                                     "eps", "actions") if k in cur}), tag)
    logs = []
    for p in sorted(IDX.glob("kdir_log_train_*.pkl")):
        import pickle
        logs += pickle.loads(p.read_bytes())["logs"]
    dv = KE.logs_for("dev_voi", KE.STREAMS["dev"], "voi")
    odv = KE.logs_for("old_dev_voi", KE.STREAMS["old_dev"], "voi")
    pol, rep = T.train(logs, dv["logs"] + odv["logs"], "all-no-director")
    base = {**pol, "actions": list(cur.get("actions") or kdir.ACTIONS), "C": cur.get("C", 1.0), "eps": cur.get("eps", 0.02)}
    r = KE.res_load()
    target = {"dev": max(r["cache_dev"]["regret_rate"], r["voi_dev_regret"]), "old_dev": r["cache_old_dev"]["regret_rate"]}
    pts = []
    for R in r_grid or sorted({max(4, round(cur["R"] * f)) for f in (0.67, 0.8, 1.0, 1.25, 1.5, 2.0)}):
        row = {"R": R}
        for sk in ("old_dev", "dev"):
            row[sk] = replay("cand", {**base, "R": R}, sk)
        row["ok"] = all(row[k]["regret_rate"] <= target[k] + 1e-9 for k in target)
        pts.append(row)
        C.log(f"  kdir R={R} dev {row['dev']['paid_per_q']}/{row['dev']['regret_rate']} "
              f"old_dev {row['old_dev']['paid_per_q']}/{row['old_dev']['regret_rate']} ok={row['ok']}")
    ok = [p for p in pts if p["ok"]]
    if not ok:
        return {"skip": "no R keeps dev regret at today's level", "pick": pts, "train_report": rep}
    best = min(ok, key=lambda p: (p["dev"]["paid_per_q"] + p["old_dev"]["paid_per_q"], p["R"]))
    cand = {k: v for k, v in base.items() if k in ("ok", "catch", "free_strong", "free_ok", "C", "eps", "actions")}
    cand.update(R=best["R"], features=kdir.NAMES, name=cur.get("name", "3c_nodir") + f"+c{cycle}",
                lat_value=cur.get("lat_value", 0.5), R_rule=cur.get("R_rule", ""),
                trained=f"eval/train/kdir_step.py cycle {cycle}: DAgger rounds {len(list(IDX.glob('kdir_log_train_*.pkl')))}")
    m = {"current": {}, "candidate": {}}
    for s in TEST_STREAMS:
        m["current"][s] = replay("cur", cur, s)
        m["candidate"][s] = replay("cand", cand, s)
    C.write_json(out / "kdir_model.json", cand)
    C.write_json(st_p, {"sig": [tag] + sorted(p.name for p in IDX.glob("kdir_log_train_*.pkl"))})
    return {**m, "train_report": rep, "rows": {"train_decisions": sum(len(L["cx"]) for L in logs),
                                                 "new_decisions": len(new["logs"])},
            "R": {"current": cur["R"], "candidate": best["R"]}, "pick": pts}


# ------------------------------------------------------------------------------------------------ TTLs
def heldout(sid: str) -> bool:
    return zlib.crc32(sid.encode()) % 5 == 0


def exposure(b: dict, now) -> tuple[float, int] | None:
    import fit_ttl as FT
    s = FT._t(b["start"]["value"])
    e = FT._t(b["end"]["value"]) if "end" in b else None
    if s is None or s.year < 1800 or s > now:
        return None
    end = e if e is not None and s < e <= now else now
    return max(0.0, (end - s).days / 365.25), int(e is not None and s < e <= now)


def loglik(lam: float, obs: list[tuple[float, int]]) -> float:
    lam = max(lam, 1e-6)
    return sum(e * math.log(lam) - lam * y for y, e in obs)


def lam_of_ttl(ttl_s: float) -> float:
    return -math.log(1 - 0.05) / (ttl_s / (365.25 * 86400))


async def statement_histories() -> dict:
    import httpx

    import fit_ttl as FT
    from server import knowledge
    from server.keystone.relkey import FROM_PROP
    out = {}
    async with httpx.AsyncClient() as client:
        for prop, rel in FROM_PROP.items():
            pid = knowledge.PROPS[prop][0][0]
            se = await FT.q(client, f"""SELECT ?s ?start ?end WHERE {{ ?i p:{pid} ?s . ?s pq:P580 ?start .
                                        OPTIONAL {{ ?s pq:P582 ?end }} }} LIMIT 1500""")
            smp = await FT.q(client, f"""SELECT ?s ?st WHERE {{ ?i p:{pid} ?s . OPTIONAL {{ ?s pq:P580 ?st }} }} LIMIT 1500""")
            out[rel] = {"se": se, "n_all": len(smp), "n_with": sum(1 for b in smp if "st" in b)}
            await asyncio.sleep(1.0)
    return out


def ttl_fit(hist: dict, cur_ttl: dict, now) -> tuple[dict, dict]:
    """Candidate table (current table with the statement-history relations refit on the 4/5 split) + held-out
    log-likelihood per observation for current and candidate."""
    import fit_ttl as FT
    cand = dict(cur_ttl)
    lc, lk, nobs, per = 0.0, 0.0, 0, {}
    for rel, h in hist.items():
        tr, te = [], []
        for b in h["se"]:
            x = exposure(b, now)
            if x is None:
                continue
            (te if heldout(b["s"]["value"]) else tr).append(x)
        share = h["n_with"] / h["n_all"] if h["n_all"] else 0.0
        w = 1.0 if share > 0.02 else share / 0.02
        years = sum(y for y, _ in tr)
        lam = (sum(e for _, e in tr) / years * w) if years > 0 else 0.0
        ttl = FT.HI if lam <= 0 else -math.log(1 - FT.P_CHANGE) / lam * 365.25 * 86400
        if rel in cur_ttl and len(tr) >= 50:
            cand[rel] = round(min(FT.HI, max(FT.LO, ttl)))
        if te and rel in cur_ttl:
            a, b = loglik(lam_of_ttl(cur_ttl[rel]), te), loglik(lam_of_ttl(cand[rel]), te)
            lc, lk, nobs = lc + a, lk + b, nobs + len(te)
            per[rel] = {"n_fit": len(tr), "n_heldout": len(te), "ttl_cur_d": round(cur_ttl[rel] / 86400, 1),
                        "ttl_cand_d": round(cand[rel] / 86400, 1)}
    return cand, {"current": round(lc / max(1, nobs), 5), "candidate": round(lk / max(1, nobs), 5), "n_obs": nobs,
                  "relations": per}


def ttl_main(cycle: int, out: Path) -> dict:
    import datetime as dt

    from server.keystone import kdir
    cur_doc = json.loads(kdir.TTL_PATH.read_text())
    # statement histories move slowly and the public SPARQL endpoint is slow: fetched at most weekly, cached
    hp = C.TRAIN / "ttl_histories.json"
    cached = C.read_json(hp, {})
    import time as _t
    if cached.get("hist") and _t.time() - cached.get("ts", 0) < 7 * 86400:
        hist = cached["hist"]
    else:
        hist = asyncio.run(statement_histories())
        if sum(len(h["se"]) for h in hist.values()) == 0:
            return {"skip": "Wikidata SPARQL returned no statement histories (timeouts)"}
        C.write_json(hp, {"ts": _t.time(), "hist": hist})
    now = dt.datetime.now(dt.timezone.utc)
    cand_ttl, ll = ttl_fit(hist, cur_doc["ttl_s"], now)
    cand_doc = {**cur_doc, "ttl_s": cand_ttl, "source": f"eval/train/kdir_step.py ttl, cycle {cycle} (4/5 statements)"}
    m = {"current": {"heldout": {"loglik_per_obs": ll["current"]}}, "candidate": {"heldout": {"loglik_per_obs": ll["candidate"]}},
         "heldout": ll}
    missing = link_inputs()
    if "kdir_test.jsonl" not in missing:
        cur_pol = json.loads(kdir.MODEL_PATH.read_text())
        for name, table in (("current", cur_doc["ttl_s"]), ("candidate", cand_ttl)):
            kdir._ttl = dict(table)
            for s in TEST_STREAMS:
                m[name][s] = replay("cur", cur_pol, s, ttl=True)
        kdir._ttl = None
    C.write_json(out / "ttl_model.json", cand_doc)
    return m


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["kdir", "ttl"])
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rep = kdir_main(a.cycle, out) if a.what == "kdir" else ttl_main(a.cycle, out)
    except Exception as e:  # noqa: BLE001  (one failed step must not end the cycle)
        import traceback
        rep = {"error": f"{type(e).__name__}: {e}"[:400], "trace": traceback.format_exc()[-2000:]}
    C.write_json(out / "metrics.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("pick", "trace")}, default=str)[:3000])
    return rep


if __name__ == "__main__":
    main()
