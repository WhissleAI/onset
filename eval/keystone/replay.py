"""Workload replay: the retrieval DECISION for every question of a multi-user stream, per policy, text-level (no audio,
no answer model, no Tavily). The recorded fixture (record_fixture.py) is what today's path returns for each question;
"paid" = a Tavily-equivalent web call (the fixture's web tier). Policies:

  fresh   no reuse at all (every web-tier question pays)
  a       today: the semantic search cache (server/searchcache.py semantics: exact QKey, embedding NN >= 0.92 with the
          same topic + subject + numbers, TTL by class, 400 entries)
  b       a + memory cells: dense + BM25 passage search over everything retrieved so far, question-vector NN (hand gate)
  c       b + Arc-keyed typed edges (subject, relation) -> value cells, Wikidata facts as typed triples
  d       c + the learned answer-equivalence hash
  e       d + activation (session / user memory as graph activation, ranking bias) and ONE structure: the separate
          search cache is retired (its job is the value cells)
  f       e + the calibrated value-of-information gate (keystone/voi.py) instead of the hand gate

Card questions (weather, fx, time, units) always take the live card (free). Facts-tier questions (free Wikidata /
local leads) may be answered locally when the gate accepts (a latency win, not a cost win).

Per question: path, paid, latency (recorded network ms for fresh lanes; measured lookup compute + a measured
query-embedding p50 for local paths; the lookup overhead is ALSO charged to every miss), correctness:
  expect   (gold or LLM-silver answer keywords): every keyword group found in the served passages
  silver   (no keywords): the served passages share a document with what today's fresh path retrieved
           (top 3) for that question (same evidence); unknown when the fresh path returned nothing

    .venv/bin/python eval/keystone/replay.py --stream runs/index/stream.jsonl --policies fresh,a,b,c,d,e,f
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np  # noqa: E402

from server import config, lanes, searchcache  # noqa: E402
from server.keystone import ahash, voi  # noqa: E402
from server.keystone.core import Flags, Keystone, Q  # noqa: E402
from server.keystone.relkey import key_of_frame  # noqa: E402

OUT = ROOT / "runs/index"
T0 = 1_800_000_000.0
LAST: dict = {}
POLICIES = {
    "b": dict(cells=True, key=False, hash=False, act=False, qnn=True, verify=False),
    "c": dict(cells=True, key=True, hash=False, act=False, qnn=True, verify=False),
    "d": dict(cells=True, key=True, hash=True, act=False, qnn=True),
    "e": dict(cells=True, key=True, hash=True, act=True, qnn=True),
    "f": dict(cells=True, key=True, hash=True, act=True, qnn=True),
    "oracle": dict(cells=True, key=True, hash=True, act=True, qnn=True),
}


class Data:
    def __init__(self, stream_path):
        self.stream = [json.loads(x) for x in Path(stream_path).read_text().splitlines() if x.strip()]
        self.fx = json.loads((OUT / "fixture.json").read_text())
        z = np.load(OUT / "fixture_vecs.npz")
        self.dvec = {k: v.astype(np.float32) for k, v in zip(z["doc_ids"], z["doc_vecs"])}
        self.qvec = {k: v.astype(np.float32) for k, v in zip(z["q_texts"], z["q_vecs"])}
        # silver evidence per family: what today's fresh path retrieved (top 3) for ANY wording of the question
        self.fam_ids: dict = {}
        self.text_fid: dict = {}
        for it in self.stream:
            self.text_fid.setdefault(it["query"], it["fid"])
            self.fam_ids.setdefault(it["fid"], set()).update(d["id"] for d in self.fx[it["query"]]["docs"][:3])
        from replay_frames import frame_for
        self.frames = {}
        for it in self.stream:
            if it["query"] not in self.frames:
                f = frame_for(it["query"])
                f.route = "web"
                self.frames[it["query"]] = f

    def docs(self, text):
        out = []
        for d in self.fx[text]["docs"]:
            out.append(SimpleNamespace(id=d["id"], title=d["title"], text=d["text"], url=d["url"], source=d["source"],
                                       meta=d.get("meta") or {}))
        return out


def prefix_frames(text: str) -> list:
    """The frame after each word, as live (one tagger pass, eval/replay_frames.py logic)."""
    from replay_frames import tagger
    from server import frame as framemod
    from server.session import clean
    t = tagger()
    t.reset()
    words = [w for w in (clean(x) for x in text.split()) if w]
    spans, current, events, prev_intent, out = [], None, set(), None, []
    for i in range(len(words)):
        t.push(words[i:i + 1])
        r = t.poll()
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            current = r["current"]
        for u in r.get("utterances") or []:
            current = u
        for e in r.get("events") or []:
            events.add(e["type"])
        f = framemod.build(words[:i + 1], spans, current, events, prev_intent)
        if f.intent.startswith("discourse.intent."):
            prev_intent = (f.intent, f.intent_p)
        out.append(f)
    return out


def key_completion(ks, it, q, served, now) -> dict:
    """C: streaming key completion. Word at which the relation is known, the full key is known (= final key from then
    on), and the typed lookup returns the cell the final question was served from (a local hit ready mid-sentence)."""
    pf = prefix_frames(it["query"])
    n = len(pf)
    keys = [key_of_frame(f) for f in pf]
    final = q.key
    out = {"n_words": n, "rel_at": None, "key_at": None, "hit_at": None}
    if final is None or not final.complete:
        return out
    for i in range(n - 1, -1, -1):          # earliest word from which the relation / key stay equal to the final
        if keys[i] is not None and keys[i].relation == final.relation:
            out["rel_at"] = i + 1
        else:
            break
    for i in range(n - 1, -1, -1):
        if keys[i] is not None and keys[i].text() == final.text():
            out["key_at"] = i + 1
        else:
            break
    if served is not None and served.path == "key":
        target = served.cells[0][0].id
        for i, k in enumerate(keys):
            if k is not None and k.complete:
                _, vals = ks.g.lookup(k.subject, k.relation, now)
                if any(v.get("cell") == target for v in vals):
                    out["hit_at"] = i + 1
                    break
    return out


def embed_ms(n=40) -> float:
    from server import embed
    embed.encode_queries(["warm up"])
    ts = []
    for i in range(n):
        t = time.perf_counter()
        embed.encode_queries([f"how tall is building number {i}"])
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def _match(groups, text: str) -> bool:
    low = text.lower()
    return all(any(re.search(r"(?<![a-z0-9])" + re.escape(alt.strip().lower()), low) for alt in g.split("|"))
               for g in groups)


def correct(item, served_text: str, served_ids: set, fresh_ids, origin_fids=()) -> bool | None:
    """expect keywords when the question has them; else silver: the served answer came from a question of the same
    family (same answer by construction), or the served passages share a document with what the fresh path retrieved
    for any wording of this question."""
    if item["expect"]:
        return _match(item["expect"], served_text)
    if item["fid"] in origin_fids:
        return True
    if not fresh_ids:
        return None
    return bool(served_ids & set(fresh_ids))


def make_q(data: Data, it) -> Q:
    f = data.frames[it["query"]]
    k = key_of_frame(f)
    fx = data.fx[it["query"]]
    live = fx["tier"] == "card" or searchcache.ttl_class(searchcache.qkey(f)) == "live" or bool(f.when)
    arc = ahash.arc_of(f.words, f.intent, f.slots, f.acts, live, k)
    return Q(text=it["query"], words=list(f.words), slots=tuple(f.slots), intent=f.intent, acts=tuple(f.acts),
             live=live, qvec=data.qvec.get(it["query"]), key=k, arc=arc, user=it["user"], session=it["session"])


class MiniCache:
    """server/searchcache.py semantics, on a simulated clock."""

    def __init__(self):
        self.e: dict = {}

    def get(self, key, qvec, now):
        x = self.e.get(key.text())
        if x and x["exp"] > now:
            return x, "exact"
        best = None
        for y in self.e.values():
            if y["exp"] > now and y["key"].topic == key.topic and y["key"].nums == key.nums and \
                    y["key"].subject == key.subject and y["key"].text() != key.text():
                s = max(float(v @ qvec) for v in y["qv"])
                if s >= config.CACHE_NN_SIM and (best is None or s > best[1]):
                    best = (y, s)
        return (best[0], "nn") if best else (None, None)

    def put(self, key, docs, qvec, now, fid=""):
        self.e[key.text()] = {"key": key, "docs": docs, "t": now, "exp": now + searchcache.ttl_s(key), "qv": [qvec],
                              "fid": fid}
        if len(self.e) > config.CACHE_MAX:
            for k in sorted(self.e, key=lambda k: self.e[k]["t"])[: len(self.e) - config.CACHE_MAX]:
                del self.e[k]


def run(data: Data, policy: str, emb_ms: float, voi_model=None, log_feats=False, theta=0.5, fold_models=None,
        anytime=False):
    ks = None
    if policy in POLICIES:
        ks = Keystone(Flags(**POLICIES[policy]))
    cache = MiniCache() if policy in ("a", "b", "c", "d") else None
    rows, feat_rows = [], []
    for it in data.stream:
        now = T0 + it["t"]
        fx = data.fx[it["query"]]
        q = make_q(data, it)
        f = data.frames[it["query"]]
        fresh_docs = data.docs(it["query"])
        fresh_ids = data.fam_ids.get(it["fid"], set())
        fresh_ok = correct(it, " ".join(f"{d.title} {d.text}" for d in fresh_docs[:5]), {d.id for d in fresh_docs},
                           fresh_ids, {it["fid"]})
        tier = fx["tier"] or ("web" if fx["plan"]["lane"] != "card" else "card")
        web_q = tier == "web"
        row = {"pos": it["pos"], "fid": it["fid"], "user": it["user"], "tier": tier, "live": q.live,
               "has_expect": bool(it["expect"]), "expect_src": it["expect_src"], "key": q.key.text() if q.key else ""}
        served, path, lat, paid = fresh_docs, tier, float(fx["ms"]), 1 if web_q else 0
        overhead = 0.0
        local = None
        if tier != "card":
            # (a) today's search cache: consulted for web-plan questions only
            if cache is not None and fx["plan"]["lane"] == "web" and q.qvec is not None:
                k = searchcache.qkey(f)
                hit, how = cache.get(k, q.qvec, now)
                if hit is not None:
                    served, path, paid = hit["docs"], "cache-" + how, 0
                    lat = 1.0 if how == "exact" else emb_ms + 1.0
            if ks is not None and path == tier and q.qvec is not None:
                plaus = ks.plausible(q)
                cands, feats = ks.lookup(q, now)
                if not plaus:              # live path: no embedding, no lookup beyond the typed key
                    cands = [c for c in cands if c.path == "key"]
                cost = feats["lookup_ms"] + (0.0 if (q.key and feats["key_hit"]) or not plaus else emb_ms)
                if policy == "oracle":     # full information: serve a candidate only if it is no-regret correct
                    best = next((c for c in cands if _noregret(correct(it, _served_text(c), {x.doc_id for x, _ in
                                                                                             c.cells}, fresh_ids,
                                                                       _origins(data, c)), fresh_ok)), None)
                elif policy == "f" and voi_model is not None:
                    m = (fold_models or {}).get(it["fid"], voi_model) if fold_models else voi_model
                    best = voi.choose(m, cands, feats, theta)
                else:
                    best = ks.heuristic(cands, feats)
                if log_feats:
                    for c in cands:
                        st = _served_text(c)
                        feat_rows.append({"pos": it["pos"], "fid": it["fid"], "path": c.path,
                                          "x": voi.vector(feats, c), "feats": feats,
                                          "y": _noregret(correct(it, st, {x.doc_id for x, _ in c.cells},
                                                                 fresh_ids, _origins(data, c)), fresh_ok)})
                if best is not None:
                    local = best
                    served = None
                    path, paid, lat = best.path, 0, cost
                    ks.served(best, now)
                    ks.touch(q, now)
                else:
                    overhead = cost
            lat += overhead
        if local is not None:
            st = _served_text(local)
            ids = {c.doc_id for c, _ in local.cells}
            origins = _origins(data, local)
            row["as_of_age_s"] = round(now - local.as_of)
        else:
            st = " ".join(f"{d.title} {d.text}" for d in served[:5])
            ids = {d.id for d in served}
            origins = {hit["fid"]} if path.startswith("cache") else ({it["fid"]} if path in ("web", "facts", "card")
                                                                    else set())
        ok = correct(it, st, ids, fresh_ids, origins)
        row.update(path=path, paid=paid, lat_ms=round(lat, 2), correct=ok, fresh_correct=fresh_ok,
                   regret=bool(fresh_ok) and ok is False, origin=sorted(origins)[:1])
        if anytime and ks is not None and q.key is not None:
            row["kc"] = key_completion(ks, it, q, local, now)
        rows.append(row)
        # writes: what was fetched fresh is remembered (local hits are not re-written)
        if local is None and path in ("web", "facts", "card"):
            if ks is not None:
                ks.write(q, fresh_docs, data.dvec, tier=tier, now=now)
            if cache is not None and path == "web" and fx["plan"]["lane"] == "web" and fresh_docs and \
                    q.qvec is not None:
                cache.put(searchcache.qkey(f), fresh_docs, q.qvec, now, it["fid"])
    stats = ks.stats() if ks is not None else {}
    LAST["ks"] = ks
    return rows, feat_rows, stats


def _noregret(ok, fresh_ok):
    """VOI label: serving locally is no worse than paying (local right, or the paid path would be wrong too)."""
    if ok is None:
        return None
    return bool(ok) or fresh_ok is False


def _origins(data, cand) -> set:
    """Families whose question wrote the served VALUE cell (passages are judged by evidence instead)."""
    c = cand.cells[0][0]
    if c.kind == "value" and cand.path in ("key", "hash", "qnn"):
        f = data.text_fid.get(c.question)
        return {f} if f else set()
    return set()


def _served_text(c) -> str:
    return " ".join(f"{x.title} {x.payload} {x.full}" for x, _ in c.cells[:5])


def summarize(rows, window=100):
    n = len(rows)

    def agg(rs):
        acc_e = [r["correct"] for r in rs if r["has_expect"] and r["correct"] is not None]
        acc_s = [r["correct"] for r in rs if not r["has_expect"] and r["correct"] is not None]
        loc = [r for r in rs if r["path"] in ("key", "hash", "qnn", "dense") or r["path"].startswith("cache")]
        lat = [r["lat_ms"] for r in rs]
        return {"n": len(rs), "paid_per_q": round(sum(r["paid"] for r in rs) / len(rs), 4),
                "local_hit_rate": round(len(loc) / len(rs), 4),
                "p50_ms": round(statistics.median(lat), 1), "p90_ms": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 1),
                "mean_ms": round(sum(lat) / len(lat), 1),
                "acc_expect": round(sum(acc_e) / len(acc_e), 4) if acc_e else None, "n_expect": len(acc_e),
                "agree_silver": round(sum(acc_s) / len(acc_s), 4) if acc_s else None, "n_silver": len(acc_s),
                "regret_rate": round(sum(1 for r in rs if r.get("regret")) / len(rs), 4),
                "local_acc": round(sum(1 for r in loc if r["correct"]) / max(1, sum(1 for r in loc
                                                                                  if r["correct"] is not None)), 4),
                "paths": _count(r["path"] for r in rs),
                "path_acc": {p: round(sum(1 for r in rs if r["path"] == p and r["correct"]) /
                                      max(1, sum(1 for r in rs if r["path"] == p and r["correct"] is not None)), 3)
                             for p in sorted({r["path"] for r in rs})}}
    out = {"all": agg(rows), "windows": []}
    for s in range(0, n, window):
        out["windows"].append({"start": s, **agg(rows[s:s + window])})
    return out


def _count(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def sweep(a):
    """Operating curve of the VOI gate: paid calls vs regret / accuracy as theta moves (test stream)."""
    data = Data(a.stream)
    emb = a.embed_ms if a.embed_ms is not None else embed_ms()
    vm = voi.load()
    out = {"embed_ms": emb, "points": []}
    for p in ("fresh", "a"):
        rows, _, _ = run(data, p, emb)
        out[p] = summarize(rows)["all"]
    for th in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]:
        rows, _, st = run(data, "f", emb, voi_model=vm, theta=th)
        al = summarize(rows)["all"]
        out["points"].append({"theta": th, **{k: al[k] for k in ("paid_per_q", "local_hit_rate", "p50_ms", "acc_expect",
                                                                 "agree_silver", "regret_rate", "local_acc")}})
        print(th, out["points"][-1], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("->", a.out)


def save_store(ks, d: Path):
    """A replayed store as the live app's KEYSTONE_DIR (warm-index A/B). Cell times are re-based to now (the replay
    runs on a simulated clock in the future)."""
    import numpy as _np
    shift = time.time() - max((c.t for c in ks.cells.cells.values()), default=time.time())
    for c in ks.cells.cells.values():
        c.t += shift
    for vals in ks.g.edges.values():
        for v in vals:
            v["t"] += shift
    d.mkdir(parents=True, exist_ok=True)
    ks.cells.save(d)
    (d / "graph.json").write_text(json.dumps(ks.g.to_json(), default=list))
    ids = sorted(ks.codes)
    _np.savez(d / "codes.npz", ids=_np.array(ids, int), codes=_np.array([ks.codes[i] for i in ids], _np.uint8))
    print("store ->", d, ks.stats())


def main(a):
    if a.sweep:
        return sweep(a)
    data = Data(a.stream)
    emb = a.embed_ms if a.embed_ms is not None else embed_ms()
    print(f"stream {a.stream}: {len(data.stream)} questions; query-embedding p50 {emb:.1f} ms", flush=True)
    res = {"stream": a.stream, "embed_ms": emb, "policies": {}}
    vm = voi.load()
    for p in a.policies.split(","):
        t = time.time()
        rows, _, st = run(data, p, emb, voi_model=vm, theta=a.theta if a.theta is not None else (vm.theta if vm else 0.5),
                          anytime=(p == a.anytime))
        s = summarize(rows)
        res["policies"][p] = {"summary": s, "store": st, "rows": rows}
        if a.save_store and p == "f" and LAST.get("ks") is not None:
            save_store(LAST["ks"], Path(a.save_store))
        al = s["all"]
        print(f"{p:6s} paid/q {al['paid_per_q']:.3f}  local {al['local_hit_rate']:.2f}  p50 {al['p50_ms']:.0f} ms  "
              f"acc(expect) {al['acc_expect']}  agree(silver) {al['agree_silver']}  local_acc {al['local_acc']}  "
              f"regret {al['regret_rate']}  "
              f"paths {al['paths']}  acc {al['path_acc']} [{time.time() - t:.0f}s]", flush=True)
    Path(a.out).write_text(json.dumps(res))
    print("->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=str(OUT / "stream.jsonl"))
    ap.add_argument("--policies", default="fresh,a,b,c,d,e,f,oracle")
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--anytime", default="f", help="policy whose replay also measures streaming key completion")
    ap.add_argument("--embed-ms", type=float, default=None)
    ap.add_argument("--out", default=str(OUT / "replay.json"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--save-store", default="")
    main(ap.parse_args())
