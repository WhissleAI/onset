"""Question families for the Keystone workload + the answer-equivalence hash: one family = one answer.

Sources (all in the repo): eval/questions.jsonl (50), eval/showcase/questions.jsonl (30), eval/multipart (26 + their
parts), eval/conversations.jsonl (21 two-turn), and a stratified sample of runs/director_queries/part_*.jsonl (~15K
spoken-search texts; follow-up chains kept as pairs).

For each family the answer model (Haiku, ~25 questions per call, JSON out) writes:
  paraphrases : 3 spoken rewordings with the SAME answer (positives for the hash, paraphrase traffic for the stream)
  near_miss   : 1 question about the same main entity with a DIFFERENT answer (hard negative)
  resolved    : for a follow-up, the question made self-contained (what memory resolution produces)
  expect      : for stable factual questions without labels, 1-3 short answer keywords (marked expect_src="llm",
                silver); [] for open-ended / live questions
  live        : the answer changes within days (news, scores, prices, weather, opening hours)

    .venv/bin/python eval/keystone/make_families.py [--n-director 380] [--out runs/index/families.jsonl]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server import answer, config  # noqa: E402

DQ = Path(__file__).resolve().parents[2] / "runs" / "director_queries"
SKIP_KINDS = {"not_search", "work_email_calendar", "smart_home_devices", "in_car_navigation", "names_homophones"}
STYLES = {"plain", "filler", "numbers", "short"}


def load(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def eval_families():
    fam = []
    for q in load(ROOT / "eval/questions.jsonl"):
        fam.append({"fid": q["id"], "canon": q["text"], "kind": q["kind"], "expect": q.get("expect") or [],
                    "expect_src": "gold" if q.get("expect") else "", "source": "eval58"})
    for q in load(ROOT / "eval/showcase/questions.jsonl"):
        fam.append({"fid": q["id"], "canon": q["text"], "kind": q["kind"], "expect": q.get("expect") or [],
                    "expect_src": "gold" if q.get("expect") else "", "source": "showcase"})
    for q in load(ROOT / "eval/multipart/questions.jsonl"):
        fam.append({"fid": q["id"], "canon": q["text"], "kind": "multipart:" + q["kind"],
                    "expect": q.get("overall") or [], "expect_src": "gold" if q.get("overall") else "",
                    "source": "multipart"})
        for j, p in enumerate(q.get("parts") or []):
            fam.append({"fid": f"{q['id']}p{j + 1}", "canon": p["q"], "kind": "part", "expect": p.get("expect") or [],
                        "expect_src": "gold" if p.get("expect") else "", "source": "multipart-part"})
    for c in load(ROOT / "eval/conversations.jsonl"):
        for j, t in enumerate(c["turns"]):
            fam.append({"fid": f"{c['id']}t{j + 1}", "canon": t, "kind": "conversation",
                        "expect": (c.get("expect") or [[]] * 2)[j] if j < len(c.get("expect") or []) else [],
                        "expect_src": "gold", "source": "conversations",
                        "followup_of": f"{c['id']}t{j}" if j else None})
    return fam


def director_families(n: int, seed: int = 7):
    rows = [json.loads(x) for i in range(1, 9) for x in (DQ / f"part_{i}.jsonl").read_text().splitlines() if x.strip()]
    by_id = {r["id"]: r for r in rows}
    rng = random.Random(seed)
    pool = [r for r in rows if r["kind"] not in SKIP_KINDS and r["style"] in STYLES and len(r["text"].split()) <= 16]
    by_kind: dict = {}
    for r in pool:
        by_kind.setdefault(r["kind"], []).append(r)
    kinds = sorted(by_kind)
    out, seen = [], set()
    while len(out) < n:
        for k in kinds:
            if by_kind[k] and len(out) < n:
                r = by_kind[k].pop(rng.randrange(len(by_kind[k])))
                if r["text"].lower() not in seen:
                    seen.add(r["text"].lower())
                    out.append(r)
    fam = [{"fid": "d" + r["id"], "canon": r["text"], "kind": r["kind"], "expect": [], "expect_src": "",
            "source": "director"} for r in out]
    # follow-up pairs (parent plain question + one follow-up)
    fus = [r for r in rows if r["style"] == "followup" and r.get("followup_of") in by_id
           and by_id[r["followup_of"]]["style"] != "followup" and r["kind"] not in SKIP_KINDS]
    rng.shuffle(fus)
    for r in fus[: max(10, n // 10)]:
        p = by_id[r["followup_of"]]
        if p["text"].lower() in seen:
            continue
        seen.add(p["text"].lower())
        fam.append({"fid": "d" + p["id"], "canon": p["text"], "kind": p["kind"], "expect": [], "expect_src": "",
                    "source": "director"})
        fam.append({"fid": "d" + r["id"], "canon": r["text"], "kind": r["kind"], "expect": [], "expect_src": "",
                    "source": "director", "followup_of": "d" + p["id"]})
    return fam


PROMPT = """You are building evaluation data for a spoken search engine. For each numbered question below, return a JSON
object with these fields:
- "i": the number
- "paraphrases": 3 different spoken rewordings that have exactly the SAME answer (vary the wording a lot: synonyms,
  word order, possessives, filler like "um" or "hey" sometimes; keep the same entities and constraints)
- "near_miss": 1 question about the SAME main entity whose answer is DIFFERENT (e.g. "who founded nvidia" ->
  "when was nvidia founded"); for a question with no entity, a different question on the same topic
- "resolved": if the question is a follow-up (it is marked FOLLOW-UP with its previous question), the follow-up
  rewritten to be self-contained; else ""
- "expect": if the question is a stable fact with a short answer, 1-3 lowercase keywords any correct answer must
  contain (e.g. ["canberra"], ["1993"]); [] for open-ended, advice, opinion, or live questions
- "live": true if the answer changes within days (news, scores, prices, weather, schedules, opening hours), else false
Reply with ONLY a JSON array, no prose.

Questions:
"""


async def label(batch, sem):
    lines = []
    for j, f in enumerate(batch):
        s = f"{j + 1}. {f['canon']}"
        if f.get("followup_of_text"):
            s += f"   [FOLLOW-UP; previous question: {f['followup_of_text']}]"
        lines.append(s)
    async with sem:
        for attempt in range(3):
            try:
                r = await answer.client().messages.create(
                    model=config.ANSWER_MODEL, max_tokens=8000,
                    messages=[{"role": "user", "content": PROMPT + "\n".join(lines)}])
                txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                m = re.search(r"\[.*\]", txt, re.S)
                arr = json.loads(m.group(0))
                usage = (r.usage.input_tokens, r.usage.output_tokens)
                return arr, usage
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, type(e).__name__, str(e)[:120], flush=True)
                await asyncio.sleep(2)
    return [], (0, 0)


async def main_async(a):
    fam = eval_families() + director_families(a.n_director)
    by = {f["fid"]: f for f in fam}
    for f in fam:
        if f.get("followup_of"):
            f["followup_of_text"] = by[f["followup_of"]]["canon"]
    batches = [fam[i:i + 25] for i in range(0, len(fam), 25)]
    sem = asyncio.Semaphore(4)
    res = await asyncio.gather(*[label(b, sem) for b in batches])
    tin = tout = 0
    for b, (arr, (i, o)) in zip(batches, res):
        tin, tout = tin + i, tout + o
        got = {int(x.get("i", 0)): x for x in arr if isinstance(x, dict)}
        for j, f in enumerate(b):
            x = got.get(j + 1) or {}
            f["paraphrases"] = [p for p in (x.get("paraphrases") or []) if isinstance(p, str)][:3]
            f["near_miss"] = x.get("near_miss") or ""
            f["resolved"] = x.get("resolved") or ""
            f["live"] = bool(x.get("live"))
            if not f["expect"] and x.get("expect"):
                f["expect"] = [str(k).lower() for k in x["expect"]][:3]
                f["expect_src"] = "llm"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(f) for f in fam) + "\n")
    print(f"{len(fam)} families -> {out}; labelled {sum(1 for f in fam if f.get('paraphrases'))}; "
          f"tokens in {tin} out {tout}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-director", type=int, default=380)
    ap.add_argument("--out", default=str(ROOT / "runs/index/families.jsonl"))
    asyncio.run(main_async(ap.parse_args()))
