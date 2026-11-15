"""Answer-prompt A/B (server/answer.py SYSTEM): the prompt at --base vs the working tree, on the SAME documents.

Questions: the web-lane questions of the 58 set (eval/questions.jsonl, scored by run_ab.score) as the live A/B heard
them (question_heard + its query from --run), and the 30 showcase questions (eval/showcase/questions.jsonl, unscored:
how often the answer opens with a preamble / says the sources don't cover it, and its length). Each question's docs
come from one Wikipedia search on the recorded query (lanes.wikipedia, cached for both arms); each arm is one
answer-model call. No Tavily, no audio, no embedding model.

  .venv/bin/python eval/prompt_ab.py --base 65d59ed --out runs/prompt_ab.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_ab import score  # noqa: E402

from server import answer, config, lanes  # noqa: E402

PREAMBLE = re.compile(r"^\s*(the|your|this) (spoken )?question (seems|appears)|^\s*(it )?(seems|sounds) like you|"
                      r"^\s*you('re| are) asking", re.I)
UNCOVERED = re.compile(r"\b(sources?|results?)\b[^.]{0,30}\b(don't|do not|doesn't|does not|didn't)\b[^.]{0,12}\b(cover|say|mention|include|give)", re.I)


def old_system(base: str) -> str:
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:server/answer.py"], capture_output=True, text=True,
                         check=True).stdout
    m = re.search(r'^SYSTEM = """(.*?)"""', src, re.S | re.M)
    return m.group(1)


def first_sentence(t: str) -> str:
    return re.split(r"(?<=[.!?])\s", t.strip(), maxsplit=1)[0]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="65d59ed")
    ap.add_argument("--run", default="runs/ab_v2b_q58_fused_r1.json", help="58-set A/B: heard text + query per question")
    ap.add_argument("--showcase-run", default="runs/ab_v2b_showcase_fused_r1.json")
    ap.add_argument("--max-llm", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "runs" / "prompt_ab.json"))
    a = ap.parse_args()
    systems = {"before": old_system(a.base), "after": answer.SYSTEM}
    assert systems["before"] != systems["after"], "the prompt did not change"

    qs = {}
    for f in ("eval/questions.jsonl", "eval/showcase/questions.jsonl"):
        for line in open(ROOT / f):
            o = json.loads(line)
            qs[o["id"]] = o
    items = []
    for path, set_name in ((a.run, "q58-web"), (a.showcase_run, "showcase")):
        rows = json.load(open(ROOT / path))["rows"]
        for r in rows:
            if r.get("mode") != "headstart" or r["id"] not in qs or not r.get("query"):
                continue
            if set_name == "q58-web" and not qs[r["id"]].get("expect"):
                continue
            items.append({"set": set_name, "id": r["id"], "heard": r["question_heard"], "query": r["query"],
                          "expect": qs[r["id"]].get("expect") or []})
    calls = {"llm": 0}
    sem = asyncio.Semaphore(3)
    wsem = asyncio.Semaphore(2)

    async def ask(system: str, question: str, docs) -> str:
        calls["llm"] += 1
        if calls["llm"] > a.max_llm:
            raise SystemExit("llm budget exceeded")
        async with sem:
            for attempt in range(3):
                try:
                    r = await answer.client().messages.create(
                        model=config.ANSWER_MODEL, max_tokens=160, system=system,
                        **answer.model_opts(config.ANSWER_MODEL),
                        messages=[{"role": "user", "content": answer.build_prompt(question, docs)}])
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
                except Exception as e:  # noqa: BLE001  (529 / transient)
                    if attempt == 2:
                        return f"<error {type(e).__name__}>"
                    await asyncio.sleep(1.5 * (attempt + 1))

    async def one(it):
        async with wsem:
            for attempt in range(4):
                try:
                    docs = await lanes.wikipedia(it["query"], 4)
                    break
                except Exception:  # noqa: BLE001  (Wikipedia 429)
                    if attempt == 3:
                        docs = []
                    await asyncio.sleep(2 ** (attempt + 1))
        docs = docs[:5]
        out = {**it, "docs": [d.title for d in docs]}
        for arm, system in systems.items():
            t = await ask(system, it["heard"], docs)
            out[arm] = {"answer": t, "correct": score(t, it["expect"]) if it["expect"] else None,
                        "preamble": bool(PREAMBLE.search(t)), "uncovered_first": bool(UNCOVERED.search(first_sentence(t))),
                        "words": len(t.split())}
        return out

    res = await asyncio.gather(*(one(it) for it in items))
    summ = {}
    for set_name in ("q58-web", "showcase"):
        rs = [r for r in res if r["set"] == set_name]
        s = {"n": len(rs)}
        for arm in systems:
            xs = [r[arm] for r in rs]
            sc = [x["correct"] for x in xs if x["correct"] is not None]
            s[arm] = {"correct": sum(sc), "scored": len(sc), "preamble": sum(x["preamble"] for x in xs),
                      "uncovered_first": sum(x["uncovered_first"] for x in xs),
                      "words_mean": round(statistics.mean(x["words"] for x in xs), 1) if xs else None,
                      "errors": sum(x["answer"].startswith("<error") for x in xs)}
        summ[set_name] = s
    out = {"base": a.base, "model": config.ANSWER_MODEL, "llm_calls": calls["llm"], "summary": summ, "rows": res}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
