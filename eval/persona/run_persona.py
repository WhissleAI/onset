"""Persona A/B on scripted multi-turn sessions (eval/persona/sessions.jsonl): each session is fed as one live stream
(turn, gap, turn, ...) through the configured ear + an Onset Session, once with PERSONA off and once on (alternating
order per session, same process, so both arms see the same load). Every session is a new user (empty profile).

  WEB_BACKEND=wikipedia .venv/bin/python eval/persona/run_persona.py --tag r1            # both arms
  ... --ids p01,p03 --arms on          # a subset / one arm
  ... --judge                          # + a cheap LLM judge (claude-haiku-5-5), tokens and $ reported

Scores per turn: gold keys (expect), the meaning picked for an ambiguous name (sense regexes), the style asked for
(short <= 20 words, fuller >= 28, more >= 45, fix = apologises, units:imperial = feet / miles / °F). Writes
runs/persona_<tag>.json."""
import argparse, asyncio, json, re, statistics, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--sessions", default=str(HERE / "sessions.jsonl"))
ap.add_argument("--ids", default="")
ap.add_argument("--arms", default="off,on")
ap.add_argument("--gap", type=float, default=3.0, help="seconds of silence between turns")
ap.add_argument("--tag", default="")
ap.add_argument("--judge", action="store_true")
ap.add_argument("--events", action="store_true", help="also keep spec / sources / route events (debug)")
ap.add_argument("--judge-only", default="", help="re-judge an existing runs/persona_<tag>.json")
args = ap.parse_args()
from server import ear, embed, lanes, answer, searchcache, router, persona, meter, config   # noqa
from server.session import Session            # noqa

JUDGE_MODEL = "claude-haiku-5-5"


def words(a):
    return len(re.sub(r"\[\d+\]", "", a or "").split())


def ok(ans, keys):
    a = (ans or "").lower()
    return all(any(k2 in a for k2 in k.split("|")) for k in keys)


def style_ok(ans, st):
    if not st:
        return None
    a, n = (ans or "").lower(), words(ans)
    if st == "short":
        return n <= 20
    if st == "fuller":
        return n >= 28
    if st == "more":
        return n >= 45
    if st == "fix":
        return bool(re.search(r"\b(sorry|apologi[sz]e|my mistake|my apologies)\b", a))
    if st == "units:imperial":
        return bool(re.search(r"\b(feet|ft|foot|miles|mph|°f|fahrenheit)\b", a))
    return None


def sense_ok(ans, s, sources=()):
    """The intended meaning: the answer names it (want) and not the other one (not), or the top source is that
    meaning's article (src) and the answer doesn't talk about the other one."""
    if not s:
        return None
    a = (ans or "").lower()
    if re.search(s["not"], a):
        return False
    top = (sources[0] if sources else "").lower()
    return bool(re.search(s["want"], a)) or bool(s.get("src") and re.search(s["src"], top))


async def run(sess):
    ev = []
    t0 = time.perf_counter()

    def sink(e):
        if args.events and e.get("t") in ("spec", "sources", "route", "plan"):
            ev.append({k: (v if k != "docs" else [(d.get("title"), d.get("score"), d.get("source")) for d in v])
                       for k, v in e.items() if k in ("t", "q", "lane", "query", "state", "docs", "reason", "n", "ms",
                                                       "route", "path")})
        if e.get("t") in ("persona", "metric", "endpoint"):
            ev.append({k: v for k, v in e.items() if k in ("t", "q", "style", "why", "sense", "title", "instead",
                                                             "profile", "ms")})

    s = Session(sink, mode="headstart", t0=t0)
    e = ear.make_ear([s])
    parts = []
    for f in sess["files"]:
        w, _ = sf.read(HERE / "audio" / f, dtype="float32")
        parts += [w, np.zeros(int(16000 * args.gap), np.float32)]
    a = np.concatenate(parts)
    for i in range(0, len(a), 1600):
        while time.perf_counter() < t0 + (i + 1600) / 16000:
            await asyncio.sleep(0.003)
        await e.feed(a[i:i + 1600])
    await e.flush()
    if s.final_task:
        try:
            await s.final_task
        except Exception:
            pass
    t_end = time.perf_counter() + 8
    while (s.busy or s.pending_finals) and time.perf_counter() < t_end:
        await asyncio.sleep(0.05)
    if getattr(s, "bg_tasks", None):
        await asyncio.wait(list(s.bg_tasks), timeout=8)
    s.close()
    return s.history, ev, s.persona.summary()


JUDGE_SYS = """You grade a voice assistant's answer for ONE user. You see the user's earlier questions in this session, how
the current question was spoken, the question, and the answer. Score 1-5 how well the answer fits THIS user: the
meaning they most plausibly intend given their earlier questions, an answer length / depth that suits how they asked
(quick terse askers want one line; hesitant or beginner askers want a short plain explanation; someone who asks for
more wants detail), the tone (if they said the last answer was wrong, a brief acknowledgement), and their units.
Ignore citation markers like [1]. Reply with only JSON: {"score": <1-5>, "why": "<10 words>"}"""


async def judge(rows):
    client = answer.client()
    tin = tout = 0
    for r in rows:
        hist = []
        for t in r["turns"]:
            how = []
            if t.get("rate", 185) >= 240:
                how.append("spoken quickly, clipped")
            if t.get("rate", 185) <= 160:
                how.append("spoken slowly with ums and pauses")
            msg = (f"Earlier questions: {json.dumps(hist) if hist else '(none)'}\nHow it was spoken: "
                   f"{', '.join(how) or 'normal'}\nQuestion: {t['said']}\nAnswer: {t['answer']}")
            hist.append(t["said"])
            if not t.get("answer"):
                t["judge"] = None
                continue
            try:
                resp = await client.messages.create(model=JUDGE_MODEL, max_tokens=60, system=JUDGE_SYS,
                                                    messages=[{"role": "user", "content": msg}],
                                                    **answer.model_opts(JUDGE_MODEL))
                txt = "".join(getattr(b, "text", "") for b in resp.content)
                tin += resp.usage.input_tokens
                tout += resp.usage.output_tokens
                m = re.search(r"\{.*\}", txt, re.S)
                t["judge"] = json.loads(m.group(0)) if m else None
            except Exception as e:
                t["judge"] = {"error": f"{type(e).__name__}: {e}"[:120]}
    p = config.METER_MODEL_PRICES.get(JUDGE_MODEL)
    usd = (tin * p[0] + tout * p[1]) / 1e6 if p else None
    return {"model": JUDGE_MODEL, "in": tin, "out": tout, "usd": None if usd is None else round(usd, 4),
            "calls": sum(1 for r in rows for t in r["turns"] if t.get("answer"))}


def summarize(res, jmeta=None):
    print("\narm   content  sense   style   judge  p50ms  p90ms  hint_tok/q  ans_in/q  ans_out/q  words/turn")
    out = {}
    for arm in args.arms.split(","):
        T = [t for r in res if r["arm"] == arm for t in r["turns"]]
        if not T:
            continue
        c = [t["correct"] for t in T if t["correct"] is not None]
        se = [t["sense_ok"] for t in T if t["sense_ok"] is not None]
        st = [t["style_ok"] for t in T if t["style_ok"] is not None]
        jd = [t["judge"]["score"] for t in T if isinstance(t.get("judge"), dict) and "score" in t["judge"]]
        lat = sorted(t["ready_ms"] for t in T if t.get("ready_ms") is not None)
        ht = [t.get("hint_tok") or 0 for t in T]
        ai = [t.get("ans_in") or 0 for t in T]
        ao = [t.get("ans_out") or 0 for t in T]
        o = {"content": f"{sum(c)}/{len(c)}", "sense": f"{sum(se)}/{len(se)}", "style": f"{sum(st)}/{len(st)}",
             "judge": round(statistics.mean(jd), 2) if jd else None,
             "p50": statistics.median(lat) if lat else None, "p90": lat[int(0.9 * (len(lat) - 1))] if lat else None,
             "hint_tok": round(statistics.mean(ht), 1), "ans_in": round(statistics.mean(ai)), "ans_out": round(statistics.mean(ao), 1),
             "words": round(statistics.mean(words(t["answer"]) for t in T), 1), "turns": len(T),
             "answered": sum(1 for t in T if t.get("answer"))}
        out[arm] = o
        print(f"{arm:5s} {o['content']:8s} {o['sense']:7s} {o['style']:7s} {o['judge']!s:6s} {o['p50']!s:6s} "
              f"{o['p90']!s:6s} {o['hint_tok']:<11} {o['ans_in']:<9} {o['ans_out']:<10} {o['words']}")
    if jmeta:
        print("judge:", jmeta)
    return out


async def main():
    out = ROOT / "runs" / (f"persona_{args.tag}.json" if args.tag else "persona.json")
    if args.judge_only:
        src = ROOT / "runs" / f"persona_{args.judge_only}.json"
        d = json.loads(src.read_text())
        jm = await judge(d["rows"])
        d["judge"] = jm
        d["summary"] = summarize(d["rows"], jm)
        src.write_text(json.dumps(d, indent=1))
        return
    ear.warmup(); embed.warmup(); await lanes.warm_web(); await answer.warm()
    persona.ENABLED = True
    persona.warmup()
    sessions = [json.loads(x) for x in Path(args.sessions).read_text().splitlines() if x.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        sessions = [x for x in sessions if x["id"] in want]
    arms = args.arms.split(",")
    res = []
    for k, sess in enumerate(sessions):
        order = arms if k % 2 == 0 else arms[::-1]
        for arm in order:
            persona.ENABLED = arm == "on"
            searchcache.reset()
            router.reset_caches()
            h, ev, prof = await run(sess)
            h = [m for m in h if not m.get("skipped")]
            row = {"id": sess["id"], "arm": arm, "n_answers": len(h), "profile": prof, "events": ev, "turns": []}
            for i, m in enumerate(h[:len(sess["turns"])]):
                g = sess["turns"][i]
                ans = m.get("answer") or ""
                pm = m.get("persona") or {}
                row["turns"].append({
                    "i": i + 1, "said": g["text"], "rate": g["rate"], "heard": m.get("question"), "query": m.get("query"),
                    "answer": ans, "sources": (m.get("sources") or [])[:3], "ready_ms": m.get("ready_ms"),
                    "path": m.get("path"), "answer_by": m.get("answer_by"),
                    "correct": ok(ans, g["expect"]) if g["expect"] else None,
                    "sense_ok": sense_ok(ans, g.get("sense"), m.get("sources") or []), "style_ok": style_ok(ans, g.get("style")),
                    "want_style": g.get("style"), "persona": pm, "hint_tok": pm.get("hint_tok", 0),
                    "ans_in": m.get("ans_in"), "ans_out": m.get("ans_out")})
            res.append(row)
            print(f'{sess["id"]} {arm:3s} ' + " | ".join(
                f'{t["i"]}:{t["heard"]!r} {t["ready_ms"]}ms c={t["correct"]} s={t["sense_ok"]} st={t["style_ok"]} '
                f'[{(t["persona"] or {}).get("why") or ""}{"; " + t["persona"]["sense"] if (t["persona"] or {}).get("sense") else ""}] '
                f'{t["answer"][:70]!r}' for t in row["turns"]), flush=True)
            out.write_text(json.dumps({"rows": res}, indent=1))
    jm = await judge(res) if args.judge else None
    summ = summarize(res, jm)
    out.write_text(json.dumps({"rows": res, "summary": summ, "judge": jm,
                               "config": {"bias": persona.BIAS, "sense": persona.SENSE, "style": persona.STYLE,
                                          "web_backend": config.WEB_BACKEND, "ear": config.EAR}}, indent=1))
    print("wrote", out)


asyncio.run(main())
