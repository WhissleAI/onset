"""One training cycle: collect -> synthesize -> refit / retrain each model -> measure current vs candidate on the
fixed held-out sets -> gate -> promote (with regressions + rollback) -> results. scripts/train_cycle.sh loops it.

Each heavy step is its own niced process started only when the user isn't testing (quiet flag) and swap is below
the cap (common.run_heavy). A failed or skipped step never stops the cycle; nothing is promoted from it.

  .venv/bin/python eval/train/cycle.py [--cycle N] [--dry-run] [--only calib,kdir,ttl,director]
  --dry-run: small synthetic batch, no Flux streaming, gates computed, NOTHING installed / written to docs.

Promotion of a candidate that passed its gate:
  1. e2e (TRAIN_E2E=1, behaviour-changing models only: kdir with KDIR=1, director with FLUX_SOFT=director): the 58
     set replayed with the current model, then (after install) with the candidate -> gates.no_regression
  2. install (registry.promote), then the regression suites (pytest: fixlive, persona, calibrate, kdir, director_v2)
  3. any failure -> registry.rollback (the previous version is re-installed)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gates as G  # noqa: E402
import registry as RG  # noqa: E402
import results as RS  # noqa: E402

STEP_TIMEOUT = {"calib": 3600, "kdir": 5400, "ttl": 1800, "director": 7200}
TTL_EVERY = int(os.environ.get("TRAIN_TTL_EVERY", "6"))
REGRESSION_TESTS = ["tests/test_fixlive.py", "tests/test_persona.py", "tests/test_calibrate.py", "tests/test_kdir.py",
                    "tests/test_director_v2.py", "tests/test_train_loop.py"]
E2E_ENV = {"kdir": {"INDEX": "1", "KDIR": "1"}, "director": {"FLUX_SOFT": "director", "DIRECTOR": "on",
                                                              "DIRECTOR_STREAMS": "1"}}
CAND_FILES = {"calib": "calib", "kdir": "kdir", "ttl": "ttl", "director": "director"}


def next_cycle() -> int:
    st = C.read_json(C.TRAIN / "state.json", {})
    return int(st.get("last_cycle", 0)) + 1


def step_cmd(model: str, cycle: int, out: Path, dry: bool) -> list[str]:
    t = str(C.ROOT / "eval/train")
    if model == "calib":
        return [C.PY, f"{t}/calib_step.py", "--cycle", str(cycle), "--out", str(out)] + (["--no-label"] if dry else [])
    if model in ("kdir", "ttl"):
        return [C.PY, f"{t}/kdir_step.py", model, "--cycle", str(cycle), "--out", str(out)]
    if model == "director":
        return [C.PY, f"{t}/director_step.py", "--cycle", str(cycle), "--out", str(out)] + (["--no-stream"] if dry else [])
    raise ValueError(model)


def e2e(model: str, cdir: Path, tag: str) -> dict:
    """Real-audio replay of the 58 set (Wikipedia lane, EAR from .env) with the model's flag on: accuracy + p50."""
    out = cdir / f"e2e_{model}_{tag}.json"
    rc = C.run_heavy([C.PY, str(C.ROOT / "eval/run_ab.py"), "--modes", "headstart", "--no-latest", "--out", str(out),
                      "--wait-lock", "1800", "--owner", "trainer"], str(cdir / "e2e.log"), env=E2E_ENV.get(model, {}),
                     timeout=5400)
    d = C.read_json(out, {})
    rows = [r for r in d.get("rows") or [] if r.get("mode") == "headstart" and r.get("ready_ms") is not None]
    if rc != 0 or not rows:
        return {"error": f"run_ab exit {rc}"}
    sc = [r for r in rows if r.get("correct") is not None]
    return {"q58": {"acc": round(sum(bool(r["correct"]) for r in sc) / max(1, len(sc)), 4),
                    "p50_ms": statistics.median(r["ready_ms"] for r in rows), "n": len(rows)},
            "ready": {r["id"]: r["ready_ms"] for r in rows}}


def regressions(cdir: Path) -> dict:
    tests = [t for t in REGRESSION_TESTS if (C.ROOT / t).exists()]
    rc = C.run_heavy([C.PY, "-m", "pytest", "-q", *tests], str(cdir / "regressions.log"), timeout=2400)
    return {"pytest": {"ok": rc == 0, "why": None if rc == 0 else f"pytest exit {rc} (see regressions.log)",
                       "tests": tests}}


def data_summary(collect: dict, synth: dict, director: dict) -> dict:
    src = (collect or {}).get("sources") or {}
    return {"real_total": (collect or {}).get("total"), "real_new": (collect or {}).get("new"),
            "app_new": (src.get("app") or {}).get("new", 0) + (src.get("app_log") or {}).get("new", 0),
            "eval_new": (src.get("eval") or {}).get("new", 0),
            "synth_sessions": (synth or {}).get("sessions", 0), "synth_turns": (synth or {}).get("turns", 0),
            "synth_by_class": (synth or {}).get("by_class"),
            "paraphrase_usd": ((synth or {}).get("paraphrase") or {}).get("usd", 0.0),
            "flux_usd": ((director or {}).get("stream") or {}).get("flux_usd_est", 0.0)}


def run_cycle(cycle: int, dry: bool = False, only: list[str] | None = None) -> dict:
    cdir = C.TRAIN / "cycles" / f"c{cycle}"
    cdir.mkdir(parents=True, exist_ok=True)
    reg = RG.Registry()
    for m in RG.MODELS:
        reg.bootstrap(m)
    rep: dict = {"cycle": cycle, "started": C.now_iso(), "dry_run": dry, "steps": {}, "gates": {}, "promoted": {}}
    C.log(f"cycle {cycle} start{' (dry run)' if dry else ''}")
    # 1. real data (light: no models)
    C.run_heavy([C.PY, str(C.ROOT / "eval/train/collect.py")] + (["--dry-run"] if dry else []), str(cdir / "collect.log"),
                timeout=900)
    rep["steps"]["collect"] = _last_json(cdir / "collect.log")
    # 2. synthetic sessions (say + Wikidata + Haiku paraphrases, measured $)
    n_s = int(os.environ.get("TRAIN_SYNTH_SESSIONS", "6" if dry else "26"))
    usd = os.environ.get("TRAIN_PARAPHRASE_USD", "0.05" if dry else "2.0")
    C.run_heavy([C.PY, str(C.ROOT / "eval/train/synth.py"), "--cycle", str(cycle), "--sessions", str(n_s),
                 "--paraphrase-usd", usd, "--noise"] + (["--no-audio"] if dry else []), str(cdir / "synth.log"),
                timeout=1800)
    rep["steps"]["synth"] = C.read_json(C.TRAIN / "synth" / f"c{cycle}" / "report.json", {})
    # 3. each model: candidate + metrics on the fixed held-out sets
    models = only or ["calib", "kdir", "ttl", "director"]
    if "ttl" in models and not only and (cycle - 1) % TTL_EVERY != 0:
        models.remove("ttl")
        rep["steps"]["ttl"] = {"skip": f"TTLs refit every {TTL_EVERY} cycles"}
    for m in models:
        if C.stopped():
            break
        out = cdir / m
        out.mkdir(parents=True, exist_ok=True)
        rc = C.run_heavy(step_cmd(m, cycle, out, dry), str(out / "step.log"), timeout=STEP_TIMEOUT[m])
        met = C.read_json(out / "metrics.json", {}) if rc != -1 else {"skip": "no slot"}
        rep["steps"][m] = {k: v for k, v in met.items() if k not in ("pick", "trace")}
        if "current" not in met or "candidate" not in met:
            C.log(f"  {m}: no candidate ({met.get('skip') or met.get('error') or f'exit {rc}'})")
            continue
        g = G.GATES[m](met["current"], met["candidate"])
        rep["gates"][m] = g
        C.log(f"  {m}: {RS.gate_text(m, g)}")
        if not g["promote"] or dry:
            continue
        rep["promoted"].update(promote(m, cycle, out, met, g, reg, cdir))
    # 4. results
    rep["finished"] = C.now_iso()
    rep["data"] = data_summary(rep["steps"].get("collect"), rep["steps"].get("synth"), rep["steps"].get("director"))
    if rep["promoted"] and not dry:
        rep["docs"] = RS.update_docs(cycle, rep, reg)["scorecard_models"]
    C.write_json(cdir / "report.json", rep)
    if not dry:
        st = C.read_json(C.TRAIN / "state.json", {})
        st["last_cycle"] = cycle
        C.write_json(C.TRAIN / "state.json", st)
        RS.write_latest(rep, reg)
    C.log(f"cycle {cycle} done: promoted {rep['promoted'] or 'nothing'}")
    return rep


def promote(m: str, cycle: int, out: Path, met: dict, g: dict, reg: RG.Registry, cdir: Path) -> dict:
    e2e_on = os.environ.get("TRAIN_E2E", "0") == "1" and m in E2E_ENV
    cur_e2e = e2e(m, cdir, "current") if e2e_on else None
    meta = {"cycle": cycle, "trained": C.now_iso()[:10], "gate": g,
            "metrics": {k: met[k] for k in ("current", "candidate")}, "run": met.get("run")}
    v = reg.promote(m, out, meta)
    C.log(f"  {m}: installed {v}")
    fails = []
    if e2e_on:
        cand_e2e = e2e(m, cdir, "candidate")
        if "error" in cur_e2e or "error" in cand_e2e:
            fails.append("e2e replay failed")
        else:
            fails += G.no_regression(cur_e2e, cand_e2e, ["q58"], acc_key="acc", p50_key="p50_ms",
                                     p50_tol_ms=float(os.environ.get("TRAIN_E2E_P50_TOL_MS", "25")))
    reg_res = regressions(cdir)
    fails += G.regressions_ok(reg_res)
    if fails:
        back = reg.rollback(m)
        C.log(f"  {m}: ROLLED BACK to {back}: {'; '.join(fails)}")
        g["promote"], g["fails"] = False, (g.get("fails") or []) + fails
        return {}
    if os.environ.get("TRAIN_COMMIT", "0") == "1":
        import subprocess
        files = RG.MODELS[m]
        subprocess.run(["git", "add", *files], cwd=str(C.ROOT))
        subprocess.run(["git", "commit", "-m", f"train cycle {cycle}: promote {m} {v}\n\n{RS.gate_text(m, g)}\n\n"
                        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *files], cwd=str(C.ROOT))
    return {m: v}


def _last_json(p: Path) -> dict:
    try:
        for ln in reversed(p.read_text().splitlines()):
            if ln.startswith("{"):
                return json.loads(ln)
    except (OSError, ValueError):
        pass
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    cyc = a.cycle or (0 if a.dry_run else next_cycle())     # a dry run is cycle 0: never counted
    rep = run_cycle(cyc, a.dry_run, [x for x in a.only.split(",") if x] or None)
    print(json.dumps({k: rep[k] for k in ("cycle", "gates", "promoted", "data")}, default=str, indent=1))


if __name__ == "__main__":
    main()
