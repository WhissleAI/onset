"""Promotion gates, one per model. A candidate replaces the current model only if it BEATS it on the model's own
metric (by more than the noise margin), loses no accuracy on any fixed held-out set, and adds no p50 latency.
Pure functions over the metric dicts the cycle measures for BOTH models on the SAME fixed sets: easy to test.

Every gate returns {"promote": bool, "beats": [...], "fails": [...]} (the reasons, in plain words, for the log).
"""
from __future__ import annotations


def _get(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _verdict(beats: list, fails: list, need_beat: bool = True) -> dict:
    return {"promote": (bool(beats) or not need_beat) and not fails, "beats": beats, "fails": fails}


def no_regression(cur: dict, cand: dict, sets: list[str], acc_key: str = "acc", p50_key: str = "p50_ms",
                  acc_tol: float = 0.0, p50_tol_ms: float = 0.0) -> list[str]:
    """Shared rule: on each fixed set, accuracy may not drop (beyond acc_tol) and p50 may not rise (beyond tol)."""
    fails = []
    for s in sets:
        a0, a1 = _get(cur, f"{s}.{acc_key}"), _get(cand, f"{s}.{acc_key}")
        if a0 is not None and (a1 is None or a1 < a0 - acc_tol - 1e-9):
            fails.append(f"{s}: accuracy {a0} -> {a1}")
        p0, p1 = _get(cur, f"{s}.{p50_key}"), _get(cand, f"{s}.{p50_key}")
        if p0 is not None and (p1 is None or p1 > p0 + p50_tol_ms + 1e-9):
            fails.append(f"{s}: p50 {p0} -> {p1} ms")
    return fails


# ------------------------------------------------------------------------------------------------ calibrator
CALIB_MIN_BRIER_GAIN = 0.002


def calib(cur: dict, cand: dict) -> dict:
    """Held-out test answers (fixed families) + the live replay rows: Brier must improve by >= 0.002 on the test
    split; ECE may not rise by > 0.01 and AUC may not fall by > 0.01 on either set. The calibrator only labels the
    answer (CALIB_ROUTE is off), so accuracy and p50 are unchanged by construction; if CALIB_ROUTE is on, the
    end-to-end sets are checked too."""
    beats, fails = [], []
    b0, b1 = _get(cur, "test.brier"), _get(cand, "test.brier")
    if b0 is None or b1 is None:
        return _verdict([], ["no test Brier"])
    if b1 <= b0 - CALIB_MIN_BRIER_GAIN:
        beats.append(f"test Brier {b0} -> {b1}")
    for s in ("test", "live"):
        e0, e1 = _get(cur, f"{s}.ece"), _get(cand, f"{s}.ece")
        if e0 is not None and e1 is not None and e1 > e0 + 0.01:
            fails.append(f"{s}: ECE {e0} -> {e1}")
        u0, u1 = _get(cur, f"{s}.auc"), _get(cand, f"{s}.auc")
        if u0 is not None and u1 is not None and u1 < u0 - 0.01:
            fails.append(f"{s}: AUC {u0} -> {u1}")
        l0, l1 = _get(cur, f"{s}.brier"), _get(cand, f"{s}.brier")
        if s == "live" and l0 is not None and l1 is not None and l1 > l0 + 0.005:
            fails.append(f"live Brier {l0} -> {l1}")
    if cur.get("e2e") or cand.get("e2e"):
        fails += no_regression(cur.get("e2e") or {}, cand.get("e2e") or {}, ["q58", "showcase"])
    return _verdict(beats, fails)


# ------------------------------------------------------------------------------------------------ Keystone director
KDIR_MIN_PAID_GAIN = 0.003


def kdir(cur: dict, cand: dict, streams=("test", "old_test")) -> dict:
    """Replays of the fixed held-out kdir test streams (same store / feedback settings for both): paid per question
    must fall by >= 0.003 averaged over the streams; on every stream regret may not rise, accuracy may not fall and
    the replayed p50 may not rise (+1 ms: decision time jitter)."""
    beats, fails = [], []
    d = []
    for s in streams:
        p0, p1 = _get(cur, f"{s}.paid_per_q"), _get(cand, f"{s}.paid_per_q")
        if p0 is None or p1 is None:
            fails.append(f"{s}: not measured")
            continue
        d.append(p0 - p1)
        r0, r1 = _get(cur, f"{s}.regret_rate"), _get(cand, f"{s}.regret_rate")
        if r1 is not None and r0 is not None and r1 > r0 + 1e-9:
            fails.append(f"{s}: regret {r0} -> {r1}")
    if d and sum(d) / len(d) >= KDIR_MIN_PAID_GAIN:
        beats.append(f"paid/q -{sum(d) / len(d):.4f} (mean over {len(d)} streams)")
    fails += no_regression(cur, cand, list(streams), acc_key="acc_all", p50_key="p50_ms", p50_tol_ms=1.0)
    return _verdict(beats, fails)


# ------------------------------------------------------------------------------------------------ TTLs
def ttl(cur: dict, cand: dict) -> dict:
    """Learned TTLs: held-out Wikidata statements (every 5th item, never fitted) -> log-likelihood of the observed
    changes under each table must improve; and the kdir test-stream replay with KDIR_TTL on may not pay more, regret
    more or answer worse."""
    beats, fails = [], []
    l0, l1 = _get(cur, "heldout.loglik_per_obs"), _get(cand, "heldout.loglik_per_obs")
    if l0 is None or l1 is None:
        fails.append("no held-out likelihood")
    elif l1 > l0 + 1e-4:
        beats.append(f"held-out log-lik/obs {l0} -> {l1}")
    for s in ("test", "old_test"):
        p0, p1 = _get(cur, f"{s}.paid_per_q"), _get(cand, f"{s}.paid_per_q")
        if p0 is not None and p1 is not None and p1 > p0 + 1e-9:
            fails.append(f"{s}: paid/q {p0} -> {p1}")
        r0, r1 = _get(cur, f"{s}.regret_rate"), _get(cand, f"{s}.regret_rate")
        if r0 is not None and r1 is not None and r1 > r0 + 1e-9:
            fails.append(f"{s}: regret {r0} -> {r1}")
    fails += no_regression(cur, cand, ["test", "old_test"], acc_key="acc_all", p50_key="p50_ms", p50_tol_ms=1.0)
    return _verdict(beats, fails)


# ------------------------------------------------------------------------------------------------ Mamba director
DIR_MIN_BCE_GAIN = 0.002        # ~ the seed noise measured in sweep5 is .005 on single runs; the gate needs both sets


def director(cur: dict, cand: dict) -> dict:
    """Offline, on the fixed held-out splits (test 1,000 / test_v2 760 / heldout80, val-chosen thresholds):
    threshold-free end-of-turn BCE must fall by >= 0.002 on test_v2 AND not rise on test; the endpoint p50 at the
    val thresholds and at the matched 10 / 20 premature targets may not rise on any split; premature per 100 q may
    not rise by > 1; addressed AUC (open mic) may not fall by > 0.01. Then (cycle.py) an end-to-end A/B with
    FLUX_SOFT=director on the 58 set: no accuracy loss, no p50 regression (no_regression)."""
    beats, fails = [], []
    b0, b1 = _get(cur, "test_v2.eot_bce"), _get(cand, "test_v2.eot_bce")
    if b0 is None or b1 is None:
        return _verdict([], ["no test_v2 eot BCE"])
    if b1 <= b0 - DIR_MIN_BCE_GAIN:
        beats.append(f"test_v2 eot BCE {b0} -> {b1}")
    t0, t1 = _get(cur, "test.eot_bce"), _get(cand, "test.eot_bce")
    if t0 is not None and t1 is not None and t1 > t0 + 1e-4:
        fails.append(f"test eot BCE {t0} -> {t1}")
    for s in ("test", "test_v2", "heldout80"):
        for k in ("p50", "p50_at10", "p50_at20"):
            p0, p1 = _get(cur, f"{s}.{k}"), _get(cand, f"{s}.{k}")
            if p0 is not None and p1 is not None and p1 > p0 + 1e-9:
                fails.append(f"{s}: {k} {p0} -> {p1} ms")
        q0, q1 = _get(cur, f"{s}.premature"), _get(cand, f"{s}.premature")
        if q0 is not None and q1 is not None and q1 > q0 + 1.0:
            fails.append(f"{s}: premature {q0} -> {q1} /100 q")
    a0, a1 = _get(cur, "addr.auc"), _get(cand, "addr.auc")
    if a0 is not None and a1 is not None and a1 < a0 - 0.01:
        fails.append(f"addressed AUC {a0} -> {a1}")
    if cur.get("e2e") or cand.get("e2e"):
        fails += no_regression(cur.get("e2e") or {}, cand.get("e2e") or {}, ["q58"])
    return _verdict(beats, fails)


GATES = {"calib": calib, "kdir": kdir, "ttl": ttl, "director": director}


def regressions_ok(reg: dict) -> list[str]:
    """The fixed regression suites run with every candidate installed (persona, fixlive, arcvalue text-level)."""
    fails = []
    for name, r in (reg or {}).items():
        if not isinstance(r, dict):
            continue
        if r.get("ok") is False:
            fails.append(f"regression {name}: {r.get('why') or 'failed'}")
    return fails
