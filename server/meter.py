"""Cost meter: per-question answer-model tokens and an estimated $ (web credits x price + tokens x price).

Fed by the speculator: every answer.Stream it created this question (drafts, soft and final answers). The stream that
answered is "used"; every other stream's tokens (drafts thrown away, a superseded soft answer, a hedge loser) are
waste and are counted, so a Head Start that drafts more shows it. Prices: config.METER_* (with sources)."""
from __future__ import annotations

from . import config


def prices() -> dict:
    """USD per million tokens for the configured answer model, and whether that price is verified (listed)."""
    listed = config.METER_MODEL_PRICES.get(config.ANSWER_MODEL)
    pin, pout = listed if listed else (None, None)
    over = config.METER_ANSWER_IN_USD_MTOK is not None and config.METER_ANSWER_OUT_USD_MTOK is not None
    if over:
        pin, pout = float(config.METER_ANSWER_IN_USD_MTOK), float(config.METER_ANSWER_OUT_USD_MTOK)
    return {"model": config.ANSWER_MODEL, "in_usd_mtok": pin, "out_usd_mtok": pout,
            "verified": bool(listed) and not over, "configured": over,
            "credit_usd": config.METER_TAVILY_CREDIT_USD, "credits_per_call": config.METER_CREDITS_PER_CALL,
            "web_backend": config.WEB_BACKEND}


def _u(stream) -> dict:
    s = getattr(stream, "primary", stream)                 # answer.Race wraps the model stream
    fn = getattr(s, "usage", None)
    if fn is None:
        return {"in": 0, "out": 0, "win_in": 0, "win_out": 0, "requests": 0, "est": False}
    return fn()


def tokens(streams, used) -> dict:
    """Token split over this question's streams. used = the stream that answered (or None)."""
    used_s = getattr(used, "primary", used) if used is not None else None
    t = {"ans_in": 0, "ans_out": 0, "waste_in": 0, "waste_out": 0, "llm_requests": 0, "waste_streams": 0,
         "tokens_est": False}
    seen = set()
    for st in streams:
        st = getattr(st, "primary", st)
        if id(st) in seen:
            continue
        seen.add(id(st))
        u = _u(st)
        t["llm_requests"] += u["requests"]
        t["tokens_est"] = t["tokens_est"] or u["est"]
        if st is used_s and getattr(st, "by", None) != "card":
            t["ans_in"] += u["win_in"]
            t["ans_out"] += u["win_out"]
            t["waste_in"] += u["in"] - u["win_in"]        # the hedge loser's request
            t["waste_out"] += u["out"] - u["win_out"]
        else:
            t["waste_in"] += u["in"]
            t["waste_out"] += u["out"]
            if u["in"] or u["out"]:
                t["waste_streams"] += 1
    return t


def cost(paid_calls: float, tok: dict, credits: float | None = None, p: dict | None = None) -> dict:
    """Estimated $ for one question. Web: the actual Tavily credits when the backend reported them (credits > 0), else
    paid calls x credits per call. Tokens: (answer + waste) x the model's price; None if the price is unknown."""
    p = p or prices()
    cr = credits if credits else paid_calls * p["credits_per_call"]
    web = cr * p["credit_usd"]
    n_in = tok.get("ans_in", 0) + tok.get("waste_in", 0)
    n_out = tok.get("ans_out", 0) + tok.get("waste_out", 0)
    if p["in_usd_mtok"] is None:
        llm = waste = None
    else:
        llm = (n_in * p["in_usd_mtok"] + n_out * p["out_usd_mtok"]) / 1e6
        waste = (tok.get("waste_in", 0) * p["in_usd_mtok"] + tok.get("waste_out", 0) * p["out_usd_mtok"]) / 1e6
    return {"usd_web": round(web, 6), "usd_llm": None if llm is None else round(llm, 7),
            "usd_llm_waste": None if waste is None else round(waste, 7),
            "usd": round(web + (llm or 0.0), 6), "usd_est": True, "price_verified": p["verified"]}


def fields(streams, used, m: dict) -> dict:
    """The meter's fields for the per-question metric (m: the metric so far, for paid_calls / credits / fires)."""
    tok = tokens(streams, used)
    paid = m.get("paid_calls", m.get("web_paid", m.get("web_calls", 0))) or 0
    c = cost(paid, tok, m.get("credits") or None)
    wasted = max(0, (m.get("spec_fires") or 0) - (m.get("spec_used") or 0))
    return {**tok, **c, "wasted_fires": wasted}
