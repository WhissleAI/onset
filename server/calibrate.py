"""Answer calibration: one confidence per answer ("right about X% of the time") and a short plain reason.

Two stages share one feature vector:
  pre   at the endpoint, before the answer model runs: retrieval (source score + margin, EmbeddingGemma rerank cosine
        + margin, how many top sources name the subject), the router (lane, its own conf, Wikidata / card / Keystone
        hit, shared-store hit and age), Arc (confidence of the question's decisive word, self-correction, follow-up,
        hesitation fillers, retrieval-in-the-loop rewrite, intent). Used by the router under CALIB_ROUTE=1.
  post  after the last token: the same + server/verify.py's check of the answer against its sources and the answer
        itself (a "the sources don't say" answer, extractive fallback, citations). Shown on the badge (CALIB_BADGE).

Model: logistic regression over standardized features, then isotonic regression on its output (both fitted by
eval/calib_fit.py on labelled answers; weights in server/calib_model.json). Missing values are imputed with the
training mean, so a feature a lane cannot supply (Wikipedia has no relevance score, a text question has no ASR
confidence) adds nothing either way. Pure and cheap (< 0.1 ms). Reuses server/confidence.py's linear / sigmoid core.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from .confidence import _lin, _sig

# Flags live here (not server/config.py) to keep merges easy.
CALIB = os.environ.get("CALIB", "1") == "1"                     # compute conf + reason for every answer (metric)
CALIB_BADGE = os.environ.get("CALIB_BADGE", "1") == "1"         # show the badge (the UI reads it from the event)
CALIB_ROUTE = os.environ.get("CALIB_ROUTE", "0") == "1"         # the router acts on the pre-answer conf
CALIB_LOW = float(os.environ.get("CALIB_LOW", "0.6"))           # band edges (calib_model.json "bands" wins)
CALIB_HIGH = float(os.environ.get("CALIB_HIGH", "0.8"))
CALIB_PUBDATE = os.environ.get("CALIB_PUBDATE", "0") == "1"   # Tavily include_published_date (source age)
CALIB_FUSE = float(os.environ.get("CALIB_FUSE", "0"))          # weight of Tavily's score in the final ranking
CALIB_ROUTE_WAIT_S = float(os.environ.get("CALIB_ROUTE_WAIT_S", "1.2"))   # cap on the router's extra lookup

MODEL_PATH = Path(__file__).resolve().parent / "calib_model.json"
_model: dict | None = None
_FILLERS = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "like"}
_LIVE = ("weather", "news", "stock", "currency", "traffic", "sports")
_CITE = re.compile(r"\[\d+")


def model() -> dict:
    global _model
    if _model is None:
        try:
            _model = json.loads(MODEL_PATH.read_text())
        except (OSError, ValueError):
            _model = {}
    return _model


def reload(m: dict | None = None):
    global _model
    _model = m
    if m is None:
        model()


# ---------------------------------------------------------------------------------------------- features
def _num(x) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _age_h(as_of: str | None, now: float | None = None) -> float | None:
    if not as_of:
        return None
    try:
        t = time.mktime(time.strptime(str(as_of)[:16], "%Y-%m-%d %H:%M"))
    except ValueError:
        return None
    return max(0.0, ((now or time.time()) - t) / 3600.0)


def _pub_age_d(pub, now: float | None = None) -> float | None:
    """Days since a source's published date (Tavily published_date, e.g. "Tue, 11 Nov 2026 14:02:00 GMT" or ISO)."""
    if not pub:
        return None
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timezone
    try:
        d = parsedate_to_datetime(str(pub))
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(str(pub)[:19])
        except ValueError:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return max(0.0, ((now or time.time()) - d.timestamp()) / 86400.0)


def _mentions(doc, phrase: str) -> bool:
    if not phrase:
        return False
    t = f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}".lower()
    return all(w in t for w in phrase.lower().split() if len(w) > 2)


def key_word(f) -> tuple[float | None, str]:
    """The question's least-sure decisive word: (min of ear / fused confidence over the content words, the word)."""
    c = list(getattr(f, "sconf", ()) or ()) or list(getattr(f, "wconf", ()) or ())
    words = list(getattr(f, "words", ()) or ())
    idx = [i for i in (getattr(f, "content_idx", ()) or ()) if i < len(c)]
    if not c or not idx:
        return None, ""
    j = min(idx, key=lambda i: c[i])
    return float(c[j]), (words[j] if j < len(words) else "")


def features(f, ranked, route: dict | None = None, v: dict | None = None, answer: str | None = None,
             fallback: str | None = None, now: float | None = None) -> dict:
    """Raw features of one answer. ranked = [(doc, rerank cosine)] best first (the answer's sources); route = {lane,
    conf, tier}; v = verify.check(...) (post stage); answer = the answer text (post stage). None = not available."""
    route = route or {}
    docs = [d for d, _ in ranked]
    rr = [float(s) for _, s in ranked if s is not None]
    src = [_num(getattr(d, "score", None)) for d in docs if getattr(d, "source", "") == "tavily"]
    src = [s for s in src if s is not None]
    metas = [getattr(d, "meta", None) or {} for d in docs]
    subject = ""
    if f is not None:
        subject = (f.entity_query() or (f.phrases[-1] if getattr(f, "phrases", ()) else "")) if hasattr(f, "entity_query") else ""
    top3 = docs[:3]
    kc, kw = key_word(f) if f is not None else (None, "")
    words = [w.lower().strip(",.?!") for w in (getattr(f, "words", ()) or ())]
    intent = (getattr(f, "intent", "") or "") if f is not None else ""
    lane = route.get("lane") or ""
    tier = route.get("tier") or ""
    ks = [m for m in metas if m.get("tier") == "index"]
    ages = [a for a in (_age_h(m.get("as_of"), now) for m in ks) if a is not None]
    pub = [a for a in (_pub_age_d(m.get("published_date"), now) for m in metas) if a is not None]
    x = {
        "src_top": src[0] if src else None,
        "src_age_d": min(pub) if pub else None,
        "src_margin": (src[0] - sum(src[1:]) / len(src[1:])) if len(src) >= 2 else None,
        "rr_top": rr[0] if rr else None,
        "rr_margin": (rr[0] - rr[1]) if len(rr) >= 2 else None,
        "rr_mean3": (sum(rr[:3]) / len(rr[:3])) if rr else None,
        "n_docs": float(min(len(docs), 8)),
        "agree_subj": float(sum(1 for d in top3 if _mentions(d, subject))) if subject else None,
        "facts": float(any(m.get("strong") or m.get("tier") == "facts" for m in metas) or tier == "facts"),
        "card": float(any(m.get("card") for m in metas) or tier == "card"),
        "ks_hit": float(bool(ks)),
        "ks_shared": float(any(m.get("index_scope") == "shared" for m in ks)),
        "ks_age_h": min(ages) if ages else None,
        "route_conf": _num(route.get("conf")),
        "lane_web": float(lane in ("", "web")),
        "lane_facts": float(lane == "facts"),
        "lane_card": float(lane == "card"),
        "key_conf": kc,
        "corrected": float(bool(getattr(f, "corrected", False))),
        "followup": float(bool(getattr(f, "followup_of", "") or getattr(f, "carried", "") or getattr(f, "inline", ""))),
        "heard_fix": float(bool(getattr(f, "resolved", ()))),
        "fillers": float(min(sum(1 for w in words if w in _FILLERS), 3)),
        "n_words": float(min(len(words), 20)),
        "intent_p": _num(getattr(f, "intent_p", None)),
        "i_factoid": float("qa_factoid" in intent or "qa_definition" in intent),
        "i_live": float(any(k in intent for k in _LIVE)),
        "_key_word": kw,                    # for the reason text (not a model input)
    }
    if v is not None or answer is not None:
        st = (v or {}).get("status") or ""
        n = (v or {}).get("n_claims") or 0
        x.update({
            "v_agree": float(st == "agree"), "v_supported": float(st == "supported"),
            "v_partial": float(st == "partial"), "v_unverified": float(st == "unverified"),
            "v_disagree": float(st == "disagree"),
            "v_found_frac": ((v or {}).get("found") or 0) / n if n else None,
            "non_answer": float(_non_answer(answer or "")),
            "ans_words": float(min(len((answer or "").split()), 80)),
            "n_cites": float(min(len(_CITE.findall(answer or "")), 6)),
            "extractive": float(fallback == "extractive"),
        })
    return x


def _non_answer(text: str) -> bool:
    from .speculator import _non_answer as na
    return na(text) if text else True


# ---------------------------------------------------------------------------------------------- scoring
def raw(feats: dict, stage: str = "post") -> float | None:
    """The fitted probability, or None when no model is installed for this stage."""
    spec = model().get(stage)
    if not spec:
        return None
    mu = dict(zip(spec["features"], spec.get("fill") or spec["mu"]))
    xs = {k: (mu.get(k, 0.0) if feats.get(k) is None else feats[k]) for k in spec["features"]}
    p = _sig(_lin(spec, xs))
    iso = spec.get("iso")
    if iso:
        p = _interp(p, iso["x"], iso["y"])
    return min(max(p, 0.01), 0.99)


def _interp(x: float, xs: list, ys: list) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            a, b = xs[i - 1], xs[i]
            return ys[i - 1] + (ys[i] - ys[i - 1]) * ((x - a) / (b - a) if b > a else 1.0)
    return ys[-1]


def bands() -> tuple[float, float]:
    b = model().get("bands") or {}
    return float(b.get("low", CALIB_LOW)), float(b.get("high", CALIB_HIGH))


def band(p: float | None) -> str:
    if p is None:
        return ""
    lo, hi = bands()
    return "high" if p >= hi else ("low" if p < lo else "mid")


def reason(x: dict, p: float | None) -> str:
    """A short plain-words reason, positive for a confident answer, the main doubt otherwise."""
    b = band(p)
    doubts, goods = [], []
    if x.get("v_disagree"):
        doubts.append("sources disagree")
    if x.get("non_answer"):
        doubts.append("sources don't say")
    kc = x.get("key_conf")
    if kc is not None and kc < 0.85 and x.get("_key_word"):
        doubts.append(f"heard '{x['_key_word']}' unclearly")
    if x.get("corrected"):
        doubts.append("you corrected yourself")
    if x.get("extractive"):
        doubts.append("quoted from the top source")
    if x.get("v_unverified") and not x.get("card"):
        doubts.append("couldn't check it against sources")
    if x.get("i_live") and (x.get("src_age_d") or 0) > 7:
        doubts.append("sources may be out of date")
    if (x.get("rr_top") or 1.0) < 0.45 and not (x.get("card") or x.get("facts")):
        doubts.append("weak match in sources")
    if x.get("card"):
        goods.append("live data")
    if x.get("ks_shared"):
        goods.append("verified by other users")
    elif x.get("ks_hit"):
        goods.append("from your earlier answers")
    if x.get("facts"):
        goods.append("Wikidata agrees")
    if x.get("v_agree"):
        goods.append(f"{max(2, int(x.get('agree_subj') or 2))} sources agree")
    elif x.get("v_supported"):
        goods.append("matches a source")
    elif (x.get("agree_subj") or 0) >= 2:
        goods.append(f"{int(x['agree_subj'])} sources on topic")
    if b == "high":
        return goods[0] if goods else "strong source match"
    if b == "low":
        return doubts[0] if doubts else "check this"
    return (doubts or goods or ["some doubt"])[0]


def assess(feats: dict, stage: str = "post") -> dict:
    """{conf, band, reason, stage} for the metric / UI; {} when no model is installed."""
    p = raw(feats, stage)
    if p is None:
        return {}
    return {"conf": round(p, 3), "band": band(p), "reason": reason(feats, p), "stage": stage}


def compact(feats: dict) -> dict:
    """The model inputs, rounded, for the metric (eval/calib_fit.py can refit from live runs)."""
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in feats.items()
            if not k.startswith("_") and v is not None}


def label(p: float | None) -> str:
    """Badge words: "High confidence" / "Medium" / "Low"."""
    return {"high": "High confidence", "mid": "Medium confidence", "low": "Low confidence"}.get(band(p), "")


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------------------------------- router use
def alt_query(f, searched: set) -> str:
    """One more search for a low-confidence answer: the subject alone (its own article), else the whole question as
    said; never a query already searched."""
    for q in ((f.entity_query() if hasattr(f, "entity_query") else ""), getattr(f, "resolved_text", "")):
        q = (q or "").strip()
        if q and q.lower() not in {s.lower() for s in searched}:
            return q
    return ""


async def free_docs(qtext: str, f) -> list:
    """Free second opinion (CALIB_ROUTE middle band): the question's Wikidata fact if it parses, and the local
    leads index's article on its subject (the same free lanes server/recheck.py uses after the answer)."""
    from . import recheck
    out = []
    try:
        _, docs = await recheck._facts(qtext)
        out += list(docs or [])
    except Exception:  # noqa: BLE001
        pass
    subject = (f.entity_query() or (f.phrases[-1] if getattr(f, "phrases", ()) else "")) if f is not None else ""
    try:
        out += list(await recheck._leads(subject, qtext))
    except Exception:  # noqa: BLE001
        pass
    return out


def route_action(p: float | None, paid_left: bool, web_ok: bool) -> str:
    """What the router does with the pre-answer confidence: "" answer now | "free" one free lookup first | "paid" one
    more paid search first (only inside the question's paid budget; else the free lookup)."""
    b = band(p)
    if b == "mid":
        return "free"
    if b == "low":
        return "paid" if (paid_left and web_ok) else "free"
    return ""
