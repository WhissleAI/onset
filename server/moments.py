"""Recording -> utterance-level moments: the code shared by the offline index build (index/build_index.py) and the
in-app ingest of the user's own recordings (server/userindex.py), so both produce the same moments, tags and vectors.

  arc_labels(out, domain)        Arc v1 predict() output -> words [(text, start, end)] + labels (names, numbers, events,
                                 roles, speakers) on the clip's clock
  segment(words, p, bounds)      word timestamps -> moments (pause / sentence / length rules, tiny moments merged)
  build_docs(src, words, lab)    moments -> docs.jsonl rows (title "<label> · mm:ss", meta.tags, neighbour context)
  tag_doc(tagger, doc)           Arc Tagger read-out on a moment's text -> meta.tags (+ meta.chips: people, places,
                                 orgs, numbers, for display)
  embed_text_of / embed_title_of what EmbeddingGemma 2 embeds for a moment's text vector
"""
from __future__ import annotations

import re

SR = 16000
# segmentation (seconds): close a moment at a pause >= GAP once it is >= MIN long; at a sentence end once >= SOFT;
# always at HARD or at a pause >= BREAK. Moments with < MIN_WORDS words are merged into a neighbour.
SEG = {"default": dict(GAP=0.55, MIN=3.0, SOFT=9.0, HARD=16.0, BREAK=1.8, MIN_WORDS=5),
       "assistant": dict(GAP=0.6, MIN=0.0, SOFT=9.0, HARD=16.0, BREAK=0.6, MIN_WORDS=1)}
AUDIO_MIN, AUDIO_MAX = 0.6, 30.0     # moment audio slices embedded (s)
CTX_WORDS = 14  # neighbour words kept in meta.context_before / context_after (not embedded)
MAX_TAGS = 14
MAX_CHIPS = 8

# Arc / Arc Tagger entity type -> the chip kind the UI shows
CHIP_KIND = {"person": "person", "place": "place", "location": "place", "gpe": "place", "city": "place",
             "country": "place", "organization": "org", "org": "org", "team": "org", "company": "org"}


def mmss(t: float) -> str:
    """Moment start as mm:ss (h:mm:ss past an hour)."""
    t = int(round(t))
    if t >= 3600:
        return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"
    return f"{t // 60:02d}:{t % 60:02d}"


def arc_labels(a: dict, domain: str) -> tuple[list, dict]:
    """Arc v1 offline predict() output (transcript='builtin') -> (words [(text, start, end)], labels
    {names:[(s,e,text,kind)], numbers:[(s,e,value,kind)], events:[(t,tag)], roles:[(s,e,role)], spk:[(s,e,speaker)],
    phases, words_source, labels_source})."""
    words = []
    for u in a.get("transcript") or []:
        for w in u.get("words") or []:
            if (w.get("text") or "").strip() and w.get("start") is not None and w.get("end") is not None:
                words.append((w["text"].strip(), float(w["start"]), float(w["end"])))
    role_of = {}
    for k, v in (a.get("speakers") or {}).items():
        r = (((v.get("groups") or {}).get("speaker.role") or {}).get("label") or "").replace("speaker.role.", "")
        if r and domain not in ("unknown", "talk") and not (r == "device_user" and not domain.startswith("assistant")):
            role_of[k] = r
    lab = {"names": [(n["start"], n["end"], (n.get("entity") or n.get("heard") or "").strip(" .,?!"),
                      (n.get("tag") or "").split(".")[-1]) for n in a.get("names") or []],
           "numbers": [(n["start"], n["end"], n.get("heard"), n.get("kind") or n.get("kind_head"))
                       for n in a.get("numbers") or []],
           "events": [(e["t"], e["tag"]) for e in a.get("events") or []
                      if not e["tag"].startswith(("speaker.", "meta."))],
           "roles": [(d["start"], d["end"], role_of.get(d["speaker"])) for d in a.get("diarization") or []
                     if role_of.get(d["speaker"])],
           "spk": [(d["start"], d["end"], d["speaker"]) for d in a.get("diarization") or []],
           "phases": [],
           "words_source": "Arc v1 built-in decoder (offline predict)", "labels_source": "Arc v1"}
    return words, lab


def shift_labels(words: list, lab: dict, dt: float, spk_prefix: str = "") -> tuple[list, dict]:
    """Move a chunk's words + labels onto the recording's clock (chunked ingest of long files). Speaker ids are
    per-chunk in Arc, so they get a chunk prefix."""
    words = [(w, a + dt, b + dt) for w, a, b in words]
    out = dict(lab)
    out["names"] = [(a + dt, b + dt, t, k) for a, b, t, k in lab["names"]]
    out["numbers"] = [(a + dt, b + dt, v, k) for a, b, v, k in lab["numbers"]]
    out["events"] = [(t + dt, tag) for t, tag in lab["events"]]
    out["roles"] = [(a + dt, b + dt, r) for a, b, r in lab["roles"]]
    out["spk"] = [(a + dt, b + dt, spk_prefix + s) for a, b, s in lab["spk"]]
    out["phases"] = [(a + dt, b + dt, p) for a, b, p in lab.get("phases") or []]
    return words, out


def merge_labels(parts: list[dict]) -> dict:
    if not parts:
        return {"names": [], "numbers": [], "events": [], "roles": [], "spk": [], "phases": [],
                "words_source": "", "labels_source": ""}
    out = {k: [] for k in ("names", "numbers", "events", "roles", "spk", "phases")}
    for p in parts:
        for k in out:
            out[k] += p.get(k) or []
    out["words_source"] = parts[0].get("words_source", "")
    out["labels_source"] = parts[0].get("labels_source", "")
    return out


_END = re.compile(r"[.?!]['\"]?$")


def segment(words, p, bounds=()):
    """bounds: extra cut times (e.g. gold speaker-turn starts); a moment never spans one."""
    bounds = sorted(bounds)
    segs, cur = [], []
    for w in words:
        if cur and any(cur[-1][2] - 0.25 <= b <= w[1] + 0.25 and b > cur[0][1] + 0.3 for b in bounds):
            segs.append(cur)
            cur = []
        if cur:
            gap = w[1] - cur[-1][2]
            dur = cur[-1][2] - cur[0][1]
            if (gap >= p["BREAK"] or (gap >= p["GAP"] and dur >= p["MIN"]) or (dur >= p["SOFT"] and _END.search(cur[-1][0]))
                    or dur >= p["HARD"]):
                segs.append(cur)
                cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    # merge tiny moments into the nearer neighbour
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, sg in enumerate(segs):
            if len(sg) >= p["MIN_WORDS"]:
                continue
            gp = sg[0][1] - segs[i - 1][-1][2] if i > 0 else 1e9
            gn = segs[i + 1][0][1] - sg[-1][2] if i + 1 < len(segs) else 1e9
            j = i - 1 if gp <= gn else i + 1
            if min(gp, gn) > 3.0:
                continue
            a, b = sorted((i, j))
            segs[a:b + 1] = [segs[a] + segs[b]]
            changed = True
            break
    return segs


def overlap_pick(items, a, b):
    """items [(s,e,label)] -> label with most overlap with [a,b]."""
    acc = {}
    for s, e, lab in items:
        ov = min(b, e) - max(a, s)
        if ov > 0 and lab:
            acc[lab] = acc.get(lab, 0) + ov
    return max(acc, key=acc.get) if acc else ""


def _chip_kind(kind: str) -> str:
    return CHIP_KIND.get((kind or "").lower(), "")


def build_docs(src: dict, words: list, lab: dict, id_prefix: str | None = None) -> list[dict]:
    """One recording's words + labels -> docs.jsonl rows. src: {id, label, domain, about?, seg?, split_on_speaker?}.
    Also records meta.chips (Arc's own names + numbers inside the moment) for display; tag_doc adds the tagger's."""
    bounds = [a for a, _, _ in lab["spk"]] if src.get("split_on_speaker") else ()
    segs = segment(words, SEG[src.get("seg", "default")], bounds)
    texts = [" ".join(w[0] for w in sg) for sg in segs]
    docs = []
    pre = id_prefix if id_prefix is not None else src["id"]
    for i, sg in enumerate(segs):
        a, b = sg[0][1], sg[-1][2]
        text = texts[i][:600]
        prev_tail = " ".join(w[0] for w in segs[i - 1][-CTX_WORDS:]) if i > 0 else ""
        next_head = " ".join(w[0] for w in segs[i + 1][:CTX_WORDS]) if i + 1 < len(segs) else ""
        tags, chips = [], []
        for ns, ne, t, k in lab["names"]:
            if ns < b + 0.2 and ne > a - 0.2 and t:
                tags.append(t)
                if _chip_kind(k):
                    chips.append({"text": t, "kind": _chip_kind(k)})
        for ns, ne, t, k in lab["numbers"]:
            if ns < b + 0.2 and ne > a - 0.2 and t:
                tags.append(str(t))
                chips.append({"text": str(t).strip(" .,?!"), "kind": "number"})
        for t, tag in lab["events"]:
            if a - 0.3 <= t <= b + 0.6:
                tags.append(tag.replace("event.", "").replace("discourse.", ""))
        docs.append({
            "id": f"{pre}-{i:03d}",
            "title": f"{src['label']} · {mmss(a)}",
            "text": text,
            "url": "",
            "t": round(a, 2),
            "meta": {"source": src["id"], "source_label": src["label"], "about": src.get("about", ""),
                     "end": round(b, 2), "speaker": overlap_pick(lab["spk"], a, b),
                     "role": overlap_pick(lab["roles"], a, b) or "",
                     "phase": overlap_pick(lab.get("phases") or [], a, b) or "",
                     "tags": tags, "domain": src["domain"], "words_source": lab["words_source"],
                     "context_before": prev_tail, "context_after": next_head, "chips": chips},
        })
    return docs


def _dedup(items, key=lambda x: x.lower()):
    seen, out = set(), []
    for t in items:
        k = key(t)
        if t and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def tag_doc(tagger, d: dict) -> dict:
    """Arc Tagger (native int8) on the moment's text: entities, numbers, events, on-intents -> meta.tags (at most 14,
    deduplicated, Arc's labels first); people / places / orgs / numbers -> meta.chips; on-intents -> meta.intents."""
    r = tagger.tag_text(d["text"])
    tags = list(d["meta"]["tags"])
    chips = list(d["meta"].get("chips") or [])
    intents = []
    for sp in r.get("spans") or []:
        fam, typ = sp.get("family"), sp.get("type", "")
        if fam in ("entity", "number") and sp.get("score", 0) >= sp.get("thr", 1) and typ != "entity.numeral":
            txt = sp["text"].strip(" .,?!")
            tags.append(txt)
            kind = "number" if fam == "number" else _chip_kind(typ.split(".")[-1])
            if kind and txt:
                chips.append({"text": txt, "kind": kind})
    for ev in r.get("events") or []:
        tags.append(ev["type"].replace("event.", ""))
    for u in r.get("utterances") or []:
        for it in u.get("intent") or []:
            if it.get("on"):
                tags.append(it["tag"].replace("discourse.", ""))
                intents.append(it["tag"].replace("discourse.", ""))
    d["meta"]["tags"] = _dedup(tags)[:MAX_TAGS]
    if "chips" in d["meta"]:
        d["meta"]["chips"] = _dedup(chips, key=lambda c: c["text"].lower())[:MAX_CHIPS]
    if intents:
        d["meta"]["intents"] = _dedup(intents)
    return d


def embed_text_of(d):
    """What the text vector sees: the moment's own words. (Adding neighbour context was tried and lowered recall@1 on
    the local question set, 0.7 -> 0.5, so the neighbours only go in meta for the answer step.)"""
    return d["text"][:600]


def embed_title_of(d):
    m = d["meta"]
    role = f" · {m['role'].replace('_', ' ')}" if m.get("role") else ""
    return f"{d['title']}{role} · {m['about']}"
