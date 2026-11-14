"""Keep the results current after a cycle: runs/train/latest.json (every cycle), and after a PROMOTION the in-app
Results data (server/scorecard.json "models": the "Model versions · last trained" line) plus a dated
"Training cycle N" section appended to PROGRESS.md in the same step, so tests/test_results_view.py's rule (every
number on the Results view is quoted verbatim from the doc it cites) keeps holding. Only numbers measured on the
fixed held-out sets go in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import registry as RG  # noqa: E402

NAMES = {"calib": "calibrator", "kdir": "Keystone director", "ttl": "TTLs", "director": "search director"}
SCORECARD = C.ROOT / "server/scorecard.json"
PROGRESS = C.ROOT / "PROGRESS.md"


def versions_line(models: dict) -> str:
    """'calibrator v001 (2026-11-14) · Keystone director v000 (2026-11-14) · ...' from the registry status."""
    parts = []
    for k in ("director", "kdir", "calib", "ttl"):
        m = models.get(k) or {}
        if m.get("version"):
            parts.append(f"{NAMES[k]} {m['version']} ({m.get('trained') or '?'})")
    return " · ".join(parts)


def model_status(reg: RG.Registry, shipped: dict) -> dict:
    out = {}
    for k in RG.MODELS:
        cur = reg.current(k)
        md = reg.meta(k, cur) if cur else {}
        trained = md.get("trained") or (md.get("created") or "")[:10] if md.get("cycle") is not None else None
        out[k] = {"version": cur, "trained": trained or (shipped.get(k) if cur else None), "cycle": md.get("cycle")}
    return out


def gate_text(model: str, g: dict) -> str:
    return f"{NAMES[model]}: " + ("; ".join(g.get("beats") or []) or "no gain") + \
        ("" if g.get("promote") else " — kept: " + "; ".join(g.get("fails") or ["did not beat"]))


def progress_section(cycle: int, date: str, line: str, rep: dict) -> str:
    lines = [f"\n## Training cycle {cycle} ({date}; scripts/train_cycle.sh, eval/train/, runs/train/cycles/c{cycle}/)",
             f"Model versions: {line}"]
    data = rep.get("data") or {}
    if data:
        lines.append(f"- Data: real rows {data.get('real_total')} (+{data.get('real_new')} this cycle: app "
                     f"{data.get('app_new', 0)}, eval replays {data.get('eval_new', 0)}); synthetic sessions "
                     f"{data.get('synth_sessions', 0)} ({data.get('synth_turns', 0)} turns, paraphrase "
                     f"${data.get('paraphrase_usd', 0):.4f}, Flux ${data.get('flux_usd', 0):.4f} est.).")
    for m, g in (rep.get("gates") or {}).items():
        lines.append(f"- {gate_text(m, g)}")
    for m, v in (rep.get("promoted") or {}).items():
        lines.append(f"- **Promoted {NAMES[m]} {v}** (fixed held-out sets; rollback: "
                     f"`.venv/bin/python eval/train/registry.py rollback {m}`).")
    return "\n".join(lines) + "\n"


def update_docs(cycle: int, rep: dict, reg: RG.Registry, dry: bool = False) -> dict:
    """Append the PROGRESS section and point server/scorecard.json's models line at it (same step)."""
    date = C.now_iso()[:10]
    sc = json.loads(SCORECARD.read_text())
    shipped = {k: v.get("trained") for k, v in ((sc.get("models") or {}).get("by_model") or {}).items()}
    st = model_status(reg, shipped)
    line = versions_line(st)
    sec = progress_section(cycle, date, line, rep)
    quote = f"Model versions: {line}"
    sc["models"] = {"line": "Model versions · last trained " + max((v.get("trained") or "") for v in st.values()),
                    "by_model": {k: {"name": NAMES[k], **v} for k, v in st.items()},
                    "source": {"doc": "PROGRESS.md", "section": f"Training cycle {cycle}", "date": date, "quote": quote}}
    if not dry:
        with PROGRESS.open("a") as f:
            f.write(sec)
        SCORECARD.write_text(with_models(SCORECARD.read_text(), sc["models"]))
    return {"section": sec, "scorecard_models": sc["models"]}


def with_models(text: str, models: dict) -> str:
    """server/scorecard.json with its last key "models" replaced (or added), the rest of the file byte-identical."""
    block = ',\n  "models": ' + json.dumps(models, ensure_ascii=False) + "\n}\n"
    i = text.find(',\n  "models": ')
    if i < 0:
        i = text.rstrip().rfind("}")
        head = text[:i].rstrip()
    else:
        head = text[:i]
    out = head + block
    json.loads(out)                                   # never write a broken file
    return out


def write_latest(rep: dict, reg: RG.Registry):
    C.write_json(C.TRAIN / "latest.json", {"updated": C.now_iso(), "cycle": rep.get("cycle"), "status": reg.status(),
                                           "last_cycle": rep})
