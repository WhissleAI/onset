"""The search director's addressed head (server/director.py, the shipped v2 model) on the open-mic test scenes, for
eval/arcvalue/textlevel.py's open-mic check: p(addressed) at the last director update of each Flux turn.

The director reads the Flux event stream plus the Arc audio streams, which a text-only replay doesn't have. The recipe
that trained it (PromptingNemo-private recipes/headstart_director) saved those inputs per update for its held-out
open-mic scenes (data/ds4/test_om/*.npz: feats + feats2 = all four v2 streams, the word-hash tokens, `turn` = the Flux
turn index of openmic_flux_turns.jsonl). This runs the shipped numpy model over each scene in order, as the recipe's
evaluate2.py does (one recurrent state per scene), and keeps the newest p(addressed) per turn.

  DIRECTOR_DS=<.../data/ds4/test_om>  (default: the recipe checkout next to this repo)
Returns {} when the features are not on this machine; textlevel then reports the director arm as missing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DS = Path(os.environ.get("DIRECTOR_DS", ROOT.parent / "pnp-headstart-director" / "recipes" / "headstart_director" /
                         "data" / "ds4" / "test_om"))
CACHE = ROOT / "runs" / "arcvalue" / "director_addr_test_om.json"


def turn_addr(ds: Path = DS, cache: Path | None = CACHE) -> dict[tuple[str, int], float]:
    if cache is not None and cache.exists():
        d = json.loads(cache.read_text())
        if d.get("ds") == str(ds):
            return {(k.split("|")[0], int(k.split("|")[1])): v for k, v in d["addr"].items()}
    from server import director
    m = director.model()
    files = sorted(ds.glob("*.npz")) if ds.is_dir() else []
    if m is None or not m.v2 or not files:
        return {}
    out: dict[tuple[str, int], float] = {}
    for fn in files:
        z = np.load(fn)
        feats = np.concatenate([z["feats"], z["feats2"]], 1).astype(np.float32)
        state = None
        for i in range(len(feats)):
            r, state = m.step(state, feats[i], z["toks"][i], z["wtok"][i], z["wfeat"][i], z["wfeat2"][i])
            t = int(z["turn"][i])
            if t >= 0:
                out[(fn.stem, t)] = round(float(r["addr"]), 4)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"ds": str(ds), "addr": {f"{s}|{t}": p for (s, t), p in out.items()}}))
    return out


if __name__ == "__main__":
    a = turn_addr(cache=None)
    print(len(a), "turns")
