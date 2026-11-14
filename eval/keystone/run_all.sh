#!/bin/bash
# Keystone pipeline: hash + verifier -> VOI weights (dev stream) -> theta on the dev operating curve (regret no worse
# than today's search cache) -> replays + operating curve on the test stream.
# Families / streams / fixture are inputs (make_families.py, make_stream.py, record_fixture.py: network, run once).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
EMB=${EMB_MS:-}
$PY eval/keystone/train_ahash.py --bits 64 --ablate > runs/index/train_ahash.log 2>&1
$PY eval/keystone/fit_voi.py > runs/index/fit_voi.log 2>&1
$PY eval/keystone/replay.py --sweep --stream runs/index/stream_s2.jsonl --out runs/index/voi_sweep_dev.json ${EMB:+--embed-ms $EMB} > runs/index/voi_sweep_dev.log 2>&1
$PY - <<'PYEOF'
import json
from pathlib import Path
d = json.loads(Path("runs/index/voi_sweep_dev.json").read_text())
ref = d["a"]["regret_rate"]
ok = [p for p in d["points"] if p["regret_rate"] <= max(ref, 0.005)]
theta = min(p["theta"] for p in ok) if ok else 0.99
m = Path("server/keystone/voi_model.json")
j = json.loads(m.read_text()); j["theta"] = theta; j["theta_rule"] = f"dev regret <= today's cache ({ref})"
m.write_text(json.dumps(j))
print("theta", theta, "dev cache regret", ref)
PYEOF
$PY eval/keystone/replay.py --out runs/index/replay.json ${EMB:+--embed-ms $EMB} > runs/index/replay.log 2>&1
$PY eval/keystone/replay.py --stream runs/index/stream_s2.jsonl --out runs/index/replay_dev.json --anytime none ${EMB:+--embed-ms $EMB} > runs/index/replay_dev.log 2>&1
$PY eval/keystone/replay.py --sweep --out runs/index/voi_sweep.json ${EMB:+--embed-ms $EMB} > runs/index/voi_sweep.log 2>&1
echo done
