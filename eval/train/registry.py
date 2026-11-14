"""Model versions with rollback.

Each trained model is a set of files the server loads (MODELS below: the path under the repo). A version is a copy of
those files in runs/models/<model>/vNNN/ with meta.json (when, trained on what, the gate metrics). CURRENT names the
version installed in server/. v000 is a snapshot of what shipped before the loop's first promotion.

  promote(model, cand_dir, meta)   copy the candidate in as the next version, install it into server/, CURRENT = it
  rollback(model[, version])       re-install the previous (or a named) version
  .venv/bin/python eval/train/registry.py status | rollback <model> [vNNN]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODELS = {
    "calib": ["server/calib_model.json"],
    "kdir": ["server/keystone/kdir_model.json"],
    "ttl": ["server/keystone/ttl_model.json"],
    "director": ["server/director/director.npz", "server/director/director.json"],
}


class Registry:
    def __init__(self, root: Path | None = None, repo: Path | None = None):
        self.root = Path(root or C.MODELS)
        self.repo = Path(repo or C.ROOT)

    def dir(self, model: str) -> Path:
        return self.root / model

    def versions(self, model: str) -> list[str]:
        d = self.dir(model)
        return sorted(p.name for p in d.glob("v[0-9][0-9][0-9]") if p.is_dir()) if d.exists() else []

    def current(self, model: str) -> str | None:
        p = self.dir(model) / "CURRENT"
        return p.read_text().strip() if p.exists() else None

    def meta(self, model: str, version: str | None = None) -> dict:
        v = version or self.current(model)
        return C.read_json(self.dir(model) / v / "meta.json", {}) if v else {}

    def files(self, model: str) -> list[Path]:
        return [self.repo / f for f in MODELS[model]]

    def _history(self, model: str, rec: dict):
        C.append_jsonl(self.dir(model) / "history.jsonl", [{"ts": C.now_iso(), **rec}])

    def _copy_in(self, model: str, src_files: list[Path], meta: dict) -> str:
        vs = self.versions(model)
        v = f"v{(int(vs[-1][1:]) + 1) if vs else 0:03d}"
        d = self.dir(model) / v
        d.mkdir(parents=True, exist_ok=False)
        for s in src_files:
            shutil.copy2(s, d / s.name)
        C.write_json(d / "meta.json", {"version": v, "model": model, "created": C.now_iso(), **meta})
        return v

    def bootstrap(self, model: str) -> str | None:
        """Snapshot the shipped files as v000 (once). None when the model's files aren't in this checkout."""
        if self.versions(model):
            return self.current(model)
        fs = self.files(model)
        if not all(f.exists() for f in fs):
            return None
        v = self._copy_in(model, fs, {"source": "shipped (before the training loop)"})
        (self.dir(model) / "CURRENT").write_text(v)
        self._history(model, {"event": "bootstrap", "version": v})
        return v

    def install(self, model: str, version: str):
        d = self.dir(model) / version
        for f in self.files(model):
            src = d / f.name
            if not src.exists():
                raise FileNotFoundError(f"{model} {version} has no {f.name}")
        for f in self.files(model):                 # all present: now copy (atomic per file)
            tmp = f.with_name(f.name + ".tmp")
            shutil.copy2(d / f.name, tmp)
            tmp.replace(f)
        (self.dir(model) / "CURRENT").write_text(version)

    def promote(self, model: str, cand_dir: Path, meta: dict) -> str:
        self.bootstrap(model)
        names = [f.name for f in self.files(model)]
        src = [Path(cand_dir) / n for n in names]
        missing = [s.name for s in src if not s.exists()]
        if missing:
            raise FileNotFoundError(f"candidate lacks {missing}")
        prev = self.current(model)
        v = self._copy_in(model, src, {**meta, "previous": prev})
        self.install(model, v)
        self._history(model, {"event": "promote", "version": v, "previous": prev})
        return v

    def rollback(self, model: str, version: str | None = None) -> str:
        cur = self.current(model)
        if version is None:
            version = self.meta(model, cur).get("previous") if cur else None
            if not version:
                vs = [v for v in self.versions(model) if v != cur]
                if not vs:
                    raise ValueError(f"{model}: nothing to roll back to")
                version = vs[-1]
        self.install(model, version)
        self._history(model, {"event": "rollback", "version": version, "from": cur})
        return version

    def status(self) -> dict:
        out = {}
        for m in MODELS:
            cur = self.current(m)
            md = self.meta(m, cur) if cur else {}
            out[m] = {"current": cur, "versions": len(self.versions(m)), "created": md.get("created"),
                      "cycle": md.get("cycle"), "present": all(f.exists() for f in self.files(m))}
        return out


if __name__ == "__main__":
    r = Registry()
    a = sys.argv[1:] or ["status"]
    if a[0] == "status":
        print(json.dumps(r.status(), indent=1))
    elif a[0] == "rollback":
        print(r.rollback(a[1], a[2] if len(a) > 2 else None))
    else:
        sys.exit("usage: registry.py status | rollback <model> [vNNN]")
