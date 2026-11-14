"""The shared Keystone: public, verified knowledge pooled across users (the network effect).

What may enter it is an explicit ALLOWLIST, enforced at the boundary (`admit`), never by the caller's good intent:
  - typed edges (subject) -[relation]-> value whose relation is a known PUBLIC relation (relkey.RELATIONS), and
  - whose provenance is "wikidata" (a fact from the facts lane / a cached Wikidata neighbourhood), or
    "verified" (an answer that server/verify.py checked against >= 1 cited source: status supported / agree).
Everything else is refused: user facts and the "you" node, conversation activation, memory edges, unverified or
disagreeing answers, live relations past their TTL, anything whose subject or value looks personal (an email, a phone
number, a handle). Each edge carries provenance, verify status, source URL, as-of time and the relation's TTL; no user
id is stored (only a contribution count).

`SharedKeystone` is the API boundary (get / publish / stats). `SQLiteShared` implements it on a local file; the same
interface could be served remotely later. Lookup order in the app: the user's own Keystone (personal overlay), then
this store, then the VOI gate decides on paid web.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .relkey import RELATIONS, TTL_S, canon

ALLOWED_PROV = {"wikidata", "verified"}
ALLOWED_VERIFY = {"supported", "agree", "wikidata"}
_PERSONAL = re.compile(r"@|\b\d{3}[ -.]?\d{3}[ -.]?\d{4}\b|\bhttps?://|\b(?:my|our|i|me|you|your)\b")


@dataclass
class SharedEdge:
    subject: str               # canonical subject name
    relation: str
    value: str
    prov: str                  # wikidata | verified
    verify: str                # supported | agree | wikidata
    as_of: float
    ttl: float
    qid: str = ""
    source: str = ""
    url: str = ""
    contributions: int = 1
    hits: int = 0

    def fresh(self, now: float) -> bool:
        return self.ttl <= 0 or now - self.as_of <= self.ttl


def admit(e: SharedEdge) -> str | None:
    """None if the edge may enter the shared store, else why not (the allowlist)."""
    if e.prov not in ALLOWED_PROV:
        return f"provenance {e.prov!r} not shareable"
    if e.verify not in ALLOWED_VERIFY:
        return f"verify status {e.verify!r} not shareable"
    if e.relation not in RELATIONS:
        return f"relation {e.relation!r} not a public relation"
    if not e.subject or e.subject.startswith(("you", "@")) or e.subject in ("you", "me", "i", "my"):
        return "personal or unresolved subject"
    if _PERSONAL.search(e.subject.lower()) or _PERSONAL.search(e.value.lower()[:200]):
        return "looks personal"
    if RELATIONS[e.relation][1] == "live" and e.ttl > TTL_S["live"]:
        return "live relation with a long TTL"
    return None


class SharedKeystone:
    """API boundary. Implementations: SQLiteShared (local file); a remote client could implement the same three."""

    def get(self, subject: str, relation: str, now: float | None = None) -> list[SharedEdge]:
        raise NotImplementedError

    def publish(self, e: SharedEdge) -> str | None:
        """Admit + store; returns None when stored, else the refusal reason."""
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError


class SQLiteShared(SharedKeystone):
    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        self.refused = 0
        with self.lock:
            self.db.execute("""CREATE TABLE IF NOT EXISTS edges (subject TEXT, relation TEXT, value TEXT, prov TEXT,
                verify TEXT, as_of REAL, ttl REAL, qid TEXT, source TEXT, url TEXT, contributions INTEGER, hits INTEGER,
                PRIMARY KEY (subject, relation, value))""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS alias (name TEXT PRIMARY KEY, subject TEXT)""")
            self.db.commit()

    def _subject(self, name: str) -> str:
        c = canon(name)
        row = self.db.execute("SELECT subject FROM alias WHERE name = ?", (c,)).fetchone()
        if row:
            return row[0]
        if c.endswith("s"):
            row = self.db.execute("SELECT subject FROM alias WHERE name = ?", (c[:-1],)).fetchone()
            if row:
                return row[0]
        return c

    def get(self, subject: str, relation: str, now: float | None = None) -> list[SharedEdge]:
        now = time.time() if now is None else now
        with self.lock:
            s = self._subject(subject)
            rows = self.db.execute("SELECT subject, relation, value, prov, verify, as_of, ttl, qid, source, url, "
                                   "contributions, hits FROM edges WHERE subject = ? AND relation = ?",
                                   (s, relation)).fetchall()
            out = [SharedEdge(*r) for r in rows]
            out = [e for e in out if e.fresh(now)]
            if out:
                self.db.execute("UPDATE edges SET hits = hits + 1 WHERE subject = ? AND relation = ?", (s, relation))
            return sorted(out, key=lambda e: (-e.contributions, -e.as_of))

    def publish(self, e: SharedEdge) -> str | None:
        why = admit(e)
        if why:
            self.refused += 1
            return why
        with self.lock:
            s = canon(e.subject)
            self.db.execute("INSERT OR IGNORE INTO alias (name, subject) VALUES (?, ?)", (s, s))
            cur = self.db.execute("UPDATE edges SET as_of = MAX(as_of, ?), contributions = contributions + 1, "
                                  "verify = CASE WHEN verify = 'agree' THEN verify ELSE ? END "
                                  "WHERE subject = ? AND relation = ? AND value = ?",
                                  (e.as_of, e.verify, s, e.relation, e.value))
            if cur.rowcount == 0:
                self.db.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (s, e.relation, e.value, e.prov, e.verify, e.as_of, e.ttl, e.qid, e.source, e.url,
                                 1, 0))
            self.db.commit()
        return None

    def alias(self, name: str, subject: str):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO alias (name, subject) VALUES (?, ?)", (canon(name), canon(subject)))
            self.db.commit()

    def stats(self) -> dict:
        with self.lock:
            n, c = self.db.execute("SELECT COUNT(*), COALESCE(SUM(contributions), 0) FROM edges").fetchone()
            by = dict(self.db.execute("SELECT prov, COUNT(*) FROM edges GROUP BY prov").fetchall())
        return {"edges": n, "contributions": c, "by_prov": by, "refused": self.refused, "path": self.path}

    def dump(self) -> list[tuple]:
        """Every row (tests / audits: prove what is in the shared store)."""
        with self.lock:
            return self.db.execute("SELECT * FROM edges").fetchall()
