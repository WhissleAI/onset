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


# ------------------------------------------------------------------------------------------------------------------
# The wider allowlist (KDIR step 6): verified question -> answer cells, so the network effect also covers questions
# that are not single-relation factoids. Same boundary discipline as edges: `admit_cell` is the allowlist.
#   allowed : a PUBLIC question (no first / second person, no personal-looking value, not a follow-up that leans on a
#             conversation) whose answer passed server/verify.py against at least one cited source (supported / agree),
#             with its public source URL; live questions only with their live TTL (minutes)
#   refused : anything personal (my / I / you / emails / phones / handles / house numbers), unverified answers, user
#             facts and activation (they never reach this API at all), local-moment / private-corpus sources
# A cell stores: the question text, its query vector (float16, 768), its Arc signature (answer type, relation,
# subject, intent), the answer payload, source + URL, as-of, TTL, contributions. No user id.
_PRIVATE_SRC = re.compile(r"\b(?:local|moments?|transcript|memory|onset index)\b", re.I)
_PERSONAL_Q = re.compile(r"\b(?:my|mine|our|ours|i|i'm|i've|me|we|us|you|your|yours)\b|@|\d{3}[ -.]?\d{3}[ -.]?\d{4}"
                         r"|\b\d{1,5}\s+[a-z]+\s+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr)\b", re.I)


@dataclass
class SharedCell:
    question: str
    payload: str
    verify: str                # supported | agree
    as_of: float
    ttl: float
    source: str = ""
    url: str = ""
    arc: dict | None = None
    contributions: int = 1


def admit_cell(c: SharedCell) -> str | None:
    if c.verify not in ("supported", "agree"):
        return f"verify status {c.verify!r} not shareable"
    if not c.url or _PRIVATE_SRC.search(c.source or ""):
        return "no public cited source"
    if _PERSONAL_Q.search(c.question or ""):
        return "personal question"
    if _PERSONAL.search((c.payload or "").lower()[:320]):
        return "looks personal"
    if (c.arc or {}).get("live") and c.ttl > TTL_S["live"]:
        return "live question with a long TTL"
    return None


class SharedCells:
    """In-memory (or SQLite-backed) store of admitted question -> answer cells with a vector index. lookup() returns
    a keystone Cand on the 'shared_q' path (question NN + the Arc answer-equivalence verifier)."""

    def __init__(self, path: Path | str = ":memory:", min_cos: float = 0.80):
        import numpy as np
        self.np = np
        self.cells: list[SharedCell] = []
        self.vecs = np.zeros((0, 768), np.float32)
        self.fids: list[str] = []
        self.refused = 0
        self.min_cos = min_cos
        self.hits = 0

    def publish(self, c: SharedCell, qvec, fid: str = "") -> str | None:
        why = admit_cell(c)
        if why:
            self.refused += 1
            return why
        np = self.np
        v = np.asarray(qvec, np.float32)[:768]
        if len(self.cells):
            s = self.vecs @ v
            j = int(np.argmax(s))
            if s[j] >= 0.985 and self.cells[j].payload == c.payload:        # the same question again: a contribution
                self.cells[j].contributions += 1
                self.cells[j].as_of = max(self.cells[j].as_of, c.as_of)
                return None
        self.cells.append(c)
        self.vecs = np.vstack([self.vecs, v[None, :]])
        self.fids.append(fid)
        return None

    def publish_verified(self, q, docs, now: float, fid: str = "") -> str | None:
        """The replay's stand-in for the live path (answer model + verify): the answer is the question's key
        sentences from its top source; shared only when verify.check finds its claims in >= 1 source."""
        from .. import verify
        from .cells import key_sentences, toks
        if not docs or q.qvec is None:
            return "no docs"
        payload = key_sentences(docs[0].text, toks(q.text))
        v = verify.check(payload, docs[:3], q.text)
        ttl = TTL_S["live"] if q.live else (q.key.ttl_s if q.key is not None else 30 * 86400.0)
        arc = dict(q.arc or {})
        arc["live"] = bool(q.live)
        return self.publish(SharedCell(q.text, payload, v["status"], now, ttl, docs[0].source, docs[0].url, arc),
                            q.qvec, fid)

    def lookup(self, q, now: float):
        from .cells import Cell
        from .core import Cand
        from . import verify_eq
        np = self.np
        if not len(self.cells) or q.qvec is None:
            return None
        s = self.vecs @ np.asarray(q.qvec, np.float32)[:768]
        for j in np.argsort(-s)[:3]:
            c = self.cells[int(j)]
            if s[j] < self.min_cos:
                return None
            if now - c.as_of > c.ttl:
                continue
            cell = Cell(id=-1, kind="value", title=c.question[:120], payload=c.payload, source=c.source or "Onset users",
                        url=c.url, question=c.question, arc={"shared": True, "verify": c.verify,
                                                             "contributions": c.contributions},
                        t=c.as_of, ttl=c.ttl, doc_id=f"sharedq:{int(j)}")
            cand = Cand("shared_q", [(cell, float(s[j]))], float(s[j]), c.as_of)
            ver = verify_eq.load()
            if ver is not None:
                cand.eq_p = ver.p(verify_eq.pair_features(float(s[j]), q.arc, c.arc or {}, q.key, None))
            cand.fid = self.fids[int(j)]
            self.hits += 1
            return cand
        return None

    def stats(self) -> dict:
        return {"cells": len(self.cells), "contributions": sum(c.contributions for c in self.cells),
                "refused": self.refused, "lookups_served": self.hits}

    def dump(self) -> list:
        return [(c.question, c.payload, c.verify, c.source, c.url) for c in self.cells]
