"""Regressions from the 0.2.0 install (2026-11-23): one question's material leaking into the next, and the on-device
writer overruling Wikidata facts.

Live: "who is the prime minister of India" then "what are five good books to read" answered the second with the
first's sources; "who is the president of India" answered Modi; "how old is he" answered 52 and Keystone stored it.
"""
import datetime as dt

import numpy as np

from server import answer, knowledge
from server.lanes import Doc
from server.speculator import Speculator


def test_prime_minister_is_head_of_government_not_head_of_state():
    assert knowledge.parse_fact("who is the prime minister of india").prop == "head_gov"
    assert knowledge.parse_fact("who is the president of india").prop == "head_state"
    assert knowledge.PROPS["head_gov"][0] == ["P6"] and knowledge.PROPS["head_state"][0] == ["P35"]


def test_how_old_is_a_birth_date_lookup_and_the_age_is_computed():
    assert knowledge.parse_fact("how old is narendra modi").prop == "born_date"
    assert answer.asks_prop("how old is he (Narendra Modi)") == "born_date"
    s = answer.fact_sentence("how old is he", "Narendra Modi", "born_date", ["September 17, 1950"], dt.date(2026, 11, 23))
    assert s == "Narendra Modi is 76 (born September 17, 1950)."
    assert answer.fact_sentence("how old is he", "X", "born_date", ["1950"]) == ""      # year only: no computed age


def test_fact_sentence_says_the_role_as_asked_and_drops_descriptions():
    assert answer.fact_sentence("who is the prime minister of india", "India", "head_gov", ["Narendra Modi"]) == \
        "The prime minister of India is Narendra Modi."
    assert answer.fact_sentence("what is the capital of australia", "Australia", "capital",
                                ["Canberra (capital city of Australia)"]) == "The capital of Australia is Canberra."


def test_fact_answer_only_for_the_property_asked(monkeypatch):
    monkeypatch.setattr(answer, "backend", lambda: "local")
    fact = Doc(id="wikidata:Q668:P6", title="India · Wikidata", text="India: head of government: Narendra Modi.",
               url="", source="Wikidata", score=1.0,
               meta={"strong": True, "fact": {"prop": "head_gov", "values": ["Narendra Modi"], "item": "India"}})

    class S(answer.Stream):
        def __init__(self, q):          # no task: just the fact path
            self.question, self.docs, self.tokens = q, [fact], []
            self.by = None

        def _finish(self, *a):
            pass
    a = S("how old is he (Narendra Modi)")
    assert not a._fact_answer()                       # a carried PM fact is no answer to "how old is he"
    b = S("who is the prime minister of India")
    assert b._fact_answer() and b.tokens == ["The prime minister of India is Narendra Modi. [1]"]


def test_local_writer_gets_the_previous_turn_only_for_a_follow_up():
    assert answer.local_context("What are five good books to read", "prev") == ""
    assert answer.local_context("how old is he", "prev") == "prev"
    assert answer.local_context("and in France?", "prev") == "prev"


def _spec():
    s = Speculator(lambda e: None, "headstart", 0.0)
    return s


def test_carried_docs_do_not_count_as_this_questions_results():
    s = _spec()
    v = np.ones(4, np.float32) / 2
    s.seed({"old": (Doc(id="old", title="Prime minister", text="...", url="", source="x"), v)})
    assert s.rank(v, 1)                  # the carried doc is in the pool (follow-ups use it) ...
    assert not s._own_results()          # ... but it is not a result of THIS question: the empty-pool retry still runs


def test_carried_docs_rank_only_for_follow_ups():
    s = _spec()
    v = np.ones(4, np.float32) / 2
    s.seed({"old": (Doc(id="old", title="Narendra Modi", text="...", url="", source="x"), v)})

    class F:
        route, carried, mem_ref, resolved_text = "web", "", None, "who is the president of india"
    s.persona = None
    assert s._rank_for(F(), v, 3) == []              # not a follow-up: last turn's sources are no answer
    F.carried = "Narendra Modi"
    assert s._rank_for(F(), v, 3)                    # "how old is he": they are
