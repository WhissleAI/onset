"""Cost presets for Head Start's speculation (COST_MODE; config.COST_*).

speed : SPEC_PAID_MAX paid web calls per question, drafts up to DRAFT_BUDGET, a meaning is searched once its words
        have held SPEC_SETTLE_MS (the policy measured in PROGRESS "Search efficiency" / "Integration").
parity: the cost of plain search. Mid-sentence speculation uses the free lanes only (cards, Wikidata facts, local
        index, search cache); the ONE paid web call goes out when the meaning is final (end-of-turn score >=
        COST_PARITY_EOT, or, without a turn score, after SPEC_LAST_SETTLE_MS of stable words) and is the endpoint's
        search too (pool reuse). The endpoint adds a call only when that call doesn't cover the final words (a real
        meaning change). Drafts: at most COST_PARITY_DRAFTS, only on a final meaning, so the draft is the answer."""
from __future__ import annotations

from dataclasses import dataclass

from . import config

MODES = ("speed", "parity")


@dataclass(frozen=True)
class Profile:
    name: str
    paid_max: int                 # paid speculative calls per question (final included in "speed")
    final_extra: int              # paid calls the endpoint may add on top of paid_max (coverage miss / empty pool)
    draft_budget: int
    final_eot: float | None       # paid fires wait for this end-of-turn score (None: settle-time policy)
    draft_eot: float | None = None   # drafts wait for this one

    @property
    def final_max(self) -> int:
        return self.paid_max + self.final_extra


def profile(name: str | None = None) -> Profile:
    n = (name or config.COST_MODE or "speed").strip().lower()
    if n == "parity":
        return Profile("parity", max(1, config.COST_PARITY_PAID_MAX), max(0, config.COST_PARITY_FINAL_EXTRA),
                       max(0, config.COST_PARITY_DRAFTS), config.COST_PARITY_EOT, config.COST_PARITY_DRAFT_EOT)
    return Profile("speed", config.SPEC_PAID_MAX, 0, config.DRAFT_BUDGET, None)


def valid(name) -> str | None:
    n = str(name or "").strip().lower()
    return n if n in MODES else None
