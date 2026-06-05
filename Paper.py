from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agent import Agent

DEFAULT_ACCRUAL_RATE = 1.0
DEFAULT_REVIEW_SHARE = 0.01
GOOD_FAITH_ACCRUAL_BUMP = 0.25
BAD_FAITH_ACCRUAL_BUMP = 0.05


class Paper:
    """Minimal paper model for the first peer-review marketplace iteration."""

    def __init__(
        self,
        author: Agent,
        accrual_rate: float = DEFAULT_ACCRUAL_RATE,
        current_ac: float = 0.0,
        share_distribution: dict[Agent, float] | None = None,
        completion_progress: float = 1.0,
        review_available: bool = True,
    ):
        self.author = author
        self.accrual_rate = float(accrual_rate)
        self.current_ac = float(current_ac)
        self.share_distribution = (
            {author: 1.0} if share_distribution is None else dict(share_distribution)
        )
        self.completion_progress = completion_progress
        self.review_available = review_available
        self.completed_peer_reviews = 0
        self.good_faith_reviews = 0
        self.bad_faith_reviews = 0
        self.review_in_progress_by: Agent | None = None
        self.reviewed_by: set[Agent] = set()

    def add_share(self, agent: Agent, share: float = DEFAULT_REVIEW_SHARE) -> float:
        """Transfer a review share from the author and apply a good-faith accrual bump."""
        review_share = self._add_review_share(agent, share, "good_faith")
        if review_share > 0.0:
            self.good_faith_reviews += 1
        return review_share

    def add_bad_share(self, agent: Agent, share: float = DEFAULT_REVIEW_SHARE) -> float:
        """Transfer a review share from the author and apply a bad-faith accrual bump."""
        review_share = self._add_review_share(agent, share, "bad_faith")
        if review_share > 0.0:
            self.bad_faith_reviews += 1
        return review_share

    def set_share(self, agent: Agent, share: float):
        self.share_distribution[agent] = max(0.0, float(share))
        self._refresh_review_available()

    def advance_accrual(self, days: int = 1):
        self.current_ac += self.accrual_rate * days

    def estimate_review_share(
        self,
        agent: Agent,
        kind: str,
        share: float = DEFAULT_REVIEW_SHARE,
    ) -> float:
        if not self.can_start_review(agent):
            return 0.0

        decayed_share = max(0.0, float(share)) / self._review_damping()
        author_share = self.share_distribution.get(self.author, 0.0)
        return min(decayed_share, max(0.0, author_share))

    def estimate_accrual_rate_after_review(self, kind: str) -> float:
        return self.accrual_rate * (1.0 + self._review_bump(kind))

    def can_start_review(self, agent: Agent) -> bool:
        if agent == self.author or not self.review_available:
            return False
        if agent in self.reviewed_by:
            return False
        return self.review_in_progress_by in {None, agent}

    def start_review(self, agent: Agent) -> bool:
        if not self.can_start_review(agent):
            return False

        self.review_in_progress_by = agent
        return True

    def finish_review(self, agent: Agent):
        if self.review_in_progress_by == agent:
            self.review_in_progress_by = None

    def _add_review_share(self, agent: Agent, share: float, kind: str) -> float:
        review_share = self.estimate_review_share(agent, kind, share)
        if review_share <= 0.0:
            return 0.0

        self.share_distribution[self.author] = (
            self.share_distribution.get(self.author, 0.0) - review_share
        )
        self.share_distribution[agent] = (
            self.share_distribution.get(agent, 0.0) + review_share
        )
        self.accrual_rate = self.estimate_accrual_rate_after_review(kind)
        self.completed_peer_reviews += 1
        self.reviewed_by.add(agent)
        self._refresh_review_available()
        return review_share

    def _review_bump(self, kind: str) -> float:
        base_bump = (
            BAD_FAITH_ACCRUAL_BUMP
            if kind in {"bad_faith", "bad_faith_review"}
            else GOOD_FAITH_ACCRUAL_BUMP
        )
        return base_bump / self._review_damping()

    def _review_damping(self) -> float:
        return math.log2(self.completed_peer_reviews + 2)

    def _refresh_review_available(self):
        self.review_available = self.share_distribution.get(self.author, 0.0) > 0.0
