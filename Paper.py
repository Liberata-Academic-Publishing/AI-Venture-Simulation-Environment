from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agent import Agent

DEFAULT_ACCRUAL_RATE = 1.0
DEFAULT_REVIEW_SHARE = 0.01
MIN_REVIEW_EFFORT_THRESHOLD = 10.0
REVIEW_EFFORT_PER_DAY = 1.0
BASE_REVIEW_ACCRUAL_BUMP = 0.20
FIRST_EXTRA_DAY_BUMP = 0.10
EXTRA_DAY_DECAY = 0.85
DEFAULT_REVIEWER_AC_THRESHOLD = 10.0
DEFAULT_HIGH_AC_REVIEW_SHARE = 0.02
DEFAULT_MAX_REVIEWER_SHARE = 0.25


def review_accrual_bump(effort: float) -> float:
    """Total accrual-rate bump fraction for a review completed at ``effort``.

    Effort below ``MIN_REVIEW_EFFORT_THRESHOLD`` yields 0. At exactly 10 the
    reviewer earns the base bump. Each day past 10 adds a positive marginal
    bump; each marginal after day 11 is smaller than the previous day's.
    """
    if effort < MIN_REVIEW_EFFORT_THRESHOLD:
        return 0.0

    bump = BASE_REVIEW_ACCRUAL_BUMP
    marginal = FIRST_EXTRA_DAY_BUMP
    extra_days = int(effort - MIN_REVIEW_EFFORT_THRESHOLD)
    for _ in range(extra_days):
        bump += marginal
        marginal *= EXTRA_DAY_DECAY
    return bump


class Paper:
    """Minimal paper model for the peer-review marketplace."""

    def __init__(
        self,
        author: Agent,
        accrual_rate: float = DEFAULT_ACCRUAL_RATE,
        current_ac: float = 0.0,
        share_distribution: dict[Agent, float] | None = None,
        completion_progress: float = 1.0,
        review_available: bool = True,
        reviewer_ac_threshold: float = DEFAULT_REVIEWER_AC_THRESHOLD,
        low_ac_review_share: float = DEFAULT_REVIEW_SHARE,
        high_ac_review_share: float = DEFAULT_HIGH_AC_REVIEW_SHARE,
        max_reviewer_share: float = DEFAULT_MAX_REVIEWER_SHARE,
    ):
        if author is None:
            raise ValueError("author cannot be None")

        self.author = author
        self.accrual_rate = self._nonnegative_float(accrual_rate, "accrual_rate")
        self.current_ac = self._nonnegative_float(current_ac, "current_ac")
        self.share_distribution = (
            {author: 1.0} if share_distribution is None else dict(share_distribution)
        )
        self._validate_share_distribution()
        self.completion_progress = self._nonnegative_float(
            completion_progress,
            "completion_progress",
        )
        self.review_available = bool(review_available)
        self.reviewer_ac_threshold = self._nonnegative_float(
            reviewer_ac_threshold,
            "reviewer_ac_threshold",
        )
        self.low_ac_review_share = self._validate_share_value(
            low_ac_review_share,
            "low_ac_review_share",
        )
        self.high_ac_review_share = self._validate_share_value(
            high_ac_review_share,
            "high_ac_review_share",
        )
        self.max_reviewer_share = self._validate_share_value(
            max_reviewer_share,
            "max_reviewer_share",
        )
        self.completed_peer_reviews = 0
        self.review_in_progress_by: Agent | None = None
        self.reviewed_by: set[Agent] = set()
        self.review_records: list[dict[str, object]] = []

    @property
    def ac_accrual_rate(self) -> float:
        """Compatibility alias for notes/code that use the longer AC name."""
        return self.accrual_rate

    @ac_accrual_rate.setter
    def ac_accrual_rate(self, value: float):
        self.accrual_rate = self._nonnegative_float(value, "ac_accrual_rate")

    def is_valid_review_effort(self, effort: float) -> bool:
        return self._nonnegative_float(effort, "review_effort") >= MIN_REVIEW_EFFORT_THRESHOLD

    def add_review(
        self,
        agent: Agent,
        effort: float,
        share: float = DEFAULT_REVIEW_SHARE,
    ) -> float:
        """Apply a completed review once effort is at least the minimum threshold."""
        review_effort = self._nonnegative_float(effort, "review_effort")
        if not self.is_valid_review_effort(review_effort):
            return 0.0

        return self._add_review_share(agent, share, effort=review_effort)

    def add_share(self, agent: Agent, share: float = DEFAULT_REVIEW_SHARE) -> float:
        """Compatibility wrapper for a minimum-threshold completed review."""
        return self.add_review(agent, MIN_REVIEW_EFFORT_THRESHOLD, share)

    def set_share(self, agent: Agent, share: float):
        """Set a contributor share.

        This method is intentionally stricter than review actions. Normal
        simulation-invalid review actions no-op, but direct invalid share edits
        raise ValueError so model mistakes are visible.
        """
        if agent is None:
            raise ValueError("agent cannot be None")

        share_value = self._validate_share_value(share, "share")
        other_total = sum(
            current_share
            for contributor, current_share in self.share_distribution.items()
            if contributor is not agent
        )
        if other_total + share_value > 1.0 + 1e-12:
            raise ValueError("total paper shares cannot exceed 1.0")

        self.share_distribution[agent] = share_value
        self._refresh_review_available()

    def advance_accrual(self, days: int = 1):
        self.accrue_ac(days)

    def accrue_ac(self, time_steps: float = 1.0) -> float:
        """Increase current AC using the current provisional accrual rate."""
        elapsed = self._nonnegative_float(time_steps, "time_steps")
        self.current_ac += self.accrual_rate * elapsed
        return self.current_ac

    def estimate_review_share(
        self,
        agent: Agent,
        share: float = DEFAULT_REVIEW_SHARE,
    ) -> float:
        if not self.can_start_review(agent):
            return 0.0

        base_share = self._base_review_share(agent, share)
        decayed_share = base_share / self._review_damping()
        author_share = self.share_distribution.get(self.author, 0.0)
        remaining_review_budget = self.max_reviewer_share - self._total_reviewer_share()
        return min(
            decayed_share,
            max(0.0, author_share),
            max(0.0, remaining_review_budget),
        )

    def estimate_accrual_rate_after_review(self, effort: float) -> float:
        bump = review_accrual_bump(effort) / self._review_damping()
        return self.accrual_rate * (1.0 + bump)

    def can_start_review(self, agent: Agent) -> bool:
        if agent == self.author:
            return False
        if agent in self.reviewed_by:
            return False
        if self.review_in_progress_by == agent:
            return True
        if not self.review_available or self.review_in_progress_by is not None:
            return False
        return True

    def start_review(self, agent: Agent) -> bool:
        if not self.can_start_review(agent):
            return False

        self.review_in_progress_by = agent
        self.review_available = False
        return True

    def finish_review(self, agent: Agent):
        if self.review_in_progress_by == agent:
            self.review_in_progress_by = None
            self._refresh_review_available()

    def _add_review_share(
        self,
        agent: Agent,
        share: float,
        effort: float,
    ) -> float:
        review_share = self.estimate_review_share(agent, share)
        if review_share <= 0.0:
            return 0.0

        self.share_distribution[self.author] = (
            self.share_distribution.get(self.author, 0.0) - review_share
        )
        self.share_distribution[agent] = (
            self.share_distribution.get(agent, 0.0) + review_share
        )
        self.accrual_rate = self.estimate_accrual_rate_after_review(effort)
        self.completed_peer_reviews += 1
        self.reviewed_by.add(agent)
        self.review_records.append(
            {
                "reviewer": agent,
                "share": review_share,
                "effort": effort,
                "accrual_bump": review_accrual_bump(effort) / self._review_damping(),
                "accrual_rate": self.accrual_rate,
            }
        )
        self._refresh_review_available()
        return review_share

    def _review_damping(self) -> float:
        return math.log2(self.completed_peer_reviews + 2)

    def _refresh_review_available(self):
        author_share = self.share_distribution.get(self.author, 0.0)
        remaining_reviewer_share = self.max_reviewer_share - self._total_reviewer_share()
        self.review_available = author_share > 0.0 and remaining_reviewer_share > 0.0

    def _base_review_share(self, agent: Agent, share: float) -> float:
        requested_share = self._validate_share_value(share, "share")
        reviewer_ac = getattr(agent, "academic_capital", 0.0)
        try:
            reviewer_ac = float(reviewer_ac)
        except (TypeError, ValueError):
            reviewer_ac = 0.0

        if requested_share == self.low_ac_review_share:
            if reviewer_ac >= self.reviewer_ac_threshold:
                return self.high_ac_review_share
            return self.low_ac_review_share

        return requested_share

    def _total_reviewer_share(self) -> float:
        return sum(
            share
            for contributor, share in self.share_distribution.items()
            if contributor is not self.author
        )

    def _validate_share_distribution(self):
        total = 0.0
        for contributor, share in self.share_distribution.items():
            if contributor is None:
                raise ValueError("share_distribution cannot contain None contributors")
            total += self._validate_share_value(share, "share_distribution share")
        if total > 1.0 + 1e-12:
            raise ValueError("initial share_distribution cannot exceed 1.0 total")

    @staticmethod
    def _nonnegative_float(value: float, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if math.isnan(number) or math.isinf(number) or number < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        return number

    @staticmethod
    def _validate_share_value(value: float, name: str) -> float:
        number = Paper._nonnegative_float(value, name)
        if number > 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
        return number
