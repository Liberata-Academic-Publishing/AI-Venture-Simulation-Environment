from __future__ import annotations

import math

from typing import Any

class Paper:

    def __init__(
        self,
        author: Any,
        current_ac: float = 0.0,
        ac_accrual_rate: float | None = None,
        completion_progress: float = 1.0,
        review_available: bool = True,
        reviewer_ac_threshold: float = 10.0,
        low_ac_good_share: float = 0.03,
        high_ac_good_share: float = 0.05,
        low_ac_bad_share: float = 0.003,
        high_ac_bad_share: float = 0.005,
        review_share_log_decay: float = 0.25,
        max_total_reviewer_share: float = 0.25,
        good_review_effort: float = 1.0,
        bad_review_effort: float = 0.05,
        max_review_boost: float = 0.50,
    ) -> None:
        if author is None:
            raise ValueError("Paper author cannot be None.")

        self.author = author
        self.current_ac = self._nonnegative_float(current_ac, "current_ac")
        self.completion_progress = self._nonnegative_float(
            completion_progress, "completion_progress"
        )
        self.review_available = bool(review_available)

        inferred_rate = self._infer_base_accrual_rate(author)
        self.base_accrual_rate = self._nonnegative_float(
            inferred_rate if ac_accrual_rate is None else ac_accrual_rate,
            "ac_accrual_rate",
        )
        self.ac_accrual_rate = self.base_accrual_rate

        self.reviewer_ac_threshold = self._nonnegative_float(
            reviewer_ac_threshold, "reviewer_ac_threshold"
        )
        self.low_ac_good_share = self._validate_share_value(
            low_ac_good_share, "low_ac_good_share"
        )
        self.high_ac_good_share = self._validate_share_value(
            high_ac_good_share, "high_ac_good_share"
        )
        self.low_ac_bad_share = self._validate_share_value(
            low_ac_bad_share, "low_ac_bad_share"
        )
        self.high_ac_bad_share = self._validate_share_value(
            high_ac_bad_share, "high_ac_bad_share"
        )
        self.review_share_log_decay = self._nonnegative_float(
            review_share_log_decay, "review_share_log_decay"
        )
        self.max_total_reviewer_share = self._validate_share_value(
            max_total_reviewer_share, "max_total_reviewer_share"
        )
        self.good_review_effort = self._nonnegative_float(
            good_review_effort, "good_review_effort"
        )
        self.bad_review_effort = self._nonnegative_float(
            bad_review_effort, "bad_review_effort"
        )
        self.max_review_boost = self._nonnegative_float(
            max_review_boost, "max_review_boost"
        )

        self.share_distribution: dict[Any, float] = {author: 1.0}
        self.review_records: list[dict[str, Any]] = []
        self.num_peer_reviews = 0
        self.num_good_faith_reviews = 0
        self.num_bad_faith_reviews = 0
        self.review_quality_score = 0.0

    def accrue_ac(self, time_steps: float = 1.0) -> float:
        """Increase current AC by accrual rate times elapsed timesteps."""
        elapsed = self._nonnegative_float(time_steps, "time_steps")
        self.current_ac += self.ac_accrual_rate * elapsed
        return self.current_ac

    def add_share(self, agent: Any) -> bool:
        """Add a good-faith peer-review contribution if valid.
        Returns True when a review is added. Returns False for normal
        simulation-invalid actions such as duplicate review, self-review,
        unavailable review slots, or exhausted reviewer share budget.
        """
        return self._add_review(agent, review_type="good_faith")

    def add_bad_share(self, agent: Any) -> bool:
        """Add a bad-faith peer-review contribution if valid.
        Bad-faith review uses lower effort and a lower share schedule. This is
        a provisional MVP distinction, not an official Liberata mechanism.
        """
        return self._add_review(agent, review_type="bad_faith")

    def set_share(self, agent: Any, share: float) -> bool:
        """Set a contributor share and keep total paper shares equal to 1.0.
        Raises:
            ValueError: if share is outside [0.0, 1.0] or if reviewer shares
            would exceed 100% of the paper.
        """
        if agent is None:
            raise ValueError("Share agent cannot be None.")

        share_value = self._validate_share_value(share, "share")

        if agent is self.author:
            reviewer_total = self._total_reviewer_share()
            if share_value + reviewer_total > 1.0 + 1e-12:
                raise ValueError("Author share plus reviewer shares exceeds 1.0.")
            self.share_distribution[self.author] = share_value
            return True

        reviewer_total_without_agent = sum(
            existing_share
            for contributor, existing_share in self.share_distribution.items()
            if contributor is not self.author and contributor is not agent
        )
        new_reviewer_total = reviewer_total_without_agent + share_value
        if new_reviewer_total > 1.0 + 1e-12:
            raise ValueError("Total reviewer shares cannot exceed 1.0.")

        self.share_distribution[agent] = share_value
        self.share_distribution[self.author] = max(0.0, 1.0 - new_reviewer_total)
        return True

    def _add_review(self, agent: Any, review_type: str) -> bool:
        if agent is None or agent is self.author:
            return False
        if not self.review_available or self._has_reviewed(agent):
            return False

        is_good_faith = review_type == "good_faith"
        share = self._reviewer_share(agent, is_good_faith=is_good_faith)
        remaining_share = self.max_total_reviewer_share - self._total_reviewer_share()
        allocated_share = min(share, max(0.0, remaining_share))
        if allocated_share <= 0.0:
            self.review_available = False
            return False

        self.set_share(agent, allocated_share)

        effort = self.good_review_effort if is_good_faith else self.bad_review_effort
        quality_delta = self._review_quality_delta(agent, effort)
        self.review_quality_score += quality_delta
        self.num_peer_reviews += 1

        if is_good_faith:
            self.num_good_faith_reviews += 1
        else:
            self.num_bad_faith_reviews += 1

        self.review_records.append(
            {
                "reviewer": agent,
                "type": review_type,
                "share": allocated_share,
                "effort": effort,
                "quality_delta": quality_delta,
            }
        )
        self._recalculate_accrual_rate()

        if self._total_reviewer_share() >= self.max_total_reviewer_share - 1e-12:
            self.review_available = False

        return True

    def _reviewer_share(self, agent: Any, is_good_faith: bool) -> float:
        reviewer_ac = self._agent_number(agent, "academic_capital", default=0.0)
        if is_good_faith:
            base_share = (
                self.high_ac_good_share
                if reviewer_ac >= self.reviewer_ac_threshold
                else self.low_ac_good_share
            )
        else:
            base_share = (
                self.high_ac_bad_share
                if reviewer_ac >= self.reviewer_ac_threshold
                else self.low_ac_bad_share
            )

        discount = 1.0 + self.review_share_log_decay * math.log1p(self.num_peer_reviews)
        return base_share / discount

    def _review_quality_delta(self, agent: Any, effort: float) -> float:
        experience = self._agent_number(agent, "intrinsic_talent", default=0.0)
        if experience == 0.0:
            experience = self._agent_number(agent, "academic_capital", default=0.0)

        effort_component = math.log1p(max(0.0, effort))
        experience_component = math.log1p(max(0.0, experience))
        return effort_component * experience_component

    def _recalculate_accrual_rate(self) -> None:
        # Provisional, not official: saturating boost from accumulated review value.
        review_boost = self.max_review_boost * (1.0 - math.exp(-self.review_quality_score))
        self.ac_accrual_rate = self.base_accrual_rate * (1.0 + review_boost)

    def _has_reviewed(self, agent: Any) -> bool:
        return any(record["reviewer"] is agent for record in self.review_records)

    def _total_reviewer_share(self) -> float:
        return sum(
            share
            for contributor, share in self.share_distribution.items()
            if contributor is not self.author
        )

    @staticmethod
    def _infer_base_accrual_rate(author: Any) -> float:
        talent = Paper._agent_number(author, "intrinsic_talent", default=1.0)
        return max(0.0, talent)

    @staticmethod
    def _agent_number(agent: Any, attribute: str, default: float) -> float:
        value = getattr(agent, attribute, default)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(number) or math.isinf(number):
            return default
        return number

    @staticmethod
    def _nonnegative_float(value: float, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric.") from exc
        if math.isnan(number) or math.isinf(number) or number < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative number.")
        return number

    @staticmethod
    def _validate_share_value(value: float, name: str) -> float:
        number = Paper._nonnegative_float(value, name)
        if number > 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0.")
        return number
