from __future__ import annotations

from Agent import Agent, EXPECTED_REVIEW_EFFORT_PER_TURN, PAPER_THRESHOLD
from Paper import (
    BAD_FAITH_ACCRUAL_BUMP,
    DEFAULT_ACCRUAL_RATE,
    DEFAULT_REVIEW_SHARE,
    GOOD_FAITH_REVIEW_EFFORT_THRESHOLD,
    GOOD_FAITH_ACCRUAL_BUMP,
    MIN_REVIEW_EFFORT_THRESHOLD,
    Paper,
)

EXPECTED_WRITE_PROGRESS = 0.5


class HeuristicAgent(Agent):
    """Agent that picks the action with the best rough capital forecast.

    Reviews are open-ended (start / continue / stop). This agent decides up
    front how much effort a review is worth -- a ``good_faith`` (full) review
    or a ``bad_faith`` (minimal) one -- then keeps reviewing until it reaches
    that effort target and stops. A learned policy could stop at any point.
    """

    # Candidate review effort levels to weigh, by classification kind. The
    # completed review is ultimately classified by the paper from real effort.
    # Subclasses can narrow this (e.g. a low-effort agent drops good_faith).
    REVIEW_TARGETS: tuple[str, ...] = ("good_faith", "bad_faith")

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        forecast_horizon_days: int = 30,
        name: str | None = None,
    ):
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            name=name,
        )
        self.forecast_horizon_days = forecast_horizon_days
        self._review_target_effort = 0.0

    def choose_action(self) -> tuple[str, Paper | None]:
        # Mid-review: keep reviewing until the chosen effort target, then stop.
        if self.active_review_paper is not None:
            if self.active_review_effort < self._review_target_effort:
                return "peer_review", self.active_review_paper
            return "stop_peer_review", None

        reviewable = [
            paper
            for paper in Agent.all_papers
            if self._can_review(paper)
        ]

        if not reviewable:
            return "write_paper", None

        best_action = "write_paper"
        best_paper = None
        best_score = self._score_write()
        best_target = 0.0

        for paper in reviewable:
            for kind in self.REVIEW_TARGETS:
                score = self._score_review(paper, kind)
                if score > best_score:
                    best_action = "peer_review"
                    best_paper = paper
                    best_score = score
                    best_target = self._target_effort_for_kind(kind)

        if best_action == "peer_review":
            self._review_target_effort = best_target
        return best_action, best_paper

    def _target_effort_for_kind(self, kind: str) -> float:
        return (
            GOOD_FAITH_REVIEW_EFFORT_THRESHOLD
            if kind == "good_faith"
            else MIN_REVIEW_EFFORT_THRESHOLD
        )

    def _score_write(self) -> float:
        score = self._forecast_capital()
        progress_after_action = self.paper_progress + EXPECTED_WRITE_PROGRESS
        if progress_after_action >= PAPER_THRESHOLD:
            remaining_days = max(0, self.forecast_horizon_days - 1)
            score += DEFAULT_ACCRUAL_RATE * remaining_days
        return score

    def _score_review(self, paper: Paper, kind: str) -> float:
        delay_days = self._review_delay_days(kind)
        remaining_days = max(0, self.forecast_horizon_days - delay_days)
        score = 0.0

        for candidate in Agent.all_papers:
            current_share = self._current_share(candidate)
            current_ac = candidate.current_ac
            current_rate = candidate.accrual_rate

            if candidate is paper:
                review_share = self._estimate_review_share(candidate, kind)
                future_rate = self._estimate_accrual_rate_after_review(candidate, kind)
                future_ac = (
                    current_ac
                    + current_rate * delay_days
                    + future_rate * remaining_days
                )
                score += (current_share + review_share) * future_ac
            else:
                future_ac = current_ac + current_rate * self.forecast_horizon_days
                score += current_share * future_ac

        return score

    def _review_delay_days(self, kind: str) -> float:
        target_effort = (
            GOOD_FAITH_REVIEW_EFFORT_THRESHOLD
            if kind == "good_faith"
            else MIN_REVIEW_EFFORT_THRESHOLD
        )
        return max(1.0, target_effort / EXPECTED_REVIEW_EFFORT_PER_TURN)

    def _forecast_capital(self) -> float:
        return sum(
            self._current_share(paper)
            * (paper.current_ac + paper.accrual_rate * self.forecast_horizon_days)
            for paper in Agent.all_papers
        )

    def _current_share(self, paper: Paper) -> float:
        return paper.share_distribution.get(self, 0.0)

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))

        return paper.author != self and getattr(paper, "review_available", True)

    def _estimate_review_share(self, paper: Paper, kind: str) -> float:
        helper = getattr(paper, "estimate_review_share", None)
        if helper is not None:
            return max(0.0, float(helper(self, kind)))

        author_share = paper.share_distribution.get(paper.author, 0.0)
        return min(DEFAULT_REVIEW_SHARE, max(0.0, author_share))

    def _estimate_accrual_rate_after_review(self, paper: Paper, kind: str) -> float:
        helper = getattr(paper, "estimate_accrual_rate_after_review", None)
        if helper is not None:
            return float(helper(kind))

        bump = (
            BAD_FAITH_ACCRUAL_BUMP
            if kind == "bad_faith"
            else GOOD_FAITH_ACCRUAL_BUMP
        )
        return paper.accrual_rate * (1.0 + bump)
