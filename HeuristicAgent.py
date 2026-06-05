from __future__ import annotations

from Agent import Agent, GOOD_FAITH_REVIEW_DAYS, PAPER_THRESHOLD
from Paper import (
    BAD_FAITH_ACCRUAL_BUMP,
    DEFAULT_ACCRUAL_RATE,
    DEFAULT_REVIEW_SHARE,
    GOOD_FAITH_ACCRUAL_BUMP,
    Paper,
)

EXPECTED_WRITE_PROGRESS = 0.5


class HeuristicAgent(Agent):
    """Agent that picks the action with the best rough capital forecast."""

    # Review options the agent will consider, as (forecast kind, action name).
    # Subclasses can narrow this (e.g. a bad-faith-only agent drops good_faith).
    REVIEW_ACTIONS: tuple[tuple[str, str], ...] = (
        ("good_faith", "peer_review"),
        ("bad_faith", "bad_faith_review"),
    )

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

    def choose_action(self) -> tuple[str, Paper | None]:
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

        for paper in reviewable:
            for kind, action in self.REVIEW_ACTIONS:
                score = self._score_review(paper, kind)
                if score > best_score:
                    best_action = action
                    best_paper = paper
                    best_score = score

        return best_action, best_paper

    def _score_write(self) -> float:
        score = self._forecast_capital()
        progress_after_action = self.paper_progress + EXPECTED_WRITE_PROGRESS
        if progress_after_action >= PAPER_THRESHOLD:
            remaining_days = max(0, self.forecast_horizon_days - 1)
            score += DEFAULT_ACCRUAL_RATE * remaining_days
        return score

    def _score_review(self, paper: Paper, kind: str) -> float:
        delay_days = GOOD_FAITH_REVIEW_DAYS if kind == "good_faith" else 1
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
