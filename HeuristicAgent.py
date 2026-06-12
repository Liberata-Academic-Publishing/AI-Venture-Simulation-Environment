from __future__ import annotations

from Agent import Agent, EXPECTED_REVIEW_EFFORT_PER_TURN, PAPER_THRESHOLD
from config import SIM
from Paper import (
    DEFAULT_ACCRUAL_RATE,
    DEFAULT_REVIEW_SHARE,
    MIN_REVIEW_EFFORT_THRESHOLD,
    Paper,
    review_accrual_bump,
)

EXPECTED_WRITE_PROGRESS = SIM.expected_write_progress
MAX_FORECAST_EFFORT = SIM.max_forecast_effort
CONTINUE_MARGINAL_WEIGHT = SIM.continue_marginal_weight
PREFERRED_EXTRA_REVIEW_DAYS = SIM.preferred_extra_review_days


class HeuristicAgent(Agent):
    """Agent that picks the action with the best projected academic capital return."""

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
        if self.should_offer_review_choice():
            return self._choose_review_fate()

        reviewable = [
            paper for paper in Agent.all_papers if self._can_review(paper)
        ]
        if not reviewable:
            return "write_paper", None

        best_action = "write_paper"
        best_paper = None
        best_score = self._score_write()

        for paper in reviewable:
            for target_effort in self._forecast_effort_targets():
                score = self._score_start_review(paper, target_effort)
                if score > best_score:
                    best_action = "peer_review"
                    best_paper = paper
                    best_score = score

        return best_action, best_paper

    def _choose_review_fate(self) -> tuple[str, Paper | None]:
        paper = self.active_review_paper
        if paper is None:
            return "write_paper", None

        best_action = "peer_review"
        best_paper = paper
        best_score = self._score_continue_review(paper)

        finish_write_score = self._score_finish_and_write(paper)
        if finish_write_score >= best_score:
            best_action = "finish_review_write_paper"
            best_paper = None
            best_score = finish_write_score

        for candidate in Agent.all_papers:
            if not self._can_review(candidate):
                continue
            for target_effort in self._forecast_effort_targets():
                score = self._score_finish_and_start_peer_review(
                    paper, candidate, target_effort
                )
                if score >= best_score:
                    best_action = "finish_review_peer_review"
                    best_paper = candidate
                    best_score = score

        return best_action, best_paper

    def _forecast_effort_targets(self) -> range:
        start = int(MIN_REVIEW_EFFORT_THRESHOLD)
        end = min(MAX_FORECAST_EFFORT, start + self.forecast_horizon_days) + 1
        return range(start, end)

    def _score_write(self) -> float:
        score = self._forecast_capital()
        progress_after_action = self.paper_progress + EXPECTED_WRITE_PROGRESS
        if progress_after_action >= PAPER_THRESHOLD:
            remaining_days = max(0, self.forecast_horizon_days - 1)
            score += DEFAULT_ACCRUAL_RATE * remaining_days
        return score

    def _score_start_review(self, paper: Paper, target_effort: float) -> float:
        delay_days = max(1.0, target_effort / EXPECTED_REVIEW_EFFORT_PER_TURN)
        return self._score_review_outcome(paper, target_effort, delay_days)

    def _score_continue_review(self, paper: Paper) -> float:
        """Forecast value of spending one more day, then finishing at higher effort."""
        current = self.active_review_effort
        if current < MIN_REVIEW_EFFORT_THRESHOLD:
            # Below the reward cliff finishing now is worthless, so value
            # continuing as the forecast payoff of pushing on to the threshold.
            days_to_threshold = MIN_REVIEW_EFFORT_THRESHOLD - current
            return self._score_review_outcome(
                paper, MIN_REVIEW_EFFORT_THRESHOLD, delay_days=days_to_threshold
            )
        if current >= MAX_FORECAST_EFFORT:
            return 0.0

        finish_now = self._score_finish_at_current_effort(paper)
        next_effort = min(
            current + EXPECTED_REVIEW_EFFORT_PER_TURN,
            MAX_FORECAST_EFFORT,
        )
        horizon_after_finish = max(0, self.forecast_horizon_days - 1)
        next_path = self._score_review_outcome(paper, next_effort, delay_days=1.0)
        next_path += self._marginal_effort_bonus(
            paper, current, next_effort, horizon_after_finish
        )
        marginal_gain = max(0.0, next_path - finish_now)
        days_past_minimum = max(0.0, current - MIN_REVIEW_EFFORT_THRESHOLD)
        gain_weight = max(
            0.0,
            1.0 - days_past_minimum / PREFERRED_EXTRA_REVIEW_DAYS,
        )
        return finish_now + marginal_gain * gain_weight

    def _marginal_effort_bonus(
        self,
        paper: Paper,
        from_effort: float,
        to_effort: float,
        horizon_days: float,
    ) -> float:
        """Extra value from a higher completion effort on the reviewed paper."""
        if to_effort <= from_effort or horizon_days <= 0:
            return 0.0

        damping = paper._review_damping()  # noqa: SLF001 — forecast mirrors paper logic
        marginal_bump = (
            review_accrual_bump(to_effort) - review_accrual_bump(from_effort)
        ) / damping
        share = self._current_share(paper) + self._estimate_review_share(paper)
        return (
            CONTINUE_MARGINAL_WEIGHT
            * share
            * paper.current_ac
            * marginal_bump
            * horizon_days
        )

    def _score_finish_and_write(self, paper: Paper) -> float:
        """Single-world forecast: finish the active review and write today."""
        if self.active_review_effort < MIN_REVIEW_EFFORT_THRESHOLD:
            return self._score_write()

        score = self._score_finish_at_current_effort(paper)
        progress_after_action = self.paper_progress + EXPECTED_WRITE_PROGRESS
        if progress_after_action >= PAPER_THRESHOLD:
            score += DEFAULT_ACCRUAL_RATE * max(0, self.forecast_horizon_days - 1)
        return score

    def _score_finish_and_start_peer_review(
        self,
        current_paper: Paper,
        candidate: Paper,
        target_effort: float,
    ) -> float:
        """Single-world forecast: finish the active review and complete a new one."""
        current_effort = self.active_review_effort
        if current_effort < MIN_REVIEW_EFFORT_THRESHOLD:
            return 0.0
        if target_effort < MIN_REVIEW_EFFORT_THRESHOLD:
            return 0.0

        candidate_delay = max(
            1.0, target_effort / EXPECTED_REVIEW_EFFORT_PER_TURN
        )
        remaining_after_candidate = max(
            0, self.forecast_horizon_days - candidate_delay
        )
        score = 0.0

        for reviewed_paper in Agent.all_papers:
            share = self._current_share(reviewed_paper)
            current_ac = reviewed_paper.current_ac
            current_rate = reviewed_paper.accrual_rate

            if reviewed_paper is current_paper:
                review_share = self._estimate_review_share(reviewed_paper)
                future_rate = self._estimate_accrual_rate_after_review(
                    reviewed_paper, current_effort
                )
                future_ac = current_ac + future_rate * self.forecast_horizon_days
                score += (share + review_share) * future_ac
            elif reviewed_paper is candidate:
                review_share = self._estimate_review_share(reviewed_paper)
                future_rate = self._estimate_accrual_rate_after_review(
                    reviewed_paper, target_effort
                )
                future_ac = (
                    current_ac
                    + current_rate * candidate_delay
                    + future_rate * remaining_after_candidate
                )
                score += (share + review_share) * future_ac
            else:
                future_ac = current_ac + current_rate * self.forecast_horizon_days
                score += share * future_ac

        return score

    def _score_finish_at_current_effort(self, paper: Paper) -> float:
        if self.active_review_effort < MIN_REVIEW_EFFORT_THRESHOLD:
            return 0.0
        return self._score_review_outcome(
            paper,
            self.active_review_effort,
            delay_days=0.0,
        )

    def _score_review_outcome(
        self,
        paper: Paper,
        completion_effort: float,
        delay_days: float,
    ) -> float:
        if completion_effort < MIN_REVIEW_EFFORT_THRESHOLD:
            return 0.0

        remaining_days = max(0, self.forecast_horizon_days - delay_days)
        score = 0.0

        for candidate in Agent.all_papers:
            current_share = self._current_share(candidate)
            current_ac = candidate.current_ac
            current_rate = candidate.accrual_rate

            if candidate is paper:
                review_share = self._estimate_review_share(candidate)
                future_rate = self._estimate_accrual_rate_after_review(
                    candidate, completion_effort
                )
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

    def _estimate_review_share(self, paper: Paper) -> float:
        helper = getattr(paper, "estimate_review_share", None)
        if helper is not None:
            return max(0.0, float(helper(self)))

        author_share = paper.share_distribution.get(paper.author, 0.0)
        return min(DEFAULT_REVIEW_SHARE, max(0.0, author_share))

    def _estimate_accrual_rate_after_review(
        self, paper: Paper, effort: float
    ) -> float:
        helper = getattr(paper, "estimate_accrual_rate_after_review", None)
        if helper is not None:
            return float(helper(effort))

        damping = paper._review_damping()  # noqa: SLF001 — forecast mirrors paper logic
        bump = review_accrual_bump(effort) / damping
        return paper.accrual_rate * (1.0 + bump)
