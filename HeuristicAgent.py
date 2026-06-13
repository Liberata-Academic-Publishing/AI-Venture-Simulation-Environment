from __future__ import annotations

from Agent import Agent
from config import SIM
from Paper import (
    BAD_FAITH_REVIEW,
    BAD_REVIEW_TIMESTEPS,
    GOOD_FAITH_REVIEW,
    GOOD_REVIEW_TIMESTEPS,
    MIN_PAPER_QUALITY,
    MIN_REVIEW_EFFORT_THRESHOLD,
    Paper,
    REVIEW_PARADIGM_DISCRETE,
    accrual_rate_from_quality,
    review_accrual_bump,
)

EXPECTED_WRITE_PROGRESS = SIM.expected_write_progress
MAX_FORECAST_EFFORT = SIM.max_forecast_effort


class HeuristicAgent(Agent):
    """Agent that claims and finishes reviews to maximize forecast academic capital.

    Each timestep it picks the action with the best projected return: in the
    marketplace phase it claims the most valuable listed paper if that beats its
    work-phase alternative, and in the work phase it either keeps investing in an
    active review (while the marginal accrual bump outweighs a timestep of
    writing) or finishes and writes.
    """

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        forecast_horizon_timesteps: int = 30,
        name: str | None = None,
    ):
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            name=name,
        )
        self.forecast_horizon_timesteps = forecast_horizon_timesteps

    # ---- marketplace phase ----------------------------------------------
    def choose_marketplace_action(self) -> Paper | None:
        listed = [
            paper
            for paper in Agent.all_papers
            if self._can_review(paper) and paper.offered_share(self) > 0.0
        ]
        if not listed:
            return None

        best = max(listed, key=self._score_claim)
        claim_value = self._score_claim(best)
        if claim_value <= 0.0:
            return None

        if self.active_review_paper is None:
            return best if claim_value > self._write_value() else None

        # Grabbing finalizes the active review at its current effort, then starts
        # the new one. Compare against the best work-phase alternative.
        current = self.active_review_paper
        finish_now = self._review_value(
            current, self.active_review_effort, self.forecast_horizon_timesteps
        )
        not_grab = max(
            finish_now + self._write_value(),
            self._score_continue_review(current),
        )
        return best if finish_now + claim_value > not_grab else None

    # ---- work phase ------------------------------------------------------
    def choose_work_action(self) -> tuple[str, Paper | None]:
        paper = self.active_review_paper
        if paper is None:
            return "write_paper", None

        if self._score_continue_review(paper) > self._score_finish_and_write(paper):
            return "peer_review", paper
        return "finish_review_write_paper", None

    def choose_review_kind(self, paper: Paper) -> str:
        """Discrete-mode choice between fixed bad- and good-faith review work."""
        bad = self._score_fixed_review(paper, BAD_REVIEW_TIMESTEPS)
        good = self._score_fixed_review(paper, GOOD_REVIEW_TIMESTEPS)
        return GOOD_FAITH_REVIEW if good > bad else BAD_FAITH_REVIEW

    # ---- scoring ---------------------------------------------------------
    def _review_value(self, paper: Paper, effort: float, horizon: float) -> float:
        """Forecast capital from owning the agreed review share of ``paper``."""
        share = self._prospective_share(paper)
        if share <= 0.0:
            return 0.0
        future_rate = paper.accrual_rate * (
            1.0 + review_accrual_bump(effort, paper.quality)
        )
        return share * (paper.current_ac + future_rate * max(0.0, horizon))

    def _score_claim(self, paper: Paper) -> float:
        """Value of claiming ``paper`` and completing a minimum-effort review."""
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            return max(
                self._score_fixed_review(paper, BAD_REVIEW_TIMESTEPS),
                self._score_fixed_review(paper, GOOD_REVIEW_TIMESTEPS),
            )
        return self._review_value(
            paper, MIN_REVIEW_EFFORT_THRESHOLD, self.forecast_horizon_timesteps
        )

    def _score_fixed_review(self, paper: Paper, effort: float) -> float:
        horizon = self.forecast_horizon_timesteps - max(0.0, effort - 1.0)
        return self._review_value(paper, effort, horizon)

    def _score_finish_and_write(self, paper: Paper) -> float:
        """Finish the active review now and spend this timestep writing."""
        return (
            self._review_value(
                paper, self.active_review_effort, self.forecast_horizon_timesteps
            )
            + self._write_value()
        )

    def _score_continue_review(self, paper: Paper) -> float:
        """Invest one more timestep, then finish at the higher effort (no writing)."""
        if self.active_review_effort >= MAX_FORECAST_EFFORT:
            return 0.0
        next_effort = self.active_review_effort + self.review_effort_delta()
        return self._review_value(
            paper, next_effort, self.forecast_horizon_timesteps - 1
        )

    def _write_value(self) -> float:
        """Forecast capital from one timestep of progress on the agent's next paper."""
        quality = (
            self.next_paper_quality
            if self.next_paper_quality is not None
            else max(MIN_PAPER_QUALITY, self.intrinsic_talent)
        )
        paper_rate = accrual_rate_from_quality(quality)
        full_value = paper_rate * self.forecast_horizon_timesteps
        timesteps_to_finish = max(
            1.0, self.paper_completion_threshold() / max(EXPECTED_WRITE_PROGRESS, 1e-9)
        )
        return full_value / timesteps_to_finish

    # ---- shares ----------------------------------------------------------
    def _prospective_share(self, paper: Paper) -> float:
        helper = getattr(paper, "estimate_review_share", None)
        if helper is not None:
            return max(0.0, float(helper(self)))
        return 0.0

    def _current_share(self, paper: Paper) -> float:
        return paper.share_distribution.get(self, 0.0)

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))
        return paper.author is not self
