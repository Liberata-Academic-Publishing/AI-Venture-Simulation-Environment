from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import SIM
from Paper import (
    DISCRETE_PAPER_TIMESTEPS,
    DISCRETE_WRITING_EFFORT_PER_TIMESTEP,
    GOOD_FAITH_REVIEW,
    MIN_REVIEW_EFFORT_THRESHOLD,
    QUALITY_SIGMA,
    REVIEW_EFFORT_PER_TIMESTEP,
    REVIEW_PARADIGM_CONTINUOUS,
    REVIEW_PARADIGM_DISCRETE,
    Paper,
    fixed_review_effort,
    quality_multiplier,
    review_action_kind,
    review_kind_from_effort,
    validate_review_paradigm,
)

PAPER_THRESHOLD = SIM.paper_threshold
EXPECTED_REVIEW_EFFORT_PER_TURN = REVIEW_EFFORT_PER_TIMESTEP


@dataclass(frozen=True)
class ActionRecord:
    """A class to describe what an agent did on a single turn."""

    kind: str
    paper: Paper | None = None
    published: bool = False
    review_effort: float | None = None
    review_kind: str | None = None
    writing_effort: float | None = None


class Agent(ABC):

    all_papers: list[Paper] = []  # class variable shared across all agents

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        name: str | None = None,
    ):
        self.intrinsic_talent = intrinsic_talent
        self.academic_capital = academic_capital
        self.paper_progress = paper_progress
        self.review_progress = review_progress
        self.name = name
        self.active_review_paper: Paper | None = None
        self.active_review_effort = 0.0
        self.active_review_kind: str | None = None
        self.active_review_target_effort: float | None = None
        self.last_review_effort: float | None = None
        self.last_review_kind: str | None = None
        self.review_paradigm = validate_review_paradigm(SIM.review_paradigm)

        # Quality of the paper currently being written (known before/while
        # working on it). Sampled lazily the first time the agent writes.
        self.next_paper_quality: float | None = None

        # Public peer-review reputation: mean AC earned per completed review.
        self.peer_review_history: float = 0.0
        self.total_ac_from_reviews: float = 0.0
        self.completed_review_count: int = 0

    # ---- action interface (two phases per timestep) ----------------------
    @abstractmethod
    def choose_marketplace_action(self) -> Paper | None:
        """Marketplace phase: return a listed paper to claim, or ``None`` to pass.

        Claiming a paper while a review is already in progress finalizes that
        review at its current effort and immediately starts the new one.
        """

    @abstractmethod
    def choose_work_action(self) -> tuple[str, Paper | None]:
        """Work phase: return ``(action, paper)`` for an agent that did not claim.

        Actions:
          - ``"write_paper"`` — make progress on own research (``paper`` is None).
          - ``"peer_review"`` — invest one more timestep in the active review.
          - ``"finish_review_write_paper"`` — finalize the active review, then write.
        """

    def choose_review_kind(self, paper: Paper) -> str:
        """Discrete-mode review choice. Subclasses can choose bad vs good faith."""
        return GOOD_FAITH_REVIEW

    def configure_review_paradigm(self, review_paradigm: str) -> None:
        self.review_paradigm = validate_review_paradigm(review_paradigm)

    def should_offer_review_choice(self) -> bool:
        """True when the agent holds an in-progress review (continue / finish)."""
        return (
            self.review_paradigm == REVIEW_PARADIGM_CONTINUOUS
            and self.active_review_paper is not None
        )

    def available_actions(self) -> tuple[str, ...]:
        if self.should_offer_review_choice():
            return ("peer_review", "finish_review_write_paper")
        return ("write_paper",)

    # ---- phase 1: marketplace selection (instantaneous, no effort) -------
    def claim_review(
        self,
        paper: Paper,
        review_kind: str | None = None,
    ) -> ActionRecord | None:
        """Select ``paper`` to review, finalizing any active review first.

        This is pure selection: the new review starts at zero effort. The
        first unit of effort is applied in the work phase via
        :meth:`apply_initial_review_effort`. Returns a record only when an
        existing review was finalized to make room (otherwise ``None``).
        """
        finished = None
        if self.active_review_paper is not None:
            finished = self._finalize_active_review()

        paper.start_review(self)
        self.active_review_paper = paper
        self.active_review_effort = 0.0
        self.review_progress = 0.0
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            self.active_review_kind = review_kind or self.choose_review_kind(paper)
            self.active_review_target_effort = fixed_review_effort(
                self.active_review_kind
            )
        else:
            self.active_review_kind = None
            self.active_review_target_effort = None

        if finished is not None:
            return ActionRecord(
                "review_finished_peer_review",
                finished.paper,
                review_effort=finished.review_effort,
                review_kind=finished.review_kind,
            )
        return None

    # ---- phase 2: effort application (one timestep of work) --------------
    def apply_initial_review_effort(self) -> ActionRecord:
        """Apply this timestep's effort to a review claimed in the marketplace."""
        self._clear_last_review_result()
        return self._advance_active_review("review_started")

    def work_turn(self) -> ActionRecord:
        """Apply this timestep's effort for an agent that did not claim a paper."""
        self._clear_last_review_result()

        if self.active_review_paper is not None:
            if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
                return self._advance_active_review("review_continued")

            action, _ = self.choose_work_action()
            if action == "finish_review_write_paper":
                return self._finish_review_and_write()
            # Default to continuing the active review.
            return self._advance_active_review("review_continued")

        papers_before = len(Agent.all_papers)
        writing_effort = self.write_paper()
        return ActionRecord(
            "write_paper",
            published=len(Agent.all_papers) > papers_before,
            writing_effort=writing_effort,
        )

    def _finish_review_and_write(self) -> ActionRecord:
        finished = self._finalize_active_review()
        papers_before = len(Agent.all_papers)
        writing_effort = self.write_paper()
        published = len(Agent.all_papers) > papers_before
        if finished is None:
            return ActionRecord(
                "write_paper",
                published=published,
                writing_effort=writing_effort,
            )
        return ActionRecord(
            "review_finished_write",
            finished.paper,
            published=published,
            review_effort=finished.review_effort,
            review_kind=finished.review_kind,
            writing_effort=writing_effort,
        )

    def _advance_active_review(self, progress_kind: str) -> ActionRecord:
        paper = self.active_review_paper
        self.active_review_effort += self.review_effort_delta()
        self.review_progress = self.active_review_effort

        if (
            self.review_paradigm == REVIEW_PARADIGM_DISCRETE
            and self.active_review_target_effort is not None
            and self.active_review_effort >= self.active_review_target_effort
        ):
            review_kind = self.active_review_kind or review_kind_from_effort(
                self.active_review_effort
            )
            completed = self._finalize_active_review(
                action_kind=review_action_kind(review_kind),
                review_kind=review_kind,
            )
            if completed is not None:
                return completed

        return ActionRecord(
            progress_kind,
            paper,
            review_effort=self.active_review_effort,
            review_kind=self.active_review_kind,
        )

    def _finalize_active_review(
        self,
        action_kind: str = "review_stopped",
        review_kind: str | None = None,
    ) -> ActionRecord | None:
        paper = self.active_review_paper
        if paper is None:
            return None

        effort = self.active_review_effort
        completed_review_kind = review_kind or self.active_review_kind
        if completed_review_kind is None:
            completed_review_kind = review_kind_from_effort(effort)
        share = paper.finish_review(self, effort, completed_review_kind)
        self._record_review_outcome(paper, share, effort, completed_review_kind)

        self.active_review_paper = None
        self.active_review_effort = 0.0
        self.active_review_kind = None
        self.active_review_target_effort = None
        self.review_progress = 0.0

        return ActionRecord(
            action_kind,
            paper,
            review_effort=effort,
            review_kind=completed_review_kind,
        )

    def _record_review_outcome(
        self,
        paper: Paper,
        share: float,
        effort: float,
        review_kind: str,
    ) -> None:
        """Update the agent's public peer-review history on completion."""
        self.completed_review_count += 1
        self.total_ac_from_reviews += share * paper.current_ac
        self.peer_review_history = (
            self.total_ac_from_reviews / self.completed_review_count
        )
        self.last_review_kind = review_kind
        if share > 0.0:
            self.last_review_effort = effort

    # ---- writing ---------------------------------------------------------
    def write_paper(self) -> float:
        """Progress the current paper; publish (and resample quality) at threshold."""
        if self.next_paper_quality is None:
            self.next_paper_quality = self._sample_quality()
        effort = self.writing_effort_delta()
        self.paper_progress += effort
        if self.paper_progress >= self.paper_completion_threshold():
            self.paper_progress = 0.0
            self.publish_paper()
            self.next_paper_quality = None
        return effort

    def publish_paper(self):
        """Create a new Paper (off-market; the env lists it next timestep)."""
        quality = (
            self.next_paper_quality
            if self.next_paper_quality is not None
            else self._sample_quality()
        )
        paper = Paper(author=self, quality=quality)
        Agent.all_papers.append(paper)

    def _sample_quality(self) -> float:
        return quality_multiplier(random.gauss(self.intrinsic_talent, QUALITY_SIGMA))

    def review_effort_delta(self) -> float:
        """Review effort contributed in one timestep."""
        return REVIEW_EFFORT_PER_TIMESTEP

    def writing_effort_delta(self) -> float:
        """Writing effort contributed in one timestep."""
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            return DISCRETE_WRITING_EFFORT_PER_TIMESTEP
        return self._clean_effort(random.random())

    def paper_completion_threshold(self) -> float:
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            return DISCRETE_PAPER_TIMESTEPS
        return PAPER_THRESHOLD

    def _clear_last_review_result(self):
        self.last_review_effort = None
        self.last_review_kind = None

    @staticmethod
    def _clean_effort(value: float) -> float:
        try:
            effort = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(effort) or math.isinf(effort) or effort < 0.0:
            return 0.0
        return effort
