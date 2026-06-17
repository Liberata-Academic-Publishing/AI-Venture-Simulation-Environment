from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import SIM
from Paper import (
    DISCRETE_WRITING_EFFORT_PER_TIMESTEP,
    GOOD_FAITH_REVIEW,
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
WRITING_EFFORT_PER_TIMESTEP = SIM.writing_effort_per_timestep
PAPER_EFFORT_MODE_FIXED = "fixed"
PAPER_EFFORT_MODE_UNIFORM = "uniform"
PAPER_EFFORT_MODE_QUALITY_SCALED = "quality_scaled"
VALID_PAPER_EFFORT_MODES = frozenset({
    PAPER_EFFORT_MODE_FIXED,
    PAPER_EFFORT_MODE_UNIFORM,
    PAPER_EFFORT_MODE_QUALITY_SCALED,
})
CONTINUOUS_PUBLISHING_CHOICE = "choice"
CONTINUOUS_PUBLISHING_THRESHOLD = "threshold"
VALID_CONTINUOUS_PUBLISHING = frozenset({
    CONTINUOUS_PUBLISHING_CHOICE,
    CONTINUOUS_PUBLISHING_THRESHOLD,
})

# Continuous-mode merged decision: one of these per agent per timestep.
CONTINUOUS_CLAIM = "claim"
CONTINUOUS_REVIEW = "review"
CONTINUOUS_RESEARCH = "research"
CONTINUOUS_RESEARCH_FINISH = "research_finish"


def validate_continuous_publishing(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in VALID_CONTINUOUS_PUBLISHING:
        allowed = ", ".join(sorted(VALID_CONTINUOUS_PUBLISHING))
        raise ValueError(f"continuous_publishing must be one of: {allowed}")
    return value


def validate_paper_effort_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in VALID_PAPER_EFFORT_MODES:
        allowed = ", ".join(sorted(VALID_PAPER_EFFORT_MODES))
        raise ValueError(f"paper_effort_mode must be one of: {allowed}")
    return value


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
    all_agents: list[Agent] = []  # live roster for cross-agent signals (e.g. RL rank)

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
        self.continuous_publishing = validate_continuous_publishing(
            SIM.continuous_publishing
        )
        self.continuous_paper_timesteps = float(SIM.continuous_paper_timesteps)
        self.discrete_paper_timesteps = float(SIM.discrete_paper_timesteps)
        self.paper_effort_mode = validate_paper_effort_mode(SIM.paper_effort_mode)
        self.paper_effort_min = float(SIM.paper_effort_min)
        self.paper_effort_max = float(SIM.paper_effort_max)

        # Quality of the paper currently being written (known before/while
        # working on it). Sampled lazily the first time the agent writes.
        self.next_paper_quality: float | None = None
        self.next_paper_required_effort: float | None = None

        # Public peer-review reputation: mean AC earned per completed review.
        self.peer_review_history: float = 0.0
        self.total_ac_from_reviews: float = 0.0
        self.peer_review_epsilon_history: float = SIM.prior_review_epsilon
        self.total_review_epsilon: float = 0.0
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

    def choose_continuous_action(self) -> tuple[str, Paper | None]:
        """Continuous merged phase: one decision that also applies a timestep.

        Returns ``(kind, paper)`` where ``kind`` is one of:

        Without an active review:
          - ``"claim"`` (with a listed ``paper``) — start reviewing it (+1 effort).
          - ``"research"`` — add one timestep of writing effort to the own paper.
          - ``"research_finish"`` — add one writing timestep, then finish the paper
            (listed next timestep) and start a fresh one.

        With an active review:
          - ``"claim"`` (with a listed ``paper``) — finalize the current review and
            start the new one (+1 effort).
          - ``"review"`` — add one timestep of effort to the current review.
          - ``"research"`` — finalize the current review, then add one writing
            timestep to the own paper.

        Default derives from the two-phase hooks; concrete agents override this.
        """
        paper = self.choose_marketplace_action()
        if paper is not None and paper.can_start_review(self):
            return (CONTINUOUS_CLAIM, paper)
        if self.active_review_paper is not None:
            action, _ = self.choose_work_action()
            if action == "finish_review_write_paper":
                return (CONTINUOUS_RESEARCH, None)
            return (CONTINUOUS_REVIEW, None)
        return (CONTINUOUS_RESEARCH, None)

    def choose_review_kind(self, paper: Paper) -> str:
        """Discrete-mode review choice. Subclasses can choose bad vs good faith."""
        return GOOD_FAITH_REVIEW

    def configure_review_paradigm(self, review_paradigm: str) -> None:
        self.review_paradigm = validate_review_paradigm(review_paradigm)

    def configure_continuous_publishing(
        self, mode: str, timesteps: float | None = None
    ) -> None:
        self.continuous_publishing = validate_continuous_publishing(mode)
        if timesteps is not None:
            self.continuous_paper_timesteps = max(0.0, float(timesteps))

    def configure_paper_effort(
        self,
        mode: str | None = None,
        min_effort: float | None = None,
        max_effort: float | None = None,
        *,
        discrete_timesteps: float | None = None,
    ) -> None:
        """Configure manuscript effort targets for this run.

        ``fixed`` preserves the existing thresholds. ``uniform`` and
        ``quality_scaled`` sample one target per paper, which matches the
        50-150 timestep range discussed in sync without changing agent APIs.
        """
        if mode is not None:
            self.paper_effort_mode = validate_paper_effort_mode(mode)
        if min_effort is not None:
            self.paper_effort_min = max(0.0, float(min_effort))
        if max_effort is not None:
            self.paper_effort_max = max(0.0, float(max_effort))
        if self.paper_effort_max < self.paper_effort_min:
            self.paper_effort_min, self.paper_effort_max = (
                self.paper_effort_max,
                self.paper_effort_min,
            )
        if discrete_timesteps is not None:
            self.discrete_paper_timesteps = max(0.0, float(discrete_timesteps))
        # Force a clean draw for the next manuscript under the new run config.
        if self.paper_progress == 0.0:
            self.next_paper_required_effort = None

    def continuous_publish_by_threshold(self) -> bool:
        """True when continuous mode auto-publishes at a fixed writing effort."""
        return (
            self.review_paradigm == REVIEW_PARADIGM_CONTINUOUS
            and self.continuous_publishing == CONTINUOUS_PUBLISHING_THRESHOLD
        )

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

    # ---- continuous merged phase (one decision + one timestep of effort) --
    def act_continuous(self) -> list[ActionRecord]:
        """Run one merged continuous-mode decision and apply its timestep.

        Returns the action record(s) produced (a claim that finalizes an active
        review yields two: the finished review and the newly started one).
        """
        self._clear_last_review_result()
        kind, paper = self.choose_continuous_action()

        if kind == CONTINUOUS_CLAIM and paper is not None and paper.can_start_review(self):
            records: list[ActionRecord] = []
            finalized = self.claim_review(paper)
            if finalized is not None:
                records.append(finalized)
            records.append(self._advance_active_review("review_started"))
            return records

        if kind == CONTINUOUS_REVIEW and self.active_review_paper is not None:
            return [self._advance_active_review("review_continued")]

        finish = (
            kind == CONTINUOUS_RESEARCH_FINISH
            and self.active_review_paper is None
            and not self.continuous_publish_by_threshold()
        )
        return [self._research_turn(finish=finish)]

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

    def _research_turn(self, finish: bool) -> ActionRecord:
        """Continuous research: optionally finalize an active review, then add one
        writing timestep; when ``finish`` is set, publish the paper and start a
        fresh one."""
        finished = None
        if self.active_review_paper is not None:
            finished = self._finalize_active_review()

        effort = self.add_research_effort()
        published = False
        if finish:
            self.finish_research_paper()
            published = True
        elif self._auto_publish_if_threshold_reached():
            published = True

        if finished is not None:
            return ActionRecord(
                "review_finished_write",
                finished.paper,
                published=published,
                review_effort=finished.review_effort,
                review_kind=finished.review_kind,
                writing_effort=effort,
            )
        return ActionRecord(
            "write_paper",
            published=published,
            writing_effort=effort,
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
        """Update public peer-review reputation and epsilon history on completion."""
        self.completed_review_count += 1
        self.total_ac_from_reviews += share * paper.current_ac
        self.peer_review_history = (
            self.total_ac_from_reviews / self.completed_review_count
        )
        epsilon = 0.0
        if getattr(paper, "review_records", None):
            try:
                epsilon = float(paper.review_records[-1].get("epsilon", 0.0))
            except (TypeError, ValueError):
                epsilon = 0.0
        self.total_review_epsilon += max(0.0, epsilon)
        self.peer_review_epsilon_history = (
            self.total_review_epsilon / self.completed_review_count
        )
        self.last_review_kind = review_kind
        if share > 0.0:
            self.last_review_effort = effort

    # ---- writing ---------------------------------------------------------
    def write_paper(self) -> float:
        """Progress the current paper; publish (and resample quality) at threshold."""
        self._ensure_next_paper_state()
        effort = self.writing_effort_delta()
        self.paper_progress += effort
        if self.paper_progress >= self.paper_completion_threshold():
            self.publish_paper()
        return effort

    def publish_paper(self) -> Paper:
        """Create a new Paper (off-market; the env lists it next timestep)."""
        self._ensure_next_paper_state()
        paper = Paper(
            author=self,
            quality=self.next_paper_quality or self._sample_quality(),
            writing_effort=self.paper_progress,
            required_writing_effort=self.next_paper_required_effort,
        )
        Agent.all_papers.append(paper)
        self._reset_next_paper_state()
        return paper

    # ---- continuous writing (agent-chosen finish, asymptotic accrual) ----
    def add_research_effort(self) -> float:
        """Add one timestep of writing effort to the in-progress paper (no publish).

        Quality is sampled lazily the first time the agent researches a paper.
        """
        self._ensure_next_paper_state()
        effort = self.writing_effort_delta()
        self.paper_progress += effort
        return effort

    def _auto_publish_if_threshold_reached(self) -> bool:
        if (
            self.continuous_publish_by_threshold()
            and self.paper_progress >= self.paper_completion_threshold()
        ):
            self.finish_research_paper()
            return True
        return False

    def finish_research_paper(self) -> Paper:
        """Publish the in-progress paper using its accumulated writing effort.

        The paper's base accrual rate is set from how much writing effort went
        in (asymptotically approaching its quality ceiling). Resets progress and
        resamples a fresh quality for the agent's next paper.
        """
        self._ensure_next_paper_state()
        paper = Paper(
            author=self,
            quality=self.next_paper_quality or self._sample_quality(),
            writing_effort=self.paper_progress,
            required_writing_effort=self.next_paper_required_effort,
        )
        Agent.all_papers.append(paper)
        self._reset_next_paper_state()
        return paper

    def _ensure_next_paper_state(self) -> None:
        if self.next_paper_quality is None:
            self.next_paper_quality = self._sample_quality()
        if self.next_paper_required_effort is None:
            self.next_paper_required_effort = self._sample_required_writing_effort(
                self.next_paper_quality
            )

    def _reset_next_paper_state(self) -> None:
        self.paper_progress = 0.0
        self.next_paper_quality = None
        self.next_paper_required_effort = None

    def _sample_required_writing_effort(self, quality: float) -> float:
        if self.paper_effort_mode == PAPER_EFFORT_MODE_FIXED:
            if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
                return self.discrete_paper_timesteps
            if self.continuous_publish_by_threshold():
                return self.continuous_paper_timesteps
            return PAPER_THRESHOLD

        lo = min(self.paper_effort_min, self.paper_effort_max)
        hi = max(self.paper_effort_min, self.paper_effort_max)
        if hi <= lo:
            return lo
        if self.paper_effort_mode == PAPER_EFFORT_MODE_UNIFORM:
            return random.uniform(lo, hi)

        # Quality-scaled mode: higher sampled paper quality requires a longer
        # manuscript target inside the configured range. The logistic map avoids
        # making low/high-quality papers collapse exactly to the endpoints.
        scaled = 1.0 / (1.0 + math.exp(-3.0 * (quality_multiplier(quality) - 1.0)))
        return lo + (hi - lo) * scaled

    def _sample_quality(self) -> float:
        return quality_multiplier(random.gauss(self.intrinsic_talent, QUALITY_SIGMA))

    def review_effort_delta(self) -> float:
        """Review effort contributed in one timestep."""
        return REVIEW_EFFORT_PER_TIMESTEP

    def writing_effort_delta(self) -> float:
        """Writing effort contributed in one timestep."""
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            return DISCRETE_WRITING_EFFORT_PER_TIMESTEP
        return WRITING_EFFORT_PER_TIMESTEP

    def paper_completion_threshold(self) -> float:
        self._ensure_next_paper_state()
        if self.next_paper_required_effort is not None:
            return self.next_paper_required_effort
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            return self.discrete_paper_timesteps
        if self.continuous_publish_by_threshold():
            return self.continuous_paper_timesteps
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
