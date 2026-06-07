from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from Paper import (
    MIN_REVIEW_EFFORT_THRESHOLD,
    REVIEW_EFFORT_PER_DAY,
    Paper,
)

PAPER_THRESHOLD = 10.0
EXPECTED_REVIEW_EFFORT_PER_TURN = REVIEW_EFFORT_PER_DAY


@dataclass(frozen=True)
class ActionRecord:
    """
    A class to describe what an agent did on a single turn.
    """

    kind: str
    paper: Paper | None = None
    published: bool = False
    review_effort: float | None = None
    review_kind: str | None = None


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
        self.last_review_effort: float | None = None
        self.last_review_kind: str | None = None
        self._review_choice_pending = False

    @abstractmethod
    def choose_action(self) -> tuple[str, Paper | None]:
        """Return ``(action, paper)`` for this turn.

        Actions:
          - ``"write_paper"`` — make progress on a paper (``paper`` is ``None``).
          - ``"peer_review"`` — start or continue a review on ``paper``.
          - ``"finish_review_write_paper"`` — finalize the active review, then
            work on own paper the same day (``paper`` is ``None``).
          - ``"finish_review_peer_review"`` — finalize the active review, then
            start reviewing ``paper`` the same day.

        Locked agents (effort below the minimum threshold) are not called; the
        environment auto-continues their review via ``auto_continue_review``.
        """

    def is_locked_in_review(self) -> bool:
        """True when the agent is mid-review and not yet eligible to choose."""
        return (
            self.active_review_paper is not None
            and not self._review_choice_pending
        )

    def should_offer_review_choice(self) -> bool:
        """True when the agent may choose continue / finish+write / finish+review."""
        return (
            self.active_review_paper is not None
            and self._review_choice_pending
        )

    def available_actions(self) -> tuple[str, ...]:
        if self.should_offer_review_choice():
            return (
                "peer_review",
                "finish_review_write_paper",
                "finish_review_peer_review",
            )
        return ("write_paper", "peer_review")

    def auto_continue_review(self) -> ActionRecord:
        """Advance a locked in-progress review by one day of effort."""
        paper = self.active_review_paper
        if paper is None:
            return ActionRecord("idle")

        self.active_review_effort += self.review_effort_delta()
        self.review_progress = self.active_review_effort
        if self.active_review_effort >= MIN_REVIEW_EFFORT_THRESHOLD:
            self._review_choice_pending = True

        return ActionRecord(
            "review_auto_continued",
            paper,
            review_effort=self.active_review_effort,
        )

    def act(self) -> ActionRecord:
        """Called by the environment when the agent may choose an action."""
        self._clear_last_review_result()
        action, paper = self.choose_action()

        if action == "peer_review":
            record = self._peer_review_turn(paper)
            if (
                record.kind == "review_continued"
                and self.active_review_effort >= MIN_REVIEW_EFFORT_THRESHOLD
            ):
                self._review_choice_pending = True
            return record
        if action == "finish_review_write_paper":
            return self._finish_review_and_write()
        if action == "finish_review_peer_review":
            return self._finish_review_and_start(paper)
        if action == "write_paper":
            papers_before = len(Agent.all_papers)
            self.write_paper()
            published = len(Agent.all_papers) > papers_before
            return ActionRecord("write_paper", published=published)
        return ActionRecord("idle")

    def write_paper(self):
        """Increment paper progress randomly and publishes once threshold is reached."""
        self.paper_progress += random.random()
        if self.paper_progress >= PAPER_THRESHOLD:
            self.paper_progress = 0.0
            self.publish_paper()

    def publish_paper(self):
        """Create a new Paper and add it to the shared paper list."""
        paper = Paper(author=self)
        Agent.all_papers.append(paper)

    # ---- reviewing -------------------------------------------------------
    def _peer_review_turn(self, paper: Paper | None) -> ActionRecord:
        """Start a new review or continue the active one."""
        if self.active_review_paper is not None:
            self.active_review_effort += self.review_effort_delta()
            self.review_progress = self.active_review_effort
            return ActionRecord(
                "review_continued",
                self.active_review_paper,
                review_effort=self.active_review_effort,
            )

        if paper is None:
            return ActionRecord("review_unavailable")
        if hasattr(paper, "start_review") and not paper.start_review(self):
            return ActionRecord("review_unavailable", paper)

        self.active_review_paper = paper
        self.active_review_effort = self.review_effort_delta()
        self.review_progress = self.active_review_effort
        self._review_choice_pending = False
        return ActionRecord(
            "review_started",
            paper,
            review_effort=self.active_review_effort,
        )

    def _finish_review_and_write(self) -> ActionRecord:
        finished = self._finalize_active_review()
        if finished is None:
            papers_before = len(Agent.all_papers)
            self.write_paper()
            return ActionRecord(
                "write_paper",
                published=len(Agent.all_papers) > papers_before,
            )

        papers_before = len(Agent.all_papers)
        self.write_paper()
        return ActionRecord(
            "review_finished_write",
            finished.paper,
            published=len(Agent.all_papers) > papers_before,
            review_effort=finished.review_effort,
            review_kind=finished.review_kind,
        )

    def _finish_review_and_start(self, paper: Paper | None) -> ActionRecord:
        finished = self._finalize_active_review()
        if finished is None:
            return ActionRecord("idle")

        if paper is None:
            return finished

        if hasattr(paper, "start_review") and not paper.start_review(self):
            return ActionRecord("review_unavailable", paper)

        self.active_review_paper = paper
        self.active_review_effort = self.review_effort_delta()
        self.review_progress = self.active_review_effort
        self._review_choice_pending = False
        return ActionRecord(
            "review_finished_peer_review",
            finished.paper,
            review_effort=finished.review_effort,
            review_kind=finished.review_kind,
        )

    def _finalize_active_review(self) -> ActionRecord | None:
        paper = self.active_review_paper
        if paper is None:
            return None

        effort = self.active_review_effort
        self.publish_peer_review(paper, effort)
        if hasattr(paper, "finish_review"):
            paper.finish_review(self)

        self.active_review_paper = None
        self.active_review_effort = 0.0
        self.review_progress = 0.0
        self._review_choice_pending = False

        return ActionRecord(
            "review_stopped",
            paper,
            review_effort=effort,
            review_kind=self.last_review_kind,
        )

    def publish_peer_review(self, paper: Paper, effort: float | None = None):
        """Register share on the reviewed paper for the effort invested."""
        review_effort = (
            MIN_REVIEW_EFFORT_THRESHOLD
            if effort is None
            else self._clean_effort(effort)
        )
        if hasattr(paper, "add_review"):
            review_share = paper.add_review(self, review_effort)
        else:
            review_share = paper.add_share(self)
        if review_share > 0.0:
            self.last_review_effort = review_effort
            self.last_review_kind = None

    def review_effort_delta(self) -> float:
        """Review effort contributed in one agent turn."""
        return REVIEW_EFFORT_PER_DAY

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
