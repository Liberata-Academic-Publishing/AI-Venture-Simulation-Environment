from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from Paper import (
    GOOD_FAITH_REVIEW_EFFORT_THRESHOLD,
    MIN_REVIEW_EFFORT_THRESHOLD,
    Paper,
)

PAPER_THRESHOLD = 10.0
GOOD_FAITH_REVIEW_DAYS = 4
EXPECTED_REVIEW_EFFORT_PER_TURN = 0.5


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
        self.active_review_days_remaining = 0
        self.active_review_kind: str | None = None
        self.active_review_effort = 0.0
        self.active_review_target_effort = 0.0
        self.last_review_effort: float | None = None
        self.last_review_kind: str | None = None

    @abstractmethod
    def choose_action(self) -> tuple[str, Paper | None]:
        """Return (action, paper). Paper is None for write_paper, a Paper object for reviews."""

    def act(self) -> ActionRecord:
        """Called by the environment each step. Returns a record of what happened
        this turn so the environment can log it (the environment, not the agent,
        knows the current day and sees every agent)."""
        self._clear_last_review_result()
        if self.active_review_paper is not None:
            paper = self.active_review_paper
            self.advance_active_review()
            if self.active_review_paper is None:
                return ActionRecord(
                    "review_completed",
                    paper,
                    review_effort=self.last_review_effort,
                    review_kind=self.last_review_kind,
                )
            return ActionRecord(
                "review_continued",
                paper,
                review_effort=self.active_review_effort,
            )

        action, paper = self.choose_action()
        if action == "write_paper":
            papers_before = len(Agent.all_papers)
            self.write_paper()
            published = len(Agent.all_papers) > papers_before
            return ActionRecord("write_paper", published=published)
        if action == "peer_review":
            self.peer_review(paper)
            if self.last_review_effort is not None:
                return ActionRecord(
                    "review_completed",
                    paper,
                    review_effort=self.last_review_effort,
                    review_kind=self.last_review_kind,
                )
            if self.active_review_paper is not None:
                return ActionRecord(
                    "review_started",
                    paper,
                    review_effort=self.active_review_effort,
                )
            return ActionRecord("review_unavailable", paper)
        if action == "bad_faith_review":
            self.bad_faith_review(paper)
            if self.last_review_effort is not None:
                return ActionRecord(
                    "bad_faith_review",
                    paper,
                    review_effort=self.last_review_effort,
                    review_kind=self.last_review_kind,
                )
            if self.active_review_paper is not None:
                return ActionRecord(
                    "bad_faith_review",
                    paper,
                    review_effort=self.active_review_effort,
                )
            return ActionRecord("review_unavailable", paper)
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

    def peer_review(
        self,
        paper: Paper | None,
        target_effort: float | None = None,
    ):
        """Start a peer review with an effort target."""
        self.start_peer_review(
            paper,
            GOOD_FAITH_REVIEW_EFFORT_THRESHOLD
            if target_effort is None
            else target_effort,
        )

    def start_peer_review(self, paper: Paper | None, target_effort: float):
        """Begin a review episode; final good/bad faith is effort-thresholded."""
        if paper is None:
            return

        if hasattr(paper, "start_review") and not paper.start_review(self):
            return

        review_target = self._clean_effort(target_effort)
        if review_target <= 0.0:
            review_target = MIN_REVIEW_EFFORT_THRESHOLD

        self.active_review_paper = paper
        self.active_review_effort = 0.0
        self.active_review_target_effort = review_target
        self.active_review_days_remaining = self._estimated_review_days_remaining()
        self.active_review_kind = None
        self.review_progress = 0.0
        self.advance_active_review()

    def advance_active_review(self):
        """Advance the active review by one effort increment and publish it when complete."""
        if self.active_review_paper is None:
            return

        self.active_review_effort += self.review_effort_delta()
        self.review_progress = self.active_review_effort

        if self.active_review_effort >= self.active_review_target_effort:
            paper = self.active_review_paper
            review_effort = self.active_review_effort
            self.active_review_paper = None
            self.active_review_days_remaining = 0
            self.active_review_kind = None
            self.active_review_effort = 0.0
            self.active_review_target_effort = 0.0
            self.review_progress = 0.0
            self.publish_peer_review(paper, review_effort)
            if hasattr(paper, "finish_review"):
                paper.finish_review(self)
        else:
            self.active_review_days_remaining = self._estimated_review_days_remaining()

    def publish_peer_review(self, paper: Paper, effort: float | None = None):
        """Register share on the reviewed paper."""
        review_effort = (
            GOOD_FAITH_REVIEW_EFFORT_THRESHOLD
            if effort is None
            else self._clean_effort(effort)
        )
        review_kind = (
            paper.classify_review_effort(review_effort)
            if hasattr(paper, "classify_review_effort")
            else "good_faith"
        )
        if hasattr(paper, "add_review"):
            review_share = paper.add_review(self, review_effort)
        else:
            review_share = paper.add_share(self)
        if review_share > 0.0:
            self.last_review_effort = review_effort
            self.last_review_kind = review_kind

    def bad_faith_review(self, paper: Paper | None):
        """Compatibility entry point for a minimum-effort review target."""
        self.peer_review(paper, target_effort=MIN_REVIEW_EFFORT_THRESHOLD)

    def review_effort_delta(self) -> float:
        """Continuous review effort contributed in one agent turn."""
        return self._clean_effort(random.random())

    def _estimated_review_days_remaining(self) -> int:
        remaining_effort = max(
            0.0,
            self.active_review_target_effort - self.active_review_effort,
        )
        if remaining_effort <= 0.0:
            return 0
        return max(1, math.ceil(remaining_effort / EXPECTED_REVIEW_EFFORT_PER_TURN))

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
