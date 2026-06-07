from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from Paper import (
    GOOD_FAITH_REVIEW_EFFORT_THRESHOLD,
    Paper,
)

PAPER_THRESHOLD = 10.0
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
        self.active_review_effort = 0.0
        self.last_review_effort: float | None = None
        self.last_review_kind: str | None = None

    @abstractmethod
    def choose_action(self) -> tuple[str, Paper | None]:
        """Return ``(action, paper)`` for this turn.

        Actions:
          - ``"write_paper"`` — make progress on a paper (``paper`` is ``None``).
          - ``"peer_review"`` — spend one turn reviewing. Starts a review on
            ``paper`` if none is active; otherwise continues the active review
            (the ``paper`` argument is ignored).
          - ``"stop_peer_review"`` — finalize the active review and collect the
            reviewer share + accrual bump. The review is classified good-/bad-faith
            from the total effort invested (``paper`` is ignored).

        See ``available_actions`` for which of these are legal given the agent's
        current review state.
        """

    def available_actions(self) -> tuple[str, ...]:
        """Legal actions given the agent's current review state.

        While a review is in progress the agent is committed to that paper and
        may only continue it or stop; otherwise it may write or start a review.
        """
        if self.active_review_paper is not None:
            return ("peer_review", "stop_peer_review")
        return ("write_paper", "peer_review")

    def act(self) -> ActionRecord:
        """Called by the environment each step. Returns a record of what happened
        this turn so the environment can log it (the environment, not the agent,
        knows the current day and sees every agent)."""
        self._clear_last_review_result()
        action, paper = self.choose_action()

        if action == "peer_review":
            return self._peer_review_turn(paper)
        if action == "stop_peer_review":
            return self._stop_peer_review()
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
        """Spend one turn reviewing: start a new review or continue the active one.

        Effort accumulates one ``review_effort_delta`` per turn; nothing is
        granted until the agent chooses ``stop_peer_review``.
        """
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
        return ActionRecord(
            "review_started",
            paper,
            review_effort=self.active_review_effort,
        )

    def _stop_peer_review(self) -> ActionRecord:
        """Finalize the active review and collect its reward (no-op when idle).

        The accumulated effort decides whether the completed review counts as
        good- or bad-faith; too little effort grants nothing (see
        ``Paper.add_review``).
        """
        paper = self.active_review_paper
        if paper is None:
            return ActionRecord("idle")

        effort = self.active_review_effort
        self.publish_peer_review(paper, effort)
        if hasattr(paper, "finish_review"):
            paper.finish_review(self)

        self.active_review_paper = None
        self.active_review_effort = 0.0
        self.review_progress = 0.0
        return ActionRecord(
            "review_stopped",
            paper,
            review_effort=effort,
            review_kind=self.last_review_kind,
        )

    def publish_peer_review(self, paper: Paper, effort: float | None = None):
        """Register share on the reviewed paper for the effort invested."""
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

    def review_effort_delta(self) -> float:
        """Continuous review effort contributed in one agent turn."""
        return self._clean_effort(random.random())

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
