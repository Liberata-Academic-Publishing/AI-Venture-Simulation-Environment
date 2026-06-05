from __future__ import annotations
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from Paper import Paper

PAPER_THRESHOLD = 10.0
GOOD_FAITH_REVIEW_DAYS = 4


@dataclass(frozen=True)
class ActionRecord:
    """
    A class to describe what an agent did on a single turn.
    """

    kind: str
    paper: Paper | None = None
    published: bool = False


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

    @abstractmethod
    def choose_action(self) -> tuple[str, Paper | None]:
        """Return (action, paper). Paper is None for write_paper, a Paper object for reviews."""

    def act(self) -> ActionRecord:
        """Called by the environment each step. Returns a record of what happened
        this turn so the environment can log it (the environment, not the agent,
        knows the current day and sees every agent)."""
        if self.active_review_paper is not None:
            paper = self.active_review_paper
            self.advance_active_review()
            kind = (
                "review_completed"
                if self.active_review_paper is None
                else "review_continued"
            )
            return ActionRecord(kind, paper)

        action, paper = self.choose_action()
        if action == "write_paper":
            papers_before = len(Agent.all_papers)
            self.write_paper()
            published = len(Agent.all_papers) > papers_before
            return ActionRecord("write_paper", published=published)
        if action == "peer_review":
            self.peer_review(paper)
            if self.active_review_paper is not None:
                return ActionRecord("review_started", paper)
            return ActionRecord("review_unavailable", paper)
        if action == "bad_faith_review":
            self.bad_faith_review(paper)
            return ActionRecord("bad_faith_review", paper)
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

    def peer_review(self, paper: Paper | None):
        """Start a good-faith review that takes four agent turns."""
        if paper is None:
            return

        if hasattr(paper, "start_review") and not paper.start_review(self):
            return

        self.active_review_paper = paper
        self.active_review_days_remaining = GOOD_FAITH_REVIEW_DAYS
        self.active_review_kind = "good_faith"
        self.advance_active_review()

    def advance_active_review(self):
        """Advance the active review by one turn and publish it when complete."""
        if self.active_review_paper is None:
            return

        self.active_review_days_remaining -= 1
        if self.active_review_days_remaining <= 0:
            paper = self.active_review_paper
            self.active_review_paper = None
            self.active_review_days_remaining = 0
            self.active_review_kind = None
            self.review_progress = 0.0
            self.publish_peer_review(paper)
            if hasattr(paper, "finish_review"):
                paper.finish_review(self)

    def publish_peer_review(self, paper: Paper):
        """Register share on the reviewed paper."""
        paper.add_share(self)

    def bad_faith_review(self, paper: Paper | None):
        """Complete a bad-faith review immediately."""
        if paper is None:
            return

        if hasattr(paper, "can_start_review") and not paper.can_start_review(self):
            return

        paper.add_bad_share(self)
