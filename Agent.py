from __future__ import annotations
import random
from abc import ABC, abstractmethod

from Paper import Paper

PAPER_THRESHOLD = 10.0
GOOD_FAITH_REVIEW_DAYS = 4


class Agent(ABC):

    all_papers: list[Paper] = []  # class variable shared across all agents

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
    ):
        self.intrinsic_talent = intrinsic_talent
        self.academic_capital = academic_capital
        self.paper_progress = paper_progress
        self.review_progress = review_progress
        self.active_review_paper: Paper | None = None
        self.active_review_days_remaining = 0
        self.active_review_kind: str | None = None

    @abstractmethod
    def choose_action(self) -> tuple[str, Paper | None]:
        """Return (action, paper). Paper is None for write_paper, a Paper object for reviews."""

    def act(self):
        """Called by the environment each step."""
        if self.active_review_paper is not None:
            self.advance_active_review()
            return

        action, paper = self.choose_action()
        if action == "write_paper":
            self.write_paper()
        elif action == "peer_review":
            self.peer_review(paper)
        elif action == "bad_faith_review":
            self.bad_faith_review(paper)

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
