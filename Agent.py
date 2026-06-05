from __future__ import annotations
import random
from abc import ABC, abstractmethod

from Paper import Paper

PAPER_THRESHOLD = 10.0
REVIEW_THRESHOLD = 1.0


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

    @abstractmethod
    def choose_action(self) -> tuple[str, Paper | None]:
        """Return (action, paper). Paper is None for write_paper, a Paper object for reviews."""

    def act(self):
        """Called by the environment each step."""
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

    def peer_review(self, paper: Paper):
        """Increment review progress; calls publish_peer_review once threshold is reached."""
        self.review_progress += random.random()
        if self.review_progress >= REVIEW_THRESHOLD:
            self.review_progress = 0.0
            self.publish_peer_review(paper)

    def publish_peer_review(self, paper: Paper):
        """Register share on the reviewed paper."""
        paper.add_share(self)

    def bad_faith_review(self, paper: Paper):
        """Complete a bad-faith review immediately."""
        paper.add_bad_share(self)
