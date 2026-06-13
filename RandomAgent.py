from __future__ import annotations
import random

from Agent import Agent
from Paper import Paper


class RandomAgent(Agent):

    def choose_marketplace_action(self) -> Paper | None:
        reviewable = [
            p
            for p in Agent.all_papers
            if self._can_review(p) and p.offered_share(self) > 0.0
        ]
        if not reviewable:
            return None
        # Pass roughly half the time so the agent also does its own research.
        if random.random() < 0.5:
            return None
        return random.choice(reviewable)

    def choose_work_action(self) -> tuple[str, Paper | None]:
        if self.active_review_paper is not None:
            return random.choice(
                [
                    ("peer_review", self.active_review_paper),
                    ("finish_review_write_paper", None),
                ]
            )
        return "write_paper", None

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))
        return paper.author is not self
