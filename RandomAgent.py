from __future__ import annotations
import random

from Agent import Agent
from Paper import Paper


class RandomAgent(Agent):

    def choose_action(self) -> tuple[str, Paper | None]:
        if self.should_offer_review_choice():
            reviewable = [p for p in Agent.all_papers if self._can_review(p)]
            options: list[tuple[str, Paper | None]] = [
                ("peer_review", self.active_review_paper),
                ("finish_review_write_paper", None),
            ]
            for paper in reviewable:
                options.append(("finish_review_peer_review", paper))
            return random.choice(options)

        reviewable = [p for p in Agent.all_papers if self._can_review(p)]
        if not reviewable:
            return "write_paper", None

        action = random.choice(["write_paper", "peer_review"])
        if action == "write_paper":
            return "write_paper", None
        return action, random.choice(reviewable)

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))

        return paper.author != self
