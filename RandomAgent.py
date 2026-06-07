from __future__ import annotations
import random

from Agent import Agent
from Paper import Paper


class RandomAgent(Agent):

    def choose_action(self) -> tuple[str, Paper | None]:
        # Mid-review: randomly keep reviewing or stop and collect.
        if self.active_review_paper is not None:
            action = random.choice(["peer_review", "stop_peer_review"])
            return action, self.active_review_paper

        reviewable = [p for p in Agent.all_papers if self._can_review(p)]

        # If there are no papers to review, fall back to writing
        if not reviewable:
            return "write_paper", None

        action = random.choice(["write_paper", "peer_review"])

        if action == "write_paper":
            return "write_paper", None
        else:
            return action, random.choice(reviewable)

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))

        return paper.author != self
