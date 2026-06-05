from __future__ import annotations
import random

from Agent import Agent
from Paper import Paper


class RandomAgent(Agent):

    def choose_action(self) -> tuple[str, Paper | None]:
        reviewable = [p for p in Agent.all_papers if self._can_review(p)]

        # If there are no papers to review, fall back to writing
        if not reviewable:
            return "write_paper", None

        action = random.choice(["write_paper", "peer_review", "bad_faith_review"])

        if action == "write_paper":
            return "write_paper", None
        else:
            return action, random.choice(reviewable)

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))

        return paper.author != self
