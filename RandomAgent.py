from __future__ import annotations
import random

from Agent import Agent
from Paper import Paper


class RandomAgent(Agent):

    def choose_action(self) -> tuple[str, Paper | None]:
        reviewable = [p for p in Agent.all_papers if p.author != self]

        # If there are no papers to review, fall back to writing
        if not reviewable:
            return "write_paper", None

        action = random.choice(["write_paper", "peer_review", "bad_faith_review"])

        if action == "write_paper":
            return "write_paper", None
        else:
            return action, random.choice(reviewable)
