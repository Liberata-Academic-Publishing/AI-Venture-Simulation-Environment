from __future__ import annotations

from Agent import Agent
from HeuristicAgent import HeuristicAgent
from Paper import Paper


class BadFaithAgent(HeuristicAgent):
    """Like HeuristicAgent, but only ever writes papers or does bad-faith reviews.

    It never performs a good-faith peer review, so it scores only write_paper vs
    bad_faith_review and picks the best-forecast option.
    """

    def choose_action(self) -> tuple[str, Paper | None]:
        reviewable = [p for p in Agent.all_papers if self._can_review(p)]
        if not reviewable:
            return "write_paper", None

        best_action = "write_paper"
        best_paper = None
        best_score = self._score_write()

        for paper in reviewable:
            bad_score = self._score_review(paper, "bad_faith")
            if bad_score > best_score:
                best_action = "bad_faith_review"
                best_paper = paper
                best_score = bad_score

        return best_action, best_paper
