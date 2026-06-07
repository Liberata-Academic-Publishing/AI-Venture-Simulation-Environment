from __future__ import annotations

from HeuristicAgent import HeuristicAgent


class BadFaithAgent(HeuristicAgent):
    """A HeuristicAgent that only targets minimum-effort reviews.

    It reuses all of HeuristicAgent's capital-forecast scoring but only ever
    considers writing a paper or a minimum-effort review (see
    ``REVIEW_ACTIONS``). The completed review is still classified by effort
    threshold inside ``Paper``.
    """

    REVIEW_ACTIONS = (("bad_faith", "bad_faith_review"),)
