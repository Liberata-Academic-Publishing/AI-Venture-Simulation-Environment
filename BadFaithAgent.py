from __future__ import annotations

from HeuristicAgent import HeuristicAgent


class BadFaithAgent(HeuristicAgent):
    """A HeuristicAgent that never gives good-faith reviews.

    It reuses all of HeuristicAgent's capital-forecast scoring but only ever
    considers writing a paper or a bad-faith review (see ``REVIEW_ACTIONS``),
    so it never performs a good-faith ``peer_review``.
    """

    REVIEW_ACTIONS = (("bad_faith", "bad_faith_review"),)
