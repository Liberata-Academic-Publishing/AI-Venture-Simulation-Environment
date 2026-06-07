from __future__ import annotations

from HeuristicAgent import HeuristicAgent


class BadFaithAgent(HeuristicAgent):
    """A HeuristicAgent that only targets minimum-effort reviews.

    It reuses all of HeuristicAgent's capital-forecast scoring but only ever
    weighs the ``bad_faith`` (minimum-effort) review target (see
    ``REVIEW_TARGETS``), so every review it starts is stopped as soon as it
    crosses the minimum effort threshold and is classified bad-faith.
    """

    REVIEW_TARGETS = ("bad_faith",)
