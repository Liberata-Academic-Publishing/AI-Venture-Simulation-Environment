from __future__ import annotations

import csv
import os
import tempfile
import unittest

from Agent import Agent
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History, gini
from Paper import (
    MIN_REVIEW_EFFORT_THRESHOLD,
    REVIEW_EFFORT_PER_DAY,
    Paper,
    review_accrual_bump,
)

try:
    import matplotlib  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class DummyAgent(Agent):
    def __init__(self, name: str = "agent", actions=None):
        super().__init__(intrinsic_talent=1.0)
        self.name = name
        self.actions = list(actions or [])

    def choose_action(self):
        if self.actions:
            return self.actions.pop(0)
        return "write_paper", None


class RecordingAgent(DummyAgent):
    def __init__(self, log: list[str], name: str):
        super().__init__(name=name)
        self.log = log

    def act(self):
        self.log.append(self.name)
        return super().act()


class SimulationTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_agentact_calls_agents_in_order(self):
        log: list[str] = []
        agents = [RecordingAgent(log, "first"), RecordingAgent(log, "second")]
        env = Environment(agents=agents)

        env.agentact()

        self.assertEqual(log, ["first", "second"])

    def test_nextstep_advances_papers_and_recomputes_agent_capital(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(
            author=author,
            current_ac=10.0,
            accrual_rate=2.0,
            share_distribution={author: 0.75, reviewer: 0.25},
        )
        env = Environment(agents=[author, reviewer], papers=[paper])

        env.nextstep()

        self.assertEqual(env.day, 1)
        self.assertEqual(paper.current_ac, 12.0)
        self.assertEqual(author.academic_capital, 9.0)
        self.assertEqual(reviewer.academic_capital, 3.0)

    def _advance_to_review_choice(self, reviewer: DummyAgent) -> None:
        """Auto-continue until effort reaches the minimum threshold."""
        while reviewer.is_locked_in_review():
            reviewer.auto_continue_review()
        self.assertTrue(reviewer.should_offer_review_choice())
        self.assertEqual(reviewer.active_review_effort, MIN_REVIEW_EFFORT_THRESHOLD)

    def test_review_locks_paper_and_grants_share_on_finish(self):
        author = DummyAgent("author")
        paper = Paper(author=author, current_ac=100.0)
        reviewer = DummyAgent(
            "reviewer",
            actions=[
                ("peer_review", paper),
                ("finish_review_write_paper", None),
            ],
        )

        started = reviewer.act()
        self.assertEqual(started.kind, "review_started")
        self.assertEqual(reviewer.active_review_effort, REVIEW_EFFORT_PER_DAY)
        self.assertFalse(paper.review_available)
        self.assertIs(paper.review_in_progress_by, reviewer)

        self._advance_to_review_choice(reviewer)
        finished = reviewer.act()

        self.assertEqual(finished.kind, "review_finished_write")
        self.assertIsNone(reviewer.active_review_paper)
        self.assertIsNone(paper.review_in_progress_by)
        self.assertTrue(paper.review_available)
        self.assertEqual(paper.share_distribution[reviewer], 0.01)
        self.assertEqual(paper.completed_peer_reviews, 1)

    def test_review_lock_blocks_other_reviewers(self):
        author = DummyAgent("author")
        paper = Paper(author=author)
        first = DummyAgent("first", actions=[("peer_review", paper)])
        second = DummyAgent("second", actions=[("peer_review", paper)])

        first.act()
        unavailable = second.act()

        self.assertEqual(unavailable.kind, "review_unavailable")
        self.assertNotIn(second, paper.share_distribution)
        self.assertIsNone(second.active_review_paper)
        self.assertIs(paper.review_in_progress_by, first)
        self.assertFalse(paper.review_available)

    def test_locked_agent_auto_continues_via_environment(self):
        author = DummyAgent("author")
        paper = Paper(author=author)
        reviewer = DummyAgent("reviewer", actions=[("peer_review", paper)])
        history = History()
        env = Environment(agents=[reviewer], history=history)

        reviewer.act()
        self.assertTrue(reviewer.is_locked_in_review())
        env.agentact()

        self.assertEqual(reviewer.active_review_effort, 2.0)
        self.assertEqual(history.actions[-1][2], "review_auto_continued")

    def test_choice_offered_day_after_threshold(self):
        author = DummyAgent("author")
        paper = Paper(author=author)
        reviewer = DummyAgent("reviewer", actions=[("peer_review", paper)])

        reviewer.act()
        self.assertEqual(reviewer.active_review_effort, 1.0)
        for expected in range(2, int(MIN_REVIEW_EFFORT_THRESHOLD)):
            self.assertTrue(reviewer.is_locked_in_review())
            reviewer.auto_continue_review()
            self.assertEqual(reviewer.active_review_effort, float(expected))

        self.assertTrue(reviewer.is_locked_in_review())
        reviewer.auto_continue_review()
        self.assertEqual(reviewer.active_review_effort, MIN_REVIEW_EFFORT_THRESHOLD)
        self.assertFalse(reviewer.is_locked_in_review())
        self.assertTrue(reviewer.should_offer_review_choice())

    def test_minimum_effort_review_completes(self):
        author = DummyAgent("author")
        paper = Paper(author=author)
        reviewer = DummyAgent(
            "reviewer",
            actions=[
                ("peer_review", paper),
                ("finish_review_write_paper", None),
            ],
        )

        reviewer.act()
        self._advance_to_review_choice(reviewer)
        finished = reviewer.act()

        self.assertEqual(finished.kind, "review_finished_write")
        self.assertIsNone(finished.review_kind)
        self.assertEqual(finished.review_effort, MIN_REVIEW_EFFORT_THRESHOLD)
        self.assertEqual(paper.completed_peer_reviews, 1)

    def test_review_below_minimum_effort_does_not_complete(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(author=author)

        review_share = paper.add_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD - 0.01)

        self.assertEqual(review_share, 0.0)
        self.assertNotIn(reviewer, paper.share_distribution)
        self.assertEqual(paper.completed_peer_reviews, 0)

    def test_review_accrual_bump_increases_with_diminishing_marginals(self):
        bump_10 = review_accrual_bump(10)
        bump_11 = review_accrual_bump(11)
        bump_12 = review_accrual_bump(12)
        bump_13 = review_accrual_bump(13)

        self.assertGreater(bump_11, bump_10)
        self.assertGreater(bump_12, bump_11)
        self.assertGreater(bump_13, bump_12)
        self.assertGreater(bump_11 - bump_10, bump_12 - bump_11)
        self.assertGreater(bump_12 - bump_11, bump_13 - bump_12)
        self.assertEqual(review_accrual_bump(9), 0.0)

    def test_higher_effort_yields_higher_paper_accrual_rate(self):
        author = DummyAgent("author")
        low_reviewer = DummyAgent("low reviewer")
        high_reviewer = DummyAgent("high reviewer")
        low_paper = Paper(author=author, accrual_rate=1.0)
        high_paper = Paper(author=author, accrual_rate=1.0)

        low_paper.add_review(low_reviewer, MIN_REVIEW_EFFORT_THRESHOLD)
        high_paper.add_review(high_reviewer, MIN_REVIEW_EFFORT_THRESHOLD + 3)

        self.assertGreater(high_paper.accrual_rate, low_paper.accrual_rate)

    def test_agent_cannot_review_same_paper_twice(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(author=author)

        first_share = paper.add_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD)
        rate_after_first_review = paper.accrual_rate
        second_share = paper.add_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD + 2)

        self.assertGreater(first_share, 0.0)
        self.assertEqual(second_share, 0.0)
        self.assertEqual(paper.completed_peer_reviews, 1)
        self.assertEqual(paper.share_distribution[reviewer], first_share)
        self.assertEqual(paper.accrual_rate, rate_after_first_review)

    def test_review_share_and_accrual_gain_decay_logarithmically(self):
        author = DummyAgent("author")
        first_reviewer = DummyAgent("first reviewer")
        second_reviewer = DummyAgent("second reviewer")
        paper = Paper(author=author, accrual_rate=1.0)

        first_share = paper.add_share(first_reviewer)
        first_rate_gain = paper.accrual_rate - 1.0
        rate_after_first = paper.accrual_rate
        second_share = paper.add_share(second_reviewer)
        second_rate_gain = paper.accrual_rate - rate_after_first

        self.assertEqual(first_share, 0.01)
        self.assertLess(second_share, first_share)
        self.assertLess(second_rate_gain, first_rate_gain)

    def test_constructor_supports_generated_agents(self):
        env = Environment(num_agents=2, agent_cls=HeuristicAgent)

        self.assertEqual(len(env.agents), 2)
        self.assertTrue(all(isinstance(agent, HeuristicAgent) for agent in env.agents))

    def test_heuristic_writes_when_no_papers_are_reviewable(self):
        agent = HeuristicAgent(intrinsic_talent=1.0)

        self.assertEqual(agent.choose_action(), ("write_paper", None))

    def test_heuristic_selects_highest_value_reviewable_paper(self):
        author = DummyAgent("author")
        reviewer = HeuristicAgent(intrinsic_talent=1.0)
        low_value = Paper(author=author, current_ac=10.0, accrual_rate=1.0)
        high_value = Paper(author=author, current_ac=200.0, accrual_rate=1.0)
        Agent.all_papers = [low_value, high_value]

        action, paper = reviewer.choose_action()

        self.assertEqual(action, "peer_review")
        self.assertIs(paper, high_value)

    def test_finish_review_and_start_new_review_same_day(self):
        author = DummyAgent("author")
        paper_a = Paper(author=author, current_ac=50.0)
        paper_b = Paper(author=author, current_ac=100.0)
        reviewer = DummyAgent(
            "reviewer",
            actions=[
                ("peer_review", paper_a),
                ("finish_review_peer_review", paper_b),
            ],
        )
        Agent.all_papers = [paper_a, paper_b]

        reviewer.act()
        self._advance_to_review_choice(reviewer)
        record = reviewer.act()

        self.assertEqual(record.kind, "review_finished_peer_review")
        self.assertEqual(record.review_effort, MIN_REVIEW_EFFORT_THRESHOLD)
        self.assertIs(record.paper, paper_a)
        self.assertIs(reviewer.active_review_paper, paper_b)
        self.assertEqual(reviewer.active_review_effort, REVIEW_EFFORT_PER_DAY)
        self.assertTrue(paper_a.review_available)
        self.assertFalse(paper_b.review_available)
        self.assertIn(reviewer, paper_a.share_distribution)

    def test_act_returns_review_action_records(self):
        author = DummyAgent("author")
        paper = Paper(author=author, current_ac=10.0)
        reviewer = DummyAgent(
            "reviewer",
            actions=[
                ("peer_review", paper),
                ("peer_review", paper),
                ("finish_review_write_paper", None),
            ],
        )

        started = reviewer.act()
        self.assertEqual(started.kind, "review_started")
        self.assertIs(started.paper, paper)

        self._advance_to_review_choice(reviewer)
        continued = reviewer.act()
        self.assertEqual(continued.kind, "review_continued")

        reviewer._review_choice_pending = True
        finished = reviewer.act()
        self.assertEqual(finished.kind, "review_finished_write")

    def test_environment_records_capital_and_actions(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(
            author=author,
            current_ac=10.0,
            accrual_rate=2.0,
            share_distribution={author: 0.75, reviewer: 0.25},
        )
        history = History()
        env = Environment(agents=[author, reviewer], papers=[paper], history=history)

        env.agentact()
        env.nextstep()

        self.assertEqual(history.days, [1])
        self.assertEqual(len(history.actions), 2)
        self.assertEqual(len(history.agent_capital["author"]), 1)
        self.assertEqual(len(history.agent_capital["reviewer"]), 1)
        self.assertAlmostEqual(history.agent_capital["author"][0], 9.0)
        self.assertAlmostEqual(history.agent_capital["reviewer"][0], 3.0)
        self.assertEqual(history.scalars["num_papers"][0], 1.0)

    def test_history_to_csv_has_header_and_one_row_per_day(self):
        author = DummyAgent("author")
        paper = Paper(author=author, current_ac=5.0, accrual_rate=1.0)
        history = History()
        env = Environment(agents=[author], papers=[paper], history=history)
        env.run(3)

        path = os.path.join(tempfile.mkdtemp(), "history.csv")
        history.to_csv(path)
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))

        self.assertEqual(rows[0][0], "day")
        self.assertIn("total_capital", rows[0])
        self.assertEqual(len(rows), 1 + 3)
        self.assertEqual([row[0] for row in rows[1:]], ["1", "2", "3"])

    def test_gini_ranges_from_equal_to_unequal(self):
        self.assertEqual(gini([]), 0.0)
        self.assertEqual(gini([5.0, 5.0, 5.0]), 0.0)
        self.assertAlmostEqual(gini([0.0, 0.0, 0.0, 10.0]), 0.75)

    @unittest.skipUnless(_HAS_MPL, "matplotlib not installed")
    def test_visualize_writes_pngs(self):
        import visualize

        author = DummyAgent("author")
        paper = Paper(author=author, current_ac=5.0)
        history = History()
        env = Environment(agents=[author], papers=[paper], history=history)
        env.run(2)

        outdir = tempfile.mkdtemp()
        paths = visualize.plot_all(history, outdir)
        for path in paths.values():
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
