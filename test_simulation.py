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
    GOOD_FAITH_REVIEW_EFFORT_THRESHOLD,
    MIN_REVIEW_EFFORT_THRESHOLD,
    Paper,
)

try:
    import matplotlib  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class DummyAgent(Agent):
    def __init__(self, name: str = "agent", actions=None, review_effort_deltas=None):
        super().__init__(intrinsic_talent=1.0)
        self.name = name
        self.actions = list(actions or [])
        self.review_effort_deltas = list(review_effort_deltas or [])

    def choose_action(self):
        if self.actions:
            return self.actions.pop(0)
        return "write_paper", None

    def review_effort_delta(self):
        if self.review_effort_deltas:
            return self.review_effort_deltas.pop(0)
        return 0.5


class RecordingAgent(DummyAgent):
    def __init__(self, log: list[str], name: str):
        super().__init__(name=name)
        self.log = log

    def act(self):
        self.log.append(self.name)


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

    def test_good_faith_review_blocks_for_four_turns(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(author=author, current_ac=100.0)

        reviewer.peer_review(paper)
        self.assertEqual(reviewer.active_review_days_remaining, 3)
        self.assertIs(paper.review_in_progress_by, reviewer)
        self.assertNotIn(reviewer, paper.share_distribution)

        reviewer.act()
        reviewer.act()
        reviewer.act()

        self.assertIsNone(reviewer.active_review_paper)
        self.assertIsNone(paper.review_in_progress_by)
        self.assertEqual(paper.share_distribution[reviewer], 0.01)

    def test_good_faith_review_lock_blocks_other_reviewers(self):
        author = DummyAgent("author")
        first_reviewer = DummyAgent("first reviewer")
        second_reviewer = DummyAgent("second reviewer")
        paper = Paper(author=author)

        first_reviewer.peer_review(paper)
        second_reviewer.bad_faith_review(paper)
        second_reviewer.peer_review(paper)

        self.assertNotIn(second_reviewer, paper.share_distribution)
        self.assertIsNone(second_reviewer.active_review_paper)
        self.assertIs(paper.review_in_progress_by, first_reviewer)

    def test_minimum_effort_review_completes_as_bad_faith(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(author=author)

        reviewer.bad_faith_review(paper)

        self.assertEqual(paper.share_distribution[reviewer], 0.01)
        self.assertEqual(reviewer.active_review_days_remaining, 0)
        self.assertEqual(paper.bad_faith_reviews, 1)
        self.assertEqual(paper.good_faith_reviews, 0)
        self.assertEqual(
            paper.review_records[-1]["effort"],
            MIN_REVIEW_EFFORT_THRESHOLD,
        )

    def test_review_below_minimum_effort_does_not_complete(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(author=author)

        review_share = paper.add_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD - 0.01)

        self.assertEqual(review_share, 0.0)
        self.assertNotIn(reviewer, paper.share_distribution)
        self.assertEqual(paper.completed_peer_reviews, 0)

    def test_share_is_same_for_bad_and_good_faith_reviews(self):
        author = DummyAgent("author")
        bad_reviewer = DummyAgent("bad reviewer")
        good_reviewer = DummyAgent("good reviewer")
        bad_paper = Paper(author=author)
        good_paper = Paper(author=author)

        bad_share = bad_paper.add_review(bad_reviewer, MIN_REVIEW_EFFORT_THRESHOLD)
        good_share = good_paper.add_review(
            good_reviewer,
            GOOD_FAITH_REVIEW_EFFORT_THRESHOLD,
        )

        self.assertEqual(bad_share, good_share)
        self.assertEqual(bad_paper.bad_faith_reviews, 1)
        self.assertEqual(good_paper.good_faith_reviews, 1)

    def test_agent_cannot_review_same_paper_twice(self):
        author = DummyAgent("author")
        reviewer = DummyAgent("reviewer")
        paper = Paper(author=author)

        reviewer.bad_faith_review(paper)
        share_after_first_review = paper.share_distribution[reviewer]
        rate_after_first_review = paper.accrual_rate
        reviewer.bad_faith_review(paper)
        reviewer.peer_review(paper)

        self.assertEqual(paper.completed_peer_reviews, 1)
        self.assertEqual(paper.share_distribution[reviewer], share_after_first_review)
        self.assertEqual(paper.accrual_rate, rate_after_first_review)
        self.assertIsNone(reviewer.active_review_paper)

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

        self.assertIn(action, {"peer_review", "bad_faith_review"})
        self.assertIs(paper, high_value)

    def test_act_returns_review_action_records(self):
        author = DummyAgent("author")
        paper = Paper(author=author, current_ac=10.0)
        reviewer = DummyAgent("reviewer", actions=[("peer_review", paper)])
        Agent.all_papers = [paper]

        started = reviewer.act()
        self.assertEqual(started.kind, "review_started")
        self.assertIs(started.paper, paper)

        kinds = [reviewer.act().kind for _ in range(3)]
        self.assertEqual(
            kinds, ["review_continued", "review_continued", "review_completed"]
        )

    def test_act_returns_write_and_bad_faith_records(self):
        author = DummyAgent("author")
        writer = DummyAgent("writer", actions=[("write_paper", None)])
        self.assertEqual(writer.act().kind, "write_paper")

        paper = Paper(author=author)
        bad = DummyAgent("bad", actions=[("bad_faith_review", paper)])
        record = bad.act()
        self.assertEqual(record.kind, "bad_faith_review")
        self.assertIs(record.paper, paper)
        self.assertEqual(record.review_kind, "bad_faith")
        self.assertEqual(record.review_effort, MIN_REVIEW_EFFORT_THRESHOLD)

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
        self.assertEqual(len(history.actions), 2)  # one entry per agent turn
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
        self.assertEqual(len(rows), 1 + 3)  # header + one row per simulated day
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
