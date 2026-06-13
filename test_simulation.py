from __future__ import annotations

import csv
import os
import statistics
import tempfile
import unittest

from Agent import Agent, PAPER_THRESHOLD
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History, gini
from Paper import (
    BAD_FAITH_REVIEW,
    BAD_REVIEW_TIMESTEPS,
    GOOD_FAITH_REVIEW,
    GOOD_REVIEW_TIMESTEPS,
    MIN_REVIEW_EFFORT_THRESHOLD,
    REVIEW_EFFORT_PER_TIMESTEP,
    Paper,
    review_accrual_bump,
)
from RandomAgent import ProbabilisticDiscreteAgent, RandomAgent

try:
    import matplotlib  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class ScriptAgent(Agent):
    """Agent driven by scripted marketplace/work decisions for deterministic tests."""

    def __init__(self, name: str = "agent", marketplace=None, work=None):
        super().__init__(intrinsic_talent=1.0)
        self.name = name
        self.marketplace = list(marketplace or [])
        self.work = list(work or [])

    def choose_marketplace_action(self):
        if self.marketplace:
            return self.marketplace.pop(0)
        return None

    def choose_work_action(self):
        if self.work:
            return self.work.pop(0)
        return "write_paper", None

    def writing_effort_delta(self):
        return 0.5


class ReviewKindScriptAgent(ScriptAgent):
    def __init__(self, review_kind: str, **kwargs):
        super().__init__(**kwargs)
        self.review_kind = review_kind

    def choose_review_kind(self, paper):
        return self.review_kind


class RecordingAgent(ScriptAgent):
    def __init__(self, log: list[str], name: str):
        super().__init__(name=name)
        self.log = log

    def work_turn(self):
        self.log.append(self.name)
        return super().work_turn()


def _listed_paper(author, **kwargs) -> Paper:
    paper = Paper(author=author, market_listed=True, **kwargs)
    return paper


class MarketplaceLifecycleTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_published_paper_lists_one_timestep_later(self):
        author = ScriptAgent("author")
        author.paper_progress = PAPER_THRESHOLD - 0.25
        env = Environment(agents=[author])

        env.run_timestep()  # timestep 1: author writes and publishes
        self.assertEqual(len(Agent.all_papers), 1)
        paper = Agent.all_papers[0]
        self.assertFalse(paper.market_listed)
        self.assertFalse(paper.review_available)

        env.run_timestep()  # timestep 2: scheduled listing happens at the start
        self.assertTrue(paper.market_listed)
        self.assertTrue(paper.review_available)

    def test_claimed_paper_leaves_market_for_everyone(self):
        author = ScriptAgent("author")
        first = ScriptAgent("first")
        second = ScriptAgent("second")
        paper = _listed_paper(author)

        self.assertTrue(paper.start_review(first))
        self.assertTrue(paper.review_claimed)
        self.assertFalse(paper.market_listed)
        self.assertFalse(paper.review_available)
        self.assertFalse(paper.can_start_review(second))

    def test_paper_can_only_be_reviewed_once(self):
        author = ScriptAgent("author")
        first = ScriptAgent("first")
        second = ScriptAgent("second")
        paper = _listed_paper(author, quality=1.0)

        paper.start_review(first)
        paper.finish_review(first, MIN_REVIEW_EFFORT_THRESHOLD)

        self.assertTrue(paper.reviewed)
        self.assertFalse(paper.review_available)
        self.assertFalse(paper.can_start_review(second))

    def test_min_effort_review_earns_reward(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0, current_ac=10.0)
        paper.update_price_table([author, reviewer], 1.0, 0.0)

        share = paper.add_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD)

        self.assertGreater(share, 0.0)
        self.assertGreater(paper.accrual_rate, 1.0)
        self.assertEqual(paper.completed_peer_reviews, 1)
        self.assertIn(reviewer, paper.share_distribution)

    def test_subthreshold_review_consumes_opportunity_without_reward(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        paper.update_price_table([reviewer], 1.0, 0.0)

        paper.start_review(reviewer)
        share = paper.finish_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD - 0.5)

        self.assertEqual(share, 0.0)
        self.assertEqual(paper.accrual_rate, 1.0)
        self.assertTrue(paper.reviewed)
        self.assertNotIn(reviewer, paper.share_distribution)


class EconomicsTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_review_bump_grows_with_diminishing_returns(self):
        self.assertEqual(review_accrual_bump(MIN_REVIEW_EFFORT_THRESHOLD - 0.5), 0.0)
        b1 = review_accrual_bump(1.0)
        b2 = review_accrual_bump(2.0)
        b3 = review_accrual_bump(3.0)
        self.assertGreater(b1, 0.0)
        self.assertGreater(b2, b1)
        self.assertGreater(b3, b2)
        self.assertGreater(b2 - b1, b3 - b2)

    def test_higher_quality_raises_bump(self):
        self.assertGreater(
            review_accrual_bump(1.0, quality=1.5),
            review_accrual_bump(1.0, quality=0.8),
        )

    def test_higher_effort_yields_higher_accrual_rate(self):
        author = ScriptAgent("author")
        low_reviewer = ScriptAgent("low")
        high_reviewer = ScriptAgent("high")
        low = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        high = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        low.update_price_table([low_reviewer], 1.0, 0.0)
        high.update_price_table([high_reviewer], 1.0, 0.0)

        low.add_review(low_reviewer, MIN_REVIEW_EFFORT_THRESHOLD)
        high.add_review(high_reviewer, MIN_REVIEW_EFFORT_THRESHOLD + 3)

        self.assertGreater(high.accrual_rate, low.accrual_rate)

    def test_price_drops_for_higher_quality_papers(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        low_q = _listed_paper(author, quality=0.8)
        high_q = _listed_paper(author, quality=1.5)
        median_q = statistics.median([0.8, 1.5])

        low_q.update_price_table([reviewer], median_q, 0.0)
        high_q.update_price_table([reviewer], median_q, 0.0)

        self.assertGreater(low_q.offered_share(reviewer), high_q.offered_share(reviewer))

    def test_price_rises_for_stronger_reviewer_history(self):
        author = ScriptAgent("author")
        rookie = ScriptAgent("rookie")
        veteran = ScriptAgent("veteran")
        veteran.peer_review_history = 5.0
        paper = _listed_paper(author, quality=1.0)

        paper.update_price_table([rookie, veteran], 1.0, mean_peer_review_history=2.5)

        self.assertGreater(paper.offered_share(veteran), paper.offered_share(rookie))


class ReviewerStateTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_peer_review_history_updates_on_completion(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer", work=[("finish_review_write_paper", None)])
        paper = _listed_paper(author, quality=1.0, current_ac=100.0, accrual_rate=1.0)
        paper.update_price_table([reviewer], 1.0, 0.0)
        Agent.all_papers = [paper]

        claimed = reviewer.claim_review(paper)
        self.assertIsNone(claimed)  # phase 1 is pure selection, no record
        self.assertEqual(reviewer.active_review_effort, 0.0)

        started = reviewer.apply_initial_review_effort()
        self.assertEqual(started.kind, "review_started")
        self.assertEqual(reviewer.active_review_effort, REVIEW_EFFORT_PER_TIMESTEP)

        finished = reviewer.work_turn()
        self.assertEqual(finished.kind, "review_finished_write")
        self.assertEqual(reviewer.completed_review_count, 1)
        self.assertGreater(reviewer.peer_review_history, 0.0)
        self.assertIsNone(reviewer.active_review_paper)

    def test_grabbing_new_paper_finalizes_active_review(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        first = _listed_paper(author, quality=1.0, current_ac=50.0)
        second = _listed_paper(author, quality=1.0, current_ac=100.0)
        first.update_price_table([reviewer], 1.0, 0.0)
        second.update_price_table([reviewer], 1.0, 0.0)

        reviewer.claim_review(first)
        reviewer.apply_initial_review_effort()  # phase 2 effort on the first review
        record = reviewer.claim_review(second)

        self.assertEqual(record.kind, "review_finished_peer_review")
        self.assertIs(record.paper, first)
        self.assertIs(reviewer.active_review_paper, second)
        self.assertEqual(reviewer.active_review_effort, 0.0)  # second not worked yet
        self.assertTrue(first.reviewed)
        self.assertFalse(second.reviewed)
        self.assertIn(reviewer, first.share_distribution)


class ReviewParadigmTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_continuous_mode_classifies_finished_reviews_by_threshold(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent(
            "reviewer",
            marketplace=[],
            work=[("peer_review", None), ("finish_review_write_paper", None)],
        )
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="continuous",
        )
        paper.update_price_table([reviewer], 1.0, 0.0)

        reviewer.claim_review(paper)
        reviewer.apply_initial_review_effort()
        reviewer.work_turn()
        record = reviewer.work_turn()

        self.assertEqual(record.review_kind, GOOD_FAITH_REVIEW)
        self.assertEqual(reviewer.last_review_kind, GOOD_FAITH_REVIEW)
        self.assertEqual(paper.review_records[-1]["review_kind"], GOOD_FAITH_REVIEW)

    def test_discrete_bad_faith_review_finishes_after_one_timestep(self):
        author = ScriptAgent("author")
        reviewer = ReviewKindScriptAgent(
            BAD_FAITH_REVIEW, name="reviewer", marketplace=[]
        )
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        reviewer.marketplace = [paper]
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="discrete",
        )

        env.run_timestep()

        self.assertTrue(paper.reviewed)
        self.assertIsNone(reviewer.active_review_paper)
        self.assertEqual(reviewer.last_review_kind, BAD_FAITH_REVIEW)
        self.assertEqual(paper.review_records[-1]["effort"], BAD_REVIEW_TIMESTEPS)
        self.assertEqual(history.completed_reviews[-1][4], BAD_FAITH_REVIEW)
        self.assertEqual(history.action_counts["bad_faith_review"], 1)
        self.assertEqual(history.scalars["bad_faith_reviews"][-1], 1.0)
        self.assertEqual(history.scalars["good_faith_reviews"][-1], 0.0)

    def test_discrete_good_faith_review_uses_fixed_five_timesteps(self):
        author = ScriptAgent("author")
        reviewer = ReviewKindScriptAgent(
            GOOD_FAITH_REVIEW, name="reviewer", marketplace=[]
        )
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        reviewer.marketplace = [paper]
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="discrete",
        )

        env.run(int(GOOD_REVIEW_TIMESTEPS) - 1)

        self.assertFalse(paper.reviewed)
        self.assertIs(reviewer.active_review_paper, paper)
        self.assertEqual(reviewer.active_review_effort, GOOD_REVIEW_TIMESTEPS - 1)

        env.run_timestep()

        self.assertTrue(paper.reviewed)
        self.assertIsNone(reviewer.active_review_paper)
        self.assertEqual(reviewer.last_review_kind, GOOD_FAITH_REVIEW)
        self.assertEqual(paper.review_records[-1]["effort"], GOOD_REVIEW_TIMESTEPS)
        self.assertEqual(history.completed_reviews[-1][4], GOOD_FAITH_REVIEW)
        self.assertEqual(history.action_counts["good_faith_review"], 1)
        self.assertEqual(history.scalars["good_faith_reviews"][-1], 1.0)

    def test_probabilistic_agents_are_discrete_only(self):
        agent = ProbabilisticDiscreteAgent(intrinsic_talent=1.0)

        with self.assertRaises(ValueError):
            Environment(agents=[agent], review_paradigm="continuous")

        env = Environment(agents=[agent], review_paradigm="discrete")
        self.assertEqual(env.review_paradigm, "discrete")

    def test_invalid_review_paradigm_is_rejected(self):
        with self.assertRaises(ValueError):
            Environment(agents=[], review_paradigm="mixed")

    def test_build_simulation_can_add_random_control_agents(self):
        from run_simulation import build_simulation

        history = History()
        env = build_simulation(
            history,
            num_agents=0,
            rl_agents=0,
            random_agents=2,
            seed=3,
        )

        self.assertEqual(len(env.agents), 2)
        self.assertTrue(all(isinstance(agent, RandomAgent) for agent in env.agents))


class HeuristicPolicyTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_writes_when_nothing_is_reviewable(self):
        agent = HeuristicAgent(intrinsic_talent=1.0)
        self.assertIsNone(agent.choose_marketplace_action())
        self.assertEqual(agent.choose_work_action(), ("write_paper", None))

    def test_claims_highest_value_listed_paper(self):
        agent = HeuristicAgent(intrinsic_talent=1.0)
        author = ScriptAgent("author")
        low = _listed_paper(author, quality=1.0, current_ac=10.0)
        high = _listed_paper(author, quality=1.0, current_ac=200.0)
        Agent.all_papers = [low, high]
        for paper in (low, high):
            paper.update_price_table([agent], 1.0, 0.0)

        self.assertIs(agent.choose_marketplace_action(), high)

    def test_work_phase_finishes_a_normal_review(self):
        agent = HeuristicAgent(intrinsic_talent=1.0, forecast_horizon_timesteps=30)
        author = ScriptAgent("author")
        paper = _listed_paper(author, quality=1.0, current_ac=10.0)
        paper.update_price_table([agent], 1.0, 0.0)
        Agent.all_papers = [paper]

        agent.claim_review(paper)
        agent.apply_initial_review_effort()
        action, _ = agent.choose_work_action()

        self.assertEqual(action, "finish_review_write_paper")

    def test_work_phase_continues_when_marginal_effort_dominates(self):
        # A low-talent agent's own research is weak, so investing another
        # timestep in a valuable review beats finishing and writing.
        agent = HeuristicAgent(intrinsic_talent=0.1, forecast_horizon_timesteps=60)
        author = ScriptAgent("author")
        paper = _listed_paper(author, quality=2.0, current_ac=10.0)
        paper.update_price_table([agent], 2.0, 0.0)
        Agent.all_papers = [paper]

        agent.claim_review(paper)
        agent.apply_initial_review_effort()
        action, target = agent.choose_work_action()

        self.assertEqual(action, "peer_review")
        self.assertIs(target, paper)


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_work_phase_runs_agents_each_timestep(self):
        log: list[str] = []
        agents = [RecordingAgent(log, "first"), RecordingAgent(log, "second")]
        env = Environment(agents=agents)

        env.run_timestep()

        self.assertEqual(sorted(log), ["first", "second"])
        self.assertEqual(env.timestep, 1)

    def test_accrual_and_capital_update_each_timestep(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = Paper(
            author=author,
            accrual_rate=2.0,
            current_ac=10.0,
            share_distribution={author: 0.75, reviewer: 0.25},
        )
        env = Environment(agents=[author, reviewer], papers=[paper])

        env.run_timestep()

        self.assertEqual(env.timestep, 1)
        self.assertEqual(paper.current_ac, 12.0)
        self.assertAlmostEqual(author.academic_capital, 9.0)
        self.assertAlmostEqual(reviewer.academic_capital, 3.0)

    def test_history_records_timesteps_and_actions(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = Paper(
            author=author,
            accrual_rate=2.0,
            current_ac=10.0,
            share_distribution={author: 0.75, reviewer: 0.25},
        )
        history = History()
        env = Environment(agents=[author, reviewer], papers=[paper], history=history)

        env.run_timestep()

        self.assertEqual(history.timesteps, [1])
        self.assertEqual(history.days, [1])  # backwards-compatible alias
        self.assertEqual(len(history.actions), 2)
        self.assertAlmostEqual(history.agent_capital["author"][0], 9.0)
        self.assertEqual(history.scalars["num_papers"][0], 1.0)

    def test_full_run_produces_reviews(self):
        from run_simulation import build_simulation

        history = History()
        env = build_simulation(history, num_agents=12, rl_agents=0, seed=3)
        env.run(120)

        completed = sum(
            getattr(p, "completed_peer_reviews", 0) for p in env.papers
        )
        self.assertGreater(len(env.papers), 0)
        self.assertGreater(completed, 0)
        # Reviewed papers leave the market permanently.
        for paper in env.papers:
            if paper.reviewed:
                self.assertFalse(paper.review_available)

    def test_history_to_csv_has_header_and_one_row_per_timestep(self):
        author = ScriptAgent("author")
        paper = Paper(author=author, accrual_rate=1.0, current_ac=5.0)
        history = History()
        env = Environment(agents=[author], papers=[paper], history=history)
        env.run(3)

        path = os.path.join(tempfile.mkdtemp(), "history.csv")
        history.to_csv(path)
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))

        self.assertEqual(rows[0][0], "timestep")
        self.assertIn("total_capital", rows[0])
        self.assertEqual(len(rows), 1 + 3)
        self.assertEqual([row[0] for row in rows[1:]], ["1", "2", "3"])


class UtilityTest(unittest.TestCase):
    def test_gini_ranges_from_equal_to_unequal(self):
        self.assertEqual(gini([]), 0.0)
        self.assertEqual(gini([5.0, 5.0, 5.0]), 0.0)
        self.assertAlmostEqual(gini([0.0, 0.0, 0.0, 10.0]), 0.75)

    @unittest.skipUnless(_HAS_MPL, "matplotlib not installed")
    def test_visualize_writes_pngs(self):
        import visualize

        Agent.all_papers = []
        author = ScriptAgent("author")
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
