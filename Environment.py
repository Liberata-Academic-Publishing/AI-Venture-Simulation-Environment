from __future__ import annotations

import inspect
import random
from collections.abc import Callable, Sequence
from statistics import median
from typing import TYPE_CHECKING

from Agent import Agent
from Paper import Paper

if TYPE_CHECKING:
    from History import History

AgentFactory = Callable[..., Agent]


class Environment:
    """Single-review marketplace environment.

    Each timestep runs in two phases over a freshly shuffled agent order: a
    marketplace phase where agents may claim at most one listed paper to review,
    then a work phase where the remaining agents write or advance their active
    review. Papers list one timestep after they are written and leave the market
    permanently the moment they are claimed.
    """

    def __init__(
        self,
        agents: Sequence[Agent] | None = None,
        papers: Sequence[Paper] | None = None,
        num_agents: int | None = None,
        agent_cls: type[Agent] | AgentFactory | None = None,
        forecast_horizon_timesteps: int = 30,
        history: "History | None" = None,
    ):
        if agents is not None and num_agents is not None:
            raise ValueError("Pass either agents or num_agents, not both.")

        self.forecast_horizon_timesteps = forecast_horizon_timesteps
        self.history = history
        self.timestep = 0

        if agents is None:
            count = 0 if num_agents is None else num_agents
            self.agents = [
                self._create_agent(agent_cls, forecast_horizon_timesteps)
                for _ in range(count)
            ]
        else:
            self.agents = list(agents)

        self.papers = list(Agent.all_papers if papers is None else papers)
        Agent.all_papers = self.papers
        self._configure_agent_forecasts()

    # ---- main loop -------------------------------------------------------
    def run_timestep(self):
        """Advance the simulation by one full timestep."""
        self.timestep += 1
        self._sync_papers()
        self._list_scheduled_papers()
        self._update_market_prices()

        order = list(self.agents)
        random.shuffle(order)
        claimers = self._marketplace_phase(order)
        self._work_phase(order, claimers)

        self._sync_papers()
        self._schedule_new_papers()

        for paper in self.papers:
            paper.accrue_ac()
        self.update_agent_capital()

        if self.history is not None:
            self.history.record_step(self)

    def run(self, timesteps: int):
        if timesteps < 0:
            raise ValueError("timesteps must be non-negative")
        for _ in range(timesteps):
            self.run_timestep()
        return self

    # ---- phases ----------------------------------------------------------
    def _marketplace_phase(self, order: list[Agent]) -> set[Agent]:
        """Phase 1 (instantaneous): each agent may select one listed paper.

        Selecting is pure choice — no effort is applied here. Returns the set of
        agents who claimed a paper so the work phase can apply their first unit
        of review effort instead of re-deciding.
        """
        claimers: set[Agent] = set()
        for agent in order:
            choose = getattr(agent, "choose_marketplace_action", None)
            if choose is None:
                continue
            paper = choose()
            if paper is None or not paper.can_start_review(agent):
                continue
            record = agent.claim_review(paper)
            claimers.add(agent)
            if self.history is not None and record is not None:
                self.history.record_action(self, agent, record)
        return claimers

    def _work_phase(self, order: list[Agent], claimers: set[Agent]) -> None:
        """Phase 2 (effort application): every agent spends one timestep.

        Agents who claimed in the marketplace apply the first unit of effort to
        the new review; everyone else writes or advances/finishes their active
        review.
        """
        for agent in order:
            if agent in claimers:
                record = agent.apply_initial_review_effort()
            else:
                record = agent.work_turn()
            if self.history is not None and record is not None:
                self.history.record_action(self, agent, record)

    # ---- marketplace bookkeeping ----------------------------------------
    def market(self):
        """Backwards-compatible alias for the price-table refresh."""
        self._update_market_prices()

    def _update_market_prices(self) -> None:
        listed = [p for p in self.papers if p.review_available]
        if not listed:
            return
        median_quality = median(p.quality for p in listed)
        histories = [getattr(a, "peer_review_history", 0.0) for a in self.agents]
        mean_history = sum(histories) / len(histories) if histories else 0.0
        for paper in listed:
            paper.update_price_table(self.agents, median_quality, mean_history)

    def _list_scheduled_papers(self) -> None:
        for paper in self.papers:
            scheduled = getattr(paper, "scheduled_listing_timestep", None)
            if (
                scheduled is not None
                and scheduled <= self.timestep
                and not paper.review_claimed
                and not paper.reviewed
            ):
                paper.market_listed = True
                paper.scheduled_listing_timestep = None

    def _schedule_new_papers(self) -> None:
        for paper in self.papers:
            already_scheduled = getattr(paper, "scheduled_listing_timestep", None)
            if (
                already_scheduled is None
                and not paper.market_listed
                and not paper.review_claimed
                and not paper.reviewed
            ):
                paper.scheduled_listing_timestep = self.timestep + 1

    # ---- capital ---------------------------------------------------------
    def update_agent_capital(self):
        agent_set = set(self.agents)
        for agent in self.agents:
            agent.academic_capital = 0.0

        for paper in self.papers:
            for agent, share in paper.share_distribution.items():
                if agent in agent_set:
                    agent.academic_capital += share * paper.current_ac

    # ---- helpers ---------------------------------------------------------
    def _sync_papers(self):
        if Agent.all_papers is not self.papers:
            self.papers = Agent.all_papers

    def _configure_agent_forecasts(self):
        for agent in self.agents:
            if hasattr(agent, "forecast_horizon_timesteps"):
                agent.forecast_horizon_timesteps = self.forecast_horizon_timesteps

    def _create_agent(
        self,
        agent_cls: type[Agent] | AgentFactory | None,
        forecast_horizon_timesteps: int,
    ) -> Agent:
        if agent_cls is None:
            from HeuristicAgent import HeuristicAgent

            agent_cls = HeuristicAgent

        kwargs = {"intrinsic_talent": 1.0}
        signature = inspect.signature(agent_cls)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs or "forecast_horizon_timesteps" in signature.parameters:
            kwargs["forecast_horizon_timesteps"] = forecast_horizon_timesteps

        return agent_cls(**kwargs)
