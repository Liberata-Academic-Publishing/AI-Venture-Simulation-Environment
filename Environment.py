from __future__ import annotations

import inspect
import random
from collections.abc import Callable, Sequence
from statistics import median
from typing import TYPE_CHECKING

from Agent import Agent, validate_continuous_publishing
from config import SIM
from Paper import (
    REVIEW_PARADIGM_DISCRETE,
    Paper,
    fair_market_price_from_epsilons,
    validate_review_paradigm,
)

if TYPE_CHECKING:
    from History import History

AgentFactory = Callable[..., Agent]


class Environment:
    """Single-review marketplace environment.

    Each timestep reshuffles the agent order, then advances the world by one
    timestep. In the ``continuous`` paradigm this is one merged decision per
    agent (claim+review, continue a review, research, or finish research). In the
    ``discrete`` paradigm it is the two-phase flow (a marketplace claim phase
    then a work phase). Papers list one timestep after they are written and leave
    the market permanently the moment they are claimed.
    """

    def __init__(
        self,
        agents: Sequence[Agent] | None = None,
        papers: Sequence[Paper] | None = None,
        num_agents: int | None = None,
        agent_cls: type[Agent] | AgentFactory | None = None,
        forecast_horizon_timesteps: int = 30,
        review_paradigm: str = "continuous",
        continuous_publishing: str = SIM.continuous_publishing,
        continuous_paper_timesteps: float = SIM.continuous_paper_timesteps,
        paper_effort_mode: str = SIM.paper_effort_mode,
        paper_effort_min: float = SIM.paper_effort_min,
        paper_effort_max: float = SIM.paper_effort_max,
        discrete_paper_timesteps: float = SIM.discrete_paper_timesteps,
        history: "History | None" = None,
    ):
        if agents is not None and num_agents is not None:
            raise ValueError("Pass either agents or num_agents, not both.")

        self.forecast_horizon_timesteps = forecast_horizon_timesteps
        self.review_paradigm = validate_review_paradigm(review_paradigm)
        self.continuous_publishing = validate_continuous_publishing(
            continuous_publishing
        )
        self.continuous_paper_timesteps = float(continuous_paper_timesteps)
        self.paper_effort_mode = paper_effort_mode
        self.paper_effort_min = float(paper_effort_min)
        self.paper_effort_max = float(paper_effort_max)
        self.discrete_paper_timesteps = float(discrete_paper_timesteps)
        self.history = history
        self.timestep = 0
        self.fair_market_price = fair_market_price_from_epsilons(
            [SIM.prior_review_epsilon]
        )

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
        Agent.all_agents = self.agents
        self._configure_agents()

    # ---- main loop -------------------------------------------------------
    def run_timestep(self):
        """Advance the simulation by one full timestep."""
        self.timestep += 1
        self._sync_papers()
        self._list_scheduled_papers()
        self._update_market_prices()

        order = list(self.agents)
        random.shuffle(order)
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            claimers = self._marketplace_phase(order)
            self._work_phase(order, claimers)
        else:
            self._continuous_phase(order)

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
    def _continuous_phase(self, order: list[Agent]) -> None:
        """Single merged phase: each agent makes one decision and spends one
        timestep of effort on it (claim+review, continue review, research, or
        finish research). Agents act in the freshly shuffled order so claims
        race for whatever papers are still listed."""
        for agent in order:
            act = getattr(agent, "act_continuous", None)
            if act is None:
                continue
            records = act()
            if self.history is None or not records:
                continue
            for record in records:
                if record is not None:
                    self.history.record_action(self, agent, record)

    def _marketplace_phase(self, order: list[Agent]) -> set[Agent]:
        """Phase 1 (instantaneous): each agent may select one listed paper.

        Selecting is pure choice — no effort is applied here. Returns the set of
        agents who claimed a paper so the work phase can apply their first unit
        of review effort instead of re-deciding.
        """
        claimers: set[Agent] = set()
        for agent in order:
            if (
                self.review_paradigm == REVIEW_PARADIGM_DISCRETE
                and agent.active_review_paper is not None
            ):
                continue
            choose = getattr(agent, "choose_marketplace_action", None)
            if choose is None:
                continue
            paper = choose()
            if paper is None or not paper.can_start_review(agent):
                continue
            review_kind = None
            if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
                review_kind = agent.choose_review_kind(paper)
            record = agent.claim_review(paper, review_kind=review_kind)
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
        epsilons = [
            getattr(a, "peer_review_epsilon_history", SIM.prior_review_epsilon)
            for a in self.agents
        ]
        mean_epsilon = (
            sum(epsilons) / len(epsilons)
            if epsilons
            else SIM.prior_review_epsilon
        )
        self.fair_market_price = fair_market_price_from_epsilons(epsilons)
        for paper in listed:
            paper.update_price_table(
                self.agents,
                median_quality,
                mean_epsilon,
                fair_market_price=self.fair_market_price,
            )

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

    def _configure_agents(self):
        for agent in self.agents:
            required = getattr(agent, "requires_review_paradigm", None)
            if required is not None and required != self.review_paradigm:
                raise ValueError(
                    f"{type(agent).__name__} requires review_paradigm={required!r}"
                )
            if hasattr(agent, "configure_review_paradigm"):
                agent.configure_review_paradigm(self.review_paradigm)
            if hasattr(agent, "configure_continuous_publishing"):
                agent.configure_continuous_publishing(
                    self.continuous_publishing,
                    self.continuous_paper_timesteps,
                )
            if hasattr(agent, "configure_paper_effort"):
                agent.configure_paper_effort(
                    self.paper_effort_mode,
                    self.paper_effort_min,
                    self.paper_effort_max,
                    discrete_timesteps=self.discrete_paper_timesteps,
                )
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
