from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

from Agent import Agent
from Paper import Paper

AgentFactory = Callable[..., Agent]


class Environment:
    """First-pass simulation environment for the peer-review marketplace."""

    def __init__(
        self,
        agents: Sequence[Agent] | None = None,
        papers: Sequence[Paper] | None = None,
        num_agents: int | None = None,
        agent_cls: type[Agent] | AgentFactory | None = None,
        forecast_horizon_days: int = 30,
    ):
        if agents is not None and num_agents is not None:
            raise ValueError("Pass either agents or num_agents, not both.")

        self.forecast_horizon_days = forecast_horizon_days
        self.day = 0

        if agents is None:
            count = 0 if num_agents is None else num_agents
            self.agents = [
                self._create_agent(agent_cls, forecast_horizon_days)
                for _ in range(count)
            ]
        else:
            self.agents = list(agents)

        self.papers = list(Agent.all_papers if papers is None else papers)
        Agent.all_papers = self.papers
        self._configure_agent_forecasts()

    def agentact(self):
        """Ask each agent to act once, in order."""
        self._sync_papers()
        for agent in self.agents:
            agent.act()
        self._sync_papers()

    def nextstep(self):
        """Advance the environment by one simulated day."""
        self._sync_papers()
        for paper in self.papers:
            if hasattr(paper, "advance_accrual"):
                paper.advance_accrual()
            else:
                paper.current_ac += paper.accrual_rate

        self.update_agent_capital()
        self.day += 1

    def run(self, days: int):
        if days < 0:
            raise ValueError("days must be non-negative")

        for _ in range(days):
            self.agentact()
            self.nextstep()
        return self

    def market(self):
        pass

    def update_agent_capital(self):
        """Recompute each environment agent's capital from current paper shares."""
        agent_set = set(self.agents)
        for agent in self.agents:
            agent.academic_capital = 0.0

        for paper in self.papers:
            for agent, share in paper.share_distribution.items():
                if agent in agent_set:
                    agent.academic_capital += share * paper.current_ac

    def _sync_papers(self):
        if Agent.all_papers is not self.papers:
            Agent.all_papers = self.papers

    def _configure_agent_forecasts(self):
        for agent in self.agents:
            if hasattr(agent, "forecast_horizon_days"):
                agent.forecast_horizon_days = self.forecast_horizon_days

    def _create_agent(
        self,
        agent_cls: type[Agent] | AgentFactory | None,
        forecast_horizon_days: int,
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
        if accepts_kwargs or "forecast_horizon_days" in signature.parameters:
            kwargs["forecast_horizon_days"] = forecast_horizon_days

        return agent_cls(**kwargs)
