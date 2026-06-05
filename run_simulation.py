from __future__ import annotations

import random
from collections import Counter

from Agent import Agent
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from Paper import Paper


class LoggingHeuristicAgent(HeuristicAgent):
    """Heuristic agent that records each action for demo output."""

    next_id = 1
    current_day = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"Agent {LoggingHeuristicAgent.next_id}"
        LoggingHeuristicAgent.next_id += 1
        self.action_history: list[str] = []
        self.action_categories: list[str] = []

    def act(self):
        if self.active_review_paper is not None:
            paper = self.active_review_paper
            self.advance_active_review()
            if self.active_review_paper is None:
                self.record_action("good-faith review completed", paper)
            else:
                self.record_action("good-faith review continued", paper)
            return

        action, paper = self.choose_action()
        if action == "write_paper":
            self.record_action("worked on own paper")
            self.write_paper()
        elif action == "peer_review":
            self.record_action("good-faith review started", paper)
            previous_active_paper = self.active_review_paper
            self.peer_review(paper)
            if previous_active_paper is None and self.active_review_paper is None:
                self.action_categories.pop()
                self.action_history.pop()
                self.record_action("review unavailable", paper)
        elif action == "bad_faith_review":
            self.record_action("bad-faith review completed", paper)
            self.bad_faith_review(paper)

    def record_action(self, category: str, paper: Paper | None = None):
        paper_text = f" of {display_title(paper)}" if paper is not None else ""
        self.action_categories.append(category)
        self.action_history.append(
            f"day {LoggingHeuristicAgent.current_day}: {category}{paper_text}"
        )


def seed_initial_papers(agents: list[LoggingHeuristicAgent]):
    for index, agent in enumerate(agents, start=1):
        paper = Paper(
            author=agent,
            current_ac=random.uniform(5.0, 20.0),
            accrual_rate=random.uniform(0.8, 1.5),
        )
        paper.title = f"Paper {index}"
        Agent.all_papers.append(paper)


def paper_title(paper: Paper, index: int) -> str:
    if not hasattr(paper, "title"):
        paper.title = f"Paper {index}"
    return paper.title


def display_title(paper: Paper) -> str:
    if not hasattr(paper, "title"):
        try:
            paper.title = f"Paper {Agent.all_papers.index(paper) + 1}"
        except ValueError:
            paper.title = "Unknown paper"
    return paper.title


def print_summary(env: Environment):
    print(f"\nSimulation finished after {env.day} days")
    print(f"Agents: {len(env.agents)}")
    print(f"Papers: {len(env.papers)}")

    action_categories = Counter(
        category
        for agent in env.agents
        for category in getattr(agent, "action_categories", [])
    )

    print("\nAction counts")
    for action, count in action_categories.most_common():
        print(f"- {action}: {count}")

    print("\nFinal agent capital")
    for agent in sorted(env.agents, key=lambda item: item.academic_capital, reverse=True):
        print(f"- {agent.name}: AC={agent.academic_capital:.2f}")

    authored_counts = Counter(paper.author for paper in env.papers)
    print("\nPapers produced by agent")
    for agent in env.agents:
        print(f"- {agent.name}: {authored_counts[agent]}")

    print("\nPapers")
    for index, paper in enumerate(env.papers, start=1):
        title = paper_title(paper, index)
        author_name = getattr(paper.author, "name", "Unknown")
        active_reviewer = getattr(paper, "review_in_progress_by", None)
        active_text = (
            getattr(active_reviewer, "name", "Unknown")
            if active_reviewer is not None
            else "none"
        )
        reviewers = [
            f"{getattr(agent, 'name', 'Unknown')}={share:.2%}"
            for agent, share in paper.share_distribution.items()
            if agent != paper.author
        ]
        reviewer_text = ", ".join(reviewers) if reviewers else "none"
        print(
            f"- {title}: author={author_name}, AC={paper.current_ac:.2f}, "
            f"rate={paper.accrual_rate:.2f}, total reviews={paper.completed_peer_reviews}, "
            f"good={paper.good_faith_reviews}, bad={paper.bad_faith_reviews}, "
            f"unique reviewers={len(paper.reviewed_by)}, "
            f"active review={active_text}, reviewer shares={reviewer_text}"
        )

    print("\nRecent agent actions")
    for agent in env.agents:
        print(f"\n{agent.name}")
        for action in agent.action_history[-10:]:
            print(f"- {action}")


def main():
    random.seed(7)
    Agent.all_papers = []
    LoggingHeuristicAgent.next_id = 1

    agents = [
        LoggingHeuristicAgent(intrinsic_talent=1.0, forecast_horizon_days=30)
        for _ in range(20)
    ]
    seed_initial_papers(agents)

    env = Environment(agents=agents, papers=Agent.all_papers, forecast_horizon_days=30)
    for _ in range(200):
        LoggingHeuristicAgent.current_day = env.day + 1
        env.agentact()
        env.nextstep()

    print_summary(env)


if __name__ == "__main__":
    main()
