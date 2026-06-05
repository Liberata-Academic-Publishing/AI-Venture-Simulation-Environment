from __future__ import annotations

import os
import random
from collections import Counter

from Agent import Agent
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from Paper import Paper

NUM_AGENTS = 20
NUM_DAYS = 200
OUTPUT_DIR = "runs"


def seed_initial_papers(agents: list[Agent]):
    for index, agent in enumerate(agents, start=1):
        paper = Paper(
            author=agent,
            current_ac=random.uniform(5.0, 20.0),
            accrual_rate=random.uniform(0.8, 1.5),
        )
        paper.title = f"Paper {index}"
        Agent.all_papers.append(paper)


def print_summary(env: Environment, history: History):
    print(f"\nSimulation finished after {env.day} days")
    print(f"Agents: {len(env.agents)}")
    print(f"Papers: {len(env.papers)}")

    print("\nAction counts")
    for action, count in history.action_counts.most_common():
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
        title = getattr(paper, "title", f"Paper {index}")
        author_name = getattr(paper.author, "name", "Unknown")
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
            f"unique reviewers={len(paper.reviewed_by)}, reviewer shares={reviewer_text}"
        )

    print("\nRecent agent actions")
    for agent in env.agents:
        recent = history.agent_actions.get(agent.name, [])
        if not recent:
            continue
        print(f"\n{agent.name}")
        for line in recent[-10:]:
            print(f"- {line}")


def save_outputs(history: History):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = history.to_csv(os.path.join(OUTPUT_DIR, "history.csv"))
    json_path = history.to_json(os.path.join(OUTPUT_DIR, "history.json"))
    print(f"\nWrote time-series to {csv_path} and {json_path}")

    try:
        import visualize
    except ImportError as exc:
        print(f"Skipping charts ({exc}).")
        print(f"Open {csv_path} in a spreadsheet to plot it instead.")
        return

    print("Wrote charts:")
    for _, path in visualize.plot_all(history, OUTPUT_DIR).items():
        print(f"- {path}")


def main():
    random.seed(7)
    Agent.all_papers = []

    agents = [
        HeuristicAgent(intrinsic_talent=1.0, forecast_horizon_days=30, name=f"Agent {i}")
        for i in range(1, NUM_AGENTS + 1)
    ]
    seed_initial_papers(agents)

    history = History()
    env = Environment(
        agents=agents,
        papers=Agent.all_papers,
        forecast_horizon_days=30,
        history=history,
    )
    for _ in range(NUM_DAYS):
        env.agentact()
        env.nextstep()

    print_summary(env, history)
    save_outputs(history)


if __name__ == "__main__":
    main()
