from __future__ import annotations

import argparse
import os
import random
from collections import Counter

from Agent import Agent
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from BadFaithAgent import BadFaithAgent
from History import History
from Paper import Paper

NUM_AGENTS = 20 #This is meant to be good agents
NUM_BAD_AGENTS = 20
NUM_DAYS = 200
OUTPUT_DIR = "runs"

# Map raw action kinds to the decision an agent actively made on its turn.
# Review continuations/completions are follow-through, not fresh choices, so
# they are excluded from the choice breakdown.
DECISION_LABELS = {
    "write_paper": "write_paper",
    "review_started": "peer_review",
    "review_unavailable": "peer_review",
    "bad_faith_review": "bad_faith_review",
    "idle": "idle",
}


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


def print_choice_breakdown(env: Environment, history: History):
    """Show, per cohort, the share of decisions that were write/peer/bad-faith.

    Good = HeuristicAgents, bad = BadFaithAgents. Continuations of an in-progress
    good-faith review are not counted as fresh choices (see ``DECISION_LABELS``).
    """
    label_to_group = {
        agent.name: ("bad" if isinstance(agent, BadFaithAgent) else "good")
        for agent in env.agents
    }
    group_counts = Counter(label_to_group.values())
    tallies = {group: Counter() for group in group_counts}

    for _, agent_label, kind, _ in history.actions:
        decision = DECISION_LABELS.get(kind)
        group = label_to_group.get(agent_label)
        if decision is None or group is None:
            continue
        tallies[group][decision] += 1

    print("\nChoice breakdown (share of decisions)")
    for group in sorted(group_counts):
        counter = tallies[group]
        total = sum(counter.values())
        print(f"- {group} agents ({group_counts[group]}):")
        if total == 0:
            print("    no decisions recorded")
            continue
        for decision in ("write_paper", "peer_review", "bad_faith_review", "idle"):
            count = counter.get(decision, 0)
            if count:
                print(f"    {decision}: {count / total:.1%} ({count})")


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


def build_simulation(history: History, *, num_agents: int = NUM_AGENTS, num_bad_agents: int =   NUM_BAD_AGENTS, seed: int = 7) -> Environment:
    """
    Construct a new simulation
    """
    random.seed(seed)
    Agent.all_papers = []

    agents = [
        HeuristicAgent(intrinsic_talent=1.0, forecast_horizon_days=30, name=f"Agent {i}")
        for i in range(1, num_agents + 1)
    ]
    for i in range(num_bad_agents):
        agents.append(BadFaithAgent(intrinsic_talent=1.0, forecast_horizon_days=30, name=f"Bad Agent {i}"))

        
    seed_initial_papers(agents)

    return Environment(
        agents=agents,
        papers=Agent.all_papers,
        forecast_horizon_days=30,
        history=history,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the venture simulation. By default, after the run it "
        "asks whether to save it to the docs/ gallery and what to call it."
    )
    parser.add_argument(
        "--name",
        metavar="TITLE",
        help="Save with this title and skip the prompt (for scripting).",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip the prompt and do not save the run.",
    )
    return parser.parse_args(argv)


def archive_run(history: History, title: str | None) -> None:
    from export_run import export_run

    run_id = export_run(
        history,
        config={
            "num_agents": NUM_AGENTS,
            "num_bad_agents": NUM_BAD_AGENTS,
            "num_days": NUM_DAYS,
            "seed": 7,
        },
        title=title,
    )
    print(f"\nArchived run to docs/data/{run_id}/ (visible in the gallery).")
    print("Publish it with: "
          "git add docs/data && git commit -m 'Add run' && git push")


def prompt_and_archive(history: History) -> None:
    """Ask whether to save this run and, if so, what to title it."""
    try:
        answer = input("\nSave this run to the gallery? [y/N]: ").strip().lower()
    except EOFError:  # non-interactive (piped/no TTY): default to not saving
        print("Not archived (no interactive input).")
        return

    if answer not in ("y", "yes"):
        print("Not archived.")
        return

    try:
        name = input("Name this run (leave blank for an auto name): ").strip()
    except EOFError:
        name = ""
    archive_run(history, name or None)


def main(argv=None):
    args = parse_args(argv)

    history = History()
    env = build_simulation(history)
    for _ in range(NUM_DAYS):
        env.agentact()
        env.nextstep()

    print_summary(env, history)
    print_choice_breakdown(env, history)
    save_outputs(history)

    if args.no_archive:
        print("\nNot archived to the gallery (--no-archive).")
    elif args.name is not None:
        archive_run(history, args.name)
    else:
        prompt_and_archive(history)


if __name__ == "__main__":
    main()
