from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter

from Agent import Agent
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from Paper import Paper

NUM_AGENTS = 20
NUM_DAYS = 200
OUTPUT_DIR = "runs"

# Map raw action kinds to the decision an agent actively made on its turn.
# Auto-continued locked reviews are follow-through, not fresh choices.
DECISION_LABELS = {
    "write_paper": "write_paper",
    "review_started": "start_review",
    "review_continued": "continue_review",
    "review_finished_write": "finish_and_write",
    "review_finished_peer_review": "finish_and_review",
    "review_unavailable": "start_review",
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


def print_choice_breakdown(history: History):
    """Show the share of top-level agent decisions (see ``DECISION_LABELS``)."""
    tallies: Counter[str] = Counter()

    for _, _, kind, _ in history.actions:
        decision = DECISION_LABELS.get(kind)
        if decision is not None:
            tallies[decision] += 1

    print("\nChoice breakdown (share of decisions)")
    total = sum(tallies.values())
    if total == 0:
        print("- no decisions recorded")
        return
    for decision in (
        "write_paper",
        "start_review",
        "continue_review",
        "finish_and_write",
        "finish_and_review",
        "idle",
    ):
        count = tallies.get(decision, 0)
        if count:
            print(f"- {decision}: {count / total:.1%} ({count})")


CHART_DESCRIPTIONS = {
    "summary": "Overview dashboard (review effort, actions, choices)",
    "action_mix": "What every agent did each day (stacked bars)",
    "choice_breakdown": "Agent decisions (write / review / finish)",
    "review_effort_histogram": "Completed peer reviews by effort level",
    "review_effort_scatter": "Completed reviews: day vs effort",
    "review_behavior": "Cumulative completed peer reviews",
    "paper_ac": "Accrued capital per paper over time",
}


def open_chart(path: str) -> None:
    """Open a saved chart with the OS default image viewer."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


def save_outputs(history: History, *, show: bool = False, open_charts: bool = True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = history.to_csv(os.path.join(OUTPUT_DIR, "history.csv"))
    json_path = history.to_json(os.path.join(OUTPUT_DIR, "history.json"))
    print(f"\nWrote time-series to {csv_path} and {json_path}")

    try:
        import visualize
    except ImportError as exc:
        print(f"Skipping charts ({exc}).")
        print("Install matplotlib with: python -m pip install matplotlib")
        print(f"Open {csv_path} in a spreadsheet to plot it instead.")
        return

    paths = visualize.plot_all(history, OUTPUT_DIR, show=show)
    print("\nWrote charts to the runs/ folder:")
    for name, path in paths.items():
        description = CHART_DESCRIPTIONS.get(name, name)
        print(f"- {path}  ({description})")

    summary_path = paths.get("summary")
    if open_charts and summary_path and os.path.exists(summary_path):
        print(f"\nOpening summary chart: {summary_path}")
        open_chart(summary_path)
    elif summary_path:
        print(f"\nView the summary chart at: {os.path.abspath(summary_path)}")


def build_simulation(history: History, *, num_agents: int = NUM_AGENTS, seed: int = 7) -> Environment:
    """Construct a new simulation with a single agent type."""
    random.seed(seed)
    Agent.all_papers = []

    agents = [
        HeuristicAgent(intrinsic_talent=1.0, forecast_horizon_days=30, name=f"Agent {i}")
        for i in range(1, num_agents + 1)
    ]

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
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pop up matplotlib chart windows after the run (in addition to saving PNGs).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the summary chart in your default image viewer.",
    )
    return parser.parse_args(argv)


def archive_run(history: History, title: str | None) -> None:
    from export_run import export_run

    run_id = export_run(
        history,
        config={
            "num_agents": NUM_AGENTS,
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
    print_choice_breakdown(history)
    save_outputs(history, show=args.show, open_charts=not args.no_open)

    if args.no_archive:
        print("\nNot archived to the gallery (--no-archive).")
    elif args.name is not None:
        archive_run(history, args.name)
    else:
        prompt_and_archive(history)


if __name__ == "__main__":
    main()
