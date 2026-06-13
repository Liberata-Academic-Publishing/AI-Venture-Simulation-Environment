from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from dataclasses import asdict

from Agent import Agent
from config import SIM, default_policy_path
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from Paper import Paper
from QLearningAgent import QLearningAgent, make_backend
from RandomAgent import ProbabilisticDiscreteAgent, RandomAgent

# Defaults come from config.py; CLI flags override them at runtime.
NUM_AGENTS = SIM.num_heuristic_agents
NUM_TIMESTEPS = SIM.num_timesteps
NUM_RL_AGENTS = SIM.num_rl_agents
NUM_RANDOM_AGENTS = SIM.num_random_agents
NUM_PROBABILISTIC_AGENTS = SIM.num_probabilistic_agents
OUTPUT_DIR = SIM.output_dir


def _talent_for(index: int, count: int) -> float:
    """Spread talents across RL agents (inert until the sim uses talent)."""
    if count <= 1:
        return SIM.talent_min
    frac = index / (count - 1)
    return SIM.talent_min + frac * (SIM.talent_max - SIM.talent_min)


# Map raw action kinds to the decision an agent actively made on its turn.
DECISION_LABELS = {
    "write_paper": "write_paper",
    "bad_faith_review": "bad_faith_review",
    "good_faith_review": "good_faith_review",
    "review_started": "start_review",
    "review_continued": "continue_review",
    "review_finished_write": "finish_and_write",
    "review_finished_peer_review": "finish_and_review",
    "idle": "idle",
}


def seed_initial_papers(agents: list[Agent]):
    """Seed starting papers per SimConfig (listed on the market from timestep 1)."""
    index = 0
    for agent in agents:
        for _ in range(SIM.init_papers_per_agent):
            index += 1
            paper = Paper(
                author=agent,
                quality=agent.intrinsic_talent,
                current_ac=random.uniform(SIM.init_ac_min, SIM.init_ac_max),
                market_listed=True,
            )
            paper.title = f"Paper {index}"
            Agent.all_papers.append(paper)


def print_summary(env: Environment, history: History):
    print(f"\nSimulation finished after {env.timestep} timesteps")
    print(f"Agents: {len(env.agents)}")
    print(f"Papers: {len(env.papers)}")
    print(f"Review paradigm: {env.review_paradigm}")

    reviewed = sum(1 for p in env.papers if p.reviewed)
    on_market = sum(1 for p in env.papers if getattr(p, "review_available", False))
    in_review = sum(1 for p in env.papers if p.review_in_progress_by is not None)
    qualities = [p.quality for p in env.papers]
    capitals = [a.academic_capital for a in env.agents]
    from History import gini

    print("\nMarketplace overview")
    print(f"- papers reviewed: {reviewed}/{len(env.papers)}")
    print(
        f"- good-faith reviews: "
        f"{int(history.scalars.get('good_faith_reviews', [0])[-1]) if history.timesteps else 0}"
    )
    print(
        f"- bad-faith reviews: "
        f"{int(history.scalars.get('bad_faith_reviews', [0])[-1]) if history.timesteps else 0}"
    )
    print(f"- papers on market (unclaimed): {on_market}")
    print(f"- papers in review right now: {in_review}")
    if qualities:
        print(
            f"- paper quality: mean={sum(qualities) / len(qualities):.2f}, "
            f"min={min(qualities):.2f}, max={max(qualities):.2f}"
        )
    if capitals:
        print(f"- capital Gini (inequality): {gini(capitals):.3f}")

    reviewers = sorted(
        (a for a in env.agents if a.completed_review_count > 0),
        key=lambda a: a.peer_review_history,
        reverse=True,
    )
    if reviewers:
        print("\nTop reviewers (by reputation = AC earned per review)")
        for agent in reviewers[:5]:
            print(
                f"- {agent.name}: reputation={agent.peer_review_history:.2f} "
                f"over {agent.completed_review_count} reviews"
            )

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
        if paper.review_available:
            status = "on market"
        elif paper.reviewed:
            status = "reviewed"
        elif paper.review_in_progress_by is not None:
            status = "in review"
        else:
            status = "unlisted"
        print(
            f"- {title}: author={author_name}, quality={paper.quality:.2f}, "
            f"AC={paper.current_ac:.2f}, rate={paper.accrual_rate:.2f}, "
            f"status={status}, reviewer shares={reviewer_text}"
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
        "bad_faith_review",
        "good_faith_review",
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
    "summary": "Overview dashboard (capital, inequality, market, quality, reputation)",
    "agent_capital": "Academic capital per agent over time",
    "system_aggregates": "Total/mean/max capital with the inequality (Gini) index",
    "marketplace_activity": "Papers on market vs cumulative reviews completed",
    "paper_quality_vs_ac": "Paper quality vs accrued capital (reviewed or not)",
    "review_reputation": "Reviewer reputation (AC earned per review) over time",
    "action_mix": "What every agent did each timestep (stacked bars)",
    "choice_breakdown": "Agent decisions (write / review / finish)",
    "review_effort_histogram": "Completed peer reviews by effort level",
    "writing_effort_distribution": "Total paper-writing effort by agent",
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


def save_outputs(history: History, *, show: bool = False, open_charts: bool = False):
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


def build_rl_agents(
    count: int,
    *,
    backend_kind: str,
    policy_path: str | None,
    freeze: bool,
) -> list[QLearningAgent]:
    """Create independent RL agents (one private backend each).

    Each starts blank or, if ``policy_path`` points at an existing file, from
    that saved baseline (then they diverge via their own online learning).
    ``run_simulation.py`` never writes policies back.
    """
    if count <= 0:
        return []

    loaded = bool(policy_path) and os.path.exists(policy_path)
    if policy_path and not loaded:
        print(f"RL: policy {policy_path} not found; starting from scratch.")

    agents: list[QLearningAgent] = []
    for i in range(count):
        backend = make_backend(backend_kind)
        if loaded:
            try:
                backend.load(policy_path)
            except (ValueError, EOFError, OSError, KeyError):
                # Policies from before the single-review overhaul are incompatible.
                if i == 0:
                    print(f"RL: policy {policy_path} is incompatible; using scratch.")
                loaded = False
                backend = make_backend(backend_kind)
        agents.append(
            QLearningAgent(
                intrinsic_talent=_talent_for(i, count),
                forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
                name=f"RL Agent {i + 1}",
                backend=backend,
                epsilon=0.0 if freeze else SIM.rl_epsilon,
                learning=not freeze,
            )
        )

    source = (
        f"loaded baseline {policy_path}" if loaded else "starting from scratch"
    )
    mode = "frozen (greedy)" if freeze else "learning online"
    print(f"RL: {count} independent {backend_kind} agents, {source}, {mode}.")
    return agents


def build_random_agents(count: int) -> list[RandomAgent]:
    if count <= 0:
        return []
    return [
        RandomAgent(intrinsic_talent=1.0, name=f"Random Agent {i + 1}")
        for i in range(count)
    ]


def build_probabilistic_agents(count: int) -> list[ProbabilisticDiscreteAgent]:
    if count <= 0:
        return []
    return [
        ProbabilisticDiscreteAgent(
            intrinsic_talent=1.0,
            name=f"Probabilistic Agent {i + 1}",
        )
        for i in range(count)
    ]


def build_simulation(
    history: History,
    *,
    num_agents: int = NUM_AGENTS,
    seed: int = SIM.seed,
    rl_agents: int = NUM_RL_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    rl_backend: str = SIM.rl_backend,
    rl_policy_path: str | None = None,
    rl_freeze: bool = False,
    review_paradigm: str = SIM.review_paradigm,
) -> Environment:
    """Construct a simulation of heuristics plus independent RL agents."""
    random.seed(seed)
    Agent.all_papers = []

    agents: list[Agent] = [
        HeuristicAgent(intrinsic_talent=1.0,
                       forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
                       name=f"Agent {i}")
        for i in range(1, num_agents + 1)
    ]
    agents.extend(build_random_agents(random_agents))
    agents.extend(build_probabilistic_agents(probabilistic_agents))
    agents.extend(
        build_rl_agents(
            rl_agents,
            backend_kind=rl_backend,
            policy_path=rl_policy_path,
            freeze=rl_freeze,
        )
    )

    seed_initial_papers(agents)

    return Environment(
        agents=agents,
        papers=Agent.all_papers,
        forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
        review_paradigm=review_paradigm,
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
        "--open",
        action="store_true",
        help="Open the summary chart in your default image viewer after the run.",
    )
    parser.add_argument(
        "--rl-agents", dest="rl_agents", type=int, default=NUM_RL_AGENTS,
        metavar="N", help="Number of RL agents (0 disables RL).",
    )
    parser.add_argument(
        "--random-agents", dest="random_agents", type=int,
        default=NUM_RANDOM_AGENTS, metavar="N",
        help="Number of random control agents.",
    )
    parser.add_argument(
        "--probabilistic-agents", dest="probabilistic_agents", type=int,
        default=NUM_PROBABILISTIC_AGENTS, metavar="N",
        help="Number of discrete-only probability agents.",
    )
    parser.add_argument(
        "--review-paradigm", dest="review_paradigm",
        choices=["continuous", "discrete"], default=SIM.review_paradigm,
        help="Review action paradigm for the whole simulation run.",
    )
    parser.add_argument(
        "--rl-backend", dest="rl_backend", choices=["tabular", "linear"],
        default=SIM.rl_backend, help="Q backend for the RL agents.",
    )
    parser.add_argument(
        "--rl-from-scratch", dest="rl_from_scratch", action="store_true",
        help="Start RL agents from a blank policy instead of the saved baseline.",
    )
    parser.add_argument(
        "--rl-policy", dest="rl_policy", metavar="PATH", default=None,
        help="Explicit baseline policy path (overrides the default baseline).",
    )
    parser.add_argument(
        "--rl-freeze", dest="rl_freeze", action="store_true",
        help="Run RL agents greedily with no online learning.",
    )
    return parser.parse_args(argv)


def build_run_config(
    *,
    rl_agents: int = NUM_RL_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    rl_backend: str = SIM.rl_backend,
    review_paradigm: str = SIM.review_paradigm,
) -> dict:
    """Full SimConfig snapshot for this run, with runtime overrides applied.

    Dumps every SimConfig field so the gallery can display the complete set of
    simulation variables, then patches in the values that CLI flags may have
    changed for this run. ``num_agents`` is kept as an alias of
    ``num_heuristic_agents`` for backward compatibility with older gallery data.
    """
    config = asdict(SIM)
    config["num_rl_agents"] = rl_agents
    config["num_random_agents"] = random_agents
    config["num_probabilistic_agents"] = probabilistic_agents
    config["rl_backend"] = rl_backend
    config["review_paradigm"] = review_paradigm
    config["num_agents"] = config["num_heuristic_agents"]
    # Aliases so the static gallery (which also reads pre-overhaul runs) keeps
    # rendering the time-unit config fields under their old names.
    config["num_days"] = config["num_timesteps"]
    config["forecast_horizon_days"] = config["forecast_horizon_timesteps"]
    config["review_effort_per_day"] = config["review_effort_per_timestep"]
    return config


def archive_run(
    history: History,
    title: str | None,
    *,
    rl_agents: int = NUM_RL_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    rl_backend: str = SIM.rl_backend,
    review_paradigm: str = SIM.review_paradigm,
) -> None:
    from export_run import export_run

    run_id = export_run(
        history,
        config=build_run_config(
            rl_agents=rl_agents,
            random_agents=random_agents,
            probabilistic_agents=probabilistic_agents,
            rl_backend=rl_backend,
            review_paradigm=review_paradigm,
        ),
        title=title,
    )
    print(f"\nArchived run to docs/data/{run_id}/ (visible in the gallery).")
    print("Publish it with: "
          "git add docs/data && git commit -m 'Add run' && git push")


def prompt_and_archive(
    history: History,
    *,
    rl_agents: int = NUM_RL_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    rl_backend: str = SIM.rl_backend,
    review_paradigm: str = SIM.review_paradigm,
) -> None:
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
    archive_run(
        history,
        name or None,
        rl_agents=rl_agents,
        random_agents=random_agents,
        probabilistic_agents=probabilistic_agents,
        rl_backend=rl_backend,
        review_paradigm=review_paradigm,
    )


def main(argv=None):
    args = parse_args(argv)

    rl_policy_path = args.rl_policy
    if rl_policy_path is None and not args.rl_from_scratch:
        rl_policy_path = default_policy_path(args.rl_backend)

    history = History()
    env = build_simulation(
        history,
        rl_agents=args.rl_agents,
        random_agents=args.random_agents,
        probabilistic_agents=args.probabilistic_agents,
        rl_backend=args.rl_backend,
        rl_policy_path=rl_policy_path,
        rl_freeze=args.rl_freeze,
        review_paradigm=args.review_paradigm,
    )
    for _ in range(NUM_TIMESTEPS):
        env.run_timestep()

    print_summary(env, history)
    print_choice_breakdown(history)
    save_outputs(history, show=args.show, open_charts=args.open)

    if args.no_archive:
        print("\nNot archived to the gallery (--no-archive).")
    elif args.name is not None:
        archive_run(
            history,
            args.name,
            rl_agents=args.rl_agents,
            random_agents=args.random_agents,
            probabilistic_agents=args.probabilistic_agents,
            rl_backend=args.rl_backend,
            review_paradigm=args.review_paradigm,
        )
    else:
        prompt_and_archive(
            history,
            rl_agents=args.rl_agents,
            random_agents=args.random_agents,
            probabilistic_agents=args.probabilistic_agents,
            rl_backend=args.rl_backend,
            review_paradigm=args.review_paradigm,
        )


if __name__ == "__main__":
    main()
