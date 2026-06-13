"""Matplotlib charts for a simulation ``History``.

This module is the *only* part that needs matplotlib; the recorder/exports in
``History`` stay dependency-free. Importing this module raises a clear error if
matplotlib is missing, so callers can ``try: import visualize`` and fall back to
the CSV/JSON export.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import TYPE_CHECKING

from Paper import MIN_REVIEW_EFFORT_THRESHOLD

if TYPE_CHECKING:
    from History import History

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "visualize requires matplotlib. Install it with: pip install matplotlib"
    ) from exc

ACTION_COLORS = {
    "write_paper": "#60a5fa",
    "bad_faith_review": "#f87171",
    "good_faith_review": "#16a34a",
    "review_started": "#4ade80",
    "review_continued": "#22c55e",
    "review_finished_write": "#f59e0b",
    "review_finished_peer_review": "#a855f7",
    "review_stopped": "#16a34a",
    "review_unavailable": "#a78bfa",
    "idle": "#6b7280",
}

# Top-level decisions (matches run_simulation.DECISION_LABELS).
DECISION_LABELS = {
    "write_paper": "write_paper",
    "bad_faith_review": "bad_faith_review",
    "good_faith_review": "good_faith_review",
    "review_started": "start_review",
    "review_continued": "continue_review",
    "review_finished_write": "finish_and_write",
    "review_finished_peer_review": "finish_and_review",
    "review_unavailable": "start_review",
    "idle": "idle",
}

DECISION_COLORS = {
    "write_paper": "#60a5fa",
    "bad_faith_review": "#f87171",
    "good_faith_review": "#16a34a",
    "start_review": "#4ade80",
    "continue_review": "#22c55e",
    "finish_and_write": "#f59e0b",
    "finish_and_review": "#a855f7",
    "idle": "#6b7280",
}


def _finish(fig, path: str | None, show: bool) -> str | None:
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _draw_agent_capital(ax, history: "History", legend: bool = True) -> None:
    if not history.agent_capital:
        ax.text(0.5, 0.5, "No agent capital recorded", ha="center", va="center")
        ax.set_axis_off()
        return
    for label, series in history.agent_capital.items():
        ax.plot(history.days, series, linewidth=1.2, label=label)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Academic capital")
    if legend and 0 < len(history.agent_capital) <= 30:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)


def plot_agent_capital(history: "History", path: str | None = None, show: bool = False):
    """One line per agent: academic capital over time (the core view)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_agent_capital(ax, history)
    ax.set_title("Academic capital per agent over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_agent_capital_by_group(
    history: "History", path: str | None = None, show: bool = False
):
    """Academic capital over time, with one color per agent class.

    Like ``plot_agent_capital`` but colors agents by their class with a single
    legend entry per group.
    """
    groups = history.agent_groups
    ordered_groups: list[str] = []
    for label in history.agent_capital:
        group = groups.get(label, "Agent")
        if group not in ordered_groups:
            ordered_groups.append(group)

    cmap = plt.get_cmap("tab10")
    color_for = {group: cmap(i % 10) for i, group in enumerate(ordered_groups)}

    fig, ax = plt.subplots(figsize=(11, 6))
    legended: set[str] = set()
    for label, series in history.agent_capital.items():
        group = groups.get(label, "Agent")
        legend_label = group if group not in legended else None
        legended.add(group)
        ax.plot(
            history.days,
            series,
            linewidth=1.2,
            alpha=0.8,
            color=color_for[group],
            label=legend_label,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Academic capital")
    ax.set_title("Academic capital per agent over time (by group)")
    if ordered_groups:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_system_aggregates(ax, history: "History") -> None:
    scalars = history.scalars
    for key in ("total_capital", "mean_capital", "max_capital"):
        if key in scalars:
            ax.plot(history.days, scalars[key], label=key.replace("_", " "))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Academic capital")
    ax.legend(loc="upper left", fontsize=8)
    if "capital_gini" in scalars:
        twin = ax.twinx()
        twin.plot(
            history.days,
            scalars["capital_gini"],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="capital gini",
        )
        twin.set_ylabel("Gini (inequality)")
        twin.set_ylim(0.0, 1.0)
        twin.legend(loc="lower right", fontsize=8)


def plot_system_aggregates(history: "History", path: str | None = None, show: bool = False):
    """Total / mean / max capital, with the Gini inequality index on a twin axis."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_system_aggregates(ax, history)
    ax.set_title("System capital & inequality (Gini)")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_review_behavior(history: "History", path: str | None = None, show: bool = False):
    """Cumulative good- vs bad-faith reviews (and paper count) over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    scalars = history.scalars
    for key in (
        "good_faith_reviews",
        "bad_faith_reviews",
        "completed_peer_reviews",
        "num_papers",
    ):
        if key in scalars:
            ax.plot(history.days, scalars[key], label=key)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Review behavior over time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_paper_ac(history: "History", path: str | None = None, show: bool = False):
    """One thin line per paper: accrued capital (AC) over time (legend suppressed)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for series in history.paper_ac.values():
        ax.plot(history.days, series, linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Accrued capital (AC)")
    ax.set_title(f"Per-paper AC over time ({len(history.paper_ac)} papers)")
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_marketplace(ax, history: "History") -> None:
    """Supply (papers waiting on the market) vs demand (reviews completed)."""
    scalars = history.scalars
    handles = []
    if "papers_on_market" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_on_market"],
            color="#f59e0b",
            label="papers on market",
        )
        handles.append(line)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Papers on market", color="#b45309")
    ax.tick_params(axis="y", labelcolor="#b45309")

    if "completed_peer_reviews" in scalars:
        twin = ax.twinx()
        (line,) = twin.plot(
            history.days,
            scalars["completed_peer_reviews"],
            color="#a855f7",
            linestyle="--",
            linewidth=1.6,
            label="cumulative reviews",
        )
        twin.set_ylabel("Reviews completed", color="#7e22ce")
        twin.tick_params(axis="y", labelcolor="#7e22ce")
        handles.append(line)

    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8)


def plot_marketplace_activity(
    history: "History", path: str | None = None, show: bool = False
):
    """Market supply (papers listed) against cumulative reviews completed."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_marketplace(ax, history)
    ax.set_title("Review marketplace: supply vs reviews completed")
    fig.tight_layout()
    return _finish(fig, path, show)


def _paper_quality_ac_points(history: "History") -> list[tuple[float, float, bool]]:
    """(quality, final AC, reviewed?) for every paper that was tracked."""
    points: list[tuple[float, float, bool]] = []
    for label, series in history.paper_ac.items():
        quality = history.paper_quality.get(label)
        if quality is None or not series:
            continue
        points.append((quality, series[-1], history.paper_reviewed.get(label, False)))
    return points


def _draw_quality_vs_ac(ax, history: "History") -> None:
    points = _paper_quality_ac_points(history)
    if not points:
        ax.text(0.5, 0.5, "No paper data", ha="center", va="center")
        ax.set_axis_off()
        return
    reviewed = [(q, a) for q, a, r in points if r]
    plain = [(q, a) for q, a, r in points if not r]
    if plain:
        qs, acs = zip(*plain)
        ax.scatter(qs, acs, s=16, alpha=0.45, color="#94a3b8", label="not reviewed")
    if reviewed:
        qs, acs = zip(*reviewed)
        ax.scatter(
            qs, acs, s=26, alpha=0.8, color="#a855f7",
            edgecolors="#4c1d95", label="reviewed",
        )
    ax.set_xlabel("Paper quality")
    ax.set_ylabel("Final accrued capital (AC)")
    ax.legend(fontsize=8)


def plot_paper_quality_vs_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """Does quality pay off? Final AC vs quality, split by whether reviewed."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_quality_vs_ac(ax, history)
    ax.set_title("Paper quality vs accrued capital")
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_review_reputation(ax, history: "History") -> None:
    """Per-agent peer-review reputation over time, with the mean highlighted."""
    series_map = getattr(history, "agent_review_history", {})
    drew = False
    for series in series_map.values():
        if any(series):
            ax.plot(history.days, series, linewidth=0.8, alpha=0.4, color="#38bdf8")
            drew = True
    mean_series = history.scalars.get("mean_peer_review_history")
    if mean_series is not None:
        ax.plot(
            history.days, mean_series, linewidth=2.0, color="#0369a1",
            label="mean reputation",
        )
        drew = True
    if not drew:
        ax.text(0.5, 0.5, "No reviews completed", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Peer-review reputation\n(AC earned per review)")
    ax.legend(fontsize=8, loc="upper left")


def plot_review_reputation(
    history: "History", path: str | None = None, show: bool = False
):
    """Reviewer reputation (AC earned per completed review) over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_review_reputation(ax, history)
    ax.set_title("Reviewer reputation over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def _daily_action_counts(history: "History") -> dict[str, list[int]]:
    """Per-day counts for each raw action kind."""
    kinds = sorted({kind for _, _, kind, _ in history.actions})
    counts = {kind: [0] * len(history.days) for kind in kinds}
    day_index = {day: i for i, day in enumerate(history.days)}

    for day, _, kind, _ in history.actions:
        idx = day_index.get(day)
        if idx is not None:
            counts[kind][idx] += 1
    return counts


def _completed_reviews(history: "History") -> list[tuple[int, float]]:
    """Return (day, effort) for each completed peer review."""
    return [(row[0], row[3]) for row in history.completed_reviews]


def _writing_effort_by_agent(history: "History") -> dict[str, float]:
    """Total writing effort accumulated by each agent."""
    totals: dict[str, float] = {}
    for _, agent, effort, _ in getattr(history, "writing_efforts", []):
        totals[agent] = totals.get(agent, 0.0) + effort
    return totals


def _effort_histogram(history: "History") -> tuple[list[int], list[int]]:
    """Bin completed reviews by integer effort level."""
    counts: Counter[int] = Counter()
    for _, effort in _completed_reviews(history):
        counts[int(effort)] += 1
    if not counts:
        return [], []
    efforts = sorted(counts)
    return efforts, [counts[e] for e in efforts]


def _draw_effort_histogram(ax, history: "History") -> None:
    efforts, values = _effort_histogram(history)
    if not efforts:
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.bar(efforts, values, width=0.9, color="#60a5fa", edgecolor="#1e3a5f")
    ax.axvline(
        MIN_REVIEW_EFFORT_THRESHOLD,
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="reward threshold",
    )
    ax.set_xlabel("Review effort")
    ax.set_ylabel("Completed peer reviews")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
    ax.legend(fontsize=8)


def _draw_effort_scatter(ax, history: "History") -> None:
    points = _completed_reviews(history)
    if not points:
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return
    timesteps, efforts = zip(*points)
    ax.scatter(timesteps, efforts, alpha=0.65, s=28, color="#a855f7", edgecolors="#4c1d95")
    ax.axhline(
        MIN_REVIEW_EFFORT_THRESHOLD,
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="reward threshold",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Review effort")
    ax.legend(fontsize=8)


def plot_review_effort_histogram(
    history: "History", path: str | None = None, show: bool = False
):
    """Histogram of completed peer reviews by effort level."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_effort_histogram(ax, history)
    ax.set_title("Completed peer reviews by effort")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_review_effort_scatter(
    history: "History", path: str | None = None, show: bool = False
):
    """Scatter of each completed review: day vs effort invested."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_effort_scatter(ax, history)
    ax.set_title("Completed peer reviews over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_writing_effort_distribution(
    history: "History", path: str | None = None, show: bool = False
):
    """Histogram of total paper-writing effort accumulated by each agent."""
    totals = _writing_effort_by_agent(history)
    fig, ax = plt.subplots(figsize=(11, 6))
    if not totals:
        ax.text(0.5, 0.5, "No writing effort recorded", ha="center", va="center")
        ax.set_axis_off()
    else:
        values = list(totals.values())
        bins = max(1, min(20, len(values)))
        ax.hist(values, bins=bins, color="#60a5fa", edgecolor="#1e3a5f")
        ax.set_xlabel("Total writing effort per agent")
        ax.set_ylabel("Agents")
        ax.set_title("Distribution of paper-writing effort")
    fig.tight_layout()
    return _finish(fig, path, show)


def _choice_tallies(history: "History") -> dict[str, int]:
    """Count top-level decisions across all agents."""
    tallies: dict[str, int] = {}
    for _, _, kind, _ in history.actions:
        decision = DECISION_LABELS.get(kind)
        if decision is None:
            continue
        tallies[decision] = tallies.get(decision, 0) + 1
    return tallies


def _stacked_action_bars(ax, history: "History") -> None:
    counts = _daily_action_counts(history)
    if not counts:
        ax.text(0.5, 0.5, "No actions recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    bottom = [0] * len(history.days)
    for kind in sorted(
        counts,
        key=lambda name: list(ACTION_COLORS).index(name)
        if name in ACTION_COLORS
        else len(ACTION_COLORS),
    ):
        values = counts[kind]
        ax.bar(
            history.days,
            values,
            bottom=bottom,
            label=kind.replace("_", " "),
            color=ACTION_COLORS.get(kind, "#94a3b8"),
            width=0.9,
        )
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Actions per timestep")
    ax.legend(fontsize=8, ncol=2, loc="upper left")


def plot_action_mix_over_time(
    history: "History", path: str | None = None, show: bool = False
):
    """Stacked daily counts of every action kind (what agents did each day)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _stacked_action_bars(ax, history)
    ax.set_title("What agents did each timestep (stacked action counts)")
    fig.tight_layout()
    return _finish(fig, path, show)


_DECISION_ORDER = (
    "write_paper",
    "bad_faith_review",
    "good_faith_review",
    "start_review",
    "continue_review",
    "finish_and_write",
    "finish_and_review",
    "idle",
)


def _draw_choice_breakdown(ax, history: "History", fontsize: int = 9) -> None:
    tallies = _choice_tallies(history)
    labels = [d for d in _DECISION_ORDER if tallies.get(d, 0)]
    if not labels:
        ax.text(0.5, 0.5, "No decisions recorded", ha="center", va="center")
        ax.set_axis_off()
        return
    values = [tallies[d] for d in labels]
    total = sum(values)
    bars = ax.bar(
        range(len(labels)),
        values,
        color=[DECISION_COLORS.get(label, "#94a3b8") for label in labels],
    )
    for rect, value in zip(bars, values):
        ax.annotate(
            f"{value / total:.0%}",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            ha="center", va="bottom", fontsize=fontsize - 1,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [label.replace("_", " ") for label in labels],
        rotation=20, ha="right", fontsize=fontsize,
    )
    ax.set_ylabel("Decision count")


def plot_choice_breakdown(
    history: "History", path: str | None = None, show: bool = False
):
    """Bar chart of top-level agent decisions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_choice_breakdown(ax, history)
    ax.set_title("Agent choices (write / review / finish)")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_run_summary(history: "History", path: str | None = None, show: bool = False):
    """Single-page dashboard encapsulating the single-review marketplace run.

    Top row tells the outcome story (who accumulates capital, how unequal the
    system becomes, and how the review market behaves); the bottom row explains
    the mechanics (quality payoff, reviewer reputation, and how agents spend
    their timesteps).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    n_papers = len(history.paper_ac)
    horizon = history.timesteps[-1] if history.timesteps else 0
    fig.suptitle(
        f"Simulation summary — {len(history.agent_capital)} agents, "
        f"{n_papers} papers, {horizon} timesteps",
        fontsize=15, fontweight="bold",
    )

    ax = axes[0, 0]
    _draw_agent_capital(ax, history, legend=len(history.agent_capital) <= 12)
    ax.set_title("Academic capital per agent")

    ax = axes[0, 1]
    _draw_system_aggregates(ax, history)
    ax.set_title("System capital & inequality (Gini)")

    ax = axes[0, 2]
    _draw_marketplace(ax, history)
    ax.set_title("Review marketplace")

    ax = axes[1, 0]
    _draw_quality_vs_ac(ax, history)
    ax.set_title("Paper quality vs accrued capital")

    ax = axes[1, 1]
    _draw_review_reputation(ax, history)
    ax.set_title("Reviewer reputation over time")

    ax = axes[1, 2]
    _draw_choice_breakdown(ax, history, fontsize=8)
    ax.set_title("How agents spent their timesteps")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, path, show)


def plot_all(
    history: "History", outdir: str = "runs", *, show: bool = False
) -> dict[str, str]:
    """Render every chart into ``outdir`` and return {name: path}."""
    os.makedirs(outdir, exist_ok=True)
    return {
        "summary": plot_run_summary(
            history, os.path.join(outdir, "summary.png"), show=show
        ),
        "agent_capital": plot_agent_capital(
            history, os.path.join(outdir, "agent_capital.png"), show=show
        ),
        "system_aggregates": plot_system_aggregates(
            history, os.path.join(outdir, "system_aggregates.png"), show=show
        ),
        "marketplace_activity": plot_marketplace_activity(
            history, os.path.join(outdir, "marketplace_activity.png"), show=show
        ),
        "paper_quality_vs_ac": plot_paper_quality_vs_ac(
            history, os.path.join(outdir, "paper_quality_vs_ac.png"), show=show
        ),
        "review_reputation": plot_review_reputation(
            history, os.path.join(outdir, "review_reputation.png"), show=show
        ),
        "action_mix": plot_action_mix_over_time(
            history, os.path.join(outdir, "action_mix.png"), show=show
        ),
        "choice_breakdown": plot_choice_breakdown(
            history, os.path.join(outdir, "choice_breakdown.png"), show=show
        ),
        "review_effort_histogram": plot_review_effort_histogram(
            history, os.path.join(outdir, "review_effort_histogram.png"), show=show
        ),
        "writing_effort_distribution": plot_writing_effort_distribution(
            history,
            os.path.join(outdir, "writing_effort_distribution.png"),
            show=show,
        ),
        "paper_ac": plot_paper_ac(
            history, os.path.join(outdir, "paper_ac.png"), show=show
        ),
    }
