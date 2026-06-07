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

if TYPE_CHECKING:
    from History import History

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "visualize requires matplotlib. Install it with: pip install matplotlib"
    ) from exc

ACTION_COLORS = {
    "write_paper": "#60a5fa",
    "review_started": "#4ade80",
    "review_continued": "#22c55e",
    "review_auto_continued": "#15803d",
    "review_finished_write": "#f59e0b",
    "review_finished_peer_review": "#a855f7",
    "review_stopped": "#16a34a",
    "review_unavailable": "#a78bfa",
    "idle": "#6b7280",
}

# Top-level decisions (matches run_simulation.DECISION_LABELS).
DECISION_LABELS = {
    "write_paper": "write_paper",
    "review_started": "start_review",
    "review_continued": "continue_review",
    "review_finished_write": "finish_and_write",
    "review_finished_peer_review": "finish_and_review",
    "review_unavailable": "start_review",
    "idle": "idle",
}

DECISION_COLORS = {
    "write_paper": "#60a5fa",
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


def plot_agent_capital(history: "History", path: str | None = None, show: bool = False):
    """One line per agent: academic capital over time (the core view)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, series in history.agent_capital.items():
        ax.plot(history.days, series, linewidth=1.2, label=label)
    ax.set_xlabel("Day")
    ax.set_ylabel("Academic capital")
    ax.set_title("Academic capital per agent over time")
    if 0 < len(history.agent_capital) <= 30:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
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
    ax.set_xlabel("Day")
    ax.set_ylabel("Academic capital")
    ax.set_title("Academic capital per agent over time (by group)")
    if ordered_groups:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_system_aggregates(history: "History", path: str | None = None, show: bool = False):
    """Total / mean / max capital, with the Gini inequality index on a twin axis."""
    fig, ax = plt.subplots(figsize=(11, 6))
    scalars = history.scalars
    for key in ("total_capital", "mean_capital", "max_capital"):
        if key in scalars:
            ax.plot(history.days, scalars[key], label=key)
    ax.set_xlabel("Day")
    ax.set_ylabel("Academic capital")
    ax.set_title("System capital aggregates")
    ax.legend(loc="upper left")
    if "capital_gini" in scalars:
        twin = ax.twinx()
        twin.plot(
            history.days,
            scalars["capital_gini"],
            color="black",
            linestyle="--",
            label="capital_gini",
        )
        twin.set_ylabel("Gini (inequality)")
        twin.legend(loc="lower right")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_review_behavior(history: "History", path: str | None = None, show: bool = False):
    """Cumulative good- vs bad-faith reviews (and paper count) over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    scalars = history.scalars
    for key in ("completed_peer_reviews", "num_papers"):
        if key in scalars:
            ax.plot(history.days, scalars[key], label=key)
    ax.set_xlabel("Day")
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
    ax.set_xlabel("Day")
    ax.set_ylabel("Accrued capital (AC)")
    ax.set_title(f"Per-paper AC over time ({len(history.paper_ac)} papers)")
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
    return [(day, effort) for day, _, _, effort in history.completed_reviews]


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
    ax.set_xlabel("Review effort")
    ax.set_ylabel("Completed peer reviews")
    ax.set_xticks(efforts)


def _draw_effort_scatter(ax, history: "History") -> None:
    points = _completed_reviews(history)
    if not points:
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return
    days, efforts = zip(*points)
    ax.scatter(days, efforts, alpha=0.65, s=28, color="#a855f7", edgecolors="#4c1d95")
    ax.set_xlabel("Day")
    ax.set_ylabel("Review effort")


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
    ax.set_xlabel("Day")
    ax.set_ylabel("Actions per day")
    ax.legend(fontsize=8, ncol=2, loc="upper left")


def plot_action_mix_over_time(
    history: "History", path: str | None = None, show: bool = False
):
    """Stacked daily counts of every action kind (what agents did each day)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _stacked_action_bars(ax, history)
    ax.set_title("What agents did each day (stacked action counts)")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_choice_breakdown(
    history: "History", path: str | None = None, show: bool = False
):
    """Bar chart of top-level agent decisions."""
    tallies = _choice_tallies(history)
    decisions = (
        "write_paper",
        "start_review",
        "continue_review",
        "finish_and_write",
        "finish_and_review",
        "idle",
    )
    labels = [d for d in decisions if tallies.get(d, 0)]
    values = [tallies[d] for d in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        range(len(labels)),
        values,
        color=[DECISION_COLORS.get(label, "#94a3b8") for label in labels],
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([label.replace("_", " ") for label in labels], rotation=20, ha="right")
    ax.set_ylabel("Decision count")
    ax.set_title("Agent choices (write / review / finish)")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_run_summary(history: "History", path: str | None = None, show: bool = False):
    """Single-page dashboard: review effort, actions, and choice breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Simulation summary", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    _draw_effort_histogram(ax, history)
    ax.set_title("Completed peer reviews by effort")

    ax = axes[0, 1]
    _draw_effort_scatter(ax, history)
    ax.set_title("Completed peer reviews over time")

    ax = axes[1, 0]
    _stacked_action_bars(ax, history)
    ax.set_title("Daily action mix")

    ax = axes[1, 1]
    tallies = _choice_tallies(history)
    decisions = (
        "write_paper",
        "start_review",
        "continue_review",
        "finish_and_write",
        "finish_and_review",
        "idle",
    )
    labels = [d for d in decisions if tallies.get(d, 0)]
    values = [tallies[d] for d in labels]
    ax.bar(
        range(len(labels)),
        values,
        color=[DECISION_COLORS.get(label, "#94a3b8") for label in labels],
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([label.replace("_", " ") for label in labels], fontsize=7, rotation=25, ha="right")
    ax.set_title("Agent choices")
    ax.set_ylabel("Count")

    fig.tight_layout()
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
        "action_mix": plot_action_mix_over_time(
            history, os.path.join(outdir, "action_mix.png"), show=show
        ),
        "choice_breakdown": plot_choice_breakdown(
            history, os.path.join(outdir, "choice_breakdown.png"), show=show
        ),
        "review_effort_histogram": plot_review_effort_histogram(
            history, os.path.join(outdir, "review_effort_histogram.png"), show=show
        ),
        "review_effort_scatter": plot_review_effort_scatter(
            history, os.path.join(outdir, "review_effort_scatter.png"), show=show
        ),
        "writing_effort_distribution": plot_writing_effort_distribution(
            history,
            os.path.join(outdir, "writing_effort_distribution.png"),
            show=show,
        ),
        "review_behavior": plot_review_behavior(
            history, os.path.join(outdir, "review_behavior.png"), show=show
        ),
        "paper_ac": plot_paper_ac(
            history, os.path.join(outdir, "paper_ac.png"), show=show
        ),
    }
