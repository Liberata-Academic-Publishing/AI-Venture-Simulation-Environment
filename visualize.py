"""Matplotlib charts for a simulation ``History``.

This module is the *only* part that needs matplotlib; the recorder/exports in
``History`` stay dependency-free. Importing this module raises a clear error if
matplotlib is missing, so callers can ``try: import visualize`` and fall back to
the CSV/JSON export.
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING, Any

from Paper import MIN_REVIEW_EFFORT_THRESHOLD

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


def _mean_completed_review_effort_series(history: "History") -> list[float]:
    """Cumulative mean effort of finished reviews at each timestep."""
    stored = history.scalars.get("mean_completed_review_effort")
    if stored is not None and len(stored) == len(history.days):
        return list(stored)

    by_day: dict[int, list[float]] = {}
    for ts, _, _, effort, _ in history.completed_reviews:
        by_day.setdefault(ts, []).append(float(effort))

    running_sum = 0.0
    running_count = 0
    series: list[float] = []
    for day in history.days:
        for effort in by_day.get(day, []):
            running_sum += effort
            running_count += 1
        series.append(running_sum / running_count if running_count else 0.0)
    return series


def _draw_mean_review_effort(ax, history: "History") -> None:
    """Running average effort invested in completed peer reviews over time."""
    series = _mean_completed_review_effort_series(history)
    if not any(series):
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(history.days, series, linewidth=2.0, color="#7c3aed", label="mean effort")
    ax.axhline(
        MIN_REVIEW_EFFORT_THRESHOLD,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"good-faith threshold ({MIN_REVIEW_EFFORT_THRESHOLD:g})",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean review effort (timesteps)")
    ax.legend(fontsize=8, loc="best")


def plot_mean_review_effort(
    history: "History", path: str | None = None, show: bool = False
):
    """Running mean effort of completed peer reviews over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_mean_review_effort(ax, history)
    ax.set_title("Average peer review effort over time")
    fig.tight_layout()
    return _finish(fig, path, show)


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


def plot_agent_group_comparison(
    history: "History", path: str | None = None, show: bool = False
):
    """Executive summary by agent type: mean capital and review behavior."""
    summary = history.agent_group_summary()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not summary:
        axes[0].text(0.5, 0.5, "No agent group data", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        return _finish(fig, path, show)

    groups = sorted(summary)
    mean_capital = [summary[g]["mean_final_capital"] for g in groups]
    good = [summary[g]["good_faith_reviews"] for g in groups]
    bad = [summary[g]["bad_faith_reviews"] for g in groups]
    x = range(len(groups))

    axes[0].bar(x, mean_capital, color="#60a5fa", edgecolor="#1e3a5f")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(groups, rotation=20, ha="right")
    axes[0].set_ylabel("Mean final AC")
    axes[0].set_title("Mean final capital by agent type")

    axes[1].bar(x, good, color="#16a34a", label="good faith")
    axes[1].bar(x, bad, bottom=good, color="#f87171", label="bad faith")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(groups, rotation=20, ha="right")
    axes[1].set_ylabel("Completed reviews")
    axes[1].set_title("Review outcomes by agent type")
    axes[1].legend(fontsize=8)

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
    """Supply (papers waiting/listed/claimed) vs demand (reviews completed)."""
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
    if "papers_listed_this_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_listed_this_timestep"],
            color="#22c55e",
            linewidth=1.2,
            alpha=0.85,
            label="papers listed this timestep",
        )
        handles.append(line)
    if "papers_claimed_this_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_claimed_this_timestep"],
            color="#ef4444",
            linewidth=1.2,
            alpha=0.85,
            label="papers claimed this timestep",
        )
        handles.append(line)
    if "papers_claimed_same_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_claimed_same_timestep"],
            color="#fb7185",
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label="cumulative instant claims",
        )
        handles.append(line)
    ax.set_xlabel("Timestep")
    # Use the bright line colors for the axis labels too, so they read on both
    # the light runs/ PNGs and the transparent dark-themed gallery PNGs.
    ax.set_ylabel("Papers on market", color="#f59e0b")
    ax.tick_params(axis="y", labelcolor="#f59e0b")

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
        twin.set_ylabel("Reviews completed", color="#a855f7")
        twin.tick_params(axis="y", labelcolor="#a855f7")
        handles.append(line)

    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8)


def plot_marketplace_activity(
    history: "History", path: str | None = None, show: bool = False
):
    """Market supply/listing/claim diagnostics against completed reviews."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_marketplace(ax, history)
    ax.set_title("Review marketplace: supply, claims, and reviews completed")
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


def _draw_writing_effort_vs_rate(ax, history: "History") -> None:
    """How much authors invested in each paper vs the base accrual rate it earned.

    Continuous mode: a paper's base rate approaches its quality ceiling as
    writing effort grows, so the cloud should trace a rising, saturating band.
    """
    efforts = getattr(history, "paper_writing_effort", {})
    rates = getattr(history, "paper_accrual_rate", {})
    points = [
        (efforts[label], rates[label])
        for label in efforts
        if label in rates
    ]
    if not points:
        ax.text(
            0.5, 0.5, "No writing-effort data\n(continuous mode only)",
            ha="center", va="center",
        )
        ax.set_axis_off()
        return
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=18, alpha=0.5, color="#0ea5e9", edgecolors="#075985")
    ax.set_xlabel("Writing effort invested (timesteps)")
    ax.set_ylabel("Base accrual rate at finish")


def plot_writing_effort_vs_rate(
    history: "History", path: str | None = None, show: bool = False
):
    """Per-paper writing effort vs the base accrual rate it locked in."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_writing_effort_vs_rate(ax, history)
    ax.set_title("Writing effort vs accrual rate (asymptote)")
    fig.tight_layout()
    return _finish(fig, path, show)


def _paper_writing_effort_over_time_points(
    history: "History",
) -> list[tuple[int, float, str]]:
    """Return (first-seen timestep, writing effort, author group) per paper."""
    efforts = getattr(history, "paper_writing_effort", {})
    first_seen = getattr(history, "paper_first_seen_timestep", {})
    authors = getattr(history, "paper_authors", {})
    groups = getattr(history, "agent_groups", {})
    paper_ac = getattr(history, "paper_ac", {})
    days = getattr(history, "days", [])

    points: list[tuple[int, float, str]] = []
    for paper, effort in efforts.items():
        timestep = first_seen.get(paper)
        if timestep is None:
            series = paper_ac.get(paper, [])
            for i, value in enumerate(series):
                if value > 0:
                    timestep = days[i] if i < len(days) else i
                    break
        if timestep is None:
            continue
        author = authors.get(paper)
        group = groups.get(author, "Agent")
        points.append((int(timestep), float(effort), group))
    return points


def _draw_paper_writing_effort_over_time(ax, history: "History") -> None:
    """Show whether papers are being published with more/less effort over time."""
    points = _paper_writing_effort_over_time_points(history)
    if not points:
        ax.text(
            0.5, 0.5, "No per-paper writing effort over time recorded",
            ha="center", va="center",
        )
        ax.set_axis_off()
        return

    points_by_group: dict[str, list[tuple[int, float]]] = {}
    for timestep, effort, group in points:
        points_by_group.setdefault(group, []).append((timestep, effort))

    cmap = plt.get_cmap("tab10")
    for i, group in enumerate(sorted(points_by_group)):
        xs, ys = zip(*points_by_group[group])
        ax.scatter(
            xs,
            ys,
            s=22,
            alpha=0.65,
            color=cmap(i % 10),
            edgecolors="#1e293b",
            linewidths=0.3,
            label=group,
        )
    ax.set_xlabel("Timestep paper appeared")
    ax.set_ylabel("Writing effort at publication")
    ax.legend(fontsize=8)


def plot_paper_writing_effort_over_time(
    history: "History", path: str | None = None, show: bool = False
):
    """Per-paper writing effort at publication, colored by author type."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_paper_writing_effort_over_time(ax, history)
    ax.set_title("Paper-writing effort over time")
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


def _completed_reviews(history: "History") -> list[tuple[int, str, float]]:
    """Return (day, reviewer label, effort) for each completed peer review."""
    return [(row[0], row[1], row[3]) for row in history.completed_reviews]


def _writing_effort_by_agent(history: "History") -> dict[str, float]:
    """Total writing effort accumulated by each agent."""
    totals: dict[str, float] = {}
    for _, agent, effort, _ in getattr(history, "writing_efforts", []):
        totals[agent] = totals.get(agent, 0.0) + effort
    return totals


def _draw_effort_histogram(
    ax, history: "History", threshold: float = MIN_REVIEW_EFFORT_THRESHOLD
) -> None:
    reviews = _completed_reviews(history)
    efforts = [effort for _, _, effort in reviews]
    if not efforts:
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return
    lo = math.floor(min(efforts))
    hi = math.ceil(max(efforts))
    efforts_by_group: dict[str, list[float]] = {}
    for _, agent, effort in reviews:
        group = history.agent_groups.get(agent, "Agent")
        efforts_by_group.setdefault(group, []).append(effort)

    bins = range(lo, hi + 2)
    if len(efforts_by_group) > 1:
        groups = sorted(efforts_by_group)
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i, _ in enumerate(groups)]
        ax.hist(
            [efforts_by_group[group] for group in groups],
            bins=bins,
            align="left",
            rwidth=0.9,
            stacked=True,
            color=colors,
            edgecolor="#1e3a5f",
            label=groups,
        )
    else:
        ax.hist(
            efforts,
            bins=bins,
            align="left",
            rwidth=0.9,
            color="#60a5fa",
            edgecolor="#1e3a5f",
        )
    # One tick per integer effort when the range is small; otherwise let
    # matplotlib thin them so wide-effort runs don't overlap their labels.
    if hi - lo <= 30:
        ax.set_xticks(range(lo, hi + 1))
    ax.axvline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="reward threshold",
    )
    ax.set_xlabel("Review effort")
    ax.set_ylabel("Completed peer reviews")
    ax.legend(fontsize=8)


def _draw_effort_scatter(
    ax, history: "History", threshold: float = MIN_REVIEW_EFFORT_THRESHOLD
) -> None:
    points = _completed_reviews(history)
    if not points:
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return
    timesteps = [day for day, _, _ in points]
    efforts = [effort for _, _, effort in points]
    ax.scatter(timesteps, efforts, alpha=0.65, s=28, color="#a855f7", edgecolors="#4c1d95")
    ax.axhline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="reward threshold",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Review effort")
    ax.legend(fontsize=8)


def plot_review_effort_histogram(
    history: "History",
    path: str | None = None,
    show: bool = False,
    *,
    threshold: float = MIN_REVIEW_EFFORT_THRESHOLD,
):
    """Histogram of completed peer reviews by effort level."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_effort_histogram(ax, history, threshold)
    ax.set_title("Completed peer reviews by effort")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_review_effort_scatter(
    history: "History",
    path: str | None = None,
    show: bool = False,
    *,
    threshold: float = MIN_REVIEW_EFFORT_THRESHOLD,
):
    """Scatter of each completed review: day vs effort invested."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_effort_scatter(ax, history, threshold)
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


def plot_paper_writing_effort_distribution(
    history: "History", path: str | None = None, show: bool = False
):
    """Histogram of writing effort across all papers (effort vs paper count)."""
    values = list(getattr(history, "paper_writing_effort", {}).values())
    fig, ax = plt.subplots(figsize=(11, 6))
    if not values:
        ax.text(
            0.5, 0.5, "No per-paper writing effort recorded\n(continuous mode only)",
            ha="center", va="center",
        )
        ax.set_axis_off()
    else:
        bins = max(1, min(20, len(values)))
        ax.hist(values, bins=bins, color="#60a5fa", edgecolor="#1e3a5f")
        ax.set_xlabel("Writing effort per paper")
        ax.set_ylabel("Papers")
        ax.set_title("Distribution of per-paper writing effort")
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

    Top row tells the outcome story (mean review effort over time, effort
    distribution, and how the review market behaves); the bottom row explains
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
    _draw_mean_review_effort(ax, history)
    ax.set_title("Average peer review effort over time")

    ax = axes[0, 1]
    _draw_effort_histogram(ax, history)
    ax.set_title("Completed peer reviews by effort")

    ax = axes[0, 2]
    _draw_marketplace(ax, history)
    ax.set_title("Review marketplace")

    ax = axes[1, 0]
    if getattr(history, "paper_writing_effort", {}):
        _draw_writing_effort_vs_rate(ax, history)
        ax.set_title("Writing effort vs accrual rate")
    else:
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



def _draw_episode_return(
    ax,
    episodes: list[int],
    returns: list[float],
    epsilon: list[float] | None = None,
) -> None:
    del epsilon  # kept for call-site compatibility; chart is raw returns only
    if not episodes or not returns:
        ax.text(0.5, 0.5, "No training episodes recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(episodes, returns, linewidth=1.5, color="#60a5fa")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean episode return (final AC)")


def plot_episode_return(
    episodes: list[int],
    returns: list[float],
    epsilon: list[float] | None = None,
    path: str | None = None,
    show: bool = False,
):
    """Episode return curve for RL training (mean final AC per episode)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_episode_return(ax, episodes, returns, epsilon)
    ax.set_title("Episode return")
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_avg_peer_review_time(
    ax,
    episodes: list[int],
    avg_review_times: list[float],
) -> None:
    if not episodes or not avg_review_times:
        ax.text(0.5, 0.5, "No peer review data recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(episodes, avg_review_times, linewidth=1.5, color="#34d399")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean peer review time (timesteps)")


def plot_avg_peer_review_time(
    episodes: list[int],
    avg_review_times: list[float],
    path: str | None = None,
    show: bool = False,
):
    """Average completed peer-review duration per training episode."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_avg_peer_review_time(ax, episodes, avg_review_times)
    ax.set_title("Average peer review time")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_avg_peer_review_time_from_log(
    log_path: str,
    png_path: str | None = None,
    *,
    show: bool = False,
) -> str | None:
    """Render ``avg_peer_review_time.png`` from a ``training_log.json`` file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as fh:
        log: list[dict[str, Any]] = json.load(fh)
    if not log or "mean_review_effort" not in log[0]:
        return None
    episodes = [int(row["episode"]) for row in log]
    avg_times = [float(row["mean_review_effort"]) for row in log]
    if png_path is None:
        png_path = os.path.join(os.path.dirname(log_path), "avg_peer_review_time.png")
    plot_avg_peer_review_time(episodes, avg_times, png_path, show=show)
    return png_path if os.path.exists(png_path) else None


def plot_episode_return_from_log(
    log_path: str,
    png_path: str | None = None,
    *,
    show: bool = False,
) -> str | None:
    """Render ``episode_return.png`` from a ``training_log.json`` file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as fh:
        log: list[dict[str, Any]] = json.load(fh)
    if not log:
        return None
    episodes = [int(row["episode"]) for row in log]
    returns = [float(row["mean_return"]) for row in log]
    epsilon = [float(row["epsilon"]) for row in log]
    if png_path is None:
        png_path = os.path.join(os.path.dirname(log_path), "episode_return.png")
    plot_episode_return(episodes, returns, epsilon, png_path, show=show)
    return png_path if os.path.exists(png_path) else None


# ---- static gallery charts (dark-themed, transparent PNGs) ----------------
#
# The committed gallery (docs/) shows the *static* charts as images instead of
# rebuilding them in the browser. They are rendered transparent with light text
# so they sit cleanly on the dark panel background (#161d24) and match the live
# Chart.js look closely. Only charts that have data for a given run are emitted,
# mirroring the front-end's per-panel ``setPanel`` gating, so simpler runs get
# fewer PNGs.
_DARK_RC = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "text.color": "#c4cdd5",
    "axes.labelcolor": "#c4cdd5",
    "axes.titlecolor": "#e6edf3",
    "axes.edgecolor": "#3a4450",
    "xtick.color": "#8b98a5",
    "ytick.color": "#8b98a5",
    "grid.color": "#1f272f",
    "legend.facecolor": "#161d24",
    "legend.edgecolor": "#30363d",
    "legend.framealpha": 0.85,
}


def render_episode_return_gallery(
    episodes: list[int],
    returns: list[float],
    epsilon: list[float] | None,
    outdir: str,
) -> str | None:
    """Render a dark-themed episode-return PNG for the static gallery."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "episode_return.png")
    with plt.rc_context(_DARK_RC):
        plot_episode_return(episodes, returns, epsilon, path)
    return path if os.path.exists(path) else None


def render_avg_peer_review_time_gallery(
    episodes: list[int],
    avg_review_times: list[float],
    outdir: str,
) -> str | None:
    """Render a dark-themed average peer-review-time PNG for the gallery."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "avg_peer_review_time.png")
    with plt.rc_context(_DARK_RC):
        plot_avg_peer_review_time(episodes, avg_review_times, path)
    return path if os.path.exists(path) else None


def _has_review_behavior(history: "History") -> bool:
    s = history.scalars
    return bool(s.get("num_papers") or s.get("completed_peer_reviews"))


def _has_marketplace(history: "History") -> bool:
    s = history.scalars
    return bool(s.get("papers_on_market") or s.get("completed_peer_reviews"))


def _has_reputation(history: "History") -> bool:
    series = getattr(history, "agent_review_history", {})
    if any(any(values) for values in series.values()):
        return True
    return bool(history.scalars.get("mean_peer_review_history"))


def _has_groups(history: "History") -> bool:
    return bool(history.agent_group_summary())


# Static charts shown in the gallery: (png name, data gate, plot function).
_GALLERY_CHARTS = (
    ("review_behavior", _has_review_behavior, plot_review_behavior),
    ("marketplace_activity", _has_marketplace, plot_marketplace_activity),
    ("review_reputation", _has_reputation, plot_review_reputation),
    ("paper_quality_vs_ac",
     lambda h: bool(_paper_quality_ac_points(h)), plot_paper_quality_vs_ac),
    ("writing_effort_vs_rate",
     lambda h: any(l in h.paper_accrual_rate for l in h.paper_writing_effort),
     plot_writing_effort_vs_rate),
    ("paper_writing_effort_over_time",
     lambda h: bool(_paper_writing_effort_over_time_points(h)),
     plot_paper_writing_effort_over_time),
    ("paper_ac", lambda h: bool(h.paper_ac), plot_paper_ac),
    ("choice_breakdown", lambda h: bool(_choice_tallies(h)), plot_choice_breakdown),
    ("review_effort_histogram",
     lambda h: bool(h.completed_reviews), plot_review_effort_histogram),
    ("review_effort_scatter",
     lambda h: bool(h.completed_reviews), plot_review_effort_scatter),
    ("writing_effort_distribution",
     lambda h: bool(_writing_effort_by_agent(h)), plot_writing_effort_distribution),
    ("paper_writing_effort_distribution",
     lambda h: bool(getattr(h, "paper_writing_effort", {})),
     plot_paper_writing_effort_distribution),
    ("agent_group_comparison", _has_groups, plot_agent_group_comparison),
)


# Charts whose reward-threshold line must use the run's own configured value
# rather than the current global default.
_THRESHOLD_CHARTS = frozenset({"review_effort_histogram", "review_effort_scatter"})


def render_gallery_charts(
    history: "History",
    outdir: str,
    *,
    threshold: float = MIN_REVIEW_EFFORT_THRESHOLD,
) -> list[str]:
    """Render the dark-themed static charts a run has data for into ``outdir``.

    ``threshold`` is the run's ``min_review_effort_threshold`` so the review
    effort charts draw the reward-threshold line at the value that run used.

    Returns the list of chart names produced (the front-end shows a panel only
    when its chart name is present), so runs with fewer recorded fields get
    fewer images.
    """
    os.makedirs(outdir, exist_ok=True)
    produced: list[str] = []
    with plt.rc_context(_DARK_RC):
        for name, has_data, plot_fn in _GALLERY_CHARTS:
            if not has_data(history):
                continue
            path = os.path.join(outdir, f"{name}.png")
            if name in _THRESHOLD_CHARTS:
                plot_fn(history, path, threshold=threshold)
            else:
                plot_fn(history, path)
            produced.append(name)
    return produced


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
        "mean_review_effort": plot_mean_review_effort(
            history, os.path.join(outdir, "mean_review_effort.png"), show=show
        ),
        "agent_group_comparison": plot_agent_group_comparison(
            history, os.path.join(outdir, "agent_group_comparison.png"), show=show
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
        "writing_effort_vs_rate": plot_writing_effort_vs_rate(
            history, os.path.join(outdir, "writing_effort_vs_rate.png"), show=show
        ),
        "paper_writing_effort_over_time": plot_paper_writing_effort_over_time(
            history,
            os.path.join(outdir, "paper_writing_effort_over_time.png"),
            show=show,
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
        "paper_writing_effort_distribution": plot_paper_writing_effort_distribution(
            history,
            os.path.join(outdir, "paper_writing_effort_distribution.png"),
            show=show,
        ),
        "paper_ac": plot_paper_ac(
            history, os.path.join(outdir, "paper_ac.png"), show=show
        ),
    }
