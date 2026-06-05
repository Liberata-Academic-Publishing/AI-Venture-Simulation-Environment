"""Matplotlib charts for a simulation ``History``.

This module is the *only* part that needs matplotlib; the recorder/exports in
``History`` stay dependency-free. Importing this module raises a clear error if
matplotlib is missing, so callers can ``try: import visualize`` and fall back to
the CSV/JSON export.
"""

from __future__ import annotations

import os
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
    """Academic capital over time, with one color per agent group/class.

    Like ``plot_agent_capital`` but colors agents by their class (e.g. good
    HeuristicAgents vs bad BadFaithAgents) so the two cohorts stand out, with a
    single legend entry per group.
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
    for key in ("good_faith_reviews", "bad_faith_reviews", "num_papers"):
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


def plot_all(history: "History", outdir: str = "runs") -> dict[str, str]:
    """Render every chart into ``outdir`` and return {name: path}."""
    os.makedirs(outdir, exist_ok=True)
    return {
        "agent_capital": plot_agent_capital(
            history, os.path.join(outdir, "agent_capital.png")
        ),
        "agent_capital_by_group": plot_agent_capital_by_group(
            history, os.path.join(outdir, "agent_capital_by_group.png")
        ),
        "system_aggregates": plot_system_aggregates(
            history, os.path.join(outdir, "system_aggregates.png")
        ),
        "review_behavior": plot_review_behavior(
            history, os.path.join(outdir, "review_behavior.png")
        ),
        "paper_ac": plot_paper_ac(history, os.path.join(outdir, "paper_ac.png")),
    }
