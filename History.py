"""Single store for one simulation run: per-day metric series + the action log.

Pure standard library (no numpy/matplotlib), so recording always works. The
``Environment`` feeds it: ``record_step(env)`` once per day from ``nextstep()``,
and ``record_action(env, agent, record)`` for each agent turn from ``agentact()``.
Export with ``to_csv`` / ``to_json`` / ``to_dict``; visualize separately (see
``visualize.py``) by reading these series.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Agent import ActionRecord
    from Environment import Environment

from Paper import MIN_REVIEW_EFFORT_THRESHOLD

MetricFn = Callable[["Environment"], float]

COMPLETED_REVIEW_KINDS = frozenset({
    "review_finished_write",
    "review_finished_peer_review",
    "review_stopped",
})


def gini(values: Iterable[float]) -> float:
    """Gini coefficient of non-negative values (0 = perfectly equal, →1 = unequal)."""
    nonneg = sorted(max(0.0, float(v)) for v in values)
    n = len(nonneg)
    total = sum(nonneg)
    if n == 0 or total == 0.0:
        return 0.0
    weighted = sum(i * value for i, value in enumerate(nonneg, start=1))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def default_metrics() -> dict[str, MetricFn]:
    """Scalar, per-day metrics recorded by default (system aggregates + review behavior)."""
    return {
        "total_capital": lambda env: sum(a.academic_capital for a in env.agents),
        "mean_capital": lambda env: (
            sum(a.academic_capital for a in env.agents) / len(env.agents)
            if env.agents
            else 0.0
        ),
        "max_capital": lambda env: max(
            (a.academic_capital for a in env.agents), default=0.0
        ),
        "capital_gini": lambda env: gini(a.academic_capital for a in env.agents),
        "num_papers": lambda env: float(len(env.papers)),
        "completed_peer_reviews": lambda env: float(
            sum(getattr(p, "completed_peer_reviews", 0) for p in env.papers)
        ),
    }


class History:
    """Time-series + action log for a run.

    Series are kept aligned to ``self.days``: every agent/paper series has the
    same length, with papers that appear mid-run back-filled with ``0.0`` for the
    days before they existed.
    """

    def __init__(
        self,
        metrics: dict[str, MetricFn] | None = None,
        *,
        track_agents: bool = True,
        track_papers: bool = True,
    ):
        self.metrics = default_metrics() if metrics is None else dict(metrics)
        self.track_agents = track_agents
        self.track_papers = track_papers

        self.days: list[int] = []
        self.scalars: dict[str, list[float]] = {name: [] for name in self.metrics}
        self.agent_capital: dict[str, list[float]] = {}
        self.agent_groups: dict[str, str] = {}  # agent label -> class name
        self.paper_ac: dict[str, list[float]] = {}

        # Action log: one entry per agent turn.
        self.actions: list[tuple[int, str, str, str | None]] = []
        self.review_efforts: list[tuple[int, str, str | None, float, str | None]] = []
        self.completed_reviews: list[tuple[int, str, str | None, float]] = []
        self.action_counts: Counter[str] = Counter()
        self.agent_actions: dict[str, list[str]] = {}

        self._labels: dict[int, str] = {}
        self._used_labels: set[str] = set()
        self._agent_counter = 0
        self._paper_counter = 0

    # ---- recording -------------------------------------------------------
    def record_step(self, env: "Environment") -> None:
        """Snapshot per-day metric series. Called from ``Environment.nextstep()``."""
        self.days.append(env.day)
        for name, fn in self.metrics.items():
            self.scalars[name].append(float(fn(env)))
        if self.track_agents:
            self._record_series(
                env.agents,
                self.agent_capital,
                lambda a: float(getattr(a, "academic_capital", 0.0)),
                "Agent",
            )
        if self.track_papers:
            self._record_series(
                env.papers,
                self.paper_ac,
                lambda p: float(getattr(p, "current_ac", 0.0)),
                "Paper",
            )

    def record_action(self, env: "Environment", agent: Any, record: "ActionRecord") -> None:
        """Log one agent turn. Called from ``Environment.agentact()`` (before the
        day is advanced, so the action belongs to the day about to be simulated)."""
        day = env.day + 1
        agent_label = self._label(agent, "Agent")
        paper_label = (
            self._label(record.paper, "Paper") if record.paper is not None else None
        )
        self.actions.append((day, agent_label, record.kind, paper_label))
        self.action_counts[record.kind] += 1
        if record.review_effort is not None and record.review_kind is not None:
            self.review_efforts.append(
                (
                    day,
                    agent_label,
                    paper_label,
                    float(record.review_effort),
                    record.review_kind,
                )
            )
        if (
            record.review_effort is not None
            and record.kind in COMPLETED_REVIEW_KINDS
        ):
            effort = float(record.review_effort)
            if effort >= MIN_REVIEW_EFFORT_THRESHOLD:
                self.completed_reviews.append(
                    (day, agent_label, paper_label, effort)
                )
        suffix = f" of {paper_label}" if paper_label else ""
        self.agent_actions.setdefault(agent_label, []).append(
            f"day {day}: {record.kind}{suffix}"
        )

    def _record_series(
        self,
        entities: Iterable[Any],
        store: dict[str, list[float]],
        value_fn: Callable[[Any], float],
        prefix: str,
    ) -> None:
        target_len = len(self.days)
        for entity in entities:
            label = self._label(entity, prefix)
            if prefix == "Agent":
                self.agent_groups[label] = type(entity).__name__
            series = store.get(label)
            if series is None:
                series = [0.0] * (target_len - 1)  # back-fill days before it existed
                store[label] = series
            series.append(value_fn(entity))

    def _label(self, obj: Any, prefix: str) -> str:
        """Stable, unique display label for an agent/paper, cached by object id."""
        key = id(obj)
        cached = self._labels.get(key)
        if cached is not None:
            return cached

        if prefix == "Agent":
            self._agent_counter += 1
            default = f"Agent {self._agent_counter}"
        else:
            self._paper_counter += 1
            default = f"Paper {self._paper_counter}"

        name = getattr(obj, "name", None) or getattr(obj, "title", None) or default
        base, n = name, 2
        while name in self._used_labels:  # defend against duplicate names/titles
            name = f"{base} ({n})"
            n += 1

        self._used_labels.add(name)
        self._labels[key] = name
        return name

    # ---- export ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "days": list(self.days),
            "scalars": {k: list(v) for k, v in self.scalars.items()},
            "agent_capital": {k: list(v) for k, v in self.agent_capital.items()},
            "paper_ac": {k: list(v) for k, v in self.paper_ac.items()},
            "actions": [
                {"day": d, "agent": a, "kind": k, "paper": p}
                for (d, a, k, p) in self.actions
            ],
            "review_efforts": [
                {
                    "day": d,
                    "agent": a,
                    "paper": p,
                    "effort": e,
                    "kind": k,
                }
                for (d, a, p, e, k) in self.review_efforts
            ],
            "completed_reviews": [
                {"day": d, "agent": a, "paper": p, "effort": e}
                for (d, a, p, e) in self.completed_reviews
            ],
            "action_counts": dict(self.action_counts),
        }

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    def to_csv(self, path: str) -> str:
        """Wide time-series: one row per day; columns are scalars + each agent + each paper."""
        scalar_names = list(self.scalars)
        agent_names = list(self.agent_capital)
        paper_names = list(self.paper_ac)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["day"] + scalar_names + agent_names + paper_names)
            for i, day in enumerate(self.days):
                row: list[Any] = [day]
                row += [self.scalars[name][i] for name in scalar_names]
                row += [self._at(self.agent_capital[name], i) for name in agent_names]
                row += [self._at(self.paper_ac[name], i) for name in paper_names]
                writer.writerow(row)
        return path

    @staticmethod
    def _at(series: list[float], i: int) -> float:
        return series[i] if i < len(series) else 0.0
