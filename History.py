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

from Paper import BAD_FAITH_REVIEW, GOOD_FAITH_REVIEW

if TYPE_CHECKING:
    from Agent import ActionRecord
    from Environment import Environment

MetricFn = Callable[["Environment"], float]

COMPLETED_REVIEW_KINDS = frozenset({
    "bad_faith_review",
    "good_faith_review",
    "review_finished_write",
    "review_finished_peer_review",
    "review_stopped",
})

# Action kinds that complete a review but carry the *new* review's starting
# effort, so they must not be logged as a completed-review effort sample.
_NON_COMPLETION_REVIEW_KINDS = frozenset({"review_started"})


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
    """Scalar, per-timestep metrics recorded by default (aggregates + review behavior)."""
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
        "papers_on_market": lambda env: float(
            sum(1 for p in env.papers if getattr(p, "review_available", False))
        ),
        "completed_peer_reviews": lambda env: float(
            sum(getattr(p, "completed_peer_reviews", 0) for p in env.papers)
        ),
        "good_faith_reviews": lambda env: float(
            sum(
                1
                for p in env.papers
                for record in getattr(p, "review_records", [])
                if record.get("review_kind") == GOOD_FAITH_REVIEW
            )
        ),
        "bad_faith_reviews": lambda env: float(
            sum(
                1
                for p in env.papers
                for record in getattr(p, "review_records", [])
                if record.get("review_kind") == BAD_FAITH_REVIEW
            )
        ),
        "mean_peer_review_history": lambda env: (
            sum(getattr(a, "peer_review_history", 0.0) for a in env.agents)
            / len(env.agents)
            if env.agents
            else 0.0
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

        self.timesteps: list[int] = []
        self.scalars: dict[str, list[float]] = {name: [] for name in self.metrics}
        self.agent_capital: dict[str, list[float]] = {}
        self.agent_review_history: dict[str, list[float]] = {}
        self.agent_groups: dict[str, str] = {}  # agent label -> class name
        self.paper_ac: dict[str, list[float]] = {}
        # Per-paper attributes (constant or final snapshot) for outcome charts.
        self.paper_quality: dict[str, float] = {}
        self.paper_reviewed: dict[str, bool] = {}

        # Action log: one entry per agent turn.
        self.actions: list[tuple[int, str, str, str | None]] = []
        self.completed_reviews: list[
            tuple[int, str, str | None, float, str | None]
        ] = []
        self.writing_efforts: list[tuple[int, str, float, bool]] = []
        self.action_counts: Counter[str] = Counter()
        self.agent_actions: dict[str, list[str]] = {}

        self._labels: dict[int, str] = {}
        self._used_labels: set[str] = set()
        self._agent_counter = 0
        self._paper_counter = 0

    @property
    def days(self) -> list[int]:
        """Backwards-compatible alias for the timestep axis."""
        return self.timesteps

    # ---- recording -------------------------------------------------------
    def record_step(self, env: "Environment") -> None:
        """Snapshot per-timestep metric series. Called from ``run_timestep()``."""
        self.timesteps.append(env.timestep)
        for name, fn in self.metrics.items():
            self.scalars[name].append(float(fn(env)))
        if self.track_agents:
            self._record_series(
                env.agents,
                self.agent_capital,
                lambda a: float(getattr(a, "academic_capital", 0.0)),
                "Agent",
            )
            self._record_series(
                env.agents,
                self.agent_review_history,
                lambda a: float(getattr(a, "peer_review_history", 0.0)),
                "Agent",
            )
        if self.track_papers:
            self._record_series(
                env.papers,
                self.paper_ac,
                lambda p: float(getattr(p, "current_ac", 0.0)),
                "Paper",
            )
            for paper in env.papers:
                label = self._label(paper, "Paper")
                self.paper_quality[label] = float(getattr(paper, "quality", 0.0))
                self.paper_reviewed[label] = bool(getattr(paper, "reviewed", False))

    def record_action(self, env: "Environment", agent: Any, record: "ActionRecord") -> None:
        """Log one agent turn. Called during a timestep's marketplace/work phases,
        so the action belongs to the timestep currently being simulated."""
        timestep = env.timestep
        agent_label = self._label(agent, "Agent")
        paper_label = (
            self._label(record.paper, "Paper") if record.paper is not None else None
        )
        self.actions.append((timestep, agent_label, record.kind, paper_label))
        self.action_counts[record.kind] += 1
        if (
            record.review_effort is not None
            and record.kind in COMPLETED_REVIEW_KINDS
            and record.kind not in _NON_COMPLETION_REVIEW_KINDS
        ):
            effort = float(record.review_effort)
            review_kind = record.review_kind
            # Record every finished review, including early stops below the
            # reward threshold, so the effort distribution shows where agents
            # actually choose to stop. The reward cliff (sub-threshold reviews
            # earn nothing) lives in Paper, not in this recording gate.
            if effort > 0:
                self.completed_reviews.append(
                    (timestep, agent_label, paper_label, effort, review_kind)
                )
        if record.writing_effort is not None:
            self.writing_efforts.append(
                (
                    timestep,
                    agent_label,
                    float(record.writing_effort),
                    bool(record.published),
                )
            )
        suffix = f" of {paper_label}" if paper_label else ""
        self.agent_actions.setdefault(agent_label, []).append(
            f"timestep {timestep}: {record.kind}{suffix}"
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
        # ``days``/``day`` are kept as aliases of the timestep axis so the static
        # gallery (which reads older runs too) keeps working unchanged.
        return {
            "timesteps": list(self.timesteps),
            "days": list(self.timesteps),
            "scalars": {k: list(v) for k, v in self.scalars.items()},
            "agent_capital": {k: list(v) for k, v in self.agent_capital.items()},
            "agent_review_history": {
                k: list(v) for k, v in self.agent_review_history.items()
            },
            "paper_ac": {k: list(v) for k, v in self.paper_ac.items()},
            "paper_quality": dict(self.paper_quality),
            "paper_reviewed": dict(self.paper_reviewed),
            "actions": [
                {"timestep": d, "day": d, "agent": a, "kind": k, "paper": p}
                for (d, a, k, p) in self.actions
            ],
            "completed_reviews": [
                {
                    "timestep": d,
                    "day": d,
                    "agent": a,
                    "paper": p,
                    "effort": e,
                    "review_kind": k,
                }
                for (d, a, p, e, k) in self.completed_reviews
            ],
            "writing_efforts": [
                {
                    "timestep": d,
                    "day": d,
                    "agent": a,
                    "effort": e,
                    "published": p,
                }
                for (d, a, e, p) in self.writing_efforts
            ],
            "action_counts": dict(self.action_counts),
        }

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    def to_csv(self, path: str) -> str:
        """Wide time-series: one row per timestep; columns are scalars + agents + papers."""
        scalar_names = list(self.scalars)
        agent_names = list(self.agent_capital)
        paper_names = list(self.paper_ac)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestep"] + scalar_names + agent_names + paper_names)
            for i, timestep in enumerate(self.timesteps):
                row: list[Any] = [timestep]
                row += [self.scalars[name][i] for name in scalar_names]
                row += [self._at(self.agent_capital[name], i) for name in agent_names]
                row += [self._at(self.paper_ac[name], i) for name in paper_names]
                writer.writerow(row)
        return path

    @staticmethod
    def _at(series: list[float], i: int) -> float:
        return series[i] if i < len(series) else 0.0
