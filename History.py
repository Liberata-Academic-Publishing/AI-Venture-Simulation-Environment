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
        "mean_peer_review_epsilon": lambda env: (
            sum(getattr(a, "peer_review_epsilon_history", 0.0) for a in env.agents)
            / len(env.agents)
            if env.agents
            else 0.0
        ),
        "fair_market_price": lambda env: float(
            getattr(env, "fair_market_price", 0.0)
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
        self.agent_review_epsilon_history: dict[str, list[float]] = {}
        self.agent_groups: dict[str, str] = {}  # agent label -> class name
        self.paper_ac: dict[str, list[float]] = {}
        # Per-paper attributes (constant or final snapshot) for outcome charts.
        self.paper_quality: dict[str, float] = {}
        self.paper_authors: dict[str, str] = {}
        self.paper_reviewed: dict[str, bool] = {}
        self.paper_writing_effort: dict[str, float] = {}
        self.paper_accrual_rate: dict[str, float] = {}

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
            self._record_series(
                env.agents,
                self.agent_review_epsilon_history,
                lambda a: float(getattr(a, "peer_review_epsilon_history", 0.0)),
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
                self.paper_authors[label] = self._label(paper.author, "Agent")
                self.paper_quality[label] = float(getattr(paper, "quality", 0.0))
                self.paper_reviewed[label] = bool(getattr(paper, "reviewed", False))
                effort = getattr(paper, "writing_effort", None)
                if effort is not None:
                    self.paper_writing_effort[label] = float(effort)
                self.paper_accrual_rate[label] = float(
                    getattr(paper, "accrual_rate", 0.0)
                )

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
    def to_dict(self, *, max_paper_series: int | None = None) -> dict[str, Any]:
        # ``days``/``day`` are kept as aliases of the timestep axis so the static
        # gallery (which reads older runs too) keeps working unchanged.
        #
        # ``max_paper_series`` slims the gallery payload: the full per-paper AC
        # time series (by far the heaviest field for large runs) is kept only for
        # the highest-final-AC papers, while ``paper_final_ac`` always carries one
        # final value per paper so the quality-vs-AC scatter stays complete. Pass
        # ``None`` (the default) for a full, lossless export.
        paper_ac_full = {k: list(v) for k, v in self.paper_ac.items()}
        paper_final_ac = {
            k: (v[-1] if v else 0.0) for k, v in paper_ac_full.items()
        }
        if max_paper_series is not None and len(paper_ac_full) > max_paper_series:
            top = sorted(
                paper_ac_full,
                key=lambda k: paper_final_ac[k],
                reverse=True,
            )[:max_paper_series]
            paper_ac_out = {k: paper_ac_full[k] for k in top}
        else:
            paper_ac_out = paper_ac_full

        return {
            "timesteps": list(self.timesteps),
            "days": list(self.timesteps),
            "scalars": {k: list(v) for k, v in self.scalars.items()},
            "agent_capital": {k: list(v) for k, v in self.agent_capital.items()},
            "agent_review_history": {
                k: list(v) for k, v in self.agent_review_history.items()
            },
            "agent_review_epsilon_history": {
                k: list(v) for k, v in self.agent_review_epsilon_history.items()
            },
            "agent_groups": dict(self.agent_groups),
            "paper_ac": paper_ac_out,
            "paper_final_ac": paper_final_ac,
            "paper_authors": dict(self.paper_authors),
            "paper_quality": dict(self.paper_quality),
            "paper_reviewed": dict(self.paper_reviewed),
            "paper_writing_effort": dict(self.paper_writing_effort),
            "paper_accrual_rate": dict(self.paper_accrual_rate),
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
            "agent_group_summary": self.agent_group_summary(),
        }

    def agent_group_summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate final outcomes by concrete agent class."""
        groups: dict[str, dict[str, Any]] = {}

        def ensure(group: str) -> dict[str, Any]:
            if group not in groups:
                groups[group] = {
                    "agent_count": 0,
                    "total_final_capital": 0.0,
                    "min_final_capital": None,
                    "max_final_capital": None,
                    "papers_authored": 0,
                    "completed_reviews": 0,
                    "good_faith_reviews": 0,
                    "bad_faith_reviews": 0,
                    "total_review_effort": 0.0,
                    "total_peer_review_history": 0.0,
                    "total_peer_review_epsilon": 0.0,
                    "total_writing_effort": 0.0,
                    "actions": Counter(),
                }
            return groups[group]

        for agent_label, series in self.agent_capital.items():
            group = self.agent_groups.get(agent_label, "Agent")
            stats = ensure(group)
            final_capital = float(series[-1]) if series else 0.0
            stats["agent_count"] += 1
            stats["total_final_capital"] += final_capital
            current_min = stats["min_final_capital"]
            current_max = stats["max_final_capital"]
            stats["min_final_capital"] = (
                final_capital if current_min is None else min(current_min, final_capital)
            )
            stats["max_final_capital"] = (
                final_capital if current_max is None else max(current_max, final_capital)
            )

            review_series = self.agent_review_history.get(agent_label, [])
            if review_series:
                stats["total_peer_review_history"] += float(review_series[-1])
            epsilon_series = self.agent_review_epsilon_history.get(agent_label, [])
            if epsilon_series:
                stats["total_peer_review_epsilon"] += float(epsilon_series[-1])

        for author_label in self.paper_authors.values():
            group = self.agent_groups.get(author_label, "Agent")
            ensure(group)["papers_authored"] += 1

        for _, agent_label, _, effort, review_kind in self.completed_reviews:
            group = self.agent_groups.get(agent_label, "Agent")
            stats = ensure(group)
            stats["completed_reviews"] += 1
            stats["total_review_effort"] += float(effort)
            if review_kind == GOOD_FAITH_REVIEW:
                stats["good_faith_reviews"] += 1
            elif review_kind == BAD_FAITH_REVIEW:
                stats["bad_faith_reviews"] += 1

        for _, agent_label, effort, _ in self.writing_efforts:
            group = self.agent_groups.get(agent_label, "Agent")
            ensure(group)["total_writing_effort"] += float(effort)

        for _, agent_label, kind, _ in self.actions:
            group = self.agent_groups.get(agent_label, "Agent")
            ensure(group)["actions"][kind] += 1

        output: dict[str, dict[str, Any]] = {}
        for group, stats in groups.items():
            count = stats["agent_count"]
            reviews = stats["completed_reviews"]
            summary = dict(stats)
            summary["mean_final_capital"] = (
                stats["total_final_capital"] / count if count else 0.0
            )
            summary["mean_peer_review_history"] = (
                stats["total_peer_review_history"] / count if count else 0.0
            )
            summary["mean_peer_review_epsilon"] = (
                stats["total_peer_review_epsilon"] / count if count else 0.0
            )
            summary["average_review_effort"] = (
                stats["total_review_effort"] / reviews if reviews else 0.0
            )
            summary["actions"] = dict(stats["actions"])
            if summary["min_final_capital"] is None:
                summary["min_final_capital"] = 0.0
            if summary["max_final_capital"] is None:
                summary["max_final_capital"] = 0.0
            output[group] = summary
        return output

    def to_gallery_dict(self) -> dict[str, Any]:
        """Slim payload for the static gallery.

        The committed gallery renders the heavy static charts from PNGs (see
        ``visualize.render_gallery_charts``), so this carries only what the
        *animated* charts, the action feed/replay, and the side tables need:
        per-agent capital, scalar series, the decisions/action log, group
        summary, and each agent's final reputation (for the ranking table).
        The full, lossless history is kept locally in ``local_data/``.
        """
        return {
            "timesteps": list(self.timesteps),
            "days": list(self.timesteps),
            "scalars": {k: list(v) for k, v in self.scalars.items()},
            "agent_capital": {k: list(v) for k, v in self.agent_capital.items()},
            "agent_groups": dict(self.agent_groups),
            "agent_group_summary": self.agent_group_summary(),
            "action_counts": dict(self.action_counts),
            # One value per agent: the final peer-review reputation, so the
            # ranking table keeps its "reliability" column without shipping the
            # full per-timestep reputation series (that chart is now a PNG).
            "agent_final_reputation": {
                label: (series[-1] if series else 0.0)
                for label, series in self.agent_review_history.items()
            },
            # Decisions/action log for the feed + replay. ``day`` is the only
            # timestep key the gallery reads, so the duplicate ``timestep`` is
            # dropped to keep this compact.
            "actions": [
                {"day": d, "agent": a, "kind": k, "paper": p}
                for (d, a, k, p) in self.actions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "History":
        """Reconstruct a History from a saved (full) ``history.json`` dict.

        Lets ``visualize`` rebuild charts for any archived run. Accepts the
        full export produced by :meth:`to_dict`; missing fields default to
        empty so partial exports still load.
        """
        history = cls()
        timesteps = data.get("timesteps") or data.get("days") or []
        history.timesteps = [int(t) for t in timesteps]

        history.scalars = {
            k: [float(x) for x in v] for k, v in (data.get("scalars") or {}).items()
        }
        history.agent_capital = {
            k: [float(x) for x in v]
            for k, v in (data.get("agent_capital") or {}).items()
        }
        history.agent_review_history = {
            k: [float(x) for x in v]
            for k, v in (data.get("agent_review_history") or {}).items()
        }
        history.agent_review_epsilon_history = {
            k: [float(x) for x in v]
            for k, v in (data.get("agent_review_epsilon_history") or {}).items()
        }
        history.agent_groups = dict(data.get("agent_groups") or {})

        history.paper_ac = {
            k: [float(x) for x in v] for k, v in (data.get("paper_ac") or {}).items()
        }
        history.paper_quality = dict(data.get("paper_quality") or {})
        history.paper_authors = dict(data.get("paper_authors") or {})
        history.paper_reviewed = dict(data.get("paper_reviewed") or {})
        history.paper_writing_effort = dict(data.get("paper_writing_effort") or {})
        history.paper_accrual_rate = dict(data.get("paper_accrual_rate") or {})

        def _ts(row: dict[str, Any]) -> int:
            return int(row.get("timestep", row.get("day", 0)))

        history.actions = [
            (_ts(r), r.get("agent"), r.get("kind"), r.get("paper"))
            for r in (data.get("actions") or [])
        ]
        history.completed_reviews = [
            (
                _ts(r),
                r.get("agent"),
                r.get("paper"),
                float(r.get("effort", 0.0)),
                r.get("review_kind"),
            )
            for r in (data.get("completed_reviews") or [])
        ]
        history.writing_efforts = [
            (
                _ts(r),
                r.get("agent"),
                float(r.get("effort", 0.0)),
                bool(r.get("published", False)),
            )
            for r in (data.get("writing_efforts") or [])
        ]

        counts = data.get("action_counts")
        if counts:
            history.action_counts = Counter(
                {k: int(v) for k, v in counts.items()}
            )
        else:
            history.action_counts = Counter(kind for _, _, kind, _ in history.actions)

        return history

    def to_json(self, path: str, *, max_paper_series: int | None = None) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(max_paper_series=max_paper_series), fh, indent=2)
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
