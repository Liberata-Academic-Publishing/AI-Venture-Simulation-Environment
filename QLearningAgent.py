"""Reinforcement-learning agent for the Liberata peer-review market.

Reward = Δ academic capital. The agent learns *which action type* to take; the
target paper for a review is chosen by the inherited heuristic forecast (so the
action space stays tiny and fixed instead of growing with the paper pool).

Action interface (matches Agent.available_actions)
--------------------------------------------------
``choose_action()`` runs every turn: when the agent is free, and on *every* day
of an active review (``should_offer_review_choice()`` is true the whole time).
There is no forced auto-continue, so the agent observes the continue-vs-stop
decision at each effort level -- the optimal-stopping signal it must learn.

* free (no active review)  -> ``("write_paper", None)``
                              or ``("peer_review", paper)`` to start a review
* review-fate choice        -> ``("peer_review", active_paper)`` to invest one
                               more day, ``("finish_review_write_paper", None)``,
                               or ``("finish_review_peer_review", candidate)``

Review value scales with invested effort (``review_accrual_bump``, with
diminishing marginal returns), so the skill being learned is *optimal stopping*:
keep investing in the current review vs. cash it out and write / start another.

Library: numpy only. Two interchangeable Q backends (tabular + linear) sit behind
a 2-method interface so the same agent can run either; tabular is the bucketed
warm-up baseline, linear function approximation is the real target (it
generalises across the continuous state instead of storing every bucket).
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from enum import IntEnum

import numpy as np

from Agent import Agent, PAPER_THRESHOLD
from HeuristicAgent import HeuristicAgent
from Paper import (
    DEFAULT_MAX_REVIEWER_SHARE,
    DEFAULT_REVIEWER_AC_THRESHOLD,
    MIN_REVIEW_EFFORT_THRESHOLD,
    Paper,
)


class QAction(IntEnum):
    WRITE = 0           # free: work on own paper
    START_REVIEW = 1    # free: begin a review of the best reviewable paper
    CONTINUE = 2        # choice: invest one more day in the active review
    FINISH_WRITE = 3    # choice: finalize the review, then write
    FINISH_REVIEW = 4   # choice: finalize the review, then start another


# Fixed-length observation. Index order is part of the contract between the
# featurizer and the Q backends, so keep it stable.
NUM_FEATURES = 8
NUM_ACTIONS = len(QAction)

# Scale used to squash invested review effort into ~[0, 1) for the feature.
EFFORT_FEATURE_SCALE = 15.0


# --------------------------------------------------------------------------- #
# Q backends: q_values(features) -> np.ndarray[NUM_ACTIONS]; update(f, a, target)
# --------------------------------------------------------------------------- #
class TabularQ:
    """Bucketized tabular Q-table. Simple, interpretable warm-up baseline."""

    def __init__(self, alpha: float = 0.1, buckets: int = 5):
        self.alpha = alpha
        self.buckets = buckets
        self.table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
        )

    def _key(self, features: np.ndarray) -> tuple[int, ...]:
        # Features are bounded to ~[0, 1]; clip then bin.
        clipped = np.clip(features, 0.0, 1.0)
        idx = np.minimum((clipped * self.buckets).astype(int), self.buckets - 1)
        return tuple(int(i) for i in idx)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.table[self._key(features)]

    def update(self, features: np.ndarray, action: int, target: float) -> None:
        q = self.table[self._key(features)]
        q[action] += self.alpha * (target - q[action])

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"alpha": self.alpha, "buckets": self.buckets,
                         "table": dict(self.table)}, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.alpha = data["alpha"]
        self.buckets = data["buckets"]
        self.table = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64), data["table"]
        )


class LinearQ:
    """Linear function approximation: Q(s, a) = w_a . [features, 1].

    Generalises across the continuous state, so the size of the state space stops
    mattering -- there are no per-state entries, just one weight row per action.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        # +1 column for a bias term.
        self.W = np.zeros((NUM_ACTIONS, NUM_FEATURES + 1), dtype=np.float64)

    @staticmethod
    def _augment(features: np.ndarray) -> np.ndarray:
        return np.append(features, 1.0)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.W @ self._augment(features)

    def update(self, features: np.ndarray, action: int, target: float) -> None:
        x = self._augment(features)
        pred = self.W[action] @ x
        self.W[action] += self.alpha * (target - pred) * x

    def save(self, path: str) -> None:
        np.save(path, self.W)

    def load(self, path: str) -> None:
        self.W = np.load(path if path.endswith(".npy") else path + ".npy")


def make_backend(kind: str = "tabular", **kwargs):
    if kind == "tabular":
        return TabularQ(**kwargs)
    if kind == "linear":
        return LinearQ(**kwargs)
    raise ValueError(f"unknown Q backend: {kind!r}")


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class QLearningAgent(HeuristicAgent):
    """Q-learning agent. Subclasses ``HeuristicAgent`` to reuse its review-target
    selection and forecast helpers (``_can_review``, ``_score_start_review``,
    ``_estimate_review_share``); only ``choose_action`` is RL-driven."""

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        forecast_horizon_days: int = 30,
        name: str | None = None,
        *,
        backend=None,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        learning: bool = True,
    ):
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            forecast_horizon_days=forecast_horizon_days,
            name=name,
        )
        # Share one backend across agents for self-play; default to a private one.
        self.backend = backend if backend is not None else make_backend("tabular")
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning = learning

        # Online-TD bookkeeping for the previous transition.
        self._last_features: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_capital: float = academic_capital

    # ---- policy ----------------------------------------------------------- #
    def choose_action(self) -> tuple[str, Paper | None]:
        review_choice = self.should_offer_review_choice()
        # In review-fate mode the active paper is in progress; a *new* review must
        # target a different paper, so exclude it from the candidate pool.
        best_paper, best_share, best_ac = self._best_reviewable(
            exclude=self.active_review_paper
        )
        features = self._features(best_share, best_ac)
        legal = self._legal_actions(review_choice, best_paper is not None)

        # TD update for the transition that ended at this state.
        if self.learning and self._last_features is not None:
            reward = self.academic_capital - self._last_capital
            q_next = self.backend.q_values(features)
            max_next = max((q_next[a] for a in legal), default=0.0)
            target = reward + self.gamma * max_next
            self.backend.update(self._last_features, self._last_action, target)

        action = self._select(features, legal)

        self._last_features = features
        self._last_action = int(action)
        self._last_capital = self.academic_capital

        return self._to_env_action(action, best_paper)

    def end_episode(self) -> None:
        """Terminal flush: bootstrap-free update for the final transition, then
        clear per-episode state. Call between training episodes."""
        if self.learning and self._last_features is not None:
            reward = self.academic_capital - self._last_capital
            self.backend.update(self._last_features, self._last_action, reward)
        self._last_features = None
        self._last_action = None

    # ---- action helpers --------------------------------------------------- #
    def _legal_actions(
        self, review_choice: bool, has_reviewable: bool
    ) -> list[int]:
        if review_choice:
            actions = [QAction.CONTINUE, QAction.FINISH_WRITE]
            if has_reviewable:
                actions.append(QAction.FINISH_REVIEW)
            return actions
        actions = [QAction.WRITE]
        if has_reviewable:
            actions.append(QAction.START_REVIEW)
        return actions

    def _select(self, features: np.ndarray, legal: list[int]) -> int:
        if self.learning and random.random() < self.epsilon:
            return random.choice(legal)
        q = self.backend.q_values(features)
        return max(legal, key=lambda a: q[a])

    def _to_env_action(
        self, action: int, best_paper: Paper | None
    ) -> tuple[str, Paper | None]:
        if action == QAction.WRITE:
            return "write_paper", None
        if action == QAction.START_REVIEW:
            return "peer_review", best_paper
        if action == QAction.CONTINUE:
            return "peer_review", self.active_review_paper
        if action == QAction.FINISH_WRITE:
            return "finish_review_write_paper", None
        return "finish_review_peer_review", best_paper

    # ---- state ------------------------------------------------------------ #
    def _best_reviewable(
        self, exclude: Paper | None = None
    ) -> tuple[Paper | None, float, float]:
        """Highest-forecast reviewable paper (reusing the heuristic score) plus
        its estimated review share and current AC."""
        reviewable = [
            p
            for p in Agent.all_papers
            if p is not exclude and self._can_review(p)
        ]
        if not reviewable:
            return None, 0.0, 0.0
        best = max(
            reviewable,
            key=lambda p: self._score_start_review(p, MIN_REVIEW_EFFORT_THRESHOLD),
        )
        share = self._estimate_review_share(best)
        return best, share, best.current_ac

    def _features(self, best_share: float, best_ac: float) -> np.ndarray:
        num_reviewable = sum(1 for p in Agent.all_papers if self._can_review(p))
        in_review = self.active_review_paper is not None
        features = np.array(
            [
                # Required input #1: progress on the paper being written.
                min(self.paper_progress / PAPER_THRESHOLD, 1.0),
                # Required input #2: effort/days already invested in the review.
                np.tanh(self.active_review_effort / EFFORT_FEATURE_SCALE),
                np.tanh(self.academic_capital / 100.0),
                1.0 if self.academic_capital >= DEFAULT_REVIEWER_AC_THRESHOLD else 0.0,
                np.tanh(num_reviewable / 10.0),
                np.tanh(best_share / DEFAULT_MAX_REVIEWER_SHARE),
                np.tanh(best_ac / 100.0),
                1.0 if in_review else 0.0,
            ],
            dtype=np.float64,
        )
        return features
