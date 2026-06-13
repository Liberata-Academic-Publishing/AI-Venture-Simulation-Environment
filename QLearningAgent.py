"""Reinforcement-learning agent for the single-review peer-review market.

Reward = Δ academic capital. The agent learns *which action type* to take each
timestep; the target paper for a claim is chosen by the inherited heuristic
forecast, so the action space stays tiny and fixed.

The agent makes one decision per timestep. The decision is computed in the
marketplace phase (``choose_marketplace_action``): if it decides to claim a
paper it returns that paper; otherwise it caches a work-phase action that
``choose_work_action`` replays.

Action space
------------
* free (no active review)  -> ``WRITE`` or ``CLAIM`` (start reviewing a paper)
* active review            -> ``CONTINUE`` (one more timestep), ``FINISH_WRITE``
                              (finalize then write), or ``CLAIM_NEW`` (finalize
                              the current review by claiming another paper)

NOTE: the action semantics and feature vector changed with the single-review
marketplace overhaul, so policies saved before that change are incompatible and
must be retrained.

Library: numpy only. Two interchangeable Q backends (tabular + linear).
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
    MIN_REVIEW_EFFORT_THRESHOLD,
    Paper,
)


class QAction(IntEnum):
    WRITE = 0          # free: work on own paper
    CLAIM = 1          # free: claim and start reviewing the best listed paper
    CONTINUE = 2       # review: invest one more timestep in the active review
    FINISH_WRITE = 3   # review: finalize the active review, then write
    CLAIM_NEW = 4      # review: finalize by claiming another listed paper


# Fixed-length observation. Index order is part of the contract between the
# featurizer and the Q backends, so keep it stable.
NUM_FEATURES = 9
NUM_ACTIONS = len(QAction)

# Scale used to squash invested review effort into ~[0, 1) for the feature.
EFFORT_FEATURE_SCALE = 5.0


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
    """Linear function approximation: Q(s, a) = w_a . [features, 1]."""

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
    selection and forecast helpers; only the action *type* is RL-driven."""

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        forecast_horizon_timesteps: int = 30,
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
            forecast_horizon_timesteps=forecast_horizon_timesteps,
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

        # Decision cached in the marketplace phase, replayed in the work phase.
        self._pending_action: int = int(QAction.WRITE)
        self._pending_paper: Paper | None = None

    # ---- policy: one decision per timestep, made in the marketplace phase -- #
    def choose_marketplace_action(self) -> Paper | None:
        in_review = self.active_review_paper is not None
        best_paper, best_share, best_ac = self._best_reviewable(
            exclude=self.active_review_paper
        )
        features = self._features(best_share, best_ac)
        legal = self._legal_actions(in_review, best_paper is not None)

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
        self._pending_action = int(action)
        self._pending_paper = best_paper

        if action in (QAction.CLAIM, QAction.CLAIM_NEW):
            return best_paper
        return None

    def choose_work_action(self) -> tuple[str, Paper | None]:
        action = self._pending_action
        if action == QAction.CONTINUE:
            return "peer_review", self.active_review_paper
        if action == QAction.FINISH_WRITE:
            return "finish_review_write_paper", None
        return "write_paper", None

    def end_episode(self) -> None:
        """Terminal flush: bootstrap-free update for the final transition."""
        if self.learning and self._last_features is not None:
            reward = self.academic_capital - self._last_capital
            self.backend.update(self._last_features, self._last_action, reward)
        self._last_features = None
        self._last_action = None

    # ---- action helpers --------------------------------------------------- #
    def _legal_actions(self, in_review: bool, has_reviewable: bool) -> list[int]:
        if in_review:
            actions = [QAction.CONTINUE, QAction.FINISH_WRITE]
            if has_reviewable:
                actions.append(QAction.CLAIM_NEW)
            return actions
        actions = [QAction.WRITE]
        if has_reviewable:
            actions.append(QAction.CLAIM)
        return actions

    def _select(self, features: np.ndarray, legal: list[int]) -> int:
        if self.learning and random.random() < self.epsilon:
            return random.choice(legal)
        q = self.backend.q_values(features)
        return max(legal, key=lambda a: q[a])

    # ---- state ------------------------------------------------------------ #
    def _best_reviewable(
        self, exclude: Paper | None = None
    ) -> tuple[Paper | None, float, float]:
        """Highest-value claimable paper (reusing the heuristic score) plus its
        estimated review share and current AC."""
        reviewable = [
            p
            for p in Agent.all_papers
            if p is not exclude
            and self._can_review(p)
            and p.offered_share(self) > 0.0
        ]
        if not reviewable:
            return None, 0.0, 0.0
        best = max(reviewable, key=self._score_claim)
        share = self._prospective_share(best)
        return best, share, best.current_ac

    def _features(self, best_share: float, best_ac: float) -> np.ndarray:
        num_reviewable = sum(
            1
            for p in Agent.all_papers
            if self._can_review(p) and p.offered_share(self) > 0.0
        )
        in_review = self.active_review_paper is not None
        features = np.array(
            [
                min(self.paper_progress / PAPER_THRESHOLD, 1.0),
                np.tanh(self.active_review_effort / EFFORT_FEATURE_SCALE),
                np.tanh(self.academic_capital / 100.0),
                np.tanh(self.peer_review_history / 10.0),
                np.tanh(num_reviewable / 10.0),
                np.tanh(best_share / DEFAULT_MAX_REVIEWER_SHARE),
                np.tanh(best_ac / 100.0),
                1.0 if in_review else 0.0,
                min(
                    self.active_review_effort / max(MIN_REVIEW_EFFORT_THRESHOLD, 1.0),
                    1.0,
                ),
            ],
            dtype=np.float64,
        )
        return features
