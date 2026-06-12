"""Central configuration for the Liberata simulation and RL training.

Single source of truth for defaults. Scripts (``run_simulation.py``,
``train_rl.py``) read these defaults and expose CLI flags that *override* them at
runtime — the standard config-first / CLI-override pattern. Edit the dataclass
defaults here to change behavior everywhere; pass flags for one-off runs.

``SimConfig`` (``SIM``) holds *every* parameter that defines a single simulation
run — world size, initial papers, paper economics, the effort/reward model, the
publishing threshold, heuristic forecasting weights, and the RL agents'
settings (the RL agents are part of the simulation). ``TrainConfig`` (``TRAIN``)
is kept separate: it holds only the training-loop knobs used by ``train_rl.py``.

Stdlib only (``dataclasses``) — no extra dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Every parameter that defines a single simulation run."""

    # --- World -----------------------------------------------------------
    num_heuristic_agents: int = 0
    num_rl_agents: int = 20
    num_days: int = 1000
    seed: int = 7
    forecast_horizon_days: int = 30
    output_dir: str = "runs"

    # --- Initial papers --------------------------------------------------
    # Papers seeded before day 1 (bootstraps review material). Set
    # init_papers_per_agent=0 for no starting papers, or init_ac_min=init_ac_max=0
    # to start every agent at zero capital.
    init_papers_per_agent: int = 0
    init_ac_min: float = 0
    init_ac_max: float = 0
    init_accrual_min: float = 0.8
    init_accrual_max: float = 1.5

    # --- Paper economics -------------------------------------------------
    default_accrual_rate: float = 1.0       # AC gained per day, before bumps
    default_review_share: float = 0.01      # base ownership share a review grants
    default_reviewer_ac_threshold: float = 10.0   # reviewer AC for the high share
    default_high_ac_review_share: float = 0.02     # share for high-AC reviewers
    default_max_reviewer_share: float = 0.25       # cap on total reviewer share

    # --- Effort & reward model -------------------------------------------
    review_effort_per_day: float = 1.0          # effort added per review day
    min_review_effort_threshold: float = 10.0   # reward cliff: below this earns 0
    base_review_accrual_bump: float = 0.20      # rate bump at exactly the threshold
    first_extra_day_bump: float = 0.10          # added by the first day past threshold
    extra_day_decay: float = 0.85               # decay of each extra day's marginal bump

    # --- Publishing ------------------------------------------------------
    paper_threshold: float = 10.0   # writing effort needed to publish a paper

    # --- Heuristic forecasting -------------------------------------------
    expected_write_progress: float = 0.5
    max_forecast_effort: int = 25
    continue_marginal_weight: float = 0.15
    preferred_extra_review_days: float = 4.0

    # --- RL agents (part of the simulation) ------------------------------
    rl_backend: str = "tabular"     # "tabular" | "linear"
    rl_epsilon: float = 0.1         # exploration when learning online
    rl_gamma: float = 0.95          # TD discount
    talent_min: float = 0.8         # talent spread (inert until the sim uses it)
    talent_max: float = 1.2
    policies_dir: str = "policies"


@dataclass(frozen=True)
class TrainConfig:
    """Defaults for the training harness (train_rl.py) — kept separate from the
    simulation parameters above."""

    episodes: int = 200
    days: int = 200
    num_rl: int = 10
    num_heuristic: int = 10
    eps_start: float = 1.0
    eps_end: float = 0.05


SIM = SimConfig()
TRAIN = TrainConfig()


def default_policy_path(backend_kind: str) -> str:
    """Canonical on-disk path for a trained policy of the given backend.

    Tabular pickles (``.pkl``); linear uses ``np.save`` (``.npy``). This is the
    one implementation both train_rl.py (save) and run_simulation.py (load) use.
    """
    ext = ".pkl" if backend_kind == "tabular" else ".npy"
    return os.path.join(SIM.policies_dir, f"policy_{backend_kind}{ext}")
