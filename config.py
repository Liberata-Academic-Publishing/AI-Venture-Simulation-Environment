"""Central configuration for the Liberata simulation and RL training.

Single source of truth for defaults. Scripts (``run_simulation.py``,
``train_rl.py``) read these defaults and expose CLI flags that *override* them at
runtime — the standard config-first / CLI-override pattern. Edit the dataclass
defaults here to change behavior everywhere; pass flags for one-off runs.

Stdlib only (``dataclasses``) — no extra dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Defaults for a single simulation run (run_simulation.py)."""

    num_agents: int = 20            # heuristic agents
    num_days: int = 1000
    seed: int = 7
    forecast_horizon_days: int = 30
    output_dir: str = "runs"


@dataclass(frozen=True)
class RLConfig:
    """Defaults for the RL agents (shared by run_simulation.py & train_rl.py)."""

    num_agents: int = 5             # RL agents added alongside heuristics
    backend: str = "tabular"        # "tabular" | "linear"
    epsilon: float = 0.1            # exploration when learning online
    gamma: float = 0.95             # TD discount
    talent_min: float = 0.8         # talent spread (inert until the sim uses it)
    talent_max: float = 1.2
    policies_dir: str = "policies"


@dataclass(frozen=True)
class TrainConfig:
    """Defaults for the training harness (train_rl.py)."""

    episodes: int = 200
    days: int = 200
    num_rl: int = 10
    num_heuristic: int = 10
    eps_start: float = 1.0
    eps_end: float = 0.05


SIM = SimConfig()
RL = RLConfig()
TRAIN = TrainConfig()


def default_policy_path(backend_kind: str) -> str:
    """Canonical on-disk path for a trained policy of the given backend.

    Tabular pickles (``.pkl``); linear uses ``np.save`` (``.npy``). This is the
    one implementation both train_rl.py (save) and run_simulation.py (load) use.
    """
    ext = ".pkl" if backend_kind == "tabular" else ".npy"
    return os.path.join(RL.policies_dir, f"policy_{backend_kind}{ext}")
