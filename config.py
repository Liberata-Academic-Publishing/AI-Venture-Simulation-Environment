"""Central configuration for the Liberata simulation and RL training.

Single source of truth for defaults. Scripts (``run_simulation.py``,
``train_rl.py``) read these defaults and expose CLI flags that *override* them at
runtime — the standard config-first / CLI-override pattern. Edit the dataclass
defaults here to change behavior everywhere; pass flags for one-off runs.

``SimConfig`` (``SIM``) holds *every* parameter that defines a single simulation
run — world size, initial papers, paper economics, the effort/reward model, the
publishing threshold, heuristic forecasting weights, and the RL agents'
settings (the RL agents are part of the simulation). ``TrainConfig`` (``TRAIN``)
and ``TrainDQNConfig`` (``TRAIN_DQN``) hold training-loop knobs for
``train_rl.py`` and ``train_dqn.py`` respectively.

Stdlib only (``dataclasses``) — no extra dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Every parameter that defines a single simulation run."""

    # --- World -----------------------------------------------------------
    num_heuristic_agents: int = 10
    num_rl_agents: int = 100
    num_dqn_agents: int = 0
    num_random_agents: int = 10
    num_probabilistic_agents: int = 0
    num_timesteps: int = 2000
    seed: int = 7
    forecast_horizon_timesteps: int = 30
    output_dir: str = "runs"

    # --- Initial papers --------------------------------------------------
    # Papers seeded before timestep 1 (bootstraps review material). Set
    # init_papers_per_agent=0 for no starting papers, or init_ac_min=init_ac_max=0
    # to start every agent at zero capital.
    init_papers_per_agent: int = 0
    init_ac_min: float = 0
    init_ac_max: float = 0
    init_accrual_min: float = 0.8
    init_accrual_max: float = 1.5

    # --- Paper economics -------------------------------------------------
    default_accrual_rate: float = 1.0       # base AC gained per timestep, before bumps
    default_review_share: float = 0.05      # legacy fallback when fair-market pricing is off
    default_max_reviewer_share: float = 0.1       # cap on a single review's share
    min_offer_share: float = 0.005          # floor on a non-zero review offer
    use_fair_market_pricing: bool = True
    prior_review_epsilon: float = 0.05      # prior reviewer effect before review history exists
    # Author-side pricing policy. ``static_fair_market`` preserves the current
    # fair-market formula. ``adaptive_multiplier`` keeps that formula as the
    # base, then lets authors raise/lower future offers based on how quickly
    # their papers are claimed from the marketplace.
    pricing_policy: str = "static_fair_market"  # "static_fair_market" | "adaptive_multiplier"
    target_market_wait_timesteps: float = 1.0
    adaptive_pricing_learning_rate: float = 0.10
    min_author_price_multiplier: float = 0.25
    max_author_price_multiplier: float = 2.0

    # --- Paper quality ---------------------------------------------------
    # Each paper's quality is drawn from N(author talent, quality_sigma) and is
    # known to the author before they start writing it. Quality scales the
    # paper's base accrual rate and the accrual bump a review can earn, and it
    # drives the per-reviewer share the author is willing to offer.
    quality_sigma: float = 0.20
    min_paper_quality: float = 0.10
    quality_price_scale: float = 1.0    # higher quality -> smaller offered share
    history_price_scale: float = 0.5    # better reviewer history -> larger offered share

    # --- Effort & reward model -------------------------------------------
    review_paradigm: str = "continuous"       # "continuous" | "discrete"
    review_effort_per_timestep: float = 1.0     # effort added per review timestep
    writing_effort_per_timestep: float = 1.0    # continuous writing effort per timestep
    min_review_effort_threshold: float = 1.0    # minimum valid review/share effort
    good_faith_review_threshold: float = 2.0    # continuous-mode classification
    bad_review_timesteps: float = 1.0           # discrete bad-faith duration (T_B)
    good_review_timesteps: float = 5.0          # discrete good-faith duration (T_G)
    review_effort_curve: str = "sigmoid"        # "sigmoid" | "log" | "jump"
    min_review_accrual_bump: float = 0.05       # sigmoid bump at one timestep
    max_review_accrual_bump: float = 0.35       # sigmoid saturation near a long review
    review_sigmoid_midpoint: float = 2.5        # review length where impact accelerates
    review_sigmoid_steepness: float = 1.4
    review_jump_threshold: float = 15.0          #f optional jump-mode high-effort cutoff
    review_jump_bump: float = 0.50               # optional extra bump after the cutoff
    review_jump_width: float = 0.75              # softness of the jump-mode transition
    base_review_accrual_bump: float = 0.20      # log-mode rate bump at exactly the threshold
    first_extra_day_bump: float = 0.10          # log-mode first extra timestep bump

    # --- Publishing ------------------------------------------------------
    paper_threshold: float = 10.0   # legacy forecast normalizer (choice-mode continuous)
    # Continuous writing: "choice" = agent picks when to finish/list; "threshold"
    # = auto-publish after a fixed amount of writing effort (no early finish).
    continuous_publishing: str = "threshold"       # "choice" | "threshold"
    continuous_paper_timesteps: float = 50.0    # writing effort to auto-publish
    discrete_paper_timesteps: float = 200.0      # discrete manuscript duration (T_M)
    discrete_writing_effort_per_timestep: float = 1.0
    # Paper effort target for each manuscript. ``fixed`` preserves the existing
    # thresholds above; ``uniform`` samples once per paper from the 50-150 band
    # discussed in sync; ``quality_scaled`` maps higher sampled paper quality to
    # a larger target inside that same band.
    paper_effort_mode: str = "uniform"             # "fixed" | "uniform" | "quality_scaled"
    paper_effort_min: float = 130.0
    paper_effort_max: float = 170.0
    # Continuous-mode asymptotic writing model: a paper's accrual rate approaches
    # its quality-defined ceiling as accumulated writing effort grows. Higher k
    # reaches the ceiling faster (k=0.2 -> ~63% at 5 ts, ~86% at 10 ts).
    writing_saturation: float = 0.2

    # --- Control / probability agents ------------------------------------
    random_claim_probability: float = 0.5
    random_good_faith_probability: float = 0.5
    probabilistic_claim_probability: float = 0.5
    probabilistic_good_faith_probability: float = 0.5

    # --- Heuristic forecasting -------------------------------------------
    expected_write_progress: float = 0.5
    max_forecast_effort: int = 25
    continue_marginal_weight: float = 0.15
    preferred_extra_review_timesteps: float = 4.0

    # --- RL agents (part of the simulation) ------------------------------
    rl_backend: str = "tabular"     # "tabular" | "linear"
    rl_epsilon: float = 0.1         # exploration when learning online
    rl_gamma: float = 0.95          # TD discount
    rl_reward_ac_weight: float = 0.0       # weight on Δ academic capital
    rl_reward_rank_weight: float = 0.0   # weight on Δ AC percentile rank (0..1)
    rl_reward_accrual_weight: float = 1.0  # weight on Δ portfolio accrual rate
    rl_autoload_policy: bool = True  # auto-load the saved baseline for RL agents
    talent_min: float = 0.6         # default talent spread; CLI can widen/narrow it
    talent_max: float = 1.4
    policies_dir: str = "policies"

    # --- Export / gallery limits -----------------------------------------
    # Full histories remain in local_data/. The committed static gallery keeps
    # only the most recent action rows for feed/replay while retaining compact
    # per-timestep action counts for full action-mix plots.
    gallery_action_limit: int = 5000

    # --- DQN agents (separate from tabular/linear RL) ------------------
    dqn_hidden_size: int = 64
    dqn_hidden_layers: int = 2
    dqn_lr: float = 0.001
    dqn_gamma: float = 0.95
    dqn_epsilon: float = 0.1
    dqn_replay_capacity: int = 10000
    dqn_batch_size: int = 32
    dqn_target_sync: int = 200
    dqn_autoload_policy: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Defaults for the training harness (train_rl.py) — kept separate from the
    simulation parameters above."""

    episodes: int = 200
    timesteps: int = 1000
    num_rl: int = 10
    num_heuristic: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05


@dataclass(frozen=True)
class TrainDQNConfig:
    """Defaults for the DQN training harness (train_dqn.py)."""

    episodes: int = 200
    timesteps: int = 1000
    num_dqn: int = 10
    num_heuristic: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05


@dataclass(frozen=True)
class TrainDiscreteConfig:
    """Defaults for the discrete RL harness (train_discrete_rl.py)."""

    episodes: int = 200
    timesteps: int = 1000
    num_rl: int = 10
    num_heuristic: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05


SIM = SimConfig()
TRAIN = TrainConfig()
TRAIN_DQN = TrainDQNConfig()
TRAIN_DISCRETE = TrainDiscreteConfig()


def default_policy_path(backend_kind: str) -> str:
    """Canonical on-disk path for a trained policy of the given backend.

    Tabular pickles (``.pkl``); linear uses ``np.save`` (``.npy``). This is the
    one implementation both train_rl.py (save) and run_simulation.py (load) use.
    """
    ext = ".pkl" if backend_kind == "tabular" else ".npy"
    return os.path.join(SIM.policies_dir, f"policy_{backend_kind}{ext}")


def default_dqn_policy_path() -> str:
    """Canonical on-disk path for a trained DQN policy."""
    return os.path.join(SIM.policies_dir, "policy_dqn.pkl")


def default_discrete_policy_path(backend_kind: str) -> str:
    """Canonical path for a discrete 3-action Q policy (``train_discrete_rl.py``)."""
    return os.path.join(SIM.policies_dir, f"policy_{backend_kind}_discrete.pkl")
