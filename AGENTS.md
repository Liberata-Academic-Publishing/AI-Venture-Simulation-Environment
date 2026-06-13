# AI-Venture-Simulation-Environment

Agent brief for the Liberata simulation. This is the source of truth for project
context; keep it concise and current. `README.md` is the human-facing doc.

## What this is

An agent-based simulation of the "Liberata" academic publishing platform. Each
day, every agent chooses how to spend its turn between two activities:

1. **Writing papers** (advancing its own research), and
2. **Peer review** (participating in a review marketplace for academic capital).

The research question is **good-faith vs bad-faith peer review**: the sim
presents both choices and observes the emergent agentic behavior. Reviewer share
does not depend on the good/bad-faith classification; the classification only
affects the reviewed paper's future accrual rate.

## Core mechanics

`config.py` is the single source of truth for all tunable parameters. Edit the
`SimConfig`/`TrainConfig` dataclass defaults to change behavior everywhere; pass
CLI flags for one-off overrides.

- **Writing effort**: each `write_paper` adds a `writing_effort_delta`; once
  cumulative progress reaches `paper_threshold`, a paper publishes and progress
  resets.
- **Review paradigms**: each environment run is either `continuous` or
  `discrete`, configured by `review_paradigm`; paradigms are not mixed inside one
  run.
- **Continuous review effort**: agents choose time spent by continuing or
  finishing reviews. The environment classifies completed reviews as bad faith
  below `good_faith_review_threshold` and good faith at or above it.
- **Discrete review effort**: agents choose `bad_faith` or `good_faith` when
  claiming a review. By default `T_B = 1`, `T_G = 5 * T_B`, and discrete
  manuscript work uses `T_M = 200 * T_B`.
- **Review-share economics**: a review grants ownership share on the reviewed
  paper (`default_review_share`, higher for high-AC reviewers, capped at
  `default_max_reviewer_share`).
- **RL settings** live in `SimConfig` too (`rl_backend`, `rl_epsilon`,
  `rl_gamma`); RL agents are part of the simulation.

## Architecture map

- `config.py` - `SimConfig` (`SIM`) and `TrainConfig` (`TRAIN`) dataclasses;
  config-first with CLI overrides. `default_policy_path()` resolves policy files.
- `Agent.py` - abstract `Agent` base + the action protocol: `write_paper`,
  `peer_review`, `finish_review_write_paper`, `finish_review_peer_review`.
  `ActionRecord` describes one turn. `Agent.all_papers` is a shared class list.
- `HeuristicAgent.py`, `QLearningAgent.py`, `RandomAgent.py` - agent variants,
  including random controls and discrete-only probability agents.
- `Paper.py` - paper economics, reviews, and accrual; defines
  `MIN_REVIEW_EFFORT_THRESHOLD`, `REVIEW_EFFORT_PER_DAY`.
- `Environment.py` - the world / turn loop (`agentact`, `nextstep`).
- `History.py` - run logging and metrics (e.g. `gini`).
- `run_simulation.py` - main entry point (run a sim, print summary, optionally
  archive to the `docs/` gallery).
- `train_rl.py` - self-play RL training + greedy evaluation; auto-saves policies.
- `visualize.py`, `dashboard_server.py` - charts + live localhost dashboard.
- `docs/` - static GitHub Pages gallery of saved runs (`docs/data/<run_id>/`).

## Commands

```bash
# Run a simulation (prompts to archive afterward)
python run_simulation.py
python run_simulation.py --no-archive            # don't save
python run_simulation.py --name "my run"         # save non-interactively
python run_simulation.py --review-paradigm discrete --random-agents 5 --no-archive
python run_simulation.py --rl-agents 20 --rl-backend tabular

# Train the RL agent (auto-saves to policies/)
python train_rl.py
python train_rl.py --backend linear --episodes 300
python train_rl.py --load policies/policy_tabular.pkl --episodes 0   # eval only

# Tests (stdlib unittest)
python test_simulation.py
python -m unittest test_simulation

# Live dashboard at http://127.0.0.1:8000
python dashboard_server.py
```

## Conventions

- **Stdlib-only where possible** (`config.py` uses only `dataclasses`); matplotlib
  is optional and guarded behind an import check in tests.
- **Config-first**: change defaults in `config.py`; use CLI flags for one-offs.
- **Determinism**: runs use a fixed `seed` (default 7) for reproducibility.
- Frozen dataclasses for config; type hints use `from __future__ import annotations`.
