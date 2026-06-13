# AI-Venture-Simulation-Environment

This project is an agent-based simulation designed to model the incentive structures, market dynamics, and quality accrual processes within the Liberata academic publishing platform. Agents strategically allocate each timestep between advancing their own research and participating in a single-review peer-review marketplace.

## Single-review marketplace

Each paper has a `quality` sampled from a Gaussian centered on its author's intrinsic talent, known to the author before they start writing. Quality sets the paper's base accrual rate and the accrual bump a review can earn. A paper is listed on the market one timestep after it is written, and it can be reviewed exactly once: the first agent to claim it takes it off the market permanently.

While a paper is listed, its author offers each potential reviewer a distinct share price (`Paper.price_table`). A higher-quality paper (relative to the market) offers a smaller share; a reviewer with a stronger peer-review history is offered a larger one. The price table refreshes every timestep because it depends on which papers are currently on the market.

`peer_review_history` is a public per-agent metric: the total academic capital an agent has earned from reviews divided by the number of reviews it has completed.

## Timestep structure

Every timestep runs in two phases over a freshly shuffled agent order:

1. Marketplace phase — each agent may claim at most one listed paper to review. Claiming a paper while already reviewing finalizes the current review (at its accumulated effort) and starts the new one.
2. Work phase — agents that did not claim either continue their own research or, if mid-review, choose between continuing the review and finishing it to write.

## Review effort model

Review effort accrues one unit per timestep. The minimum threshold is one timestep, so a review can complete the timestep after it is taken on (earning the smallest, quality-scaled accrual bump). Additional timesteps add a logarithmically diminishing bump.

## Writing Effort Model

Paper writing effort is continuous. Each `write_paper` action contributes a `writing_effort_delta` to the agent's current paper progress. Once cumulative progress reaches `PAPER_THRESHOLD`, the agent publishes a paper and progress resets.

## Features
Our environment stresses a few main features:
- Flexible interfaces for agent, environment, market, and paper classes. This allows for multiple implementations of various algorithms.
- Various methods for more complex simulation. This can be chosen to be turned on or off depending on the simulation we want to run.

## Live dashboard
Watch a run unfold in the browser.

```
python dashboard_server.py
```

Then open http://127.0.0.1:8000. The server is stdlib-only and runs entirely on
localhost. Each browser connection streams a fresh, deterministic run with a small delay every step so the run is watchable.

## Logging runs
Separately from the live dashboard, you can save completed runs and browse them later in a static web page . This is gets published to GitHub
Pages.

After running the simulation, the terminal will prompt you whether or not to save this run to the log and ask for a name.

## Reinforcement-learning agents

`train_rl.py` trains a Q-learning agent against heuristic opponents. Note that the action space and feature vector changed with the single-review marketplace overhaul, so any policy saved before that change (in `policies/`) is incompatible and must be retrained. `run_simulation.py` defaults to heuristic agents; pass `--rl-agents N` to include RL agents.
