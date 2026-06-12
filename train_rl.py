"""Train and evaluate the Q-learning agent on the Liberata market.

Self-play across many short episodes: a pool of ``QLearningAgent``s sharing
one Q backend learns online (reward = Δ academic capital), with ε decayed each
episode. After training we run a greedy (ε=0) evaluation against
``HeuristicAgent``s and report the capital gap.

The trained policy (the Q backend) is auto-saved to ``policies/`` and can be
reloaded with ``--load`` to resume training or to run a frozen policy.

Usage:
    python train_rl.py                       # train + auto-save to policies/
    python train_rl.py --backend linear --episodes 300
    python train_rl.py --load policies/policy_tabular.pkl --episodes 0   # eval only
"""

from __future__ import annotations

import argparse
import os
import random

from Agent import Agent
from config import SIM, TRAIN, default_policy_path
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from Paper import Paper
from QLearningAgent import QLearningAgent, make_backend


def seed_initial_papers(agents: list[Agent], rng: random.Random) -> None:
    """Seed starting papers per SimConfig (count + AC/accrual ranges)."""
    index = 0
    for agent in agents:
        for _ in range(SIM.init_papers_per_agent):
            index += 1
            paper = Paper(
                author=agent,
                current_ac=rng.uniform(SIM.init_ac_min, SIM.init_ac_max),
                accrual_rate=rng.uniform(
                    SIM.init_accrual_min, SIM.init_accrual_max
                ),
            )
            paper.title = f"Paper {index}"
            Agent.all_papers.append(paper)


def build_env(
    *,
    backend,
    epsilon: float,
    learning: bool,
    num_rl: int,
    num_heuristic: int,
    horizon: int,
    seed: int,
    gamma: float = 0.95,
) -> tuple[Environment, list[QLearningAgent], list[HeuristicAgent]]:
    """Fresh env: shared-backend RL agents vs. heuristic opponents."""
    rng = random.Random(seed)
    Agent.all_papers = []

    rl_agents = [
        QLearningAgent(
            intrinsic_talent=1.0,
            forecast_horizon_days=horizon,
            name=f"RL {i}",
            backend=backend,
            epsilon=epsilon,
            learning=learning,
            gamma=gamma,
        )
        for i in range(num_rl)
    ]
    heuristics = [
        HeuristicAgent(intrinsic_talent=1.0, forecast_horizon_days=horizon,
                       name=f"Heuristic {i}")
        for i in range(num_heuristic)
    ]
    agents: list[Agent] = [*rl_agents, *heuristics]

    seed_initial_papers(agents, rng)
    env = Environment(agents=agents, papers=Agent.all_papers,
                      forecast_horizon_days=horizon)
    return env, rl_agents, heuristics


def mean_capital(agents) -> float:
    if not agents:
        return 0.0
    return sum(a.academic_capital for a in agents) / len(agents)


def train(args) -> None:
    backend = make_backend(args.backend, alpha=args.alpha)

    if args.load:
        backend.load(args.load)
        print(f"Loaded policy from {args.load}")
    elif args.episodes == 0:
        print("Warning: --episodes 0 with no --load evaluates an empty policy.")

    if args.episodes:
        print(f"Training: backend={args.backend} episodes={args.episodes} "
              f"days={args.days} rl={args.num_rl} heuristic={args.num_heuristic}")
    for episode in range(args.episodes):
        # Linear ε decay from eps_start to eps_end.
        frac = episode / max(1, args.episodes - 1)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        env, rl_agents, _ = build_env(
            backend=backend, epsilon=epsilon, learning=True,
            num_rl=args.num_rl, num_heuristic=args.num_heuristic,
            horizon=args.horizon, seed=args.seed + episode, gamma=args.gamma,
        )
        env.run(args.days)
        for agent in rl_agents:
            agent.end_episode()

        last = episode == args.episodes - 1
        if episode % max(1, args.episodes // 10) == 0 or last:
            print(f"  ep {episode:4d}  eps={epsilon:.3f}  "
                  f"RL mean AC={mean_capital(rl_agents):8.2f}")

    # Persist the trained policy unless told not to (nothing new if no training).
    if args.episodes and not args.no_save:
        save_path = args.save or default_policy_path(args.backend)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        backend.save(save_path)
        print(f"Saved policy to {save_path}")

    evaluate(backend, args)


def evaluate(backend, args) -> None:
    """Greedy (ε=0, no learning) RL agents vs. heuristics on a fresh seed."""
    env, rl_agents, heuristics = build_env(
        backend=backend, epsilon=0.0, learning=False,
        num_rl=args.num_rl, num_heuristic=args.num_heuristic,
        horizon=args.horizon, seed=args.seed + 10_000,
    )
    env.run(args.days)

    rl_ac = mean_capital(rl_agents)
    heur_ac = mean_capital(heuristics)
    print("\nEvaluation (greedy):")
    print(f"  RL mean AC        = {rl_ac:8.2f}")
    print(f"  Heuristic mean AC = {heur_ac:8.2f}")
    if heur_ac:
        print(f"  RL / Heuristic    = {rl_ac / heur_ac:6.2%}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the Liberata Q-learning agent."
    )
    p.add_argument("--backend", choices=["tabular", "linear"],
                   default=SIM.rl_backend)
    p.add_argument("--episodes", type=int, default=TRAIN.episodes)
    p.add_argument("--days", type=int, default=TRAIN.days)
    p.add_argument("--num-rl", dest="num_rl", type=int, default=TRAIN.num_rl)
    p.add_argument("--num-heuristic", dest="num_heuristic", type=int,
                   default=TRAIN.num_heuristic)
    p.add_argument("--horizon", type=int, default=SIM.forecast_horizon_days)
    p.add_argument("--alpha", type=float, default=None,
                   help="learning rate (defaults: 0.1 tabular / 0.01 linear)")
    p.add_argument("--gamma", type=float, default=SIM.rl_gamma)
    p.add_argument("--eps-start", dest="eps_start", type=float,
                   default=TRAIN.eps_start)
    p.add_argument("--eps-end", dest="eps_end", type=float,
                   default=TRAIN.eps_end)
    p.add_argument("--seed", type=int, default=SIM.seed)
    p.add_argument("--save", default=None,
                   help="policy save path (default: policies/policy_<backend>)")
    p.add_argument("--no-save", dest="no_save", action="store_true",
                   help="do not persist the trained policy")
    p.add_argument("--load", default=None,
                   help="load a saved policy before training/evaluating")
    args = p.parse_args(argv)
    if args.alpha is None:
        args.alpha = 0.1 if args.backend == "tabular" else 0.01
    return args


if __name__ == "__main__":
    train(parse_args())
