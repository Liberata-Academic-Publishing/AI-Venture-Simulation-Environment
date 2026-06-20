"""Adaptive surplus sweep: tune accrual bumps until authors and reviewers both win.

Each step trains RL (100 episodes), runs a simulation, logs surplus metrics to a
live CSV (flushed every run for Excel), and adapts bump parameters for the next
step. Archives to the gallery every N runs.

Usage:
    python3 -u sweep_surplus.py
    python3 -u sweep_surplus.py 2>&1 | tee runs/sweep_surplus.log
    python3 sweep_surplus.py --max-runs 15 --archive-every 5
"""

from __future__ import annotations

import os
import sys

# Force line-buffered stdout/stderr when piped (e.g. through ``tee``).
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any

import config as config_module
import Paper
from config import SIM
from train_rl import parse_args as parse_train_args, train as run_training
from run_simulation import OUTPUT_DIR, parse_args as parse_sim_args, _run_once


# One-time RL reward fix (not swept).
RL_REWARD_AC_WEIGHT = 0.5
RL_REWARD_ACCRUAL_WEIGHT = 0.5
RL_REWARD_RANK_WEIGHT = 0.0

TRAIN_EPISODES = 100
HARD_CAP_RUNS = 30
STALL_RUNS = 3
NEGATIVE_STREAK_FOR_PERMANENT = 3
MIN_REVIEWER_SHARE = 0.01
MAX_QUALITY_PRICE_SCALE = 3.0


@dataclass
class BumpParams:
    min_review_accrual_bump: float = 0.05
    max_review_accrual_bump: float = 0.35
    review_bump_duration: str = "decay"
    review_bump_decay_rate: float = 0.05
    review_bump_decay_cap_timesteps: float | None = 100.0
    default_max_reviewer_share: float = 0.10
    review_sigmoid_saturation_effort: float = 18.0
    review_sigmoid_midpoint: float = 3.0
    history_price_scale: float = 0.5
    quality_price_scale: float = 1.0


def params_equal(a: BumpParams, b: BumpParams) -> bool:
    return asdict(a) == asdict(b)


def bump_params_from_config() -> BumpParams:
    return BumpParams(
        min_review_accrual_bump=SIM.min_review_accrual_bump,
        max_review_accrual_bump=SIM.max_review_accrual_bump,
        review_bump_duration=SIM.review_bump_duration,
        review_bump_decay_rate=SIM.review_bump_decay_rate,
        review_bump_decay_cap_timesteps=SIM.review_bump_decay_cap_timesteps,
        default_max_reviewer_share=SIM.default_max_reviewer_share,
        review_sigmoid_saturation_effort=SIM.review_sigmoid_saturation_effort,
        review_sigmoid_midpoint=SIM.review_sigmoid_midpoint,
        history_price_scale=SIM.history_price_scale,
        quality_price_scale=SIM.quality_price_scale,
    )


def apply_sim_overrides(
    params: BumpParams,
    *,
    rl_reward_ac_weight: float = RL_REWARD_AC_WEIGHT,
    rl_reward_accrual_weight: float = RL_REWARD_ACCRUAL_WEIGHT,
    rl_reward_rank_weight: float = RL_REWARD_RANK_WEIGHT,
) -> None:
    """Patch runtime SIM and re-sync Paper module-level constants."""
    config_module.SIM = replace(
        config_module.SIM,
        min_review_accrual_bump=params.min_review_accrual_bump,
        max_review_accrual_bump=params.max_review_accrual_bump,
        review_bump_duration=params.review_bump_duration,
        review_bump_decay_rate=params.review_bump_decay_rate,
        review_bump_decay_cap_timesteps=params.review_bump_decay_cap_timesteps,
        default_max_reviewer_share=params.default_max_reviewer_share,
        review_sigmoid_saturation_effort=params.review_sigmoid_saturation_effort,
        review_sigmoid_midpoint=params.review_sigmoid_midpoint,
        history_price_scale=params.history_price_scale,
        quality_price_scale=params.quality_price_scale,
        rl_reward_ac_weight=rl_reward_ac_weight,
        rl_reward_accrual_weight=rl_reward_accrual_weight,
        rl_reward_rank_weight=rl_reward_rank_weight,
    )
    sim = config_module.SIM
    Paper.MIN_REVIEW_ACCRUAL_BUMP = sim.min_review_accrual_bump
    Paper.MAX_REVIEW_ACCRUAL_BUMP = sim.max_review_accrual_bump
    Paper.DEFAULT_MAX_REVIEWER_SHARE = sim.default_max_reviewer_share
    Paper.REVIEW_SIGMOID_SATURATION_EFFORT = sim.review_sigmoid_saturation_effort
    Paper.REVIEW_SIGMOID_MIDPOINT = sim.review_sigmoid_midpoint
    Paper.HISTORY_PRICE_SCALE = sim.history_price_scale
    Paper.QUALITY_PRICE_SCALE = sim.quality_price_scale


def _decay_cap_cli(value: float | None) -> str:
    if value is None:
        return "none"
    return str(int(value)) if float(value).is_integer() else str(value)


def build_train_args(seed: int) -> argparse.Namespace:
    argv = [
        "--episodes", str(TRAIN_EPISODES),
        "--no-archive",
        "--seed", str(seed),
    ]
    return parse_train_args(argv)


def build_sim_args(
    params: BumpParams,
    *,
    seed: int,
    archive: bool,
    title: str | None,
) -> argparse.Namespace:
    argv = [
        "--quiet",
        "--rl-agents", "20",
        "--heuristic-agents", "10",
        "--random-agents", "10",
        "--probabilistic-agents", "0",
        "--seed", str(seed),
        "--review-bump-duration", params.review_bump_duration,
        "--review-bump-decay-rate", str(params.review_bump_decay_rate),
        "--review-bump-decay-cap", _decay_cap_cli(params.review_bump_decay_cap_timesteps),
    ]
    if archive and title:
        argv.extend(["--name", title])
    else:
        argv.append("--no-archive")
    return parse_sim_args(argv)


@dataclass
class SurplusMetrics:
    author_net_good: float = 0.0
    author_net_bad: float = 0.0
    author_net_total: float = 0.0
    reviewer_benefit_good: float = 0.0
    reviewer_benefit_bad: float = 0.0
    reviewer_total: float = 0.0
    value_created_good: float = 0.0
    value_created_bad: float = 0.0
    value_created_total: float = 0.0
    gap: float = 0.0
    both_positive: bool = False


def read_surplus_metrics(history_path: str | None = None) -> SurplusMetrics:
    path = history_path or os.path.join(OUTPUT_DIR, "history.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    scalars = data.get("scalars", {})

    def last(key: str) -> float:
        values = scalars.get(key) or []
        return float(values[-1]) if values else 0.0

    author_good = last("author_net_good")
    author_bad = last("author_net_bad")
    author_total = author_good + author_bad
    reviewer_good = last("reviewer_benefit_good")
    reviewer_bad = last("reviewer_benefit_bad")
    reviewer_total = reviewer_good + reviewer_bad
    value_good = last("value_created_good")
    value_bad = last("value_created_bad")
    value_total = value_good + value_bad
    gap = reviewer_total - value_total

    return SurplusMetrics(
        author_net_good=author_good,
        author_net_bad=author_bad,
        author_net_total=author_total,
        reviewer_benefit_good=reviewer_good,
        reviewer_benefit_bad=reviewer_bad,
        reviewer_total=reviewer_total,
        value_created_good=value_good,
        value_created_bad=value_bad,
        value_created_total=value_total,
        gap=gap,
        both_positive=author_total > 0.0 and reviewer_total > 0.0,
    )


def propose_next_params(
    params: BumpParams,
    metrics: SurplusMetrics,
    *,
    negative_streak: int,
) -> tuple[BumpParams, str]:
    """Return updated bump params and a human-readable adjustment note."""
    notes: list[str] = []
    next_params = replace(params)

    if metrics.author_net_total < 0.0:
        old_max = next_params.max_review_accrual_bump
        next_params.max_review_accrual_bump = min(
            0.60, next_params.max_review_accrual_bump + 0.05
        )
        if next_params.max_review_accrual_bump != old_max:
            notes.append(f"max_bump {old_max:.2f}->{next_params.max_review_accrual_bump:.2f}")

        old_min = next_params.min_review_accrual_bump
        next_params.min_review_accrual_bump = min(
            0.15, next_params.min_review_accrual_bump + 0.02
        )
        if next_params.min_review_accrual_bump != old_min:
            notes.append(f"min_bump {old_min:.2f}->{next_params.min_review_accrual_bump:.2f}")

        old_share = next_params.default_max_reviewer_share
        next_params.default_max_reviewer_share = max(
            MIN_REVIEWER_SHARE, next_params.default_max_reviewer_share - 0.01
        )
        if next_params.default_max_reviewer_share != old_share:
            notes.append(
                f"max_share {old_share:.3f}->{next_params.default_max_reviewer_share:.3f}"
            )

        if next_params.review_bump_duration == "decay":
            if negative_streak >= NEGATIVE_STREAK_FOR_PERMANENT:
                next_params.review_bump_duration = "permanent"
                notes.append("duration decay->permanent")
            else:
                old_rate = next_params.review_bump_decay_rate
                next_params.review_bump_decay_rate = max(
                    0.005, next_params.review_bump_decay_rate * 0.5
                )
                if next_params.review_bump_decay_rate != old_rate:
                    notes.append(
                        f"decay_rate {old_rate:.3f}->{next_params.review_bump_decay_rate:.3f}"
                    )
                cap = next_params.review_bump_decay_cap_timesteps
                if cap is not None:
                    new_cap = min(2000.0, cap * 2.0)
                    next_params.review_bump_decay_cap_timesteps = new_cap
                    notes.append(f"decay_cap {cap:.0f}->{new_cap:.0f}")

        old_mid = next_params.review_sigmoid_midpoint
        next_params.review_sigmoid_midpoint = max(
            1.0, next_params.review_sigmoid_midpoint - 0.1
        )
        if next_params.review_sigmoid_midpoint != old_mid:
            notes.append(
                f"sigmoid_mid {old_mid:.1f}->{next_params.review_sigmoid_midpoint:.1f}"
            )

        # When bump/share knobs are exhausted, keep pushing author-fair pricing.
        if not notes or (
            next_params.max_review_accrual_bump >= 0.60 - 1e-9
            and next_params.min_review_accrual_bump >= 0.15 - 1e-9
            and next_params.default_max_reviewer_share <= MIN_REVIEWER_SHARE + 1e-9
            and next_params.review_sigmoid_midpoint <= 1.0 + 1e-9
        ):
            old_quality = next_params.quality_price_scale
            next_params.quality_price_scale = min(
                MAX_QUALITY_PRICE_SCALE, next_params.quality_price_scale + 0.25
            )
            if next_params.quality_price_scale != old_quality:
                notes.append(
                    f"quality_price {old_quality:.2f}->"
                    f"{next_params.quality_price_scale:.2f}"
                )
            old_history = next_params.history_price_scale
            next_params.history_price_scale = max(
                0.0, next_params.history_price_scale - 0.1
            )
            if next_params.history_price_scale != old_history:
                notes.append(
                    f"history_price {old_history:.2f}->"
                    f"{next_params.history_price_scale:.2f}"
                )
            if next_params.default_max_reviewer_share > MIN_REVIEWER_SHARE + 1e-9:
                old_share = next_params.default_max_reviewer_share
                next_params.default_max_reviewer_share = max(
                    MIN_REVIEWER_SHARE, next_params.default_max_reviewer_share - 0.005
                )
                if next_params.default_max_reviewer_share != old_share:
                    notes.append(
                        f"max_share {old_share:.3f}->"
                        f"{next_params.default_max_reviewer_share:.3f}"
                    )

    elif metrics.reviewer_total <= 0.0:
        old_share = next_params.default_max_reviewer_share
        next_params.default_max_reviewer_share = min(
            0.15, next_params.default_max_reviewer_share + 0.005
        )
        notes.append(f"max_share {old_share:.3f}->{next_params.default_max_reviewer_share:.3f}")

    if metrics.value_created_total > 1e-9 and metrics.gap > 2.0 * metrics.value_created_total:
        old_share = next_params.default_max_reviewer_share
        next_params.default_max_reviewer_share = max(
            MIN_REVIEWER_SHARE, next_params.default_max_reviewer_share - 0.015
        )
        if next_params.default_max_reviewer_share != old_share:
            notes.append(
                f"gap_cut max_share {old_share:.3f}->{next_params.default_max_reviewer_share:.3f}"
            )

    if not notes:
        notes.append("no change")
    return next_params, "; ".join(notes)


def _csv_fieldnames() -> list[str]:
    return [
        "run",
        "timestamp",
        "min_review_accrual_bump",
        "max_review_accrual_bump",
        "review_bump_duration",
        "review_bump_decay_rate",
        "review_bump_decay_cap_timesteps",
        "default_max_reviewer_share",
        "review_sigmoid_midpoint",
        "review_sigmoid_saturation_effort",
        "history_price_scale",
        "quality_price_scale",
        "rl_reward_ac_weight",
        "rl_reward_accrual_weight",
        "rl_reward_rank_weight",
        "author_net_good",
        "author_net_bad",
        "author_net_total",
        "reviewer_benefit_good",
        "reviewer_benefit_bad",
        "reviewer_total",
        "value_created_total",
        "gap",
        "both_positive",
        "author_net_delta",
        "adjustment_notes",
        "archived",
    ]


def append_csv_row(path: str, row: dict[str, Any], *, write_header: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = _csv_fieldnames()
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def _archive_title(run: int, params: BumpParams, metrics: SurplusMetrics) -> str:
    return (
        f"surplus sweep run {run}: "
        f"max_bump={params.max_review_accrual_bump:.2f} "
        f"share={params.default_max_reviewer_share:.3f} "
        f"dur={params.review_bump_duration} "
        f"author_net={metrics.author_net_total:.0f} "
        f"reviewer={metrics.reviewer_total:.0f}"
    )


def _score_run(metrics: SurplusMetrics) -> float:
    """Higher is better; success runs rank highest."""
    if metrics.both_positive:
        return 1_000_000.0 + metrics.author_net_total + metrics.reviewer_total
    return metrics.author_net_total + 0.1 * metrics.reviewer_total


def parse_sweep_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive surplus parameter sweep.")
    parser.add_argument("--max-runs", type=int, default=20, metavar="N",
                        help="Default stop after N runs (may extend to 30 if improving).")
    parser.add_argument("--archive-every", type=int, default=5, metavar="N",
                        help="Archive simulation to gallery every N runs.")
    parser.add_argument("--output", default=os.path.join("runs", "sweep_surplus.csv"),
                        help="Live CSV log path (flushed after each run).")
    parser.add_argument("--seed", type=int, default=SIM.seed,
                        help="Base random seed (incremented per run).")
    return parser.parse_args(argv)


def _log(msg: str) -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_sweep_args(argv)
    params = bump_params_from_config()
    apply_sim_overrides(params)

    _log("Adaptive surplus sweep")
    _log(f"  max_runs={args.max_runs} (hard cap {HARD_CAP_RUNS})")
    _log(f"  train_episodes={TRAIN_EPISODES}")
    _log("  agents: 20 RL + 10 heuristic + 10 random")
    _log(f"  RL reward (fixed): ac={RL_REWARD_AC_WEIGHT}, "
          f"accrual={RL_REWARD_ACCRUAL_WEIGHT}, rank={RL_REWARD_RANK_WEIGHT}")
    _log(f"  CSV: {args.output}")
    _log(f"  archive every {args.archive_every} runs")
    _log("")

    prev_author_net: float | None = None
    negative_streak = 0
    improve_streak = 0
    stall_streak = 0
    effective_max = args.max_runs
    best_run: dict[str, Any] | None = None
    best_score = float("-inf")
    csv_exists = os.path.exists(args.output)

    for run in range(1, HARD_CAP_RUNS + 1):
        if run > effective_max:
            break

        seed = args.seed + run
        apply_sim_overrides(params)
        _log(f"=== Run {run}/{effective_max} (seed={seed}) ===")
        _log(
            f"  bumps: min={params.min_review_accrual_bump:.2f} "
            f"max={params.max_review_accrual_bump:.2f} "
            f"duration={params.review_bump_duration} "
            f"decay_rate={params.review_bump_decay_rate} "
            f"decay_cap={params.review_bump_decay_cap_timesteps} "
            f"max_share={params.default_max_reviewer_share:.3f} "
            f"sigmoid_mid={params.review_sigmoid_midpoint:.1f} "
            f"sat_effort={params.review_sigmoid_saturation_effort:.1f} "
            f"hist_price={params.history_price_scale:.2f} "
            f"qual_price={params.quality_price_scale:.2f}"
        )

        _log("  Training RL...")
        _log(
            "  (100 episodes x 1000 timesteps each — first training "
            "progress line may take several minutes)"
        )
        run_training(build_train_args(seed))

        archive = args.archive_every > 0 and run % args.archive_every == 0
        title = (
            (
                f"surplus sweep run {run}: "
                f"max_bump={params.max_review_accrual_bump:.2f} "
                f"share={params.default_max_reviewer_share:.3f} "
                f"dur={params.review_bump_duration}"
            )
            if archive
            else None
        )

        _log("  Running simulation...")
        sim_args = build_sim_args(params, seed=seed, archive=archive, title=title)
        _run_once(sim_args, seed=seed, title=title if archive else None)

        metrics = read_surplus_metrics()
        author_delta = (
            0.0 if prev_author_net is None
            else metrics.author_net_total - prev_author_net
        )

        row = {
            "run": run,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **asdict(params),
            "rl_reward_ac_weight": RL_REWARD_AC_WEIGHT,
            "rl_reward_accrual_weight": RL_REWARD_ACCRUAL_WEIGHT,
            "rl_reward_rank_weight": RL_REWARD_RANK_WEIGHT,
            "author_net_good": round(metrics.author_net_good, 2),
            "author_net_bad": round(metrics.author_net_bad, 2),
            "author_net_total": round(metrics.author_net_total, 2),
            "reviewer_benefit_good": round(metrics.reviewer_benefit_good, 2),
            "reviewer_benefit_bad": round(metrics.reviewer_benefit_bad, 2),
            "reviewer_total": round(metrics.reviewer_total, 2),
            "value_created_total": round(metrics.value_created_total, 2),
            "gap": round(metrics.gap, 2),
            "both_positive": metrics.both_positive,
            "author_net_delta": round(author_delta, 2),
            "adjustment_notes": "",
            "archived": archive,
        }

        score = _score_run(metrics)
        if score > best_score:
            best_score = score
            best_run = {**row, "metrics": metrics}

        _log(
            f"  author_net={metrics.author_net_total:,.1f} "
            f"reviewer={metrics.reviewer_total:,.1f} "
            f"value_created={metrics.value_created_total:,.1f} "
            f"both_positive={metrics.both_positive}"
        )

        if metrics.both_positive:
            row["adjustment_notes"] = "success; stopping"
            append_csv_row(args.output, row, write_header=not csv_exists)
            csv_exists = True
            _log("\nSuccess: both author and reviewer surplus are positive.")
            break

        next_params, notes = propose_next_params(
            params, metrics, negative_streak=negative_streak
        )
        row["adjustment_notes"] = notes
        append_csv_row(args.output, row, write_header=not csv_exists)
        csv_exists = True
        _log(f"  next adjustment: {notes}")

        if notes == "no change" or params_equal(params, next_params):
            _log(
                "\nStopping: no further parameters to adjust. "
                "Bump/share/pricing knobs exhausted — authors still negative. "
                "May need a market code change (share on future value only)."
            )
            break

        if metrics.author_net_total < 0.0:
            negative_streak += 1
        else:
            negative_streak = 0

        if prev_author_net is not None and metrics.author_net_total > prev_author_net + 1e-6:
            improve_streak += 1
            stall_streak = 0
        elif prev_author_net is not None:
            improve_streak = 0
            stall_streak += 1

        if (
            run >= args.max_runs
            and improve_streak >= 3
            and effective_max < HARD_CAP_RUNS
        ):
            effective_max = HARD_CAP_RUNS
            _log(f"  extending to {HARD_CAP_RUNS} runs (author_net improving 3x)")

        if run >= effective_max and stall_streak >= STALL_RUNS:
            _log(f"\nStopping: no author_net improvement for {STALL_RUNS} runs.")
            break

        prev_author_net = metrics.author_net_total
        params = next_params
        _log("")

    _log("\n=== Sweep finished ===")
    _log(f"CSV: {os.path.abspath(args.output)}")
    if best_run:
        m = best_run["metrics"]
        _log(
            f"Best run {best_run['run']}: author_net={m.author_net_total:,.1f} "
            f"reviewer={m.reviewer_total:,.1f} both_positive={m.both_positive}"
        )
        _log(
            f"  params: max_bump={best_run['max_review_accrual_bump']:.2f} "
            f"min_bump={best_run['min_review_accrual_bump']:.2f} "
            f"max_share={best_run['default_max_reviewer_share']:.3f} "
            f"duration={best_run['review_bump_duration']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
