"""Archive a finished run into the static Pages gallery (``docs/``).

GitHub Pages serves files only, so each run is saved statically and read at view
time with no backend. To keep the committed gallery small, the heavy *static*
charts are rendered to dark-themed PNGs (``docs/data/<run_id>/charts/*.png``) and
the per-run ``docs/data/<run_id>/history.json`` carries only the slim data the
*animated* charts, the action feed/replay, and the side tables need. The full,
lossless history for every run is written to ``local_data/<run_id>/history.json``
(gitignored) so diagrams can always be rebuilt later. ``docs/data/index.json``
is the manifest the gallery reads to populate its run picker.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from History import History

# Final-day scalars surfaced in the manifest so the picker can show a quick
# summary without loading each run's full history.json.
SUMMARY_KEYS = (
    "total_capital",
    "mean_capital",
    "capital_gini",
    "num_papers",
    "completed_peer_reviews",
    "good_faith_reviews",
    "bad_faith_reviews",
    "fair_market_price",
    "mean_peer_review_epsilon",
)

# Full, lossless per-run history lives here (gitignored). The committed gallery
# only carries the slim JSON + chart PNGs.
LOCAL_DATA_DIR = "local_data"


def _unique_run_id(data_dir: str) -> str:
    """Timestamp-based id, with a numeric suffix if that folder already exists."""
    base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id, n = base, 2
    while os.path.exists(os.path.join(data_dir, run_id)):
        run_id = f"{base}_{n}"
        n += 1
    return run_id


def _summary(history: "History") -> dict[str, float]:
    out: dict[str, float] = {}
    for key in SUMMARY_KEYS:
        series = history.scalars.get(key) or []
        if series:
            out[key] = round(float(series[-1]), 4)
    return out


def write_full_history(history: "History", run_id: str, local_dir: str = LOCAL_DATA_DIR) -> str:
    """Write the full, lossless history to ``local_dir/<run_id>/history.json``."""
    run_dir = os.path.join(local_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "history.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history.to_dict(), fh)  # compact; this is the local archive
    return path


def write_gallery_run(
    history: "History", run_dir: str, *, threshold: float | None = None
) -> list[str]:
    """Render the static-chart PNGs + slim ``history.json`` into ``run_dir``.

    ``threshold`` is the run's ``min_review_effort_threshold`` so the review
    effort charts mark the reward cliff at the value that run used.

    Returns the list of chart names produced (empty if matplotlib is missing).
    """
    os.makedirs(run_dir, exist_ok=True)
    charts: list[str] = []
    try:
        import visualize

        kwargs = {} if threshold is None else {"threshold": threshold}
        charts = visualize.render_gallery_charts(
            history, os.path.join(run_dir, "charts"), **kwargs
        )
    except ImportError:
        print(
            "matplotlib not installed; skipping static chart PNGs. "
            "Install it with: python -m pip install matplotlib"
        )

    payload = history.to_gallery_dict()
    payload["charts"] = charts
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)  # no indent: gallery payload stays small
    return charts


def export_run(
    history: "History",
    *,
    config: dict[str, Any],
    title: str | None = None,
    docs_dir: str = "docs",
    local_dir: str = LOCAL_DATA_DIR,
) -> str:
    """Archive ``history``: full data local, slim JSON + PNGs to the gallery.

    Returns the run id.
    """
    data_dir = os.path.join(docs_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    run_id = _unique_run_id(data_dir)

    # Full, lossless history -> local (gitignored) archive.
    write_full_history(history, run_id, local_dir)

    # Slim JSON + static-chart PNGs -> committed gallery. The review effort
    # charts mark the reward cliff at this run's configured threshold.
    run_dir = os.path.join(data_dir, run_id)
    write_gallery_run(history, run_dir, threshold=config.get("min_review_effort_threshold"))

    entry = {
        "id": run_id,
        "title": title or f"Run {run_id}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "num_timesteps": len(history.timesteps),
        "num_days": len(history.timesteps),  # alias for older gallery builds
        "config": config,
        "summary": _summary(history),
    }

    manifest_path = os.path.join(data_dir, "index.json")
    runs: list[dict[str, Any]] = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, list):
                runs = loaded
        except (json.JSONDecodeError, OSError):
            runs = []  # start fresh if the manifest is missing/corrupt

    runs.append(entry)
    runs.sort(key=lambda r: r.get("created", ""), reverse=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=2)

    return run_id
