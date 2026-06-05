"""Archive a finished run into the static Pages gallery (``docs/``).

GitHub Pages serves files only, so each run is saved as a static
``docs/data/<run_id>/history.json`` (the existing ``History.to_json`` output) and
appended to ``docs/data/index.json``, the manifest the gallery reads to populate
its run picker. No backend is involved at view time.
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
SUMMARY_KEYS = ("total_capital", "mean_capital", "capital_gini", "num_papers")


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


def export_run(
    history: "History",
    *,
    config: dict[str, Any],
    title: str | None = None,
    docs_dir: str = "docs",
) -> str:
    """Write ``history`` into the gallery and update the manifest. Returns the run id."""
    data_dir = os.path.join(docs_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    run_id = _unique_run_id(data_dir)
    run_dir = os.path.join(data_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    history.to_json(os.path.join(run_dir, "history.json"))

    entry = {
        "id": run_id,
        "title": title or f"Run {run_id}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "num_days": len(history.days),
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
