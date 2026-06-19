"""Re-render static chart PNGs for gallery runs without re-simulating.

Loads each run's history (preferring ``local_data/<id>/history.json`` when
present), renders any charts the updated ``visualize`` module can produce,
merges them with charts already on disk, and updates ``history.json``'s
``charts`` list.

Usage:
    python refresh_gallery_charts.py
    python refresh_gallery_charts.py --run 2026-06-19_004503
"""

from __future__ import annotations

import argparse
import json
import os

from export_run import LOCAL_DATA_DIR
from History import History
from migrate_gallery import _thresholds_by_id


def _load_history(run_id: str, run_dir: str, local_dir: str) -> History:
    local_path = os.path.join(local_dir, run_id, "history.json")
    slim_path = os.path.join(run_dir, "history.json")
    source = local_path if os.path.exists(local_path) else slim_path
    with open(source, encoding="utf-8") as fh:
        return History.from_dict(json.load(fh))


def _charts_on_disk(charts_dir: str) -> list[str]:
    if not os.path.isdir(charts_dir):
        return []
    return sorted(
        name[:-4]
        for name in os.listdir(charts_dir)
        if name.endswith(".png")
    )


def refresh_run(
    run_dir: str,
    *,
    local_dir: str = LOCAL_DATA_DIR,
    threshold: float | None = None,
) -> list[str]:
    """Refresh one ``docs/data/<id>`` directory. Returns the merged chart list."""
    run_id = os.path.basename(os.path.normpath(run_dir))
    slim_path = os.path.join(run_dir, "history.json")
    if not os.path.exists(slim_path):
        print(f"  skip {run_id}: no history.json")
        return []

    with open(slim_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    history = _load_history(run_id, run_dir, local_dir)
    charts_dir = os.path.join(run_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    produced: list[str] = []
    try:
        import visualize

        kwargs = {} if threshold is None else {"threshold": threshold}
        produced = visualize.render_gallery_charts(history, charts_dir, **kwargs)
    except ImportError:
        print(
            f"  skip {run_id}: matplotlib not installed "
            "(python -m pip install matplotlib)"
        )
        return list(payload.get("charts") or [])

    merged = sorted(set(payload.get("charts") or []) | set(produced) | set(_charts_on_disk(charts_dir)))
    added = sorted(set(produced) - set(payload.get("charts") or []))
    payload["charts"] = merged
    with open(slim_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    added = sorted(set(produced) - set(payload.get("charts") or []))
    print(f"  refreshed {run_id}: {len(produced)} rendered, {len(merged)} total chart(s)")
    if added:
        print(f"    new: {', '.join(added)}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        help="Refresh a single run id (default: all runs under docs/data/)",
    )
    args = parser.parse_args()

    docs_data = os.path.join("docs", "data")
    if not os.path.isdir(docs_data):
        print(f"No {docs_data} directory; nothing to refresh.")
        return

    thresholds = _thresholds_by_id(docs_data)
    if args.run:
        run_dirs = [os.path.join(docs_data, args.run)]
    else:
        run_dirs = sorted(
            os.path.join(docs_data, name)
            for name in os.listdir(docs_data)
            if os.path.isdir(os.path.join(docs_data, name))
        )

    print(f"Refreshing charts for {len(run_dirs)} run(s) ...")
    for run_dir in run_dirs:
        refresh_run(
            run_dir,
            threshold=thresholds.get(os.path.basename(run_dir)),
        )
    print("Done.")


if __name__ == "__main__":
    main()
