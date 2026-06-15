"""One-time migration to the local-full + gallery-slim layout.

For every run under ``docs/data/<id>/`` this:
  * picks the fullest available source (``history.full.json`` if present, else
    the committed ``history.json``),
  * copies the full history to ``local_data/<id>/history.json`` (gitignored),
  * renders the dark-themed static-chart PNGs into ``docs/data/<id>/charts/``,
  * rewrites ``docs/data/<id>/history.json`` as the slim gallery payload
    (plus the ``charts`` list of produced PNGs),
  * deletes ``docs/data/<id>/history.full.json``.

No simulations are re-run: the raw per-event arrays already live in the source
files. Re-running this script is safe; already-migrated runs are skipped.

Usage:
    python migrate_gallery.py
"""

from __future__ import annotations

import json
import os

from export_run import LOCAL_DATA_DIR, write_full_history, write_gallery_run
from History import History


def _thresholds_by_id(docs_data: str) -> dict[str, float]:
    """Map run id -> its configured ``min_review_effort_threshold`` (manifest)."""
    manifest_path = os.path.join(docs_data, "index.json")
    out: dict[str, float] = {}
    if not os.path.exists(manifest_path):
        return out
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            runs = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return out
    for entry in runs if isinstance(runs, list) else []:
        config = entry.get("config") or {}
        if "min_review_effort_threshold" in config:
            out[entry.get("id")] = float(config["min_review_effort_threshold"])
    return out


def migrate_run(
    run_dir: str, *, local_dir: str = LOCAL_DATA_DIR, threshold: float | None = None
) -> bool:
    """Migrate one ``docs/data/<id>`` directory. Returns True if migrated."""
    run_id = os.path.basename(os.path.normpath(run_dir))
    full_path = os.path.join(run_dir, "history.full.json")
    slim_path = os.path.join(run_dir, "history.json")

    source = full_path if os.path.exists(full_path) else slim_path
    if not os.path.exists(source):
        print(f"  skip {run_id}: no history found")
        return False

    with open(source, encoding="utf-8") as fh:
        data = json.load(fh)

    # Already migrated: slim file (has `charts`) and no full source to rebuild
    # from. Skip so we don't overwrite the local full archive with slim data.
    if not os.path.exists(full_path) and isinstance(data, dict) and "charts" in data:
        print(f"  skip {run_id}: already migrated")
        return False

    history = History.from_dict(data)
    write_full_history(history, run_id, local_dir)
    charts = write_gallery_run(history, run_dir, threshold=threshold)

    if os.path.exists(full_path):
        os.remove(full_path)

    print(f"  migrated {run_id}: {len(charts)} chart(s)")
    return True


def main() -> None:
    docs_data = os.path.join("docs", "data")
    if not os.path.isdir(docs_data):
        print(f"No {docs_data} directory; nothing to migrate.")
        return

    thresholds = _thresholds_by_id(docs_data)
    run_dirs = sorted(
        os.path.join(docs_data, name)
        for name in os.listdir(docs_data)
        if os.path.isdir(os.path.join(docs_data, name))
    )
    print(f"Migrating {len(run_dirs)} run(s) under {docs_data} ...")
    migrated = sum(
        migrate_run(
            run_dir, threshold=thresholds.get(os.path.basename(run_dir))
        )
        for run_dir in run_dirs
    )
    print(f"Done. Migrated {migrated} run(s). Full data archived in {LOCAL_DATA_DIR}/.")


if __name__ == "__main__":
    main()
