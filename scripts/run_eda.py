#!/usr/bin/env python3
"""Run automated exploratory data analysis and write artifacts/eda/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda import run_eda
from src.io_utils import artifacts_dir, build_schema_summary, project_root


def main() -> int:
    """Load schema artifacts, run EDA, and write eda_summary.json."""
    root = project_root()
    adir = artifacts_dir(root)
    schema_path = adir / "schema_summary.json"

    if schema_path.exists():
        schema_summary = json.loads(schema_path.read_text(encoding="utf-8"))
    else:
        print("schema_summary.json missing; building schema summary on demand")
        schema_summary = build_schema_summary(root=root)
        if not schema_summary.get("files"):
            print("Cannot run EDA without schema information")
            return 1

    modeling_log_path = adir / "modeling_log.json"
    modeling_log = None
    if modeling_log_path.exists():
        modeling_log = json.loads(modeling_log_path.read_text(encoding="utf-8"))

    summary = run_eda(schema_summary, modeling_log=modeling_log, root=root)

    print("Sentinelle EDA")
    print(f"  status: {summary.get('status')}")
    print(f"  output_dir: {summary.get('output_dir')}")
    print(f"  tables: {len(summary.get('tables') or [])}")
    print(f"  figures: {len(summary.get('figures') or [])}")

    warnings = summary.get("warnings") or []
    if warnings:
        print("  warnings:")
        for warning in warnings[:10]:
            print(f"    - {warning}")
        if len(warnings) > 10:
            print(f"    - ... (+{len(warnings) - 10} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
