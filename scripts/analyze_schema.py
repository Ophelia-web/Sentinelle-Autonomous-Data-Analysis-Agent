#!/usr/bin/env python3
"""Scan ./data, infer CSV roles and columns, and write schema summary artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import build_schema_summary, project_root, save_schema_summary


def _role_path(summary: Dict[str, Any], role: str) -> str:
    profile = (summary.get("inferred_roles") or {}).get(role)
    if profile is None:
        return "not identified"
    return str(profile.get("rel_path", "unknown"))


def print_concise_summary(summary: Dict[str, Any], json_path: Path, md_path: Path) -> None:
    """Print a human-readable summary to stdout."""
    print("Sentinelle schema analysis")
    print(f"  status: {summary.get('status')}")
    print(f"  data_dir: {summary.get('data_dir')}")
    print(f"  data_description: {summary.get('data_description_path') or 'not found'}")
    print(f"  csv_files: {summary.get('n_csv_files')}")
    print(f"  row_id_column: {summary.get('row_id_column')}")
    print(f"  target_column: {summary.get('target_column')}")
    print(f"  submission_columns: {summary.get('submission_columns')}")
    print("  inferred_roles:")
    for role in ("sample_submission", "train_target", "train_features", "validation_features"):
        print(f"    {role}: {_role_path(summary, role)}")

    warnings: List[str] = summary.get("warnings") or []
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    print(f"  wrote: {json_path}")
    print(f"  wrote: {md_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer dataset schema from files under ./data and write artifacts.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (default: repository root inferred from src layout).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run schema inference and persist artifacts. Always exits successfully unless argparse fails."""
    args = parse_args(argv)
    root = args.root or project_root()

    summary = build_schema_summary(root=root)
    json_path, md_path = save_schema_summary(summary, root=root)
    print_concise_summary(summary, json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
