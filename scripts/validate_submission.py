#!/usr/bin/env python3
"""Validate submission.csv against the inferred schema and sample submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import artifacts_dir, build_schema_summary, project_root


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with a latin-1 fallback."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")


def load_schema_summary(root: Path) -> Dict[str, Any]:
    """Load schema summary from artifacts or build it on demand."""
    schema_path = artifacts_dir(root) / "schema_summary.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    return build_schema_summary(root=root)


def profile_path(summary: Dict[str, Any], role: str) -> Optional[Path]:
    """Return the filesystem path for an inferred role, if available."""
    profile = (summary.get("inferred_roles") or {}).get(role)
    if not profile:
        return None
    path = profile.get("path")
    return Path(path) if path else None


def infer_submission_columns(
    sample_df: pd.DataFrame,
    row_id_column: Optional[str],
    target_column: Optional[str],
    schema_columns: Optional[List[str]],
) -> List[str]:
    """Infer required submission columns from schema summary or sample submission."""
    if schema_columns and len(schema_columns) >= 2:
        return list(schema_columns)

    if row_id_column and target_column and row_id_column in sample_df.columns and target_column in sample_df.columns:
        return [row_id_column, target_column]

    if len(sample_df.columns) >= 2:
        return [sample_df.columns[0], sample_df.columns[-1]]

    return list(sample_df.columns)


def validation_report_to_markdown(report: Dict[str, Any]) -> str:
    """Render a validation report as Markdown."""
    lines = [
        "# Validation Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Submission path: `{report.get('submission_path')}`",
        f"- Sample submission path: `{report.get('sample_submission_path')}`",
        f"- Row id column: `{report.get('row_id_column')}`",
        f"- Target column: `{report.get('target_column')}`",
        f"- Required columns: `{report.get('required_columns')}`",
        f"- Submission rows: `{report.get('n_submission_rows')}`",
        f"- Sample rows: `{report.get('n_sample_rows')}`",
        "",
    ]

    errors = report.get("errors") or []
    if errors:
        lines.append("## Errors")
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")

    warnings = report.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def save_validation_report(report: Dict[str, Any], root: Path) -> tuple[Path, Path]:
    """Write validation report artifacts."""
    adir = artifacts_dir(root)
    json_path = adir / "validation_report.json"
    md_path = adir / "validation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path.write_text(validation_report_to_markdown(report), encoding="utf-8")
    return json_path, md_path


def print_validation_summary(report: Dict[str, Any], json_path: Path, md_path: Path) -> None:
    """Print a concise validation summary to stdout."""
    print("Sentinelle submission validation")
    print(f"  status: {report.get('status')}")
    print(f"  submission_path: {report.get('submission_path')}")
    print(f"  sample_submission_path: {report.get('sample_submission_path')}")
    print(f"  required_columns: {report.get('required_columns')}")
    print(f"  n_submission_rows: {report.get('n_submission_rows')}")
    print(f"  n_sample_rows: {report.get('n_sample_rows')}")

    errors = report.get("errors") or []
    if errors:
        print("  errors:")
        for error in errors:
            print(f"    - {error}")

    warnings = report.get("warnings") or []
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    print(f"  wrote: {json_path}")
    print(f"  wrote: {md_path}")


def validate_submission(root: Path) -> Dict[str, Any]:
    """Validate repository submission.csv against inferred schema requirements."""
    errors: List[str] = []
    warnings: List[str] = []

    submission_path = root / "submission.csv"
    summary = load_schema_summary(root)

    row_id_column = summary.get("row_id_column")
    target_column = summary.get("target_column")
    required_columns = summary.get("submission_columns")

    sample_path = profile_path(summary, "sample_submission")
    sample_submission_path = str(sample_path) if sample_path else None

    report: Dict[str, Any] = {
        "status": "failed",
        "submission_path": str(submission_path),
        "sample_submission_path": sample_submission_path,
        "row_id_column": row_id_column,
        "target_column": target_column,
        "required_columns": required_columns,
        "n_submission_rows": None,
        "n_sample_rows": None,
        "errors": errors,
        "warnings": warnings,
    }

    if not submission_path.exists():
        errors.append("submission.csv does not exist in repository root")
        report["errors"] = errors
        return report

    if sample_path is None or not sample_path.exists():
        errors.append("sample_submission file not found under data/")
        report["errors"] = errors
        return report

    if row_id_column is None:
        errors.append("row_id_column could not be inferred from schema summary")
    if target_column is None:
        errors.append("target_column could not be inferred from schema summary")

    try:
        submission_df = load_csv(submission_path)
        sample_df = load_csv(sample_path)
    except Exception as exc:
        errors.append(f"failed to load CSV files: {type(exc).__name__}: {exc}")
        report["errors"] = errors
        return report

    report["n_submission_rows"] = int(len(submission_df))
    report["n_sample_rows"] = int(len(sample_df))

    required_columns = infer_submission_columns(
        sample_df,
        row_id_column,
        target_column,
        required_columns if isinstance(required_columns, list) else None,
    )
    report["required_columns"] = required_columns
    report["row_id_column"] = required_columns[0]
    report["target_column"] = required_columns[1]

    row_id_column = required_columns[0]
    target_column = required_columns[1]

    submission_cols = list(submission_df.columns)
    if submission_cols != required_columns:
        if set(submission_cols) != set(required_columns):
            missing = [c for c in required_columns if c not in submission_cols]
            extra = [c for c in submission_cols if c not in required_columns]
            if missing:
                errors.append(f"missing required columns: {missing}")
            if extra:
                errors.append(f"unexpected extra columns: {extra}")
        else:
            errors.append(
                f"submission columns {submission_cols} do not exactly match required order {required_columns}"
            )

    for col in required_columns:
        if col not in submission_df.columns:
            errors.append(f"required column missing: {col}")

    if row_id_column not in submission_df.columns:
        errors.append(f"row_id column not found: {row_id_column}")
    if target_column not in submission_df.columns:
        errors.append(f"prediction/target column not found: {target_column}")

    if row_id_column in submission_df.columns:
        submission_ids = submission_df[row_id_column]
        if submission_ids.isna().any():
            errors.append("submission row_id column contains missing values")
        if submission_ids.duplicated().any():
            dup_count = int(submission_ids.duplicated().sum())
            errors.append(f"submission row_id column contains {dup_count} duplicate values")

    if target_column in submission_df.columns:
        target_series = pd.to_numeric(submission_df[target_column], errors="coerce")
        if not pd.api.types.is_numeric_dtype(submission_df[target_column]) and target_series.isna().all():
            errors.append(f"prediction column is not numeric: {target_column}")
        if target_series.isna().any():
            errors.append("prediction column contains NaN values")
        if np.isinf(target_series.to_numpy(dtype=float)).any():
            errors.append("prediction column contains infinite values")

    if row_id_column in sample_df.columns and row_id_column in submission_df.columns:
        sample_ids = sample_df[row_id_column]
        submission_ids = submission_df[row_id_column]

        missing_ids = sorted(set(sample_ids.dropna()) - set(submission_ids.dropna()))
        extra_ids = sorted(set(submission_ids.dropna()) - set(sample_ids.dropna()))

        if missing_ids:
            errors.append(f"submission is missing {len(missing_ids)} row_id values present in sample_submission")
        if extra_ids:
            errors.append(
                f"submission contains {len(extra_ids)} row_id values not present in sample_submission"
            )
        
        if len(submission_ids) == len(sample_ids) and not submission_ids.reset_index(drop=True).equals(
            sample_ids.reset_index(drop=True)
        ):
            errors.append("submission row_id order does not match sample_submission order")
    if len(submission_df) != len(sample_df):
        errors.append(
            f"submission row count ({len(submission_df)}) does not equal sample_submission row count ({len(sample_df)})"
        )

    placeholder_cols = [c for c in sample_df.columns if c not in required_columns]
    if placeholder_cols:
        warnings.append(
            f"sample_submission contains additional placeholder columns ignored for validation: {placeholder_cols}"
        )

    if not required_columns:
        errors.append("required submission columns could not be determined")

    report["errors"] = errors
    report["warnings"] = warnings
    report["status"] = "passed" if not errors else "failed"
    return report


def main() -> int:
    """Run submission validation and write report artifacts."""
    root = project_root()
    report = validate_submission(root)
    json_path, md_path = save_validation_report(report, root)
    print_validation_summary(report, json_path, md_path)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
