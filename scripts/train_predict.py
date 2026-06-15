#!/usr/bin/env python3
"""Train a baseline tabular model and write submission.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_sklearn_preprocessor, prepare_feature_frames
from src.io_utils import artifacts_dir, build_schema_summary, project_root
from src.join_inference import safe_merge_tables
from src.modeling import (
    detect_third_party_booster_status,
    fit_best_model,
    predict_global_strategy,
    predict_per_block_strategy,
    run_strategy_selection,
)


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


def build_training_table(
    train_target: Optional[pd.DataFrame],
    train_features: Optional[pd.DataFrame],
    join_keys: Optional[List[str]],
    warnings: List[str],
) -> Optional[pd.DataFrame]:
    """Construct the training table using safe join-key inference."""
    if train_target is None:
        return None
    if train_features is None:
        return train_target.copy()
    merged, merge_warnings = safe_merge_tables(
        train_target,
        train_features,
        join_keys,
        how="inner",
        preserve_left_rows=False,
    )
    warnings.extend(merge_warnings)
    if join_keys is None:
        warnings.append("training feature merge skipped; using train_target only")
        return train_target.copy()
    return merged


def build_prediction_table(
    sample_submission: pd.DataFrame,
    validation_features: Optional[pd.DataFrame],
    join_keys: Optional[List[str]],
    warnings: List[str],
) -> pd.DataFrame:
    """Construct the prediction table preserving sample_submission row order."""
    merged, merge_warnings = safe_merge_tables(
        sample_submission,
        validation_features,
        join_keys,
        how="left",
        preserve_left_rows=True,
    )
    warnings.extend(merge_warnings)
    return merged


def resolve_training_target_column(
    train_df: pd.DataFrame,
    preferred: Optional[str],
    row_id_col: Optional[str],
) -> Optional[str]:
    """Resolve the target column name present in the training table."""
    if preferred and preferred in train_df.columns:
        return preferred

    for key in ("target", "label", "y", "response", "outcome", "value", "rate", "score", "target_value"):
        if key in train_df.columns:
            return key

    numeric_cols = [
        c
        for c in train_df.columns
        if c != row_id_col and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    return numeric_cols[-1] if numeric_cols else None


def fallback_predictions(
    sample_submission: pd.DataFrame,
    train_df: pd.DataFrame,
    train_target_col: str,
    block_col: Optional[str],
    metric_spec: Dict[str, Any],
) -> np.ndarray:
    """Generate median-based fallback predictions."""
    y = pd.to_numeric(train_df[train_target_col], errors="coerce")
    global_median = float(y.median()) if y.notna().any() else 0.0
    preds = np.full(len(sample_submission), global_median, dtype=float)

    if (
        block_col
        and block_col in sample_submission.columns
        and block_col in train_df.columns
        and metric_spec.get("metric_mode") == "block_averaged"
    ):
        block_medians = train_df.groupby(block_col)[train_target_col].median()
        for i, block_value in enumerate(sample_submission[block_col].tolist()):
            key = str(block_value)
            if key in block_medians.index.astype(str).tolist() or block_value in block_medians.index:
                try:
                    preds[i] = float(block_medians.loc[block_value])
                except Exception:
                    if key in block_medians:
                        preds[i] = float(block_medians[key])
    return preds


def validate_predictions(preds: np.ndarray) -> List[str]:
    """Return warnings for invalid predictions."""
    warnings: List[str] = []
    if len(preds) == 0:
        warnings.append("no predictions generated")
    if not np.all(np.isfinite(preds)):
        warnings.append("predictions contain non-finite values")
    return warnings


def print_summary(
    summary: Dict[str, Any],
    modeling_log: Dict[str, Any],
    submission_path: Optional[Path],
    warnings: List[str],
) -> None:
    """Print a concise training and prediction summary."""
    print("Sentinelle train/predict")
    print(f"  status: {modeling_log.get('status')}")
    print(f"  modeling_strategy: {modeling_log.get('modeling_strategy')}")
    print(f"  metric_mode: {(modeling_log.get('metric_spec') or {}).get('metric_mode')}")
    print(f"  selected_model: {modeling_log.get('selected_model')}")
    print(f"  n_train_rows: {modeling_log.get('n_train_rows')}")
    print(f"  n_predictions: {modeling_log.get('n_predictions')}")
    print(f"  row_id_column: {summary.get('row_id_column')}")
    print(f"  target_column: {summary.get('target_column')}")
    print(f"  submission_columns: {summary.get('submission_columns')}")
    if submission_path is not None:
        print(f"  wrote: {submission_path}")
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")


def main() -> int:
    """Run baseline modeling and write submission.csv when possible."""
    root = project_root()
    warnings: List[str] = []
    modeling_log: Dict[str, Any] = {"status": "failed"}

    summary = load_schema_summary(root)
    metric_spec = summary.get("metric_spec") or {
        "metric_name": "mae",
        "metric_mode": "global",
        "higher_is_better": False,
    }
    block_col = metric_spec.get("block_column")
    join_keys_info = summary.get("join_keys") or {}

    row_id_col = summary.get("row_id_column")
    target_col = summary.get("target_column")
    submission_columns = summary.get("submission_columns")

    sample_path = profile_path(summary, "sample_submission")
    train_target_path = profile_path(summary, "train_target")
    train_features_path = profile_path(summary, "train_features")
    val_features_path = profile_path(summary, "validation_features")

    if sample_path is None or not sample_path.exists():
        warnings.append("sample_submission file not found; cannot create submission.csv")
        print_summary(summary, modeling_log, None, warnings)
        return 1

    if target_col is None:
        warnings.append("target column not identified")
        print_summary(summary, modeling_log, None, warnings)
        return 1

    if submission_columns is None or len(submission_columns) < 2:
        if row_id_col and target_col:
            submission_columns = [row_id_col, target_col]
        else:
            submission_columns = list(load_csv(sample_path).columns[:2])
        warnings.append("submission columns inferred from available metadata")

    row_id_col = submission_columns[0]
    submission_target_col = submission_columns[1]

    sample_submission = load_csv(sample_path)
    train_target = load_csv(train_target_path) if train_target_path and train_target_path.exists() else None
    train_features = load_csv(train_features_path) if train_features_path and train_features_path.exists() else None
    validation_features = load_csv(val_features_path) if val_features_path and val_features_path.exists() else None

    train_df = build_training_table(
        train_target,
        train_features,
        join_keys_info.get("train_keys"),
        warnings,
    )
    if train_df is None:
        warnings.append("training target file not found; cannot train model")
        print_summary(summary, modeling_log, None, warnings)
        return 1

    train_target_col = resolve_training_target_column(train_df, target_col, row_id_col)
    if train_target_col is None:
        warnings.append("could not resolve training target column")
        print_summary(summary, modeling_log, None, warnings)
        return 1
    if train_target_col != submission_target_col:
        warnings.append(
            f"training target column '{train_target_col}' differs from submission target '{submission_target_col}'"
        )

    train_df = train_df.dropna(subset=[train_target_col]).copy()
    if train_df.empty:
        warnings.append("no training rows with valid target values")
        print_summary(summary, modeling_log, None, warnings)
        return 1

    predict_df = build_prediction_table(
        sample_submission,
        validation_features,
        join_keys_info.get("predict_keys"),
        warnings,
    )

    blocks_train = (
        train_df[block_col].reset_index(drop=True)
        if block_col and block_col in train_df.columns
        else None
    )
    blocks_predict = (
        predict_df[block_col].reset_index(drop=True)
        if block_col and block_col in predict_df.columns
        else None
    )

    train_feature_df = train_df.drop(columns=[block_col], errors="ignore") if block_col else train_df
    predict_feature_df = predict_df.drop(columns=[submission_target_col, block_col], errors="ignore")

    preds: np.ndarray
    selected_model = "dummy_median"
    fit_meta: Dict[str, Any] = {}
    strategy_selection: Dict[str, Any] = {}
    cv_results = pd.DataFrame()
    fitted_artifact: Any = None
    used_fallback = False
    modeling_status = "ok"
    model_fit_failed = False
    fallback_reason = ""
    column_groups: Dict[str, Any] = {}

    try:
        join_keys_for_images = (join_keys_info.get("train_keys") or []) + (join_keys_info.get("predict_keys") or [])
        X_train, y_train, X_test, column_groups = prepare_feature_frames(
            train_feature_df,
            predict_feature_df,
            target_col=train_target_col,
            row_id_col=row_id_col,
            join_key_columns=list(dict.fromkeys(join_keys_for_images)),
        )

        train_mask = y_train.notna() & np.isfinite(y_train.to_numpy(dtype=float))
        X_train = X_train.loc[train_mask].reset_index(drop=True)
        y_train = y_train.loc[train_mask].reset_index(drop=True)
        if blocks_train is not None:
            blocks_train = blocks_train.iloc[train_mask.to_numpy()].reset_index(drop=True)

        preprocessor = build_sklearn_preprocessor(X_train, column_groups=column_groups)
        strategy_selection = run_strategy_selection(
            X_train,
            y_train,
            blocks_train,
            preprocessor,
            metric_spec,
            block_col,
            random_state=42,
        )
        warnings.extend(strategy_selection.get("warnings", []))
        cv_results = strategy_selection["cv_results"]

        fitted_artifact, selected_model, fit_meta = fit_best_model(
            X_train,
            y_train,
            preprocessor,
            cv_results,
            random_state=42,
            strategy_selection=strategy_selection,
            blocks=blocks_train,
        )

        fallback_value = float(fit_meta.get("fallback_value", np.median(y_train)))
        if strategy_selection.get("modeling_strategy") == "per_block" and isinstance(fitted_artifact, dict):
            preds = predict_per_block_strategy(
                fitted_artifact.get("fitted_by_block", {}),
                fitted_artifact.get("block_medians", {}),
                X_test,
                blocks_predict if blocks_predict is not None else pd.Series([None] * len(X_test)),
                fallback_value,
            )
        else:
            preds = predict_global_strategy(fitted_artifact, X_test, fallback_value=fallback_value)
    except Exception as exc:
        modeling_status = "fallback"
        model_fit_failed = True
        fallback_reason = f"{type(exc).__name__}: {exc}"
        warnings.append(f"model training failed ({fallback_reason}); using fallback predictions")

        preds = fallback_predictions(
            sample_submission,
            train_df,
            train_target_col,
            block_col,
            metric_spec,
        )

        selected_model = "median_fallback"
        fit_meta = {
            "fallback_value": float(np.median(train_df[train_target_col])),
            "fallback_reason": fallback_reason,
        }
        used_fallback = True

        # The final prediction source is fallback. Do not let stale CV-selected
        # strategy metadata pretend that the fitted model was per-block/global.
        strategy_selection = {
            "modeling_strategy": "fallback",
            "global_strategy_score": strategy_selection.get("global_strategy_score") if isinstance(strategy_selection, dict) else None,
            "per_block_strategy_score": strategy_selection.get("per_block_strategy_score") if isinstance(strategy_selection, dict) else None,
            "selected_model_by_block": None,
        }

    y_train_numeric = pd.to_numeric(train_df[train_target_col], errors="coerce").to_numpy(dtype=float)
    finite_train = y_train_numeric[np.isfinite(y_train_numeric)]
    if len(finite_train) > 0 and np.min(finite_train) >= 0:
        preds = np.clip(preds, 0.0, None)

    invalid = ~np.isfinite(preds)
    if invalid.any():
        fallback_value = float(fit_meta.get("fallback_value", np.median(finite_train) if len(finite_train) else 0.0))
        preds = preds.copy()
        preds[invalid] = fallback_value
        warnings.append("replaced non-finite predictions with fallback values")
        used_fallback = True

    pred_warnings = validate_predictions(preds)
    warnings.extend(pred_warnings)

    if len(preds) != len(sample_submission):
        warnings.append(
            f"prediction length ({len(preds)}) does not match sample_submission ({len(sample_submission)})"
        )
        preds = np.resize(preds, len(sample_submission))

    submission = sample_submission[[row_id_col]].copy()
    submission[submission_target_col] = preds
    submission = submission[submission_columns]

    submission_path = root / "submission.csv"
    submission.to_csv(submission_path, index=False)

    adir = artifacts_dir(root)
    cv_path = adir / "cv_results.csv"
    if not cv_results.empty:
        cv_results.to_csv(cv_path, index=False)

    final_strategy = "fallback" if model_fit_failed else strategy_selection.get("modeling_strategy", "global")
    final_selected_by_block = None if model_fit_failed else strategy_selection.get("selected_model_by_block")

    modeling_log = {
        "status": modeling_status,
        "modeling_strategy": final_strategy,
        "metric_spec": metric_spec,
        "global_strategy_score": strategy_selection.get("global_strategy_score"),
        "per_block_strategy_score": strategy_selection.get("per_block_strategy_score"),
        "selected_model": selected_model,
        "selected_model_by_block": final_selected_by_block,
        "block_column": block_col,
        "block_values": metric_spec.get("block_values"),
        "join_keys": join_keys_info,
        "n_train_rows": int(len(train_df)),
        "n_predictions": int(len(submission)),
        "row_id_column": row_id_col,
        "target_column": submission_target_col,
        "training_target_column": train_target_col,
        "submission_columns": submission_columns,
        "column_groups": column_groups if isinstance(column_groups, dict) else {},
        "fit_metadata": fit_meta,
        "used_fallback": used_fallback,
        "model_fit_failed": model_fit_failed,
        "fallback_reason": fallback_reason,
        "final_prediction_source": "median_fallback" if model_fit_failed else selected_model,
        "third_party_booster_status": detect_third_party_booster_status(),
        "warnings": warnings,
        "submission_path": str(submission_path),
        "cv_results_path": str(cv_path) if not cv_results.empty else None,
    }
    log_path = adir / "modeling_log.json"
    log_path.write_text(json.dumps(modeling_log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print_summary(summary, modeling_log, submission_path, warnings)
    print(f"  wrote: {log_path}")
    if not cv_results.empty:
        print(f"  wrote: {cv_path}")
    return 0


if __name__ == "__main__":
    # Some sklearn/joblib backends can leave non-daemon worker threads alive in
    # constrained notebook/evaluation containers. This script writes all files
    # before returning, so force process termination after flushing streams.
    import os

    exit_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
