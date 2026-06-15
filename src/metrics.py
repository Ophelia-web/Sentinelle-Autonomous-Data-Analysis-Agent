"""Regression metrics for model evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.Series, list]


def mean_absolute_error_np(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute MAE while ignoring non-finite pairs."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true_arr[mask] - y_pred_arr[mask])))


def mean_absolute_error_safe(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Backward-compatible alias for global MAE."""
    return mean_absolute_error_np(y_true, y_pred)


def _normalize_block_values(values: Optional[Sequence[Any]]) -> Optional[list[str]]:
    """Normalize official scoring block values."""
    if not values:
        return None
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip().strip("`'\" ")
        if not text:
            continue
        cleaned.append(text.lower())
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned or None


def block_averaged_mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    blocks: ArrayLike,
    scoring_blocks: Optional[Sequence[Any]] = None,
) -> float:
    """Compute unweighted mean of per-block MAEs.

    If scoring_blocks is provided, only those official scoring blocks are used.
    This is important when training data contains extra categories that are not
    part of the official submission/evaluation blocks.
    """
    y_true_arr = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    y_pred_arr = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
    block_arr = np.asarray(blocks)

    frame = pd.DataFrame({"y_true": y_true_arr, "y_pred": y_pred_arr, "block": block_arr})
    frame = frame[np.isfinite(frame["y_true"]) & np.isfinite(frame["y_pred"])]
    if frame.empty:
        return float("nan")

    official = _normalize_block_values(scoring_blocks)
    if official:
        frame["_block_norm"] = frame["block"].astype(str).str.strip().str.lower()
        frame = frame[frame["_block_norm"].isin(official)]
        if frame.empty:
            return float("nan")

    frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
    per_block_mae = frame.groupby("block", dropna=False)["abs_error"].mean()
    if per_block_mae.empty:
        return float("nan")
    return float(per_block_mae.mean())


def score_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_spec: Optional[Dict[str, Any]] = None,
    block_values: Optional[ArrayLike] = None,
    *,
    block: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Score predictions using the detected official metric."""
    blocks = block_values if block_values is not None else block
    spec = metric_spec or {
        "metric_name": "mae",
        "metric_mode": "global",
        "higher_is_better": False,
    }

    global_mae = mean_absolute_error_np(y_true, y_pred)
    result: Dict[str, Any] = {
        "metric_name": spec.get("metric_name", "mae"),
        "metric_mode": spec.get("metric_mode", "global"),
        "score": global_mae,
        "mae": global_mae,
        "higher_is_better": bool(spec.get("higher_is_better", False)),
    }

    if spec.get("metric_mode") == "block_averaged" and blocks is not None:
        official_blocks = spec.get("block_values") or None
        block_score = block_averaged_mae(
            y_true,
            y_pred,
            blocks,
            scoring_blocks=official_blocks,
        )
        result["block_averaged_mae"] = block_score
        result["score"] = block_score
        result["mae"] = block_score

    return result
