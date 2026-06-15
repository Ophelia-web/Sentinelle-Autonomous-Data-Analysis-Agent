"""Model selection, cross-validation, and prediction helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.metrics import mean_absolute_error_np, score_predictions

MIN_BLOCK_ROWS = 15
DEFAULT_CV_TIME_BUDGET_SEC = 20 * 60


@dataclass
class ModelCandidate:
    name: str
    estimator: Any
    family: str
    priority: int = 100
    is_heavy: bool = False
    is_third_party: bool = False


def _cv_time_budget_sec() -> float:
    """Return the CV time budget in seconds, configurable by environment."""
    raw = os.environ.get("SENTINELLE_CV_TIME_BUDGET_SEC")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return float(DEFAULT_CV_TIME_BUDGET_SEC)


def _sklearn_model_candidates(random_state: int) -> List[ModelCandidate]:
    """Return CPU-safe sklearn model candidates with small hyperparameter grids."""
    candidates: List[ModelCandidate] = []

    candidates.append(
        ModelCandidate(
            name="dummy_median",
            estimator=DummyRegressor(strategy="median"),
            family="dummy",
            priority=900,
            is_heavy=False,
        )
    )

    for alpha in (0.1, 1.0, 10.0):
        candidates.append(
            ModelCandidate(
                name=f"ridge_alpha{alpha:g}",
                estimator=Ridge(alpha=alpha, random_state=random_state),
                family="ridge",
                priority=300,
                is_heavy=False,
            )
        )

    for alpha, l1_ratio in ((0.001, 0.1), (0.01, 0.3), (0.1, 0.5)):
        candidates.append(
            ModelCandidate(
                name=f"elastic_net_alpha{alpha:g}_l1{l1_ratio:g}",
                estimator=ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=5000,
                    random_state=random_state,
                ),
                family="elastic_net",
                priority=320,
                is_heavy=False,
            )
        )

    candidates.append(
        ModelCandidate(
            name="huber_epsilon1p35",
            estimator=HuberRegressor(epsilon=1.35, max_iter=300),
            family="huber",
            priority=420,
            is_heavy=False,
        )
    )

    for max_depth, min_leaf, n_estimators in (
        (6, 2, 80),
        (10, 2, 100),
        (None, 2, 140),
    ):
        depth_label = "none" if max_depth is None else str(max_depth)
        candidates.append(
            ModelCandidate(
                name=f"extra_trees_depth{depth_label}_leaf{min_leaf}_n{n_estimators}",
                estimator=ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    random_state=random_state,
                    n_jobs=1,
                ),
                family="extra_trees",
                priority=120,
                is_heavy=(max_depth is None or n_estimators >= 140),
            )
        )

    for max_depth, min_leaf, n_estimators in (
        (6, 2, 80),
        (10, 2, 100),
        (None, 2, 140),
    ):
        depth_label = "none" if max_depth is None else str(max_depth)
        candidates.append(
            ModelCandidate(
                name=f"random_forest_depth{depth_label}_leaf{min_leaf}_n{n_estimators}",
                estimator=RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    random_state=random_state,
                    n_jobs=1,
                ),
                family="random_forest",
                priority=150,
                is_heavy=(max_depth is None or n_estimators >= 140),
            )
        )

    for n_estimators, learning_rate, max_depth in (
        (60, 0.05, 2),
        (80, 0.05, 3),
        (120, 0.03, 3),
    ):
        candidates.append(
            ModelCandidate(
                name=f"gradient_boosting_n{n_estimators}_lr{learning_rate:g}_depth{max_depth}",
                estimator=GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
                family="gradient_boosting",
                priority=220,
                is_heavy=n_estimators >= 120,
            )
        )

    candidates.append(
        ModelCandidate(
            name="bagging_tree_depth8_n30",
            estimator=BaggingRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=random_state,
                ),
                n_estimators=30,
                random_state=random_state,
                n_jobs=1,
            ),
            family="bagging_tree",
            priority=500,
            is_heavy=True,
        )
    )

    return candidates


def detect_third_party_booster_status() -> Dict[str, str]:
    """Record whether optional third-party boosters can be imported."""
    booster_status: Dict[str, str] = {
        "xgboost": "not_installed",
        "lightgbm": "not_installed",
        "catboost": "not_installed",
    }

    try:
        from xgboost import XGBRegressor  # noqa: F401

        booster_status["xgboost"] = "available"
    except Exception as exc:
        booster_status["xgboost"] = f"unavailable: {type(exc).__name__}"

    try:
        from lightgbm import LGBMRegressor  # noqa: F401

        booster_status["lightgbm"] = "available"
    except Exception as exc:
        booster_status["lightgbm"] = f"unavailable: {type(exc).__name__}"

    try:
        from catboost import CatBoostRegressor  # noqa: F401

        booster_status["catboost"] = "available"
    except Exception as exc:
        booster_status["catboost"] = f"unavailable: {type(exc).__name__}"

    return booster_status


def _third_party_booster_candidates(random_state: int) -> List[ModelCandidate]:
    """Return third-party booster candidates if their packages are installed.

    These are auto-detected. No environment variable is required.
    Missing packages or incompatible versions must not crash the pipeline.
    """
    candidates: List[ModelCandidate] = []
    booster_status = detect_third_party_booster_status()

    if booster_status.get("xgboost") == "available":
        from xgboost import XGBRegressor

        for n_estimators, learning_rate, max_depth in (
            (120, 0.05, 3),
            (180, 0.03, 4),
        ):
            candidates.append(
                ModelCandidate(
                    name=f"xgboost_n{n_estimators}_lr{learning_rate:g}_depth{max_depth}",
                    estimator=XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=1,
                        verbosity=0,
                    ),
                    family="xgboost",
                    priority=40,
                    is_heavy=True,
                    is_third_party=True,
                )
            )

    if booster_status.get("lightgbm") == "available":
        from lightgbm import LGBMRegressor

        for n_estimators, learning_rate, num_leaves in (
            (150, 0.05, 31),
            (220, 0.03, 31),
        ):
            candidates.append(
                ModelCandidate(
                    name=f"lightgbm_n{n_estimators}_lr{learning_rate:g}_leaves{num_leaves}",
                    estimator=LGBMRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                    family="lightgbm",
                    priority=45,
                    is_heavy=True,
                    is_third_party=True,
                )
            )

    if booster_status.get("catboost") == "available":
        from catboost import CatBoostRegressor

        for iterations, learning_rate, depth in (
            (150, 0.05, 4),
            (220, 0.03, 6),
        ):
            candidates.append(
                ModelCandidate(
                    name=f"catboost_iter{iterations}_lr{learning_rate:g}_depth{depth}",
                    estimator=CatBoostRegressor(
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        loss_function="MAE",
                        random_seed=random_state,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=1,
                    ),
                    family="catboost",
                    priority=35,
                    is_heavy=True,
                    is_third_party=True,
                )
            )

    return candidates


def select_model_candidates(
    n_rows: int,
    n_features: int,
    strategy: str = "global",
    block_rows: Optional[int] = None,
    random_state: int = 42,
) -> List[ModelCandidate]:
    """Choose runtime-aware model candidates.

    Strong third-party boosters are prioritized if installed, but the total
    candidate count is capped to protect the 2-hour Award B runtime.
    """
    candidates = _sklearn_model_candidates(random_state)
    booster_candidates = _third_party_booster_candidates(random_state)

    small_data = n_rows < 5000 and n_features < 500
    medium_data = n_rows < 20000 and n_features < 1000

    selected: List[ModelCandidate] = []

    if strategy == "per_block":
        if block_rows is not None and block_rows < MIN_BLOCK_ROWS:
            allowed_families = {"dummy", "ridge", "elastic_net"}
            max_candidates = 4
        else:
            allowed_families = {
                "catboost",
                "xgboost",
                "lightgbm",
                "extra_trees",
                "random_forest",
                "ridge",
                "elastic_net",
                "dummy",
            }
            max_candidates = 8

        pool = booster_candidates + candidates
        for cand in sorted(pool, key=lambda c: c.priority):
            if cand.family not in allowed_families:
                continue
            if cand.is_heavy and not small_data and cand.is_third_party:
                # For larger per-block data, keep only one heavy third-party
                # candidate per booster family via priority ordering.
                pass
            selected.append(cand)
            if len(selected) >= max_candidates:
                break

        # Always include dummy median fallback.
        if not any(c.family == "dummy" for c in selected):
            selected.append(next(c for c in candidates if c.family == "dummy"))
        return selected

    # Global strategy.
    if small_data:
        max_candidates = 12
        pool = booster_candidates + candidates
    elif medium_data:
        max_candidates = 8
        pool = booster_candidates[:3] + [c for c in candidates if not c.is_heavy]
    else:
        max_candidates = 6
        pool = booster_candidates[:2] + [
            c
            for c in candidates
            if c.family in {"dummy", "ridge", "elastic_net", "extra_trees", "random_forest"}
            and not c.is_heavy
        ]

    for cand in sorted(pool, key=lambda c: c.priority):
        selected.append(cand)
        if len(selected) >= max_candidates:
            break

    if not any(c.family == "dummy" for c in selected):
        selected.append(next(c for c in candidates if c.family == "dummy"))

    # Deterministic de-duplication by name.
    deduped: List[ModelCandidate] = []
    seen = set()
    for cand in selected:
        if cand.name not in seen:
            deduped.append(cand)
            seen.add(cand.name)

    return deduped


def _all_model_candidates(random_state: int = 42) -> Dict[str, ModelCandidate]:
    """Lookup table of every defined candidate by name."""
    all_cands = _sklearn_model_candidates(random_state) + _third_party_booster_candidates(random_state)
    return {cand.name: cand for cand in all_cands}


def _estimator_for_name(model_name: str, random_state: int = 42) -> Any:
    """Resolve a fitted estimator from a CV-selected candidate name."""
    lookup = _all_model_candidates(random_state)
    if model_name in lookup:
        return clone(lookup[model_name].estimator)
    return DummyRegressor(strategy="median")


def select_candidate_models(
    n_rows: int,
    n_features: int,
    strategy: str = "global",
    block_rows: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Backward-compatible wrapper returning name -> estimator."""
    candidates = select_model_candidates(
        n_rows=n_rows,
        n_features=n_features,
        strategy=strategy,
        block_rows=block_rows,
        random_state=random_state,
    )
    return {cand.name: cand.estimator for cand in candidates}


def make_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Backward-compatible default model pool."""
    return select_candidate_models(
        n_rows=1000,
        n_features=100,
        strategy="global",
        random_state=random_state,
    )


def _choose_n_splits(n_rows: int) -> Optional[int]:
    """Choose CV folds with runtime safety for large datasets."""
    if n_rows >= 1000:
        return 3
    if n_rows >= 50:
        return 5
    if n_rows >= 20:
        return 3
    return None


def _metric_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    blocks: Optional[np.ndarray],
    metric_spec: Dict[str, Any],
) -> float:
    scored = score_predictions(y_true, y_pred, metric_spec, block_values=blocks)
    return float(scored["score"])


def _make_pipeline(preprocessor: Any, model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ]
    )


def _cv_model_scores(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    blocks: Optional[np.ndarray],
    preprocessor: Any,
    metric_spec: Dict[str, Any],
    random_state: int,
) -> Tuple[float, float, int, str, str]:
    n_rows = len(X)
    n_splits = _choose_n_splits(n_rows)
    if n_splits is None:
        return np.nan, np.nan, 0, "skipped_insufficient_rows", ""

    pipeline = _make_pipeline(preprocessor, model)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []

    try:
        for train_idx, val_idx in cv.split(X):
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y[val_idx]
            fold_blocks = blocks[val_idx] if blocks is not None else None

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            fold_scores.append(_metric_score(y_val, np.asarray(preds, dtype=float), fold_blocks, metric_spec))
        return float(np.mean(fold_scores)), float(np.std(fold_scores)), n_splits, "ok", ""
    except Exception as exc:
        return (
            np.nan,
            np.nan,
            n_splits,
            "failed",
            f"{type(exc).__name__}: {str(exc)[:180]}",
        )


def _cv_results_row(
    strategy: str,
    block: Optional[str],
    model: str,
    cv_folds: int,
    score_mean: float,
    score_std: float,
    metric_mode: str,
    status: str,
    *,
    is_selected: bool = False,
    strategy_score: Optional[float] = None,
    notes: str = "",
    model_family: str = "",
    is_third_party: bool = False,
    is_heavy: bool = False,
) -> Dict[str, Any]:
    return {
        "strategy": strategy,
        "block": block,
        "model": model,
        "cv_folds": cv_folds,
        "score_mean": score_mean,
        "score_std": score_std,
        "mae_mean": score_mean,
        "mae_std": score_std,
        "metric_mode": metric_mode,
        "status": status,
        "is_selected": is_selected,
        "strategy_score": strategy_score,
        "notes": notes,
        "model_family": model_family,
        "is_third_party": is_third_party,
        "is_heavy": is_heavy,
    }


def _candidate_meta_fields(candidate: Optional[ModelCandidate]) -> Dict[str, Any]:
    if candidate is None:
        return {"model_family": "", "is_third_party": False, "is_heavy": False}
    return {
        "model_family": candidate.family,
        "is_third_party": candidate.is_third_party,
        "is_heavy": candidate.is_heavy,
    }


def _global_block_label(metric_spec: Dict[str, Any]) -> Optional[str]:
    if metric_spec.get("metric_mode") == "block_averaged":
        return "**all_blocks**"
    return None


def _annotate_cv_results(
    cv_results: pd.DataFrame,
    final_strategy: str,
    selected_global_model: Optional[str],
    selected_by_block: Dict[str, str],
    global_score: float,
    per_block_score: float,
) -> pd.DataFrame:
    """Mark selected rows and ensure summary columns exist."""
    if cv_results.empty:
        return cv_results

    out = cv_results.copy()
    for col, default in (
        ("is_selected", False),
        ("strategy_score", np.nan),
        ("notes", ""),
    ):
        if col not in out.columns:
            out[col] = default

    for idx, row in out.iterrows():
        strategy = str(row.get("strategy", ""))
        block = row.get("block")
        model = str(row.get("model", ""))
        status = str(row.get("status", ""))

        if status == "strategy_summary":
            out.at[idx, "is_selected"] = (
                final_strategy == "per_block" and strategy == "per_block"
            ) or (final_strategy == "global" and strategy == "global")
            if strategy == "per_block":
                out.at[idx, "strategy_score"] = per_block_score
            elif strategy == "global":
                out.at[idx, "strategy_score"] = global_score
            continue

        if strategy == "global" and model == selected_global_model and status == "ok":
            out.at[idx, "is_selected"] = final_strategy == "global"
            out.at[idx, "strategy_score"] = global_score if final_strategy == "global" else np.nan
            continue

        if strategy == "per_block" and block is not None:
            block_key = str(block)
            if block_key in ("**overall**", "nan", "None"):
                continue
            if model == selected_by_block.get(block_key) and status in {"ok", "fallback_insufficient_rows"}:
                out.at[idx, "is_selected"] = final_strategy == "per_block"

    return out


def cross_validate_global_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    blocks: Optional[pd.Series],
    preprocessor: Any,
    metric_spec: Dict[str, Any],
    random_state: int = 42,
    start_time: Optional[float] = None,
) -> Tuple[pd.DataFrame, Optional[str], float]:
    """Cross-validate one global model per candidate using the official metric."""
    candidates = select_model_candidates(
        n_rows=len(X),
        n_features=X.shape[1],
        strategy="global",
        random_state=random_state,
    )
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    block_arr = blocks.to_numpy() if blocks is not None else None
    rows: List[Dict[str, Any]] = []
    best_name: Optional[str] = None
    best_score = float("inf")
    budget_start = start_time if start_time is not None else time.time()
    metric_mode = metric_spec.get("metric_mode", "global")
    block_label = _global_block_label(metric_spec)

    for cand in candidates:
        name = cand.name
        model = cand.estimator
        meta_fields = _candidate_meta_fields(cand)
        n_splits = _choose_n_splits(len(X)) or 0

        if time.time() - budget_start > _cv_time_budget_sec():
            rows.append(
                _cv_results_row(
                    "global",
                    block_label,
                    name,
                    0,
                    np.nan,
                    np.nan,
                    metric_mode,
                    "skipped_time_budget",
                    notes="Skipped due to CV time budget",
                    **meta_fields,
                )
            )
            continue

        try:
            score_mean, score_std, folds, status, cv_notes = _cv_model_scores(
                name,
                model,
                X,
                y_arr,
                block_arr,
                preprocessor,
                metric_spec,
                random_state,
            )
            row_notes = cv_notes or (
                "Block-averaged CV score" if metric_mode == "block_averaged" else ""
            )
            rows.append(
                _cv_results_row(
                    "global",
                    block_label,
                    name,
                    folds,
                    score_mean,
                    score_std,
                    metric_mode,
                    status,
                    notes=row_notes,
                    **meta_fields,
                )
            )
            if status == "ok" and np.isfinite(score_mean) and score_mean < best_score:
                best_score = score_mean
                best_name = name
        except Exception as exc:
            rows.append(
                _cv_results_row(
                    "global",
                    block_label,
                    name,
                    n_splits,
                    np.nan,
                    np.nan,
                    metric_mode,
                    "failed",
                    notes=f"{type(exc).__name__}: {str(exc)[:180]}",
                    **meta_fields,
                )
            )
            continue

    if best_name is None:
        best_name = "dummy_median"
        best_score = float("inf")
    return pd.DataFrame(rows), best_name, best_score


def cross_validate_per_block_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    blocks: pd.Series,
    preprocessor: Any,
    metric_spec: Dict[str, Any],
    random_state: int = 42,
    start_time: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], float, List[str]]:
    """Cross-validate separate models per block; overall score is mean block CV score."""
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    selected_by_block: Dict[str, str] = {}
    block_scores: List[float] = []
    budget_start = start_time if start_time is not None else time.time()
    metric_mode = "block_averaged"
    within_block_spec = {"metric_name": "mae", "metric_mode": "global", "higher_is_better": False}

    block_values = metric_spec.get("block_values") or []
    observed_blocks = list(blocks.dropna().unique())
    observed_by_text = {str(v).strip().lower(): v for v in observed_blocks}

    if block_values:
        unique_blocks = []
        missing_scoring_blocks = []
        for value in block_values:
            key = str(value).strip().lower()
            if key in observed_by_text:
                unique_blocks.append(observed_by_text[key])
            else:
                missing_scoring_blocks.append(str(value))

        if missing_scoring_blocks:
            warnings.append(
                "official scoring block values not found in training rows: "
                + ", ".join(missing_scoring_blocks)
            )

        # If official scoring blocks are named, do not add extra non-scoring
        # categories from the training file. They may be useful for training a
        # global model, but they should not contribute to the per-block strategy
        # score or per-block model list.
        if not unique_blocks:
            warnings.append(
                "no official scoring blocks were found in the block column; "
                "falling back to all observed blocks for per-block CV"
            )
            unique_blocks = sorted(observed_blocks, key=lambda x: str(x))
    else:
        unique_blocks = sorted(observed_blocks, key=lambda x: str(x))

    for block_value in unique_blocks:
        mask = (blocks == block_value).to_numpy()
        X_block = X.iloc[mask].reset_index(drop=True)
        y_block = pd.to_numeric(y.iloc[mask], errors="coerce").reset_index(drop=True)
        valid = (y_block.notna() & np.isfinite(y_block.to_numpy(dtype=float))).to_numpy()
        X_block = X_block.iloc[valid].reset_index(drop=True)
        y_arr = y_block.iloc[valid].to_numpy(dtype=float)
        block_key = str(block_value)

        if len(X_block) < MIN_BLOCK_ROWS:
            median_val = float(np.median(y_arr)) if len(y_arr) else float("nan")
            fallback_preds = np.full_like(y_arr, median_val, dtype=float)
            fallback_score = float(mean_absolute_error_np(y_arr, fallback_preds))
            rows.append(
                _cv_results_row(
                    "per_block",
                    block_key,
                    "median_fallback",
                    0,
                    fallback_score,
                    0.0,
                    metric_mode,
                    "fallback_insufficient_rows",
                    notes="Too few rows for CV; median fallback used",
                )
            )
            selected_by_block[block_key] = "median_fallback"
            if np.isfinite(fallback_score):
                block_scores.append(fallback_score)
            warnings.append(f"block '{block_value}' has too few rows; using median fallback")
            continue

        block_candidates = select_model_candidates(
            n_rows=len(X_block),
            n_features=X_block.shape[1],
            strategy="per_block",
            block_rows=len(X_block),
            random_state=random_state,
        )
        best_name = "dummy_median"
        best_score = float("inf")

        for cand in block_candidates:
            name = cand.name
            model = cand.estimator
            meta_fields = _candidate_meta_fields(cand)
            n_splits = _choose_n_splits(len(X_block)) or 0

            if time.time() - budget_start > _cv_time_budget_sec():
                rows.append(
                    _cv_results_row(
                        "per_block",
                        block_key,
                        name,
                        0,
                        np.nan,
                        np.nan,
                        metric_mode,
                        "skipped_time_budget",
                        notes="Skipped due to CV time budget",
                        **meta_fields,
                    )
                )
                continue

            try:
                score_mean, score_std, folds, status, cv_notes = _cv_model_scores(
                    name,
                    model,
                    X_block,
                    y_arr,
                    None,
                    preprocessor,
                    within_block_spec,
                    random_state,
                )
                rows.append(
                    _cv_results_row(
                        "per_block",
                        block_key,
                        name,
                        folds,
                        score_mean,
                        score_std,
                        metric_mode,
                        status,
                        notes=cv_notes or "Per-block within-block MAE",
                        **meta_fields,
                    )
                )
                if status == "ok" and np.isfinite(score_mean) and score_mean < best_score:
                    best_score = score_mean
                    best_name = name
            except Exception as exc:
                rows.append(
                    _cv_results_row(
                        "per_block",
                        block_key,
                        name,
                        n_splits,
                        np.nan,
                        np.nan,
                        metric_mode,
                        "failed",
                        notes=f"{type(exc).__name__}: {str(exc)[:180]}",
                        **meta_fields,
                    )
                )
                continue

        if not np.isfinite(best_score):
            median_val = float(np.median(y_arr)) if len(y_arr) else float("nan")
            fallback_preds = np.full_like(y_arr, median_val, dtype=float)
            fallback_score = float(mean_absolute_error_np(y_arr, fallback_preds))
            rows.append(
                _cv_results_row(
                    "per_block",
                    block_key,
                    "median_fallback",
                    0,
                    fallback_score,
                    0.0,
                    metric_mode,
                    "fallback_insufficient_rows",
                    notes="CV unavailable; median fallback used",
                )
            )
            selected_by_block[block_key] = "median_fallback"
            block_scores.append(fallback_score)
            warnings.append(f"block '{block_value}' CV unavailable; using median fallback score")
            continue

        selected_by_block[block_key] = best_name
        block_scores.append(best_score)

    overall = float(np.mean(block_scores)) if block_scores else float("inf")
    overall_std = float(np.std(block_scores)) if len(block_scores) > 1 else 0.0
    if block_scores:
        rows.append(
            _cv_results_row(
                "per_block",
                "**overall**",
                "mean_of_selected_block_models",
                0,
                overall,
                overall_std,
                metric_mode,
                "strategy_summary",
                strategy_score=overall,
                notes="Arithmetic mean of selected per-block CV MAEs",
            )
        )
    return pd.DataFrame(rows), selected_by_block, overall, warnings


def run_strategy_selection(
    X: pd.DataFrame,
    y: pd.Series,
    blocks: Optional[pd.Series],
    preprocessor: Any,
    metric_spec: Dict[str, Any],
    block_column: Optional[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compare global and per-block strategies and choose the better one."""
    warnings: List[str] = []
    start_time = time.time()
    block_averaged = metric_spec.get("metric_mode") == "block_averaged"

    per_block_enabled = (
        block_averaged
        and block_column is not None
        and blocks is not None
        and blocks.notna().any()
    )

    per_block_cv = pd.DataFrame()
    per_block_models: Dict[str, str] = {}
    per_block_score = float("inf")

    if block_averaged and not per_block_enabled:
        warnings.append(
            "Block-averaged scoring detected, but block column could not be confidently resolved; "
            "per-block modeling was not run."
        )

    if per_block_enabled:
        per_block_cv, per_block_models, per_block_score, block_warnings = cross_validate_per_block_strategy(
            X,
            y,
            blocks,
            preprocessor,
            metric_spec,
            random_state,
            start_time=start_time,
        )
        warnings.extend(block_warnings)

    global_cv = pd.DataFrame()
    global_model: Optional[str] = "dummy_median"
    global_score = float("inf")
    if time.time() - start_time <= _cv_time_budget_sec():
        global_cv, global_model, global_score = cross_validate_global_strategy(
            X,
            y,
            blocks,
            preprocessor,
            metric_spec,
            random_state,
            start_time=start_time,
        )
        if block_averaged and np.isfinite(global_score):
            global_cv = pd.concat(
                [
                    global_cv,
                    pd.DataFrame(
                        [
                            _cv_results_row(
                                "global",
                                "**all_blocks**",
                                global_model or "dummy_median",
                                0,
                                global_score,
                                0.0,
                                "block_averaged",
                                "strategy_summary",
                                strategy_score=global_score,
                                notes="Best global model scored with block-averaged MAE",
                            )
                        ]
                    ),
                ],
                ignore_index=True,
            )
    elif block_averaged:
        warnings.append("Global strategy comparison skipped or truncated due to CV time budget")

    if not block_averaged:
        pass  # global-only mode
    elif not per_block_enabled:
        warnings.append(
            "Per-block modeling skipped because block column could not be resolved; "
            "global models evaluated with block-averaged metric when blocks are unavailable."
        )

    if per_block_enabled and np.isfinite(per_block_score) and per_block_score < global_score:
        strategy = "per_block"
        selected_score = per_block_score
    else:
        strategy = "global"
        selected_score = global_score

    cv_parts = [df for df in (per_block_cv, global_cv) if not df.empty]
    cv_results = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
    cv_results = _annotate_cv_results(
        cv_results,
        strategy,
        global_model,
        per_block_models,
        global_score,
        per_block_score if per_block_enabled else float("inf"),
    )

    return {
        "modeling_strategy": strategy,
        "metric_spec": metric_spec,
        "global_strategy_score": global_score if np.isfinite(global_score) else None,
        "per_block_strategy_score": per_block_score if per_block_enabled else None,
        "selected_global_model": global_model,
        "selected_model_by_block": per_block_models if per_block_enabled else None,
        "cv_results": cv_results,
        "warnings": warnings,
        "selected_score": selected_score,
        "block_column": block_column,
    }


def fit_global_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    model_name: str,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    model = _estimator_for_name(model_name, random_state)
    y_arr = pd.to_numeric(y, errors="coerce")
    valid = y_arr.notna() & np.isfinite(y_arr.to_numpy(dtype=float))
    X_fit = X.loc[valid].reset_index(drop=True)
    y_fit = y_arr.loc[valid].to_numpy(dtype=float)

    pipeline = _make_pipeline(preprocessor, model)
    pipeline.fit(X_fit, y_fit)
    preds = predict_with_model(pipeline, X_fit, fallback_value=float(np.median(y_fit)))
    metadata = {
        "selected_model": model_name,
        "n_train_rows": int(len(X_fit)),
        "train_mae": float(mean_absolute_error_np(y_fit, preds)),
        "fallback_value": float(np.median(y_fit)),
    }
    return pipeline, metadata


def fit_per_block_models(
    X: pd.DataFrame,
    y: pd.Series,
    blocks: pd.Series,
    preprocessor: Any,
    selected_by_block: Dict[str, str],
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    fitted: Dict[str, Any] = {}
    block_medians: Dict[str, float] = {}
    y_arr = pd.to_numeric(y, errors="coerce")

    for block_value in blocks.dropna().unique():
        mask = (blocks == block_value).to_numpy()
        y_block = y_arr.iloc[mask]
        valid = (y_block.notna() & np.isfinite(y_block.to_numpy(dtype=float))).to_numpy()
        y_vals = y_block.iloc[valid].to_numpy(dtype=float)
        block_key = str(block_value)
        block_medians[block_key] = float(np.median(y_vals)) if len(y_vals) else float("nan")

        model_name = selected_by_block.get(block_key, "median_fallback")
        if model_name == "median_fallback" or len(y_vals) < MIN_BLOCK_ROWS:
            fitted[block_key] = None
            continue

        X_block = X.iloc[mask].iloc[valid].reset_index(drop=True)
        pipeline = _make_pipeline(preprocessor, _estimator_for_name(model_name, random_state))
        pipeline.fit(X_block, y_vals)
        fitted[block_key] = pipeline

    global_median = float(np.median(y_arr[np.isfinite(y_arr.to_numpy(dtype=float))]))
    out_medians = dict(block_medians)
    out_medians["__global__"] = global_median
    return fitted, out_medians


def predict_with_model(
    model: Any,
    X_test: pd.DataFrame,
    fallback_value: float,
) -> np.ndarray:
    """Generate predictions, filling failures with a fallback value."""
    try:
        preds = model.predict(X_test)
        preds = np.asarray(preds, dtype=float)
    except Exception:
        preds = np.full(len(X_test), fallback_value, dtype=float)

    invalid = ~np.isfinite(preds)
    if invalid.any():
        preds = preds.copy()
        preds[invalid] = fallback_value
    return preds


def predict_global_strategy(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    fallback_value: float,
) -> np.ndarray:
    return predict_with_model(pipeline, X_test, fallback_value)


def predict_per_block_strategy(
    fitted_by_block: Dict[str, Any],
    block_medians: Dict[str, float],
    X_test: pd.DataFrame,
    blocks_test: pd.Series,
    global_fallback: float,
) -> np.ndarray:
    preds = np.full(len(X_test), global_fallback, dtype=float)
    for idx in range(len(X_test)):
        block_key = str(blocks_test.iloc[idx]) if pd.notna(blocks_test.iloc[idx]) else "__unknown__"
        fallback = block_medians.get(block_key, global_fallback)
        model = fitted_by_block.get(block_key)
        if model is None:
            preds[idx] = fallback
            continue
        row = X_test.iloc[[idx]]
        preds[idx] = float(predict_with_model(model, row, fallback_value=fallback)[0])
    return preds


def cross_validate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    random_state: int = 42,
    metric_spec: Optional[Dict[str, Any]] = None,
    blocks: Optional[pd.Series] = None,
    block_column: Optional[str] = None,
) -> pd.DataFrame:
    spec = metric_spec or {"metric_name": "mae", "metric_mode": "global", "higher_is_better": False}
    selection = run_strategy_selection(X, y, blocks, preprocessor, spec, block_column, random_state)
    return selection["cv_results"]


def fit_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    cv_results: pd.DataFrame,
    random_state: int = 42,
    strategy_selection: Optional[Dict[str, Any]] = None,
    blocks: Optional[pd.Series] = None,
) -> Tuple[Any, str, Dict[str, Any]]:
    """Fit the selected global or per-block strategy."""
    if strategy_selection is None:
        global_rows = cv_results[cv_results.get("strategy", "global") == "global"] if "strategy" in cv_results.columns else cv_results
        ok = global_rows[global_rows["status"] == "ok"] if not global_rows.empty else cv_results
        model_name = str(ok.sort_values("score_mean").iloc[0]["model"]) if not ok.empty else "dummy_median"
        pipeline, meta = fit_global_model(X, y, preprocessor, model_name, random_state)
        return pipeline, model_name, meta

    if strategy_selection.get("modeling_strategy") == "per_block" and blocks is not None:
        fitted, medians = fit_per_block_models(
            X,
            y,
            blocks,
            preprocessor,
            strategy_selection.get("selected_model_by_block") or {},
            random_state,
        )
        meta = {
            "selected_model": "per_block",
            "selected_model_by_block": strategy_selection.get("selected_model_by_block"),
            "n_train_rows": int(len(X)),
            "fallback_value": medians.get("__global__", float(np.nan)),
            "block_medians": medians,
            "fitted_by_block": True,
        }
        return {"fitted_by_block": fitted, "block_medians": medians}, "per_block", meta

    model_name = strategy_selection.get("selected_global_model", "dummy_median")
    pipeline, meta = fit_global_model(X, y, preprocessor, model_name, random_state)
    return pipeline, model_name, meta
