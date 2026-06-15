"""Automated exploratory data analysis for Sentinelle pipeline artifacts."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.io_utils import artifacts_dir, data_dir, project_root

TIME_NAME_PATTERNS = re.compile(
    r"(date|time|period|month|year|week|quarter|day|timestamp|datetime)",
    re.IGNORECASE,
)
MAX_CATEGORICAL_FIGURES = 3
TOP_MISSING = 20
TOP_CORRELATIONS = 15
TOP_SHIFT = 15
FIGSIZE_WIDE = (9.0, 5.0)
FIGSIZE_BAR = (8.0, 4.5)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isfinite(obj):
            return float(obj)
        return None
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def _read_csv_safe(path: Path, warnings_list: List[str]) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception as exc:
            warnings_list.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
            return None


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _format_float(value: Any) -> Any:
    try:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (float, np.floating)):
            return round(float(value), 4)
        return value
    except Exception:
        return value


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(_format_float)
    return out


def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_count": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean()), 4),
                "nunique": int(s.nunique(dropna=True)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["missing_pct", "missing_count"], ascending=False)
    return out


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
    ]


def _looks_like_text(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    as_str = non_null.astype(str)
    avg_len = float(as_str.str.len().mean())
    if avg_len >= 40:
        return True
    alpha_ratio = as_str.str.contains(r"[A-Za-z]", regex=True).mean()
    space_ratio = as_str.str.contains(r"\s", regex=True).mean()
    return avg_len >= 25 and alpha_ratio > 0.5 and space_ratio > 0.2


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    sample = non_null.head(min(50, len(non_null)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.7


def _safe_sample(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def _skew(series: pd.Series) -> float:
    try:
        from scipy.stats import skew

        vals = pd.to_numeric(series, errors="coerce").dropna()
        if len(vals) < 3:
            return float("nan")
        return float(skew(vals.to_numpy(dtype=float)))
    except Exception:
        return float("nan")


def _resolve_role_path(schema_summary: Dict[str, Any], role: str) -> Optional[Path]:
    roles = schema_summary.get("inferred_roles") or {}
    profile = roles.get(role)
    if profile and profile.get("path"):
        return Path(profile["path"])
    for file_info in schema_summary.get("files") or []:
        if file_info.get("role") == role and file_info.get("path"):
            return Path(file_info["path"])
    return None


def _memory_mb(df: pd.DataFrame) -> float:
    try:
        return round(float(df.memory_usage(deep=True).sum()) / (1024**2), 4)
    except Exception:
        return float("nan")


def _dataset_overview_row(role: str, rel_path: str, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None:
        return {
            "role": role,
            "relative_path": rel_path,
            "rows": 0,
            "columns": 0,
            "memory_mb": np.nan,
            "duplicate_rows": 0,
            "missing_cells": 0,
            "missing_pct": np.nan,
        }
    missing_cells = int(df.isna().sum().sum())
    total_cells = max(len(df) * max(len(df.columns), 1), 1)
    return {
        "role": role,
        "relative_path": rel_path,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_mb": _memory_mb(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells, 4),
    }


def _column_type_summary(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"table": table_name, "rows": 0, "columns": 0}
    numeric = 0
    categorical = 0
    datetime_like = 0
    text_like = 0
    all_missing = 0
    high_missing = 0
    high_cardinality = 0
    n_rows = max(len(df), 1)

    for col in df.columns:
        s = df[col]
        miss_pct = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        if miss_pct >= 1.0:
            all_missing += 1
        if miss_pct > 0.5:
            high_missing += 1
        if nunique / n_rows > 0.95 and nunique > 20:
            high_cardinality += 1

        if pd.api.types.is_numeric_dtype(s):
            numeric += 1
        elif _looks_like_datetime(s):
            datetime_like += 1
        elif _looks_like_text(s):
            text_like += 1
        else:
            categorical += 1

    return {
        "table": table_name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_like_columns": datetime_like,
        "text_like_columns": text_like,
        "all_missing_columns": all_missing,
        "high_missing_columns": high_missing,
        "high_cardinality_columns": high_cardinality,
    }


def _safe_merge_train(
    train_target: pd.DataFrame,
    train_features: pd.DataFrame,
    join_keys: Optional[List[str]],
    warnings_list: List[str],
) -> Optional[pd.DataFrame]:
    if train_target is None or train_features is None:
        return None
    if not join_keys:
        warnings_list.append("train join keys unavailable; skipping target-feature joined EDA")
        return None
    missing = [k for k in join_keys if k not in train_target.columns or k not in train_features.columns]
    if missing:
        warnings_list.append(f"train join keys missing for merge: {missing}")
        return None

    n_target = len(train_target)
    try:
        merged = train_target.merge(train_features, on=join_keys, how="inner", suffixes=("", "_feat"))
    except Exception as exc:
        warnings_list.append(f"train target/features merge failed: {type(exc).__name__}: {exc}")
        return None

    n_merged = len(merged)
    if n_merged > 1.05 * n_target:
        warnings_list.append(
            f"merged rows ({n_merged}) exceed train_target rows ({n_target}); skipping joined EDA"
        )
        return None
    if n_merged < 0.75 * n_target:
        warnings_list.append(
            f"merged rows ({n_merged}) below 75% of train_target ({n_target}); skipping joined EDA"
        )
        return None
    return merged


def _resolve_target_column(
    schema_summary: Dict[str, Any],
    modeling_log: Optional[Dict[str, Any]],
    train_target: Optional[pd.DataFrame],
) -> Optional[str]:
    candidates: List[Optional[str]] = []
    if modeling_log:
        candidates.append(modeling_log.get("training_target_column"))
    candidates.append(schema_summary.get("target_column"))
    if train_target is not None:
        for key in ("target", "label", "y", "response", "outcome", "value", "rate", "score", "target_value"):
            if key in train_target.columns:
                candidates.append(key)
        numeric = [c for c in _numeric_columns(train_target) if c not in {"row_id", "id"}]
        if numeric:
            candidates.append(numeric[-1])
    for col in candidates:
        if col and train_target is not None and col in train_target.columns:
            return col
    return None


def _run_dataset_overview(
    loaded: Dict[str, Tuple[str, Optional[pd.DataFrame]]],
    out_dir: Path,
    tables: List[str],
    filename: str = "raw_file_overview.csv",
) -> pd.DataFrame:
    rows = [_dataset_overview_row(role, rel, df) for role, (rel, df) in loaded.items()]
    df = pd.DataFrame(rows)
    path = out_dir / filename
    _write_csv(_round_df(df), path)
    tables.append(str(path))
    return df


def _raw_file_overview_from_schema(schema_summary: Dict[str, Any]) -> pd.DataFrame:
    """Build raw CSV overview rows from schema file profiles."""
    rows: List[Dict[str, Any]] = []
    for profile in schema_summary.get("files") or []:
        rows.append(
            {
                "role": profile.get("role", "unknown"),
                "relative_path": profile.get("rel_path", profile.get("path", "unknown")),
                "rows": profile.get("n_rows", 0),
                "columns": profile.get("n_cols", 0),
                "memory_mb": np.nan,
                "duplicate_rows": np.nan,
                "missing_cells": np.nan,
                "missing_pct": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _count_modeling_columns(column_groups: Dict[str, Any]) -> int:
    if not column_groups:
        return 0
    counted: set[str] = set()
    for key in (
        "numeric_columns",
        "low_cardinality_categorical",
        "engineered_categorical_columns",
        "frequency_encoded_categorical",
        "simple_text_features",
        "text_svd_features",
        "datetime_features_created",
        "image_feature_columns",
    ):
        for col in column_groups.get(key) or []:
            counted.add(str(col))
    return len(counted)


def _build_modeling_frame_overview(
    modeling_log: Optional[Dict[str, Any]],
    merged: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize model-ready row counts, which may differ from raw CSV files."""
    rows: List[Dict[str, Any]] = []
    ml = modeling_log or {}
    groups = ml.get("column_groups") or {}
    n_feature_cols = _count_modeling_columns(groups)

    n_train = ml.get("n_train_rows")
    if n_train is not None:
        rows.append(
            {
                "role": "train_modeling_frame",
                "relative_path": "modeling training table",
                "rows": int(n_train),
                "columns": n_feature_cols if n_feature_cols else np.nan,
                "memory_mb": np.nan,
                "duplicate_rows": np.nan,
                "missing_cells": np.nan,
                "missing_pct": np.nan,
            }
        )

    n_pred = ml.get("n_predictions")
    if n_pred is not None:
        rows.append(
            {
                "role": "validation_modeling_frame",
                "relative_path": "modeling prediction table",
                "rows": int(n_pred),
                "columns": n_feature_cols if n_feature_cols else np.nan,
                "memory_mb": np.nan,
                "duplicate_rows": np.nan,
                "missing_cells": np.nan,
                "missing_pct": np.nan,
            }
        )

    if merged is not None and not merged.empty:
        joined_row = _dataset_overview_row(
            "train_joined_eda",
            "inner-joined train target+features (EDA)",
            merged,
        )
        if n_train is None or int(joined_row.get("rows", 0)) != int(n_train):
            rows.append(joined_row)

    return pd.DataFrame(rows)


def _run_column_type_summary(
    tables_dict: Dict[str, Optional[pd.DataFrame]],
    out_dir: Path,
    tables: List[str],
) -> pd.DataFrame:
    rows = [_column_type_summary(df, name) for name, df in tables_dict.items()]
    df = pd.DataFrame(rows)
    path = out_dir / "column_type_summary.csv"
    _write_csv(_round_df(df), path)
    tables.append(str(path))
    return df


def _run_missingness(
    df: Optional[pd.DataFrame],
    out_dir: Path,
    tables: List[str],
    figures: List[str],
    warnings_list: List[str],
) -> bool:
    if df is None or df.empty:
        return False
    miss = _missing_summary(df).head(TOP_MISSING)
    path = out_dir / "missingness_top.csv"
    _write_csv(miss, path)
    tables.append(str(path))

    if miss.empty or float(miss["missing_pct"].max()) <= 0:
        warnings_list.append("No missing values detected in main EDA table; missingness figure skipped")
        return True

    top = miss[miss["missing_pct"] > 0].head(TOP_MISSING)
    if top.empty:
        return True

    fig_path = out_dir / "missingness_top.png"
    try:
        plt.figure(figsize=FIGSIZE_BAR)
        plt.barh(top["column"][::-1], top["missing_pct"][::-1])
        plt.xlabel("Missing fraction")
        plt.ylabel("Column")
        plt.title("Top columns by missingness")
        _save_fig(fig_path)
        figures.append(str(fig_path))
    except Exception as exc:
        warnings_list.append(f"missingness figure failed: {type(exc).__name__}: {exc}")
    return True


def _run_target_distribution(
    train_target: Optional[pd.DataFrame],
    target_col: Optional[str],
    out_dir: Path,
    tables: List[str],
    figures: List[str],
    warnings_list: List[str],
) -> bool:
    if train_target is None or not target_col or target_col not in train_target.columns:
        return False
    series = pd.to_numeric(train_target[target_col], errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return False

    summary = {
        "target_column": target_col,
        "count": int(valid.count()),
        "mean": _format_float(valid.mean()),
        "std": _format_float(valid.std()),
        "min": _format_float(valid.min()),
        "q25": _format_float(valid.quantile(0.25)),
        "median": _format_float(valid.median()),
        "q75": _format_float(valid.quantile(0.75)),
        "max": _format_float(valid.max()),
        "skew": _format_float(_skew(valid)),
        "zero_count": int((valid == 0).sum()),
        "negative_count": int((valid < 0).sum()),
        "missing_count": int(series.isna().sum()),
    }
    path = out_dir / "target_summary.csv"
    _write_csv(pd.DataFrame([summary]), path)
    tables.append(str(path))

    try:
        fig_path = out_dir / "target_distribution.png"
        plt.figure(figsize=FIGSIZE_WIDE)
        plt.hist(valid.to_numpy(dtype=float), bins=min(40, max(10, int(np.sqrt(len(valid))))))
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.title(f"Target distribution: {target_col}")
        _save_fig(fig_path)
        figures.append(str(fig_path))

        if (valid >= 0).all() and (valid.max() > 0):
            log_vals = np.log1p(valid.to_numpy(dtype=float))
            log_path = out_dir / "target_distribution_log1p.png"
            plt.figure(figsize=FIGSIZE_WIDE)
            plt.hist(log_vals, bins=min(40, max(10, int(np.sqrt(len(valid))))))
            plt.xlabel(f"log1p({target_col})")
            plt.ylabel("Count")
            plt.title(f"log1p target distribution: {target_col}")
            _save_fig(log_path)
            figures.append(str(log_path))
    except Exception as exc:
        warnings_list.append(f"target distribution figure failed: {type(exc).__name__}: {exc}")
    return True


def _run_target_by_block(
    df: pd.DataFrame,
    target_col: str,
    block_col: str,
    out_dir: Path,
    tables: List[str],
    figures: List[str],
    warnings_list: List[str],
) -> bool:
    if block_col not in df.columns or target_col not in df.columns:
        return False
    grouped = df.groupby(block_col, dropna=False)
    rows: List[Dict[str, Any]] = []
    for block_value, g in grouped:
        t = pd.to_numeric(g[target_col], errors="coerce")
        rows.append(
            {
                "block": str(block_value),
                "rows": int(len(g)),
                "mean_target": _format_float(t.mean()),
                "std_target": _format_float(t.std()),
                "median_target": _format_float(t.median()),
                "min_target": _format_float(t.min()),
                "max_target": _format_float(t.max()),
                "missing_target_count": int(t.isna().sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("rows", ascending=False)
    if out.empty:
        return False

    if len(out) > 20:
        plot_df = out.head(20)
    else:
        plot_df = out

    path = out_dir / "target_by_block.csv"
    _write_csv(out, path)
    tables.append(str(path))

    try:
        fig_path = out_dir / "target_by_block_mean.png"
        plt.figure(figsize=FIGSIZE_WIDE)
        plt.bar(plot_df["block"].astype(str), plot_df["mean_target"].astype(float))
        plt.xlabel(block_col)
        plt.ylabel(f"Mean {target_col}")
        plt.title(f"Mean target by {block_col}")
        plt.xticks(rotation=45, ha="right")
        _save_fig(fig_path)
        figures.append(str(fig_path))
    except Exception as exc:
        warnings_list.append(f"target by block figure failed: {type(exc).__name__}: {exc}")
    return True


def _run_numeric_feature_summary(
    merged: Optional[pd.DataFrame],
    train_features: Optional[pd.DataFrame],
    target_col: Optional[str],
    out_dir: Path,
    tables: List[str],
    figures: List[str],
    warnings_list: List[str],
    max_rows: int,
    random_state: int,
) -> bool:
    base = merged if merged is not None else train_features
    if base is None or base.empty:
        return False

    sample = _safe_sample(base, max_rows, random_state)
    num_cols = [c for c in _numeric_columns(sample) if c != target_col]
    if not num_cols:
        return False

    rows: List[Dict[str, Any]] = []
    target_series = None
    if target_col and target_col in sample.columns:
        target_series = pd.to_numeric(sample[target_col], errors="coerce")

    for col in num_cols:
        s = pd.to_numeric(sample[col], errors="coerce")
        row: Dict[str, Any] = {
            "column": col,
            "count": int(s.notna().sum()),
            "missing_pct": round(float(s.isna().mean()), 4),
            "mean": _format_float(s.mean()),
            "std": _format_float(s.std()),
            "min": _format_float(s.min()),
            "q25": _format_float(s.quantile(0.25)),
            "median": _format_float(s.median()),
            "q75": _format_float(s.quantile(0.75)),
            "max": _format_float(s.max()),
            "skew": _format_float(_skew(s)),
            "abs_corr_with_target": np.nan,
        }
        if target_series is not None and s.notna().sum() > 2 and target_series.notna().sum() > 2:
            aligned = pd.concat([s, target_series], axis=1).dropna()
            if len(aligned) > 2:
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if corr is not None and np.isfinite(corr):
                    row["abs_corr_with_target"] = _format_float(abs(float(corr)))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("abs_corr_with_target", ascending=False, na_position="last")
    path = out_dir / "numeric_feature_summary.csv"
    _write_csv(summary, path)
    tables.append(str(path))

    corr_df = summary.dropna(subset=["abs_corr_with_target"]).head(TOP_CORRELATIONS)
    if len(corr_df) >= 2:
        corr_path = out_dir / "top_numeric_correlations.csv"
        _write_csv(corr_df[["column", "abs_corr_with_target"]], corr_path)
        tables.append(str(corr_path))
        try:
            fig_path = out_dir / "top_numeric_correlations.png"
            plt.figure(figsize=FIGSIZE_BAR)
            plt.barh(corr_df["column"][::-1], corr_df["abs_corr_with_target"][::-1])
            plt.xlabel("|Pearson correlation with target|")
            plt.title("Top numeric feature correlations with target")
            _save_fig(fig_path)
            figures.append(str(fig_path))
        except Exception as exc:
            warnings_list.append(f"numeric correlation figure failed: {type(exc).__name__}: {exc}")
    return True


def _run_categorical_summary(
    train_features: Optional[pd.DataFrame],
    out_dir: Path,
    tables: List[str],
    figures: List[str],
    warnings_list: List[str],
    max_rows: int,
    random_state: int,
) -> bool:
    if train_features is None or train_features.empty:
        return False
    sample = _safe_sample(train_features, max_rows, random_state)
    cat_cols = _categorical_columns(sample)
    if not cat_cols:
        return False

    rows: List[Dict[str, Any]] = []
    fig_count = 0
    n_rows = max(len(sample), 1)

    for col in cat_cols:
        s = sample[col].astype("string")
        miss_pct = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        counts = s.value_counts(dropna=True)
        top_value = counts.index[0] if len(counts) else ""
        top_freq = int(counts.iloc[0]) if len(counts) else 0
        rows.append(
            {
                "column": col,
                "nunique": nunique,
                "missing_pct": round(miss_pct, 4),
                "top_value": str(top_value),
                "top_value_freq": top_freq,
                "top_value_pct": round(top_freq / n_rows, 4),
                "high_cardinality": bool(nunique > 50 or nunique / n_rows > 0.5),
            }
        )

        if fig_count < MAX_CATEGORICAL_FIGURES and 2 <= nunique <= 50:
            try:
                top = counts.head(15)
                fig_path = out_dir / f"top_categorical_counts_{col}.png"
                plt.figure(figsize=FIGSIZE_WIDE)
                plt.bar(top.index.astype(str), top.to_numpy())
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.title(f"Top categories: {col}")
                plt.xticks(rotation=45, ha="right")
                _save_fig(fig_path)
                figures.append(str(fig_path))
                fig_count += 1
            except Exception as exc:
                warnings_list.append(f"categorical figure for {col} failed: {type(exc).__name__}: {exc}")

    path = out_dir / "categorical_feature_summary.csv"
    _write_csv(pd.DataFrame(rows), path)
    tables.append(str(path))
    return True


def _run_train_validation_shift(
    train_features: Optional[pd.DataFrame],
    val_features: Optional[pd.DataFrame],
    out_dir: Path,
    tables: List[str],
    figures: List[str],
    warnings_list: List[str],
    max_rows: int,
    random_state: int,
) -> bool:
    if train_features is None or val_features is None:
        return False

    train_s = _safe_sample(train_features, max_rows, random_state)
    val_s = _safe_sample(val_features, max_rows, random_state)
    shared_num = [c for c in _numeric_columns(train_s) if c in val_s.columns]
    ran = False

    if shared_num:
        num_rows: List[Dict[str, Any]] = []
        for col in shared_num:
            tr = pd.to_numeric(train_s[col], errors="coerce")
            va = pd.to_numeric(val_s[col], errors="coerce")
            tr_mean = float(tr.mean()) if tr.notna().any() else float("nan")
            va_mean = float(va.mean()) if va.notna().any() else float("nan")
            pooled_std = float(np.nanstd(np.concatenate([tr.dropna().to_numpy(), va.dropna().to_numpy()])))
            if not np.isfinite(pooled_std) or pooled_std == 0:
                smd = 0.0 if tr_mean == va_mean else float("nan")
            else:
                smd = (va_mean - tr_mean) / pooled_std
            num_rows.append(
                {
                    "column": col,
                    "train_mean": _format_float(tr_mean),
                    "validation_mean": _format_float(va_mean),
                    "standardized_mean_diff": _format_float(smd),
                    "train_missing_pct": round(float(tr.isna().mean()), 4),
                    "validation_missing_pct": round(float(va.isna().mean()), 4),
                    "missing_pct_diff": round(float(va.isna().mean() - tr.isna().mean()), 4),
                }
            )
        num_df = pd.DataFrame(num_rows).sort_values(
            "standardized_mean_diff", key=lambda s: s.abs(), ascending=False
        )
        num_path = out_dir / "train_validation_shift_numeric.csv"
        _write_csv(num_df, num_path)
        tables.append(str(num_path))
        ran = True

        top = num_df.reindex(num_df["standardized_mean_diff"].abs().sort_values(ascending=False).index).head(
            TOP_SHIFT
        )
        if not top.empty:
            try:
                fig_path = out_dir / "train_validation_shift_top.png"
                plt.figure(figsize=FIGSIZE_BAR)
                plt.barh(top["column"][::-1], top["standardized_mean_diff"][::-1])
                plt.xlabel("Standardized mean difference (validation - train)")
                plt.title("Top numeric covariate shift")
                _save_fig(fig_path)
                figures.append(str(fig_path))
            except Exception as exc:
                warnings_list.append(f"shift figure failed: {type(exc).__name__}: {exc}")

    shared_cat = [c for c in _categorical_columns(train_s) if c in val_s.columns]
    if shared_cat:
        cat_rows: List[Dict[str, Any]] = []
        for col in shared_cat:
            tr = train_s[col].astype("string")
            va = val_s[col].astype("string")
            train_cats = set(tr.dropna().unique())
            val_cats = set(va.dropna().unique())
            unseen = val_cats - train_cats
            unseen_count = sum(int((va == cat).sum()) for cat in unseen)
            val_non_missing = int(va.notna().sum())
            cat_rows.append(
                {
                    "column": col,
                    "train_nunique": int(tr.nunique(dropna=True)),
                    "validation_nunique": int(va.nunique(dropna=True)),
                    "unseen_validation_categories_count": int(len(unseen)),
                    "unseen_validation_categories_pct": round(
                        unseen_count / max(val_non_missing, 1), 4
                    ),
                    "train_missing_pct": round(float(tr.isna().mean()), 4),
                    "validation_missing_pct": round(float(va.isna().mean()), 4),
                }
            )
        cat_path = out_dir / "train_validation_shift_categorical.csv"
        _write_csv(pd.DataFrame(cat_rows), cat_path)
        tables.append(str(cat_path))
        ran = True

    if not shared_num and not shared_cat:
        warnings_list.append("No shared columns for train/validation shift analysis")
    return ran


def _time_candidate_columns(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        if TIME_NAME_PATTERNS.search(str(col)):
            candidates.append(col)
        elif _looks_like_datetime(df[col]):
            candidates.append(col)
    return list(dict.fromkeys(candidates))


def _run_time_patterns(
    merged: Optional[pd.DataFrame],
    train_features: Optional[pd.DataFrame],
    target_col: Optional[str],
    out_dir: Path,
    tables: List[str],
    figures: List[str],
    warnings_list: List[str],
) -> bool:
    base = merged if merged is not None else train_features
    if base is None or base.empty:
        return False

    candidates = _time_candidate_columns(base)
    if not candidates:
        return False

    ran = False
    time_rows: List[Dict[str, Any]] = []
    for col in candidates[:3]:
        try:
            parsed = pd.to_datetime(base[col], errors="coerce")
            if parsed.notna().mean() < 0.3:
                continue
            period = parsed.dt.to_period("M").astype(str)
            counts = period.value_counts().sort_index()
            if len(counts) > 60:
                continue

            for idx, cnt in counts.items():
                time_rows.append(
                    {"source_column": col, "period": str(idx), "rows": int(cnt)}
                )
            ran = True

            if target_col and target_col in base.columns:
                tmp = base[[target_col]].copy()
                tmp["period"] = period
                means = pd.to_numeric(tmp[target_col], errors="coerce").groupby(tmp["period"]).mean()
                if len(means) > 1:
                    fig_path = out_dir / "target_over_time.png"
                    plt.figure(figsize=FIGSIZE_WIDE)
                    plt.plot(means.index.astype(str), means.to_numpy(dtype=float))
                    plt.xlabel(col)
                    plt.ylabel(f"Mean {target_col}")
                    plt.title(f"Mean target over {col}")
                    plt.xticks(rotation=45, ha="right")
                    _save_fig(fig_path)
                    figures.append(str(fig_path))
                    break
        except Exception as exc:
            warnings_list.append(f"time pattern for {col} failed: {type(exc).__name__}: {exc}")

    if time_rows:
        path = out_dir / "time_summary.csv"
        _write_csv(pd.DataFrame(time_rows), path)
        tables.append(str(path))
    return ran


def _extract_text_image_availability(modeling_log: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract text/image feature availability from modeling_log column_groups.

    EDA should report the same text/image feature availability that modeling
    used. It should not independently report zero images/text if feature
    engineering already found them.
    """
    groups: Dict[str, Any] = {}
    if modeling_log:
        groups = modeling_log.get("column_groups") or {}

    simple_text = groups.get("simple_text_features") or []
    text_svd = groups.get("text_svd_features") or []

    text_columns = groups.get("text_columns") or groups.get("text_cols") or []
    if not text_columns and simple_text:
        text_columns = sorted(
            {
                str(name).split("__")[0]
                for name in simple_text
                if "__" in str(name)
            }
        )

    return {
        "text_columns_detected": text_columns,
        "simple_text_features": simple_text,
        "text_svd_features": text_svd,
        "text_feature_warnings": groups.get("text_feature_warnings")
        or groups.get("feature_warnings")
        or [],
        "image_features_used": bool(groups.get("image_features_used", False)),
        "n_images_found": int(groups.get("n_images_found") or 0),
        "n_rows_matched": int(groups.get("n_rows_matched") or 0),
        "image_match_rate": float(groups.get("image_match_rate") or 0.0),
        "image_feature_columns": groups.get("image_feature_columns") or [],
        "image_feature_warnings": groups.get("image_feature_warnings") or [],
        "availability_source": "modeling_log.column_groups" if groups else "unavailable_no_column_groups",
    }


def _modality_summary(modeling_log: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    groups = (modeling_log or {}).get("column_groups") or {}
    return {
        "text_columns_detected": groups.get("text_columns") or [],
        "simple_text_features": groups.get("simple_text_features") or [],
        "text_svd_features": groups.get("text_svd_features") or [],
        "image_features_used": bool(groups.get("image_features_used", False)),
        "n_images_found": int(groups.get("n_images_found", 0)),
        "n_rows_matched": int(groups.get("n_rows_matched", 0)),
        "image_match_rate": _format_float(groups.get("image_match_rate", 0.0)),
    }


def run_eda(
    schema_summary: Dict[str, Any],
    modeling_log: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
    max_rows: int = 50000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run automated EDA and write artifacts under artifacts/eda/."""
    root = root or project_root()
    out_dir = artifacts_dir(root) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: List[str] = []
    figures: List[str] = []
    tables: List[str] = []
    flags: Dict[str, Any] = {
        "target_eda_available": False,
        "block_target_eda_available": False,
        "joined_eda_available": False,
        "numeric_correlation_available": False,
        "shift_analysis_available": False,
        "n_files_analyzed": 0,
    }

    summary: Dict[str, Any] = {
        "status": "ok",
        "warnings": warnings_list,
        "figures": figures,
        "tables": tables,
        "output_dir": str(out_dir),
        "flags": flags,
        "modality_summary": {},
    }

    try:
        role_files = {
            "sample_submission": _resolve_role_path(schema_summary, "sample_submission"),
            "train_target": _resolve_role_path(schema_summary, "train_target"),
            "train_features": _resolve_role_path(schema_summary, "train_features"),
            "validation_features": _resolve_role_path(schema_summary, "validation_features"),
        }

        loaded: Dict[str, Tuple[str, Optional[pd.DataFrame]]] = {}
        for role, path in role_files.items():
            rel = _safe_rel(path, root) if path else "missing"
            df = _read_csv_safe(path, warnings_list) if path and path.exists() else None
            if df is not None:
                flags["n_files_analyzed"] += 1
            loaded[role] = (rel, df)

        sample_df = loaded["sample_submission"][1]
        train_target = loaded["train_target"][1]
        train_features = loaded["train_features"][1]
        val_features = loaded["validation_features"][1]

        try:
            raw_overview = _raw_file_overview_from_schema(schema_summary)
            if raw_overview.empty:
                raw_overview = _run_dataset_overview(loaded, out_dir, tables, filename="raw_file_overview.csv")
            else:
                raw_path = out_dir / "raw_file_overview.csv"
                _write_csv(_round_df(raw_overview), raw_path)
                tables.append(str(raw_path))
            summary["raw_file_overview"] = raw_overview.to_dict(orient="records")
        except Exception as exc:
            warnings_list.append(f"raw file overview failed: {type(exc).__name__}: {exc}")
            raw_overview = pd.DataFrame()

        join_keys = (schema_summary.get("join_keys") or {}).get("train_keys")
        merged = _safe_merge_train(train_target, train_features, join_keys, warnings_list)
        if merged is not None:
            flags["joined_eda_available"] = True

        try:
            modeling_overview = _build_modeling_frame_overview(modeling_log, merged)
            modeling_path = out_dir / "modeling_frame_overview.csv"
            _write_csv(_round_df(modeling_overview), modeling_path)
            tables.append(str(modeling_path))
            summary["modeling_frame_overview"] = modeling_overview.to_dict(orient="records")
        except Exception as exc:
            warnings_list.append(f"modeling frame overview failed: {type(exc).__name__}: {exc}")

        # Backward-compatible alias for older report loaders.
        try:
            if not raw_overview.empty:
                legacy_path = out_dir / "dataset_overview.csv"
                _write_csv(_round_df(raw_overview), legacy_path)
                if str(legacy_path) not in tables:
                    tables.append(str(legacy_path))
        except Exception:
            pass

        try:
            _run_column_type_summary(
                {
                    "train_features": train_features,
                    "train_target": train_target,
                    "validation_features": val_features,
                },
                out_dir,
                tables,
            )
        except Exception as exc:
            warnings_list.append(f"column type summary failed: {type(exc).__name__}: {exc}")

        main_table = merged if merged is not None else train_features
        try:
            _run_missingness(main_table, out_dir, tables, figures, warnings_list)
        except Exception as exc:
            warnings_list.append(f"missingness EDA failed: {type(exc).__name__}: {exc}")

        target_col = _resolve_target_column(schema_summary, modeling_log, train_target)
        try:
            flags["target_eda_available"] = _run_target_distribution(
                train_target, target_col, out_dir, tables, figures, warnings_list
            )
        except Exception as exc:
            warnings_list.append(f"target distribution EDA failed: {type(exc).__name__}: {exc}")

        metric_spec = (modeling_log or {}).get("metric_spec") or schema_summary.get("metric_spec") or {}
        block_col = metric_spec.get("block_column") or (modeling_log or {}).get("block_column")
        block_df = merged if merged is not None else train_target
        if (
            metric_spec.get("metric_mode") == "block_averaged"
            and block_col
            and target_col
            and block_df is not None
        ):
            try:
                flags["block_target_eda_available"] = _run_target_by_block(
                    block_df, target_col, block_col, out_dir, tables, figures, warnings_list
                )
            except Exception as exc:
                warnings_list.append(f"target by block EDA failed: {type(exc).__name__}: {exc}")

        try:
            flags["numeric_correlation_available"] = _run_numeric_feature_summary(
                merged,
                train_features,
                target_col,
                out_dir,
                tables,
                figures,
                warnings_list,
                max_rows,
                random_state,
            )
        except Exception as exc:
            warnings_list.append(f"numeric feature EDA failed: {type(exc).__name__}: {exc}")

        try:
            _run_categorical_summary(
                train_features, out_dir, tables, figures, warnings_list, max_rows, random_state
            )
        except Exception as exc:
            warnings_list.append(f"categorical feature EDA failed: {type(exc).__name__}: {exc}")

        try:
            flags["shift_analysis_available"] = _run_train_validation_shift(
                train_features,
                val_features,
                out_dir,
                tables,
                figures,
                warnings_list,
                max_rows,
                random_state,
            )
        except Exception as exc:
            warnings_list.append(f"train/validation shift EDA failed: {type(exc).__name__}: {exc}")

        try:
            _run_time_patterns(merged, train_features, target_col, out_dir, tables, figures, warnings_list)
        except Exception as exc:
            warnings_list.append(f"time pattern EDA failed: {type(exc).__name__}: {exc}")

        summary["modality_summary"] = _modality_summary(modeling_log)
        summary["target_column"] = target_col
        summary["block_column"] = block_col

    except Exception as exc:
        warnings_list.append(f"EDA failed: {type(exc).__name__}: {exc}")
        summary["status"] = "failed"

    if warnings_list and summary["status"] == "ok":
        summary["status"] = "partial"

    summary["text_image_availability"] = _extract_text_image_availability(modeling_log)

    summary_path = out_dir / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    tables.append(str(summary_path))
    return summary
