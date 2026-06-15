"""Feature detection and CPU-friendly preprocessing for tabular data."""

from __future__ import annotations

import warnings
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.image_features import build_image_features
from src.io_utils import artifacts_dir, data_dir, project_root
from src.text_features import add_text_features

LOW_CARDINALITY_MAX = 50
MEDIUM_CARDINALITY_MAX = 500
HIGH_CARDINALITY_UNIQUE_RATIO = 0.95


def normalize_column_name(name: str) -> str:
    """Normalize a column name for stable comparisons."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _is_id_like(col: str, nunique: int, n_rows: int) -> bool:
    col_norm = normalize_column_name(col)
    unique_ratio = nunique / max(n_rows, 1)
    if col_norm in {"row_id", "id", "uuid", "index"} or col_norm.endswith("_id"):
        return True
    return unique_ratio > HIGH_CARDINALITY_UNIQUE_RATIO and nunique > 20


def _looks_like_text(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    as_str = non_null.astype(str)
    avg_len = float(as_str.str.len().mean())
    max_len = int(as_str.str.len().max())
    if avg_len >= 40 or max_len >= 120:
        return True
    alpha_ratio = as_str.str.contains(r"[A-Za-z]", regex=True).mean()
    space_ratio = as_str.str.contains(r"\s", regex=True).mean()
    return avg_len >= 25 and alpha_ratio > 0.5 and space_ratio > 0.2


def detect_feature_columns(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    row_id_col: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Classify modeling columns into numeric, categorical, text, datetime, and id-like groups."""
    exclude = {c for c in (target_col, row_id_col) if c}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    text_cols: List[str] = []
    datetime_cols: List[str] = []
    id_like_cols: List[str] = []

    n = max(len(df), 1)

    for col in df.columns:
        if col in exclude:
            continue

        series = df[col]
        non_null = series.dropna()
        nunique = int(non_null.nunique()) if len(non_null) else 0

        if _is_id_like(col, nunique, n):
            id_like_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        as_str = non_null.astype(str)
        if len(as_str) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(as_str.head(min(50, len(as_str))), errors="coerce")
            if parsed.notna().mean() > 0.7:
                datetime_cols.append(col)
                continue

            if _looks_like_text(series):
                text_cols.append(col)
                continue

        categorical_cols.append(col)

    return {
        "numeric_columns": sorted(set(numeric_cols)),
        "categorical_columns": sorted(set(categorical_cols)),
        "text_columns": sorted(set(text_cols)),
        "datetime_columns": sorted(set(datetime_cols)),
        "id_like_columns": sorted(set(id_like_cols)),
    }


def classify_categorical_columns(
    df: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, List[str]]:
    """Split categorical columns by cardinality for encoding strategy selection."""
    n_rows = max(len(df), 1)
    low: List[str] = []
    medium: List[str] = []
    high: List[str] = []
    dropped: List[str] = []

    for col in categorical_cols:
        if col not in df.columns:
            continue
        nunique = int(df[col].nunique(dropna=True))
        unique_ratio = nunique / n_rows

        if _is_id_like(col, nunique, n_rows):
            dropped.append(col)
        elif nunique <= LOW_CARDINALITY_MAX:
            low.append(col)
        elif nunique <= MEDIUM_CARDINALITY_MAX:
            medium.append(col)
        elif unique_ratio > 0.99:
            dropped.append(col)
        else:
            high.append(col)

    return {
        "low_cardinality_categorical": sorted(low),
        "medium_cardinality_categorical": sorted(medium),
        "high_cardinality_categorical": sorted(high),
        "dropped_high_cardinality_categorical": sorted(dropped),
    }


def apply_frequency_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Frequency-encode categorical columns using training data statistics only."""
    train_out = train_df.copy()
    test_out = test_df.copy()
    created: List[str] = []

    for col in columns:
        if col not in train_out.columns:
            continue
        counts = train_out[col].astype("string").fillna("__MISSING__").value_counts(dropna=False)
        freq_map = (counts / max(len(train_out), 1)).to_dict()
        new_col = f"{col}__freq"
        train_out[new_col] = (
            train_out[col].astype("string").fillna("__MISSING__").map(freq_map).fillna(0.0).astype(float)
        )
        if col in test_out.columns:
            test_out[new_col] = (
                test_out[col].astype("string").fillna("__MISSING__").map(freq_map).fillna(0.0).astype(float)
            )
        else:
            test_out[new_col] = 0.0
        created.append(new_col)
        train_out.drop(columns=[col], inplace=True, errors="ignore")
        test_out.drop(columns=[col], inplace=True, errors="ignore")

    return train_out, test_out, created


def add_datetime_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    datetime_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Replace datetime-like columns with simple calendar features."""
    train_out = train_df.copy()
    test_out = test_df.copy()
    created: List[str] = []

    for col in datetime_cols:
        if col not in train_out.columns and col not in test_out.columns:
            continue

        for frame in (train_out, test_out):
            if col not in frame.columns:
                continue
            parsed = pd.to_datetime(frame[col], errors="coerce")
            frame[f"{col}__year"] = parsed.dt.year.astype("float")
            frame[f"{col}__month"] = parsed.dt.month.astype("float")
            frame[f"{col}__day"] = parsed.dt.day.astype("float")
            frame[f"{col}__dayofweek"] = parsed.dt.dayofweek.astype("float")
            frame[f"{col}__is_missing_datetime"] = parsed.isna().astype("float")
            frame.drop(columns=[col], inplace=True)
        created.extend(
            [
                f"{col}__year",
                f"{col}__month",
                f"{col}__day",
                f"{col}__dayofweek",
                f"{col}__is_missing_datetime",
            ]
        )

    return train_out, test_out, sorted(set(created))


def prepare_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    row_id_col: Optional[str],
    *,
    data_path: Optional[Path] = None,
    join_key_columns: Optional[List[str]] = None,
    artifacts_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, List[str]]]:
    """Engineer features and return modeling matrices excluding ids and target."""
    feature_warnings: List[str] = []
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    column_groups: Dict[str, Any] = detect_feature_columns(combined, target_col=target_col, row_id_col=row_id_col)
    column_groups["feature_warnings"] = feature_warnings

    train_raw = train_df.copy()
    test_raw = test_df.copy()
    train_eng = train_df.copy()
    test_eng = test_df.copy()

    train_eng, test_eng, text_meta = add_text_features(train_eng, test_eng, column_groups["text_columns"])
    column_groups["simple_text_features"] = text_meta.get("simple_text_features", [])
    column_groups["text_svd_features"] = text_meta.get("text_svd_features", [])
    column_groups["text_feature_warnings"] = text_meta.get("feature_warnings", [])
    feature_warnings.extend(text_meta.get("feature_warnings", []))

    train_eng, test_eng, datetime_created = add_datetime_features(
        train_eng,
        test_eng,
        column_groups["datetime_columns"],
    )
    column_groups["datetime_features_created"] = datetime_created

    cat_split = classify_categorical_columns(train_eng, column_groups["categorical_columns"])
    column_groups.update(cat_split)

    freq_cols = (
        cat_split["medium_cardinality_categorical"] + cat_split["high_cardinality_categorical"]
    )
    train_eng, test_eng, freq_created = apply_frequency_encoding(train_eng, test_eng, freq_cols)
    column_groups["frequency_encoded_categorical"] = freq_created

    dropped_cols = cat_split["dropped_high_cardinality_categorical"] + column_groups.get("id_like_columns", [])
    train_eng.drop(columns=[c for c in dropped_cols if c in train_eng.columns], inplace=True, errors="ignore")
    test_eng.drop(columns=[c for c in dropped_cols if c in test_eng.columns], inplace=True, errors="ignore")

    root = project_root()
    dpath = data_path or data_dir(root)
    apath = artifacts_path or artifacts_dir(root)
    image_cache = apath / "image_features.csv"
    image_keys = list(dict.fromkeys((join_key_columns or []) + ([row_id_col] if row_id_col else [])))

    raw_drop = {target_col}
    if row_id_col:
        raw_drop.add(row_id_col)
    train_raw_for_images = train_raw.drop(columns=[c for c in raw_drop if c in train_raw.columns], errors="ignore")
    test_raw_for_images = test_raw.drop(columns=[c for c in raw_drop if c in test_raw.columns], errors="ignore")

    train_img, test_img, image_meta = build_image_features(
        train_raw_for_images,
        test_raw_for_images,
        dpath,
        image_keys,
        cache_path=image_cache,
    )
    if image_meta.get("image_feature_columns"):
        for col in image_meta["image_feature_columns"]:
            train_eng[col] = train_img[col].to_numpy()
            test_eng[col] = test_img[col].to_numpy()

    image_notes = image_meta.get("image_feature_notes", [])
    image_warnings = image_meta.get("image_feature_warnings", [])
    feature_warnings.extend(image_warnings)
    column_groups["image_features_used"] = bool(image_meta.get("image_features_used", False))
    column_groups["n_images_found"] = int(image_meta.get("n_images_found", 0))
    column_groups["n_rows_matched"] = int(image_meta.get("n_rows_matched", 0))
    column_groups["image_match_rate"] = float(image_meta.get("image_match_rate", 0.0))
    column_groups["image_feature_columns"] = list(image_meta.get("image_feature_columns", []))
    column_groups["image_feature_warnings"] = list(image_warnings)
    column_groups["image_feature_notes"] = list(image_notes)

    y_train = pd.to_numeric(train_eng[target_col], errors="coerce")
    drop_cols = {target_col}
    if row_id_col:
        drop_cols.add(row_id_col)

    X_train = train_eng.drop(columns=[c for c in drop_cols if c in train_eng.columns], errors="ignore")
    X_test = test_eng.drop(columns=[c for c in drop_cols if c in test_eng.columns], errors="ignore")

    engineered_groups = detect_feature_columns(X_train, target_col=None, row_id_col=None)
    column_groups["engineered_numeric_columns"] = engineered_groups["numeric_columns"]
    column_groups["engineered_categorical_columns"] = cat_split["low_cardinality_categorical"]

    shared_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[shared_cols]
    X_test = X_test[shared_cols]
    column_groups["feature_warnings"] = feature_warnings

    return X_train, y_train, X_test, column_groups


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_sklearn_preprocessor(
    X: pd.DataFrame,
    column_groups: Optional[Dict[str, Any]] = None,
) -> ColumnTransformer:
    """Build a column-wise preprocessing pipeline for numeric and low-card categorical features."""
    low_card = []
    if column_groups:
        low_card = column_groups.get("low_cardinality_categorical") or column_groups.get(
            "engineered_categorical_columns", []
        )

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in low_card if c in X.columns and c not in numeric_cols]
    if not categorical_cols:
        categorical_cols = [
            c
            for c in X.columns
            if c not in numeric_cols and c in (column_groups or {}).get("engineered_categorical_columns", X.columns)
        ]
        categorical_cols = [c for c in categorical_cols if c in X.columns]

    transformers: List[Tuple[str, Any, List[str]]] = []

    if numeric_cols:
        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_one_hot_encoder()),
            ],
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        return ColumnTransformer(transformers=[], remainder="drop")

    return ColumnTransformer(transformers=transformers, remainder="drop")
