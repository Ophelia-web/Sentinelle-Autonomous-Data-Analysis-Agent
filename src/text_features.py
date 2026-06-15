"""Text column feature extraction with optional TF-IDF + SVD."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def _sentence_count(text: str) -> float:
    if not text:
        return 0.0
    parts = re.split(r"[.!?]+", text)
    return float(max(len([p for p in parts if p.strip()]), 1 if text.strip() else 0))


def _digit_count(text: str) -> float:
    return float(sum(ch.isdigit() for ch in text))


def _punctuation_count(text: str) -> float:
    return float(sum(not ch.isalnum() and not ch.isspace() for ch in text))


def _uppercase_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return float(sum(ch.isupper() for ch in alpha) / len(alpha))


def _add_simple_text_stats(frame: pd.DataFrame, col: str) -> List[str]:
    """Add cheap numeric text statistics for one column."""
    created: List[str] = []
    as_str = frame[col].astype("string")
    missing = as_str.isna() | (as_str.str.strip() == "")
    filled = as_str.fillna("")

    specs = {
        f"{col}__char_len": filled.str.len().astype(float),
        f"{col}__word_count": filled.str.split().str.len().astype(float),
        f"{col}__sentence_count": filled.map(_sentence_count).astype(float),
        f"{col}__digit_count": filled.map(_digit_count).astype(float),
        f"{col}__punctuation_count": filled.map(_punctuation_count).astype(float),
        f"{col}__uppercase_ratio": filled.map(_uppercase_ratio).astype(float),
        f"{col}__is_missing": missing.astype(float),
    }
    for name, values in specs.items():
        frame[name] = values
        created.append(name)
    frame.drop(columns=[col], inplace=True)
    return created


def _choose_svd_components(n_train: int) -> int:
    if n_train < 100:
        return 5
    if n_train < 1000:
        return 10
    return 25


def _choose_max_features(n_train: int) -> int:
    return int(min(2000, max(100, n_train * 5)))


def _choose_min_df(n_train: int) -> int:
    return int(max(1, min(2, n_train // 50)))


def _fit_tfidf_svd(train_text: pd.Series, test_text: pd.Series, col: str, n_train: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Fit TF-IDF + TruncatedSVD on training text and transform both splits."""
    warnings: List[str] = []
    created: List[str] = []

    train_docs = train_text.fillna("").astype(str).tolist()
    test_docs = test_text.fillna("").astype(str).tolist()
    n_components = _choose_svd_components(n_train)

    if n_train < 20 or not any(doc.strip() for doc in train_docs):
        warnings.append(f"skipped TF-IDF/SVD for '{col}' due to insufficient non-empty training text")
        return pd.DataFrame(index=train_text.index), pd.DataFrame(index=test_text.index), created, warnings

    try:
        vectorizer = TfidfVectorizer(
            max_features=_choose_max_features(n_train),
            min_df=_choose_min_df(n_train),
            ngram_range=(1, 2),
            strip_accents="unicode",
        )
        train_tfidf = vectorizer.fit_transform(train_docs)
        test_tfidf = vectorizer.transform(test_docs)

        if train_tfidf.shape[1] == 0:
            warnings.append(f"TF-IDF produced no features for '{col}'")
            return pd.DataFrame(index=train_text.index), pd.DataFrame(index=test_text.index), created, warnings

        n_components = min(n_components, max(1, train_tfidf.shape[1] - 1), n_train - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)

        train_out = pd.DataFrame(
            train_svd,
            index=train_text.index,
            columns=[f"{col}__svd_{i}" for i in range(train_svd.shape[1])],
        )
        test_out = pd.DataFrame(
            test_svd,
            index=test_text.index,
            columns=[f"{col}__svd_{i}" for i in range(test_svd.shape[1])],
        )
        created.extend(train_out.columns.tolist())
        return train_out, test_out, created, warnings
    except Exception as exc:
        warnings.append(f"TF-IDF/SVD failed for '{col}': {type(exc).__name__}: {exc}")
        return pd.DataFrame(index=train_text.index), pd.DataFrame(index=test_text.index), created, warnings


def add_text_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Replace text columns with simple stats and optional TF-IDF/SVD features."""
    train_out = train_df.copy()
    test_out = test_df.copy()
    metadata: Dict[str, Any] = {
        "simple_text_features": [],
        "text_svd_features": [],
        "feature_warnings": [],
    }

    n_train = len(train_out)
    for col in text_cols:
        if col not in train_out.columns and col not in test_out.columns:
            continue

        for frame in (train_out, test_out):
            if col in frame.columns:
                metadata["simple_text_features"].extend(_add_simple_text_stats(frame, col))

        if col not in train_df.columns:
            continue

        train_svd, test_svd, svd_cols, svd_warnings = _fit_tfidf_svd(
            train_df[col],
            test_df[col] if col in test_df.columns else pd.Series([""] * len(test_out)),
            col,
            n_train,
        )
        metadata["feature_warnings"].extend(svd_warnings)
        if not train_svd.empty:
            train_out = pd.concat([train_out, train_svd], axis=1)
            test_out = pd.concat([test_out, test_svd], axis=1)
            metadata["text_svd_features"].extend(svd_cols)
            
    metadata["simple_text_features"] = list(dict.fromkeys(metadata.get("simple_text_features", [])))
    metadata["text_svd_features"] = list(dict.fromkeys(metadata.get("text_svd_features", [])))
    metadata["feature_warnings"] = list(dict.fromkeys(metadata.get("feature_warnings", [])))
    return train_out, test_out, metadata
