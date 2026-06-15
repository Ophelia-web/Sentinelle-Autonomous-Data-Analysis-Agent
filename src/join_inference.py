"""Safe join-key inference for training and prediction tables."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from src.io_utils import normalize_name
from src.schema_parser import parse_join_keys_from_description

KEY_HINT_TOKENS = (
    "id",
    "key",
    "period",
    "date",
    "time",
    "month",
    "year",
    "state",
    "region",
    "group",
    "jurisdiction",
    "location",
    "entity",
)

ROW_RATIO_UPPER = 1.05
ROW_RATIO_LOWER = 0.75
MATCH_RATE_LOWER = 0.75


def _missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean()) if len(series) else 1.0


def _candidate_columns(df: pd.DataFrame, exclude: Set[str]) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if _missing_rate(df[col]) > 0.5:
            continue
        cols.append(col)
    return cols


def _name_hint_score(col: str) -> float:
    norm = normalize_name(col)
    score = 0.0
    for token in KEY_HINT_TOKENS:
        if norm == token or norm.endswith(f"_{token}") or norm.startswith(f"{token}_"):
            score += 2.0
        elif token in norm:
            score += 1.0
    return score


def _shared_columns(*dfs: pd.DataFrame) -> List[str]:
    shared: Optional[Set[str]] = None
    for df in dfs:
        cols = set(df.columns)
        shared = cols if shared is None else shared & cols
    return sorted(shared or [])


def _keys_present(keys: Optional[Sequence[str]], *dfs: Optional[pd.DataFrame]) -> bool:
    """Return True if every key is present in every provided dataframe."""
    if not keys:
        return False
    for df in dfs:
        if df is None:
            return False
        if not all(k in df.columns for k in keys):
            return False
    return True


def _right_duplicate_rate(right: pd.DataFrame, keys: Sequence[str]) -> float:
    """Fraction of right-hand rows that are duplicates on the candidate key set."""
    if len(right) == 0:
        return 0.0
    unique_rows = len(right.drop_duplicates(subset=list(keys), keep="first"))
    return 1.0 - (unique_rows / len(right))


def _right_feature_match_rate(
    left: pd.DataFrame,
    merged: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
) -> float:
    """Estimate how many left rows received non-null right-side feature values."""
    right_only = [c for c in right.columns if c not in left.columns]
    if right_only:
        present = [c for c in right_only if c in merged.columns]
        if present:
            return float(merged[present].notna().any(axis=1).mean())

    overlap = [c for c in right.columns if c in left.columns and c not in keys]
    suffixed = [f"{c}_right" for c in overlap if f"{c}_right" in merged.columns]
    if suffixed:
        return float(merged[suffixed].notna().any(axis=1).mean())

    if len(merged) == len(left):
        return 1.0
    return 0.0


def _merge_metadata(
    keys: Sequence[str],
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    preserve_left_rows: bool,
    use_deduplicated_right: bool,
) -> Dict[str, Any]:
    """Compute merge diagnostics for a candidate key set."""
    right_dup_rate = _right_duplicate_rate(right, keys)
    right_unique = right.drop_duplicates(subset=list(keys), keep="first")
    right_for_merge = right_unique if use_deduplicated_right else right
    deduplicated = len(right_unique) < len(right)
    dedup_loss_rate = 1.0 - (len(right_unique) / max(len(right), 1))
    how = "left" if preserve_left_rows else "inner"
    merged = left.merge(right_for_merge, on=list(keys), how=how, suffixes=("", "_right"))

    left_rows = max(len(left), 1)
    row_ratio = len(merged) / left_rows
    metadata: Dict[str, Any] = {
        "keys": list(keys),
        "left_rows": len(left),
        "right_rows": len(right),
        "merged_rows": len(merged),
        "row_ratio": row_ratio,
        "right_duplicate_rate": right_dup_rate,
        "dedup_loss_rate": dedup_loss_rate,
        "deduplicated": deduplicated,
        "preserve_left_rows": preserve_left_rows,
    }

    if preserve_left_rows:
        metadata["feature_match_rate"] = _right_feature_match_rate(left, merged, right, keys)
        metadata["unmatched_rate"] = 1.0 - metadata["feature_match_rate"]
    else:
        metadata["feature_match_rate"] = None
        metadata["unmatched_rate"] = 1.0 - min(row_ratio, 1.0) if row_ratio <= 1.0 else 0.0

    return metadata


def _score_key_set(
    keys: Sequence[str],
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    expect_left_rows: int,
    preserve_left_rows: bool,
) -> float:
    """Score a candidate join key set; higher is better."""
    if not keys:
        return -1.0

    for key in keys:
        if key not in left.columns or key not in right.columns:
            return -1.0
        if _missing_rate(left[key]) > 0.3 or _missing_rate(right[key]) > 0.3:
            return -1.0

    right_dup_rate = _right_duplicate_rate(right, keys)
    name_bonus = sum(_name_hint_score(k) for k in keys)

    if preserve_left_rows:
        meta = _merge_metadata(keys, left, right, preserve_left_rows=True, use_deduplicated_right=True)
        if meta["merged_rows"] != expect_left_rows:
            return -1e6

        match_rate = float(meta.get("feature_match_rate") or 0.0)
        if match_rate < MATCH_RATE_LOWER:
            return -100.0 + match_rate * 50.0

        score = 100.0 * match_rate
        score -= 60.0 * right_dup_rate
        score -= 10.0 if meta["deduplicated"] else 0.0
        score += name_bonus
        return score

    # Training inner merge: score the production-like merge with a deduplicated right table.
    left_rows = max(len(left), 1)
    right_unique = right.drop_duplicates(subset=list(keys), keep="first")
    dedup_loss_rate = 1.0 - (len(right_unique) / max(len(right), 1))
    merged = left.merge(right_unique, on=list(keys), how="inner")
    row_ratio = len(merged) / left_rows

    if row_ratio > ROW_RATIO_UPPER:
        return -1e6
    if row_ratio < ROW_RATIO_LOWER:
        return -100.0 + row_ratio * 40.0

    score = 100.0 - abs(row_ratio - 1.0) * 250.0
    score -= 50.0 * right_dup_rate
    if row_ratio < 0.99:
        score -= 80.0 * dedup_loss_rate
    score += name_bonus
    score += 8.0 * len(keys)

    return score


def _generate_key_candidates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    exclude: Set[str],
    max_combo: int = 3,
) -> List[List[str]]:
    shared = [c for c in _shared_columns(left, right) if c not in exclude]
    shared.sort(key=lambda c: (-_name_hint_score(c), c))

    candidates: List[List[str]] = []
    pool = shared[:16]
    for size in range(1, min(max_combo, len(pool)) + 1):
        for combo in combinations(pool, size):
            candidates.append(list(combo))
    return candidates


def infer_join_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    exclude: Set[str],
    desc_text: str = "",
    *,
    expect_left_rows: Optional[int] = None,
    preserve_left_rows: bool = False,
    preferred_shared: Optional[Sequence[str]] = None,
) -> tuple[Optional[List[str]], str, Dict[str, Any]]:
    """Choose join keys that preserve rows and avoid many-to-many explosions."""
    expect_left_rows = expect_left_rows or len(left)
    metadata: Dict[str, Any] = {}

    explicit = parse_join_keys_from_description(desc_text)
    if explicit:
        score = _score_key_set(
            explicit,
            left,
            right,
            expect_left_rows=expect_left_rows,
            preserve_left_rows=preserve_left_rows,
        )
        if score > 20:
            metadata = _merge_metadata(
                explicit,
                left,
                right,
                preserve_left_rows=preserve_left_rows,
                use_deduplicated_right=preserve_left_rows,
            )
            return list(explicit), "description", metadata

    if preferred_shared:
        pref = [c for c in preferred_shared if c in left.columns and c in right.columns and c not in exclude]
        if pref:
            score = _score_key_set(
                pref,
                left,
                right,
                expect_left_rows=expect_left_rows,
                preserve_left_rows=preserve_left_rows,
            )
            if score > 20:
                metadata = _merge_metadata(
                    pref,
                    left,
                    right,
                    preserve_left_rows=preserve_left_rows,
                    use_deduplicated_right=preserve_left_rows,
                )
                return pref, "cross-table shared keys", metadata

    best_keys: Optional[List[str]] = None
    best_score = -1.0
    best_metadata: Dict[str, Any] = {}

    for keys in _generate_key_candidates(left, right, exclude):
        score = _score_key_set(
            keys,
            left,
            right,
            expect_left_rows=expect_left_rows,
            preserve_left_rows=preserve_left_rows,
        )
        if score > best_score:
            best_score = score
            best_keys = keys
            best_metadata = _merge_metadata(
                keys,
                left,
                right,
                preserve_left_rows=preserve_left_rows,
                use_deduplicated_right=preserve_left_rows,
            )

    if best_keys and best_score > 20:
        return best_keys, "heuristic candidate scoring", best_metadata
    return None, "no confident join keys found", metadata


def infer_pipeline_join_keys(
    train_target: Optional[pd.DataFrame],
    train_features: Optional[pd.DataFrame],
    sample_submission: Optional[pd.DataFrame],
    validation_features: Optional[pd.DataFrame],
    exclude: Set[str],
    desc_text: str = "",
) -> Dict[str, Any]:
    """Infer join keys for train and prediction merges.

    Prediction keys are inferred first because sample_submission and validation
    features usually describe the rows that must be predicted. When those keys
    also exist in train_target and train_features, they are strong candidates
    for training merge keys as well.
    """
    result: Dict[str, Any] = {
        "train_keys": None,
        "predict_keys": None,
        "train_source": None,
        "predict_source": None,
        "train_merge_metadata": None,
        "predict_merge_metadata": None,
        "warnings": [],
    }

    train_shared: Optional[List[str]] = None
    if train_target is not None and train_features is not None:
        train_shared = _shared_columns(train_target, train_features)

    prediction_preferred: Optional[List[str]] = None
    if (
        train_shared
        and sample_submission is not None
        and validation_features is not None
    ):
        prediction_preferred = [
            c
            for c in train_shared
            if c in sample_submission.columns
            and c in validation_features.columns
            and c not in exclude
        ]

    # 1. Infer prediction keys first.
    if sample_submission is not None and validation_features is not None:
        predict_keys, predict_source, predict_meta = infer_join_keys(
            sample_submission,
            validation_features,
            exclude,
            desc_text,
            expect_left_rows=len(sample_submission),
            preserve_left_rows=True,
            preferred_shared=prediction_preferred,
        )
        result["predict_keys"] = predict_keys
        result["predict_source"] = predict_source
        result["predict_merge_metadata"] = predict_meta

        if predict_meta.get("deduplicated"):
            result["warnings"].append(
                f"prediction features had duplicate keys {predict_keys}; deduplicated before merge"
            )
        if predict_meta.get("right_duplicate_rate", 0.0) > 0.01 and predict_keys:
            result["warnings"].append(
                f"predict join keys {predict_keys} observed right duplicate rate "
                f"{predict_meta.get('right_duplicate_rate'):.3f}"
            )
        if predict_keys is None:
            result["warnings"].append(
                "no safe prediction join keys found; using sample_submission without validation feature merge"
            )

    # 2. Infer train keys, preferring prediction keys when they are available
    #    and present in both training tables.
    train_preferred: Optional[List[str]] = None
    if _keys_present(result.get("predict_keys"), train_target, train_features):
        train_preferred = list(result["predict_keys"])
    elif train_shared:
        train_preferred = [
            c
            for c in train_shared
            if c not in exclude
            and (
                sample_submission is None
                or c in sample_submission.columns
                or validation_features is None
                or c in validation_features.columns
            )
        ]

    if train_target is not None and train_features is not None:
        train_keys, train_source, train_meta = infer_join_keys(
            train_target,
            train_features,
            exclude,
            desc_text,
            expect_left_rows=len(train_target),
            preserve_left_rows=False,
            preferred_shared=train_preferred,
        )

        # If prediction-derived composite keys are present in both train tables
        # and are unique on the right-hand feature table, trust them over a
        # weaker single-column heuristic.
        if (
            train_preferred
            and _keys_present(train_preferred, train_target, train_features)
        ):
            preferred_meta = _merge_metadata(
                train_preferred,
                train_target,
                train_features,
                preserve_left_rows=False,
                use_deduplicated_right=False,
            )
            preferred_dup = float(preferred_meta.get("right_duplicate_rate", 1.0))
            preferred_ratio = float(preferred_meta.get("row_ratio", 0.0))

            if preferred_dup <= 0.01 and ROW_RATIO_LOWER <= preferred_ratio <= ROW_RATIO_UPPER:
                if (
                    train_keys is None
                    or len(train_preferred) > len(train_keys)
                    or float((train_meta or {}).get("right_duplicate_rate", 1.0)) > preferred_dup
                ):
                    train_keys = list(train_preferred)
                    train_source = "prediction-derived composite keys"
                    train_meta = preferred_meta

        result["train_keys"] = train_keys
        result["train_source"] = train_source
        result["train_merge_metadata"] = train_meta

        if train_meta.get("deduplicated"):
            result["warnings"].append(
                f"train right table had duplicate keys {train_keys}; deduplication would be needed for left merge"
            )
        if train_meta.get("right_duplicate_rate", 0.0) > 0.01 and train_keys:
            result["warnings"].append(
                f"train join keys {train_keys} observed right duplicate rate "
                f"{train_meta.get('right_duplicate_rate'):.3f}"
            )
        if train_keys is None:
            result["warnings"].append("no safe train join keys found; using train_target without feature merge")

    return result


def safe_merge_tables(
    left: pd.DataFrame,
    right: Optional[pd.DataFrame],
    keys: Optional[List[str]],
    *,
    how: str,
    preserve_left_rows: bool,
) -> tuple[pd.DataFrame, List[str]]:
    """Merge tables using deduplicated right-hand keys."""
    warnings: List[str] = []
    if right is None:
        return left.copy(), warnings
    if not keys:
        return left.copy(), ["merge skipped because join keys are unavailable"]

    missing = [k for k in keys if k not in left.columns or k not in right.columns]
    if missing:
        return left.copy(), [f"merge skipped; missing join keys: {missing}"]

    dup_rate = _right_duplicate_rate(right, keys)
    if dup_rate > 0.01:
        warnings.append(f"right table duplicate rate on keys {keys}: {dup_rate:.3f}")

    right_unique = right.drop_duplicates(subset=keys, keep="first")
    if len(right_unique) < len(right):
        warnings.append(f"deduplicated validation/features table on keys {keys}")

    merged = left.merge(right_unique, on=keys, how=how, suffixes=("", "_feat"))
    if preserve_left_rows and len(merged) != len(left):
        warnings.append("prediction merge changed row count unexpectedly")

    row_ratio = len(merged) / max(len(left), 1)
    if not preserve_left_rows and row_ratio > ROW_RATIO_UPPER:
        warnings.append(f"train merge row explosion detected (row_ratio={row_ratio:.3f}) on keys {keys}")

    return merged, warnings
