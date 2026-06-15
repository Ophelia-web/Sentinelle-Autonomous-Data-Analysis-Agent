"""Parse metric specifications and join keys from DATA_DESCRIPTION.md."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.io_utils import normalize_name

BLOCK_AVERAGED_PATTERNS = [
    r"block[-\s]*averaged\s+mae",
    r"mean\s+over\s+\{[^}]+\}\s+of\s+mean\s+over\s+rows",
    r"mean\s+over\s+(?:categories?|groups?|blocks?|classes?)\s+of\s+mean",
    r"each\s+(?:category|group|block|class)\s+contributes?\s+equally",
    r"each\s+(?:category|group|block|class)\s+contributes?\s+1\s*/\s*n",
    r"score\s*=\s*mean\s+over\s+(?:categories?|groups?|blocks?|classes?)",
    r"averaged\s+over\s+scoring\s+(?:categories?|groups?|blocks?)",
    r"(?:categories?|groups?|blocks?)\s+contribute\s+1\s*/\s*n",
    r"unweighted\s+mean\s+of\s+(?:per[-\s]*)?(?:category|group|block)\s+mae",
]

BLOCK_AWARE_SIGNAL_PATTERNS = [
    r"scoring\s+categories?\s*:",
    r"scoring\s+blocks?\s*:",
    r"scoring\s+groups?\s*:",
    r"evaluated\s+categories?\s*:",
    r"evaluated\s+groups?\s*:",
    r"official\s+scoring\s+categories?",
    r"submission\s+categories?\s*:",
    r"blocks?\s+scored\s*:",
    r"categories?\s+are\s+nested",
    r"categories?\s+are\s+non[-\s]*exclusive",
    r"never\s+sum\s+across",
    r"do\s+not\s+sum\s+across\s+categories?",
]

SCORING_VALUE_LINE_PATTERNS = [
    r"scoring\s+categories?\s*[:=]\s*(.+?)(?:\n|$)",
    r"scoring\s+blocks?\s*[:=]\s*(.+?)(?:\n|$)",
    r"scoring\s+groups?\s*[:=]\s*(.+?)(?:\n|$)",
    r"evaluated\s+categories?\s*[:=]\s*(.+?)(?:\n|$)",
    r"evaluated\s+groups?\s*[:=]\s*(.+?)(?:\n|$)",
    r"submission\s+categories?\s*[:=]\s*(.+?)(?:\n|$)",
    r"blocks?\s+scored\s*[:=]\s*(.+?)(?:\n|$)",
]

JOIN_KEY_PATTERNS = [
    r"join\s+keys?\s*[:=]\s*([^\n.]+)",
    r"merge\s+keys?\s*[:=]\s*([^\n.]+)",
    r"join\s+on\s*[:=]\s*([^\n.]+)",
    r"keys?\s+for\s+merge\s*[:=]\s*([^\n.]+)",
    r"(?:train(?:ing)?\s+rows?\s+)?merge\s+on\s+([^\n.]+)",
    r"(?:rows?\s+)?(?:align|match|join)\s+(?:on|by|via)\s+([^\n.]+)",
]

BLOCK_COLUMN_PATTERNS = [
    r"(?:block|category|group|class)\s+column\s*[:=]\s*[`'\"]?([A-Za-z0-9_]+)",
    r"(?:optional\s+)?block/?category\s+column\s*[:=]\s*([A-Za-z0-9_,\s]+)",
]


def _clean_block_token(token: str) -> str:
    """Clean one parsed block/category value."""
    cleaned = re.sub(r"[`'\"]", "", str(token)).strip()
    cleaned = cleaned.strip("|")
    cleaned = re.sub(r"[.,;:!?]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Drop obvious explanatory fragments, not actual category values.
    lowered = cleaned.lower()
    bad_fragments = {
        "categories are nested",
        "non-exclusive",
        "non exclusive",
        "never sum across them",
        "do not sum across them",
        "n/a",
        "na",
        "none",
    }
    if lowered in bad_fragments:
        return ""
    if lowered.startswith(("categories are ", "groups are ", "blocks are ")):
        return ""
    if "never sum" in lowered:
        return ""
    return cleaned


def _split_list_values(text: str) -> List[str]:
    """Split a short natural-language list into clean values.

    This function intentionally prioritizes backtick-enclosed values. For
    Markdown table text like:
        Scoring categories: `a`, `b`, `c`. Categories are nested...
    the desired output is only [a, b, c], not the explanatory sentence.
    """
    raw = str(text)

    backtick_values = re.findall(r"`([^`]+)`", raw)
    if backtick_values:
        return _dedupe_preserve_order(
            [_clean_block_token(v) for v in backtick_values if _clean_block_token(v)]
        )

    # Remove explanatory continuation after the actual list.
    cleaned = raw
    cleaned = re.split(
        r"\.\s+(?:categories?|groups?|blocks?|classes?)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.split(
        r"\b(?:categories?|groups?|blocks?|classes?)\s+(?:are|is)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.split(
        r"\b(?:never|do\s+not)\s+sum\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    cleaned = re.sub(r"[`'\"]", "", cleaned)
    cleaned = re.sub(r"\b(?:and|or)\b", ",", cleaned, flags=re.IGNORECASE)
    parts = re.split(r"[,;|]+", cleaned)

    values = [_clean_block_token(p) for p in parts]
    return _dedupe_preserve_order([v for v in values if v])


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(v for v in values if v))


def _split_column_names(text: str) -> List[str]:
    """Extract plausible column names from a short join-key phrase."""
    values: List[str] = []
    for value in _split_list_values(text):
        candidate = re.sub(r"\s+", "_", value.strip())
        candidate = candidate.strip("`'\"()[]{}")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
            values.append(candidate)
    return values

def _extract_brace_values(desc_text: str) -> List[str]:
    values: List[str] = []
    for match in re.finditer(r"\{([^}]+)\}", desc_text):
        values.extend(_split_list_values(match.group(1)))
    return _dedupe_preserve_order(values)


def _parse_scoring_block_values(desc_text: str) -> List[str]:
    """Parse explicit scoring category/block names from DATA_DESCRIPTION text."""
    values: List[str] = []

    for pattern in SCORING_VALUE_LINE_PATTERNS:
        for match in re.finditer(pattern, desc_text, flags=re.IGNORECASE):
            raw = match.group(1)
            values.extend(_split_list_values(raw))

    # Also support exact mathematical notation such as {a, b, c}.
    if not values:
        values.extend(_extract_brace_values(desc_text))

    # Fallback for prose like "categories are a, b, c" only when the line is
    # clearly scoring-related.
    if not values:
        list_match = re.search(
            r"(?:scoring|evaluated|official)\s+(?:categories?|groups?|blocks?|classes?)\s*(?:are|:)\s*([^\n.]+)",
            desc_text,
            flags=re.IGNORECASE,
        )
        if list_match:
            values.extend(_split_list_values(list_match.group(1)))

    return _dedupe_preserve_order(values)


def _block_aware_signal_detected(desc_text: str) -> bool:
    lower = desc_text.lower()
    if any(re.search(pat, lower, flags=re.IGNORECASE) for pat in BLOCK_AVERAGED_PATTERNS):
        return True
    return any(re.search(pat, lower, flags=re.IGNORECASE) for pat in BLOCK_AWARE_SIGNAL_PATTERNS)


def _metric_source_for_block_aware(desc_text: str) -> str:
    lower = desc_text.lower()
    if any(re.search(pat, lower, flags=re.IGNORECASE) for pat in BLOCK_AVERAGED_PATTERNS):
        return "block-averaged scoring language detected in DATA_DESCRIPTION.md"
    return "scoring categories / nested non-exclusive category language detected in DATA_DESCRIPTION.md"


def default_metric_spec() -> Dict[str, Any]:
    return {
        "metric_name": "mae",
        "metric_mode": "global",
        "block_column": None,
        "block_values": None,
        "metric_source": "default global MAE",
        "higher_is_better": False,
    }


def parse_metric_spec(desc_text: str) -> Dict[str, Any]:
    """Infer evaluation metric mode from a data description."""
    spec = default_metric_spec()
    if not desc_text or not desc_text.strip():
        return spec

    if not _block_aware_signal_detected(desc_text):
        return spec

    block_values = _parse_scoring_block_values(desc_text)
    spec["metric_mode"] = "block_averaged"
    spec["metric_name"] = "mae"
    spec["block_values"] = block_values or None
    spec["metric_source"] = _metric_source_for_block_aware(desc_text)
    return spec


def parse_join_keys_from_description(desc_text: str) -> Optional[List[str]]:
    """Parse explicit join/merge keys from the data description."""
    if not desc_text:
        return None
    for pattern in JOIN_KEY_PATTERNS:
        match = re.search(pattern, desc_text, flags=re.IGNORECASE)
        if match:
            keys = _split_column_names(match.group(1))
            if keys:
                return keys
    return None


def parse_block_column_from_description(desc_text: str) -> Optional[str]:
    """Parse an explicitly named block/category column from the description."""
    if not desc_text:
        return None
    for pattern in BLOCK_COLUMN_PATTERNS:
        match = re.search(pattern, desc_text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            if "," in raw:
                return _split_list_values(raw)[0]
            return raw
    return None


def _column_value_overlap(column: pd.Series, block_values: Sequence[str]) -> float:
    """Return share of official block values found in a candidate column."""
    cleaned_blocks = [_clean_block_token(v) for v in block_values]
    block_vals = {str(v).strip().lower() for v in cleaned_blocks if str(v).strip()}
    if not block_vals:
        return 0.0

    col_vals = {str(v).strip().lower() for v in column.dropna().unique()}
    if not col_vals:
        return 0.0

    return len(col_vals & block_vals) / len(block_vals)


def resolve_block_column(
    metric_spec: Dict[str, Any],
    train_target: Optional[pd.DataFrame],
    sample_submission: Optional[pd.DataFrame],
    desc_text: str = "",
) -> tuple[Optional[str], List[str]]:
    """Resolve block column when block-averaged scoring is enabled."""
    warnings: List[str] = []
    if metric_spec.get("metric_mode") != "block_averaged":
        return None, warnings

    block_values = metric_spec.get("block_values") or []
    explicit = parse_block_column_from_description(desc_text)
    tables = [df for df in (train_target, sample_submission) if df is not None]
    n_tables = len(tables)

    if explicit:
        for df in tables:
            if explicit in df.columns:
                metric_spec["block_column"] = explicit
                return explicit, warnings
        warnings.append(f"explicit block column '{explicit}' not found in available tables")

    if not block_values:
        warnings.append(
            "Block-averaged scoring detected, but block column could not be confidently resolved."
        )
        return None, warnings

    scored: List[tuple[float, str, int, float]] = []
    for df in tables:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 30:
                continue
            overlap = _column_value_overlap(df[col], block_values)
            if overlap <= 0:
                continue
            in_both = sum(1 for other in tables if col in other.columns)
            cardinality = int(df[col].nunique(dropna=True))
            score = overlap * 10.0 + in_both * 5.0 - min(cardinality, 50) * 0.01
            if n_tables >= 2 and in_both < 2:
                score -= 3.0
            scored.append((score, col, in_both, overlap))

    if not scored:
        warnings.append(
            "Block-averaged scoring detected, but block column could not be confidently resolved."
        )
        return None, warnings

    scored.sort(key=lambda x: (-x[0], -x[2], -x[3], x[1]))
    best_col, best_in_both, best_overlap = scored[0][1], scored[0][2], scored[0][3]
    if scored[0][0] < 5 or best_overlap < 0.5:
        warnings.append(
            "Block-averaged scoring detected, but block column could not be confidently resolved."
        )
        return None, warnings
    if n_tables >= 2 and best_in_both < 2:
        warnings.append(f"block column '{best_col}' found in only one table; match may be weak")

    metric_spec["block_column"] = best_col
    return best_col, warnings


def enrich_schema_with_metric_and_joins(
    summary: Dict[str, Any],
    desc_text: str,
    train_target: Optional[pd.DataFrame] = None,
    sample_submission: Optional[pd.DataFrame] = None,
    join_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach metric_spec, block column, and join key metadata to a schema summary."""
    metric_spec = parse_metric_spec(desc_text)
    if metric_spec.get("metric_mode") == "block_averaged":
        block_col, block_warnings = resolve_block_column(
            metric_spec,
            train_target,
            sample_submission,
            desc_text,
        )
        for w in block_warnings:
            summary.setdefault("warnings", []).append(w)
        if block_col is None:
            summary.setdefault("warnings", []).append(
                "Block-averaged scoring detected, but block column could not be confidently resolved."
            )

    summary["metric_spec"] = metric_spec
    summary["join_keys"] = join_info or summary.get("join_keys")
    return summary
