from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings


DATA_DESCRIPTION_CANDIDATES = [
    "DATA_DESCRIPTION.md",
    "Data_Description.md",
    "data_description.md",
    "README.md",
]


@dataclass
class CsvProfile:
    path: str
    rel_path: str
    name: str
    n_rows: Optional[int]
    n_cols: Optional[int]
    columns: List[str]
    dtypes: Dict[str, str]
    sample_records: List[Dict[str, Any]]
    role: str
    role_score: float
    role_reasons: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    text_columns: List[str]
    datetime_like_columns: List[str]
    id_like_columns: List[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir(root: Optional[Path] = None) -> Path:
    root = root or project_root()
    return root / "data"


def artifacts_dir(root: Optional[Path] = None) -> Path:
    root = root or project_root()
    path = root / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")
    except FileNotFoundError:
        return ""


def find_data_description(data_path: Path) -> Optional[Path]:
    for name in DATA_DESCRIPTION_CANDIDATES:
        candidate = data_path / name
        if candidate.exists():
            return candidate

    md_files = sorted(data_path.rglob("*.md"))
    if md_files:
        preferred = [p for p in md_files if "description" in p.name.lower()]
        return preferred[0] if preferred else md_files[0]
    return None


def read_data_description(data_path: Path) -> Tuple[Optional[str], Optional[str]]:
    desc_path = find_data_description(data_path)
    if desc_path is None:
        return None, None
    return str(desc_path), read_text_safely(desc_path)


def find_csv_files(data_path: Path) -> List[Path]:
    if not data_path.exists():
        return []
    return sorted(data_path.rglob("*.csv"))


def load_csv_head(path: Path, nrows: int = 200) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1")


def load_csv_for_schema_inference(path: Path, max_rows: int = 200_000) -> pd.DataFrame:
    """Load enough rows for schema/join inference.

    `load_csv_head(..., nrows=200)` is often enough for column detection, but
    it can be unsafe for join-key inference when files are sorted differently.
    For moderately sized competition files, read the full file. For very large
    files, cap the read for safety.
    """
    n_rows = count_csv_rows(path)
    try:
        if n_rows is not None and n_rows <= max_rows:
            return pd.read_csv(path)
        return pd.read_csv(path, nrows=max_rows)
    except Exception:
        if n_rows is not None and n_rows <= max_rows:
            return pd.read_csv(path, encoding="latin-1")
        return pd.read_csv(path, nrows=max_rows, encoding="latin-1")


def count_csv_rows(path: Path) -> Optional[int]:
    try:
        with path.open("rb") as f:
            count = sum(1 for _ in f)
        return max(count - 1, 0)
    except Exception:
        return None


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def column_name_score(columns: List[str], patterns: List[str]) -> float:
    normalized = [normalize_name(c) for c in columns]
    score = 0.0
    for p in patterns:
        p_norm = normalize_name(p)
        if p_norm in normalized:
            score += 2.0
        elif any(p_norm in c for c in normalized):
            score += 1.0
    return score


def infer_row_id_column(columns: List[str], sample_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    normalized = {normalize_name(c): c for c in columns}

    preferred = [
        "row_id",
        "id",
        "record_id",
        "sample_id",
        "submission_id",
        "index",
    ]
    for key in preferred:
        if key in normalized:
            return normalized[key]

    if sample_df is not None:
        for col in columns:
            s = sample_df[col]
            if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.is_unique:
                return col

    return columns[0] if columns else None


def looks_like_prediction_placeholder(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() < 0.8:
        return False
    if numeric.nunique(dropna=True) <= 2:
        return True
    return False


def infer_target_column(
    train_df: Optional[pd.DataFrame],
    sample_submission_df: Optional[pd.DataFrame],
    row_id_col: Optional[str],
) -> Optional[str]:
    if sample_submission_df is not None and len(sample_submission_df.columns) >= 2:
        candidates = [c for c in sample_submission_df.columns if c != row_id_col]
        numeric_candidates = []
        for c in candidates:
            if pd.api.types.is_numeric_dtype(sample_submission_df[c]) or looks_like_prediction_placeholder(sample_submission_df[c]):
                numeric_candidates.append(c)
        if numeric_candidates:
            return numeric_candidates[-1]
        return candidates[-1] if candidates else None

    if train_df is not None:
        cols = list(train_df.columns)
        normalized = {normalize_name(c): c for c in cols}
        for key in ["target", "label", "y", "response", "outcome", "value", "rate", "score"]:
            if key in normalized:
                return normalized[key]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]
        if numeric_cols:
            return numeric_cols[-1]

    return None


def detect_column_types(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Dict[str, List[str]]:
    exclude_set = set(exclude or [])

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    text_cols: List[str] = []
    datetime_like_cols: List[str] = []
    id_like_cols: List[str] = []

    n = max(len(df), 1)

    for col in df.columns:
        if col in exclude_set:
            continue

        s = df[col]
        col_norm = normalize_name(col)

        non_null = s.dropna()
        nunique = int(non_null.nunique()) if len(non_null) else 0
        unique_ratio = nunique / n

        if any(token in col_norm for token in ["id", "uuid", "key"]) or unique_ratio > 0.9:
            id_like_cols.append(col)

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_like_cols.append(col)
            continue

        as_str = non_null.astype(str)
        if len(as_str) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(as_str.head(50), errors="coerce")
            if parsed.notna().mean() > 0.7:
                datetime_like_cols.append(col)
                continue

            avg_len = float(as_str.str.len().mean())
            max_len = int(as_str.str.len().max())
            if avg_len >= 50 or max_len >= 200:
                text_cols.append(col)
                continue

        categorical_cols.append(col)

    return {
        "numeric_columns": sorted(set(numeric_cols)),
        "categorical_columns": sorted(set(categorical_cols)),
        "text_columns": sorted(set(text_cols)),
        "datetime_like_columns": sorted(set(datetime_like_cols)),
        "id_like_columns": sorted(set(id_like_cols)),
    }


def infer_csv_role(path: Path, df: pd.DataFrame, desc_text: str, data_path: Path) -> Tuple[str, float, List[str]]:
    rel = path.relative_to(data_path).as_posix().lower()
    name = path.name.lower()
    cols = list(df.columns)
    col_norms = [normalize_name(c) for c in cols]

    scores = {
        "sample_submission": 0.0,
        "train_target": 0.0,
        "train_features": 0.0,
        "validation_features": 0.0,
        "unknown": 0.1,
    }
    reasons: Dict[str, List[str]] = {k: [] for k in scores}

    def add(role: str, value: float, reason: str) -> None:
        scores[role] += value
        reasons[role].append(reason)

    if "submission" in name or "sample_submission" in rel:
        add("sample_submission", 5.0, "filename/path contains submission")
    if "row_id" in col_norms:
        add("sample_submission", 2.0, "contains row_id-like column")
    if len(cols) >= 2 and any("id" in c for c in col_norms):
        add("sample_submission", 0.5, "contains id-like column")

    if any(token in rel for token in ["train", "training"]):
        add("train_features", 1.5, "path indicates train")
        add("train_target", 1.5, "path indicates train")
    if any(token in rel for token in ["val", "valid", "validation", "test", "holdout"]):
        add("validation_features", 3.0, "path indicates validation/test")

    target_col_score = column_name_score(cols, ["target", "label", "response", "outcome", "y", "rate", "value", "score"])
    if target_col_score > 0:
        add("train_target", target_col_score, "contains target-like column name")

    feature_hint_score = column_name_score(cols, ["feature", "covariate", "predictor"])
    if feature_hint_score > 0:
        add("train_features", feature_hint_score, "contains feature/covariate-like column name")
        add("validation_features", feature_hint_score, "contains feature/covariate-like column name")

    if "target" in name or "label" in name or "train_y" in name:
        add("train_target", 4.0, "filename indicates target/label")
    if "covariate" in name or "feature" in name or "x_" in name:
        if any(token in rel for token in ["val", "valid", "validation", "test"]):
            add("validation_features", 4.0, "filename indicates validation/test features")
        elif any(token in rel for token in ["train", "training"]):
            add("train_features", 4.0, "filename indicates train features")
        else:
            add("train_features", 1.0, "filename indicates features")
            add("validation_features", 1.0, "filename indicates features")

    desc_lower = desc_text.lower()
    if path.name.lower() in desc_lower:
        if "sample_submission" in desc_lower or "submission" in desc_lower:
            pass
        add("unknown", 0.1, "file mentioned in data description")

    # Heuristic: a target file is often narrower than a feature file and contains at least one numeric outcome.
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) <= 8 and len(numeric_cols) >= 1 and any(token in rel for token in ["train", "training"]):
        add("train_target", 1.0, "narrow train CSV with numeric column")

    # Avoid classifying sample submission as train target.
    if scores["sample_submission"] >= 5:
        scores["train_target"] *= 0.2
        scores["train_features"] *= 0.2
        scores["validation_features"] *= 0.2

    best_role = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_role, float(scores[best_role]), reasons[best_role]


def profile_csv(path: Path, data_path: Path, desc_text: str) -> CsvProfile:
    df = load_csv_head(path)
    n_rows = count_csv_rows(path)
    n_cols = int(df.shape[1])
    role, role_score, role_reasons = infer_csv_role(path, df, desc_text, data_path)

    type_info = detect_column_types(df)

    sample_records = df.head(5).replace({np.nan: None}).to_dict(orient="records")

    return CsvProfile(
        path=str(path),
        rel_path=path.relative_to(data_path).as_posix(),
        name=path.name,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=list(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        sample_records=sample_records,
        role=role,
        role_score=role_score,
        role_reasons=role_reasons,
        numeric_columns=type_info["numeric_columns"],
        categorical_columns=type_info["categorical_columns"],
        text_columns=type_info["text_columns"],
        datetime_like_columns=type_info["datetime_like_columns"],
        id_like_columns=type_info["id_like_columns"],
    )


def choose_best_profile(profiles: List[CsvProfile], role: str) -> Optional[CsvProfile]:
    candidates = [p for p in profiles if p.role == role]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p.role_score, p.n_rows or 0), reverse=True)[0]


def build_schema_summary(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or project_root()
    ddir = data_dir(root)
    desc_path, desc_text = read_data_description(ddir)
    desc_text = desc_text or ""

    csv_paths = find_csv_files(ddir)

    profiles: List[CsvProfile] = []
    for path in csv_paths:
        try:
            profiles.append(profile_csv(path, ddir, desc_text))
        except Exception as exc:
            profiles.append(
                CsvProfile(
                    path=str(path),
                    rel_path=path.relative_to(ddir).as_posix() if ddir.exists() else str(path),
                    name=path.name,
                    n_rows=None,
                    n_cols=None,
                    columns=[],
                    dtypes={},
                    sample_records=[],
                    role="unknown",
                    role_score=0.0,
                    role_reasons=[f"profiling failed: {type(exc).__name__}: {exc}"],
                    numeric_columns=[],
                    categorical_columns=[],
                    text_columns=[],
                    datetime_like_columns=[],
                    id_like_columns=[],
                )
            )

    sample_profile = choose_best_profile(profiles, "sample_submission")
    train_target_profile = choose_best_profile(profiles, "train_target")
    train_features_profile = choose_best_profile(profiles, "train_features")
    val_features_profile = choose_best_profile(profiles, "validation_features")

    sample_df = load_csv_head(Path(sample_profile.path)) if sample_profile else None
    train_target_df = load_csv_head(Path(train_target_profile.path)) if train_target_profile else None

    # Head samples are fine for row-id/target inference, but join-key inference
    # needs enough rows to avoid false non-overlap when files are sorted differently.
    sample_join_df = load_csv_for_schema_inference(Path(sample_profile.path)) if sample_profile else None
    train_target_join_df = (
        load_csv_for_schema_inference(Path(train_target_profile.path))
        if train_target_profile
        else None
    )

    row_id_col = infer_row_id_column(sample_profile.columns, sample_df) if sample_profile else None
    target_col = infer_target_column(train_target_df, sample_df, row_id_col)

    submission_columns = None
    if sample_profile is not None:
        if row_id_col and target_col and row_id_col in sample_profile.columns:
            submission_columns = [row_id_col, target_col]
        elif len(sample_profile.columns) >= 2:
            submission_columns = [sample_profile.columns[0], sample_profile.columns[-1]]

    summary: Dict[str, Any] = {
        "project_root": str(root),
        "data_dir": str(ddir),
        "data_dir_exists": ddir.exists(),
        "data_description_path": desc_path,
        "data_description_present": desc_path is not None,
        "n_csv_files": len(csv_paths),
        "files": [asdict(p) for p in profiles],
        "inferred_roles": {
            "sample_submission": asdict(sample_profile) if sample_profile else None,
            "train_target": asdict(train_target_profile) if train_target_profile else None,
            "train_features": asdict(train_features_profile) if train_features_profile else None,
            "validation_features": asdict(val_features_profile) if val_features_profile else None,
        },
        "row_id_column": row_id_col,
        "target_column": target_col,
        "submission_columns": submission_columns,
        "status": "ok" if csv_paths else "no_csv_files_found",
        "warnings": [],
    }

    if not ddir.exists():
        summary["warnings"].append("data directory does not exist")
    if desc_path is None:
        summary["warnings"].append("no DATA_DESCRIPTION.md file found")
    if not csv_paths:
        summary["warnings"].append("no CSV files found under data/")
    if sample_profile is None:
        summary["warnings"].append("sample submission file not identified")
    if train_target_profile is None:
        summary["warnings"].append("training target file not identified")
    if val_features_profile is None:
        summary["warnings"].append("validation/test feature file not identified")
    if target_col is None:
        summary["warnings"].append("target column not identified")
    if row_id_col is None:
        summary["warnings"].append("row id column not identified")

    from src.join_inference import infer_pipeline_join_keys
    from src.schema_parser import parse_metric_spec, resolve_block_column

    metric_spec = parse_metric_spec(desc_text)
    block_col: Optional[str] = None
    if metric_spec.get("metric_mode") == "block_averaged":
        block_col, block_warnings = resolve_block_column(
            metric_spec,
            train_target_join_df if train_target_join_df is not None else train_target_df,
            sample_join_df if sample_join_df is not None else sample_df,
            desc_text,
        )
        for warning in block_warnings:
            summary["warnings"].append(warning)
        if block_col is None:
            summary["warnings"].append(
                "Block-averaged scoring detected, but block column could not be confidently resolved."
            )

    exclude_cols = {c for c in (row_id_col, target_col, block_col) if c}
    train_features_df = (
        load_csv_for_schema_inference(Path(train_features_profile.path))
        if train_features_profile
        else None
    )
    val_features_df = (
        load_csv_for_schema_inference(Path(val_features_profile.path))
        if val_features_profile
        else None
    )

    join_info = infer_pipeline_join_keys(
        train_target_join_df if train_target_join_df is not None else train_target_df,
        train_features_df,
        sample_join_df if sample_join_df is not None else sample_df,
        val_features_df,
        exclude_cols,
        desc_text,
    )
    for warning in join_info.get("warnings", []):
        summary["warnings"].append(warning)

    summary["metric_spec"] = metric_spec
    summary["join_keys"] = {
        "train_keys": join_info.get("train_keys"),
        "predict_keys": join_info.get("predict_keys"),
        "train_source": join_info.get("train_source"),
        "predict_source": join_info.get("predict_source"),
        "train_merge_metadata": join_info.get("train_merge_metadata"),
        "predict_merge_metadata": join_info.get("predict_merge_metadata"),
        "warnings": join_info.get("warnings", []),
    }

    return summary


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def schema_summary_to_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Schema Summary")
    lines.append("")
    lines.append(f"- Status: `{summary.get('status')}`")
    lines.append(f"- Data directory: `{summary.get('data_dir')}`")
    lines.append(f"- Data description present: `{summary.get('data_description_present')}`")
    lines.append(f"- Number of CSV files: `{summary.get('n_csv_files')}`")
    lines.append(f"- Row id column: `{summary.get('row_id_column')}`")
    lines.append(f"- Target column: `{summary.get('target_column')}`")
    lines.append(f"- Submission columns: `{summary.get('submission_columns')}`")
    lines.append("")

    warnings = summary.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Inferred roles")
    roles = summary.get("inferred_roles", {})
    for role, profile in roles.items():
        if profile is None:
            lines.append(f"- `{role}`: not identified")
        else:
            lines.append(
                f"- `{role}`: `{profile.get('rel_path')}` "
                f"({profile.get('n_rows')} rows, {profile.get('n_cols')} cols, score={profile.get('role_score')})"
            )
    lines.append("")

    lines.append("## CSV profiles")
    for p in summary.get("files", []):
        lines.append(f"### `{p.get('rel_path')}`")
        lines.append(f"- Role: `{p.get('role')}`")
        lines.append(f"- Shape: `{p.get('n_rows')}` rows × `{p.get('n_cols')}` columns")
        lines.append(f"- Columns: `{', '.join(p.get('columns') or [])}`")
        reasons = p.get("role_reasons") or []
        if reasons:
            lines.append("- Role reasons:")
            for r in reasons:
                lines.append(f"  - {r}")
        lines.append("")

    return "\n".join(lines)


def save_schema_summary(summary: Dict[str, Any], root: Optional[Path] = None) -> Tuple[Path, Path]:
    root = root or project_root()
    adir = artifacts_dir(root)
    json_path = adir / "schema_summary.json"
    md_path = adir / "schema_summary.md"
    write_json(summary, json_path)
    md_path.write_text(schema_summary_to_markdown(summary), encoding="utf-8")
    return json_path, md_path
