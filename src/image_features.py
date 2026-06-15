"""Optional lightweight image sidecar feature extraction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_MATCH_RATE = 0.30
WEAK_PLACEHOLDERS = {"", "na", "n/a", "none", "null", "missing", "__missing__"}


@dataclass
class ImageEntry:
    path: Path
    norm_stem: str
    token_set: Set[str]


def find_image_files(data_dir: Path) -> List[Path]:
    """Recursively find image files under a data directory."""
    if not data_dir.exists():
        return []
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _normalize_stem(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")


def _filename_tokens(path: Path) -> Set[str]:
    stem = _normalize_stem(path)
    tokens = {part for part in stem.split("_") if part}
    if stem:
        tokens.add(stem.replace("_", ""))
    return tokens


def _is_weak_token(token: str, column_name: str, only_row_id_keys: bool) -> bool:
    cleaned = str(token).strip()
    if not cleaned or cleaned.lower() in WEAK_PLACEHOLDERS:
        return True

    col_norm = re.sub(r"[^a-z0-9]+", "_", column_name.lower()).strip("_")
    if col_norm == "row_id" and not only_row_id_keys:
        return True
    if col_norm == "row_id" and cleaned.isdigit() and len(cleaned) <= 4:
        return True
    if cleaned.isdigit() and len(cleaned) <= 2:
        return True
    return False


def _row_informative_tokens(
    row: pd.Series,
    key_columns: Sequence[str],
) -> List[Tuple[str, str, str]]:
    """Return informative (column, normalized_token, original_value) tuples for a row."""
    available = [col for col in key_columns if col in row.index and not pd.isna(row[col])]
    non_row_id = [
        col
        for col in available
        if re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_") != "row_id"
    ]
    only_row_id_keys = len(non_row_id) == 0

    tokens: List[Tuple[str, str, str]] = []
    for col in key_columns:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        original = str(value).strip()
        normalized = _normalize_token(original)
        if _is_weak_token(original, col, only_row_id_keys):
            continue
        if normalized:
            tokens.append((col, normalized, original))
    return tokens


def _token_matches_image(token_norm: str, token_orig: str, entry: ImageEntry) -> bool:
    if token_norm in entry.token_set:
        return True
    if token_norm and token_norm in entry.norm_stem.replace("_", ""):
        return True
    lowered_name = entry.path.name.lower()
    if token_orig.lower() in lowered_name:
        return True
    return False


def _score_image_match(row_tokens: Sequence[Tuple[str, str, str]], entry: ImageEntry) -> int:
    if not row_tokens:
        return 0
    matched = sum(1 for _, norm, orig in row_tokens if _token_matches_image(norm, orig, entry))
    if matched < len(row_tokens):
        return 0
    return matched * 10 + len(row_tokens)


def _build_image_index(image_paths: Sequence[Path]) -> Tuple[List[ImageEntry], Dict[str, Set[Path]]]:
    entries: List[ImageEntry] = []
    token_to_paths: Dict[str, Set[Path]] = {}
    for path in image_paths:
        entry = ImageEntry(path=path, norm_stem=_normalize_stem(path), token_set=_filename_tokens(path))
        entries.append(entry)
        for token in entry.token_set:
            token_to_paths.setdefault(token, set()).add(path)
    return entries, token_to_paths


def _candidate_paths_for_row(
    row_tokens: Sequence[Tuple[str, str, str]],
    token_to_paths: Dict[str, Set[Path]],
    image_index: Sequence[ImageEntry],
) -> Set[Path]:
    if not row_tokens:
        return set()

    candidate_sets: List[Set[Path]] = []
    for _, norm, orig in row_tokens:
        paths = set(token_to_paths.get(norm, set()))
        if not paths:
            paths = {
                entry.path
                for entry in image_index
                if _token_matches_image(norm, orig, entry)
            }
        candidate_sets.append(paths)

    if not candidate_sets:
        return set()
    candidates = set.intersection(*candidate_sets)
    return candidates


def _match_row_to_image(
    row_tokens: Sequence[Tuple[str, str, str]],
    candidates: Set[Path],
    image_index: Sequence[ImageEntry],
) -> Tuple[Optional[Path], Optional[str]]:
    if not row_tokens or not candidates:
        return None, None

    path_to_entry = {entry.path: entry for entry in image_index}
    scored: List[Tuple[int, Path]] = []
    for path in candidates:
        entry = path_to_entry.get(path)
        if entry is None:
            continue
        score = _score_image_match(row_tokens, entry)
        if score > 0:
            scored.append((score, path))

    if not scored:
        return None, "no composite-token match"
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    best_score = scored[0][0]
    best_paths = [path for score, path in scored if score == best_score]
    if len(best_paths) == 1:
        return best_paths[0], None
    return None, f"ambiguous image match among {len(best_paths)} files"


def _build_image_lookup(
    df: pd.DataFrame,
    image_paths: Sequence[Path],
    key_columns: Sequence[str],
) -> Tuple[Dict[int, Path], float, List[str]]:
    image_index, token_to_paths = _build_image_index(image_paths)
    matches: Dict[int, Path] = {}
    warnings: List[str] = []

    for idx, row in df.iterrows():
        row_tokens = _row_informative_tokens(row, key_columns)
        candidates = _candidate_paths_for_row(row_tokens, token_to_paths, image_index)
        matched_path, warning = _match_row_to_image(row_tokens, candidates, image_index)
        if matched_path is not None:
            matches[int(idx)] = matched_path
        elif warning and len(row_tokens) > 0:
            warnings.append(f"row {idx}: {warning}")

    match_rate = len(matches) / max(len(df), 1)
    return matches, match_rate, warnings


def _image_signature(image_paths: Sequence[Path]) -> str:
    payload = "|".join(str(path.resolve()) for path in image_paths)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_image_array(path: Path) -> Optional[np.ndarray]:
    """Load an image as a float numpy array."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt

        arr = np.asarray(plt.imread(path), dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr
    except Exception:
        return None


def extract_image_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute cheap image statistics from an array."""
    flat = arr.reshape(-1)
    stats = {
        "img_mean": float(np.mean(flat)),
        "img_std": float(np.std(flat)),
        "img_min": float(np.min(flat)),
        "img_max": float(np.max(flat)),
        "img_q25": float(np.quantile(flat, 0.25)),
        "img_q50": float(np.quantile(flat, 0.50)),
        "img_q75": float(np.quantile(flat, 0.75)),
        "img_nonzero_ratio": float(np.mean(flat > 0)),
    }
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        for channel in ("r", "g", "b"):
            channel_vals = arr[..., ["r", "g", "b"].index(channel)].reshape(-1)
            stats[f"img_{channel}_mean"] = float(np.mean(channel_vals))
            stats[f"img_{channel}_std"] = float(np.std(channel_vals))
    return stats


def _cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def _read_cache(
    cache_path: Path,
    expected_rows: int,
    key_columns: Sequence[str],
    image_paths: Sequence[Path],
    *,
    train_len: int,
    test_len: int,
    data_dir: Optional[Path],
) -> Optional[pd.DataFrame]:
    meta_path = _cache_metadata_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        cached = pd.read_csv(cache_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("n_rows", -1)) != expected_rows:
            return None
        if list(meta.get("key_columns") or []) != list(key_columns):
            return None
        if int(meta.get("train_len", -1)) != int(train_len):
            return None
        if int(meta.get("test_len", -1)) != int(test_len):
            return None
        if str(meta.get("data_dir", "")) != str(data_dir.resolve() if data_dir else ""):
            return None
        if meta.get("image_signature") != _image_signature(image_paths):
            return None
        return cached
    except Exception:
        return None


def _write_cache(cache_path: Path, frame: pd.DataFrame, meta: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    _cache_metadata_path(cache_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_image_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: Optional[Path],
    key_columns: Optional[Sequence[str]],
    cache_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Return image-only feature frames aligned to train/test row indices."""
    metadata: Dict[str, Any] = {
        "image_features_used": False,
        "n_images_found": 0,
        "n_rows_matched": 0,
        "image_match_rate": 0.0,
        "image_feature_columns": [],
        "image_feature_warnings": [],
        "image_feature_notes": [],
    }

    empty_train = pd.DataFrame(index=train_df.index)
    empty_test = pd.DataFrame(index=test_df.index)

    if data_dir is None or not key_columns:
        metadata["image_feature_notes"].append("image features skipped: data directory or key columns unavailable")
        return empty_train, empty_test, metadata

    image_paths = find_image_files(data_dir)
    metadata["n_images_found"] = len(image_paths)
    if not image_paths:
        metadata["image_feature_notes"].append("no image files found under data/; image features not used")
        return empty_train, empty_test, metadata

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    available_keys = [col for col in key_columns if col in combined.columns]
    if not available_keys:
        metadata["image_feature_warnings"].append("image features skipped: no usable key columns for filename matching")
        return empty_train, empty_test, metadata

    if cache_path is not None:
        cached = _read_cache(
            cache_path,
            len(combined),
            available_keys,
            image_paths,
            train_len=len(train_df),
            test_len=len(test_df),
            data_dir=data_dir,
        )
        if cached is not None:
            feature_cols = [c for c in cached.columns if c.startswith("img_")]
            if feature_cols:
                metadata["image_features_used"] = True
                metadata["image_feature_columns"] = feature_cols
                metadata["n_rows_matched"] = int((cached[feature_cols] != 0).any(axis=1).sum())
                metadata["image_match_rate"] = metadata["n_rows_matched"] / max(len(combined), 1)
                train_len = len(train_df)
                return cached.iloc[:train_len][feature_cols].reset_index(drop=True), cached.iloc[train_len:][feature_cols].reset_index(drop=True), metadata

    lookup, match_rate, match_warnings = _build_image_lookup(combined, image_paths, available_keys)
    metadata["n_rows_matched"] = len(lookup)
    metadata["image_match_rate"] = match_rate
    metadata["image_feature_warnings"].extend(match_warnings[:20])

    if match_rate < MIN_MATCH_RATE:
        metadata["image_feature_warnings"].append(
            f"image features skipped: confident match rate too low ({match_rate:.3f})"
        )
        return empty_train, empty_test, metadata

    feature_rows: List[Dict[str, float]] = []
    feature_cols: List[str] = []
    for idx in range(len(combined)):
        path = lookup.get(idx)
        if path is None:
            feature_rows.append({})
            continue
        arr = _load_image_array(path)
        if arr is None:
            feature_rows.append({})
            metadata["image_feature_warnings"].append(f"failed to load image: {path}")
            continue
        stats = extract_image_stats(arr)
        feature_rows.append(stats)
        feature_cols = sorted({*feature_cols, *stats.keys()})

    if not feature_cols:
        metadata["image_feature_warnings"].append("image features skipped: no image statistics could be extracted")
        return empty_train, empty_test, metadata

    feature_frame = pd.DataFrame(feature_rows).reindex(columns=feature_cols).fillna(0.0)
    if cache_path is not None:
        try:
            _write_cache(
                cache_path,
                feature_frame,
                {
                    "n_rows": len(combined),
                    "train_len": len(train_df),
                    "test_len": len(test_df),
                    "data_dir": str(data_dir.resolve() if data_dir else ""),
                    "n_images_found": len(image_paths),
                    "key_columns": list(available_keys),
                    "image_signature": _image_signature(image_paths),
                },
            )
        except Exception as exc:
            metadata["image_feature_warnings"].append(f"failed to write image feature cache: {type(exc).__name__}: {exc}")

    metadata["image_features_used"] = True
    metadata["image_feature_columns"] = feature_cols
    train_len = len(train_df)
    return (
        feature_frame.iloc[:train_len].reset_index(drop=True),
        feature_frame.iloc[train_len:].reset_index(drop=True),
        metadata,
    )
