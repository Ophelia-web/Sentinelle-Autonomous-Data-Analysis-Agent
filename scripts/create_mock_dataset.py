#!/usr/bin/env python3
"""Create a small synthetic generic tabular dataset under ./data for smoke testing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import data_dir, project_root

MOCK_PATHS = [
    "DATA_DESCRIPTION.md",
    "sample_submission.csv",
    "train/target.csv",
    "train/features.csv",
    "val/features.csv",
]

SEGMENTS = ["group_a", "group_b", "group_c"]
RNG = np.random.default_rng(42)


def remove_known_mock_files(data_path: Path) -> None:
    """Delete only files and folders created by this mock generator."""
    for rel in MOCK_PATHS:
        path = data_path / rel
        if path.exists():
            path.unlink()

    for folder in ("train", "val"):
        folder_path = data_path / folder
        if folder_path.exists() and folder_path.is_dir():
            try:
                folder_path.rmdir()
            except OSError:
                pass


def _make_unique_keys(n_rows: int, entity_offset: int) -> pd.DataFrame:
    """Create rows with unique (entity_id, time_key) pairs for reliable merges."""
    entity_ids = np.arange(entity_offset, entity_offset + n_rows)
    time_keys = 2020 + (np.arange(n_rows) % 6)
    segments = RNG.choice(SEGMENTS, size=n_rows)
    return pd.DataFrame(
        {
            "entity_id": entity_ids,
            "time_key": time_keys,
            "segment": segments,
        }
    )


def _make_features(keys: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(keys)
    features = keys.copy()
    features["numeric_feature_1"] = RNG.normal(10, 3, size=n_rows)
    features["numeric_feature_2"] = RNG.normal(0, 1, size=n_rows)
    features["categorical_feature"] = RNG.choice(["low", "medium", "high"], size=n_rows)

    text_values = [f"note_{i} example text" for i in range(n_rows)]
    empty_idx = RNG.choice(n_rows, size=max(1, n_rows // 10), replace=False)
    for idx in empty_idx:
        text_values[idx] = ""
    features["text_feature"] = text_values

    start = pd.Timestamp("2020-01-01")
    features["event_date"] = [
        (start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in RNG.integers(0, 1200, size=n_rows)
    ]

    missing_idx = RNG.choice(n_rows, size=max(1, n_rows // 8), replace=False)
    features.loc[missing_idx, "numeric_feature_1"] = np.nan
    features.loc[missing_idx[: max(1, len(missing_idx) // 2)], "numeric_feature_2"] = np.nan

    return features


def _make_target(keys: pd.DataFrame) -> pd.DataFrame:
    target = keys.copy()
    base = (
        2.0
        + 0.05 * target["entity_id"]
        + 0.01 * target["time_key"]
        + target["segment"].map({"group_a": 0.0, "group_b": 0.5, "group_c": 1.0})
    )
    noise = RNG.normal(0, 0.3, size=len(target))
    target["target_value"] = np.clip(base.to_numpy(dtype=float) + noise, 0.0, None)
    return target


def create_mock_dataset(root: Path | None = None) -> None:
    """Write the generic mock dataset under ./data."""
    root = root or project_root()
    data_path = data_dir(root)
    data_path.mkdir(parents=True, exist_ok=True)

    remove_known_mock_files(data_path)

    train_keys = _make_unique_keys(120, entity_offset=1)
    val_keys = _make_unique_keys(30, entity_offset=1001)

    train_target = _make_target(train_keys)
    train_features = _make_features(train_keys)
    val_features = _make_features(val_keys)

    sample_submission = val_keys.copy()
    sample_submission.insert(0, "row_id", np.arange(1, len(sample_submission) + 1))
    sample_submission["target_value"] = 0.0

    (data_path / "train").mkdir(parents=True, exist_ok=True)
    (data_path / "val").mkdir(parents=True, exist_ok=True)

    train_target.to_csv(data_path / "train" / "target.csv", index=False)
    train_features.to_csv(data_path / "train" / "features.csv", index=False)
    val_features.to_csv(data_path / "val" / "features.csv", index=False)
    sample_submission.to_csv(data_path / "sample_submission.csv", index=False)

    description = """# Mock Generic Tabular Prediction Dataset

This synthetic dataset is for local smoke testing of the Sentinelle pipeline.

## Files

- training target file: train/target.csv
- training features file: train/features.csv
- validation features file: val/features.csv
- sample submission file: sample_submission.csv

## Columns

- row id column: row_id
- target column: target_value
- required submission columns: row_id,target_value
- optional block/category column: segment

## Notes

- Training rows merge on entity_id and time_key.
- Sample submission rows align with validation feature rows via entity_id and time_key.
- target_value placeholders in sample_submission are 0.0.
"""
    (data_path / "DATA_DESCRIPTION.md").write_text(description, encoding="utf-8")


def main() -> int:
    """Create mock dataset files and print a short summary."""
    root = project_root()
    create_mock_dataset(root)
    data_path = data_dir(root)
    print("Sentinelle mock dataset created")
    print(f"  data_dir: {data_path}")
    for rel in MOCK_PATHS:
        path = data_path / rel
        print(f"  wrote: {path} ({'ok' if path.exists() else 'missing'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
