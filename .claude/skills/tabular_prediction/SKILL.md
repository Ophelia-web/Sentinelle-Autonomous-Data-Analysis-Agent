---
name: tabular_prediction
description: General-purpose skill for hidden-target tabular prediction with submission.csv and report.pdf outputs.
---

# Tabular Prediction Skill

Use this skill when `./data` contains a training target, feature files, validation/test rows with hidden targets, and a required `submission.csv` format.

The hidden dataset schema is **unknown**. Infer everything from `DATA_DESCRIPTION.md`, CSV contents, and `sample_submission.csv`.

---

## 1. Purpose

Deliver a **valid, metric-aligned** tabular prediction submission for an unseen dataset:

- Infer schema and scoring metric
- Train CPU-friendly models with adaptive model selection
- Write `submission.csv` and `report.pdf` to the repository root
- Validate before finishing

---

## 2. Schema inference

**File role detection** (`scripts/analyze_schema.py` → `artifacts/schema_summary.json`):

- `sample_submission` — defines required prediction rows and placeholder target
- `train_target` — labeled training outcomes
- `train_features` — training covariates
- `validation_features` — covariates for rows to predict

**Column inference:**

- **Row id** — from `DATA_DESCRIPTION.md` and/or `sample_submission` (e.g. `row_id`)
- **Target** — from description and training files; resolve training vs submission column names via `resolve_training_target_column()`
- **Submission columns** — typically `[row_id, prediction_column]` exactly as specified

**Join key inference** (`src/join_inference.py`):

- Prefer composite keys that merge without row explosion
- Penalize keys that duplicate training rows
- Use the same inferred keys for train merge and prediction merge (may differ slightly between tables)

**Always read** `data/DATA_DESCRIPTION.md` before trusting heuristics.

---

## 3. Metric interpretation

**Metric-driven modeling is mandatory.**

| Detected metric | Modeling implication |
|-----------------|---------------------|
| `metric_mode: global` | Train/select one global model; CV uses global MAE |
| `metric_mode: block_averaged` | Compare global vs per-block strategies; CV uses block-averaged MAE |

**Scoring-category detection** (`src/schema_parser.py`):

Block-averaged mode is enabled when `DATA_DESCRIPTION.md` includes explicit block-averaged language **or**
scoring-category signals such as:

- `Scoring categories:`
- `scoring blocks` / `evaluated groups`
- `categories are nested` / `non-exclusive`
- `never sum across them`

Parsed category names become `metric_spec.block_values`. The block column is resolved by matching those
values to columns in `train_target` and `sample_submission` (not by hardcoding column names).

**Critical rules:**

- **Do not** train per-block models just because a low-cardinality column exists.
- **Do** run per-block modeling when scoring categories are named and the block column resolves.
- **Do** score global comparison models with block-averaged MAE when `metric_mode` is `block_averaged`.
- **Never sum** MAE across nested, non-exclusive categories unless the description explicitly requires it.
- Global model ≠ global metric: selection uses the detected official metric.
- If block column cannot be resolved → global fallback + documented warning.

Metric spec is stored in `schema_summary.json` / `modeling_log.json`. `cv_results.csv` includes per-block
rows, an `**overall**` summary row, and global comparison rows with `metric_mode`, `is_selected`, and `status`.

---

## 4. Modeling strategy

**Strategy selection** (`src/modeling.py` → `run_strategy_selection()`):

1. Cross-validate **global** candidate models using the detected metric.
2. If block-averaged metric is active → cross-validate **per-block** models (separate model per block).
3. Pick the strategy with the better CV score.

**Adaptive model pool** (`select_model_candidates()` / `select_candidate_models()`):

- Multiple hyperparameter candidates per sklearn family (ridge alphas, elastic-net grids, tree depths, etc.)
- Heavier models only on smaller datasets; candidate count capped by runtime
- CV time budget (~20 min default) — remaining models get `skipped_time_budget`
- **No KNN**
- XGBoost, LightGBM, and CatBoost are auto-detected if installed. They are prioritized in model selection when available, but the pipeline falls back to scikit-learn models if they are missing or fail.

**Fallback:**

- Training failure → median predictions (block medians when block-averaged mode applies)
- Small blocks (< 15 rows) → `median_fallback` with scored fallback MAE included in strategy comparison
- Non-finite predictions → replaced with fallback value

---

## 5. Feature engineering

Handled in `src/feature_engineering.py` → `prepare_feature_frames()`:

| Type | Treatment |
|------|-----------|
| Numeric | Median imputation in sklearn preprocessor |
| Low-cardinality categorical | One-hot encoding |
| Medium/high-cardinality categorical | Frequency encoding (train statistics only) |
| Very high-cardinality / ID-like | Dropped from modeling features (keys may be kept for image matching) |
| Text | Length/word stats; optional TF-IDF + SVD when enough text rows |
| Datetime | Year, month, day, day-of-week, missing indicator |
| Image sidecars | Lightweight pixel stats when filename matches **all** informative row key tokens |

**Image matching notes:**

- Uses composite key tokens — not single ambiguous tokens
- Skipped safely when match rate is low or images absent
- Raw key columns preserved for matching even if dropped from model features

Feature metadata is stored in `modeling_log.json` → `column_groups`.

---

## 6. Validation

Run `python scripts/validate_submission.py` after `submission.csv` is written.

Checks:

- File exists in repository root
- **Exact** required columns (correct names and order)
- Row count matches `sample_submission`
- Row-id coverage — no missing or extra ids; no duplicates
- Predictions are numeric and finite
- No unexpected extra columns

Failures must be fixed or clearly documented before stopping.

---

## 7. Reporting

Run `python scripts/make_report.py` → `report.pdf` in repo root.

The report includes:

1. Executive summary (status, strategy, metric, model)
2. Dataset understanding (files, join keys, schema warnings)
3. **Metric interpretation** (global vs block-averaged explanation)
4. Feature engineering summary (`column_groups`)
5. Model selection (global vs per-block scores, fallback usage)
6. Cross-validation results (`cv_results.csv`)
7. Submission validation
8. Output files and reproducibility
9. Limitations

Missing artifacts are noted in the report — report generation should not crash.

---

## Exploratory data analysis

After modeling artifacts exist, run `python scripts/run_eda.py` (or use `run_pipeline.py`, which calls it automatically).

- Writes `artifacts/eda/eda_summary.json`, CSV tables, and PNG figures
- Diagnostic only — EDA failures must not block submission or report generation
- Report section **3. Exploratory Data Analysis** summarizes EDA when artifacts are available

## Workflow checklist

```text
1. Read data/DATA_DESCRIPTION.md
2. python scripts/run_pipeline.py   # schema, train, EDA, validate, report
3. python scripts/validate_submission.py
4. Confirm submission.csv and report.pdf in repo root
5. Summarize: metric, strategy, validation, EDA status, warnings
```

## Restrictions

- No hardcoded STAI-X overdose column names
- No external datasets
- No GPU requirement
- No assumed row count
- No per-block modeling unless block-averaged metric is explicitly detected
