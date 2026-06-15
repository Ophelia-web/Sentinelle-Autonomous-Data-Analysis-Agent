# Sentinelle — an autonomous data-analysis agent for STAI-X Award B

**Sentinelle** is our entry for STAI-X Challenge 2026, Award B: an autonomous pipeline that reads an unfamiliar tabular dataset, trains a model, writes `submission.csv`, and leaves behind a PDF explaining what it did.

<p align="center">
  <img src="assets/sentinelle_logo.png" alt="Sentinelle logo" width="300">
</p>

## Why “Sentinelle”

*Sentinelle* means “sentinel” or “watcher.” The name is deliberate.

When the competition organizers drop a hidden dataset into `./data`, this agent’s job is to **stand watch** over it: read the data description, check the schema, look for the scoring rule, build a safe model, validate the output, and document the run. It is meant to be careful, not flashy — a watcher that notices join keys, metric wording, and row coverage before it ever fits a model.

We would rather ship a valid, well-explained submission than a fragile “clever” one that breaks on edge cases.

## What this repo does

Given only files under `./data` (especially `./data/DATA_DESCRIPTION.md`), Sentinelle:

1. **Understands the dataset** — discovers train, validation, and sample-submission files; infers row id, target, and required submission columns.
2. **Parses the scoring metric** — global MAE by default; **block-averaged MAE** only when `DATA_DESCRIPTION.md` explicitly says scores are averaged over blocks, categories, groups, or classes.
3. **Trains models metric-first** — compares global vs per-block strategies only when the detected metric calls for it; uses an adaptive CPU-friendly model pool with median fallback.
4. **Engineers features** — numeric, categorical, text (with optional TF-IDF/SVD), datetime, and optional image sidecar stats when filenames can be confidently matched to row keys.
5. **Writes `submission.csv`** — in the repository root, with exactly the required columns, finite numeric predictions, and full row-id coverage.
6. **Writes `report.pdf`** — a structured summary of schema, metric interpretation, features, model selection, CV, validation, and limitations.
7. **Validates outputs** — checks column schema, row counts, row-id alignment, and prediction finiteness.

The pipeline is **general-purpose**. It does not assume overdose columns, fixed row counts, or any particular target name.

## Official Award B evaluation workflow

This is the workflow judges use:

1. Clone this repository.
2. Place the hidden dataset in `./data`.
3. Ensure `./data/DATA_DESCRIPTION.md` is present.
4. Open the repo in Claude Code.
5. Send exactly: **Do the data analysis.**

The repo should then autonomously produce, in the **repository root**:

- `submission.csv`
- `report.pdf`

Claude should not stop until both files exist and validation has been attempted. If something fundamental is missing (no target, no submission schema), the failure should be documented clearly — ideally in `report.pdf`.

## Repository structure

```text
.
├── data/                 # Empty at submit time; organizers populate during evaluation
├── scripts/              # Entry points: schema, train, validate, report, full pipeline
├── src/                  # Core Python modules
├── .claude/              # Agent and skill guidance for Claude Code
├── artifacts/            # Per-run diagnostics (gitignored)
├── CLAUDE.md             # Primary operational instructions for Claude Code
├── requirements.txt
├── submission.csv        # Required output (gitignored until evaluation run)
└── report.pdf            # Required output (gitignored until evaluation run)
```

**Main scripts**

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | End-to-end run (preferred) |
| `scripts/analyze_schema.py` | Infer schema, roles, join keys, metric |
| `scripts/train_predict.py` | Feature engineering, modeling, `submission.csv` |
| `scripts/validate_submission.py` | Validate `submission.csv` |
| `scripts/make_report.py` | Generate `report.pdf` |
| `scripts/create_mock_dataset.py` | Small generic dataset for local smoke tests |

## Local dry run

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/validate_submission.py
python scripts/make_report.py
```

For a fresh generic mock dataset under `./data`:

```bash
python scripts/create_mock_dataset.py
python scripts/run_pipeline.py
```

XGBoost, LightGBM, and CatBoost are auto-detected if installed. They are prioritized in model selection when available, but the pipeline falls back to scikit-learn models if they are missing or fail.

## Design principles

- **Metric-driven modeling** — optimize what `DATA_DESCRIPTION.md` describes, not what column names happen to look like.
- **No hardcoded target names, block names, or row counts** — everything is inferred per dataset.
- **Block-aware modeling only when the metric requires it** — a low-cardinality category column alone is not enough to trigger per-block training.
- **Never sum nested scoring categories** unless the description explicitly says to.
- **No external datasets** — only repo files and `./data`.
- **CPU-friendly methods** — scikit-learn first; third-party boosters auto-detected when installed.
- **Fallback when needed** — median (or block median) predictions if training fails; validity over bravado.
- **Transparent reporting** — `report.pdf` explains metric, strategy, features, CV, validation, and limitations.
- **Safe merges** — join-key inference penalizes row explosions; sample-submission row order is preserved.

## What should not be committed

Do not commit local evaluation outputs:

- `data/` (competition data)
- `artifacts/`
- `submission.csv`
- `report.pdf`
- image feature caches and other local test outputs

These are listed in `.gitignore`.

## Limitations (honest)

- Schema and file-role inference are heuristic — unusual layouts may need manual fixes.
- Metric parsing depends on wording in `DATA_DESCRIPTION.md`; ambiguous text may default to global MAE.
- Text and image sidecars are used only when matching is confident; otherwise they are skipped safely.
- Optional external boosting libraries are not required and may be skipped on runtime budget.
- Fallback predictions prioritize **valid, finite submissions** over accuracy.
- Cross-validation uses a time budget; very large datasets may not evaluate every heavy model.

## Pre-flight checklist

Before considering a run complete:

- [ ] `DATA_DESCRIPTION.md` was read.
- [ ] Metric mode was detected (global vs block-averaged).
- [ ] Submission schema was inferred.
- [ ] Join keys were checked for merge safety.
- [ ] CV used the detected official metric.
- [ ] `submission.csv` was written to the repo root.
- [ ] `report.pdf` was written to the repo root.
- [ ] `validate_submission.py` passed, or errors were documented.
- [ ] No external data were used.

## License and challenge context

Built for **STAI-X Challenge 2026, Award B**. The repository is submitted without competition data; organizers supply the hidden dataset at evaluation time.
