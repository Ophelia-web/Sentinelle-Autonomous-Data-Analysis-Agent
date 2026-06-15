#!/usr/bin/env python3
"""Run the full Sentinelle pipeline end to end."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_schema, run_eda, train_predict, validate_submission
from src.io_utils import artifacts_dir, project_root


def _step_timeout_sec() -> float:
    """Return per-step timeout for isolated subprocess steps."""
    raw = os.environ.get("SENTINELLE_STEP_TIMEOUT_SEC")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 90 * 60.0


def _step_callables() -> List[tuple[str, Callable[[], int]]]:
    """Return in-process pipeline steps.

    Running core tabular steps in process avoids occasional subprocess pipe/joblib
    hangs in constrained evaluation containers while keeping the component scripts
    available for manual debugging.
    """
    return [
        ("analyze_schema", lambda: int(analyze_schema.main([]))),
        ("train_predict", lambda: int(train_predict.main())),
        ("run_eda", lambda: int(run_eda.main())),
        ("validate_submission", lambda: int(validate_submission.main())),
    ]


def run_callable_step(name: str, func: Callable[[], int]) -> Dict[str, Any]:
    """Run one in-process pipeline step and capture status metadata."""
    print(f"\n=== Step: {name} ===")
    start = time.time()

    try:
        returncode = int(func())
        error = None
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        returncode = int(code)
        error = None if returncode == 0 else f"SystemExit({exc.code})"
    except Exception as exc:
        returncode = 1
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    elapsed = time.time() - start
    print(f"Return code: {returncode}")
    print(f"Elapsed seconds: {elapsed:.2f}")

    return {
        "name": name,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "error": error,
    }


def run_report_step(root: Path) -> Dict[str, Any]:
    """Run report generation in a subprocess to isolate matplotlib state."""
    name = "make_report"
    command = [sys.executable, "scripts/make_report.py"]

    print(f"\n=== Step: {name} ===")
    print("Command:", " ".join(command))
    start = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_step_timeout_sec(),
        )
        returncode = int(result.returncode)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        error = None
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        error = f"make_report timed out after {_step_timeout_sec():.0f} seconds"

    if stdout:
        print(str(stdout).rstrip())
    if stderr:
        print(str(stderr).rstrip(), file=sys.stderr)

    elapsed = time.time() - start
    print(f"Return code: {returncode}")
    print(f"Elapsed seconds: {elapsed:.2f}")

    return {
        "name": name,
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "error": error,
    }


def write_pipeline_log(root: Path, run_log: Dict[str, Any]) -> Path:
    log_path = artifacts_dir(root) / "pipeline_run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return log_path


def clean_previous_outputs(root: Path) -> None:
    """Remove generated outputs from previous runs without touching data or source code."""
    for path in [
        artifacts_dir(root),
        root / "submission.csv",
        root / "report.pdf",
    ]:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    artifacts_dir(root).mkdir(parents=True, exist_ok=True)


def evaluate_pipeline_result(steps: List[Dict[str, Any]], root: Path) -> Dict[str, Any]:
    """Determine overall pipeline status from step results and output files."""
    submission_path = root / "submission.csv"
    report_path = root / "report.pdf"

    submission_exists = submission_path.exists()
    report_exists = report_path.exists()
    validation_ok = any(
        step.get("name") == "validate_submission" and step.get("returncode") == 0
        for step in steps
    )

    # Award B requires both root artifacts. A valid run should also pass the
    # repository validator. If validation fails, still create report.pdf but
    # do not claim pipeline success.
    passed = submission_exists and report_exists and validation_ok

    return {
        "status": "passed" if passed else "failed",
        "steps": steps,
        "submission_exists": submission_exists,
        "report_exists": report_exists,
        "validation_ok": validation_ok,
    }


def main() -> int:
    """Execute all pipeline steps and write a run log."""
    root = project_root()
    clean_previous_outputs(root)
    steps: List[Dict[str, Any]] = []

    for name, func in _step_callables():
        step_result = run_callable_step(name, func)
        steps.append(step_result)

        # Keep going so report.pdf can document failures when possible.
        if name in {"train_predict", "run_eda", "validate_submission"} and step_result["returncode"] != 0:
            print(f"Warning: {name} failed; continuing pipeline.")

    pre_report_log = evaluate_pipeline_result(steps, root)
    pre_report_log["status"] = "pre_report"
    pre_report_log["report_generation_pending"] = True
    write_pipeline_log(root, pre_report_log)

    steps.append(run_report_step(root))

    run_log = evaluate_pipeline_result(steps, root)
    run_log["report_generation_pending"] = False
    log_path = write_pipeline_log(root, run_log)

    print("\n=== Pipeline summary ===")
    print(f"status: {run_log['status']}")
    print(f"submission_exists: {run_log['submission_exists']}")
    print(f"report_exists: {run_log['report_exists']}")
    print(f"validation_ok: {run_log['validation_ok']}")
    print(f"wrote: {log_path}")

    return 0 if run_log["status"] == "passed" else 1


if __name__ == "__main__":
    # Force termination after artifacts are written. Some sklearn/joblib backends
    # can leave worker threads alive in constrained containers, which would
    # otherwise make the official one-command run appear to hang.
    code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
