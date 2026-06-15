#!/usr/bin/env python3
"""Generate report.pdf summarizing Sentinelle pipeline artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import artifacts_dir, project_root
from src.report import create_pdf_report


def print_summary(report_log: Dict[str, Any], log_path: Path) -> None:
    """Print a concise report generation summary."""
    print("Sentinelle report generation")
    print(f"  status: {report_log.get('status')}")
    print(f"  output_path: {report_log.get('output_path')}")
    print(f"  pages_written: {report_log.get('pages_written')}")
    print(f"  artifacts_used: {report_log.get('artifacts_used')}")
    print(f"  eda_status: {report_log.get('eda_status')}")

    warnings = report_log.get("warnings") or []
    eda_warnings = report_log.get("eda_warnings") or []
    if eda_warnings:
        print("  eda_warnings:")
        for warning in eda_warnings[:5]:
            print(f"    - {warning}")
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    print(f"  wrote: {log_path}")


def main() -> int:
    """Generate report.pdf and write artifacts/report_log.json."""
    root = project_root()
    output_path = root / "report.pdf"
    report_log = create_pdf_report(root, output_path)

    log_path = artifacts_dir(root) / "report_log.json"
    log_path.write_text(json.dumps(report_log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print_summary(report_log, log_path)

    if output_path.exists():
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
