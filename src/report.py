"""PDF report generation for Sentinelle pipeline artifacts."""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.io_utils import artifacts_dir, build_schema_summary, project_root

# Letter size (inches) — fixed for every page.
PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0
MARGIN_LEFT = 0.75
MARGIN_RIGHT = 0.75
MARGIN_TOP = 0.55
MARGIN_BOTTOM = 0.70

CONTENT_LEFT = MARGIN_LEFT / PAGE_WIDTH
CONTENT_BOTTOM = MARGIN_BOTTOM / PAGE_HEIGHT
CONTENT_WIDTH = 1.0 - (MARGIN_LEFT + MARGIN_RIGHT) / PAGE_WIDTH
CONTENT_HEIGHT = 1.0 - (MARGIN_TOP + MARGIN_BOTTOM) / PAGE_HEIGHT

TABLE_FONT_SIZE = 7.0
TABLE_HEADER_FONT_SIZE = 7.2
TABLE_SMALL_FONT_SIZE = 6.3
BODY_FONT_SIZE = 9.0
SECTION_FONT_SIZE = 13.0
TITLE_FONT_SIZE = 16.0
LINE_HEIGHT = 0.028
TABLE_ROW_HEIGHT = 0.038

REPORT_COLUMN_LABELS = {
    "validation_mean": "val_mean",
    "standardized_mean_diff": "std_mean_diff",
    "train_missing_pct": "train_miss_pct",
    "validation_missing_pct": "val_miss_pct",
    "missing_pct_diff": "miss_pct_diff",
    "block_averaged_mae": "block_avg_mae",
    "is_third_party": "third_party",
    "model_family": "family",
    "metric_mode": "metric",
    "cv_folds": "folds",
}
REPORT_COLUMN_LABELS.update(
    {
        "abs_corr_with_target": "abs_corr",
        "std_mean_diff": "std_diff",
        "train_miss_pct": "train_miss",
        "val_miss_pct": "val_miss",
        "miss_pct_diff": "miss_diff",
        "relative_path": "file",
        "nunique": "unique",
    }
)

MODEL_SELECTION_LABELS = {
    "modeling_strategy": "Modeling strategy",
    "global_strategy_score": "Global CV score",
    "per_block_strategy_score": "Per-block CV score",
    "selected_model": "Selected model",
    "selected_model (final)": "Selected model",
    "final_prediction_source": "Final prediction source",
    "model_fit_failed": "Model fit failed",
    "fallback_reason": "Fallback reason",
    "selected_model_by_block": "Selected block models",
    "fallback_value": "Fallback value",
    "used_fallback": "Fallback used",
    "train rows": "Training rows",
    "prediction rows": "Prediction rows",
    "training MAE (fit)": "Training MAE",
}


def load_json_safely(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning an empty dict on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV file, returning None on failure."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return None


def wrap_long_text(text: str, width: int = 95) -> str:
    """Wrap text to a maximum line width."""
    text = str(text)
    if len(text) <= width:
        return text
    return "\n".join(
        textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    )


def format_list_compactly(
    items: Optional[Sequence[Any]],
    max_items: int = 12,
    max_item_len: int = 60,
) -> str:
    """Format a list for compact display in tables or text."""
    if not items:
        return "(none)"
    values = [str(item) for item in items]
    truncated = [v if len(v) <= max_item_len else v[: max_item_len - 3] + "..." for v in values]
    if len(truncated) > max_items:
        shown = truncated[:max_items]
        return ", ".join(shown) + f" ... (+{len(truncated) - max_items} more)"
    return ", ".join(truncated)


def _truncate(text: Any, max_len: int = 48) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_report_value(value: Any) -> str:
    """Format values for PDF display; floats always use 4 decimal places."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "n/a"
        return f"{float(value):.4f}"
    if isinstance(value, (list, tuple, dict)):
        if isinstance(value, dict):
            return _truncate(json.dumps(value, default=str), 80)
        return format_list_compactly(value)
    return _truncate(value, 80)

def _format_table_value(value: Any, max_len: int = 80) -> str:
    """Format a table cell value with stable decimals and safe text."""
    if value is None:
        return "n/a"

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))

    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{float(value):.4f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (list, tuple)):
        text = ", ".join(str(x) for x in value)
    elif isinstance(value, dict):
        text = json.dumps(value, default=str, ensure_ascii=False)
    else:
        text = str(value)

    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Do not aggressively truncate ordinary table text. Wrapping handles it.
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."

    return text


def _wrap_cell_text(value: Any, width: int = 18, max_len: int = 90) -> str:
    """Wrap table cell text with explicit newlines."""
    text = _format_table_value(value, max_len=max_len)
    if len(text) <= width:
        return text

    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    )


def _cell_line_count(text: Any) -> int:
    return max(1, str(text).count("\n") + 1)


def _row_height_from_cells(cells: Sequence[Any], base: float = TABLE_ROW_HEIGHT) -> float:
    max_lines = max(_cell_line_count(cell) for cell in cells) if cells else 1
    return base * max_lines * 1.18


def _display_path(path_value: Any, root: Path) -> str:
    """Display paths as repo-relative paths when possible."""
    if path_value is None:
        return "n/a"
    text = str(path_value)
    try:
        path = Path(text)
        if path.is_absolute():
            try:
                return str(path.relative_to(root)).replace("\\", "/")
            except Exception:
                return path.name
    except Exception:
        pass
    return text.replace("\\", "/")


def _format_per_block_models_full(modeling_log: Dict[str, Any]) -> List[str]:
    selected = modeling_log.get("selected_model_by_block") or {}
    if not isinstance(selected, dict) or not selected:
        return ["n/a"]

    items = []
    for block, model in sorted(selected.items(), key=lambda x: str(x[0])):
        items.append(f"{block}: {_display_model_name(model)}")
    return items


def _display_model_name(name: Any) -> str:
    text = str(name)
    mapping = {
        "extra_trees_depthnone_leaf2_n140": "ExtraTrees(d=None, leaf=2, n=140)",
        "extra_trees_depth6_leaf2_n80": "ExtraTrees(d=6, leaf=2, n=80)",
        "extra_trees_depth10_leaf2_n100": "ExtraTrees(d=10, leaf=2, n=100)",
        "random_forest_depth6_leaf2_n80": "RandomForest(d=6, leaf=2, n=80)",
        "random_forest_depth10_leaf2_n100": "RandomForest(d=10, leaf=2, n=100)",
        "random_forest_depthnone_leaf2_n140": "RandomForest(d=None, leaf=2, n=140)",
        "dummy_median": "Dummy median",
    }
    return mapping.get(text, text)


def _format_cv_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply readable model names for CV table display."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "model" in out.columns:
        out["model"] = out["model"].map(_display_model_name)
    return out


def _drop_all_na_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_cols = []
    for col in out.columns:
        vals = out[col].astype(str).str.lower().str.strip()
        if vals.isin({"n/a", "nan", "none", ""}).all():
            drop_cols.append(col)
    return out.drop(columns=drop_cols, errors="ignore")


def _format_feature_list_summary(values: Any, label: str = "features", max_items: int = 5) -> str:
    """Format long feature lists as count + examples without broken ellipses."""
    if values is None:
        return "(none)"

    if isinstance(values, str):
        if not values.strip() or values.strip().lower() in {"none", "(none)", "n/a"}:
            return "(none)"
        values = [v.strip() for v in values.split(",") if v.strip()]

    values = list(values)
    if not values:
        return "(none)"

    shown = [str(v) for v in values[:max_items]]
    more = len(values) - len(shown)
    label_text = _pluralize(label, len(values))

    if more > 0:
        return f"{len(values)} {label_text}; examples: " + ", ".join(shown) + f"; plus {more} more"
    return f"{len(values)} {label_text}: " + ", ".join(shown)


def _pluralize(label: str, n: int) -> str:
    if n == 1 and label.endswith("s"):
        return label[:-1]
    return label


def _friendly_booster_status(status: Any) -> str:
    text = str(status)
    if "ModuleNotFoundError" in text:
        return "not installed"
    if text.startswith("available"):
        return "available"
    if text.startswith("unavailable"):
        return text.replace("unavailable:", "unavailable:").strip()
    return text


def _scripts_run_items(pipeline_log: Dict[str, Any]) -> List[str]:
    items: List[str] = []
    for step in pipeline_log.get("steps") or []:
        name = step.get("name")
        rc = step.get("returncode")
        if name is not None:
            items.append(f"{name} (exit {rc})")
    if not any("make_report" in item for item in items):
        items.append("make_report (current step)")
    return items if items else ["make_report (current step)"]


def _format_scripts_run(pipeline_log: Dict[str, Any]) -> str:
    return format_list_compactly(_scripts_run_items(pipeline_log))


def _prepare_eda_overview_table(overview_df: pd.DataFrame) -> pd.DataFrame:
    """Select readable columns for EDA raw/modeling overview tables."""
    overview_cols = [
        c
        for c in [
            "role",
            "relative_path",
            "rows",
            "columns",
            "missing_pct",
            "duplicate_rows",
        ]
        if c in overview_df.columns
    ]
    if not overview_cols:
        return overview_df

    display = overview_df[overview_cols].copy()
    if "relative_path" in display.columns:
        display["relative_path"] = display["relative_path"].map(
            lambda x: Path(str(x)).name if pd.notna(x) and str(x).strip() else str(x)
        )
    return display


def _model_selection_table_rows(
    modeling_log: Dict[str, Any],
    fit_meta: Dict[str, Any],
) -> List[Tuple[str, Any]]:
    strategy = modeling_log.get("modeling_strategy", "n/a")
    selected_by_block = modeling_log.get("selected_model_by_block") or {}
    used_fallback = bool(modeling_log.get("used_fallback", False))

    raw_rows: List[Tuple[str, Any]] = [
        ("modeling_strategy", strategy),
        ("global_strategy_score", modeling_log.get("global_strategy_score")),
        ("per_block_strategy_score", modeling_log.get("per_block_strategy_score")),
        ("selected_model (final)", _display_model_name(modeling_log.get("selected_model"))),
        ("final_prediction_source", modeling_log.get("final_prediction_source")),
        ("model_fit_failed", modeling_log.get("model_fit_failed", False)),
        ("fallback_reason", modeling_log.get("fallback_reason", "")),
        (
            "selected_model_by_block",
            "per-block (see list below)" if selected_by_block else "n/a",
        ),
        ("fallback_value", fit_meta.get("fallback_value")),
        ("used_fallback", used_fallback),
        ("train rows", modeling_log.get("n_train_rows")),
        ("prediction rows", modeling_log.get("n_predictions")),
        ("training MAE (fit)", fit_meta.get("train_mae")),
    ]

    rows: List[Tuple[str, Any]] = []
    for key, value in raw_rows:
        if key == "fallback_reason" and not str(value or "").strip():
            continue
        if key == "fallback_value" and not used_fallback:
            continue
        if key == "training MAE (fit)" and (value is None or str(value) == "n/a"):
            continue
        label = MODEL_SELECTION_LABELS.get(key, key)
        rows.append((label, value))
    return rows


def _safe_value(value: Any) -> str:
    return format_report_value(value)


def add_page_header_footer(
    fig: Figure,
    page_number: int,
    document_title: str = "Sentinelle Report",
) -> None:
    """Draw consistent footer on a fixed-size page."""
    fig.text(
        0.5,
        MARGIN_BOTTOM / (2.0 * PAGE_HEIGHT),
        f"Page {page_number}",
        fontsize=8,
        ha="center",
        va="center",
        color="#555555",
    )


class PdfReportWriter:
    """Render a multi-page PDF with fixed layout and pagination."""

    def __init__(
        self,
        pdf: PdfPages,
        document_title: str,
        logo_path: Optional[Path] = None,
    ) -> None:
        self.pdf = pdf
        self.document_title = document_title
        self.logo_path = logo_path
        self.page_number = 0
        self.fig: Optional[Figure] = None
        self.ax = None
        self.y = 1.0
        self.current_section = ""

    def _draw_report_logo(self) -> None:
        """Draw a small text brand mark only; no image logo."""
        if self.fig is not None:
            self.fig.text(
                1.0 - MARGIN_RIGHT / PAGE_WIDTH,
                0.982,
                "Sentinelle",
                fontsize=8,
                color="#555555",
                ha="right",
                va="top",
            )

    def _start_page(self, section_title: str = "") -> None:
        if self.fig is not None:
            self._finish_page()
        self.page_number += 1
        self.current_section = section_title
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.ax = self.fig.add_axes(
            [CONTENT_LEFT, CONTENT_BOTTOM, CONTENT_WIDTH, CONTENT_HEIGHT]
        )
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        add_page_header_footer(
            self.fig,
            self.page_number,
            document_title=self.document_title,
        )
        self._draw_report_logo()
        self.y = 0.965

    def _finish_page(self) -> None:
        if self.fig is None:
            return
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def close(self) -> int:
        """Finalize the last page and return total pages written."""
        if self.fig is not None:
            self._finish_page()
        return self.page_number

    def _ensure_space(self, needed: float, section_title: str = "") -> None:
        if self.fig is None or self.y - needed < 0.02:
            self._start_page(section_title or self.current_section)

    def add_section_page(self, title: str, subtitle: str = "") -> None:
        """Start a new section with a prominent heading."""
        self._start_page(title)
        self.ax.text(
            0.0,
            self.y,
            title,
            fontsize=SECTION_FONT_SIZE,
            fontweight="bold",
            va="top",
            transform=self.ax.transAxes,
        )
        self.y -= 0.05
        if subtitle:
            for line in wrap_long_text(subtitle, width=100).split("\n"):
                self.ax.text(
                    0.0,
                    self.y,
                    line,
                    fontsize=BODY_FONT_SIZE,
                    va="top",
                    color="#444444",
                    transform=self.ax.transAxes,
                )
                self.y -= LINE_HEIGHT

    def add_paragraph(self, text: str, fontsize: float = BODY_FONT_SIZE) -> None:
        """Add wrapped paragraph text, continuing on new pages if needed."""
        for line in wrap_long_text(text, width=100).split("\n"):
            self._ensure_space(LINE_HEIGHT, self.current_section)
            self.ax.text(
                0.0,
                self.y,
                line,
                fontsize=fontsize,
                va="top",
                transform=self.ax.transAxes,
            )
            self.y -= LINE_HEIGHT

    def add_subsection_title(self, title: str, min_following_space: float = 0.18) -> None:
        """Add subsection title and avoid leaving it alone at page bottom."""
        self._ensure_space(0.04 + min_following_space, self.current_section)
        self.ax.text(
            0.0,
            self.y,
            title,
            fontsize=BODY_FONT_SIZE,
            fontweight="bold",
            va="top",
            transform=self.ax.transAxes,
        )
        self.y -= 0.035

    def add_bullet_list(self, items: Sequence[str], fontsize: float = BODY_FONT_SIZE) -> None:
        """Add a bullet list with wrapping."""
        for item in items:
            wrapped = wrap_long_text(item, width=96)
            first = True
            for subline in wrapped.split("\n"):
                self._ensure_space(LINE_HEIGHT, self.current_section)
                prefix = "• " if first else "  "
                self.ax.text(
                    0.0,
                    self.y,
                    prefix + subline,
                    fontsize=fontsize,
                    va="top",
                    transform=self.ax.transAxes,
                )
                self.y -= LINE_HEIGHT
                first = False
            self.y -= 0.004

    def add_image(
        self,
        image_path: Path,
        caption: str = "",
        max_height: float = 0.34,
        suppress_caption: bool = True,
    ) -> None:
        """Insert an image into the report while preserving aspect ratio."""
        if not image_path.exists():
            self.add_paragraph(f"Figure unavailable: {image_path.name}")
            return

        try:
            img = plt.imread(str(image_path))
            height_px, width_px = img.shape[:2]
            aspect = width_px / max(height_px, 1)

            max_width = 0.86
            width = min(max_width, max_height * aspect)
            height = min(max_height, width / max(aspect, 1e-6))

            self._ensure_space(height + 0.07, self.current_section)

            x0 = 0.5 - width / 2
            x1 = 0.5 + width / 2
            y1 = self.y
            y0 = self.y - height

            self.ax.imshow(
                img,
                extent=[x0, x1, y0, y1],
                aspect="auto",
                interpolation="lanczos",
                transform=self.ax.transAxes,
            )
            self.y = y0 - 0.02

            # Most EDA plots already contain their own matplotlib title.
            # Avoid duplicating the same text as a stray caption under the figure.
            if caption and not suppress_caption:
                self.add_paragraph(f"Figure: {caption}", fontsize=7)
        except Exception as exc:
            self.add_paragraph(f"Could not render figure {image_path.name}: {type(exc).__name__}: {exc}")

    def add_titled_image(
        self,
        title: str,
        image_path: Path,
        max_height: float = 0.34,
    ) -> None:
        """Add a subsection title and image as one layout unit."""
        if not image_path.exists():
            return

        try:
            img = plt.imread(str(image_path))
            height_px, width_px = img.shape[:2]
            aspect = width_px / max(height_px, 1)

            max_width = 0.86
            width = min(max_width, max_height * aspect)
            height = min(max_height, width / max(aspect, 1e-6))

            # Reserve title + image together.
            self._ensure_space(0.04 + height + 0.07, self.current_section)

            self.ax.text(
                0.0,
                self.y,
                title,
                fontsize=BODY_FONT_SIZE,
                fontweight="bold",
                va="top",
                transform=self.ax.transAxes,
            )
            self.y -= 0.035

            x0 = 0.5 - width / 2
            x1 = 0.5 + width / 2
            y1 = self.y
            y0 = self.y - height

            self.ax.imshow(
                img,
                extent=[x0, x1, y0, y1],
                aspect="auto",
                interpolation="lanczos",
                transform=self.ax.transAxes,
            )
            self.y = y0 - 0.025

        except Exception as exc:
            self.add_paragraph(
                f"Could not render figure {image_path.name}: {type(exc).__name__}: {exc}"
            )

    def add_titled_dataframe_table(
        self,
        title: str,
        df: pd.DataFrame,
        max_rows_per_page: int = 8,
        keep_together: bool = False,
        note: str = "",
    ) -> None:
        """Render a title and dataframe table as one unit."""
        if df is None or df.empty:
            return

        self.add_dataframe_table(
            df,
            title=title,
            max_rows_per_page=max_rows_per_page,
            keep_together=keep_together,
            note=note,
        )

    def add_key_value_table(
        self,
        rows: Sequence[Tuple[str, Any]],
        title: str = "",
        keep_together: bool = False,
    ) -> None:
        """Render a two-column key/value table with wrapping and safe pagination."""
        if not rows:
            return

        display_rows: List[List[str]] = []
        row_heights: List[float] = []

        for key, value in rows:
            key_text = _wrap_cell_text(key, width=22, max_len=80)
            value_text = _wrap_cell_text(value, width=58, max_len=280)
            display_rows.append([key_text, value_text])
            row_heights.append(_row_height_from_cells([key_text, value_text], base=TABLE_ROW_HEIGHT))

        header = ["Field", "Value"]
        header_height = _row_height_from_cells(header, base=TABLE_ROW_HEIGHT)

        total = len(display_rows)
        start = 0

        title_height = 0.04 if title else 0.0
        if keep_together and total <= 8:
            required_height = title_height + header_height + sum(row_heights) + 0.06
        else:
            first_rows_height = sum(row_heights[: min(2, len(row_heights))]) if row_heights else TABLE_ROW_HEIGHT
            required_height = title_height + header_height + first_rows_height + 0.06

        self._ensure_space(required_height, self.current_section)

        if title:
            self.ax.text(
                0.0,
                self.y,
                title,
                fontsize=BODY_FONT_SIZE,
                fontweight="bold",
                va="top",
                transform=self.ax.transAxes,
            )
            self.y -= 0.035

        while start < total:
            available_height = self.y - 0.055
            min_useful_height = header_height + min(row_heights[start], TABLE_ROW_HEIGHT) + 0.04
            if available_height < min_useful_height:
                self._start_page(self.current_section)
                available_height = self.y - 0.055

            chunk: List[List[str]] = []
            heights: List[float] = []
            used_height = header_height + 0.018
            idx = start

            while idx < total:
                next_height = row_heights[idx]
                if chunk and used_height + next_height > available_height:
                    break
                chunk.append(display_rows[idx])
                heights.append(next_height)
                used_height += next_height
                idx += 1

            if len(chunk) == 1 and idx < total:
                self._start_page(self.current_section)
                continue

            if not chunk:
                self._start_page(self.current_section)
                continue

            table_height = min(used_height + 0.01, self.y - 0.045)
            y_bottom = max(self.y - table_height, 0.045)

            table = self.ax.table(
                cellText=chunk,
                colLabels=header,
                colWidths=[0.32, 0.68],
                cellLoc="left",
                loc="upper left",
                bbox=[0.0, y_bottom, 1.0, table_height],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(TABLE_FONT_SIZE)

            for col_idx in range(2):
                cell = table[(0, col_idx)]
                cell.set_height(header_height)
                cell.set_text_props(fontweight="bold", va="center", fontsize=TABLE_HEADER_FONT_SIZE)
                cell.set_facecolor("#EEF2F7")
                cell.set_edgecolor("#CCCCCC")
                cell.get_text().set_wrap(True)

            for local_row_idx, row_height in enumerate(heights, start=1):
                for col_idx in range(2):
                    cell = table[(local_row_idx, col_idx)]
                    cell.set_height(row_height)
                    cell.set_edgecolor("#CCCCCC")
                    cell.get_text().set_wrap(True)
                    cell.get_text().set_va("center")

            self.y = y_bottom - 0.025
            start = idx

            if start < total:
                self._start_page(self.current_section)

    def add_dataframe_table(
        self,
        df: pd.DataFrame,
        title: str = "",
        max_rows_per_page: int = 8,
        note: str = "",
        keep_together: bool = False,
    ) -> None:
        """Render a dataframe as a paginated wrapped table.

        This version intentionally avoids row-range notes and avoids one-row
        orphan table chunks. Wide tables are split across pages with repeated
        headers. Cell text is wrapped and row heights grow with line count.
        """
        if df is None or df.empty:
            self.add_paragraph("No table data available.")
            return

        display_df = df.copy()
        display_df = display_df.rename(
            columns={c: REPORT_COLUMN_LABELS.get(str(c), str(c)) for c in display_df.columns}
        )
        n_cols = len(display_df.columns)

        if n_cols >= 8:
            wrap_width = 10
            header_width = 10
            max_len = 70
            font_size = TABLE_SMALL_FONT_SIZE
            header_font_size = TABLE_SMALL_FONT_SIZE
            max_rows_per_page = min(max_rows_per_page, 6)
        elif n_cols >= 6:
            wrap_width = 12
            header_width = 11
            max_len = 80
            font_size = TABLE_SMALL_FONT_SIZE
            header_font_size = TABLE_SMALL_FONT_SIZE
            max_rows_per_page = min(max_rows_per_page, 7)
        else:
            wrap_width = 18
            header_width = 16
            max_len = 120
            font_size = TABLE_FONT_SIZE
            header_font_size = TABLE_HEADER_FONT_SIZE

        wrapped_headers = [
            _wrap_cell_text(str(c), width=header_width, max_len=60)
            for c in display_df.columns
        ]

        wrapped_rows: List[List[str]] = []
        row_heights: List[float] = []

        for _, row in display_df.iterrows():
            cells = [
                _wrap_cell_text(row[col], width=wrap_width, max_len=max_len)
                for col in display_df.columns
            ]
            wrapped_rows.append(cells)
            row_heights.append(_row_height_from_cells(cells, base=TABLE_ROW_HEIGHT))

        header_height = _row_height_from_cells(wrapped_headers, base=TABLE_ROW_HEIGHT)

        title_height = 0.04 if title else 0.0

        if keep_together and len(wrapped_rows) <= 8:
            total_table_height = title_height + header_height + sum(row_heights) + 0.08
            self._ensure_space(total_table_height, self.current_section)
            max_rows_per_page = max(max_rows_per_page, len(wrapped_rows))
        else:
            first_rows_height = sum(row_heights[: min(2, len(row_heights))]) if row_heights else TABLE_ROW_HEIGHT
            initial_need = title_height + header_height + first_rows_height + 0.06
            self._ensure_space(initial_need, self.current_section)

        if title:
            self.ax.text(
                0.0,
                self.y,
                title,
                fontsize=BODY_FONT_SIZE,
                fontweight="bold",
                va="top",
                transform=self.ax.transAxes,
            )
            self.y -= 0.035

        preferred_widths = {
            "strategy": 0.11,
            "selected": 0.08,
            "block": 0.14,
            "model": 0.15,
            "folds": 0.08,
            "cv_folds": 0.08,
            "score_mean": 0.11,
            "score_std": 0.11,
            "block_mae": 0.11,
            "block_averaged_mae": 0.14,
            "block_avg_mae": 0.14,
            "metric": 0.12,
            "metric_mode": 0.12,
            "family": 0.10,
            "model_family": 0.10,
            "third_party": 0.08,
            "is_third_party": 0.08,
            "status": 0.10,
            "notes": 0.18,
            "column": 0.20,
            "relative_path": 0.22,
            "file": 0.18,
            "artifact": 0.95,
            "val_mean": 0.11,
            "std_mean_diff": 0.12,
            "std_diff": 0.12,
            "train_miss_pct": 0.11,
            "train_miss": 0.11,
            "val_miss_pct": 0.11,
            "val_miss": 0.11,
            "miss_pct_diff": 0.11,
            "miss_diff": 0.11,
            "abs_corr": 0.12,
            "unique": 0.10,
        }
        raw_widths = [
            preferred_widths.get(str(col), 1.0 / max(n_cols, 1))
            for col in display_df.columns
        ]
        total_width = sum(raw_widths)
        col_widths = [w / total_width for w in raw_widths]

        total = len(wrapped_rows)
        start = 0

        while start < total:
            # If remaining vertical space is too small for a useful table chunk,
            # start a new page before drawing anything.
            available_height = self.y - 0.055
            min_useful_height = header_height + min(row_heights[start], TABLE_ROW_HEIGHT) + 0.04
            if available_height < min_useful_height:
                self._start_page(self.current_section)
                available_height = self.y - 0.055

            rows_this_page: List[List[str]] = []
            heights_this_page: List[float] = []
            used_height = header_height + 0.018
            idx = start

            while idx < total and len(rows_this_page) < max_rows_per_page:
                next_height = row_heights[idx]
                if rows_this_page and used_height + next_height > available_height:
                    break
                rows_this_page.append(wrapped_rows[idx])
                heights_this_page.append(next_height)
                used_height += next_height
                idx += 1

            # Avoid leaving a single orphan row on the next page.
            remaining_after_chunk = total - idx
            if remaining_after_chunk == 1 and len(rows_this_page) >= 3:
                rows_this_page.pop()
                moved_height = heights_this_page.pop()
                idx -= 1
                used_height -= moved_height

            # Avoid avoidable one-row chunks when more rows remain. Move the row to
            # a fresh page so it can be displayed with following rows.
            if len(rows_this_page) == 1 and idx < total:
                self._start_page(self.current_section)
                continue

            # If this is the last chunk and it has only one row, it is acceptable only
            # when the whole table genuinely could not be balanced. Otherwise previous
            # chunk logic should have moved one row forward.

            if not rows_this_page:
                self._start_page(self.current_section)
                continue

            table_height = min(used_height + 0.01, self.y - 0.045)
            if table_height <= 0.08:
                self._start_page(self.current_section)
                continue

            y_bottom = max(self.y - table_height, 0.045)

            table = self.ax.table(
                cellText=rows_this_page,
                colLabels=wrapped_headers,
                colWidths=col_widths,
                cellLoc="left",
                loc="upper left",
                bbox=[0.0, y_bottom, 1.0, table_height],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)

            for col_idx in range(n_cols):
                cell = table[(0, col_idx)]
                cell.set_height(header_height)
                cell.set_text_props(fontweight="bold", va="center", fontsize=header_font_size)
                cell.set_facecolor("#EEF2F7")
                cell.set_edgecolor("#CCCCCC")
                cell.get_text().set_wrap(True)

            for local_row_idx, row_height in enumerate(heights_this_page, start=1):
                for col_idx in range(n_cols):
                    cell = table[(local_row_idx, col_idx)]
                    cell.set_height(row_height)
                    cell.set_edgecolor("#CCCCCC")
                    cell.get_text().set_wrap(True)
                    cell.get_text().set_va("center")

            self.y = y_bottom - 0.025
            start = idx

            if start < total:
                self._start_page(self.current_section)

        if note:
            self.add_paragraph(note, fontsize=8)


def _role_path(schema: Dict[str, Any], role: str) -> str:
    profile = (schema.get("inferred_roles") or {}).get(role)
    if not profile:
        return "not identified"
    return str(profile.get("rel_path", "unknown"))


def _metric_explanation(metric_spec: Dict[str, Any]) -> str:
    mode = metric_spec.get("metric_mode", "global")
    name = metric_spec.get("metric_name", "mae")
    if mode == "block_averaged":
        block_col = metric_spec.get("block_column") or "the detected block column"
        return (
            f"Evaluation uses block-averaged {name.upper()}: the score is the unweighted mean of "
            f"per-block {name.upper()} values computed within each category of '{block_col}'. "
            "Global and per-block modeling strategies are compared using this metric."
        )
    return (
        f"Evaluation uses global {name.upper()}: the score is the mean absolute error computed "
        "across all rows (or the detected global metric from DATA_DESCRIPTION.md)."
    )


def _prepare_cv_display(df: pd.DataFrame, max_rows: int = 36) -> Tuple[pd.DataFrame, str]:
    """Sort CV results and trim to a readable subset when large."""
    if df.empty:
        return df, ""

    sort_cols = [c for c in ["strategy", "block", "score_mean"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    if len(df) <= max_rows:
        return df, ""

    parts: List[pd.DataFrame] = []
    if "strategy" in df.columns:
        for strategy in df["strategy"].dropna().unique():
            sub = df[df["strategy"] == strategy]
            if "block" in sub.columns:
                global_rows = sub[sub["block"].isna() | (sub["block"].astype(str) == "")]
                if not global_rows.empty:
                    parts.append(global_rows.head(12))
                block_rows = sub[sub["block"].notna() & (sub["block"].astype(str) != "")]
                if not block_rows.empty and "score_mean" in block_rows.columns:
                    ok = block_rows[block_rows["status"] == "ok"] if "status" in block_rows.columns else block_rows
                    if not ok.empty:
                        best_idx = ok.groupby("block")["score_mean"].idxmin()
                        parts.append(block_rows.loc[best_idx])
                    fallback = block_rows[block_rows["status"].astype(str).str.contains("fallback", na=False)] if "status" in block_rows.columns else pd.DataFrame()
                    if not fallback.empty:
                        parts.append(fallback.drop_duplicates(subset=["block", "model"]))
            else:
                parts.append(sub.head(12))
    else:
        parts.append(df.head(max_rows))

    if parts:
        trimmed = pd.concat(parts, ignore_index=True).drop_duplicates()
        if len(trimmed) > max_rows:
            trimmed = trimmed.head(max_rows)
        note = f"Showing {len(trimmed)} of {len(df)} CV rows (top global models and best/fallback per block)."
        return trimmed, note

    return df.head(max_rows), f"Showing first {max_rows} of {len(df)} CV rows."


def _submission_checks(root: Path, validation_report: Dict[str, Any]) -> Dict[str, Any]:
    """Derive submission quality checks for the report."""
    checks: Dict[str, Any] = {
        "row_id_coverage_ok": None,
        "predictions_finite": None,
        "n_finite_predictions": None,
        "n_non_finite_predictions": None,
    }
    submission_path = root / "submission.csv"
    if not submission_path.exists():
        return checks

    errors = validation_report.get("errors") or []
    checks["row_id_coverage_ok"] = not any("row_id" in str(e).lower() for e in errors)
    checks["predictions_finite"] = not any(
        kw in " ".join(errors).lower() for kw in ("nan", "infinite", "not numeric")
    )

    try:
        sub = pd.read_csv(submission_path)
        target_col = validation_report.get("target_column")
        if target_col and target_col in sub.columns:
            vals = pd.to_numeric(sub[target_col], errors="coerce")
            checks["n_finite_predictions"] = int(np.isfinite(vals.to_numpy(dtype=float)).sum())
            checks["n_non_finite_predictions"] = int((~np.isfinite(vals.to_numpy(dtype=float))).sum())
            if checks["predictions_finite"] is None:
                checks["predictions_finite"] = checks["n_non_finite_predictions"] == 0
    except Exception:
        pass

    return checks


def build_report_context(root: Path) -> Dict[str, Any]:
    """Collect schema and artifact information for report generation."""
    root = root.resolve()
    adir = artifacts_dir(root)
    warnings: List[str] = []
    artifacts_used: List[str] = []

    schema_path = adir / "schema_summary.json"
    if schema_path.exists():
        schema = load_json_safely(schema_path)
        artifacts_used.append(str(schema_path))
    else:
        warnings.append("schema_summary.json missing; built schema summary on demand")
        schema = build_schema_summary(root=root)

    modeling_log_path = adir / "modeling_log.json"
    modeling_log = load_json_safely(modeling_log_path)
    if modeling_log:
        artifacts_used.append(str(modeling_log_path))

    cv_path = adir / "cv_results.csv"
    cv_results = load_csv_safely(cv_path)
    if cv_results is not None:
        artifacts_used.append(str(cv_path))

    validation_path = adir / "validation_report.json"
    validation_report = load_json_safely(validation_path)
    if validation_report:
        artifacts_used.append(str(validation_path))

    pipeline_log_path = adir / "pipeline_run_log.json"
    pipeline_log = load_json_safely(pipeline_log_path)
    if pipeline_log:
        artifacts_used.append(str(pipeline_log_path))

    eda_summary_path = adir / "eda" / "eda_summary.json"
    eda_summary = load_json_safely(eda_summary_path)
    eda_artifacts_used: List[str] = []
    if eda_summary:
        artifacts_used.append(str(eda_summary_path))
        eda_artifacts_used.append(str(eda_summary_path))
        for rel in (eda_summary.get("tables") or []) + (eda_summary.get("figures") or []):
            if rel and rel not in eda_artifacts_used:
                eda_artifacts_used.append(str(rel))

    submission_path = root / "submission.csv"
    report_path = root / "report.pdf"

    metric_spec = modeling_log.get("metric_spec") or schema.get("metric_spec") or {}
    submission_checks = _submission_checks(root, validation_report)

    status_parts = [
        f"schema: {schema.get('status', 'unknown')}",
        f"modeling: {modeling_log.get('status', 'unavailable') if modeling_log else 'unavailable'}",
        f"validation: {validation_report.get('status', 'unavailable') if validation_report else 'unavailable'}",
    ]

    return {
        "root": str(root),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "schema": schema,
        "modeling_log": modeling_log,
        "cv_results": cv_results,
        "validation_report": validation_report,
        "pipeline_log": pipeline_log,
        "submission_exists": submission_path.exists(),
        "submission_path": str(submission_path),
        "report_path": str(report_path),
        "status_summary": "; ".join(status_parts),
        "artifacts_used": artifacts_used,
        "warnings": warnings,
        "metric_spec": metric_spec,
        "submission_checks": submission_checks,
        "validation_passed": validation_report.get("status") == "passed",
        "eda_summary": eda_summary,
        "eda_dir": str(adir / "eda"),
        "eda_artifacts_used": eda_artifacts_used,
        "eda_status": eda_summary.get("status") if eda_summary else None,
        "eda_warnings": eda_summary.get("warnings") if eda_summary else [],
    }



def _render_cover_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    modeling_log = context.get("modeling_log") or {}
    metric_spec = context.get("metric_spec") or {}
    modeling_status = modeling_log.get("status", "unavailable")
    strategy = modeling_log.get("modeling_strategy", "unavailable")
    selected_model = modeling_log.get("selected_model", "n/a")
    model_fit_failed = bool(modeling_log.get("model_fit_failed", False))

    if model_fit_failed or strategy == "fallback":
        selected_display = "median_fallback"
    else:
        selected_display = _display_model_name(selected_model) if strategy != "per_block" else "per-block (see summary)"

    writer._start_page("1. Executive Summary")
    writer.ax.text(
        0.0,
        writer.y,
        "Sentinelle Automated Data Analysis Report",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        va="top",
        transform=writer.ax.transAxes,
    )
    writer.y -= 0.07
    writer.add_paragraph(f"Generated (UTC): {context.get('timestamp')}")
    writer.y -= 0.01

    writer.add_key_value_table(
        [
            ("Pipeline status", context.get("status_summary")),
            ("Modeling status", modeling_status),
            ("submission.csv exists", context.get("submission_exists")),
            ("Validation passed", context.get("validation_passed")),
            ("Modeling strategy", strategy),
            ("Metric mode", metric_spec.get("metric_mode", "n/a")),
            ("Metric name", metric_spec.get("metric_name", "n/a")),
            ("Selected model", selected_display),
        ],
        title="1. Executive Summary",
    )

    if strategy == "per_block" and not model_fit_failed and modeling_log.get("selected_model_by_block"):
        writer.add_subsection_title("Selected per-block models", min_following_space=0.22)
        writer.add_bullet_list(_format_per_block_models_full(modeling_log), fontsize=8)


def _render_dataset_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    schema = context.get("schema") or {}
    modeling_log = context.get("modeling_log") or {}
    join_keys = modeling_log.get("join_keys") or schema.get("join_keys") or {}
    root = Path(context.get("root") or project_root())

    writer.add_section_page(
        "2. Dataset Understanding",
        "Inferred schema, file roles, and merge keys detected from DATA_DESCRIPTION.md and CSV inspection.",
    )

    writer.add_key_value_table(
        [
            (
                "DATA_DESCRIPTION path",
                _display_path(schema.get("data_description_path"), root),
            ),
            ("DATA_DESCRIPTION present", schema.get("data_description_present", False)),
            ("Number of CSV files", schema.get("n_csv_files", "n/a")),
            ("Row id column", schema.get("row_id_column", "n/a")),
            ("Target column", schema.get("target_column", "n/a")),
            ("Submission columns", format_list_compactly(schema.get("submission_columns"))),
            ("Train merge keys", format_list_compactly(join_keys.get("train_keys"))),
            ("Train key source", join_keys.get("train_source", "n/a")),
            ("Prediction merge keys", format_list_compactly(join_keys.get("predict_keys"))),
            ("Prediction key source", join_keys.get("predict_source", "n/a")),
        ]
    )

    writer.add_paragraph("Inferred file roles:")
    writer.add_bullet_list(
        [
            f"sample_submission: {_role_path(schema, 'sample_submission')}",
            f"train_target: {_role_path(schema, 'train_target')}",
            f"train_features: {_role_path(schema, 'train_features')}",
            f"validation_features: {_role_path(schema, 'validation_features')}",
        ]
    )

    files = schema.get("files") or []
    if files:
        profile_rows = []
        for profile in files:
            profile_rows.append(
                {
                    "File": profile.get("rel_path", "unknown"),
                    "Role": profile.get("role", "unknown"),
                    "Rows": profile.get("n_rows", "n/a"),
                    "Cols": profile.get("n_cols", "n/a"),
                }
            )
        writer.add_subsection_title("File profiles", min_following_space=0.18)
        writer.add_dataframe_table(pd.DataFrame(profile_rows))
    else:
        writer.add_paragraph("File profiles unavailable.")

    schema_warnings = schema.get("warnings") or []
    if schema_warnings:
        writer.add_paragraph("Schema warnings:")
        writer.add_bullet_list([str(w) for w in schema_warnings])
    else:
        writer.add_paragraph("Schema warnings: none recorded.")


def _load_eda_table(root: Path, filename: str) -> Optional[pd.DataFrame]:
    path = artifacts_dir(root) / "eda" / filename
    return load_csv_safely(path)


def _render_eda_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    root = Path(context.get("root") or project_root())
    eda_summary = context.get("eda_summary") or {}
    eda_dir = Path(context.get("eda_dir") or artifacts_dir(root) / "eda")

    writer.add_section_page(
        "3. Exploratory Data Analysis",
        "Automated summaries of file structure, missingness, target behavior, and covariate shift.",
    )

    if not eda_summary:
        writer.add_paragraph("EDA artifacts were unavailable; this section was skipped.")
        return

    flags = eda_summary.get("flags") or {}
    writer.add_paragraph(
        f"EDA status: {eda_summary.get('status', 'unknown')}. "
        f"Files analyzed: {flags.get('n_files_analyzed', 0)}. "
        f"Target EDA: {'yes' if flags.get('target_eda_available') else 'no'}. "
        f"Block/category target summary: {'yes' if flags.get('block_target_eda_available') else 'no'}."
    )

    eda_warnings = eda_summary.get("warnings") or []
    if eda_warnings:
        writer.add_paragraph("EDA warnings:")
        writer.add_bullet_list([str(w) for w in eda_warnings[:8]])

    writer.add_paragraph(
        "Raw file counts and modeling-frame counts may differ when feature rows are "
        "deduplicated, joined, or expanded by target categories."
    )

    raw_overview = _load_eda_table(root, "raw_file_overview.csv")
    if raw_overview is None or raw_overview.empty:
        raw_overview = _load_eda_table(root, "dataset_overview.csv")
    if raw_overview is not None and not raw_overview.empty:
        raw_display = _drop_all_na_display_columns(
            _prepare_eda_overview_table(raw_overview.head(12))
        )
        writer.add_titled_dataframe_table(
            "Raw file overview",
            raw_display,
            max_rows_per_page=12,
            keep_together=len(raw_display) <= 6,
        )

    modeling_overview = _load_eda_table(root, "modeling_frame_overview.csv")
    if modeling_overview is None or modeling_overview.empty:
        modeling_records = eda_summary.get("modeling_frame_overview") or []
        if modeling_records:
            modeling_overview = pd.DataFrame(modeling_records)
    if modeling_overview is not None and not modeling_overview.empty:
        modeling_display = _drop_all_na_display_columns(
            _prepare_eda_overview_table(modeling_overview)
        )
        writer.add_titled_dataframe_table(
            "Modeling frame overview",
            modeling_display,
            keep_together=len(modeling_display) <= 6,
        )

    missing = _load_eda_table(root, "missingness_top.csv")
    miss_fig = eda_dir / "missingness_top.png"
    if miss_fig.exists():
        writer.add_titled_image("Top columns by missingness", miss_fig, max_height=0.30)
    if missing is not None and not missing.empty:
        writer.add_titled_dataframe_table(
            "Top missingness values",
            missing.head(8),
            max_rows_per_page=8,
            keep_together=True,
        )

    target_fig = eda_dir / "target_distribution.png"
    if target_fig.exists():
        writer.add_titled_image("Target distribution", target_fig, max_height=0.30)

    target_log_fig = eda_dir / "target_distribution_log1p.png"
    if target_log_fig.exists():
        writer.add_titled_image("log1p target distribution", target_log_fig, max_height=0.30)

    block_fig = eda_dir / "target_by_block_mean.png"
    if block_fig.exists():
        writer.add_titled_image("Mean target by block/category", block_fig, max_height=0.30)
    by_block = _load_eda_table(root, "target_by_block.csv")
    if by_block is not None and not by_block.empty:
        writer.add_titled_dataframe_table(
            "Target summary by block/category",
            by_block.head(8),
            max_rows_per_page=8,
            keep_together=True,
        )

    corr_fig = eda_dir / "top_numeric_correlations.png"
    if corr_fig.exists():
        writer.add_titled_image("Top numeric feature correlations", corr_fig, max_height=0.30)
    top_corr = _load_eda_table(root, "top_numeric_correlations.csv")
    if top_corr is not None and not top_corr.empty:
        writer.add_titled_dataframe_table(
            "Top numeric correlation values",
            top_corr.head(6),
            max_rows_per_page=6,
            keep_together=True,
        )

    shift_fig = eda_dir / "train_validation_shift_top.png"
    if shift_fig.exists():
        writer.add_titled_image("Top numeric covariate shift", shift_fig, max_height=0.30)
    shift = _load_eda_table(root, "train_validation_shift_numeric.csv")
    if shift is not None and not shift.empty:
        writer.add_titled_dataframe_table(
            "Top train-validation numeric shift values",
            shift.head(6),
            max_rows_per_page=6,
            keep_together=True,
        )

    availability = eda_summary.get("text_image_availability") or {}
    model_groups = (context.get("modeling_log") or {}).get("column_groups") or {}

    writer.add_key_value_table(
        [
            (
                "Text columns detected",
                _format_feature_list_summary(
                    availability.get("text_columns_detected") or [],
                    label="columns",
                ),
            ),
            (
                "Simple text features",
                _format_feature_list_summary(
                    availability.get("simple_text_features") or [],
                    label="features",
                ),
            ),
            (
                "Text SVD features",
                _format_feature_list_summary(
                    availability.get("text_svd_features") or [],
                    label="features",
                ),
            ),
        ],
        title="Text feature availability",
        keep_together=True,
    )

    writer.add_key_value_table(
        [
            ("Image features used", bool(availability.get("image_features_used", False))),
            ("Images found", availability.get("n_images_found", 0)),
            ("Image rows matched", availability.get("n_rows_matched", 0)),
            ("Image match rate", availability.get("image_match_rate", 0.0)),
            (
                "Image feature columns",
                _format_feature_list_summary(
                    availability.get("image_feature_columns") or [],
                    label="features",
                ),
            ),
            ("Availability source", availability.get("availability_source", "n/a")),
        ],
        title="Image feature availability",
        keep_together=True,
    )

    if availability.get("availability_source") == "unavailable_no_column_groups":
        writer.add_paragraph(
            "Text/image feature availability could not be summarized because modeling_log.json did not contain column_groups metadata. This usually means feature engineering did not complete or final modeling fell back before metadata was preserved."
        )

    if model_groups.get("image_features_used") and not availability.get("image_features_used"):
        writer.add_paragraph(
            "Warning: modeling artifacts indicate image features were used, but EDA image availability metadata did not record them."
        )

    if (model_groups.get("simple_text_features") or model_groups.get("text_svd_features")) and not (
        availability.get("simple_text_features") or availability.get("text_svd_features")
    ):
        writer.add_paragraph(
            "Warning: modeling artifacts indicate text-derived features were used, but EDA text availability metadata did not record them."
        )


def _render_analytical_summary_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    """Render a deterministic narrative analytical summary.

    This is Claude-style narrative text generated from artifacts, without
    calling any external LLM/API.
    """
    schema = context.get("schema") or {}
    modeling_log = context.get("modeling_log") or {}
    metric_spec = context.get("metric_spec") or {}
    validation_report = context.get("validation_report") or {}
    eda_summary = context.get("eda_summary") or {}
    groups = modeling_log.get("column_groups") or {}

    writer.add_section_page(
        "9. Analytical Summary",
        "Narrative synthesis generated deterministically from pipeline artifacts.",
    )

    target_col = schema.get("target_column") or modeling_log.get("target_column") or "the inferred target"
    row_id_col = schema.get("row_id_column") or modeling_log.get("row_id_column") or "the inferred row id"
    submission_cols = format_list_compactly(schema.get("submission_columns") or modeling_log.get("submission_columns"))
    n_train = modeling_log.get("n_train_rows", "n/a")
    n_pred = modeling_log.get("n_predictions", "n/a")

    writer.add_paragraph(
        f"The pipeline inferred a supervised tabular prediction task with target column "
        f"'{target_col}' and row identifier '{row_id_col}'. The required submission schema was "
        f"{submission_cols}. The modeling table contained {n_train} training rows, and the final "
        f"submission contains {n_pred} prediction rows."
    )

    metric_mode = metric_spec.get("metric_mode", "global")
    block_col = metric_spec.get("block_column") or modeling_log.get("block_column")
    block_values = metric_spec.get("block_values") or modeling_log.get("block_values")

    if metric_mode == "block_averaged":
        writer.add_paragraph(
            f"The detected scoring mode was block-averaged MAE. The resolved block column was "
            f"'{block_col}', with scoring blocks {format_list_compactly(block_values)}. In this mode, "
            f"per-block rows are evaluated within each scoring block and the overall score is the "
            f"unweighted mean of those block-level MAEs. Global models may still be compared, but they "
            f"must be scored with the same block-averaged metric."
        )
    else:
        writer.add_paragraph(
            "The detected scoring mode was global MAE. Under this interpretation, cross-validation "
            "and model selection used the mean absolute error across all evaluated rows, and per-block "
            "modeling was not used as the primary strategy."
        )

    strategy = modeling_log.get("modeling_strategy", "n/a")
    global_score = modeling_log.get("global_strategy_score")
    per_block_score = modeling_log.get("per_block_strategy_score")
    selected_model = modeling_log.get("selected_model")
    selected_by_block = modeling_log.get("selected_model_by_block") or {}

    if selected_by_block:
        block_model_text = "; ".join(_format_per_block_models_full(modeling_log))
        writer.add_paragraph(
            f"Model selection chose the '{strategy}' strategy. The global strategy score was "
            f"{format_report_value(global_score)}, and the per-block strategy score was "
            f"{format_report_value(per_block_score)}. The selected per-block models were: "
            f"{block_model_text}."
        )
    else:
        writer.add_paragraph(
            f"Model selection chose the '{strategy}' strategy with selected model "
            f"'{_display_model_name(selected_model)}'. "
            f"The global strategy score was {format_report_value(global_score)}"
            + (
                f", while the per-block strategy score was {format_report_value(per_block_score)}."
                if per_block_score is not None
                else "."
            )
        )

    feature_bits: List[str] = []
    if groups.get("numeric_columns"):
        feature_bits.append(f"{len(groups.get('numeric_columns') or [])} numeric columns")
    if groups.get("low_cardinality_categorical") or groups.get("frequency_encoded_categorical"):
        feature_bits.append("categorical encodings")
    if groups.get("simple_text_features") or groups.get("text_svd_features"):
        feature_bits.append("text-derived features")
    if groups.get("image_features_used"):
        feature_bits.append(
            f"image sidecar features with match rate {format_report_value(groups.get('image_match_rate'))}"
        )
    if not feature_bits:
        feature_bits.append("the available tabular features")

    writer.add_paragraph(
        "Feature engineering used " + ", ".join(feature_bits) + ". "
        "Higher-cardinality categorical columns were handled conservatively, and optional text/image "
        "signals were used only when they could be processed safely."
    )

    validation_status = validation_report.get("status", "unavailable")
    n_sub = validation_report.get("n_submission_rows", "n/a")
    n_sample = validation_report.get("n_sample_rows", "n/a")

    writer.add_paragraph(
        f"The final validation status was '{validation_status}'. The submission contained {n_sub} rows "
        f"against {n_sample} expected rows, with finite numeric predictions checked by the validator."
    )

    if modeling_log.get("model_fit_failed"):
        writer.add_paragraph(
            "Final model fitting failed after cross-validation, so the submitted predictions were generated using the median fallback strategy. The CV results remain useful for diagnostics but should not be described as the fitted final model."
        )
    elif modeling_log.get("used_fallback"):
        writer.add_paragraph(
            "Fallback handling was applied to some predictions or blocks, as recorded in modeling_log.json."
        )
    else:
        writer.add_paragraph(
            "Fallback predictions were not used in the final output."
        )

    eda_status = eda_summary.get("status")
    if eda_status:
        flags = eda_summary.get("flags") or {}
        writer.add_paragraph(
            f"EDA completed with status '{eda_status}'. Target EDA was "
            f"{'available' if flags.get('target_eda_available') else 'not available'}, "
            f"joined target-feature EDA was "
            f"{'available' if flags.get('joined_eda_available') else 'not available'}, and "
            f"train-validation shift analysis was "
            f"{'available' if flags.get('shift_analysis_available') else 'not available'}."
        )

    warnings = (modeling_log.get("warnings") or [])[:5]
    if warnings:
        writer.add_paragraph("Important cautions recorded during the run:")
        writer.add_bullet_list([str(w) for w in warnings])


def _render_metric_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    metric_spec = context.get("metric_spec") or {}
    modeling_log = context.get("modeling_log") or {}

    writer.add_section_page(
        "4. Metric Interpretation",
        "How predictions are scored according to the detected evaluation metric.",
    )

    writer.add_key_value_table(
        [
            ("metric_name", metric_spec.get("metric_name", "n/a")),
            ("metric_mode", metric_spec.get("metric_mode", "n/a")),
            ("metric_source", metric_spec.get("metric_source", "n/a")),
            ("higher_is_better", metric_spec.get("higher_is_better", False)),
            ("block_column", metric_spec.get("block_column") or modeling_log.get("block_column") or "none"),
            (
                "block_values",
                format_list_compactly(metric_spec.get("block_values") or modeling_log.get("block_values")),
            ),
        ],
        keep_together=True,
    )

    writer.add_paragraph("Interpretation:")
    writer.add_paragraph(_metric_explanation(metric_spec))

    if metric_spec.get("metric_mode") == "block_averaged":
        writer.add_paragraph(
            "When block-averaged MAE is active, the pipeline compares a single global model against "
            "per-block models. The strategy with the lower block-averaged score is selected."
        )
    else:
        writer.add_paragraph(
            "With a global metric, model selection uses cross-validated global scores. "
            "Per-block modeling is disabled unless block-averaged evaluation is detected."
        )


def _render_feature_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    modeling_log = context.get("modeling_log") or {}
    groups = modeling_log.get("column_groups") or {}

    writer.add_section_page(
        "5. Feature Engineering Summary",
        "Feature handling recorded during prepare_feature_frames() and stored in modeling_log.json.",
    )

    if not groups:
        writer.add_paragraph(
            "column_groups metadata unavailable. Run scripts/train_predict.py to populate feature engineering details."
        )
        return

    tabular_rows = [
        ("Numeric columns", _format_feature_list_summary(groups.get("numeric_columns"), label="columns")),
        ("Categorical columns", format_list_compactly(groups.get("categorical_columns"))),
        (
            "Low-cardinality categorical (one-hot)",
            format_list_compactly(groups.get("low_cardinality_categorical") or groups.get("engineered_categorical_columns")),
        ),
        (
            "Frequency-encoded categorical",
            format_list_compactly(groups.get("frequency_encoded_categorical")),
        ),
        (
            "Dropped high-cardinality categorical",
            format_list_compactly(groups.get("dropped_high_cardinality_categorical")),
        ),
        ("ID-like columns (excluded)", format_list_compactly(groups.get("id_like_columns"))),
        ("Datetime features created", format_list_compactly(groups.get("datetime_features_created"))),
    ]
    text_image_rows = [
        ("Simple text features", _format_feature_list_summary(groups.get("simple_text_features"), label="features")),
        ("Text SVD features", _format_feature_list_summary(groups.get("text_svd_features"), label="features")),
        ("Image features used", groups.get("image_features_used", False)),
        ("Images found", groups.get("n_images_found", 0)),
        ("Image rows matched", groups.get("n_rows_matched", 0)),
        (
            "Image match rate",
            format_report_value(groups.get("image_match_rate", 0.0))
            if groups.get("image_match_rate") is not None
            else "n/a",
        ),
        ("Image feature columns", _format_feature_list_summary(groups.get("image_feature_columns"), label="features")),
    ]
    writer.add_key_value_table(tabular_rows, title="Tabular feature handling", keep_together=True)
    writer.add_key_value_table(text_image_rows, title="Text and image feature handling", keep_together=True)

    warning_groups = [
        ("Feature warnings", groups.get("feature_warnings") or []),
        ("Text feature warnings", groups.get("text_feature_warnings") or []),
        ("Image feature warnings", groups.get("image_feature_warnings") or []),
        ("Image feature notes", groups.get("image_feature_notes") or []),
    ]
    for label, items in warning_groups:
        if items:
            writer.add_paragraph(f"{label}:")
            writer.add_bullet_list([str(i) for i in items])


def _render_model_selection_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    modeling_log = context.get("modeling_log") or {}
    fit_meta = modeling_log.get("fit_metadata") or {}

    writer.add_section_page(
        "6. Model Selection Summary",
        "Strategy comparison, selected models, and training/prediction row counts.",
    )

    if not modeling_log:
        writer.add_paragraph("modeling_log.json unavailable. Modeling summary not available.")
        return

    strategy = modeling_log.get("modeling_strategy", "n/a")
    selected_by_block = modeling_log.get("selected_model_by_block") or {}
    writer.add_key_value_table(_model_selection_table_rows(modeling_log, fit_meta))

    if strategy == "per_block" and selected_by_block:
        writer.add_subsection_title("Selected per-block models", min_following_space=0.22)
        writer.add_bullet_list(_format_per_block_models_full(modeling_log), fontsize=8)

    booster_status = modeling_log.get("third_party_booster_status") or {}
    if booster_status:
        writer.add_subsection_title("Third-party booster availability", min_following_space=0.35)
        writer.add_key_value_table(
            [
                ("XGBoost", _friendly_booster_status(booster_status.get("xgboost", "not recorded"))),
                ("LightGBM", _friendly_booster_status(booster_status.get("lightgbm", "not recorded"))),
                ("CatBoost", _friendly_booster_status(booster_status.get("catboost", "not recorded"))),
            ],
            keep_together=True,
        )

    if modeling_log.get("model_fit_failed"):
        writer.add_paragraph(
            "Note: cross-validation results are reported for transparency, but final model fitting failed and the submitted predictions used fallback values."
        )

    model_warnings = modeling_log.get("warnings") or []
    if model_warnings:
        writer.add_paragraph("Modeling warnings:")
        writer.add_bullet_list([str(w) for w in model_warnings])


def _cv_is_block_averaged(cv_results: pd.DataFrame, metric_spec: Dict[str, Any]) -> bool:
    if metric_spec.get("metric_mode") == "block_averaged":
        return True
    if cv_results is None or cv_results.empty or "metric_mode" not in cv_results.columns:
        return False
    return (cv_results["metric_mode"] == "block_averaged").any()


def _selected_per_block_cv_table(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Return one best row per scoring block for report display.

    If the final selected strategy is global, per-block rows will not have
    is_selected=True. In that case, still show the best per-block candidate for
    each official scoring block so readers can see the per-block comparison.
    """
    if cv_results is None or cv_results.empty:
        return pd.DataFrame()

    per_block = cv_results[cv_results.get("strategy", "") == "per_block"].copy()
    if per_block.empty or "block" not in per_block.columns:
        return pd.DataFrame()

    summary_labels = {"**overall**", "__overall__", "nan", "None", ""}
    per_block = per_block[~per_block["block"].astype(str).isin(summary_labels)]
    if per_block.empty:
        return pd.DataFrame()

    usable_status = {"ok", "fallback_insufficient_rows"}
    if "status" in per_block.columns:
        candidates = per_block[per_block["status"].astype(str).isin(usable_status)].copy()
    else:
        candidates = per_block.copy()

    if candidates.empty:
        return pd.DataFrame()

    selected = pd.DataFrame()
    if "is_selected" in candidates.columns:
        selected = candidates[candidates["is_selected"] == True].copy()  # noqa: E712

    if selected.empty:
        candidates["_score_numeric"] = pd.to_numeric(candidates.get("score_mean"), errors="coerce")
        candidates = candidates[np.isfinite(candidates["_score_numeric"])]
        if candidates.empty:
            return pd.DataFrame()
        best_idx = candidates.groupby("block")["_score_numeric"].idxmin()
        selected = candidates.loc[best_idx].copy()

    cols = [c for c in ["block", "model", "score_mean", "cv_folds", "status"] if c in selected.columns]
    out = selected[cols].copy()
    if "score_mean" in out.columns:
        out = out.rename(columns={"score_mean": "block_mae"})
    return out.sort_values("block").reset_index(drop=True)


def _cv_strategy_comparison_table(
    cv_results: pd.DataFrame,
    modeling_log: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    final_strategy = modeling_log.get("modeling_strategy", "global")

    summary = cv_results[cv_results.get("status", "") == "strategy_summary"] if not cv_results.empty else pd.DataFrame()
    for _, row in summary.iterrows():
        rows.append(
            {
                "strategy": row.get("strategy"),
                "selected": str(row.get("strategy")) == final_strategy,
                "score_mean": row.get("score_mean"),
                "metric_mode": row.get("metric_mode"),
                "status": row.get("status"),
                "notes": row.get("notes", row.get("model", "")),
            }
        )

    if not rows:
        rows.append(
            {
                "strategy": "global",
                "selected": final_strategy == "global",
                "score_mean": modeling_log.get("global_strategy_score"),
                "metric_mode": (modeling_log.get("metric_spec") or {}).get("metric_mode"),
                "status": "ok",
                "notes": modeling_log.get("selected_model"),
            }
        )
        if modeling_log.get("per_block_strategy_score") is not None:
            rows.append(
                {
                    "strategy": "per_block",
                    "selected": final_strategy == "per_block",
                    "score_mean": modeling_log.get("per_block_strategy_score"),
                    "metric_mode": (modeling_log.get("metric_spec") or {}).get("metric_mode"),
                    "status": "ok",
                    "notes": "mean of selected block models",
                }
            )
    return pd.DataFrame(rows)


def _global_comparison_cv_table(cv_results: pd.DataFrame, max_rows: int = 8) -> pd.DataFrame:
    if cv_results.empty:
        return cv_results
    global_rows = cv_results[
        (cv_results.get("strategy", "") == "global")
        & (cv_results.get("status", "") == "ok")
    ].copy()
    if global_rows.empty:
        return global_rows
    if "score_mean" in global_rows.columns:
        global_rows = global_rows.sort_values("score_mean", na_position="last")
    if "score_mean" in global_rows.columns:
        global_rows = global_rows.rename(columns={"score_mean": "block_avg_mae"})
    elif "block_averaged_mae" in global_rows.columns:
        global_rows = global_rows.rename(columns={"block_averaged_mae": "block_avg_mae"})

    top_global_cols = ["model", "family", "block_avg_mae", "folds", "status"]
    rename_map = {
        "model_family": "family",
        "cv_folds": "folds",
    }
    global_rows = global_rows.rename(columns=rename_map)
    display_cols = [c for c in top_global_cols if c in global_rows.columns]
    if not display_cols:
        display_cols = [
            c
            for c in ["model", "model_family", "block_avg_mae", "cv_folds", "status"]
            if c in global_rows.columns
        ]
    return global_rows[display_cols].head(max_rows).reset_index(drop=True)


def _render_cv_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    cv_results = context.get("cv_results")
    modeling_log = context.get("modeling_log") or {}
    metric_spec = context.get("metric_spec") or modeling_log.get("metric_spec") or {}

    writer.add_section_page(
        "7. Cross-Validation Results",
        "Cross-validated scores used for global vs per-block strategy and model selection.",
    )

    if cv_results is None or cv_results.empty:
        writer.add_paragraph("cv_results.csv unavailable or empty.")
        return

    if _cv_is_block_averaged(cv_results, metric_spec):
        writer.add_paragraph(
            "The detected scoring mode is block-averaged MAE. Per-block rows show MAE within each "
            "scoring block. The overall per-block score is the arithmetic mean of the selected block MAEs."
        )

        strategy_table = _cv_strategy_comparison_table(cv_results, modeling_log)
        strategy_display = strategy_table.copy()
        strategy_display = strategy_display.rename(
            columns={c: REPORT_COLUMN_LABELS.get(str(c), str(c)) for c in strategy_display.columns}
        )
        strategy_cols = [c for c in ["strategy", "selected", "score_mean", "metric", "status"] if c in strategy_display.columns]
        writer.add_titled_dataframe_table(
            "Strategy comparison",
            strategy_display[strategy_cols],
            keep_together=len(strategy_display) <= 6,
        )
        if "notes" in strategy_table.columns:
            notes: List[str] = []
            for _, row in strategy_table.iterrows():
                note = row.get("notes")
                if note and str(note).strip():
                    notes.append(f"{row.get('strategy')}: {_display_model_name(note)}")
            if notes:
                writer.add_bullet_list(notes, fontsize=8)

        per_block_table = _format_cv_display_df(_selected_per_block_cv_table(cv_results))
        if not per_block_table.empty:
            writer.add_titled_dataframe_table(
                "Selected per-block models",
                per_block_table,
                keep_together=True,
            )
            block_maes = per_block_table.get("block_mae")
            if block_maes is not None and len(block_maes) > 0:
                mean_mae = float(pd.to_numeric(block_maes, errors="coerce").mean())
                writer.add_paragraph(
                    f"Overall per-block MAE (mean of selected block MAEs): {format_report_value(mean_mae)}"
                )
        else:
            writer.add_paragraph("No per-block model results available.")

        global_table = _format_cv_display_df(_global_comparison_cv_table(cv_results))
        if not global_table.empty:
            writer.add_titled_dataframe_table(
                "Top global models",
                global_table,
                max_rows_per_page=8,
                keep_together=True,
            )
        return

    display_cols = [
        c
        for c in [
            "strategy",
            "block",
            "model",
            "model_family",
            "is_third_party",
            "cv_folds",
            "score_mean",
            "score_std",
            "metric_mode",
            "status",
        ]
        if c in cv_results.columns
    ]
    if not display_cols:
        display_cols = list(cv_results.columns)

    trimmed, note = _prepare_cv_display(cv_results[display_cols])
    trimmed = _format_cv_display_df(trimmed)
    writer.add_titled_dataframe_table("CV results", trimmed, note=note)


def _render_validation_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    validation_report = context.get("validation_report") or {}
    checks = context.get("submission_checks") or {}

    writer.add_section_page(
        "8. Submission Validation",
        "Validation of submission.csv against sample_submission and inferred schema.",
    )

    if not validation_report:
        writer.add_paragraph("validation_report.json unavailable. Run scripts/validate_submission.py.")
        return

    writer.add_key_value_table(
        [
            ("Validation status", validation_report.get("status", "n/a")),
            ("Required columns", format_list_compactly(validation_report.get("required_columns"))),
            ("Submission row count", validation_report.get("n_submission_rows")),
            ("Sample row count", validation_report.get("n_sample_rows")),
            ("Row id column", validation_report.get("row_id_column")),
            ("Target column", validation_report.get("target_column")),
            ("Row id coverage OK", checks.get("row_id_coverage_ok")),
            ("Predictions finite", checks.get("predictions_finite")),
            ("Finite predictions", checks.get("n_finite_predictions")),
            ("Non-finite predictions", checks.get("n_non_finite_predictions")),
        ],
        keep_together=True,
    )

    errors = validation_report.get("errors") or []
    if errors:
        writer.add_paragraph("Validation errors:")
        writer.add_bullet_list([str(e) for e in errors])
    else:
        writer.add_paragraph("Validation errors: none.")

    val_warnings = validation_report.get("warnings") or []
    if val_warnings:
        writer.add_paragraph("Validation warnings:")
        writer.add_bullet_list([str(w) for w in val_warnings])


def _render_output_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    root = Path(context.get("root") or project_root())
    pipeline_log = context.get("pipeline_log") or {}
    report_log = load_json_safely(artifacts_dir(root) / "report_log.json")

    writer.add_section_page(
        "10. Output Files and Reproducibility",
        "Generated artifacts, pipeline steps, and execution constraints.",
    )

    scripts_run_items = _scripts_run_items(pipeline_log)

    pipeline_status = pipeline_log.get("status", "n/a")
    report_pending = bool(pipeline_log.get("report_generation_pending", False))

    if pipeline_status == "pre_report":
        # The report is being generated from a pre-report log, but in the final PDF
        # this should be presented as a completed pipeline up to report generation.
        pipeline_status_display = "schema/modeling/validation completed; report generated"
    elif pipeline_status in {"passed", "ok"}:
        pipeline_status_display = "passed"
    else:
        pipeline_status_display = pipeline_status

    writer.add_key_value_table(
        [
            ("submission.csv", "submission.csv"),
            ("report.pdf", "report.pdf"),
            ("Pipeline status", pipeline_status_display),
        ],
        keep_together=True,
    )

    writer.add_subsection_title("Scripts run", min_following_space=0.12)
    writer.add_bullet_list(scripts_run_items, fontsize=8)

    writer.add_subsection_title("Artifacts used", min_following_space=0.12)
    artifact_values = context.get("artifacts_used") or report_log.get("artifacts_used") or []
    if artifact_values:
        artifact_items = [_display_path(p, root) for p in artifact_values]
        writer.add_bullet_list(artifact_items, fontsize=8)
    else:
        writer.add_paragraph("No artifact list was recorded.")

    writer.add_subsection_title("Reproducibility notes", min_following_space=0.12)
    writer.add_bullet_list(
        [
            "No external datasets are downloaded or used.",
            "Modeling uses CPU-friendly tabular methods; scikit-learn models are always available, and XGBoost, LightGBM, or CatBoost are auto-detected when installed.",
            "Schema inference and metric parsing are heuristic and depend on DATA_DESCRIPTION.md wording.",
        ]
    )


def _render_limitations_section(writer: PdfReportWriter, context: Dict[str, Any]) -> None:
    writer.add_section_page(
        "11. Limitations",
        "Known assumptions, skipped modalities, and fallback behaviors.",
    )

    writer.add_bullet_list(
        [
            "Schema inference is heuristic: file roles, join keys, and column types may be incorrect for unusual layouts.",
            "Metric parsing depends on DATA_DESCRIPTION.md wording; block-aware scoring is enabled when explicit or strongly implied scoring-language cues are detected.",
            "Text and image sidecar features are used when columns/files can be matched confidently; otherwise the pipeline skips them safely and records the availability metadata.",
            "Image matching requires composite key agreement; ambiguous filename matches are ignored.",
            "If modeling fails or rows are too few for CV, median-based fallback predictions are used.",
            "Per-block models use median fallback for blocks with insufficient rows; these scores are included in strategy comparison.",
            "Cross-validation uses a runtime budget; heavier models may be skipped on large datasets.",
            "Missing artifacts are reported in this document rather than aborting report generation.",
        ]
    )

    report_warnings = context.get("warnings") or []
    if report_warnings:
        writer.add_paragraph("Report generation warnings:")
        writer.add_bullet_list([str(w) for w in report_warnings])


def create_pdf_report(root: Path, output_path: Path) -> Dict[str, Any]:
    """Create the Sentinelle PDF report and return a generation log."""
    root = root.resolve()
    output_path = output_path.resolve()
    context = build_report_context(root)
    warnings: List[str] = list(context.get("warnings") or [])
    pages_written = 0

    try:
        with PdfPages(output_path) as pdf:
            writer = PdfReportWriter(
                pdf,
                document_title="Sentinelle Report",
                logo_path=None,
            )
            _render_cover_section(writer, context)
            _render_dataset_section(writer, context)
            _render_eda_section(writer, context)
            _render_metric_section(writer, context)
            _render_feature_section(writer, context)
            _render_model_selection_section(writer, context)
            _render_cv_section(writer, context)
            _render_validation_section(writer, context)
            _render_analytical_summary_section(writer, context)
            _render_output_section(writer, context)
            _render_limitations_section(writer, context)
            pages_written = writer.close()

        status = "ok" if output_path.exists() and pages_written > 0 else "failed"
    except Exception as exc:
        warnings.append(f"report generation failed: {type(exc).__name__}: {exc}")
        status = "failed"
        if output_path.exists():
            output_path.unlink(missing_ok=True)

    if not context.get("eda_summary"):
        warnings.append("eda_summary.json not available; EDA section may be empty")

    return {
        "status": status,
        "output_path": str(output_path),
        "pages_written": pages_written,
        "artifacts_used": context.get("artifacts_used") or [],
        "warnings": warnings,
        "eda_status": context.get("eda_status"),
        "eda_warnings": context.get("eda_warnings") or [],
        "eda_artifacts_used": context.get("eda_artifacts_used") or [],
    }
