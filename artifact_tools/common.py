"""Shared helpers for publication displays built only from saved results."""
from __future__ import annotations

import hashlib
import json
import os
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT
CORE = ROOT
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CAMERA_READY_OUT", str(ROOT / "artifacts"))).resolve()
TABLES, FIGURES, DATA, NOTES = (OUT / name for name in ("tables", "figures", "data", "notes"))

# NeurIPS text width is 5.5 in; figure fonts are sized for that final width.
TEXT_WIDTH_IN = 5.5
COLORS = {
    "A+B": "#1976D2",
    "A-only": "#FF8A65",
    "zero": "#9E9E9E",
    "A_nu": "#607D8B",
    "empirical": "#795548",
    "isotonic": "#43A047",
    "binned": "#8E24AA",
    "spline": "#D32F2F",
    "mlp": "#00897B",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


class Group:
    """Registry of the inputs and outputs of one display group."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.displays: dict[str, dict] = {}
        for folder in (TABLES, FIGURES, DATA, NOTES):
            folder.mkdir(parents=True, exist_ok=True)

    def display(self, name: str, placement: str, description: str) -> "Display":
        if not name.startswith(self.prefix + "_"):
            raise ValueError(f"{name} must start with {self.prefix}_")
        record = dict(placement=placement, description=description, inputs={}, outputs={})
        self.displays[name] = record
        return Display(name, record)

    def write_manifest(self) -> Path:
        path = OUT / f"manifest_{self.prefix}.json"
        path.write_text(json.dumps(self.displays, indent=2, sort_keys=True) + "\n")
        return path


class Display:
    def __init__(self, name: str, record: dict):
        self.name, self.record = name, record

    def read_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        path = Path(path)
        self.record["inputs"][rel(path)] = sha256(path)
        return pd.read_csv(path, **kwargs)

    def input(self, path: Path) -> Path:
        """Register a non-CSV input (or a CSV read some other way)."""
        self.record["inputs"][rel(path)] = sha256(path)
        return Path(path)

    def _output(self, path: Path) -> Path:
        self.record["outputs"][rel(path)] = sha256(path)
        return path

    def write_data(self, frame: pd.DataFrame, suffix: str = "") -> Path:
        path = DATA / f"{self.name}{suffix}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        return self._output(path)

    def write_table(self, colspec: str, header: Sequence[str] | Sequence[Sequence[str]],
                    rows: Iterable[Sequence[str]], suffix: str = "",
                    midrules_after: Iterable[int] = ()) -> Path:
        """Write a booktabs tabular body. `rows` are pre-formatted strings."""
        headers = [header] if header and isinstance(header[0], str) else list(header)
        lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
        lines += [" & ".join(h) + r" \\" for h in headers]
        lines.append(r"\midrule")
        breaks = set(midrules_after)
        for index, row in enumerate(rows):
            lines.append(" & ".join(row) + r" \\")
            if index in breaks:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
        path = TABLES / f"{self.name}{suffix}.tex"
        path.write_text("\n".join(lines))
        return self._output(path)

    def save_figure(self, fig, suffix: str = "") -> Path:
        pdf = FIGURES / f"{self.name}{suffix}.pdf"
        fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None})
        fig.savefig(pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
        plt.close(fig)
        self._output(pdf.with_suffix(".png"))
        return self._output(pdf)

    def write_note(self, text: str) -> Path:
        path = NOTES / f"{self.name}.md"
        path.write_text(text.strip() + "\n")
        return self._output(path)


def fmt(value, digits: int, percent: bool = False, bold: bool = False) -> str:
    """Fixed-decimal LaTeX number; `$-$` for negatives, `--` for missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    value = float(value) * (100 if percent else 1)
    text = f"{abs(value):.{digits}f}"
    if value < 0 and float(text) != 0:
        text = "$-$" + text
    if percent:
        text += r"\%"
    return rf"\textbf{{{text}}}" if bold else text


def fmt_ci(low, high, digits: int) -> str:
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def figure_style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "savefig.pad_inches": 0.02,
    })


def require_wmt_fixed(frame: pd.DataFrame, *, label: str, unit_col: str | None = None,
                      path_col: str | None = None) -> None:
    """Reject the superseded 139-path WMT lineage in WMT rows.

    Only frames with a unit or path column can be checked; summaries without
    either must be filtered by source instead (see README).
    """
    text = frame.select_dtypes(include=["object", "string"])
    is_wmt = text.apply(lambda column: column.astype(str).str.contains("wmt", case=False)).any(axis=1)
    wmt = frame[is_wmt]
    if wmt.empty:
        return
    if unit_col is not None:
        units = wmt[unit_col].astype(int)
        bad = wmt[(units % 139 == 0) & (units % 141 != 0)]
        if not bad.empty:
            raise AssertionError(f"{label}: superseded WMT units {sorted(bad[unit_col].unique())}")
    if path_col is not None and wmt[path_col].nunique() not in (141,):
        raise AssertionError(f"{label}: expected 141 WMT paths, got {wmt[path_col].nunique()}")


def check(value: float, expected: float, tol: float, label: str) -> None:
    if not abs(float(value) - expected) <= tol:
        raise AssertionError(f"{label}: {value} != audited {expected} (tol {tol})")
