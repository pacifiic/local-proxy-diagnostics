"""Unified artifact-generation command runner.

The existing make_*.py scripts remain the source of truth for individual
artifact families. This module provides a single command-line entrypoint that
dispatches to those scripts with consistent roots and output directories.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent


@dataclass(frozen=True)
class BuildTask:
    key: str
    description: str
    command: Callable[[argparse.Namespace], list[str]]


def _script(name: str) -> str:
    return str(TOOLS_ROOT / name)


def _path(value: Path | str) -> str:
    return str(value)


def _core_tables_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        _script("make_paper_tables.py"),
        "--root",
        _path(args.results_root),
        "--out",
        _path(args.core_tables_out),
    ]


def _accept_boost_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        _script("make_accept_boost_tables.py"),
        "--root",
        _path(args.accept_root),
        "--out",
        _path(args.accept_out),
    ]


def _metric_audits_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        _script("make_metric_audit_tables.py"),
        "--results-root",
        _path(args.results_root),
        "--repo-root",
        _path(args.repo_root),
        "--out",
        _path(args.metric_audit_out),
    ]


def _robustness_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        _script("make_robustness_tables.py"),
        "--results-root",
        _path(args.results_root),
        "--robustness-root",
        _path(args.robustness_root),
        "--out",
        _path(args.robustness_out),
    ]


def _figures_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        _script("make_paper_figures.py"),
        "--results",
        _path(args.results_root),
        "--figures",
        _path(args.figures_out),
        "--tables-root",
        _path(args.core_tables_out),
        "--data-root",
        _path(args.data_root),
    ]


def _appendix_composites_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        _script("make_appendix_composite_figures.py"),
        "--results-root",
        _path(args.results_root),
        "--outdir",
        _path(args.figures_out),
    ]


def _kl_radius_audit_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        _script("make_kl_radius_audit_figure.py"),
        "--results-root",
        _path(args.results_root),
        "--outdir",
        _path(args.figures_out),
        "--stem",
        "fig_kl_radius_audit",
    ]


TASKS: dict[str, BuildTask] = {
    "core_tables": BuildTask(
        "core_tables",
        "core table exports",
        _core_tables_cmd,
    ),
    "accept_boost": BuildTask(
        "accept_boost",
        "accept-boost appendix tables",
        _accept_boost_cmd,
    ),
    "metric_audits": BuildTask(
        "metric_audits",
        "metric-audit tables",
        _metric_audits_cmd,
    ),
    "robustness": BuildTask(
        "robustness",
        "robustness tables",
        _robustness_cmd,
    ),
    "figures": BuildTask(
        "figures",
        "figure exports",
        _figures_cmd,
    ),
    "appendix_composites": BuildTask(
        "appendix_composites",
        "appendix composite figure exports",
        _appendix_composites_cmd,
    ),
    "kl_radius_audit": BuildTask(
        "kl_radius_audit",
        "KL-radius audit figure export",
        _kl_radius_audit_cmd,
    ),
}


TARGETS: dict[str, list[str]] = {
    "core-tables": ["core_tables"],
    "accept-boost": ["accept_boost"],
    "metric-audits": ["metric_audits"],
    "robustness": ["robustness"],
    "figures": ["figures", "appendix_composites", "kl_radius_audit"],
    "tables": [
        "core_tables",
        "accept_boost",
        "metric_audits",
        "robustness",
    ],
    "all": [
        "core_tables",
        "accept_boost",
        "metric_audits",
        "robustness",
        "figures",
        "appendix_composites",
        "kl_radius_audit",
    ],
}


def _default_accept_root(results_root: Path) -> Path:
    return results_root / "12_accept_boost"


def _default_robustness_root(results_root: Path) -> Path:
    return results_root / "extended_analyses"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build submission tables and figures through one entrypoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=sorted(TARGETS),
        help="Artifact family to build.",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--accept-root", type=Path, default=None)
    parser.add_argument("--robustness-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    default_tables = REPO_ROOT / "artifacts/tables"
    parser.add_argument("--core-tables-out", type=Path, default=default_tables)
    parser.add_argument("--accept-out", type=Path, default=default_tables)
    parser.add_argument("--metric-audit-out", type=Path, default=default_tables)
    parser.add_argument(
        "--robustness-out",
        type=Path,
        default=default_tables,
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=REPO_ROOT / "artifacts/figures",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke the underlying make_*.py scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser


def _finalize_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.accept_root is None:
        args.accept_root = _default_accept_root(args.results_root)
    if args.robustness_root is None:
        args.robustness_root = _default_robustness_root(args.results_root)
    return args


def _quote_command(cmd: Sequence[str]) -> str:
    return " ".join(_shell_quote(part) for part in cmd)


def _shell_quote(part: str) -> str:
    if not part:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=+-")
    if all(ch in safe for ch in part):
        return part
    return "'" + part.replace("'", "'\"'\"'") + "'"


def run(args: argparse.Namespace) -> int:
    args = _finalize_defaults(args)
    selected = TARGETS[args.target]
    for key in selected:
        task = TASKS[key]
        cmd = task.command(args)
        print(f"==> {task.description}", flush=True)
        print(_quote_command(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
