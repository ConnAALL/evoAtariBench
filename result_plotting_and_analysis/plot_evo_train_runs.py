"""
Simple plotting script for evo_train_runs.db that compares compression methods.

This script handles cases where multiple compression methods (e.g., "none" and "dct")
with the same nonlinearity method are stored in the same database.

This is a **simple, no-CLI** plotting utility:
- Edit the CONFIG section at the top of this file.
- Run: `python3 result_plotting_and_analysis/plot_evo_train_runs.py`

Outputs (two PNGs):
- avg_of_avgs: mean of the runs' `avg` curves (per game, per compression method)
- best_overall: best-so-far curve of the single best run (per game, per compression method)
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Optional: match repo plotting style if available.
    import scienceplots  # type: ignore  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    pass

###############################################################################
# CONFIG (edit these values; no command line args)
###############################################################################

# Database file path (absolute path or relative to repo root)
DB_PATH: str | None = None  # if None, uses: data/evo_train_runs.db

# Which compression methods to plot (leave empty to plot all available)
COMPRESSION_METHODS: list[str] = []  # examples: ["none", "dct"] or [] for all

# Which nonlinearity method to filter by (leave None to use all, but must match across compressions)
NONLINEARITY_FILTER: str | None = "sparsification"

# Output directory:
# - set to None to use the repo default: out/plots/evo_train_runs/
# - or set to an absolute path.
OUT_DIR: str | None = None

# Plot formatting.
COLS: int = 3
LIMIT_GAMES: int | None = None
DPI: int = 500
TITLE: str | None = "evo_train_runs"  # if None, uses DB filename


def _repo_root() -> str:
    """This script lives in result_plotting_and_analysis/, so repo root is 1 level up."""
    here = os.path.abspath(__file__)
    d = os.path.dirname(here)  # .../result_plotting_and_analysis
    parent = os.path.dirname(d)
    if os.path.isdir(os.path.join(parent, "data")):
        return parent
    return os.path.dirname(os.path.dirname(here))


def _safe_slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "method"


def _env_to_game(env_name: str) -> str:
    # "ALE/SpaceInvaders-v5" -> "SpaceInvaders"
    s = str(env_name)
    if "/" in s:
        s = s.split("/", 1)[1]
    s = re.sub(r"-v\d+$", "", s, flags=re.IGNORECASE)
    return s


@dataclass(frozen=True)
class Run:
    run_id: int
    env_name: str
    task: dict[str, Any]
    plot_data: list[list[float]]
    compression: str
    nonlinearity: str


def _load_runs(db_path: str) -> list[Run]:
    """Load all runs from the database, extracting compression and nonlinearity from task_json."""
    con = sqlite3.connect(db_path, timeout=30.0)
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT run_id, env_name, task_json, plot_data_json FROM runs "
            "WHERE env_name IS NOT NULL AND plot_data_json IS NOT NULL AND task_json IS NOT NULL"
        ).fetchall()
        out: list[Run] = []
        for run_id, env_name, task_json, plot_data_json in rows:
            try:
                task = json.loads(task_json) if task_json is not None else {}
                plot_data = json.loads(plot_data_json)
                if not isinstance(task, dict):
                    task = {}
                if not isinstance(plot_data, list) or not plot_data:
                    continue

                # Extract compression and nonlinearity from task
                compression = _safe_slug(task.get("compression", "unknown"))
                nonlinearity = _safe_slug(task.get("nonlinearity", "unknown"))

                out.append(Run(int(run_id), str(env_name), task, plot_data, compression, nonlinearity))
            except Exception:
                continue
        return out
    finally:
        con.close()


def _extract_curve(plot_data: list[list[float]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """
    plot_data rows are [generation, best, avg] (from scripts/single_run.py).
    """
    arr = np.asarray(plot_data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected plot_data shape: {arr.shape}")
    x = arr[:, 0]
    if metric in ("best", "best_overall"):
        y = arr[:, 1]
    elif metric == "avg":
        y = arr[:, 2]
    else:
        raise ValueError("metric must be one of: best, avg, best_overall")
    return x, y


def _align_and_mean(curves: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Align curves by trimming to the shortest length; assumes x's are comparable."""
    curves = list(curves)
    if not curves:
        raise ValueError("No curves to align")
    min_len = min(int(len(y)) for _x, y in curves)
    x0 = curves[0][0][:min_len]
    ys = [y[:min_len] for _x, y in curves]
    y_stack = np.vstack(ys)
    mean_y = np.mean(y_stack, axis=0)
    return x0, ys, mean_y


def _best_run_by_peak_best(rs: list[Run]) -> Run | None:
    """Choose the single best run for a (compression, env) group by peak best value."""
    best_run: Run | None = None
    best_score: float | None = None
    for r in rs:
        try:
            _x, y_best = _extract_curve(r.plot_data, metric="best")
            score = float(np.max(y_best))
        except Exception:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_run = r
    return best_run


def main():
    # Determine database path
    if DB_PATH:
        db_path = os.path.abspath(DB_PATH)
    else:
        db_path = os.path.join(_repo_root(), "data", "evo_train_runs.db")

    if not os.path.exists(db_path):
        raise SystemExit(f"Database not found: {db_path}")

    # Determine output directory
    if OUT_DIR:
        out_dir = os.path.abspath(OUT_DIR)
    else:
        out_dir = os.path.join(_repo_root(), "out", "plots", "evo_train_runs")
    os.makedirs(out_dir, exist_ok=True)

    # Load all runs
    print(f"[load] Loading runs from: {db_path}")
    all_runs = _load_runs(db_path)
    print(f"[load] Found {len(all_runs)} total runs")

    # Filter by nonlinearity if specified
    if NONLINEARITY_FILTER:
        nonlinearity_filter = _safe_slug(NONLINEARITY_FILTER)
        all_runs = [r for r in all_runs if r.nonlinearity == nonlinearity_filter]
        print(f"[filter] After nonlinearity filter ({nonlinearity_filter}): {len(all_runs)} runs")

    # Group by compression and env: compression -> env -> runs
    by_comp_env: dict[str, dict[str, list[Run]]] = {}
    compressions_set: set[str] = set()

    for r in all_runs:
        compressions_set.add(r.compression)
        by_comp_env.setdefault(r.compression, {}).setdefault(r.env_name, []).append(r)

    compressions = sorted(compressions_set)

    # Filter compressions if specified
    if COMPRESSION_METHODS:
        requested = {_safe_slug(c) for c in COMPRESSION_METHODS}
        compressions = [c for c in compressions if c in requested]
        if not compressions:
            raise SystemExit(f"No runs found for requested compression methods: {COMPRESSION_METHODS}")

    print(f"[found] Compression methods: {compressions}")

    if not compressions:
        raise SystemExit("No compression methods found in database")

    # Determine env list: union across compressions
    env_set = set()
    for comp in compressions:
        env_set.update(by_comp_env.get(comp, {}).keys())
    envs = sorted(env_set)

    if LIMIT_GAMES is not None:
        envs = envs[: max(0, int(LIMIT_GAMES))]

    if not envs:
        raise SystemExit("No games/envs to plot.")

    print(f"[found] Environments: {len(envs)} ({', '.join([_env_to_game(e) for e in envs[:5]])}{'...' if len(envs) > 5 else ''})")

    # Setup plot layout
    cols = max(1, int(COLS))
    n_rows = int(math.ceil(len(envs) / float(cols)))

    cell_w, cell_h = 2.05, 1.55
    fig_w, fig_h = cols * cell_w, n_rows * cell_h

    title_prefix = TITLE if TITLE is not None else os.path.splitext(os.path.basename(db_path))[0]
    comp_colors = {c: f"C{i}" for i, c in enumerate(compressions)}

    def plot_metric(metric: str) -> None:
        fig, axes = plt.subplots(n_rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        if metric == "avg_of_avgs":
            fig.suptitle(f"{title_prefix} — avg of avgs", fontsize=12)
        elif metric == "best_overall":
            fig.suptitle(f"{title_prefix} — best overall", fontsize=12)
        else:
            raise ValueError(f"unknown metric: {metric}")

        handle_by_comp: dict[str, Any] = {}

        for i, env in enumerate(envs):
            ax = axes[i // cols][i % cols]
            plotted_any = False

            for comp in compressions:
                rs = by_comp_env.get(comp, {}).get(env, [])
                if not rs:
                    continue

                if metric == "best_overall":
                    br = _best_run_by_peak_best(rs)
                    if br is None:
                        continue
                    try:
                        x, y_best = _extract_curve(br.plot_data, metric="best")
                        y = np.maximum.accumulate(y_best)
                    except Exception:
                        continue
                else:  # avg_of_avgs
                    curves = []
                    for r in rs:
                        try:
                            curves.append(_extract_curve(r.plot_data, metric="avg"))
                        except Exception:
                            continue
                    if not curves:
                        continue
                    try:
                        x, _ys, y = _align_and_mean(curves)
                    except Exception:
                        continue

                (line,) = ax.plot(x, y, color=comp_colors[comp], linewidth=1.25, alpha=0.95)
                plotted_any = True
                handle_by_comp.setdefault(comp, line)

            ax.set_title(_env_to_game(env), fontsize=7)
            ax.tick_params(axis="both", which="major", labelsize=6, length=2)
            ax.grid(True, alpha=0.12, linewidth=0.4)
            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    fontsize=6,
                    transform=ax.transAxes,
                    alpha=0.6,
                )

        # Turn off any unused axes
        for j in range(len(envs), n_rows * cols):
            axes[j // cols][j % cols].set_axis_off()

        if handle_by_comp:
            legend_labels = sorted(handle_by_comp.keys())
            legend_handles = [handle_by_comp[c] for c in legend_labels]
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(len(legend_labels), cols),
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
            )

        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        slug = "__".join([_safe_slug(c) for c in compressions])
        if len(slug) > 120:
            slug = "compressions"
        out_path = os.path.join(out_dir, f"{_safe_slug(title_prefix)}_{slug}_{metric}_grid.png")
        fig.savefig(out_path, dpi=int(DPI))
        plt.close(fig)
        print(f"[wrote] {out_path}")

    plot_metric("avg_of_avgs")
    plot_metric("best_overall")

    print(f"Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
