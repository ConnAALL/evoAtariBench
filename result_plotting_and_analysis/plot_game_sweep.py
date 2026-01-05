"""
Plot "game sweep" results from a directory of SQLite DBs (one DB per nonlinearity method).

For a fixed compression method (--comp), this script:
  - finds all matching DBs in data/run_data/<comp>_game_sweep/ (or --run-dir)
  - loads runs grouped by env_name (game) and method (from the DB filename)
  - writes ONE large PNG per metric (best + avg + best_overall) containing a grid of tiny per-game plots

Each per-game subplot overlays the mean curves for all methods (different colors).

Example:
  python3 result_plotting_and_analysis/plot_game_sweep.py --comp dct
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _repo_root() -> str:
    """
    This script lives in result_plotting_and_analysis/, so repo root is 1 level up.
    (Fallback included in case the file is moved elsewhere.)
    """
    here = os.path.abspath(__file__)
    d = os.path.dirname(here)  # .../result_plotting_and_analysis
    parent = os.path.dirname(d)
    if os.path.isdir(os.path.join(parent, "data")):
        return parent
    # fallback
    return os.path.dirname(os.path.dirname(here))


def _safe_slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "method"


def _method_from_db_filename(comp: str, filename: str) -> str:
    """
    Examples:
      comp=dct, filename=dct_dropout_250gen.db -> dropout
      comp=dct, filename=dct_quantization_250gen.db -> quantization
    """
    base = os.path.basename(filename)
    if base.lower().endswith(".db"):
        base = base[:-3]
    prefix = f"{comp.lower()}_"
    if base.lower().startswith(prefix):
        base = base[len(prefix) :]
    base = re.sub(r"_\d+gen$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_gen$", "", base, flags=re.IGNORECASE)
    return _safe_slug(base)


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


def _load_runs(db_path: str) -> list[Run]:
    con = sqlite3.connect(db_path, timeout=30.0)
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT run_id, env_name, task_json, plot_data_json FROM runs "
            "WHERE env_name IS NOT NULL AND plot_data_json IS NOT NULL"
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
                out.append(Run(int(run_id), str(env_name), task, plot_data))
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
    """
    Align curves by trimming to the shortest length; assumes x's are comparable.
    Returns (x, ys_trimmed, mean_y).
    """
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
    """
    Choose the single best run for a (method, env) group.

    We define "best" as the run with the highest peak `best` value across generations.
    (If a run never reaches a higher best score than others, it won't be selected.)
    """
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


def _default_run_dir(comp: str) -> str:
    return os.path.join(_repo_root(), "data", "run_data", f"{comp}_game_sweep")


def _default_out_dir(comp: str) -> str:
    return os.path.join(_repo_root(), "out", "plots", "game_sweep", comp)


def _find_db_paths(run_dir: str, comp: str) -> list[str]:
    comp_l = str(comp).lower()
    out = []
    for fn in os.listdir(run_dir):
        if not fn.lower().endswith(".db"):
            continue
        # Prefer only matching comp_*.db, but fall back to including everything if none match.
        if fn.lower().startswith(f"{comp_l}_"):
            out.append(os.path.join(run_dir, fn))
    if out:
        return sorted(out)
    # fallback: include all DBs
    return sorted([os.path.join(run_dir, fn) for fn in os.listdir(run_dir) if fn.lower().endswith(".db")])


def _load_envs_from_csv(csv_path: str) -> list[str]:
    """
    Load an ordered list of envs from a CSV that has an `environment` column.
    Used to force a stable "all games" list even if some games are missing from the sweep DBs.
    """
    if not csv_path:
        return []
    if not os.path.exists(csv_path):
        raise SystemExit(f"--games-csv not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "environment" not in reader.fieldnames:
            raise SystemExit(f"--games-csv must have an 'environment' column: {csv_path}")
        envs = []
        for r in reader:
            e = (r.get("environment") or "").strip()
            if e:
                envs.append(e)
        return envs


def _get_args():
    p = argparse.ArgumentParser(description="Plot game sweep results into combined PNG grids (one per method DB).")
    p.add_argument("--comp", required=True, choices=["dct"], help="Compression method (determines default run-dir).")
    p.add_argument("--run-dir", default=None, help="Directory containing per-method sweep DBs.")
    p.add_argument("--out-dir", default=None, help="Directory to write PNGs to.")
    p.add_argument("--cols", type=int, default=6, help="Number of columns in the output grid.")
    p.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Optional cap on number of games per figure (useful for quick sanity-checks).",
    )
    p.add_argument("--dpi", type=int, default=260, help="Output image DPI.")
    p.add_argument("--title", default=None, help="Optional title prefix.")
    p.add_argument(
        "--games-csv",
        default=None,
        help="Optional CSV (with `environment` column) to define the game list/order for the grid. "
        "If omitted, uses the union of envs found across the sweep DBs.",
    )
    return p.parse_args()


def main():
    args = _get_args()
    comp = str(args.comp).lower()

    run_dir = os.path.abspath(args.run_dir) if args.run_dir else os.path.abspath(_default_run_dir(comp))
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.abspath(_default_out_dir(comp))
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(run_dir):
        raise SystemExit(f"--run-dir not found or not a directory: {run_dir}")

    db_paths = _find_db_paths(run_dir, comp=comp)
    if not db_paths:
        raise SystemExit(f"No .db files found in: {run_dir}")

    # Delay-import matplotlib so the script errors cleanly if it isn't installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load everything once: method -> env -> runs
    methods: list[str] = []
    by_method_env: dict[str, dict[str, list[Run]]] = {}
    for db_path in db_paths:
        method = _method_from_db_filename(comp, db_path)
        runs = _load_runs(db_path)
        if not runs:
            print(f"[skip] no runs in: {db_path}")
            continue
        methods.append(method)
        env_map: dict[str, list[Run]] = {}
        for r in runs:
            env_map.setdefault(r.env_name, []).append(r)
        by_method_env[method] = env_map

    methods = sorted(set(methods))
    if not methods:
        raise SystemExit(f"No usable runs found across DBs in: {run_dir}")

    # Determine env list: either from CSV, or union across methods.
    if args.games_csv:
        envs = _load_envs_from_csv(os.path.abspath(args.games_csv))
    else:
        env_set = set()
        for env_map in by_method_env.values():
            env_set.update(env_map.keys())
        envs = sorted(env_set)

    if args.limit_games is not None:
        envs = envs[: max(0, int(args.limit_games))]

    if not envs:
        raise SystemExit("No games/envs to plot.")

    cols = max(1, int(args.cols))
    n_rows = int(math.ceil(len(envs) / float(cols)))

    # Small per-subplot footprint; the overall PNG grows with number of games.
    cell_w, cell_h = 2.05, 1.55
    fig_w, fig_h = cols * cell_w, n_rows * cell_h

    title_prefix = args.title if args.title is not None else comp

    method_colors = {m: f"C{i}" for i, m in enumerate(methods)}

    for metric in ("best", "avg", "best_overall"):
        fig, axes = plt.subplots(n_rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(f"{title_prefix} — methods overlay ({metric})", fontsize=12)

        handle_by_method = {}

        for i, env in enumerate(envs):
            ax = axes[i // cols][i % cols]

            plotted_any = False
            for m in methods:
                rs = by_method_env.get(m, {}).get(env, [])
                if not rs:
                    continue

                if metric == "best_overall":
                    # Plot ONLY the best run for this method+env.
                    br = _best_run_by_peak_best(rs)
                    if br is None:
                        continue
                    try:
                        x, y_best = _extract_curve(br.plot_data, metric="best")
                        y = np.maximum.accumulate(y_best)  # best-so-far for that best run
                    except Exception:
                        continue
                else:
                    # Plot the mean curve across runs (per-generation average).
                    curves = []
                    for r in rs:
                        try:
                            curves.append(_extract_curve(r.plot_data, metric=metric))
                        except Exception:
                            continue
                    if not curves:
                        continue
                    try:
                        x, _ys, y = _align_and_mean(curves)
                    except Exception:
                        continue

                (line,) = ax.plot(x, y, color=method_colors[m], linewidth=1.25, alpha=0.95)
                plotted_any = True
                if m not in handle_by_method:
                    handle_by_method[m] = line

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

        # Turn off any unused axes.
        for j in range(len(envs), n_rows * cols):
            axes[j // cols][j % cols].set_axis_off()

        if handle_by_method:
            legend_labels = sorted(handle_by_method.keys())
            legend_handles = [handle_by_method[m] for m in legend_labels]
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
        out_path = os.path.join(out_dir, f"{comp}_methods_{metric}_grid.png")
        fig.savefig(out_path, dpi=int(args.dpi))
        plt.close(fig)
        print(f"[wrote] {out_path}")

    print(f"Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()


