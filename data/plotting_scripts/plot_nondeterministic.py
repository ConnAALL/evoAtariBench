"""
Plot nondeterminism experiment results from a SQLite DB produced by scripts/evo_train_tasks.py.

Expected experiment: 4 configs x 12 repeats (48 rows total), grouped by:
  - frameskip
  - frameskip + sticky actions (REPEAT_ACTION_PROBABILITY=0.25)
  - frameskip + random noop (RANDOM_INIT=True)
  - frameskip + sticky actions + random noop

For each config, plots all runs + a mean curve, and also produces one combined plot of the mean curves.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np


def _repo_root() -> str:
    """
    This script lives in data/plotting_scripts/, so repo root is 3 levels up.
    (Fallback included in case the file is moved elsewhere.)
    """
    here = os.path.abspath(__file__)
    d = os.path.dirname(here)  # .../data/plotting_scripts
    d_data = os.path.dirname(d)  # .../data
    if os.path.basename(d_data) == "data":
        return os.path.dirname(d_data)  # .../repo
    # fallback (old layout: scripts/ -> repo is 2 up)
    return os.path.dirname(os.path.dirname(here))


def _default_db() -> str:
    return os.path.join(_repo_root(), "data", "nonDeterministic_Experiment.db")


def _default_out_dir() -> str:
    return os.path.join(_repo_root(), "data", "plots", "nondeterministic")


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
            "SELECT run_id, env_name, task_json, plot_data_json FROM runs ORDER BY id ASC;"
        ).fetchall()
        out: list[Run] = []
        for run_id, env_name, task_json, plot_data_json in rows:
            task = json.loads(task_json)
            plot_data = json.loads(plot_data_json)
            out.append(Run(int(run_id), str(env_name), task, plot_data))
        return out
    finally:
        con.close()


def _variant_label(task: dict[str, Any]) -> str:
    rap = float(task.get("REPEAT_ACTION_PROBABILITY", 0.0) or 0.0)
    random_init = bool(task.get("RANDOM_INIT", False))
    sticky = rap > 0.0
    if (not sticky) and (not random_init):
        return "frameskip"
    if sticky and (not random_init):
        return "frameskip+sticky"
    if (not sticky) and random_init:
        return "frameskip+noop"
    return "frameskip+sticky+noop"


def _variant_sort_key(label: str) -> int:
    order = {
        "frameskip": 0,
        "frameskip+sticky": 1,
        "frameskip+noop": 2,
        "frameskip+sticky+noop": 3,
    }
    return order.get(label, 999)


def _extract_curve(plot_data: list[list[float]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """
    plot_data rows are [generation, best, avg] (from scripts/single_run.py).
    """
    arr = np.asarray(plot_data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected plot_data shape: {arr.shape}")
    x = arr[:, 0]
    if metric == "best":
        y = arr[:, 1]
    elif metric == "avg":
        y = arr[:, 2]
    else:
        raise ValueError("--metric must be one of: best, avg")
    return x, y


def get_args():
    p = argparse.ArgumentParser(description="Plot nondeterministic experiment runs from a SQLite DB.")
    p.add_argument("--db", default=_default_db(), help="Path to input SQLite DB.")
    p.add_argument("--out-dir", default=_default_out_dir(), help="Directory to write plots into.")
    p.add_argument("--metric", choices=["best", "avg"], default="best", help="Which curve to plot from plot_data_json.")
    p.add_argument("--title", default=None, help="Optional title prefix.")
    p.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return p.parse_args()


def main():
    args = get_args()
    db_path = os.path.abspath(args.db)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    runs = _load_runs(db_path)
    if not runs:
        raise ValueError(f"No runs found in DB: {db_path}")

    # Delay-import matplotlib so the script still errors cleanly if it isn't installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group runs into the 4 variants.
    groups: dict[str, list[Run]] = {}
    for r in runs:
        label = _variant_label(r.task)
        groups.setdefault(label, []).append(r)

    # Infer some common metadata for titles.
    env_name = runs[0].env_name
    frameskip = runs[0].task.get("FRAMESKIP", None)
    title_prefix = args.title if args.title is not None else env_name

    # Per-variant plots: all runs + mean.
    mean_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in sorted(groups.keys(), key=_variant_sort_key):
        rs = groups[label]
        if not rs:
            continue

        xs, ys = [], []
        for r in rs:
            x, y = _extract_curve(r.plot_data, metric=args.metric)
            xs.append(x)
            ys.append(y)

        x0 = xs[0]
        y_stack = np.vstack([y for y in ys])
        y_mean = np.mean(y_stack, axis=0)
        mean_curves[label] = (x0, y_mean)

        plt.figure(figsize=(10, 6))
        for y in ys:
            plt.plot(x0, y, color="C0", alpha=0.22, linewidth=1.0)
        plt.plot(x0, y_mean, color="C1", linewidth=2.5, label="mean")
        plt.xlabel("generation")
        plt.ylabel(args.metric)
        fs = f", frameskip={frameskip}" if frameskip is not None else ""
        plt.title(f"{title_prefix}{fs} — {label} ({len(rs)} runs)")
        plt.legend(loc="best")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{label}_{args.metric}.png")
        plt.savefig(out_path, dpi=int(args.dpi))
        plt.close()

    # Combined plot of mean curves.
    plt.figure(figsize=(10, 6))
    for label in sorted(mean_curves.keys(), key=_variant_sort_key):
        x, y = mean_curves[label]
        plt.plot(x, y, linewidth=2.5, label=label)
    plt.xlabel("generation")
    plt.ylabel(args.metric)
    fs = f", frameskip={frameskip}" if frameskip is not None else ""
    plt.title(f"{title_prefix}{fs} — mean curves (nondeterminism)")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"means_{args.metric}.png")
    plt.savefig(out_path, dpi=int(args.dpi))
    plt.close()

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()


