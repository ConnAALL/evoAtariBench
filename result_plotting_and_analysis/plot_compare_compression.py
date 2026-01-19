"""
Compare two (or more) (compression, nonlinearity) configurations head-to-head, plotted
in the same grid style as `plot_game_sweep.py`.

This is a **simple, no-CLI** plotting utility:
- Edit the CONFIG section at the top of this file.
- Run: `python3 result_plotting_and_analysis/plot_compare_compression.py`

Outputs (two PNGs):
- avg_of_avgs: mean of the runs' `avg` curves (per game, per label)
- best_overall: best-so-far curve of the single best run (per game, per label)
- top5_best_mean: mean of the top-5 runs' `best` values at each generation (per game, per label)
"""

from __future__ import annotations

import csv
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
import matplotlib.pyplot as plt  # noqa: E402

try:  # Optional: match repo plotting style if available.
    import scienceplots  # type: ignore  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    pass

###############################################################################
# CONFIG (edit these values; no command line args)
###############################################################################

# Each entry is "comp:nonlinearity". Repeat to compare multiple configurations.
# Notes:
# - For dft subset DBs, methods may be stored with suffix `_complex` and this script
#   allows prefix matching (e.g. "sparsification" -> "sparsification_complex").
PAIRS: list[str] = [
    "dct:sparsification",
    "none:sparsification",
    "dct_k25:sparsification",
    "dct_k1:sparsification",
]

# Optional per-label task filters.
# Use this when a DB contains runs that should be separated into multiple curves,
# e.g. DCT with different k values. Keys are the *comp* strings used in `PAIRS`.
# Example:
#   PAIRS = ["dct:sparsification", "none:sparsification", "dct_k25:sparsification"]
#   TASK_FILTERS = {"dct_k25": {"k": 25}}
TASK_FILTERS: dict[str, dict[str, Any]] = {"dct_k25": {"k": 25}, "dct_k1": {"k": 1}}

# Optional data overrides: comp -> absolute path to directory or single .db file.
# If empty, uses repo defaults under data/run_data/.
RUN_PATH_OVERRIDE: dict[str, str] = {
    "dct": "/home/CS_data/students/dgezgin/evoAtariBench/data/run_data/dct_none_comparison_1000/dct_none_comparison_1000.db",
    "none": "/home/CS_data/students/dgezgin/evoAtariBench/data/run_data/dct_none_comparison_1000/dct_none_comparison_1000.db",
    "dct_k25": "/home/CS_data/students/dgezgin/evoAtariBench/data/run_data/dct_k25_1000/dct_k25_1000.db",
    "dct_k1": "/home/CS_data/students/dgezgin/evoAtariBench/data/run_data/overfit_test/overfit_test.db",
}

# Output directory:
# - set to None to use the repo default: out/plots/compare_compression/subset_runs/
# - or set to an absolute path.
OUT_DIR: str | None = None

# Plot formatting.
COLS: int = 3
LIMIT_GAMES: int | None = None
DPI: int = 260
TITLE: str | None = None  # if None, uses "compare_compression"

# For the `top5_best_mean` metric: take the top-K (by best fitness) at each generation and average.
TOPK_BEST_MEAN: int = 5

# Optional: fix game list/order using a CSV that has an `environment` column.
GAMES_CSV: str | None = None


def _repo_root() -> str:
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


def _infer_method_from_task(task: dict[str, Any]) -> str | None:
    if not isinstance(task, dict) or not task:
        return None
    preferred_keys = [
        "nonlinearity",
        "NONLINEARITY",
        "nonlinearity_method",
        "NONLINEARITY_METHOD",
        "method",
        "METHOD",
        "activation",
        "ACTIVATION",
    ]
    for k in preferred_keys:
        if k in task:
            v = task.get(k)
            if isinstance(v, (str, int, float, bool)):
                return _safe_slug(v)
    for k, v in task.items():
        kl = str(k).lower()
        if "method" in kl or "nonlin" in kl or "activation" in kl:
            if isinstance(v, (str, int, float, bool)):
                return _safe_slug(v)
    return None


def _infer_compression_from_task(task: dict[str, Any]) -> str | None:
    """
    Some DBs contain multiple compression settings in the same file; in that case we
    must filter runs by `task_json["compression"]` (or similar keys) in addition to
    selecting the nonlinearity method.
    """
    if not isinstance(task, dict) or not task:
        return None
    preferred_keys = [
        "compression",
        "COMPRESSION",
        "compression_method",
        "COMPRESSION_METHOD",
        "comp",
        "COMP",
        "encoder",
        "ENCODER",
        "transform",
        "TRANSFORM",
        "representation",
        "REPRESENTATION",
    ]
    for k in preferred_keys:
        if k in task:
            v = task.get(k)
            if isinstance(v, (str, int, float, bool)):
                return _safe_slug(v)
    for k, v in task.items():
        kl = str(k).lower()
        if "compress" in kl or "encoder" in kl or "transform" in kl or "representation" in kl:
            if isinstance(v, (str, int, float, bool)):
                return _safe_slug(v)
    return None


def _base_comp(comp: str) -> str:
    """
    Allow alias labels like `dct_k25` while still filtering compression as `dct`.
    """
    comp = _safe_slug(comp)
    if "_" in comp:
        return comp.split("_", 1)[0]
    return comp


def _task_matches_filters(task: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """
    Exact-match filters on task_json fields after slugging both sides where possible.
    """
    if not filters:
        return True
    if not isinstance(task, dict):
        return False
    for k, expected in filters.items():
        if k not in task:
            return False
        actual = task.get(k)
        # Compare slugged strings/bools/nums in a forgiving way.
        if isinstance(expected, (str, int, float, bool)) and isinstance(actual, (str, int, float, bool)):
            if _safe_slug(actual) != _safe_slug(expected):
                return False
        else:
            if actual != expected:
                return False
    return True


def _method_from_db_filename(comp: str, filename: str) -> str:
    base = os.path.basename(filename)
    if base.lower().endswith(".db"):
        base = base[:-3]
    prefix = f"{str(comp).lower()}_"
    if base.lower().startswith(prefix):
        base = base[len(prefix) :]
    base = re.sub(r"_\d+gen$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_gen$", "", base, flags=re.IGNORECASE)
    return _safe_slug(base)


def _find_db_paths(run_path: str, comp: str) -> list[str]:
    run_path = os.path.abspath(run_path)
    if os.path.isfile(run_path):
        return [run_path] if run_path.lower().endswith(".db") else []
    if not os.path.isdir(run_path):
        return []
    comp_l = str(comp).lower()
    preferred = [
        os.path.join(run_path, fn)
        for fn in os.listdir(run_path)
        if fn.lower().endswith(".db") and fn.lower().startswith(f"{comp_l}_")
    ]
    if preferred:
        return sorted(preferred)
    return sorted([os.path.join(run_path, fn) for fn in os.listdir(run_path) if fn.lower().endswith(".db")])


def _default_run_path(comp: str) -> str:
    root = os.path.join(_repo_root(), "data", "run_data")
    # Prefer subset-runs by default (the 5-game tests), because that's the intended use-case
    # for compression comparisons in this repo.
    cand_subset_dir = os.path.join(root, f"{comp}_subset_runs")
    if os.path.isdir(cand_subset_dir):
        return cand_subset_dir

    # Fallback: full game sweep directory if it exists.
    cand_sweep_dir = os.path.join(root, f"{comp}_game_sweep")
    if os.path.isdir(cand_sweep_dir):
        return cand_sweep_dir

    # Final fallback: keep the sweep naming (may not exist; caller will error cleanly).
    return cand_subset_dir


def _default_out_dir() -> str:
    return os.path.join(_repo_root(), "out", "plots", "compare_compression", "subset_runs")


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
    curves = list(curves)
    if not curves:
        raise ValueError("No curves to align")
    min_len = min(int(len(y)) for _x, y in curves)
    x0 = curves[0][0][:min_len]
    ys = [y[:min_len] for _x, y in curves]
    y_stack = np.vstack(ys)
    mean_y = np.mean(y_stack, axis=0)
    return x0, ys, mean_y


def _align_and_topk_mean(
    curves: Iterable[tuple[np.ndarray, np.ndarray]], k: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    curves = list(curves)
    if not curves:
        raise ValueError("No curves to align")
    k = max(1, int(k))
    min_len = min(int(len(y)) for _x, y in curves)
    x0 = curves[0][0][:min_len]
    ys = [y[:min_len] for _x, y in curves]
    y_stack = np.vstack(ys)  # (n_runs, T)
    kk = min(k, int(y_stack.shape[0]))
    topk = np.sort(y_stack, axis=0)[-kk:, :]  # largest kk per timestep
    topk_mean_y = np.mean(topk, axis=0)
    return x0, ys, topk_mean_y


def _topk_runs_by_final_best(rs: list[Run], k: int) -> list[Run]:
    k = max(1, int(k))
    scored: list[tuple[float, Run]] = []
    for r in rs:
        try:
            _x, y_best = _extract_curve(r.plot_data, metric="best")
            if len(y_best) == 0:
                continue
            scored.append((float(y_best[-1]), r))
        except Exception:
            continue
    if not scored:
        return []
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _s, r in scored[: min(k, len(scored))]]


def _best_run_by_peak_best(rs: list[Run]) -> Run | None:
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


def _load_envs_from_csv(csv_path: str) -> list[str]:
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


def _parse_pair(s: str) -> tuple[str, str]:
    if ":" not in str(s):
        raise SystemExit(f"--pair must be 'comp:nonlinearity', got: {s!r}")
    comp, method = s.split(":", 1)
    comp = _safe_slug(comp)
    method = _safe_slug(method)
    if not comp or not method:
        raise SystemExit(f"Bad --pair: {s!r}")
    return comp, method


def _resolve_method(requested: str, available: set[str]) -> str:
    """
    Resolve requested method name to an available method label.

    - Prefer exact match
    - Else allow prefix match (useful for dft_*_complex names when user passes 'sparsification')
    """
    requested = _safe_slug(requested)
    if requested in available:
        return requested
    pref = sorted([m for m in available if m.startswith(requested)])
    if len(pref) == 1:
        return pref[0]
    if len(pref) > 1:
        raise SystemExit(f"Method {requested!r} is ambiguous; matches: {pref}. Please pass exact name.")
    raise SystemExit(f"Method {requested!r} not found. Available: {sorted(available)}")


def main() -> None:
    pairs = [_parse_pair(s) for s in (PAIRS or [])]
    if not pairs:
        raise SystemExit("CONFIG error: PAIRS is empty. Add at least one entry like 'dct:sparsification'.")

    override = {_safe_slug(k): os.path.abspath(v) for k, v in (RUN_PATH_OVERRIDE or {}).items()}
    task_filters = {_safe_slug(k): v for k, v in (TASK_FILTERS or {}).items()}

    out_dir = os.path.abspath(OUT_DIR) if OUT_DIR else os.path.abspath(_default_out_dir())
    os.makedirs(out_dir, exist_ok=True)

    # Load data per pair: label -> env -> runs
    by_label_env: dict[str, dict[str, list[Run]]] = {}
    labels: list[str] = []
    all_envs: set[str] = set()

    for comp, requested_method in pairs:
        run_path = override.get(comp) or _default_run_path(comp)
        if not os.path.exists(run_path):
            raise SystemExit(f"Run path not found for comp={comp}: {run_path}")

        db_paths = _find_db_paths(run_path, comp=comp)
        if not db_paths:
            raise SystemExit(f"No .db files found for comp={comp} in: {run_path}")

        # First pass: collect all runs and available method labels.
        runs_all: list[tuple[str, Run]] = []  # (method_label, run)
        available: set[str] = set()
        for db in db_paths:
            file_method = _method_from_db_filename(comp, db)
            rs = _load_runs(db)
            for r in rs:
                # If the DB stores multiple compressions together, filter by task_json.
                # If the key is missing (older/single-comp DBs), keep the run.
                comp_in_task = _infer_compression_from_task(r.task)
                if comp_in_task is not None and _safe_slug(comp_in_task) != _safe_slug(_base_comp(comp)):
                    continue
                if not _task_matches_filters(r.task, task_filters.get(comp)):
                    continue
                m = _infer_method_from_task(r.task) or file_method
                available.add(m)
                runs_all.append((m, r))

        chosen_method = _resolve_method(requested_method, available)

        label = f"{comp}+{chosen_method}"
        labels.append(label)
        env_map: dict[str, list[Run]] = {}
        for m, r in runs_all:
            if m != chosen_method:
                continue
            env_map.setdefault(r.env_name, []).append(r)
        by_label_env[label] = env_map
        all_envs.update(env_map.keys())

        print(f"[load] {label}: dbs={len(db_paths)} envs={len(env_map)} runs={sum(len(v) for v in env_map.values())}")

    # Determine env list: either from CSV, or union across labels.
    if GAMES_CSV:
        envs = _load_envs_from_csv(os.path.abspath(GAMES_CSV))
    else:
        envs = sorted(all_envs)

    if LIMIT_GAMES is not None:
        envs = envs[: max(0, int(LIMIT_GAMES))]

    if not envs:
        raise SystemExit("No games/envs to plot.")

    cols = max(1, int(COLS))
    n_rows = int(math.ceil(len(envs) / float(cols)))

    # Small per-subplot footprint; the overall PNG grows with number of games.
    cell_w, cell_h = 2.05, 1.55
    fig_w, fig_h = cols * cell_w, n_rows * cell_h

    title_prefix = TITLE if TITLE is not None else "compare_compression"
    colors = {lab: f"C{i}" for i, lab in enumerate(labels)}

    for metric in ("avg_of_avgs", "best_overall", "top5_best_mean"):
        fig, axes = plt.subplots(n_rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        if metric == "avg_of_avgs":
            fig.suptitle(f"{title_prefix} — avg of avgs", fontsize=12)
        elif metric == "best_overall":
            fig.suptitle(f"{title_prefix} — best overall", fontsize=12)
        else:  # top5_best_mean
            fig.suptitle(f"{title_prefix} — top-{int(TOPK_BEST_MEAN)} mean (best)", fontsize=12)

        handle_by_label = {}

        for i, env in enumerate(envs):
            ax = axes[i // cols][i % cols]
            plotted_any = False

            for lab in labels:
                rs = by_label_env.get(lab, {}).get(env, [])
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
                elif metric == "avg_of_avgs":
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
                else:  # top5_best_mean
                    curves = []
                    for r in rs:
                        try:
                            curves.append(_extract_curve(r.plot_data, metric="best"))
                        except Exception:
                            continue
                    if not curves:
                        continue
                    try:
                        x, _ys, y = _align_and_topk_mean(curves, k=int(TOPK_BEST_MEAN))
                    except Exception:
                        continue

                (line,) = ax.plot(x, y, color=colors[lab], linewidth=1.25, alpha=0.95)
                plotted_any = True
                if lab not in handle_by_label:
                    handle_by_label[lab] = line

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

        # Turn off unused axes.
        for j in range(len(envs), n_rows * cols):
            axes[j // cols][j % cols].set_axis_off()

        if handle_by_label:
            legend_labels = list(handle_by_label.keys())
            legend_handles = [handle_by_label[k] for k in legend_labels]
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

        slug = "__".join([_safe_slug(l) for l in labels])
        if len(slug) > 120:
            slug = "pairs"
        out_path = os.path.join(out_dir, f"compare_{slug}_{metric}_grid.png")
        fig.savefig(out_path, dpi=int(DPI))
        plt.close(fig)
        print(f"[wrote] {out_path}")

    print(f"Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()


