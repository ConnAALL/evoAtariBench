"""
Statistical tests / uncertainty estimates for EvoAtari sweep-style runs stored in SQLite.

Supports BOTH layouts:
- Directory of DBs (one DB per method; method inferred from filename like `dct_dropout_250gen.db`)
- Single DB containing multiple methods encoded in `runs.task_json` (e.g. `task_json["nonlinearity"]`)

Included methods (as requested):
✅ Option B: Mixed-effects model (statsmodels; optional dependency)
    score ~ C(nonlinearity) + (1 | game)
    Optionally add a seed variance component (approx. (1 | seed)) if `--include-seed-re` is passed.

✅ Option C: Bootstrap confidence intervals (no extra deps; default)
    Two-level bootstrap:
      - resample seeds within each (game, method)
      - aggregate per-game means equally across games
    Produces 95% CI for pairwise method differences Δ = mean(method1) - mean(method2).

✅ Option ca2: Bayesian hierarchical model (PyMC; optional dependency)
    score[g,m,i] ~ Normal(mu_method[m] + u_game[g], sigma)
    u_game[g] ~ Normal(0, sigma_game)
    Reports posterior win-probabilities like P(method A > method B).

Example (single DB with multiple methods):
  python3 result_plotting_and_analysis/stat_test_game_sweep.py \
    --run /path/to/dct_subset_runs.db \
    --score best_fitness

Example (directory of per-method DBs):
  python3 result_plotting_and_analysis/stat_test_game_sweep.py \
    --run /path/to/data/run_data/dct_game_sweep/ \
    --comp dct
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


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
    s = re.sub(r"-v\\d+$", "", s, flags=re.IGNORECASE)
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


def _method_from_db_filename(comp: str | None, filename: str) -> str:
    base = os.path.basename(filename)
    if base.lower().endswith(".db"):
        base = base[:-3]
    if comp:
        prefix = f"{str(comp).lower()}_"
        if base.lower().startswith(prefix):
            base = base[len(prefix) :]
    base = re.sub(r"_\\d+gen$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_gen$", "", base, flags=re.IGNORECASE)
    return _safe_slug(base)


def _find_db_paths(run_path: str, comp: str | None) -> list[str]:
    run_path = os.path.abspath(run_path)
    if os.path.isfile(run_path):
        return [run_path] if run_path.lower().endswith(".db") else []
    if not os.path.isdir(run_path):
        return []
    fns = [fn for fn in os.listdir(run_path) if fn.lower().endswith(".db")]
    if not fns:
        return []
    if comp:
        comp_l = str(comp).lower()
        preferred = [fn for fn in fns if fn.lower().startswith(f"{comp_l}_")]
        if preferred:
            return sorted([os.path.join(run_path, fn) for fn in preferred])
    return sorted([os.path.join(run_path, fn) for fn in fns])


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    cur = con.cursor()
    cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(c[1]) for c in cols}


def _extract_curve(plot_data: list[list[float]], which: str) -> np.ndarray:
    """
    plot_data rows are [generation, best, avg] (from scripts/single_run.py).
    Returns a y-array.
    """
    arr = np.asarray(plot_data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected plot_data shape: {arr.shape}")
    if which == "best":
        return arr[:, 1]
    if which == "avg":
        return arr[:, 2]
    raise ValueError("which must be 'best' or 'avg'")


def _score_from_row(score_mode: str, best_fitness: float | None, plot_data: list[list[float]] | None) -> float | None:
    """
    score_mode:
      - best_fitness: uses runs.best_fitness (recommended when present)
      - peak_best: max(best over gens)
      - final_best: last(best)
      - final_avg: last(avg)
    """
    if score_mode == "best_fitness":
        return None if best_fitness is None else float(best_fitness)
    if plot_data is None:
        return None
    try:
        if score_mode == "peak_best":
            y = _extract_curve(plot_data, "best")
            return float(np.max(y))
        if score_mode == "final_best":
            y = _extract_curve(plot_data, "best")
            return float(y[-1])
        if score_mode == "final_avg":
            y = _extract_curve(plot_data, "avg")
            return float(y[-1])
    except Exception:
        return None
    raise ValueError(f"Unknown --score: {score_mode}")


@dataclass(frozen=True)
class Row:
    run_id: int
    env_name: str
    game: str
    method: str
    seed: str
    score: float


def _normalize_rows(rows: list[Row], mode: str) -> list[Row]:
    """
    Optional normalization across games.

    NOTE: Raw Atari scores have very different scales per game. Normalization can be helpful for
    cross-game aggregation / modeling, but it changes the question being asked.
    """
    mode = str(mode or "none").lower()
    if mode == "none":
        return rows
    if mode != "per_game_z":
        raise SystemExit(f"Unknown --normalize: {mode} (expected: none, per_game_z)")

    by_game: dict[str, list[float]] = {}
    for r in rows:
        by_game.setdefault(r.game, []).append(r.score)

    mu = {g: float(np.mean(v)) for g, v in by_game.items()}
    sd = {g: float(np.std(v, ddof=0)) for g, v in by_game.items()}

    out: list[Row] = []
    for r in rows:
        s = sd.get(r.game, 0.0)
        z = 0.0 if s <= 0.0 else (r.score - mu[r.game]) / s
        out.append(Row(r.run_id, r.env_name, r.game, r.method, r.seed, float(z)))
    return out


def _load_rows_from_db(db_path: str, comp: str | None, score_mode: str) -> list[Row]:
    con = sqlite3.connect(db_path, timeout=30.0)
    try:
        cols = _table_columns(con, "runs")
        need_best = score_mode == "best_fitness"
        need_plot = score_mode in {"peak_best", "final_best", "final_avg"}
        if need_best and "best_fitness" not in cols:
            raise SystemExit(f"--score best_fitness requested but runs.best_fitness missing in: {db_path}")
        if need_plot and "plot_data_json" not in cols:
            raise SystemExit(f"--score {score_mode} requested but runs.plot_data_json missing in: {db_path}")

        select_cols = ["run_id", "env_name", "task_json"]
        if "best_fitness" in cols:
            select_cols.append("best_fitness")
        else:
            select_cols.append("NULL as best_fitness")
        if "plot_data_json" in cols:
            select_cols.append("plot_data_json")
        else:
            select_cols.append("NULL as plot_data_json")

        q = (
            "SELECT "
            + ", ".join(select_cols)
            + " FROM runs WHERE env_name IS NOT NULL AND task_json IS NOT NULL"
        )

        file_method = _method_from_db_filename(comp, db_path)
        cur = con.cursor()
        rows = cur.execute(q).fetchall()

        # Determine whether this DB contains multiple task-inferred methods.
        inferred = []
        parsed_tasks: list[dict[str, Any]] = []
        for _run_id, _env_name, task_json, _best_fitness, _plot_data_json in rows:
            try:
                task = json.loads(task_json) if task_json is not None else {}
            except Exception:
                task = {}
            if not isinstance(task, dict):
                task = {}
            parsed_tasks.append(task)
            m = _infer_method_from_task(task)
            if m:
                inferred.append(m)
        inferred_unique = sorted(set(inferred))
        use_task_method = bool(inferred_unique) and len(inferred_unique) > 1

        out: list[Row] = []
        for (run_id, env_name, _task_json, best_fitness, plot_data_json), task in zip(rows, parsed_tasks, strict=False):
            env_name = str(env_name)
            game = _env_to_game(env_name)

            method = _infer_method_from_task(task) if use_task_method else None
            method = method or file_method

            # Seed: this DB schema doesn't guarantee a seed field; we use run_id (string) as a stable proxy.
            seed = str(run_id)

            plot_data = None
            if plot_data_json is not None:
                try:
                    plot_data = json.loads(plot_data_json)
                except Exception:
                    plot_data = None

            score = _score_from_row(score_mode, best_fitness, plot_data)
            if score is None or not np.isfinite(score):
                continue

            out.append(Row(int(run_id), env_name, game, method, seed, float(score)))

        return out
    finally:
        con.close()


def _bootstrap_pairwise(
    scores_by_game_method: dict[str, dict[str, np.ndarray]],
    methods: list[str],
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    games_all = sorted(scores_by_game_method.keys())

    # Only include games where all requested methods exist (for fair pairwise comparisons).
    games = [g for g in games_all if all(m in scores_by_game_method[g] for m in methods)]
    if not games:
        raise SystemExit("No games have complete method coverage; cannot bootstrap pairwise differences.")

    # Pre-pack arrays for speed
    arr = {(g, m): scores_by_game_method[g][m] for g in games for m in methods}
    for (g, m), a in arr.items():
        if a.size < 2:
            raise SystemExit(f"Not enough seeds for bootstrap in game={g} method={m} (n={a.size}).")

    pairs = list(itertools.combinations(methods, 2))
    deltas = {pair: np.empty(n_boot, dtype=np.float64) for pair in pairs}

    # Observed aggregate (equal weight per game)
    obs_mean = {m: float(np.mean([float(np.mean(arr[(g, m)])) for g in games])) for m in methods}

    for b in range(n_boot):
        boot_mean = {}
        for m in methods:
            per_game_means = []
            for g in games:
                a = arr[(g, m)]
                idx = rng.integers(0, a.size, size=a.size)
                per_game_means.append(float(np.mean(a[idx])))
            boot_mean[m] = float(np.mean(per_game_means))
        for m1, m2 in pairs:
            deltas[(m1, m2)][b] = boot_mean[m1] - boot_mean[m2]

    results: list[dict[str, Any]] = []
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    for m1, m2 in pairs:
        d = deltas[(m1, m2)]
        lo = float(np.percentile(d, lo_q))
        hi = float(np.percentile(d, hi_q))
        obs = float(obs_mean[m1] - obs_mean[m2])
        # Two-sided bootstrap p-value proxy: mass on opposite side of 0
        p = 2.0 * min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0)))
        results.append(
            {
                "method_1": m1,
                "method_2": m2,
                "delta_obs": obs,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_two_sided": min(1.0, p),
                "n_boot": int(n_boot),
                "n_games": int(len(games)),
            }
        )
    return results


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _try_fit_mixedlm(df_rows: list[Row], include_seed_re: bool) -> str:
    """
    Returns a human-readable summary string, or raises SystemExit if deps missing.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"MixedLM requested but pandas is not installed: {e}")

    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"MixedLM requested but statsmodels is not installed: {e}")

    df = pd.DataFrame(
        {
            "score": [r.score for r in df_rows],
            "nonlinearity": [r.method for r in df_rows],
            "game": [r.game for r in df_rows],
            "seed": [r.seed for r in df_rows],
        }
    )

    # Random intercept by game; optional seed variance component.
    if include_seed_re:
        model = smf.mixedlm(
            "score ~ C(nonlinearity)",
            data=df,
            groups=df["game"],
            vc_formula={"seed": "0 + C(seed)"},
        )
    else:
        model = smf.mixedlm(
            "score ~ C(nonlinearity)",
            data=df,
            groups=df["game"],
        )

    try:
        res = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    except Exception:
        res = model.fit(reml=False)

    return str(res.summary())


def _try_fit_bayesian_hierarchical(
    rows: list[Row],
    methods: list[str],
    games: list[str],
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    progressbar: bool,
) -> tuple[list[dict[str, Any]], str]:
    """
    Bayesian hierarchical model (PyMC):
      score ~ Normal(mu_method[method] + u_game[game], sigma)
      u_game ~ Normal(0, sigma_game)

    Returns:
      - list of pairwise method comparison dicts
      - a short human-readable summary string
    """
    try:
        import pymc as pm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Bayesian model requested but pymc is not installed: {e}")

    method_to_i = {m: i for i, m in enumerate(methods)}
    game_to_i = {g: i for i, g in enumerate(games)}

    y = np.asarray([r.score for r in rows], dtype=np.float64)
    method_idx = np.asarray([method_to_i[r.method] for r in rows], dtype=np.int32)
    game_idx = np.asarray([game_to_i[r.game] for r in rows], dtype=np.int32)

    # Scale-aware priors
    y_sd = float(np.std(y, ddof=0))
    y_sd = y_sd if y_sd > 0 else 1.0

    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=y_sd)
        sigma_game = pm.HalfNormal("sigma_game", sigma=y_sd)

        # Global method effects and game random intercepts
        mu_method = pm.Normal("mu_method", mu=0.0, sigma=2.0 * y_sd, shape=len(methods))
        u_game = pm.Normal("u_game", mu=0.0, sigma=sigma_game, shape=len(games))

        mu = mu_method[method_idx] + u_game[game_idx]
        pm.Normal("score", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            target_accept=float(target_accept),
            random_seed=int(seed),
            progressbar=bool(progressbar),
        )

    # Extract posterior samples for mu_method
    posterior = idata.posterior["mu_method"].values  # (chains, draws, n_methods)
    samples = posterior.reshape(-1, posterior.shape[-1])  # (n_samples, n_methods)

    pairwise: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(methods)), 2):
        delta = samples[:, i] - samples[:, j]
        prob = float(np.mean(delta > 0.0))
        lo = float(np.quantile(delta, 0.025))
        hi = float(np.quantile(delta, 0.975))
        pairwise.append(
            {
                "method_1": methods[i],
                "method_2": methods[j],
                "delta_mean": float(np.mean(delta)),
                "ci_lo": lo,
                "ci_hi": hi,
                "p_method1_gt_method2": prob,
                "n_samples": int(delta.size),
            }
        )

    # Also compute "best method" probability under mu_method (ignoring u_game since it's shared across methods)
    best_idx = np.argmax(samples, axis=1)
    best_probs = {methods[k]: float(np.mean(best_idx == k)) for k in range(len(methods))}
    best_line = ", ".join([f"{m}={best_probs[m]:.3f}" for m in methods])
    summary = f"Posterior P(best method by mu_method): {best_line}"

    return pairwise, summary


def _get_args():
    p = argparse.ArgumentParser(description="Statistical tests for sweep/subset EvoAtari runs (SQLite).")
    p.add_argument(
        "--run",
        required=True,
        help="Path to a directory of .db files, or a single .db file.",
    )
    p.add_argument(
        "--comp",
        default=None,
        help="Optional compression prefix used to parse method names from filenames (e.g. 'dct').",
    )
    p.add_argument(
        "--score",
        default="best_fitness",
        choices=["best_fitness", "peak_best", "final_best", "final_avg"],
        help="What score to use per run.",
    )
    p.add_argument(
        "--normalize",
        default="none",
        choices=["none", "per_game_z"],
        help="Optional normalization of scores across games.",
    )
    p.add_argument("--out-dir", default=None, help="Directory to write outputs (CSV report).")

    boot_g = p.add_mutually_exclusive_group()
    boot_g.add_argument("--bootstrap", action="store_true", help="(legacy) Run bootstrap CI analysis (Option C).")
    boot_g.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap CI analysis.")

    p.add_argument("--n-bootstrap", type=int, default=100000, help="Bootstrap iterations.")
    p.add_argument("--alpha", type=float, default=0.05, help="Alpha for CI; 0.05 -> 95% CI.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap.")

    mix_g = p.add_mutually_exclusive_group()
    mix_g.add_argument("--mixedlm", action="store_true", help="(legacy) Fit MixedLM (Option B) using statsmodels.")
    mix_g.add_argument("--no-mixedlm", action="store_true", help="Disable MixedLM (statsmodels).")
    p.add_argument(
        "--include-seed-re",
        action="store_true",
        help="For MixedLM, add a seed variance component (approx. (1|seed)).",
    )

    bayes_g = p.add_mutually_exclusive_group()
    bayes_g.add_argument("--bayes", action="store_true", help="(legacy) Fit Bayesian hierarchical model (PyMC).")
    bayes_g.add_argument("--no-bayes", action="store_true", help="Disable Bayesian hierarchical model (PyMC).")
    p.add_argument("--bayes-draws", type=int, default=1000, help="PyMC posterior draws.")
    p.add_argument("--bayes-tune", type=int, default=1000, help="PyMC tuning steps.")
    p.add_argument("--bayes-chains", type=int, default=2, help="PyMC chains.")
    p.add_argument("--bayes-target-accept", type=float, default=0.9, help="PyMC target_accept.")
    p.add_argument("--bayes-progressbar", action="store_true", help="Show PyMC progress bar.")
    return p.parse_args()


def main() -> None:
    args = _get_args()
    run_path = os.path.abspath(args.run)
    comp = args.comp
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.join(_repo_root(), "out", "stats", "game_sweep")
    )
    os.makedirs(out_dir, exist_ok=True)

    db_paths = _find_db_paths(run_path, comp=comp)
    if not db_paths:
        raise SystemExit(f"No .db files found in: {run_path}")

    rows: list[Row] = []
    for db in db_paths:
        rows.extend(_load_rows_from_db(db, comp=comp, score_mode=str(args.score)))

    if not rows:
        raise SystemExit("No usable rows loaded (check --score and DB contents).")

    rows = _normalize_rows(rows, mode=str(args.normalize))

    games = sorted(set(r.game for r in rows))
    methods = sorted(set(r.method for r in rows))
    print(f"Loaded rows: {len(rows)} | games: {len(games)} | methods: {len(methods)} | dbs: {len(db_paths)}")
    print("Methods:", ", ".join(methods))
    print("Games:", ", ".join(games))
    if str(args.normalize).lower() != "none":
        print(f"Normalize: {args.normalize}")

    # Group scores by game/method
    scores_by_game_method: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        scores_by_game_method.setdefault(r.game, {}).setdefault(r.method, []).append(r.score)

    # Print a small coverage table
    print("\nPer-game coverage (n seeds per method):")
    for g in games:
        parts = []
        for m in methods:
            n = len(scores_by_game_method.get(g, {}).get(m, []))
            parts.append(f"{m}={n}")
        print(f"  {g}: " + ", ".join(parts))

    # Run ALL tests by default. Use --no-* flags to disable individual analyses.
    do_bootstrap = not bool(getattr(args, "no_bootstrap", False))
    do_mixedlm = not bool(getattr(args, "no_mixedlm", False))
    do_bayes = not bool(getattr(args, "no_bayes", False))

    if do_bootstrap:
        rng = np.random.default_rng(int(args.seed))
        packed = {g: {m: np.asarray(v, dtype=np.float64) for m, v in mm.items()} for g, mm in scores_by_game_method.items()}
        boot = _bootstrap_pairwise(
            packed,
            methods=methods,
            n_boot=int(args.n_bootstrap),
            alpha=float(args.alpha),
            rng=rng,
        )
        print("\nBootstrap pairwise differences (equal-weight across games):")
        for r in boot:
            m1, m2 = r["method_1"], r["method_2"]
            print(
                f"  Δ({m1} - {m2}) = {r['delta_obs']:.3f} "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  p≈{r['p_two_sided']:.4f}"
            )

        out_csv = os.path.join(out_dir, "bootstrap_pairwise.csv")
        _write_csv(out_csv, boot)
        print(f"\n[wrote] {out_csv}")

    if do_mixedlm:
        print("\nMixedLM (statsmodels) summary:")
        try:
            summary = _try_fit_mixedlm(rows, include_seed_re=bool(args.include_seed_re))
        except SystemExit as e:
            print(f"[skip] MixedLM: {e}")
            summary = None
        if summary:
            print(summary)
            out_txt = os.path.join(out_dir, "mixedlm_summary.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(summary)
                f.write("\n")
            print(f"[wrote] {out_txt}")

    if do_bayes:
        print("\nBayesian hierarchical model (PyMC) summary:")
        try:
            bayes_rows, bayes_summary = _try_fit_bayesian_hierarchical(
                rows=rows,
                methods=methods,
                games=games,
                draws=int(args.bayes_draws),
                tune=int(args.bayes_tune),
                chains=int(args.bayes_chains),
                target_accept=float(args.bayes_target_accept),
                seed=int(args.seed),
                progressbar=bool(args.bayes_progressbar),
            )
        except SystemExit as e:
            print(f"[skip] Bayesian: {e}")
            bayes_rows, bayes_summary = [], ""

        if bayes_rows:
            print(bayes_summary)
            for r in bayes_rows:
                m1, m2 = r["method_1"], r["method_2"]
                print(
                    f"  P({m1} > {m2}) = {r['p_method1_gt_method2']:.3f}  "
                    f"Δ={r['delta_mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                )
            out_csv = os.path.join(out_dir, "bayes_pairwise.csv")
            _write_csv(out_csv, bayes_rows)
            print(f"\n[wrote] {out_csv}")
            out_txt = os.path.join(out_dir, "bayes_summary.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(bayes_summary)
                f.write("\n")
            print(f"[wrote] {out_txt}")


if __name__ == "__main__":
    main()


