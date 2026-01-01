import json
import os
import re
import sqlite3
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "no-latex"])

class RunSeries:
    def __init__(self, row_id, run_id, env_name, compression, nonlinearity, signature, gens, vals):
        self.row_id = int(row_id)
        self.run_id = int(run_id)
        self.env_name = str(env_name)
        self.compression = str(compression)
        self.nonlinearity = str(nonlinearity)
        self.signature = str(signature)
        self.gens = gens
        self.vals = vals


def _slug(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


def _task_signature(task):
    keep_keys = [
        "ENV_NAME",
        "compression",
        "k",
        "norm",
        "nonlinearity",
        "percentile",
        "num_levels",
        "rate",
        "GENERATIONS",
        "POPULATION_SIZE",
        "CMA_SIGMA",
        "EPISODES_PER_INDIVIDUAL",
        "MAX_STEPS_PER_EPISODE",
    ]
    sub = {k: task.get(k) for k in keep_keys if k in task}
    return json.dumps(sub, sort_keys=True, separators=(",", ":"))


def _extract_group_keys(task):
    compression = str(task.get("compression", "unknown"))
    nonlinearity = task.get("nonlinearity", None)
    nonlinearity = "none" if nonlinearity is None else str(nonlinearity)
    return compression, nonlinearity


def _series_from_plot_data(plot_data, metric):
    if not isinstance(plot_data, list):
        raise ValueError("plot_data_json must decode to a list")
    if not plot_data:
        raise ValueError("plot_data_json is empty")

    col = 2 if metric == "avg" else 1
    gens = []
    vals = []
    for row in plot_data:
        if not (isinstance(row, (list, tuple)) and len(row) >= 3):
            continue
        gens.append(int(row[0]))
        vals.append(float(row[col]))
    if not gens:
        raise ValueError("plot_data_json had no usable rows")

    x = np.asarray(gens, dtype=np.int32)
    y = np.asarray(vals, dtype=np.float64)
    order = np.argsort(x)
    return x[order], y[order]


def _mean_curve(series):
    acc = defaultdict(list)
    for s in series:
        for g, v in zip(s.gens.tolist(), s.vals.tolist()):
            acc[int(g)].append(float(v))
    xs = np.asarray(sorted(acc.keys()), dtype=np.int32)
    means = np.asarray([float(np.mean(acc[int(g)])) for g in xs], dtype=np.float64)
    stds = np.asarray([float(np.std(acc[int(g)])) for g in xs], dtype=np.float64)
    return xs, means, stds


def _load_runs(db_path, metric, env_filter):
    con = sqlite3.connect(db_path, timeout=30.0)
    try:
        cur = con.cursor()
        cur.execute("SELECT id, run_id, env_name, task_json, plot_data_json FROM runs")
        out = []
        for row_id, run_id, env_name, task_json, plot_data_json in cur.fetchall():
            if env_filter is not None and str(env_name) != env_filter:
                continue
            try:
                task = json.loads(task_json)
                plot_data = json.loads(plot_data_json)
                if not isinstance(task, dict):
                    continue
                compression, nonlinearity = _extract_group_keys(task)
                sig = _task_signature(task)
                gens, vals = _series_from_plot_data(plot_data, metric=metric)
            except Exception:
                continue
            out.append(
                RunSeries(
                    row_id=row_id,
                    run_id=run_id,
                    env_name=env_name,
                    compression=compression,
                    nonlinearity=nonlinearity,
                    signature=sig,
                    gens=gens,
                    vals=vals,
                )
            )
        return out
    finally:
        con.close()


def _plot_individuals_and_mean(
    *,
    series,
    title,
    ylabel,
    out_path,
):
    xs, mean, _std = _mean_curve(series)

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in series:
        ax.plot(s.gens, s.vals, linewidth=1.0, alpha=0.25)
    ax.plot(xs, mean, linewidth=2.5, color="black", label="mean")

    ax.set_title(title)
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_mean_only(
    *,
    series,
    title,
    ylabel,
    out_path,
):
    xs, mean, std = _mean_curve(series)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, mean, linewidth=2.5, label="mean")
    ax.fill_between(xs, mean - std, mean + std, alpha=0.2, label="±1 std")

    ax.set_title(title)
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_compression_compare_nonlinearity(
    *,
    by_nl,
    title,
    ylabel,
    out_path,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for nl, series in sorted(by_nl.items(), key=lambda kv: kv[0]):
        xs, mean, _std = _mean_curve(series)
        ax.plot(xs, mean, linewidth=2.25, label=nl)

    ax.set_title(title)
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.abspath(os.path.join(repo_root, "data", "evo_train_runs.db"))
    out_dir = os.path.abspath(os.path.join(repo_root, "plots", "evo_train_runs"))
    metric = "avg"
    write_mean_only = False
    env_filter = None

    runs = _load_runs(db_path, metric=metric, env_filter=env_filter)
    if not runs:
        print(f"No runs found in DB for env={env_filter!r}. db={db_path}")
        return 1

    ylabel = "avg reward" if metric == "avg" else "best reward"

    groups = defaultdict(list)
    for r in runs:
        groups[(r.compression, r.nonlinearity)].append(r)

    for (comp, nl), series in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sigs = {s.signature for s in series}
        if len(sigs) > 1:
            print(
                f"[warn] mixed parameter settings in group (compression={comp}, nonlinearity={nl}); "
                f"signatures={len(sigs)} (mean curve may be meaningless)"
            )

        title_base = f"{comp} + {nl} ({len(series)} runs) [{metric}]"
        base = f"{_slug(comp)}__{_slug(nl)}__{metric}"

        _plot_individuals_and_mean(
            series=series,
            title=title_base + " individuals + mean",
            ylabel=ylabel,
            out_path=os.path.join(out_dir, f"{base}__individuals.png"),
        )
        if bool(write_mean_only):
            _plot_mean_only(
                series=series,
                title=title_base + " mean",
                ylabel=ylabel,
                out_path=os.path.join(out_dir, f"{base}__mean.png"),
            )

    by_comp = defaultdict(lambda: defaultdict(list))
    for (comp, nl), series in groups.items():
        by_comp[comp][nl] = series

    for comp, by_nl in sorted(by_comp.items(), key=lambda kv: kv[0]):
        if len(by_nl) < 2:
            continue
        title = f"{comp} compare nonlinearities (mean curves) [{metric}]"
        out_path = os.path.join(out_dir, f"{_slug(comp)}__compare_nonlinearity__{metric}.png")
        _plot_compression_compare_nonlinearity(
            by_nl=by_nl,
            title=title,
            ylabel=ylabel,
            out_path=out_path,
        )

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())