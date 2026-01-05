"""
Plot runtime histogram from evo_train task logs and print slowest tasks/games.

This parses log lines like:
  [YYYY-mm-dd HH:MM:SS] [Task Start] [run_id=1] [Task: {...}]
  [YYYY-mm-dd HH:MM:SS] [Task End]   [run_id=1] [env=ALE/Game-v5] ... [Task: {...}]

It computes runtime = end_timestamp - start_timestamp per run_id, then:
  - writes a histogram PNG of runtimes
  - prints the longest-running individual tasks
  - prints the slowest games (by mean and by max runtime)

Example:
  python3 result_plotting_and_analysis/plot_task_runtimes.py \
    --log data/run_data/dct_game_sweep/evo_train_tasks_20260103_034434.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
    return os.path.dirname(os.path.dirname(here))


def _env_to_game(env_name: str) -> str:
    # "ALE/SpaceInvaders-v5" -> "SpaceInvaders"
    s = str(env_name or "")
    if "/" in s:
        s = s.split("/", 1)[1]
    s = re.sub(r"-v\d+$", "", s, flags=re.IGNORECASE)
    return s or "UNKNOWN"


def _norm_game_name(s: str) -> str:
    """
    Normalize game names for robust matching:
      "Beam Rider" -> "beamrider"
      "SpaceInvaders" -> "spaceinvaders"
    """
    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
_RUN_RE = re.compile(r"\[run_id=(\d+)\]")
_ENV_END_RE = re.compile(r"\[env=([^\]]+)\]")
_TASK_JSON_RE = re.compile(r"\[Task:\s*(\{.*\})\]\s*$")


def _parse_ts(line: str) -> datetime | None:
    m = _TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _parse_run_id(line: str) -> int | None:
    m = _RUN_RE.search(line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_task_json(line: str) -> dict[str, Any]:
    m = _TASK_JSON_RE.search(line)
    if not m:
        return {}
    s = m.group(1)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _parse_env_from_end(line: str) -> str | None:
    m = _ENV_END_RE.search(line)
    if not m:
        return None
    return str(m.group(1)).strip() or None


def _convert_seconds(seconds: float, units: str) -> float:
    u = str(units).lower().strip()
    if u in ("s", "sec", "secs", "second", "seconds"):
        return float(seconds)
    if u in ("m", "min", "mins", "minute", "minutes"):
        return float(seconds) / 60.0
    if u in ("h", "hr", "hrs", "hour", "hours"):
        return float(seconds) / 3600.0
    raise ValueError(f"Unknown units: {units}")


def _units_label(units: str) -> str:
    u = str(units).lower().strip()
    if u in ("s", "sec", "secs", "second", "seconds"):
        return "seconds"
    if u in ("m", "min", "mins", "minute", "minutes"):
        return "minutes"
    if u in ("h", "hr", "hrs", "hour", "hours"):
        return "hours"
    return str(units)


@dataclass(frozen=True)
class TaskRuntime:
    run_id: int
    env_name: str
    game: str
    start_ts: datetime
    end_ts: datetime
    runtime_s: float
    task: dict[str, Any]


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    # linear interpolation between closest ranks
    n = len(sorted_vals)
    x = (p / 100.0) * (n - 1)
    i = int(x)
    frac = x - i
    if i >= n - 1:
        return float(sorted_vals[-1])
    return float(sorted_vals[i]) * (1.0 - frac) + float(sorted_vals[i + 1]) * frac


def _parse_log(path: str) -> tuple[list[TaskRuntime], dict[str, int]]:
    start_by_run: dict[int, datetime] = {}
    task_by_run: dict[int, dict[str, Any]] = {}
    env_by_run: dict[int, str] = {}

    runtimes: list[TaskRuntime] = []
    stats = {
        "start_lines": 0,
        "end_lines": 0,
        "paired": 0,
        "missing_start": 0,
        "missing_end": 0,
        "unparsed_lines": 0,
    }

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "[Task Start]" in line:
                stats["start_lines"] += 1
                ts = _parse_ts(line)
                rid = _parse_run_id(line)
                if ts is None or rid is None:
                    stats["unparsed_lines"] += 1
                    continue
                if rid not in start_by_run:
                    start_by_run[rid] = ts
                task = _parse_task_json(line)
                if task and rid not in task_by_run:
                    task_by_run[rid] = task
                env = task.get("ENV_NAME") if isinstance(task, dict) else None
                if isinstance(env, str) and env.strip():
                    env_by_run[rid] = env.strip()
                continue

            if "[Task End]" in line:
                stats["end_lines"] += 1
                ts = _parse_ts(line)
                rid = _parse_run_id(line)
                if ts is None or rid is None:
                    stats["unparsed_lines"] += 1
                    continue
                end_task = _parse_task_json(line)
                end_env = _parse_env_from_end(line)
                if isinstance(end_env, str) and end_env.strip():
                    env_by_run[rid] = end_env.strip()
                if end_task and rid not in task_by_run:
                    task_by_run[rid] = end_task

                start_ts = start_by_run.get(rid)
                if start_ts is None:
                    stats["missing_start"] += 1
                    continue

                env = env_by_run.get(rid) or (task_by_run.get(rid, {}).get("ENV_NAME") if rid in task_by_run else None)
                env_s = str(env).strip() if env is not None else "UNKNOWN"
                rt_s = (ts - start_ts).total_seconds()
                if rt_s < 0:
                    # clock issues or log artifacts; keep absolute value but flag via sign
                    rt_s = abs(rt_s)

                runtimes.append(
                    TaskRuntime(
                        run_id=rid,
                        env_name=env_s,
                        game=_env_to_game(env_s),
                        start_ts=start_ts,
                        end_ts=ts,
                        runtime_s=float(rt_s),
                        task=task_by_run.get(rid, {}),
                    )
                )
                stats["paired"] += 1
                continue

    # Anything with a start and no end
    stats["missing_end"] = max(0, len(start_by_run) - len({r.run_id for r in runtimes}))
    return runtimes, stats


def _default_out_path(log_path: str, units: str) -> str:
    repo_root = _repo_root()
    out_dir = os.path.join(repo_root, "out", "plots", "task_runtimes")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(log_path))[0]
    return os.path.join(out_dir, f"{stem}_runtime_hist_{_units_label(units)}.png")


def _get_args():
    p = argparse.ArgumentParser(description="Plot histogram of per-task runtimes from evo_train task logs.")
    p.add_argument(
        "--log",
        required=True,
        help="Path to the evo_train task log file (e.g. data/run_data/dct_game_sweep/evo_train_tasks_*.log).",
    )
    p.add_argument("--out", default=None, help="Output PNG path. Defaults to out/plots/task_runtimes/…")
    p.add_argument("--bins", type=int, default=40, help="Histogram bin count.")
    p.add_argument(
        "--units",
        default="minutes",
        choices=["seconds", "minutes", "hours"],
        help="Units for histogram x-axis and printed tables.",
    )
    p.add_argument("--top-k", type=int, default=15, help="How many longest-running tasks to print.")
    p.add_argument("--top-games", type=int, default=15, help="How many slowest games to print (by mean / by max).")
    p.add_argument(
        "--report-games",
        nargs="*",
        default=["Asterix", "Beam Rider", "Freeway", "Seaquest", "Space Invaders"],
        help="Additionally print avg runtime for these games (names are matched loosely; spaces/case don't matter). "
        "Example: --report-games Asterix 'Beam Rider' Freeway Seaquest 'Space Invaders'",
    )
    p.add_argument("--min-runtime-s", type=float, default=0.0, help="Ignore tasks shorter than this (in seconds).")
    p.add_argument("--max-runtime-s", type=float, default=None, help="Ignore tasks longer than this (in seconds).")
    p.add_argument("--title", default=None, help="Optional plot title override.")
    p.add_argument("--log-x", action="store_true", help="Use a log-scaled x-axis (runtime).")
    p.add_argument("--log-y", action="store_true", help="Use a log-scaled y-axis (counts).")
    return p.parse_args()


def main():
    args = _get_args()
    log_path = os.path.abspath(args.log)
    if not os.path.exists(log_path):
        raise SystemExit(f"--log not found: {log_path}")

    runtimes, stats = _parse_log(log_path)
    if not runtimes:
        raise SystemExit(f"No completed tasks parsed from log: {log_path}")

    # Filter
    filtered: list[TaskRuntime] = []
    for r in runtimes:
        if args.min_runtime_s and r.runtime_s < float(args.min_runtime_s):
            continue
        if args.max_runtime_s is not None and r.runtime_s > float(args.max_runtime_s):
            continue
        filtered.append(r)
    runtimes = filtered
    if not runtimes:
        raise SystemExit("No runtimes left after filtering.")

    # Prepare values for plotting (converted units)
    units = str(args.units)
    xs = [_convert_seconds(r.runtime_s, units=units) for r in runtimes]
    xs_sorted = sorted(xs)

    # Summary
    n = len(xs_sorted)
    mean = sum(xs_sorted) / float(n)
    p50 = _percentile(xs_sorted, 50) or 0.0
    p90 = _percentile(xs_sorted, 90) or 0.0
    p99 = _percentile(xs_sorted, 99) or 0.0
    min_v = xs_sorted[0]
    max_v = xs_sorted[-1]

    print("")
    print(f"[parsed] {os.path.relpath(log_path, _repo_root())}")
    print(
        f"[counts] completed={len(runtimes)}  starts={stats['start_lines']}  ends={stats['end_lines']}  "
        f"missing_start={stats['missing_start']}  missing_end={stats['missing_end']}"
    )
    print(
        f"[runtimes] units={_units_label(units)}  min={min_v:.3f}  p50={p50:.3f}  p90={p90:.3f}  p99={p99:.3f}  "
        f"max={max_v:.3f}  mean={mean:.3f}"
    )

    # Print longest tasks
    top_k = max(1, int(args.top_k))
    longest = sorted(runtimes, key=lambda r: r.runtime_s, reverse=True)[:top_k]
    print("")
    print(f"Top {top_k} longest tasks:")
    for i, r in enumerate(longest, start=1):
        rt = _convert_seconds(r.runtime_s, units=units)
        nonlin = r.task.get("nonlinearity", "NA") if isinstance(r.task, dict) else "NA"
        k = r.task.get("k", "NA") if isinstance(r.task, dict) else "NA"
        comp = r.task.get("compression", "NA") if isinstance(r.task, dict) else "NA"
        print(
            f"{i:>2}. {r.game:<18}  run_id={r.run_id:<4}  runtime={rt:>9.3f} {_units_label(units):<7}  "
            f"comp={comp}  nonlinearity={nonlin}  k={k}"
        )

    # Aggregate by game/env
    by_env: dict[str, list[TaskRuntime]] = {}
    for r in runtimes:
        by_env.setdefault(r.env_name, []).append(r)

    game_rows = []
    for env, rs in by_env.items():
        vals = sorted(_convert_seconds(x.runtime_s, units=units) for x in rs)
        game_rows.append(
            {
                "env": env,
                "game": _env_to_game(env),
                "game_norm": _norm_game_name(_env_to_game(env)),
                "n": len(vals),
                "mean": sum(vals) / float(len(vals)),
                "max": float(vals[-1]),
                "p50": float(_percentile(vals, 50) or 0.0),
            }
        )

    # Requested game averages (by game name, not env string)
    requested = [g for g in (args.report_games or []) if str(g).strip()]
    if requested:
        rows_by_game_norm = {}
        for row in game_rows:
            rows_by_game_norm.setdefault(row["game_norm"], []).append(row)

        print("")
        print(f"Requested game averages (units={_units_label(units)}):")
        for name in requested:
            key = _norm_game_name(name)
            matches = rows_by_game_norm.get(key, [])
            if not matches:
                print(f"- {name}: not found in log")
                continue
            # Normally there should be exactly 1 match per game.
            # If there are multiple env variants, aggregate them.
            total_n = sum(int(m["n"]) for m in matches)
            # Weighted mean by sample count
            mean = sum(float(m["mean"]) * float(m["n"]) for m in matches) / float(total_n) if total_n else 0.0
            max_v = max(float(m["max"]) for m in matches)
            # For p50, we avoid re-deriving percentiles across merged envs; report the max of per-env p50s.
            p50 = max(float(m["p50"]) for m in matches)
            game_label = matches[0]["game"]
            print(f"- {game_label}: n={total_n}  mean={mean:.3f}  p50~={p50:.3f}  max={max_v:.3f}")

    top_games = max(1, int(args.top_games))

    print("")
    print(f"Slowest games by MEAN runtime (top {top_games}):")
    for i, row in enumerate(sorted(game_rows, key=lambda d: d["mean"], reverse=True)[:top_games], start=1):
        print(
            f"{i:>2}. {row['game']:<18}  n={row['n']:<3}  mean={row['mean']:>9.3f} {_units_label(units):<7}  "
            f"max={row['max']:>9.3f}  p50={row['p50']:>9.3f}"
        )

    print("")
    print(f"Slowest games by MAX runtime (top {top_games}):")
    for i, row in enumerate(sorted(game_rows, key=lambda d: d["max"], reverse=True)[:top_games], start=1):
        print(
            f"{i:>2}. {row['game']:<18}  n={row['n']:<3}  max={row['max']:>9.3f} {_units_label(units):<7}  "
            f"mean={row['mean']:>9.3f}  p50={row['p50']:>9.3f}"
        )

    # Plot histogram
    out_path = os.path.abspath(args.out) if args.out else os.path.abspath(_default_out_path(log_path, units=units))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.hist(xs, bins=max(1, int(args.bins)), color="#4C78A8", alpha=0.9, edgecolor="white", linewidth=0.7)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_xlabel(f"Runtime ({_units_label(units)})")
    ax.set_ylabel("Task count")

    if args.log_x:
        ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")

    title = args.title
    if not title:
        title = f"Task runtime histogram ({os.path.basename(log_path)})"
    ax.set_title(title)

    # Annotation with quick stats
    ax.text(
        0.98,
        0.98,
        f"n={n}\nmin={min_v:.3f}\np50={p50:.3f}\np90={p90:.3f}\nmax={max_v:.3f}\nmean={mean:.3f}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.0),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print("")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()


