#!/usr/bin/env python3
"""
Select Atari games from combined_benchmarks.csv by action-space size, and
rank them by how many benchmark results are available.

Example:
  ./select_games.py --action-space 18
  ./select_games.py --action-space 18 --top 20
  ./select_games.py --action-space 18 --include-na
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _is_missing(v: str | None, *, include_na: bool) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if not s:
        return True
    if not include_na and s.lower() in {"n/a", "na"}:
        return True
    return False


def main() -> None:
    here = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description=(
            "Filter games by action-space size and sort by how many benchmark "
            "results are available (non-missing metric cells)."
        )
    )
    p.add_argument(
        "--combined",
        type=Path,
        default=here / "combined_benchmarks.csv",
        help="Path to combined_benchmarks.csv (default: ./combined_benchmarks.csv)",
    )
    p.add_argument(
        "--action-space",
        type=int,
        required=True,
        help="Action space size to filter by (e.g. 18).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, only print the top N games (default: 0 = all).",
    )
    p.add_argument(
        "--include-na",
        action="store_true",
        help="Count 'N/A' cells as available results (default: excluded).",
    )
    args = p.parse_args()

    combined_path: Path = args.combined.resolve()
    if not combined_path.exists():
        raise SystemExit(f"Error: combined CSV not found: {combined_path}")

    with combined_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"Error: empty header in {combined_path}")

        required = {"game", "environment", "action_space_size"}
        missing = required.difference(set(r.fieldnames))
        if missing:
            raise SystemExit(
                "Error: combined CSV missing required columns: " + ", ".join(sorted(missing))
            )

        metric_cols = [c for c in r.fieldnames if c not in required]
        if not metric_cols:
            raise SystemExit("Error: no metric columns found in combined CSV")

        rows_out: list[tuple[int, str, str, int]] = []  # (rank later, game, env, count)
        for row in r:
            game = (row.get("game") or "").strip()
            env = (row.get("environment") or "").strip()
            action_size_raw = (row.get("action_space_size") or "").strip()

            try:
                action_size = int(action_size_raw)
            except Exception:
                continue

            if action_size != int(args.action_space):
                continue

            count = 0
            for c in metric_cols:
                if not _is_missing(row.get(c), include_na=bool(args.include_na)):
                    count += 1

            rows_out.append((0, game, env, count))

    # Sort by most benchmark coverage, then game name, then env.
    rows_out.sort(key=lambda t: (-t[3], t[1].lower(), t[2].lower()))

    if args.top and args.top > 0:
        rows_out = rows_out[: args.top]

    # Print TSV for easy copy/paste.
    print("rank\tavailable_results\tgame\tenvironment\taction_space_size")
    for i, (_rank, game, env, count) in enumerate(rows_out, start=1):
        print(f"{i}\t{count}\t{game}\t{env}\t{args.action_space}")


if __name__ == "__main__":
    main()

