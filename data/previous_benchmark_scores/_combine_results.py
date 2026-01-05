#!/usr/bin/env python3
"""
Combine benchmark CSVs from multiple papers into one wide CSV.

Rules:
- Input: all *.csv files in a directory that do NOT start with "_" and are not
  "official_atari_game_info.csv".
- Each input CSV should have at least a `game` column. If `environment` is
  missing, we attempt to derive it using official_atari_game_info.csv.
- Output is merged on `environment` (union of all environments).
- The output always contains `game` and `environment` columns, plus all metric
  columns from the inputs.
- Deduplication:
  - Exact duplicates: if two metric columns have the exact same (env -> value)
    mapping over the games they cover, we keep one and skip the other.
  - Subset duplicates: if column B reports fewer games than column A, but every
    value in B matches A for those games, then B is considered redundant and is
    skipped (or replaced if A is encountered later). This is reported.

- Optional: can also write a small `subset_results.csv` for the specific Table-3
  comparison requested (Space Invaders; selected methods from specific papers).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _extract_game_from_env(env: str) -> str:
    # "ALE/MsPacman-v5" -> "MsPacman"
    s = str(env).strip()
    if "/" in s:
        s = s.split("/", 1)[1]
    if "-v" in s:
        s = s.split("-v", 1)[0]
    return s


def _normalize_game_key(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("’", "'")
    # Roman numerals -> digits (token-based)
    s = re.sub(r"\biii\b", "3", s)
    s = re.sub(r"\bii\b", "2", s)
    s = re.sub(r"\biv\b", "4", s)
    s = re.sub(r"\bv\b", "5", s)
    # Drop non-alphanumerics
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _load_official_env_map(path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "environment" not in r.fieldnames:
            raise SystemExit(f"Error: {path} must have an 'environment' column")
        for row in r:
            env = (row.get("environment") or "").strip()
            if not env:
                continue
            game = _extract_game_from_env(env)
            env_map.setdefault(_normalize_game_key(game), env)
    return env_map


def _clean_value(v: str | None) -> str | None:
    """
    Canonicalize metric cell values for comparisons/merging.
    - Treat empty / '-' as missing
    - Strip whitespace, remove trailing LaTeX percent "\%" / "%" tokens
    - Remove digit thousands separators: 12,345.6 -> 12345.6
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s or s == "-":
        return None
    # common percent formatting
    s = s.replace("\\%", "%")
    if s.endswith("%"):
        s = s[:-1].strip()
    # remove commas inside numbers
    s = re.sub(r"(?<=\\d),(?=\\d)", "", s)
    if not s:
        return None
    # Canonicalize numeric formatting when possible so "3166" == "3166.0"
    try:
        num = float(s)
        # 15 significant digits is plenty for these tables
        s = f"{num:.15g}"
    except Exception:
        pass
    return s


def _find_col_case_insensitive(fieldnames: list[str], desired: str) -> str | None:
    m = {c.strip().lower(): c for c in fieldnames}
    return m.get(desired.strip().lower())


def _load_value_for_env(
    csv_path: Path,
    *,
    env: str,
    desired_col: str,
    env_map: dict[str, str],
    strict_env: bool,
) -> tuple[str | None, str | None]:
    """
    Returns (game, value) for the row that matches env.
    - If `environment` column exists: match directly.
    - Else: infer env from game using env_map.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"Error: empty header in {csv_path.name}")

        game_col = _find_col_case_insensitive(list(r.fieldnames), "game")
        env_col = _find_col_case_insensitive(list(r.fieldnames), "environment")
        val_col = _find_col_case_insensitive(list(r.fieldnames), desired_col)

        if val_col is None:
            raise SystemExit(f"Error: {csv_path.name} missing column {desired_col!r}")
        if game_col is None and env_col is None:
            raise SystemExit(f"Error: {csv_path.name} missing both 'game' and 'environment'")

        for row in r:
            game = ((row.get(game_col) if game_col else "") or "").strip()
            row_env = ((row.get(env_col) if env_col else "") or "").strip()
            if not row_env and env_map and game:
                row_env = env_map.get(_normalize_game_key(game), "")
            if not row_env:
                if strict_env:
                    raise SystemExit(
                        f"Error: {csv_path.name} could not infer environment for game {game!r}"
                    )
                continue
            if row_env != env:
                continue
            return game or _extract_game_from_env(env), _clean_value(row.get(val_col))

    return None, None


def write_table3_subset(folder: Path, subset_out: Path, strict_env: bool) -> None:
    """
    Write subset_results.csv for all available games using the related papers:
      - UCT (Bellemare et al. 2013)
      - OpenAI-ES, DQN, A2C FF, A3C FF (1 Day) (Salimans et al. 2017)
      - hneat_object (Hausknecht et al. 2014)   <-- IMPORTANT: use hneat_object (not hneat_noise)
    """
    folder = folder.resolve()
    subset_out = subset_out.resolve()

    official_csv = folder / "official_atari_game_info.csv"
    env_map = _load_official_env_map(official_csv) if official_csv.exists() else {}

    # (output_col, source_file, source_col)
    wanted = [
        ("UCT", "bellemare_2013_planning_table5.csv", "UCT"),
        ("hneat_object", "hausknecht_2014_hyperneat_table1.csv", "hneat_object"),
        ("DQN", "salimans_2017_openAIES__table2.csv", "DQN"),
        ("OpenAI-ES", "salimans_2017_openAIES__table2.csv", "ES FF, 1 hour"),
        ("A2C FF", "salimans_2017_openAIES__table2.csv", "A2C FF"),
        ("A3C FF (1 Day)", "salimans_2017_openAIES__table2.csv", "A3C FF, 1 day"),
    ]

    # Load each column for all environments it covers.
    col_maps: dict[str, dict[str, str]] = {}  # out_col -> {env: value}
    game_by_env: dict[str, str] = {}

    for out_col, fname, src_col in wanted:
        p = folder / fname
        if not p.exists():
            raise SystemExit(f"Error: required source file missing for subset_results.csv: {p}")

        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                raise SystemExit(f"Error: empty header in {p.name}")

            game_col = _find_col_case_insensitive(list(r.fieldnames), "game")
            env_col = _find_col_case_insensitive(list(r.fieldnames), "environment")
            val_col = _find_col_case_insensitive(list(r.fieldnames), src_col)
            if val_col is None:
                raise SystemExit(f"Error: {p.name} missing column {src_col!r}")
            if game_col is None and env_col is None:
                raise SystemExit(f"Error: {p.name} missing both 'game' and 'environment'")

            m: dict[str, str] = {}
            for row in r:
                game = ((row.get(game_col) if game_col else "") or "").strip()
                env = ((row.get(env_col) if env_col else "") or "").strip()
                if not env and env_map and game:
                    env = env_map.get(_normalize_game_key(game), "")
                if not env:
                    msg = f"{p.name}: could not infer environment for game {game!r}"
                    if strict_env:
                        raise SystemExit("Error: " + msg)
                    print("Warning:", msg, file=sys.stderr)
                    continue

                if env not in game_by_env:
                    game_by_env[env] = game or _extract_game_from_env(env)

                val = _clean_value(row.get(val_col))
                if val is None:
                    continue
                m[env] = val

            col_maps[out_col] = m

    # Union of all environments covered by any wanted column.
    all_envs: set[str] = set()
    for m in col_maps.values():
        all_envs.update(m.keys())

    subset_out.parent.mkdir(parents=True, exist_ok=True)
    out_fields = ["game", "environment"] + [c[0] for c in wanted]
    with subset_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for env in sorted(all_envs):
            row_out: dict[str, str] = {
                "environment": env,
                "game": game_by_env.get(env, _extract_game_from_env(env)),
            }
            for out_col, _fname, _src_col in wanted:
                row_out[out_col] = col_maps.get(out_col, {}).get(env, "")
            w.writerow(row_out)

    print(f"Wrote subset CSV: {subset_out} ({len(all_envs)} games)")


def _iter_input_csvs(folder: Path, *, out_path: Path | None = None) -> list[Path]:
    out: list[Path] = []
    out_resolved = out_path.resolve() if out_path is not None else None
    for p in sorted(folder.glob("*.csv")):
        if p.name.startswith("_"):
            continue
        if p.name == "official_atari_game_info.csv":
            continue
        # Never treat previous combined output as an input.
        if p.name == "combined_benchmarks.csv":
            continue
        # Never treat subset output as an input.
        if p.name == "subset_results.csv":
            continue
        # Also skip the configured output path (in case user uses a different name).
        if out_resolved is not None and p.resolve() == out_resolved:
            continue
        out.append(p)
    return out


@dataclass(frozen=True)
class ColumnSource:
    file: str
    column: str


@dataclass
class MetricCandidate:
    source: ColumnSource
    base_name: str
    file_stem: str
    values_by_env: dict[str, str]  # only non-missing values


def _dedupe_signature(env_to_val: dict[str, str | None]) -> tuple[tuple[str, str], ...]:
    """
    Signature used to detect duplicates.
    Only environments with a non-empty value participate.
    """
    items = [(env, val) for env, val in env_to_val.items() if val is not None]
    items.sort(key=lambda x: x[0])
    return tuple((e, v) for e, v in items)


def _safe_unique_name(base: str, used: set[str], suffix: str) -> str:
    if base not in used:
        used.add(base)
        return base
    cand = f"{base}__{suffix}"
    if cand not in used:
        used.add(cand)
        return cand
    i = 2
    while True:
        cand2 = f"{base}__{suffix}__{i}"
        if cand2 not in used:
            used.add(cand2)
            return cand2
        i += 1


def _is_subset(sub: dict[str, str], sup: dict[str, str]) -> bool:
    """
    True iff for every (env,val) in sub, sup has the same val for env.
    """
    for env, val in sub.items():
        if sup.get(env) != val:
            return False
    return True


def combine(folder: Path, out_path: Path, strict_env: bool) -> None:
    folder = folder.resolve()
    out_path = out_path.resolve()

    official_csv = folder / "official_atari_game_info.csv"
    env_map = _load_official_env_map(official_csv) if official_csv.exists() else {}
    if not env_map:
        print(
            f"Warning: official env map not found/empty at {official_csv}; "
            "will not be able to infer missing `environment` columns.",
            file=sys.stderr,
        )

    files = _iter_input_csvs(folder, out_path=out_path)
    if not files:
        raise SystemExit(f"Error: no input CSVs found in {folder}")

    # Master table keyed by environment
    game_by_env: dict[str, str] = {}

    # Dedupe tracking across all metric columns
    used_colnames: set[str] = set()
    final_metric_cols: list[str] = []
    outname_to_values: dict[str, dict[str, str]] = {}
    outname_to_sig: dict[str, tuple[tuple[str, str], ...]] = {}
    sig_to_outname: dict[tuple[tuple[str, str], ...], str] = {}

    skipped_exact: list[tuple[ColumnSource, str]] = []
    skipped_subset: list[tuple[ColumnSource, str]] = []
    replaced: list[tuple[ColumnSource, str]] = []
    renamed: list[tuple[ColumnSource, str]] = []

    for path in files:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                print(f"Skipping {path.name}: empty header", file=sys.stderr)
                continue

            fcols = list(r.fieldnames)
            lower = {c.lower(): c for c in fcols}
            game_col = lower.get("game")
            env_col = lower.get("environment")
            if not game_col and not env_col:
                print(f"Skipping {path.name}: missing both 'game' and 'environment' columns", file=sys.stderr)
                continue

            # Collect per-column env->value for this file for dedupe.
            # Only include metric columns (exclude game/environment).
            metric_cols = [c for c in fcols if c not in {game_col, env_col} and c is not None]

            env_to_vals_by_col: dict[str, dict[str, str | None]] = {c: {} for c in metric_cols}

            # First pass read rows
            file_rows = list(r)
            for row in file_rows:
                game = (row.get(game_col) if game_col else None) or ""
                game = game.strip()
                env = (row.get(env_col) if env_col else None) or ""
                env = env.strip()
                if not env:
                    if env_map and game:
                        env = env_map.get(_normalize_game_key(game), "")
                if not env:
                    msg = f"{path.name}: could not determine environment for game {game!r}"
                    if strict_env:
                        raise SystemExit("Error: " + msg)
                    print("Warning:", msg, file=sys.stderr)
                    continue

                if env not in game_by_env:
                    game_by_env[env] = game or _extract_game_from_env(env)

                for c in metric_cols:
                    env_to_vals_by_col[c][env] = _clean_value(row.get(c))

            # Decide which metric cols to keep based on dedupe
            for c in metric_cols:
                src = ColumnSource(file=path.name, column=c)
                # Build compact mapping for comparisons (only non-missing values)
                values_by_env = {e: v for e, v in env_to_vals_by_col[c].items() if v is not None}
                if not values_by_env:
                    # Nothing to contribute; skip silently.
                    continue

                sig = _dedupe_signature(env_to_vals_by_col[c])
                if sig and sig in sig_to_outname:
                    skipped_exact.append((src, sig_to_outname[sig]))
                    continue

                # Subset/superset duplicates:
                # - If this column is fully explained by an existing column, skip it.
                # - If this column fully explains an existing column and has more coverage,
                #   replace the existing column.
                to_replace: list[str] = []
                skip_as_subset_of: str | None = None
                for existing_name, existing_vals in outname_to_values.items():
                    if _is_subset(values_by_env, existing_vals):
                        skip_as_subset_of = existing_name
                        break
                    if _is_subset(existing_vals, values_by_env):
                        # existing is redundant w.r.t this new column
                        to_replace.append(existing_name)

                if skip_as_subset_of is not None:
                    skipped_subset.append((src, skip_as_subset_of))
                    continue

                # Add this metric column
                out_name = _safe_unique_name(c, used_colnames, suffix=Path(path.name).stem)
                if out_name != c:
                    renamed.append((src, out_name))

                # Replace any smaller-coverage columns that are subsets of this one
                for old in to_replace:
                    # Remove old from outputs
                    if old in final_metric_cols:
                        final_metric_cols.remove(old)
                    outname_to_values.pop(old, None)
                    old_sig = outname_to_sig.pop(old, None)
                    if old_sig is not None:
                        sig_to_outname.pop(old_sig, None)
                    replaced.append((src, old))
                    # Also remove it from exact-dup table if present

                final_metric_cols.append(out_name)
                outname_to_values[out_name] = values_by_env
                if sig:
                    outname_to_sig[out_name] = sig
                    sig_to_outname[sig] = out_name

    # Build output rows
    out_fields = ["game", "environment"] + final_metric_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        all_envs = sorted(game_by_env.keys())
        for env in all_envs:
            row_out = {"environment": env, "game": game_by_env.get(env, _extract_game_from_env(env))}
            for col in final_metric_cols:
                v = outname_to_values.get(col, {}).get(env)
                if v is not None:
                    row_out[col] = v
            w.writerow(row_out)

    print(f"Scanned {len(files)} CSVs from {folder}")
    print(f"Wrote combined CSV: {out_path} ({len(game_by_env)} environments, {len(final_metric_cols)} metrics)")
    if renamed:
        print("\nRenamed columns (to avoid name collisions):")
        for src, new in renamed:
            print(f"- {src.file}:{src.column} -> {new}")
    if replaced:
        print("\nReplaced smaller-coverage duplicate columns (new column is a superset match):")
        for src, old in replaced:
            print(f"- {src.file}:{src.column} replaced {old}")
    if skipped_exact:
        print("\nSkipped exact duplicate columns (identical results over covered games):")
        for src, kept in skipped_exact:
            print(f"- {src.file}:{src.column} (duplicate of {kept})")
    if skipped_subset:
        print("\nSkipped subset duplicate columns (values match an existing column on all covered games):")
        for src, kept in skipped_subset:
            print(f"- {src.file}:{src.column} (subset-duplicate of {kept})")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        dest="folder",
        type=Path,
        default=here,
        help="Folder containing benchmark CSVs (default: this folder)",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=here / "combined_benchmarks.csv",
        help="Output CSV path (default: combined_benchmarks.csv in this folder)",
    )
    p.add_argument(
        "--strict-env",
        action="store_true",
        help="Fail if any row lacks an environment and cannot be inferred.",
    )
    p.add_argument(
        "--write-subset-table3",
        action="store_true",
        help="Also write subset_results.csv for the Space Invaders Table-3 comparison.",
    )
    p.add_argument(
        "--subset-out",
        type=Path,
        default=here / "subset_results.csv",
        help="Output path for subset CSV (default: subset_results.csv in this folder)",
    )
    args = p.parse_args()
    combine(folder=args.folder, out_path=args.out_path, strict_env=bool(args.strict_env))
    if args.write_subset_table3:
        write_table3_subset(
            folder=args.folder, subset_out=args.subset_out, strict_env=bool(args.strict_env)
        )


if __name__ == "__main__":
    main()


