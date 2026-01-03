"""
Write a filtered subset of atari_game_infos.csv containing only games that appear
in at least one other CSV under data/baselines/.

This is the "coverage" (union) of games across baseline/result CSVs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    d = os.path.dirname(here)  # .../data/baselines
    d_data = os.path.dirname(d)  # .../data
    if os.path.basename(d_data) == "data":
        return os.path.dirname(d_data)  # .../repo
    return os.path.dirname(os.path.dirname(here))


def _default_baselines_dir() -> str:
    return os.path.join(_repo_root(), "data", "baselines")


def _default_infos_csv() -> str:
    return os.path.join(_default_baselines_dir(), "atari_game_infos.csv")


def _default_out_csv() -> str:
    return os.path.join(_default_baselines_dir(), "atari_game_infos_covered.csv")


def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("\\_", "_")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _env_game_key(env_name: str) -> str:
    s = str(env_name).strip()
    if s.startswith("ALE/"):
        s = s[len("ALE/") :]
    if s.endswith("-v5"):
        s = s[: -len("-v5")]
    return _norm_key(s)


def load_infos_map(infos_csv: str) -> tuple[dict[str, str], list[str]]:
    """
    Returns (normalized_game_key -> env_name, fieldnames)
    """
    with open(infos_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "environment" not in r.fieldnames:
            raise ValueError(f"Expected column 'environment' in {infos_csv}")
        m: dict[str, str] = {}
        for row in r:
            env = (row.get("environment") or "").strip()
            if not env:
                continue
            m[_env_game_key(env)] = env
        return m, list(r.fieldnames)


def _detect_game_column(fieldnames: list[str]) -> str | None:
    lower = {c.lower(): c for c in fieldnames}
    if "game" in lower:
        return lower["game"]
    if "games" in lower:
        return lower["games"]
    return None


def iter_other_csvs(baselines_dir: str, infos_name: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(baselines_dir)):
        if not name.lower().endswith(".csv"):
            continue
        if name == infos_name:
            continue
        # Exclude generated/derived outputs in this folder.
        if name == "atari_game_infos_covered.csv":
            continue
        out.append(name)
    return out


def covered_env_sources(baselines_dir: str, infos_csv: str) -> dict[str, set[str]]:
    infos_map, _fieldnames = load_infos_map(infos_csv)
    infos_envs = set(infos_map.values())
    infos_name = os.path.basename(infos_csv)
    csv_files = iter_other_csvs(baselines_dir, infos_name=infos_name)

    sources: dict[str, set[str]] = {}
    for name in csv_files:
        path = os.path.join(baselines_dir, name)
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                if r.fieldnames is None:
                    continue
                fieldnames = list(r.fieldnames)
                lower = {c.lower(): c for c in fieldnames}

                game_col = _detect_game_column(fieldnames)
                env_col = None
                if "ale_env" in lower:
                    env_col = lower["ale_env"]
                elif "environment" in lower:
                    env_col = lower["environment"]

                if game_col is None and env_col is None:
                    continue

                # Special-case atari_scores.csv: only include it when a row has at least one benchmark value.
                is_atari_scores = (name == "atari_scores.csv")
                meta_cols = {"game", "games", "ale_env", "environment", "action_space", "obs_rows", "obs_cols"}
                for row in r:
                    if is_atari_scores:
                        has_benchmark = False
                        for k, v in row.items():
                            if not k:
                                continue
                            if k.strip().lower() in meta_cols:
                                continue
                            sv = ("" if v is None else str(v)).strip()
                            if sv and sv.upper() != "NA" and sv.lower() != "nan":
                                has_benchmark = True
                                break
                        if not has_benchmark:
                            continue

                    env = None
                    if env_col is not None:
                        ev = (row.get(env_col) or "").strip()
                        if ev and ev in infos_envs:
                            env = ev

                    if env is None and game_col is not None:
                        g = (row.get(game_col) or "").strip()
                        if not g:
                            continue
                        env = infos_map.get(_norm_key(g))

                    if env is not None:
                        sources.setdefault(env, set()).add(name)
        except Exception:
            # If a CSV is malformed, just skip it.
            continue

    return sources


def write_filtered_infos(infos_csv: str, out_csv: str, env_sources: dict[str, set[str]]) -> int:
    with open(infos_csv, "r", encoding="utf-8", newline="") as fin:
        r = csv.DictReader(fin)
        if r.fieldnames is None:
            raise ValueError(f"No header row found in {infos_csv}")
        fieldnames = list(r.fieldnames)
        extra_col = "source_csvs"
        if extra_col not in fieldnames:
            fieldnames.append(extra_col)

        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=fieldnames)
            w.writeheader()

            n = 0
            for row in r:
                env = (row.get("environment") or "").strip()
                if env and env in env_sources:
                    row[extra_col] = "|".join(sorted(env_sources.get(env, set())))
                    w.writerow(row)
                    n += 1
            return n


def get_args():
    p = argparse.ArgumentParser(description="Filter atari_game_infos.csv to covered games from baseline CSVs.")
    p.add_argument("--baselines-dir", default=_default_baselines_dir(), help="Directory containing baseline CSVs.")
    p.add_argument("--infos-csv", default=_default_infos_csv(), help="Path to atari_game_infos.csv.")
    p.add_argument("--out", default=_default_out_csv(), help="Output CSV path.")
    return p.parse_args()


def main():
    args = get_args()
    baselines_dir = os.path.abspath(args.baselines_dir)
    infos_csv = os.path.abspath(args.infos_csv)
    out_csv = os.path.abspath(args.out)

    env_sources = covered_env_sources(baselines_dir=baselines_dir, infos_csv=infos_csv)
    n = write_filtered_infos(infos_csv=infos_csv, out_csv=out_csv, env_sources=env_sources)
    print(f"Wrote {n} rows to {out_csv}")


if __name__ == "__main__":
    main()


