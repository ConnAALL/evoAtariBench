"""
Convert LaTeX table rows into a CSV.
"""

import argparse
import csv
import os
import re
import sys
from difflib import get_close_matches
from pathlib import Path


def _read_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _clean_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    # Drop common LaTeX noise
    if s.startswith("%"):
        return ""
    # Drop captions (often appear after \end{tabular})
    if s.startswith("\\caption") or s.startswith("\\captionof"):
        return ""
    # Drop tabular wrappers and other common block wrappers
    if s.startswith("\\begin{") or s.startswith("\\end{"):
        return ""
    if s.startswith("\\hline") or s == "\\hline":
        return ""
    # booktabs-style rules
    if s.startswith("\\midrule") or s.startswith("\\toprule") or s.startswith("\\bottomrule"):
        return ""
    # Remove trailing row terminator (\\)
    if s.endswith("\\\\"):
        s = s[:-2].rstrip()
    return s


def _extract_brace_group(s: str, i: int) -> tuple[str | None, int]:
    """
    Parse a balanced {...} group starting at s[i] == '{'.
    Returns (content, next_index_after_group). If parse fails, returns (None, i).
    """
    if i < 0 or i >= len(s) or s[i] != "{":
        return None, i
    depth = 0
    j = i
    while j < len(s):
        ch = s[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # content excludes outer braces
                return s[i + 1 : j], j + 1
        j += 1
    return None, i


def _unwrap_known_latex(s: str) -> str:
    """
    Best-effort stripping of common LaTeX wrappers.
    Examples:
      \\textbf{X} -> X
      \\textcolor{blue}{X} -> X
    Supports nesting by repeated passes.
    """
    if not s or "\\" not in s:
        return s

    known_one_arg = {
        "textbf",
        "textit",
        "textsc",
        "emph",
        "mathbf",
        "mathrm",
        "mathit",
        "underline",
        "gamename",
    }

    changed = True
    while changed:
        changed = False
        i = 0
        out = []
        while i < len(s):
            if s[i] != "\\":
                out.append(s[i])
                i += 1
                continue

            # Parse command name: \foo or \foo*
            j = i + 1
            while j < len(s) and (s[j].isalpha() or s[j] == "*"):
                j += 1
            cmd = s[i + 1 : j]
            if not cmd:
                out.append(s[i])
                i += 1
                continue

            # Allow optional whitespace before brace groups: \cmd {x}
            k0 = j
            while k0 < len(s) and s[k0].isspace():
                k0 += 1

            # \textcolor{color}{content}
            if cmd == "textcolor" and k0 < len(s) and s[k0] == "{":
                color, k = _extract_brace_group(s, k0)
                if color is not None and k < len(s) and s[k] == "{":
                    content, k2 = _extract_brace_group(s, k)
                    if content is not None:
                        out.append(content)
                        i = k2
                        changed = True
                        continue

            # \cmd{content} for known wrappers
            if cmd in known_one_arg and k0 < len(s) and s[k0] == "{":
                content, k = _extract_brace_group(s, k0)
                if content is not None:
                    out.append(content)
                    i = k
                    changed = True
                    continue

            # Unknown command: keep as-is
            out.append(s[i])
            i += 1

        s2 = "".join(out)
        if s2 != s:
            s = s2
            changed = True

    return s


def _clean_cell(cell: str) -> str:
    s = cell.strip()

    # Unescape common LaTeX escapes in plain text cells.
    # e.g. chopper\_command -> chopper_command
    s = s.replace("\\_", "_")

    # Drop tiny uncertainty annotations, e.g. "{\\tiny($\\pm$ 268)}"
    # Commonly appears as a suffix after a mean score; we want to keep only the mean.
    s = re.sub(r"\{\s*\\tiny\s*\(.*?\)\s*\}", "", s).strip()

    # Handle brace-wrapped bold like "{\\bf 74.1 }" (common in some tables).
    # Do a couple passes to handle light nesting.
    for _ in range(2):
        s2 = re.sub(r"^\{\s*\\bf\s+(.*?)\s*\}$", r"\1", s)
        s2 = re.sub(r"^\{\s*\\bfseries\s+(.*?)\s*\}$", r"\1", s2)
        if s2 == s:
            break
        s = s2.strip()

    # Strip common LaTeX wrappers (can be nested)
    s = _unwrap_known_latex(s)
    s = s.strip()

    # Strip math mode wrappers
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)") and len(s) >= 4:
        s = s[2:-2].strip()

    # Remove thousands separators inside numbers: 2,354.5 -> 2354.5
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


def _parse_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        s = _clean_line(raw)
        if not s:
            continue
        parts = [_clean_cell(p) for p in s.split("&")]
        if len(parts) == 1 and parts[0] == "":
            continue
        # Skip typical header/footer rows when user pastes a full LaTeX table.
        first = (parts[0] or "").strip().lower()
        if first == "game":
            continue
        if first.replace(" ", "") in {"timesbest", "timesbest."}:
            continue
        rows.append(parts)
    return rows


def _parse_table(text: str) -> tuple[list[str] | None, list[list[str]]]:
    """
    Parse a full LaTeX tabular or a set of LaTeX rows.

    If a header row starting with "Game" exists, returns (header, data_rows).
    Otherwise returns (None, data_rows).
    """
    header: list[str] | None = None
    rows: list[list[str]] = []

    for raw in text.splitlines():
        s = _clean_line(raw)
        if not s:
            continue
        parts = [_clean_cell(p) for p in s.split("&")]
        if len(parts) == 1 and parts[0] == "":
            continue

        first = (parts[0] or "").strip().lower()
        if first == "game":
            header = parts
            continue
        if first.replace(" ", "") in {"timesbest", "timesbest."}:
            continue

        rows.append(parts)

    return header, rows


def _normalize_game_name(name: str) -> str:
    """
    Normalize a game name for matching across different formatting styles:
    - case-insensitive
    - ignores spaces/punctuation
    - handles roman numeral tokens like "II" -> "2"
    """
    s = str(name).strip().lower()
    # Convert common roman numeral tokens to digits (token-based, before punctuation stripping)
    s = re.sub(r"\biii\b", "3", s)
    s = re.sub(r"\bii\b", "2", s)
    s = re.sub(r"\biv\b", "4", s)
    s = re.sub(r"\bv\b", "5", s)
    # Drop all non-alphanumeric characters
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _extract_game_from_env(env: str) -> str:
    # "ALE/MsPacman-v5" -> "MsPacman"
    s = str(env)
    if "/" in s:
        s = s.split("/", 1)[1]
    if "-v" in s:
        s = s.split("-v", 1)[0]
    return s


def _load_official_env_map(official_csv_path: str) -> dict[str, str]:
    """
    Returns {normalized_game: official_environment_name}.
    """
    env_map: dict[str, str] = {}
    with open(official_csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "environment" not in r.fieldnames:
            raise SystemExit(
                f"Error: official CSV {official_csv_path} must contain an 'environment' column"
            )
        for row in r:
            env = (row.get("environment") or "").strip()
            if not env:
                continue
            game = _extract_game_from_env(env)
            key = _normalize_game_name(game)
            # If duplicates ever exist, keep the first.
            env_map.setdefault(key, env)
    return env_map


def _match_environment(
    game: str,
    env_map: dict[str, str],
    *,
    strict: bool,
    row_idx: int,
) -> str:
    key = _normalize_game_name(game)
    if key in env_map:
        return env_map[key]

    # Fuzzy fallback: pick closest normalized key.
    candidates = get_close_matches(key, list(env_map.keys()), n=1, cutoff=0.85)
    if candidates:
        return env_map[candidates[0]]

    msg = f"Warning: could not match game {game!r} to an official environment (row {row_idx})"
    if strict:
        raise SystemExit("Error: " + msg)
    print(msg, file=sys.stderr)
    return ""


def get_args():
    p = argparse.ArgumentParser(description="Convert LaTeX table rows into a CSV.")
    p.add_argument("--in", dest="in_path", default="_in.txt", help="Input file path (default: stdin). Use '-' for stdin.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument(
        "--colnames",
        default=None,
        help="Comma-separated column names, e.g. game,test1,test2,test3,test4",
    )
    p.add_argument(
        "--official-csv",
        default=str(Path(__file__).resolve().parent / "official_atari_game_info.csv"),
        help="Path to official Atari env CSV (default: data/previous_benchmark_scores/official_atari_game_info.csv)",
    )
    p.add_argument(
        "--env-colname",
        default="environment",
        help="Name of the added environment column (default: environment)",
    )
    p.add_argument(
        "--strict-env-match",
        action="store_true",
        help="Fail if any game cannot be matched to an official environment.",
    )
    return p.parse_args()


def main():
    args = get_args()
    colnames_in = (
        [c.strip() for c in str(args.colnames).split(",") if c.strip()]
        if args.colnames is not None
        else []
    )
    env_colname = str(args.env_colname).strip() or "environment"

    text = _read_text(args.in_path)
    header, rows = _parse_table(text)
    if not rows:
        raise SystemExit("Error: no rows parsed from input")

    if not colnames_in:
        if header is None:
            raise SystemExit(
                "Error: input did not include a header row (starting with 'Game') "
                "and --colnames was not provided."
            )
        colnames_in = header

    # Determine where the game column is (default to first column if not found).
    lower_cols = [c.lower() for c in colnames_in]
    game_idx = lower_cols.index("game") if "game" in lower_cols else 0

    # Add environment column while keeping game column.
    env_in_cols = env_colname.lower() in lower_cols
    if env_in_cols:
        env_idx = lower_cols.index(env_colname.lower())
        colnames_out = list(colnames_in)
        expected_input = len(colnames_in) - 1  # we will insert env value
    else:
        env_idx = game_idx + 1
        colnames_out = list(colnames_in)
        colnames_out.insert(env_idx, env_colname)
        expected_input = len(colnames_in)

    bad = [i for i, r in enumerate(rows, start=1) if len(r) != expected_input]
    if bad:
        example_i = bad[0]
        raise SystemExit(
            f"Error: row {example_i} has {len(rows[example_i-1])} columns but expected {expected_input}"
        )

    env_map = _load_official_env_map(str(args.official_csv))
    out_rows: list[list[str]] = []
    for i, r in enumerate(rows, start=1):
        game = r[game_idx] if game_idx < len(r) else ""
        env = _match_environment(
            game,
            env_map,
            strict=bool(args.strict_env_match),
            row_idx=i,
        )
        rr = list(r)
        rr.insert(env_idx, env)
        out_rows.append(rr)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(colnames_out)
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()


