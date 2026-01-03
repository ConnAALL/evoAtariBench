"""
Convert LaTeX table rows into a CSV.

Input example (from stdin or --in):
Venture & 0.00 & 1187.50 & 380.00 & 93.00 \\\\
Video Pinball & 16256.90 & 17297.60 & 42684.07 & 70009.00 \\\\

Usage:
  python3 data/baselines/latex_table_to_csv.py --colnames game,test1,test2,test3,test4 --out out.csv < rows.txt
"""

import argparse
import csv
import os
import re
import sys


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
    if s.startswith("\\hline") or s == "\\hline":
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
        "emph",
        "mathbf",
        "mathrm",
        "mathit",
        "underline",
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

            # \textcolor{color}{content}
            if cmd == "textcolor" and j < len(s) and s[j] == "{":
                color, k = _extract_brace_group(s, j)
                if color is not None and k < len(s) and s[k] == "{":
                    content, k2 = _extract_brace_group(s, k)
                    if content is not None:
                        out.append(content)
                        i = k2
                        changed = True
                        continue

            # \cmd{content} for known wrappers
            if cmd in known_one_arg and j < len(s) and s[j] == "{":
                content, k = _extract_brace_group(s, j)
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
        rows.append(parts)
    return rows


def get_args():
    p = argparse.ArgumentParser(description="Convert LaTeX table rows into a CSV.")
    p.add_argument("--in", dest="in_path", default="in.txt", help="Input file path (default: stdin). Use '-' for stdin.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument(
        "--colnames",
        required=True,
        help="Comma-separated column names, e.g. game,test1,test2,test3,test4",
    )
    return p.parse_args()


def main():
    args = get_args()
    colnames = [c.strip() for c in str(args.colnames).split(",") if c.strip()]
    if not colnames:
        raise SystemExit("Error: --colnames must contain at least 1 name")

    text = _read_text(args.in_path)
    rows = _parse_rows(text)
    if not rows:
        raise SystemExit("Error: no rows parsed from input")

    expected = len(colnames)
    bad = [i for i, r in enumerate(rows, start=1) if len(r) != expected]
    if bad:
        example_i = bad[0]
        raise SystemExit(
            f"Error: row {example_i} has {len(rows[example_i-1])} columns but --colnames has {expected}"
        )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(colnames)
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()


