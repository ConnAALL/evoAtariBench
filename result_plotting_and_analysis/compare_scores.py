import argparse
import csv
import os
import sqlite3
import re

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
except Exception:
    Console = None
    Table = None
    Text = None

try:
    import tabulate
except Exception:
    tabulate = None

EXCLUDE_COLS = {"action_space", "obs_rows", "obs_cols"}


def _load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _select_columns(fieldnames, include_human=False):
    wanted_first = ["game", "environment"]
    cols = []
    for c in wanted_first:
        if c in fieldnames and c not in EXCLUDE_COLS and c not in cols:
            cols.append(c)
    for c in fieldnames:
        if c == "human" and not include_human:
            continue
        if c in EXCLUDE_COLS:
            continue
        if c in cols:
            continue
        cols.append(c)
    return cols


def _load_best_scores(db_path):
    if not os.path.exists(db_path):
        return {}
    con = sqlite3.connect(db_path, timeout=30.0)
    try:
        cur = con.cursor()
        cur.execute("SELECT env_name, MAX(best_fitness) FROM runs GROUP BY env_name")
        out = {}
        for env_name, best in cur.fetchall():
            if env_name is None or best is None:
                continue
            out[str(env_name)] = float(best)
        return out
    finally:
        con.close()


def _fmt_score(x):
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return "NA"
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return f"{xf:.2f}"


def _to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() == "NA":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _style_for_score(v, scope_v):
    if scope_v is None:
        return None
    vv = _to_float(v)
    if vv is None:
        return None
    if vv > scope_v:
        return "red"
    if vv < scope_v:
        return "green"
    return None


def _ansi_colorize(s, color):
    if color == "red":
        return f"\033[31m{s}\033[0m"
    if color == "green":
        return f"\033[32m{s}\033[0m"
    if color == "blue":
        return f"\033[34m{s}\033[0m"
    if color == "orange":
        return f"\033[38;5;208m{s}\033[0m"
    return s


def _normalize_cell(v):
    if v is None:
        return "NA"
    s = str(v)
    if s.strip() == "":
        return "NA"
    return s


def _fill_empty_with_na(rows, headers):
    for r in rows:
        for h in headers:
            r[h] = _normalize_cell(r.get(h, "NA"))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--comp",
        dest="comp",
        required=True,
        choices=["dct"],
        help="Compressor to compare. Currently supported: dct",
    )
    p.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help="Directory containing per-method sqlite DBs for the selected compressor. "
        "Defaults to data/run_data/<comp>_game_sweep.",
    )
    p.add_argument(
        "--base-csv",
        dest="base_csv",
        default=None,
        help="Base benchmark CSV to use as the starting table. Defaults to data/previous_benchmark_scores/subset_results.csv.",
    )
    p.add_argument(
        "--filter",
        "--require-baselines",
        dest="require_baselines",
        type=int,
        default=None,
        help="If set (e.g. 6), only show games that have at least this many non-SCOPE baseline scores present.",
    )
    p.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        help="Save the printed table (after filtering + summary rows) to data/baselines/compare_scores_out.csv.",
    )
    p.add_argument(
        "--include-human",
        dest="include_human",
        action="store_true",
        help="Include the human baseline column in the table/summaries/matrix.",
    )
    return p.parse_args()


def _scope_color_from_rank(rank_v):
    if rank_v is None:
        return None
    try:
        rk = int(rank_v)
    except Exception:
        return None
    if rk == 1:
        return "green"
    if rk in (2, 3):
        return "blue"
    if rk in (4, 5):
        return "orange"
    return None


def _is_summary_row(r):
    return bool(r.get("__summary__"))


def _scope_rank(scope_v, row, method_cols):
    if scope_v is None:
        return None
    vals = []
    for c in method_cols:
        v = _to_float(row.get(c))
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    unique = sorted(set(vals), reverse=True)
    for i, v in enumerate(unique):
        if abs(float(scope_v) - float(v)) < 1e-9:
            return i + 1
    return None

def _method_cols_from_headers(headers):
    return [h for h in headers if h not in ("game", "environment")]

def _times_rank_summaries(rows, method_cols, max_rank=3):
    counts_by_rank = {k: {c: 0 for c in method_cols} for k in range(1, int(max_rank) + 1)}

    for r in rows:
        if _is_summary_row(r):
            continue
        vals = {}
        for c in method_cols:
            v = _to_float(r.get(c))
            if v is not None:
                vals[c] = float(v)
        if not vals:
            continue

        unique = sorted(set(vals.values()), reverse=True)
        for rank in range(1, int(max_rank) + 1):
            if rank > len(unique):
                continue
            target = unique[rank - 1]
            for c, v in vals.items():
                if abs(v - target) < 1e-9:
                    counts_by_rank[rank][c] += 1

    out_rows = []
    for rank in range(1, int(max_rank) + 1):
        out = {
            "__summary__": True,
            "game": f"Times {rank}st" if rank == 1 else (f"Times {rank}nd" if rank == 2 else f"Times {rank}rd"),
        }
        for c in method_cols:
            out[c] = str(int(counts_by_rank[rank][c]))
        out_rows.append(out)
    return out_rows


def _result_reported_summary(rows, method_cols):
    counts = {c: 0 for c in method_cols}

    for r in rows:
        if _is_summary_row(r):
            continue
        for c in method_cols:
            if _to_float(r.get(c)) is not None:
                counts[c] += 1

    out = {"__summary__": True, "game": "Result Reported"}
    for c in method_cols:
        out[c] = str(int(counts[c]))
    return out

def _best_cols_for_row(row, method_cols):
    vals = {c: _to_float(row.get(c)) for c in method_cols}
    present = {c: v for c, v in vals.items() if v is not None}
    if not present:
        return set()
    best = max(present.values())
    return {c for c, v in present.items() if abs(float(v) - float(best)) < 1e-9}


def _print_rich(headers, rows):
    if Console is None or Table is None or Text is None:
        return False

    table = Table(show_header=True, header_style="bold")
    for h in headers:
        table.add_column(h, overflow="fold")
    for r in rows:
        if _is_summary_row(r):
            table.add_row(*[str(r.get(h, "")) for h in headers])
            continue
        method_cols = _method_cols_from_headers(headers)
        best_cols = _best_cols_for_row(r, method_cols)
        cells = []
        for h in headers:
            v = r.get(h, "")
            s = str(v)
            if h in best_cols:
                cells.append(Text(s, style="green"))
            else:
                cells.append(s)
        table.add_row(*cells)

    Console().print(table)
    return True


def _print_tabulate(headers, rows):
    if tabulate is None:
        return False

    body = []
    for r in rows:
        if _is_summary_row(r):
            body.append([str(r.get(h, "")) for h in headers])
            continue
        method_cols = _method_cols_from_headers(headers)
        best_cols = _best_cols_for_row(r, method_cols)
        row = []
        for h in headers:
            v = r.get(h, "")
            s = str(v)
            if h in best_cols:
                s = _ansi_colorize(s, "green")
            row.append(s)
        body.append(row)
    print(tabulate(body, headers=headers, tablefmt="github"))
    return True


def _print_fallback(headers, rows):
    body = []
    for r in rows:
        if _is_summary_row(r):
            body.append([str(r.get(h, "")) for h in headers])
            continue
        method_cols = _method_cols_from_headers(headers)
        best_cols = _best_cols_for_row(r, method_cols)
        row = []
        for h in headers:
            v = r.get(h, "")
            s = str(v)
            if h in best_cols:
                s = _ansi_colorize(s, "green")
            row.append(s)
        body.append(row)
    widths = [len(h) for h in headers]
    for row in body:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    def fmt(row):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in body:
        print(fmt(row))


def _save_csv(headers, rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {h: _normalize_cell(r.get(h, "NA")) for h in headers}
            w.writerow(out)


def _pairwise_beats_counts(rows, method_cols):
    counts = {a: {b: 0 for b in method_cols} for a in method_cols}
    for r in rows:
        if _is_summary_row(r):
            continue
        vals = {c: _to_float(r.get(c)) for c in method_cols}
        for a in method_cols:
            va = vals.get(a)
            if va is None:
                continue
            for b in method_cols:
                if a == b:
                    continue
                vb = vals.get(b)
                if vb is None:
                    continue
                if va > vb:
                    counts[a][b] += 1
    return counts


def _print_pairwise_matrix_rich(method_cols, counts):
    if Console is None or Table is None:
        return False
    table = Table(title="Pairwise wins (row beats column)", show_header=True, header_style="bold")
    table.add_column("method", overflow="fold")
    for c in method_cols:
        table.add_column(str(c), justify="right", overflow="fold")
    for a in method_cols:
        row = [str(a)]
        for b in method_cols:
            if a == b:
                row.append("—")
            else:
                row.append(str(int(counts[a][b])))
        table.add_row(*row)
    Console().print(table)
    return True


def _print_pairwise_matrix_tabulate(method_cols, counts):
    if tabulate is None:
        return False
    headers = ["method"] + list(method_cols)
    body = []
    for a in method_cols:
        row = [a]
        for b in method_cols:
            row.append("—" if a == b else int(counts[a][b]))
        body.append(row)
    print(tabulate(body, headers=headers, tablefmt="github"))
    return True


def _print_pairwise_matrix_fallback(method_cols, counts):
    headers = ["method"] + list(method_cols)
    body = []
    for a in method_cols:
        row = [str(a)]
        for b in method_cols:
            row.append("—" if a == b else str(int(counts[a][b])))
        body.append(row)

    widths = [len(h) for h in headers]
    for row in body:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(str(v)))

    def fmt(row):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))

    print("")
    print("Pairwise wins (row beats column)")
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in body:
        print(fmt(row))

def _safe_slug(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "method"

def _dct_method_from_db_filename(filename):
    # Examples:
    # - dct_dropout_250gen.db -> dropout
    # - dct_quantization_250gen.db -> quantization
    base = os.path.basename(filename)
    if base.lower().endswith(".db"):
        base = base[:-3]
    if base.lower().startswith("dct_"):
        base = base[4:]
    base = re.sub(r"_\d+gen$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_gen$", "", base, flags=re.IGNORECASE)
    return _safe_slug(base)


def main():
    args = _parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    base_csv_path = (
        os.path.abspath(args.base_csv)
        if args.base_csv
        else os.path.join(repo_root, "data", "previous_benchmark_scores", "subset_results.csv")
    )
    fieldnames, rows = _load_rows(base_csv_path)

    # Remember which columns are "baseline" from the base CSV so filters can ignore added columns.
    baseline_cols = [h for h in fieldnames if h not in ("game", "environment")]

    added_method_cols = []
    if args.comp == "dct":
        run_dir = (
            os.path.abspath(args.run_dir)
            if args.run_dir
            else os.path.join(repo_root, "data", "run_data", "dct_game_sweep")
        )
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--run-dir not found or not a directory: {run_dir}")

        db_paths = sorted(
            [
                os.path.join(run_dir, fn)
                for fn in os.listdir(run_dir)
                if fn.lower().endswith(".db")
            ]
        )
        if not db_paths:
            raise SystemExit(f"No .db files found in: {run_dir}")

        scores_by_method = {}
        for db_path in db_paths:
            method = _dct_method_from_db_filename(db_path)
            col = f"dct_{method}"
            scores_by_method[col] = _load_best_scores(db_path)
            added_method_cols.append(col)

        for r in rows:
            env = r.get("environment", "")
            for col, by_env in scores_by_method.items():
                r[col] = _fmt_score(by_env.get(env))

    if args.require_baselines is not None:
        req = int(args.require_baselines)
        if req < 0:
            req = 0

        def ok(r):
            n = 0
            for c in baseline_cols:
                if _to_float(r.get(c)) is not None:
                    n += 1
            return n >= req

        rows = [r for r in rows if ok(r)]

    headers = _select_columns(list(fieldnames) + added_method_cols, include_human=bool(args.include_human))
    method_cols = _method_cols_from_headers(headers)

    rows.extend(_times_rank_summaries(rows, method_cols, max_rank=3))
    rows.append(_result_reported_summary(rows, method_cols))

    _fill_empty_with_na(rows, headers)

    if args.save_csv:
        out_path = os.path.join(here, "baselines", "compare_scores_out.csv")
        _save_csv(headers, rows, out_path)
        print(f"[saved] {out_path}")

    printed = _print_rich(headers, rows)
    if not printed:
        printed = _print_tabulate(headers, rows)
    if not printed:
        _print_fallback(headers, rows)

    counts = _pairwise_beats_counts(rows, method_cols)
    printed_matrix = _print_pairwise_matrix_rich(method_cols, counts)
    if not printed_matrix:
        printed_matrix = _print_pairwise_matrix_tabulate(method_cols, counts)
    if not printed_matrix:
        _print_pairwise_matrix_fallback(method_cols, counts)


if __name__ == "__main__":
    main()
