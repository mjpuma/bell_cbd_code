#!/usr/bin/env python3
"""
inspect_wrds_data.py
====================
A friendly, standalone tool for poking at the extracted WRDS / CbD parquet data
WITHOUT touching any of the analysis code. Use it to sanity-check what's in each
file, preview rows, get quick summary stats, and (carefully) export to CSV so you
can open things in Excel.

Nothing here connects to WRDS or modifies any data -- it only READS the parquet
files that `extract_wrds_sp500.py` already produced in `wrds_sp500_data/`.

----------------------------------------------------------------------------
QUICK START  (run from the repo root)
----------------------------------------------------------------------------
List every dataset, its size on disk, row/column counts:
    python src/inspect_wrds_data.py list

Look at one dataset's schema (column names, types, null counts, date ranges):
    python src/inspect_wrds_data.py info daily_returns

Preview the first rows (like df.head()):
    python src/inspect_wrds_data.py head daily_returns -n 20

Grab a random sample instead of the top rows:
    python src/inspect_wrds_data.py sample membership -n 15

Quick numeric summary (like df.describe()):
    python src/inspect_wrds_data.py stats daily_returns

Export to CSV so you can open it in Excel (lands in the `exports/` folder):
    python src/inspect_wrds_data.py csv membership
    python src/inspect_wrds_data.py csv daily_returns --max-rows 100000   # cap the size
    python src/inspect_wrds_data.py csv daily_returns --force             # I really want the whole thing

You can point at a different data folder with --data-dir, e.g. the w120 variant:
    python src/inspect_wrds_data.py --data-dir wrds_sp500_data_w120 list

----------------------------------------------------------------------------
USING IT INSIDE PYTHON / A NOTEBOOK
----------------------------------------------------------------------------
    from src.inspect_wrds_data import load
    df = load("daily_returns")            # returns a pandas DataFrame
    df = load("daily_returns", columns=["permno", "date", "ret"])  # only some cols
    df.head()

DEPENDENCIES
------------
    pip install pandas pyarrow numpy
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    import pyarrow.parquet as pq
    import pyarrow.dataset as pa_ds
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install pandas pyarrow numpy")


# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "wrds_sp500_data"
DEFAULT_EXPORT_DIR = "exports"   # CSV exports land here (kept out of git)

# Excel's hard row limit is 1,048,576. Past this a CSV is useless in Excel and
# slow everywhere, so we refuse to dump it unless the user opts in.
EXCEL_ROW_LIMIT = 1_048_576
CSV_SOFT_LIMIT = 1_000_000   # warn / require --force above this many rows


# ---------------------------------------------------------------------------
# Discovering datasets
# ---------------------------------------------------------------------------
# A "dataset" here is either a single `name.parquet` file OR a directory of
# parquet shards (e.g. `pair_window_stats/shard_*.parquet`). Both are read the
# same way by pyarrow, so we treat them uniformly and key them by a short name.
def resolve_path(data_dir: str, name: str) -> str:
    """Map a short dataset name to a parquet file or a directory of parquet shards."""
    name = name.strip()
    if name.endswith(".parquet"):
        name = name[: -len(".parquet")]
    file_path = os.path.join(data_dir, name + ".parquet")
    dir_path = os.path.join(data_dir, name)
    if os.path.isfile(file_path):
        return file_path
    if os.path.isdir(dir_path) and _dir_has_parquet(dir_path):
        return dir_path
    available = ", ".join(d["name"] for d in discover(data_dir)) or "(none found)"
    sys.exit(f"Dataset '{name}' not found in '{data_dir}'.\nAvailable: {available}")


def _dir_has_parquet(path: str) -> bool:
    try:
        return any(f.endswith(".parquet") for f in os.listdir(path))
    except OSError:
        return False


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def discover(data_dir: str) -> list[dict]:
    """Find all datasets in `data_dir` (single files + shard directories)."""
    if not os.path.isdir(data_dir):
        sys.exit(f"Data directory '{data_dir}' does not exist. "
                 f"Run the extractor first, or pass --data-dir.")
    found: list[dict] = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if os.path.isfile(full) and entry.endswith(".parquet"):
            found.append({"name": entry[: -len(".parquet")],
                          "path": full, "kind": "file",
                          "size": os.path.getsize(full)})
        elif os.path.isdir(full) and _dir_has_parquet(full):
            found.append({"name": entry, "path": full, "kind": "dir (sharded)",
                          "size": _dir_size_bytes(full)})
    return found


# ---------------------------------------------------------------------------
# Cheap metadata (no full load)
# ---------------------------------------------------------------------------
def quick_meta(path: str) -> dict:
    """Row count + column names WITHOUT reading the data into memory.
    Reads only the parquet footer/metadata, so it's fast even for huge files."""
    if os.path.isdir(path):
        ds = pa_ds.dataset(path, format="parquet")
        return {"rows": ds.count_rows(), "columns": list(ds.schema.names)}
    pf = pq.ParquetFile(path)
    return {"rows": pf.metadata.num_rows, "columns": list(pf.schema_arrow.names)}


def human_size(n: int) -> str:
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:,.1f} {unit}" if unit != "B" else f"{int(val)} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(name: str, data_dir: str = DEFAULT_DATA_DIR,
         columns: list[str] | None = None) -> pd.DataFrame:
    """Load a dataset (file or sharded dir) into a pandas DataFrame.
    Importable: `from src.inspect_wrds_data import load`."""
    path = resolve_path(data_dir, name)
    return pd.read_parquet(path, columns=columns)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_list(args) -> None:
    datasets = discover(args.data_dir)
    if not datasets:
        print(f"No parquet datasets found in '{args.data_dir}'.")
        return
    print(f"Datasets in '{args.data_dir}':\n")
    header = f"{'name':<28} {'kind':<14} {'size':>11} {'rows':>14} {'cols':>5}"
    print(header)
    print("-" * len(header))
    for d in datasets:
        try:
            meta = quick_meta(d["path"])
            rows, ncols = f"{meta['rows']:,}", str(len(meta["columns"]))
        except Exception as e:  # noqa: BLE001
            rows, ncols = f"(error: {e})", "?"
        print(f"{d['name']:<28} {d['kind']:<14} {human_size(d['size']):>11} "
              f"{rows:>14} {ncols:>5}")
    print("\nTip:  python src/inspect_wrds_data.py info <name>   for column details.")


def cmd_info(args) -> None:
    path = resolve_path(args.data_dir, args.name)
    meta = quick_meta(path)
    print(f"Dataset : {args.name}")
    print(f"Path    : {path}")
    print(f"Rows    : {meta['rows']:,}")
    print(f"Columns : {len(meta['columns'])}")

    # Read a small slice to infer dtypes + show real example values cheaply.
    head = pd.read_parquet(path).head(args.scan) if os.path.isdir(path) \
        else _read_head(path, args.scan)

    print("\nSchema (dtype, non-null in scan, example):")
    print(f"  {'column':<22} {'dtype':<16} {'non-null':>9}  example")
    print("  " + "-" * 70)
    for col in meta["columns"]:
        if col in head.columns:
            s = head[col]
            dtype = str(s.dtype)
            nonnull = f"{s.notna().sum()}/{len(s)}"
            ex = s.dropna()
            ex_val = repr(ex.iloc[0]) if len(ex) else "(all null in scan)"
            ex_val = ex_val if len(ex_val) <= 30 else ex_val[:27] + "..."
        else:
            dtype, nonnull, ex_val = "?", "?", ""
        print(f"  {col:<22} {dtype:<16} {nonnull:>9}  {ex_val}")

    # Date-like columns: show min/max so the student can see the span quickly.
    date_cols = [c for c in head.columns
                 if pd.api.types.is_datetime64_any_dtype(head[c])]
    if date_cols:
        full = pd.read_parquet(path, columns=date_cols)
        print("\nDate ranges (full dataset):")
        for c in date_cols:
            print(f"  {c}: {full[c].min()}  ..  {full[c].max()}")
    print(f"\n(Schema/example scanned from first {args.scan:,} rows.)")


def _read_head(path: str, n: int) -> pd.DataFrame:
    """Read just the first ~n rows of a single parquet file without loading it all."""
    pf = pq.ParquetFile(path)
    batches = pf.iter_batches(batch_size=max(1, n))
    try:
        first = next(batches)
    except StopIteration:
        return pd.read_parquet(path).head(0)
    return first.to_pandas().head(n)


def cmd_head(args) -> None:
    path = resolve_path(args.data_dir, args.name)
    df = _read_head(path, args.n) if os.path.isfile(path) \
        else pd.read_parquet(path).head(args.n)
    _print_df(df, f"First {len(df)} rows of '{args.name}'")


def cmd_sample(args) -> None:
    df = load(args.name, args.data_dir)
    n = min(args.n, len(df))
    sample = df.sample(n, random_state=args.seed) if len(df) else df
    _print_df(sample, f"Random sample of {len(sample)} rows from '{args.name}' "
                      f"(seed={args.seed})")


def cmd_stats(args) -> None:
    df = load(args.name, args.data_dir)
    print(f"Summary stats for '{args.name}'  ({len(df):,} rows, {df.shape[1]} cols)\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        # include='all' covers non-numeric columns too (counts, unique, top, freq).
        print(df.describe(include="all").transpose())
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("\nColumns with nulls:")
        for c, n in nulls.items():
            print(f"  {c}: {n:,} ({n / len(df):.1%})")
    else:
        print("\nNo nulls in any column.")


def cmd_csv(args) -> None:
    path = resolve_path(args.data_dir, args.name)
    meta = quick_meta(path)
    rows = meta["rows"]

    # Guard against accidentally producing a giant CSV that Excel can't open.
    cap = args.max_rows
    if cap is None and rows > CSV_SOFT_LIMIT and not args.force:
        print(f"'{args.name}' has {rows:,} rows. That's large:")
        if rows > EXCEL_ROW_LIMIT:
            print(f"  - Excel can only show {EXCEL_ROW_LIMIT:,} rows, so the file "
                  f"won't fully open there.")
        print("  Options:")
        print(f"    --max-rows N   write only the first N rows (e.g. --max-rows 100000)")
        print(f"    --force        write the entire thing anyway")
        sys.exit(1)

    df = pd.read_parquet(path)
    if cap is not None and cap < len(df):
        df = df.head(cap)

    out = args.out or os.path.join(DEFAULT_EXPORT_DIR, args.name + ".csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    size = human_size(os.path.getsize(out))
    note = f" (capped at {cap:,} rows)" if cap is not None and cap < rows else ""
    print(f"Wrote {len(df):,} rows{note} -> {out}  [{size}]")


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------
def _print_df(df: pd.DataFrame, title: str) -> None:
    print(title + "\n")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.max_colwidth", 40):
        print(df.to_string(index=False) if len(df) else "(no rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect / preview / export the extracted WRDS parquet data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"folder holding the parquet files (default: {DEFAULT_DATA_DIR})")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="list all datasets with size/row/column counts")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("info", help="show one dataset's schema, dtypes, date ranges")
    sp.add_argument("name")
    sp.add_argument("--scan", type=int, default=10000,
                    help="rows to scan for dtype/example inference (default 10000)")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("head", help="preview the first N rows")
    sp.add_argument("name")
    sp.add_argument("-n", type=int, default=10, help="rows to show (default 10)")
    sp.set_defaults(func=cmd_head)

    sp = sub.add_parser("sample", help="show a random sample of N rows")
    sp.add_argument("name")
    sp.add_argument("-n", type=int, default=10, help="rows to show (default 10)")
    sp.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("stats", help="numeric/categorical summary (describe + nulls)")
    sp.add_argument("name")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("csv", help="export a dataset to CSV (with a size guard)")
    sp.add_argument("name")
    sp.add_argument("-o", "--out", default=None,
                    help=f"output path (default: {DEFAULT_EXPORT_DIR}/<name>.csv)")
    sp.add_argument("--max-rows", type=int, default=None, help="write only the first N rows")
    sp.add_argument("--force", action="store_true", help="write even very large files")
    sp.set_defaults(func=cmd_csv)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
