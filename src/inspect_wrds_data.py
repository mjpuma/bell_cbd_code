#!/usr/bin/env python3
"""
inspect_wrds_data.py
====================
A tiny, read-only viewer for the ORIGINAL WRDS data. Use it to see what's in each
raw file and (optionally) export a slice to CSV for Excel. It never connects to
WRDS, never writes into the data folder, and ignores the analysis outputs under
`wrds_sp500_data/processed/` -- it only shows the raw pull:

    daily_returns, membership, identifiers, trading_calendar, sp500_index

----------------------------------------------------------------------------
USAGE  (run from the repo root)
----------------------------------------------------------------------------
    python src/inspect_wrds_data.py list                 # the raw files: size, rows, cols
    python src/inspect_wrds_data.py view daily_returns    # columns, types, first rows
    python src/inspect_wrds_data.py view membership -n 20 # show 20 rows
    python src/inspect_wrds_data.py csv membership        # export to exports/membership.csv

Big files (daily_returns is ~7M rows) won't fit in Excel, so `csv` exports a
capped slice by default; use --max-rows N or --force to change that.

In Python / a notebook:
    from src.inspect_wrds_data import load
    df = load("daily_returns")

DEPENDENCIES:  pip install pandas pyarrow
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install pandas pyarrow")

DATA_DIR = "wrds_sp500_data"   # raw WRDS pull lives at the top level of this folder
EXPORT_DIR = "exports"         # CSV exports land here (gitignored)
EXCEL_ROW_LIMIT = 1_048_576    # Excel's hard row cap; default CSV slice stays under it


def _path(name: str) -> str:
    """Resolve a raw dataset name to its parquet file (top level only)."""
    name = name[:-8] if name.endswith(".parquet") else name
    p = os.path.join(DATA_DIR, name + ".parquet")
    if not os.path.isfile(p):
        avail = ", ".join(_raw_files()) or "(none found)"
        sys.exit(f"'{name}' is not a raw data file in '{DATA_DIR}'.\nAvailable: {avail}")
    return p


def _raw_files() -> list[str]:
    """Names of the raw parquet files at the top level of DATA_DIR (no subfolders)."""
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"Data folder '{DATA_DIR}' not found. Put the WRDS data there first.")
    return sorted(f[:-8] for f in os.listdir(DATA_DIR)
                  if f.endswith(".parquet") and os.path.isfile(os.path.join(DATA_DIR, f)))


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{int(x)} {unit}" if unit == "B" else f"{x:,.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


def load(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a raw dataset into a pandas DataFrame (importable helper)."""
    return pd.read_parquet(_path(name), columns=columns)


def cmd_list(_args) -> None:
    names = _raw_files()
    if not names:
        print(f"No raw parquet files in '{DATA_DIR}'.")
        return
    print(f"Raw WRDS data in '{DATA_DIR}/':\n")
    header = f"{'name':<20} {'size':>11} {'rows':>14} {'cols':>5}"
    print(header)
    print("-" * len(header))
    for name in names:
        p = os.path.join(DATA_DIR, name + ".parquet")
        md = pq.ParquetFile(p).metadata
        print(f"{name:<20} {_human(os.path.getsize(p)):>11} "
              f"{md.num_rows:>14,} {md.num_columns:>5}")
    print(f"\n(Analysis outputs live in '{DATA_DIR}/processed/' and are not shown here.)")


def cmd_view(args) -> None:
    p = _path(args.name)
    pf = pq.ParquetFile(p)
    nrows = pf.metadata.num_rows
    # Read only the first rows we need to display, without loading the whole file.
    head = next(pf.iter_batches(batch_size=max(1, args.n))).to_pandas().head(args.n)

    print(f"{args.name}  —  {nrows:,} rows, {pf.metadata.num_columns} columns\n")
    print("columns:")
    for col in head.columns:
        s = head[col]
        ex = s.dropna()
        ex_val = repr(ex.iloc[0]) if len(ex) else "(null)"
        print(f"  {col:<16} {str(s.dtype):<16} e.g. {ex_val}")
    date_cols = [c for c in head.columns if pd.api.types.is_datetime64_any_dtype(head[c])]
    if date_cols:
        full = pd.read_parquet(p, columns=date_cols)
        print("\ndate range:")
        for c in date_cols:
            print(f"  {c}: {full[c].min()}  ..  {full[c].max()}")
    print(f"\nfirst {len(head)} rows:\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(head.to_string(index=False))


def cmd_csv(args) -> None:
    p = _path(args.name)
    nrows = pq.ParquetFile(p).metadata.num_rows

    cap = args.max_rows
    if cap is None and not args.force:
        if nrows > EXCEL_ROW_LIMIT:
            cap = 100_000
            print(f"'{args.name}' has {nrows:,} rows (> Excel's {EXCEL_ROW_LIMIT:,} limit). "
                  f"Exporting the first {cap:,}. Use --max-rows N or --force for more.")
    df = pd.read_parquet(p)
    if cap is not None and cap < len(df):
        df = df.head(cap)

    out = args.out or os.path.join(EXPORT_DIR, args.name + ".csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    note = f" (capped at {cap:,})" if cap is not None and cap < nrows else ""
    print(f"Wrote {len(df):,} rows{note} -> {out}  [{_human(os.path.getsize(out))}]")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="View the original WRDS data (read-only).")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the raw data files").set_defaults(func=cmd_list)

    sp = sub.add_parser("view", help="show a file's columns, types, and first rows")
    sp.add_argument("name")
    sp.add_argument("-n", type=int, default=10, help="rows to show (default 10)")
    sp.set_defaults(func=cmd_view)

    sp = sub.add_parser("csv", help="export to CSV (exports/<name>.csv) for Excel")
    sp.add_argument("name")
    sp.add_argument("-o", "--out", default=None, help=f"output path (default {EXPORT_DIR}/<name>.csv)")
    sp.add_argument("--max-rows", type=int, default=None, help="write only the first N rows")
    sp.add_argument("--force", action="store_true", help="write the whole file (may be huge)")
    sp.set_defaults(func=cmd_csv)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
