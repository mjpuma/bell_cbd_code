"""Re-derive window_eligibility at an alternative rolling-window LENGTH from the
on-disk panel (membership + trading_calendar), WITHOUT re-pulling from WRDS.

Used for the R4 window-length sensitivity check (default 120 trading days, which
roughly doubles the both-large CHSH cell occupancy vs the canonical ~60 td). The
extractor's windowing helpers are reused verbatim so the eligibility definition
cannot drift; only WINDOW_DAYS changes. A new data dir is populated with the fresh
eligibility frame plus symlinks back to the (large, unchanged) daily_returns,
membership, trading_calendar, and sp500_index, so cbd_analysis / networks can run
against it unchanged.

    python src/rebuild_windows.py --src wrds_sp500_data --dst wrds_sp500_data_w120 \
        --window-days 120
"""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from extract_wrds_sp500 import build_windows, compute_eligibility, STEP_DAYS, COMEMBER_FRAC

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rebuild_windows")

# files reused as-is from the source panel (symlinked, never copied/re-pulled)
LINK_FILES = ("daily_returns.parquet", "membership.parquet",
              "trading_calendar.parquet", "sp500_index.parquet")


def _read(src: str, name: str) -> pd.DataFrame:
    path = os.path.join(src, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def rebuild(src: str, dst: str, window_days: int, step_days: int) -> None:
    os.makedirs(dst, exist_ok=True)
    membership = _read(src, "membership.parquet")
    calendar = _read(src, "trading_calendar.parquet")
    cal_dates = list(pd.to_datetime(calendar["date"]).sort_values())

    windows = build_windows(cal_dates, window_days, step_days)
    log.info(f"{len(windows)} rolling windows ({window_days} td, step {step_days} td)")
    if not windows:
        raise SystemExit("no full-length windows; widen the panel or shorten window")

    elig = compute_eligibility(membership, windows, cal_dates, COMEMBER_FRAC)
    proc = os.path.join(dst, "processed")          # derived artifact -> processed/
    os.makedirs(proc, exist_ok=True)
    out = os.path.join(proc, "window_eligibility.parquet")
    elig.to_parquet(out, index=False)
    per_win = elig.groupby("window_id").permno.nunique()
    log.info(f"eligible names/window: median {int(per_win.median())}, "
             f"min {int(per_win.min())}, max {int(per_win.max())}")
    log.info(f"approx pairs/window at the median: "
             f"{int(per_win.median()) * (int(per_win.median()) - 1) // 2:,}")
    log.info(f"  saved window_eligibility: {len(elig):,} rows -> {out}")

    src_abs = os.path.abspath(src)
    for fn in LINK_FILES:
        s = os.path.join(src_abs, fn)
        d = os.path.join(dst, fn)
        if not os.path.exists(s):
            log.warning(f"  source missing, skipping link: {fn}")
            continue
        if os.path.islink(d) or os.path.exists(d):
            os.remove(d)
        os.symlink(s, d)
        log.info(f"  linked {fn} -> {s}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="wrds_sp500_data", help="source data dir")
    ap.add_argument("--dst", default="wrds_sp500_data_w120", help="target data dir")
    ap.add_argument("--window-days", type=int, default=120, help="rolling window length (td)")
    ap.add_argument("--step-days", type=int, default=STEP_DAYS, help="window step (td)")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    rebuild(a.src, a.dst, a.window_days, a.step_days)
