#!/usr/bin/env python3
"""
extract_wrds_sp500.py
=====================
Extract S&P 500 daily returns + point-in-time membership from CRSP (the new
CIZ / "_v2" format) via the WRDS API, and compute *continuous co-membership*
pair-eligibility per rolling window, for the CbD crisis-period analysis.

AUTHENTICATION (no credentials are ever stored in this file)
------------------------------------------------------------
This uses the `wrds` package's interactive prompt. On the FIRST run it asks for
your WRDS username and password, then offers to create a `~/.pgpass` file so
every later run connects with no password. You never hardcode anything here.
  - To set your username without typing it each time, optionally export it:
        export WRDS_USERNAME=your_columbia_wrds_id
    (the password is still entered interactively the first time only).

WHERE IT RUNS
-------------
  - Laptop first: leave SAMPLE_TEST = True to pull a short span (2007-2009) and
    a handful of windows, so you can confirm the plumbing before the full pull.
  - Empire AI / HPC later: see the "HPC / EMPIRE AI NOTES" block at the bottom.
    Short version: run the *extraction* on a login node (compute nodes usually
    have no outbound internet), write parquet to shared storage, then run the
    heavy CbD analysis on compute nodes offline.

OUTPUTS  (parquet, into OUTPUT_DIR)
-----------------------------------
  membership.parquet          SPX membership spells: permno, mbrstartdt, mbrenddt
  daily_returns.parquet       panel: permno, date, ret, [prc, vol, shrout]
  trading_calendar.parquet    distinct trading dates (the NYSE calendar)
  window_eligibility.parquet  window_id, win_start, win_end, permno (continuously resident)
  identifiers.parquet         best-effort permno -> permco/ticker/name (if available)

DEPENDENCIES
------------
  pip install wrds pandas pyarrow numpy
"""

import os
import sys
import math
import logging

import numpy as np
import pandas as pd

try:
    import wrds
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install wrds pandas pyarrow numpy")

# ============================ CONFIG ============================
WRDS_USERNAME = os.environ.get("WRDS_USERNAME")   # None -> wrds will prompt for it

SAMPLE_START  = "1990-01-01"
SAMPLE_END    = "2025-12-31"

WINDOW_DAYS   = 60        # trading days per rolling window
STEP_DAYS     = 21        # ~one month of trading days between window starts
COMEMBER_FRAC = 1.00      # 1.00 = strict continuous co-membership; 0.95 = permissive

PERMNO_BATCH  = 400       # chunk size for the daily-returns query (respects WRDS limits)
OUTPUT_DIR    = "wrds_sp500_data"

# Laptop smoke test: pull a short span / few windows first to confirm everything works.
SAMPLE_TEST   = True
TEST_START    = "2007-06-01"
TEST_END      = "2009-12-31"
# ===============================================================

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("wrds_sp500")


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------
def connect():
    """Open a WRDS connection. Prompts for credentials on first run and offers
    to save ~/.pgpass so later runs are passwordless."""
    log.info("Connecting to WRDS (you may be prompted for username/password the first time)...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME) if WRDS_USERNAME else wrds.Connection()
    try:
        db.create_pgpass_file()       # no-op if it already exists; enables passwordless reconnects
        log.info("  ~/.pgpass is in place (future runs will not prompt for a password).")
    except Exception as e:            # noqa: BLE001
        log.warning(f"  could not auto-create ~/.pgpass ({e}); you can still proceed this session.")
    return db


def table_columns(db, schema, table):
    """Lower-cased set of a table's column names. CIZ renamed many columns, so we
    introspect rather than assume names."""
    try:
        desc = db.describe_table(schema, table)
        return set(desc["name"].str.lower())
    except Exception as e:            # noqa: BLE001
        log.warning(f"  describe_table({schema}.{table}) failed: {e}")
        return set()


# --------------------------------------------------------------------------
# Membership  (crsp.dsp500list_v2)
# --------------------------------------------------------------------------
def get_membership(db, start, end):
    """S&P 500 (SPX) membership spells from CRSP CIZ. One row = one spell.
    NB: the old Compustat route (comp.idxcst_his) is defunct; this is the
    current source."""
    log.info("Pulling S&P 500 membership spells (crsp.dsp500list_v2)...")
    df = db.raw_sql(
        f"""
        select permno, mbrstartdt, mbrenddt
        from crsp.dsp500list_v2
        where mbrenddt >= '{start}' and mbrstartdt <= '{end}'
        """,
        date_cols=["mbrstartdt", "mbrenddt"],
    )
    df["permno"] = df["permno"].astype(int)
    log.info(f"  {len(df):,} spells across {df.permno.nunique()} distinct PERMNOs")
    return df


# --------------------------------------------------------------------------
# Daily returns  (crsp.dsf_v2)
# --------------------------------------------------------------------------
def detect_return_column(db):
    """The daily-return field is 'ret' in the WRDS dsf_v2 convenience table on
    some installs and 'dlyret' (the underlying CIZ name) on others. Pick the one
    that exists."""
    cols = table_columns(db, "crsp", "dsf_v2")
    for cand in ("ret", "dlyret"):
        if cand in cols:
            log.info(f"  daily-return column detected: '{cand}'")
            return cand, cols
    log.warning("  could not detect a return column; defaulting to 'ret'")
    return "ret", cols


def get_daily_returns(db, permnos, start, end):
    """Daily returns panel from crsp.dsf_v2, chunked by PERMNO to respect query limits."""
    ret_col, cols = detect_return_column(db)
    # CIZ (_v2) renamed the date/price/volume fields (date -> dlycaldt, prc -> dlyprc,
    # vol -> dlyvol). Detect them the same way as the return column and alias back to
    # the stable output schema (date, ret, [prc, vol, shrout]) so downstream is unchanged.
    date_col = next((c for c in ("date", "dlycaldt", "caldt") if c in cols), "date")
    select_cols = ["permno", f"{date_col} as date", f"{ret_col} as ret"]
    for want, cands in (("prc", ("prc", "dlyprc")), ("vol", ("vol", "dlyvol")),
                        ("shrout", ("shrout", "dlyshrout"))):
        hit = next((c for c in cands if c in cols), None)
        if hit:
            select_cols.append(f"{hit} as {want}")

    permnos = sorted({int(p) for p in permnos})
    n_batches = math.ceil(len(permnos) / PERMNO_BATCH)
    frames = []
    for b in range(n_batches):
        batch = permnos[b * PERMNO_BATCH:(b + 1) * PERMNO_BATCH]
        in_list = ",".join(str(p) for p in batch)
        log.info(f"  returns batch {b + 1}/{n_batches} ({len(batch)} permnos)...")
        chunk = db.raw_sql(
            f"""
            select {", ".join(select_cols)}
            from crsp.dsf_v2
            where {date_col} between '{start}' and '{end}'
              and permno in ({in_list})
            """,
            date_cols=["date"],
        )
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    df["permno"] = df["permno"].astype(int)
    if "prc" in df.columns:
        df["prc"] = df["prc"].abs()   # negative prc = bid/ask midpoint; take magnitude
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    log.info(f"  {len(df):,} return rows")
    return df


# --------------------------------------------------------------------------
# Identifiers (best-effort; CIZ field names vary, so introspect)
# --------------------------------------------------------------------------
def get_identifiers(db, permnos):
    cols = table_columns(db, "crsp", "stksecurityinfohist")
    if "permno" not in cols:
        log.info("  identifiers: stksecurityinfohist not available as expected; skipping.")
        return None
    wanted = [c for c in ("permno", "permco", "ticker", "securitynm", "issuernm",
                          "primaryexch", "sharetype", "securitytype") if c in cols]
    try:
        df = db.raw_sql(f"select distinct {', '.join(wanted)} from crsp.stksecurityinfohist")
        df["permno"] = df["permno"].astype(int)
        keep = set(int(p) for p in permnos)
        return df[df["permno"].isin(keep)].reset_index(drop=True)
    except Exception as e:            # noqa: BLE001
        log.warning(f"  identifiers pull failed (non-fatal): {e}")
        return None


# --------------------------------------------------------------------------
# Windows + continuous co-membership eligibility
# --------------------------------------------------------------------------
def build_windows(trading_days, window_days, step_days):
    """List of (window_id, start_date, end_date) over the trading calendar."""
    td = list(trading_days)
    wins, wid, i = [], 0, 0
    while i + window_days <= len(td):
        wins.append((wid, td[i], td[i + window_days - 1]))
        wid += 1
        i += step_days
    return wins


def compute_eligibility(membership, windows, trading_days, frac):
    """For each window, the PERMNOs continuously resident for >= `frac` of the
    window's trading days (frac=1.0 -> present every trading day of the window).
    Vectorized via a (permno x calendar) membership mask."""
    cal = np.array(pd.to_datetime(pd.Series(trading_days)).values, dtype="datetime64[D]")
    cal_pos = {d: i for i, d in enumerate(cal)}

    permnos = np.sort(membership.permno.unique())
    pidx = {int(p): i for i, p in enumerate(permnos)}
    mask = np.zeros((len(permnos), len(cal)), dtype=bool)

    ms = pd.to_datetime(membership.mbrstartdt).values.astype("datetime64[D]")
    me = pd.to_datetime(membership.mbrenddt).values.astype("datetime64[D]")
    for p, s, e in zip(membership.permno.values, ms, me):
        mask[pidx[int(p)]] |= (cal >= s) & (cal <= e)

    rows = []
    for wid, ws, we in windows:
        i0 = cal_pos[np.datetime64(pd.Timestamp(ws), "D")]
        i1 = cal_pos[np.datetime64(pd.Timestamp(we), "D")] + 1
        coverage = mask[:, i0:i1].mean(axis=1)
        for p in permnos[coverage >= frac]:
            rows.append((wid, ws, we, int(p)))
    return pd.DataFrame(rows, columns=["window_id", "win_start", "win_end", "permno"])


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def save(df, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name + ".parquet")
    try:
        df.to_parquet(path, index=False)
    except Exception as e:            # noqa: BLE001  (pyarrow missing, etc.)
        path = os.path.join(OUTPUT_DIR, name + ".csv")
        df.to_csv(path, index=False)
        log.warning(f"  parquet unavailable ({e}); wrote CSV instead")
    log.info(f"  saved {name}: {len(df):,} rows -> {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    start, end = (TEST_START, TEST_END) if SAMPLE_TEST else (SAMPLE_START, SAMPLE_END)
    if SAMPLE_TEST:
        log.info(f"SAMPLE_TEST mode ON -> short span {start} .. {end} "
                 f"(set SAMPLE_TEST=False for the full {SAMPLE_START}..{SAMPLE_END} pull)")

    db = connect()
    try:
        membership = get_membership(db, start, end)
        save(membership, "membership")

        permnos = membership.permno.unique()

        returns = get_daily_returns(db, permnos, start, end)
        save(returns, "daily_returns")

        calendar = pd.DataFrame({"date": pd.to_datetime(sorted(returns.date.unique()))})
        save(calendar, "trading_calendar")

        idents = get_identifiers(db, permnos)
        if idents is not None and len(idents):
            save(idents, "identifiers")

        windows = build_windows(list(calendar.date), WINDOW_DAYS, STEP_DAYS)
        log.info(f"{len(windows)} rolling windows ({WINDOW_DAYS} td, step {STEP_DAYS} td)")
        if not windows:
            log.warning("No full-length windows in this span (widen the sample). Done.")
            return

        elig = compute_eligibility(membership, windows, list(calendar.date), COMEMBER_FRAC)
        save(elig, "window_eligibility")

        per_win = elig.groupby("window_id").permno.nunique()
        med = int(per_win.median())
        log.info(f"eligible names/window: median {med}, min {int(per_win.min())}, "
                 f"max {int(per_win.max())}")
        log.info(f"approx pairs/window at the median: {med * (med - 1) // 2:,}")
        log.info("Extraction complete.")
    finally:
        db.close()
        log.info("WRDS connection closed.")


if __name__ == "__main__":
    main()


# ==========================================================================
# HPC / EMPIRE AI NOTES
# ==========================================================================
# 1) CREDENTIALS ON THE CLUSTER
#    Run this once interactively on a LOGIN node so the wrds prompt can create
#    ~/.pgpass (it sets file mode 600 for you). Cluster home dirs are shared
#    across nodes, so later batch/compute runs reuse the same ~/.pgpass with no
#    prompt. Confirm:  ls -l ~/.pgpass  ->  -rw------- .
#
# 2) NETWORK EGRESS (the usual gotcha)
#    The WRDS server is PostgreSQL at wrds-pgdata.wharton.upenn.edu:9737.
#    HPC COMPUTE nodes frequently have NO outbound internet. So:
#       - run the EXTRACTION (this script) on a login node or an egress-enabled
#         node, writing parquet to shared/scratch storage; then
#       - run the heavy CbD ANALYSIS on compute nodes fully OFFLINE from those
#         parquet files. Extraction is a one-time, I/O-bound step.
#    If even login nodes are blocked, ask Empire AI support to whitelist
#    wrds-pgdata.wharton.upenn.edu:9737, or extract on your laptop and upload.
#
# 3) ENVIRONMENT
#       module load python            # or: conda activate <env>
#       python -m venv ~/wrdsenv && source ~/wrdsenv/bin/activate
#       pip install wrds pandas pyarrow numpy
#
# 4) DON'T sbatch the extraction
#    The first-ever connect is interactive (it prompts for the password). Do
#    that once on a login node / via `salloc` interactive session. Only after
#    ~/.pgpass exists can a non-interactive sbatch job connect -- and even then,
#    only on a node with egress (see #2).
#
# 5) SCALE
#    Full 1990-2025 pull: ~1,500 unique PERMNOs x ~9,000 trading days ~ 10-15M
#    return rows (a few hundred MB as parquet). The PERMNO_BATCH chunking keeps
#    each query well within WRDS limits. Persist only per-pair summaries from the
#    downstream analysis, not the raw cells.
# ==========================================================================
