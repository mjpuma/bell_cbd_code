#!/usr/bin/env python3
"""
cbd_analysis.py
===============
Downstream Contextuality-by-Default (CbD) analysis for the S&P 500 crisis study.
Consumes the four parquet files written by `extract_wrds_sp500.py` and produces
per-pair, per-window CbD statistics.

SOURCE OF TRUTH
---------------
All definitions and notation follow `cbd_sp500_analysis.(tex|pdf)`. The core
estimators (s_odd, Delta, CTX) are implemented and unit-tested against the
spec's worked numbers below. Do NOT change those formulas without re-deriving;
run `python cbd_analysis.py --test` after any edit to the math.

INPUTS (parquet in --data-dir, from the extractor)
--------------------------------------------------
  membership.parquet         permno, mbrstartdt, mbrenddt
  daily_returns.parquet      permno, date, ret, [prc, vol, shrout]
  trading_calendar.parquet   date
  window_eligibility.parquet window_id, win_start, win_end, permno

OUTPUT
------
  pair_window_stats.parquet  one row per (window_id, permno_a, permno_b):
     E00,E01,E10,E11, a00..a11, b00..b11, N00..N11, s_odd, delta, ctx, valid,
     regime (crisis/calm)

WHAT IS DONE vs. WHAT IS A STUB
-------------------------------
  DONE  : core math (tested), parquet I/O, regime classification, per-pair
          statistics, the per-window driver, the --sample smoke mode.
  STUB  : crisis/calm labeling source (load_crisis_labels) and the classical
          deflation/null reproduction (classical_null_reproduction). Both are
          modeling choices flagged with TODO; interfaces are pinned.

This script cannot connect to WRDS or run on real data in an AI sandbox.
Validate logic on synthetic inputs (see the tests) before trusting any run.
"""

from __future__ import annotations
import os
import argparse
import logging
from itertools import combinations

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cbd")

# Fixed cell order used EVERYWHERE: index 0->(0,0) 1->(0,1) 2->(1,0) 3->(1,1).
# Regime convention (per spec): 0 = large-move (|R| >= theta), 1 = small-move.
CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]


# ==========================================================================
# CORE MATH  (implemented and unit-tested; matches the spec exactly)
# ==========================================================================
def chsh_s_values(E):
    """The four one-minus-sign CHSH combinations, in cell order [00,01,10,11].
        S1 = E00+E01+E10-E11 ; S2 = E00+E01-E10+E11
        S3 = E00-E01+E10+E11 ; S4 = -E00+E01+E10+E11
    """
    E00, E01, E10, E11 = E
    return np.array([
        E00 + E01 + E10 - E11,
        E00 + E01 - E10 + E11,
        E00 - E01 + E10 + E11,
        -E00 + E01 + E10 + E11,
    ])


def s_odd(E):
    """s_odd = max over the four one-minus-sign combinations of |S_i|."""
    return float(np.max(np.abs(chsh_s_values(E))))


def delta(a_marg, b_marg):
    """Inconsistency (direct-influence) term. Connections (spec):
        A0: (0,0)~(0,1) ; A1: (1,0)~(1,1) ; B0: (0,0)~(1,0) ; B1: (0,1)~(1,1).
    a_marg, b_marg are length-4 arrays in cell order [00,01,10,11].
    """
    a00, a01, a10, a11 = a_marg
    b00, b01, b10, b11 = b_marg
    return float(abs(a00 - a01) + abs(a10 - a11) + abs(b00 - b10) + abs(b01 - b11))


def ctx(E, a_marg, b_marg, n=4):
    """CbD criterion statistic: CTX = s_odd - Delta - (n-2). CTX > 0 == contextual."""
    return s_odd(E) - delta(a_marg, b_marg) - (n - 2)


# ==========================================================================
# REGIME CLASSIFICATION + PER-PAIR STATISTICS  (implemented)
# ==========================================================================
def window_threshold(abs_ret, q=0.5):
    """Per-stock magnitude threshold within a window: the q-quantile of |R| over
    the window. Spec default q=0.5 (the median, ~balanced regimes). The robustness
    sweep raises q so 'large-move' (regime 0, |R| >= theta) becomes rarer; this is
    the per-stock, within-window quantile (cross-sectional vol differences must NOT
    enter the regime definition)."""
    return float(np.quantile(abs_ret, q))


def assign_regime(abs_ret, theta):
    """0 = large-move (|R| >= theta), 1 = small-move (|R| < theta)."""
    return np.where(abs_ret >= theta, 0, 1).astype(int)


def compute_pair_window(a, b, x, y, n_min):
    """CbD statistics for one pair over one window.
    a, b : daily signs in {-1,+1} (days where either sign is 0 must be removed
           by the caller). x, y : daily regimes in {0,1} for stock A, B.
    Returns a dict with E[4], a_marg[4], b_marg[4], N[4], s_odd, delta, ctx, valid.
    """
    a = np.asarray(a); b = np.asarray(b); x = np.asarray(x); y = np.asarray(y)
    E = np.full(4, np.nan); am = np.full(4, np.nan); bm = np.full(4, np.nan)
    N = np.zeros(4, dtype=int)
    for k, (cx, cy) in enumerate(CELLS):
        m = (x == cx) & (y == cy)
        N[k] = int(m.sum())
        if N[k] > 0:
            E[k] = float(np.mean(a[m] * b[m]))
            am[k] = float(np.mean(a[m]))
            bm[k] = float(np.mean(b[m]))
    valid = bool(N.min() >= n_min)
    out = dict(E=E, a_marg=am, b_marg=bm, N=N, valid=valid,
               s_odd=np.nan, delta=np.nan, ctx=np.nan)
    if valid:
        out["s_odd"] = s_odd(E)
        out["delta"] = delta(am, bm)
        out["ctx"] = ctx(E, am, bm)
    return out


# ==========================================================================
# PER-WINDOW DRIVER
# ==========================================================================
def _run_window_loop(window_id, win_start, win_end, permnos, ret_wide, n_min,
                     theta_q=0.5, thresholds=None):
    """Reference O(n^2) implementation. Kept verbatim as the correctness oracle
    for `run_window` (see `_test_vectorized_equivalence`). Not called in the hot
    path. Compute CbD stats for every eligible pair in one window.
    ret_wide : DataFrame indexed by date, columns = permno, values = ret,
               already restricted to this window's trading days and permnos.
    theta_q  : per-stock |R| quantile defining the large/small regime split.
    thresholds : optional {permno: theta} to FIX the per-stock regime threshold
                 instead of recomputing it on `ret_wide` (used by the
                 regime-preserving split-half reliability). This changes only the
                 threshold selection, never the locked s_odd/delta/ctx formulas.
    Returns a list of result dicts (one per valid-or-invalid pair).
    """
    cols = [p for p in permnos if p in ret_wide.columns]
    R = ret_wide[cols]
    absR = R.abs()
    if thresholds is not None:
        thetas = {p: thresholds.get(p, np.nan) for p in cols}
    else:
        thetas = {p: window_threshold(absR[p].dropna().values, theta_q) for p in cols}
    sign = {p: np.sign(R[p].values) for p in cols}                 # in {-1,0,1}
    regime = {p: assign_regime(absR[p].values, thetas[p]) for p in cols}

    rows = []
    for pa, pb in combinations(cols, 2):
        a = sign[pa]; b = sign[pb]; x = regime[pa]; y = regime[pb]
        keep = (a != 0) & (b != 0) & ~np.isnan(R[pa].values) & ~np.isnan(R[pb].values)
        res = compute_pair_window(a[keep], b[keep], x[keep], y[keep], n_min)
        rows.append(_pack_row(window_id, win_start, win_end, pa, pb,
                              res["E"], res["a_marg"], res["b_marg"], res["N"],
                              res["s_odd"], res["delta"], res["ctx"], res["valid"]))
    return rows


def _pack_row(window_id, win_start, win_end, pa, pb, E, am, bm, N,
              s_odd_v, delta_v, ctx_v, valid_v):
    """Assemble one output row in the fixed schema (shared by loop + vectorized)."""
    return {
        "window_id": window_id, "win_start": win_start, "win_end": win_end,
        "permno_a": pa, "permno_b": pb,
        "E00": E[0], "E01": E[1], "E10": E[2], "E11": E[3],
        "a00": am[0], "a01": am[1], "a10": am[2], "a11": am[3],
        "b00": bm[0], "b01": bm[1], "b10": bm[2], "b11": bm[3],
        "N00": int(N[0]), "N01": int(N[1]), "N10": int(N[2]), "N11": int(N[3]),
        "s_odd": s_odd_v, "delta": delta_v, "ctx": ctx_v, "valid": bool(valid_v),
    }


def _safe_div(num, den):
    """Element-wise num/den, returning NaN where den == 0 (matches the loop's
    'NaN in empty cell' convention)."""
    out = np.full(num.shape, np.nan, dtype=float)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def run_window(window_id, win_start, win_end, permnos, ret_wide, n_min, theta_q=0.5,
               thresholds=None):
    """Vectorized per-window driver. Mathematically identical to
    `_run_window_loop` (asserted bit-for-bit in `_test_vectorized_equivalence`),
    but replaces the O(n^2) Python pair loop + per-pair boolean masking with a
    handful of (stocks x stocks) matrix products over the sign / regime matrices.

    ret_wide : DataFrame indexed by date, columns = permno, values = ret,
               already restricted to this window's trading days and permnos.
    theta_q  : per-stock |R| quantile defining the large/small regime split.
    thresholds : optional {permno: theta} to FIX the per-stock regime threshold
                 instead of recomputing it on `ret_wide` (regime-preserving
                 split-half reliability). Threshold selection only -- the locked
                 s_odd/delta/ctx formulas are unchanged.
    Returns a list of result dicts (one per pair), in `combinations(cols, 2)`
    order so callers see the same ordering as the reference loop.
    """
    cols = [p for p in permnos if p in ret_wide.columns]
    S = len(cols)
    if S < 2:
        return []

    R = ret_wide[cols].to_numpy(dtype=float)               # days x stocks
    absR = np.abs(R)
    nan = np.isnan(R)
    sgn = np.sign(R)
    sgn[nan] = 0.0                                          # keep matmuls NaN-free
    valid = (~nan) & (sgn != 0.0)                          # per-stock usable days

    # Per-stock threshold = q-quantile of finite |R| over the window (== loop),
    # unless fixed thresholds are injected (regime-preserving reliability).
    if thresholds is not None:
        thetas = np.array([thresholds.get(cols[i], np.nan) for i in range(S)])
    else:
        thetas = np.empty(S)
        for i in range(S):
            col = absR[~nan[:, i], i]
            thetas[i] = np.quantile(col, theta_q) if col.size else np.nan
    with np.errstate(invalid="ignore"):
        large = absR >= thetas[None, :]                    # NaN -> False -> small

    # Per-leg masks (days x stocks): U0 = usable large-move, U1 = usable small-move.
    U = {0: (valid & large).astype(float), 1: (valid & ~large).astype(float)}
    AU = {0: sgn * U[0], 1: sgn * U[1]}

    iu, ju = np.triu_indices(S, 1)                         # combinations(cols, 2) order
    N_cols, E_cols, A_cols, B_cols = [], [], [], []
    for (cx, cy) in CELLS:
        Nk = U[cx].T @ U[cy]                               # cell counts, S x S
        SAB = AU[cx].T @ AU[cy]                            # sum a*b in cell
        SA = AU[cx].T @ U[cy]                              # sum a in cell
        SB = U[cx].T @ AU[cy]                              # sum b in cell
        nk = Nk[iu, ju]
        N_cols.append(np.rint(nk).astype(int))
        E_cols.append(_safe_div(SAB[iu, ju], nk))
        A_cols.append(_safe_div(SA[iu, ju], nk))
        B_cols.append(_safe_div(SB[iu, ju], nk))

    N = np.column_stack(N_cols)                            # npairs x 4 (cell order)
    E = np.column_stack(E_cols)
    A = np.column_stack(A_cols)
    B = np.column_stack(B_cols)
    valid_pair = N.min(axis=1) >= n_min

    # Vectorized s_odd / delta / ctx over the four one-minus-sign combinations.
    E0, E1, E2, E3 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
    Smat = np.column_stack([
        E0 + E1 + E2 - E3,
        E0 + E1 - E2 + E3,
        E0 - E1 + E2 + E3,
        -E0 + E1 + E2 + E3,
    ])
    s_odd_v = np.max(np.abs(Smat), axis=1)
    delta_v = (np.abs(A[:, 0] - A[:, 1]) + np.abs(A[:, 2] - A[:, 3])
               + np.abs(B[:, 0] - B[:, 2]) + np.abs(B[:, 1] - B[:, 3]))
    ctx_v = s_odd_v - delta_v - 2.0
    # Non-valid pairs carry NaN scores (== loop, which never computes them).
    s_odd_v = np.where(valid_pair, s_odd_v, np.nan)
    delta_v = np.where(valid_pair, delta_v, np.nan)
    ctx_v = np.where(valid_pair, ctx_v, np.nan)

    rows = []
    for r in range(len(iu)):
        rows.append(_pack_row(
            window_id, win_start, win_end, cols[iu[r]], cols[ju[r]],
            E[r], A[r], B[r], N[r],
            s_odd_v[r], delta_v[r], ctx_v[r], valid_pair[r]))
    return rows


# ==========================================================================
# I/O  (wired to the extractor's exact schema)
# ==========================================================================
def _read(data_dir, name):
    p = os.path.join(data_dir, name + ".parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    p = os.path.join(data_dir, name + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=True)
    raise FileNotFoundError(f"missing {name}.parquet/.csv in {data_dir}")


def load_data(data_dir):
    d = {n: _read(data_dir, n) for n in
         ("membership", "daily_returns", "trading_calendar", "window_eligibility")}
    d["daily_returns"]["date"] = pd.to_datetime(d["daily_returns"]["date"])
    d["window_eligibility"]["win_start"] = pd.to_datetime(d["window_eligibility"]["win_start"])
    d["window_eligibility"]["win_end"] = pd.to_datetime(d["window_eligibility"]["win_end"])
    for c in ("permno",):
        d["daily_returns"][c] = d["daily_returns"][c].astype(int)
        d["window_eligibility"][c] = d["window_eligibility"][c].astype(int)
    log.info(f"loaded: {len(d['daily_returns']):,} return rows, "
             f"{d['window_eligibility'].window_id.nunique()} windows")
    return d


# ==========================================================================
# CRISIS / CALM LABELS  (external indicator -- never from the pairs' returns)
# ==========================================================================
# US recessions (NBER peak-month start .. trough-month end), covering the sample.
# Macroeconomic and fully exogenous to any constituent pair's daily return path.
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def _read_indicator(path):
    """Read a crisis-indicator file (parquet or csv)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Named crisis taxonomy -- a CROSS-CRISIS OVERLAY, *not* the binary labeler.
# --------------------------------------------------------------------------
DEFAULT_CRISES_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "crises.csv")


def load_crisis_taxonomy(path=None):
    """Load the named crisis sub-periods (config/crises.csv) as a tidy DataFrame
    with parsed Timestamp `start`/`end` columns.

    This is a shared overlay for cross-crisis comparison (financial/food/energy/
    mixed). It is deliberately SEPARATE from `load_crisis_labels`: the binary
    crisis/calm driver stays VIX/NBER-based; do NOT wire this taxonomy into it.
    """
    path = path or DEFAULT_CRISES_CSV
    if not os.path.exists(path):
        log.warning(f"load_crisis_taxonomy: '{path}' not found; returning empty.")
        return pd.DataFrame(columns=["name", "start", "end", "type", "notes"])
    tax = pd.read_csv(path)
    tax["start"] = pd.to_datetime(tax["start"])
    tax["end"] = pd.to_datetime(tax["end"])
    return tax


def tag_windows_with_crises(window_ids, win_bounds, taxonomy=None):
    """Overlay: map each window_id -> list of named crises it overlaps (may be
    empty, or several). For cross-crisis comparison only; does not affect the
    binary label. A food crisis that overlaps NO NBER/VIX-stressed window (so it
    is 'calm' under the binary labeler) is itself an important mismatch to flag."""
    tax = taxonomy if taxonomy is not None else load_crisis_taxonomy()
    out = {}
    for wid in window_ids:
        ws, we = pd.Timestamp(win_bounds[wid][0]), pd.Timestamp(win_bounds[wid][1])
        hits = [r["name"] for _, r in tax.iterrows()
                if not (we < r["start"] or ws > r["end"])]
        out[wid] = hits
    return out


def _label_by_spans(window_ids, win_bounds, spans):
    """A window is 'crisis' iff [win_start, win_end] overlaps any (start, end) span."""
    labels = {}
    for wid in window_ids:
        ws, we = pd.Timestamp(win_bounds[wid][0]), pd.Timestamp(win_bounds[wid][1])
        hit = any(not (we < s or ws > e) for s, e in spans)
        labels[wid] = "crisis" if hit else "calm"
    return labels


def _label_by_vix(window_ids, win_bounds, df, cols, threshold, vix_quantile, vix_agg):
    """A window is 'crisis' iff its aggregated VIX exceeds a threshold.
    Default threshold: the sample median of the per-window VIX aggregate
    (a rolling/relative cut). Pass an explicit `threshold` for a fixed level."""
    if "date" not in cols:
        log.warning("VIX source has no 'date' column; labeling all 'calm'.")
        return {wid: "calm" for wid in window_ids}
    vix = pd.Series(pd.to_numeric(df[cols["vix"]], errors="coerce").values,
                    index=pd.to_datetime(df[cols["date"]])).sort_index().dropna()
    agg = {}
    for wid in window_ids:
        ws, we = pd.Timestamp(win_bounds[wid][0]), pd.Timestamp(win_bounds[wid][1])
        w = vix[(vix.index >= ws) & (vix.index <= we)]
        agg[wid] = float(getattr(w, vix_agg)()) if len(w) else np.nan
    vals = np.array([agg[w] for w in window_ids], dtype=float)
    thr = float(threshold) if threshold is not None else float(np.nanquantile(vals, vix_quantile))
    log.info(f"load_crisis_labels(VIX): threshold={thr:.2f} "
             f"(agg='{vix_agg}', quantile={vix_quantile}, fixed={threshold is not None})")
    return {wid: ("crisis" if (not np.isnan(agg[wid]) and agg[wid] > thr) else "calm")
            for wid in window_ids}


def load_crisis_labels(window_ids, win_bounds, source=None, *,
                       threshold=None, vix_quantile=0.5, vix_agg="mean"):
    """Return {window_id: 'crisis'|'calm'} from an indicator EXTERNAL to the
    pairs' own returns (spec §2.6). The label is a clean, swappable input; it is
    *never* derived from the returns being analyzed.

    Parameters
    ----------
    window_ids : iterable of window_id
    win_bounds : {window_id: (win_start, win_end)}  (Timestamps or parseable)
    source     : selects the indicator (the --crisis-source CLI value):
        * None            -> every window 'calm' (lets the pipeline run end-to-end).
        * "nber"          -> built-in US NBER recession spans (see NBER_RECESSIONS).
        * path to a file  -> parquet/csv, auto-detected by its columns:
            - VIX file : columns {date, vix}; one row per trading day. Window is
              'crisis' if its aggregated VIX (mean over the window's days, by
              default) exceeds `threshold` (default: sample-median of the window
              aggregates -- a relative cut; pass a number for a fixed level).
            - NBER/recession-span file : columns {start, end}; one row per
              recession. Window is 'crisis' if it overlaps any span.
    threshold    : fixed VIX level; if None, use the sample-median aggregate.
    vix_quantile : quantile used to derive the relative threshold (default 0.5).
    vix_agg      : per-window VIX aggregator ('mean' | 'median' | 'max').

    The expected file formats above are intentionally minimal so the indicator
    is trivially swappable (CBOE VIX export, WRDS OptionMetrics, NBER table, ...).
    """
    window_ids = list(window_ids)
    if source is None:
        log.info("load_crisis_labels: no --crisis-source; labeling every window 'calm'.")
        return {wid: "calm" for wid in window_ids}

    src = str(source)
    if src.lower() in ("nber", "nber:builtin"):
        spans = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in NBER_RECESSIONS]
        log.info(f"load_crisis_labels: NBER built-in ({len(spans)} recession spans).")
        return _label_by_spans(window_ids, win_bounds, spans)

    if not os.path.exists(src):
        log.warning(f"load_crisis_labels: source '{src}' not found; labeling all 'calm'.")
        return {wid: "calm" for wid in window_ids}

    df = _read_indicator(src)
    cols = {c.lower(): c for c in df.columns}
    if "vix" in cols:
        return _label_by_vix(window_ids, win_bounds, df, cols,
                             threshold, vix_quantile, vix_agg)
    if "start" in cols and "end" in cols:
        spans = [(pd.Timestamp(s), pd.Timestamp(e))
                 for s, e in zip(df[cols["start"]], df[cols["end"]])]
        log.info(f"load_crisis_labels: recession-span file ({len(spans)} spans).")
        return _label_by_spans(window_ids, win_bounds, spans)

    log.warning("load_crisis_labels: unrecognized source schema "
                f"(columns={list(df.columns)}); expected a 'vix' column or "
                "'start'/'end' columns. Labeling all 'calm'.")
    return {wid: "calm" for wid in window_ids}


# ==========================================================================
# CLASSICAL NULL / DEFLATION  (the central result)
# ==========================================================================
# A purely classical generator of a pair's daily returns -- NO contextual
# ingredient. Three ingredients, mapped to the spec (§pipeline Step 3):
#   (1) two-state (large/small) magnitude mixture per stock  -> the regimes;
#   (2) asymmetric magnitude->sign 'leverage' link           -> per-cell marginals;
#   (3) magnitude-state-dependent sign correlation rho_xy    -> per-cell coupling.
# Signs are drawn from a single classical joint per day, conditioned on the
# (endogenous) magnitude cell. Pushing the simulation back through the SAME
# median-split pipeline reproduces the empirical (s_odd, delta, ctx) -- so any
# apparent contextuality is the post-selection artifact (spec §post-selection).

def _draw_signs(pA, pB, rho, n, rng):
    """Draw n iid sign pairs (a, b) in {-1,+1} with P(a=+1)=pA, P(b=+1)=pB and
    Pearson correlation rho, via the bivariate-Bernoulli joint (clipped to the
    simplex if rho is infeasible for the given marginals)."""
    sA = np.sqrt(max(pA * (1 - pA), 0.0))
    sB = np.sqrt(max(pB * (1 - pB), 0.0))
    p11 = pA * pB + rho * sA * sB
    p10 = pA - p11
    p01 = pB - p11
    p00 = 1.0 - pA - pB + p11
    probs = np.clip(np.array([p00, p01, p10, p11]), 0.0, None)
    tot = probs.sum()
    probs = probs / tot if tot > 0 else np.full(4, 0.25)
    idx = rng.choice(4, size=n, p=probs)
    amap = np.array([-1, -1, 1, 1])      # 00,01,10,11 -> a
    bmap = np.array([-1, 1, -1, 1])      # 00,01,10,11 -> b
    return amap[idx].astype(float), bmap[idx].astype(float)


def _corr_pm1(a, b):
    """Pearson correlation of two {-1,+1} arrays; 0 if either is constant."""
    if len(a) < 2:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def fit_pair_generator(retA, retB, n_min=10, theta_q=0.5):
    """Fit the classical generator to one pair-window's aligned daily returns.
    Returns a params dict, or None if there are too few usable days.

    theta_q : per-stock |R| quantile for the large/small regime split, so the
              null can be fit under the SAME threshold condition as the empirical
              analysis it is the null for (matters for the high-theta sweep).

    Params schema (also accepted directly by `simulate_pair`, e.g. in tests):
      scaleA/scaleB : (mean|R| large-state, mean|R| small-state)
      pos_freq      : overall P(usable day) helper (unused at sim time)
      cells         : {(x,y): (pA, pB, rho)} for the four magnitude cells, where
                      pA=P(a=+1 | cell), pB=P(b=+1 | cell), rho=corr(a,b | cell).
    """
    retA = np.asarray(retA, dtype=float)
    retB = np.asarray(retB, dtype=float)
    keep = (~np.isnan(retA) & ~np.isnan(retB)
            & (np.sign(retA) != 0) & (np.sign(retB) != 0))
    a = np.sign(retA[keep]); b = np.sign(retB[keep])
    mA = np.abs(retA[keep]); mB = np.abs(retB[keep])
    if keep.sum() < 4 * n_min:
        return None
    thA = window_threshold(mA, theta_q); thB = window_threshold(mB, theta_q)
    x = assign_regime(mA, thA); y = assign_regime(mB, thB)   # 0=large, 1=small

    def _scale(m, reg):
        lo = m[reg == 0]; hi = m[reg == 1]
        return (float(lo.mean()) if lo.size else float(m.mean()),
                float(hi.mean()) if hi.size else float(m.mean()))

    cells = {}
    for (cx, cy) in CELLS:
        sel = (x == cx) & (y == cy)
        if sel.sum() >= 2:
            aa, bb = a[sel], b[sel]
            pA = float((aa == 1).mean()); pB = float((bb == 1).mean())
            cells[(cx, cy)] = (pA, pB, _corr_pm1(aa, bb))
        else:
            cells[(cx, cy)] = (0.5, 0.5, 0.0)
    return {"scaleA": _scale(mA, x), "scaleB": _scale(mB, y), "cells": cells}


def simulate_pair(params, n_days, rng, theta_q=0.5):
    """Simulate n_days of (retA, retB) from a fitted/explicit generator.

    Magnitudes come from a two-state mixture per stock; the *regime* is then the
    pipeline's quantile split at `theta_q` (so it matches the downstream estimator
    exactly, including under the high-theta sweep), and signs are drawn from the
    per-cell classical joint conditioned on that regime.
    Returns (retA, retB) as signed-magnitude arrays.
    """
    scaleA = params["scaleA"]; scaleB = params["scaleB"]; cells = params["cells"]

    def _mags(scale):
        state = rng.integers(0, 2, size=n_days)               # 0=large,1=small mixture
        sc = np.where(state == 0, max(scale[0], 1e-12), max(scale[1], 1e-12))
        return rng.exponential(sc)

    mA = _mags(scaleA); mB = _mags(scaleB)
    xs = assign_regime(mA, window_threshold(mA, theta_q))     # pipeline quantile split
    ys = assign_regime(mB, window_threshold(mB, theta_q))

    a = np.empty(n_days); b = np.empty(n_days)
    for (cx, cy) in CELLS:
        sel = (xs == cx) & (ys == cy)
        n = int(sel.sum())
        if n:
            pA, pB, rho = cells[(cx, cy)]
            aa, bb = _draw_signs(pA, pB, rho, n, rng)
            a[sel] = aa; b[sel] = bb
    return a * mA, b * mB


def classical_null_reproduction(pair_window_stats, returns_panel, *,
                                n_min=10, max_pairs=None, seed=0, valid_only=True,
                                theta_q=0.5):
    """The deflation null. For each (valid) empirical pair-window: fit the
    purely-classical generator to that pair's real returns, simulate a panel of
    equal length, push it through the SAME estimator, and record the simulated
    (s_odd, delta, ctx). Returns a DataFrame in the pair_window_stats schema
    (tagged source='classical_null'); compare its distributions to the empirical
    ones (reproduction => apparent contextuality is the post-selection artifact).

    returns_panel : long daily_returns frame (permno, date, ret).
    max_pairs     : optional cap (the deliverable is a distribution, not every
                    pair); pairs are sampled reproducibly with `seed`.
    """
    rng = np.random.default_rng(seed)
    df = pair_window_stats
    if valid_only and "valid" in df.columns:
        df = df[df["valid"]]
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(int(max_pairs), random_state=seed)
    if df.empty:
        log.warning("classical_null_reproduction: no pairs to simulate.")
        return pd.DataFrame()

    ret = returns_panel.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    by_permno = {int(p): g.set_index("date")["ret"] for p, g in ret.groupby("permno")}

    rows = []
    for _, r in df.iterrows():
        pa, pb = int(r["permno_a"]), int(r["permno_b"])
        ws, we = pd.Timestamp(r["win_start"]), pd.Timestamp(r["win_end"])
        if pa not in by_permno or pb not in by_permno:
            continue
        sa = by_permno[pa]; sb = by_permno[pb]
        sa = sa[(sa.index >= ws) & (sa.index <= we)]
        sb = sb[(sb.index >= ws) & (sb.index <= we)]
        joined = pd.concat([sa.rename("a"), sb.rename("b")], axis=1)
        if joined.empty:
            continue
        params = fit_pair_generator(joined["a"].to_numpy(), joined["b"].to_numpy(),
                                    n_min, theta_q=theta_q)
        if params is None:
            continue
        ra, rb = simulate_pair(params, len(joined), rng, theta_q=theta_q)
        res = _pair_stats_from_returns(ra, rb, n_min, theta_q=theta_q)
        row = _pack_row(r["window_id"], ws, we, pa, pb, res["E"], res["a_marg"],
                        res["b_marg"], res["N"], res["s_odd"], res["delta"],
                        res["ctx"], res["valid"])
        row["regime"] = r.get("regime", "calm")
        row["source"] = "classical_null"
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        v = out[out["valid"]]
        log.info(f"classical_null_reproduction: simulated {len(out):,} pairs "
                 f"({len(v):,} valid); naive rate {(v.s_odd > 2).mean():.1%}, "
                 f"CbD rate {(v.ctx > 0).mean():.1%}")
    return out


def _pair_stats_from_returns(retA, retB, n_min, theta_q=0.5):
    """Push one pair's (retA, retB) through the SAME estimator the driver uses:
    drop NaN/zero-sign days, per-stock theta_q-quantile threshold, regime,
    compute_pair_window.
    """
    retA = np.asarray(retA, dtype=float); retB = np.asarray(retB, dtype=float)
    nanA = np.isnan(retA); nanB = np.isnan(retB)
    thA = window_threshold(np.abs(retA[~nanA]), theta_q) if (~nanA).any() else np.nan
    thB = window_threshold(np.abs(retB[~nanB]), theta_q) if (~nanB).any() else np.nan
    sa = np.sign(retA); sb = np.sign(retB)
    x = assign_regime(np.abs(retA), thA); y = assign_regime(np.abs(retB), thB)
    keep = (sa != 0) & (sb != 0) & ~nanA & ~nanB
    return compute_pair_window(sa[keep], sb[keep], x[keep], y[keep], n_min)


# ==========================================================================
# STEP 4 -- EXOGENOUS-CONTEXT CONTRAST  (spec §pipeline Step 4)
# ==========================================================================
def make_vix_regime(vix_df, *, rolling=None):
    """Map each date to a SHARED exogenous regime in {0 (high), 1 (low)} from a
    VIX series (columns {date, vix}). High/low split by the rolling median
    (window=`rolling` days) or, if rolling is None, the full-sample median.
    Returns a dict {Timestamp: 0|1}. This context is external to any pair's
    returns and is the same for both legs (x = y)."""
    cols = {c.lower(): c for c in vix_df.columns}
    s = pd.Series(pd.to_numeric(vix_df[cols["vix"]], errors="coerce").values,
                  index=pd.to_datetime(vix_df[cols["date"]])).sort_index().dropna()
    ref = s.rolling(int(rolling), min_periods=1).median() if rolling else s.median()
    high = s >= ref                                     # high-vol day
    return {d: (0 if h else 1) for d, h in zip(s.index, high)}


def exogenous_context_contrast(pair_window_stats, returns_panel, exo_regime, *,
                               n_min=10, max_pairs=None, seed=0, valid_only=True):
    """Recompute pair statistics with a SHARED, exogenous context (x = y), per
    spec §Step 4. Because both legs share the regime each day, the off-diagonal
    cells (0,1)/(1,0) are empty by construction, so the comparison uses the
    diagonal cells: co-movement on high-vol days (E00) vs low-vol days (E11) and
    the diagonal marginal drift. Attenuation of any contextuality signal here
    (relative to the endogenous magnitude split) is evidence the endogenous
    sorting was the source.

    exo_regime : {date: 0|1} shared regime (see `make_vix_regime`).
    Returns a DataFrame: window_id, permno_a, permno_b, E00, E11, a00, a11,
    b00, b11, N00, N11, dcmov (=E00-E11), valid_diag, regime.
    """
    df = pair_window_stats
    if valid_only and "valid" in df.columns:
        df = df[df["valid"]]
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(int(max_pairs), random_state=seed)
    if df.empty:
        return pd.DataFrame()

    ret = returns_panel.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    by_permno = {int(p): g.set_index("date")["ret"] for p, g in ret.groupby("permno")}
    exo = {pd.Timestamp(k): int(v) for k, v in exo_regime.items()}

    rows = []
    for _, r in df.iterrows():
        pa, pb = int(r["permno_a"]), int(r["permno_b"])
        ws, we = pd.Timestamp(r["win_start"]), pd.Timestamp(r["win_end"])
        if pa not in by_permno or pb not in by_permno:
            continue
        sa = by_permno[pa]; sb = by_permno[pb]
        j = pd.concat([sa.rename("a"), sb.rename("b")], axis=1)
        j = j[(j.index >= ws) & (j.index <= we)].dropna()
        if j.empty:
            continue
        reg = np.array([exo.get(d, -1) for d in j.index])
        a = np.sign(j["a"].to_numpy()); b = np.sign(j["b"].to_numpy())
        keep = (a != 0) & (b != 0) & (reg >= 0)
        a, b, reg = a[keep], b[keep], reg[keep]
        out = {"window_id": r["window_id"], "permno_a": pa, "permno_b": pb,
               "regime": r.get("regime", "calm")}
        diag = {}
        for cell, lab in ((0, "00"), (1, "11")):
            m = reg == cell
            n = int(m.sum())
            out[f"N{lab}"] = n
            out[f"E{lab}"] = float(np.mean(a[m] * b[m])) if n else np.nan
            out[f"a{lab}"] = float(np.mean(a[m])) if n else np.nan
            out[f"b{lab}"] = float(np.mean(b[m])) if n else np.nan
            diag[cell] = n
        out["dcmov"] = (out["E00"] - out["E11"]
                        if not (np.isnan(out["E00"]) or np.isnan(out["E11"])) else np.nan)
        out["valid_diag"] = bool(diag[0] >= n_min and diag[1] >= n_min)
        rows.append(out)
    return pd.DataFrame(rows)


# ==========================================================================
# STEP 5 -- CONTROLS  (spec §pipeline Step 5)
# ==========================================================================
def permutation_placebo(pair_window_stats, returns_panel, *, n_min=10, n_perm=1,
                        seed=0, max_pairs=None, valid_only=True):
    """Permutation placebo (spec §Step 5(i)). Within each pair-window, shuffle the
    regime labels relative to the signs (break the magnitude->sign linkage), then
    recompute. The CTX distribution must collapse below 0 -- if it does not, the
    apparent signal is not coming from the endogenous sorting. Returns a DataFrame
    of placebo stats (schema like pair_window_stats, with a `perm` index)."""
    rng = np.random.default_rng(seed)
    df = pair_window_stats
    if valid_only and "valid" in df.columns:
        df = df[df["valid"]]
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(int(max_pairs), random_state=seed)
    ret = returns_panel.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    by_permno = {int(p): g.set_index("date")["ret"] for p, g in ret.groupby("permno")}

    rows = []
    for _, r in df.iterrows():
        pa, pb = int(r["permno_a"]), int(r["permno_b"])
        ws, we = pd.Timestamp(r["win_start"]), pd.Timestamp(r["win_end"])
        if pa not in by_permno or pb not in by_permno:
            continue
        sa = by_permno[pa]; sb = by_permno[pb]
        j = pd.concat([sa.rename("a"), sb.rename("b")], axis=1)
        j = j[(j.index >= ws) & (j.index <= we)]
        a_all = np.sign(j["a"].to_numpy()); b_all = np.sign(j["b"].to_numpy())
        mA = np.abs(j["a"].to_numpy()); mB = np.abs(j["b"].to_numpy())
        nanA = np.isnan(j["a"].to_numpy()); nanB = np.isnan(j["b"].to_numpy())
        thA = window_threshold(mA[~nanA]) if (~nanA).any() else np.nan
        thB = window_threshold(mB[~nanB]) if (~nanB).any() else np.nan
        x = assign_regime(mA, thA); y = assign_regime(mB, thB)
        keep = (a_all != 0) & (b_all != 0) & ~nanA & ~nanB
        a, b, x, y = a_all[keep], b_all[keep], x[keep], y[keep]
        for p in range(n_perm):
            perm = rng.permutation(len(a))             # shuffle regimes vs signs
            res = compute_pair_window(a, b, x[perm], y[perm], n_min)
            row = _pack_row(r["window_id"], ws, we, pa, pb, res["E"], res["a_marg"],
                            res["b_marg"], res["N"], res["s_odd"], res["delta"],
                            res["ctx"], res["valid"])
            row["perm"] = p
            rows.append(row)
    return pd.DataFrame(rows)


def small_sample_null(n_values, *, n_sims=2000, seed=0):
    """Finite-N null (spec §Step 5(ii)). Push iid fair +-1 signs through the
    estimator at equal per-cell count N to quantify finite-N inflation of s_odd.
    Returns a DataFrame: N, s_odd_mean, s_odd_q95, ctx_mean, frac_sodd_gt2."""
    rng = np.random.default_rng(seed)
    rows = []
    for N in n_values:
        s_list = np.empty(n_sims); c_list = np.empty(n_sims)
        for k in range(n_sims):
            E = np.array([np.mean(rng.choice([-1.0, 1.0], N) * rng.choice([-1.0, 1.0], N))
                          for _ in range(4)])
            s = s_odd(E)
            s_list[k] = s; c_list[k] = s - 0.0 - 2.0    # iid marginals ~ 0 => delta~0
        rows.append({"N": int(N), "s_odd_mean": float(s_list.mean()),
                     "s_odd_q95": float(np.quantile(s_list, 0.95)),
                     "ctx_mean": float(c_list.mean()),
                     "frac_sodd_gt2": float((s_list > 2).mean())})
    return pd.DataFrame(rows)


def max_stat_null(pair_window_stats, returns_panel, *, n_min=10, n_perm=200,
                  alpha=0.05, seed=0, max_pairs_per_window=None):
    """Multiple-comparison control (spec §Step 5(iii)). For each window, build the
    permutation distribution of the MAX ctx across its pairs (shuffling regimes
    vs signs each pair), and return the (1-alpha) quantile as the family-wise
    threshold a single pair's ctx must clear to be 'noteworthy'. Returns a
    DataFrame: window_id, n_pairs, ctx_threshold, max_observed."""
    rng = np.random.default_rng(seed)
    df = pair_window_stats
    if "valid" in df.columns:
        df = df[df["valid"]]
    ret = returns_panel.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    by_permno = {int(p): g.set_index("date")["ret"] for p, g in ret.groupby("permno")}

    rows = []
    for wid, g in df.groupby("window_id"):
        if max_pairs_per_window is not None and len(g) > max_pairs_per_window:
            g = g.sample(int(max_pairs_per_window), random_state=seed)
        prepared = []
        for _, r in g.iterrows():
            pa, pb = int(r["permno_a"]), int(r["permno_b"])
            ws, we = pd.Timestamp(r["win_start"]), pd.Timestamp(r["win_end"])
            if pa not in by_permno or pb not in by_permno:
                continue
            j = pd.concat([by_permno[pa].rename("a"), by_permno[pb].rename("b")], axis=1)
            j = j[(j.index >= ws) & (j.index <= we)]
            a = np.sign(j["a"].to_numpy()); b = np.sign(j["b"].to_numpy())
            mA = np.abs(j["a"].to_numpy()); mB = np.abs(j["b"].to_numpy())
            nA = np.isnan(j["a"].to_numpy()); nB = np.isnan(j["b"].to_numpy())
            thA = window_threshold(mA[~nA]) if (~nA).any() else np.nan
            thB = window_threshold(mB[~nB]) if (~nB).any() else np.nan
            x = assign_regime(mA, thA); y = assign_regime(mB, thB)
            keep = (a != 0) & (b != 0) & ~nA & ~nB
            prepared.append((a[keep], b[keep], x[keep], y[keep]))
        if not prepared:
            continue
        maxes = np.empty(n_perm)
        for p in range(n_perm):
            best = -np.inf
            for (a, b, x, y) in prepared:
                perm = rng.permutation(len(a))
                res = compute_pair_window(a, b, x[perm], y[perm], n_min)
                if res["valid"] and res["ctx"] > best:
                    best = res["ctx"]
            maxes[p] = best if np.isfinite(best) else np.nan
        thr = float(np.nanquantile(maxes, 1 - alpha)) if np.isfinite(maxes).any() else np.nan
        rows.append({"window_id": wid, "n_pairs": len(prepared),
                     "ctx_threshold": thr, "max_observed": float(g["ctx"].max())})
    return pd.DataFrame(rows)


def sector_stratified_rates(pair_window_stats, sector_map):
    """Sector stratification (spec §Step 5(iv)). `sector_map`: {permno: sector}.
    Each pair is 'within:<sector>' if both legs share a sector, else 'cross'.
    Returns naive (s_odd>2) and CbD (ctx>0) rates by (regime, pair_sector)."""
    df = pair_window_stats
    if "valid" in df.columns:
        df = df[df["valid"]].copy()
    else:
        df = df.copy()
    if df.empty:
        return pd.DataFrame()
    sa = df["permno_a"].astype(int).map(sector_map)
    sb = df["permno_b"].astype(int).map(sector_map)
    df["pair_sector"] = np.where(sa.eq(sb) & sa.notna(), "within:" + sa.astype(str), "cross")
    df["regime"] = df["regime"] if "regime" in df.columns else "calm"
    out = df.groupby(["regime", "pair_sector"]).agg(
        n_pairs=("ctx", "size"),
        naive_rate=("s_odd", lambda s: float((s > 2).mean())),
        cbd_rate=("ctx", lambda c: float((c > 0).mean()))).reset_index()
    return out


# ==========================================================================
# MAIN
# ==========================================================================
def _iter_window_panels(elig, ret, window_ids, win_bounds):
    """Yield (wid, win_start, win_end, permnos, ret_wide) per window. Centralizes
    the per-window slice+pivot so the analyzer and the threshold sweep share it."""
    for wid in window_ids:
        ws, we = win_bounds[wid]
        permnos = elig.loc[elig.window_id == wid, "permno"].tolist()
        sub = ret[(ret.permno.isin(permnos)) & (ret.date >= ws) & (ret.date <= we)]
        if sub.empty:
            continue
        ret_wide = sub.pivot_table(index="date", columns="permno", values="ret")
        yield wid, ws, we, permnos, ret_wide


def analyze(data_dir, out_path, n_min, sample=None, crisis_source=None,
            crisis_threshold=None, run_null=False, null_out=None,
            null_max_pairs=2000, seed=0, theta_q=0.5):
    d = load_data(data_dir)
    elig, ret = d["window_eligibility"], d["daily_returns"]
    window_ids = sorted(elig.window_id.unique())
    if sample:
        window_ids = window_ids[:sample]
        log.info(f"--sample: limiting to first {len(window_ids)} window(s)")

    win_bounds = {wid: (g.win_start.iloc[0], g.win_end.iloc[0])
                  for wid, g in elig.groupby("window_id")}
    labels = load_crisis_labels(window_ids, win_bounds, source=crisis_source,
                                threshold=crisis_threshold)
    log.info(f"theta quantile = {theta_q} (regime 0 = |R| >= per-stock q{theta_q} of |R|)")

    all_rows = []
    for wid, ws, we, permnos, ret_wide in _iter_window_panels(elig, ret, window_ids, win_bounds):
        rows = run_window(wid, ws, we, permnos, ret_wide, n_min, theta_q=theta_q)
        for r in rows:
            r["regime"] = labels.get(wid, "calm")
        all_rows.extend(rows)
        log.info(f"  window {wid}: {len(permnos)} names, {len(rows)} pairs")

    df = pd.DataFrame(all_rows)
    try:
        df.to_parquet(out_path, index=False)
    except Exception:                                  # noqa: BLE001
        out_path = out_path.replace(".parquet", ".csv"); df.to_csv(out_path, index=False)
    log.info(f"wrote {len(df):,} pair-window rows -> {out_path}")
    if len(df):
        v = df[df.valid]
        log.info(f"valid pairs: {len(v):,}; naive violation rate (s_odd>2): "
                 f"{(v.s_odd > 2).mean():.1%}; CbD rate (ctx>0): {(v.ctx > 0).mean():.1%}")

    if run_null and len(df):
        log.info("running classical_null_reproduction (the deflation null)...")
        null_df = classical_null_reproduction(df, ret, n_min=n_min,
                                              max_pairs=null_max_pairs, seed=seed)
        if null_out is None:
            null_out = out_path.replace("pair_window_stats", "classical_null_stats")
        try:
            null_df.to_parquet(null_out, index=False)
        except Exception:                              # noqa: BLE001
            null_out = null_out.replace(".parquet", ".csv"); null_df.to_csv(null_out, index=False)
        log.info(f"wrote {len(null_df):,} classical-null rows -> {null_out}")
    return df


# Fixed bin edges for the streamed per-pair distributions (shared with plots.py so
# the full-span run never materializes the ~5e7-row pair-window frame in memory).
HIST_EDGES = {
    "s_odd": np.linspace(0.0, 4.0, 81),
    "delta": np.linspace(0.0, 2.0, 41),
    "ctx": np.linspace(-4.0, 2.0, 121),
}


def analyze_streaming(data_dir, out_dir, n_min, *, sample=None, crisis_source=None,
                      crisis_threshold=None, theta_q=0.5, run_null=True,
                      null_pairs_per_window=400, scatter_cap=20000, seed=0):
    """Memory-safe full-span driver. Writes one pair_window_stats SHARD per window
    (partitioned by window_id) and accumulates only compact running aggregates --
    never the whole ~5e7-row frame. Aggregates (all gitignored parquet in out_dir):

      headline_rates.parquet  : per-regime n_total/n_valid/n_naive(s_odd>2)/n_cbd(ctx>0)
      stat_hist.parquet       : per-(var,regime) histogram counts on HIST_EDGES
      cell_summary.parquet    : per-regime mean of N00..N11 (the N_min health check)
      scatter_subsample.parquet: bounded reservoir of (s_odd,delta,ctx,regime)
      classical_null_hist.parquet : streamed null ctx histogram (deflation overlay)

    Downstream (plots.py, networks.py) stream these / the shards window-by-window.
    """
    from collections import defaultdict
    d = load_data(data_dir)
    elig, ret = d["window_eligibility"], d["daily_returns"]
    window_ids = sorted(elig.window_id.unique())
    if sample:
        window_ids = window_ids[:sample]
    win_bounds = {wid: (g.win_start.iloc[0], g.win_end.iloc[0])
                  for wid, g in elig.groupby("window_id")}
    labels = load_crisis_labels(window_ids, win_bounds, source=crisis_source,
                                threshold=crisis_threshold)
    shard_dir = out_dir
    os.makedirs(shard_dir, exist_ok=True)
    agg_dir = data_dir                                     # aggregates live beside other outputs

    tally = defaultdict(lambda: np.zeros(4))               # regime -> [total,valid,naive,cbd]
    hist = defaultdict(lambda: {v: np.zeros(len(e) - 1) for v, e in HIST_EDGES.items()})
    cell = defaultdict(lambda: [0, np.zeros(4)])           # regime -> [n, sum N00..N11]
    scatter = []
    per_win_quota = max(1, scatter_cap // max(1, len(window_ids)))
    null_hist = np.zeros(len(HIST_EDGES["ctx"]) - 1)
    null_tally = np.zeros(3)                               # valid, naive, cbd

    n_rows = 0
    for wid, ws, we, permnos, ret_wide in _iter_window_panels(elig, ret, window_ids, win_bounds):
        reg = labels.get(wid, "calm")
        rows = run_window(wid, ws, we, permnos, ret_wide, n_min, theta_q=theta_q)
        if not rows:
            continue
        wdf = pd.DataFrame(rows)
        wdf["regime"] = reg
        wdf.to_parquet(os.path.join(shard_dir, f"shard_{int(wid):05d}.parquet"), index=False)
        n_rows += len(wdf)
        v = wdf[wdf["valid"]]
        t = tally[reg]
        t[0] += len(wdf); t[1] += len(v)
        t[2] += int((v["s_odd"] > 2).sum()); t[3] += int((v["ctx"] > 0).sum())
        for var, e in HIST_EDGES.items():
            hist[reg][var] += np.histogram(v[var].dropna().to_numpy(), bins=e)[0]
        cell[reg][0] += len(v)
        cell[reg][1] += v[["N00", "N01", "N10", "N11"]].to_numpy().sum(axis=0)
        # bounded scatter subsample: an even per-window quota (fast, vectorized)
        if len(v):
            take = v if len(v) <= per_win_quota else v.sample(per_win_quota, random_state=seed)
            scatter.append(take[["s_odd", "delta", "ctx", "regime"]])
        # streamed deflation null on a per-window sample
        if run_null and len(v):
            samp = v if len(v) <= null_pairs_per_window else v.sample(null_pairs_per_window,
                                                                      random_state=seed)
            nd = classical_null_reproduction(samp, ret, n_min=n_min, max_pairs=None,
                                             seed=seed, theta_q=theta_q)
            if len(nd):
                nv = nd[nd["valid"]]
                null_tally += [len(nv), int((nv.s_odd > 2).sum()), int((nv.ctx > 0).sum())]
                null_hist += np.histogram(nv["ctx"].dropna().to_numpy(),
                                          bins=HIST_EDGES["ctx"])[0]
        log.info(f"  window {wid} ({reg}): {len(wdf):,} pairs, {len(v):,} valid "
                 f"[running total {n_rows:,} rows]")

    # ---- write compact aggregates -------------------------------------------
    hr = pd.DataFrame([{"regime": r, "n_total": int(t[0]), "n_valid": int(t[1]),
                        "n_naive": int(t[2]), "n_cbd": int(t[3]),
                        "naive_rate": (t[2] / t[1]) if t[1] else np.nan,
                        "cbd_rate": (t[3] / t[1]) if t[1] else np.nan}
                       for r, t in sorted(tally.items())])
    _stream_save(hr, os.path.join(agg_dir, "headline_rates.parquet"))
    hrows = [{"var": var, "regime": r, "bin_left": HIST_EDGES[var][i],
              "bin_right": HIST_EDGES[var][i + 1], "count": int(c)}
             for r, hv in hist.items() for var, arr in hv.items()
             for i, c in enumerate(arr)]
    _stream_save(pd.DataFrame(hrows), os.path.join(agg_dir, "stat_hist.parquet"))
    crows = [{"regime": r, "n_valid": int(n),
              "N00": s[0] / n if n else np.nan, "N01": s[1] / n if n else np.nan,
              "N10": s[2] / n if n else np.nan, "N11": s[3] / n if n else np.nan}
             for r, (n, s) in cell.items()]
    _stream_save(pd.DataFrame(crows), os.path.join(agg_dir, "cell_summary.parquet"))
    scatter_df = pd.concat(scatter, ignore_index=True) if scatter else pd.DataFrame(
        columns=["s_odd", "delta", "ctx", "regime"])
    _stream_save(scatter_df, os.path.join(agg_dir, "scatter_subsample.parquet"))
    nh = pd.DataFrame([{"bin_left": HIST_EDGES["ctx"][i], "bin_right": HIST_EDGES["ctx"][i + 1],
                        "count": int(c)} for i, c in enumerate(null_hist)])
    _stream_save(nh, os.path.join(agg_dir, "classical_null_hist.parquet"))

    log.info(f"STREAMING DONE: {n_rows:,} pair-window rows across {len(window_ids)} "
             f"windows -> shards in {shard_dir}")
    for _, r in hr.iterrows():
        log.info(f"  {r['regime']:>6}: naive(s_odd>2)={r['naive_rate']:.2%}, "
                 f"CbD(ctx>0)={r['cbd_rate']:.2%}, N_valid={r['n_valid']:,}")
    if null_tally[0]:
        log.info(f"  classical-null (streamed): naive={null_tally[1]/null_tally[0]:.2%}, "
                 f"CbD={null_tally[2]/null_tally[0]:.2%}, N={int(null_tally[0]):,}")
    return hr


def _stream_save(df, path):
    try:
        df.to_parquet(path, index=False)
    except Exception:                                      # noqa: BLE001
        path = path.replace(".parquet", ".csv"); df.to_csv(path, index=False)
    log.info(f"  wrote {os.path.basename(path)} ({len(df):,} rows)")


# ==========================================================================
# STEP 2 -- PERCENTILE-THRESHOLD ROBUSTNESS SWEEP
# ==========================================================================
def sweep_thresholds(data_dir, quantiles=(0.25, 0.40, 0.42, 0.45, 0.48, 0.50, 0.75, 0.90, 0.95),
                     n_min=10, crisis_source=None, crisis_threshold=None, sample=None,
                     relaxed_n_min=3, relaxed_quantiles=(0.90, 0.95),
                     null_relative=True, null_cap=3000, seed=0):
    """Sweep the magnitude-context threshold theta over `quantiles` and report the
    deflation (naive s_odd>2 vs CbD ctx>0) by regime at each theta. Tallies are
    accumulated per-window (never materializing all pair-windows), so this scales.

    TWO ROLES of the sweep points (do not conflate):
      * sub-median {0.25, 0.40, 0.50}: the both-large (E00) cell stays populated,
        so these test the ROBUSTNESS OF THE DEFLATION (the headline claim that the
        CbD-corrected rate ~ 0 should hold across them).
      * {0.75, 0.90, 0.95}: the large-move regime gets rare, the four-cell CHSH
        structure loses support and the strict valid denominator collapses -- these
        DOCUMENT THE WELL-POSEDNESS BOUNDARY, not a failure of deflation.
    theta=0.50 remains the canonical split for the E00 fragility network elsewhere
    (networks.py); this sweep is a separate robustness exhibit.

    Strict headline holds N_min at `n_min` (= the spec's 10). A SMALL-SAMPLE
    DIAGNOSTIC at the high (relaxed_) quantiles relaxes N_min to `relaxed_n_min`;
    when `null_relative`, the classical null is ALSO run under the SAME
    relaxed-N_min/high-theta condition so the diagnostic is null-relative: if the
    empirical relaxed ctx>0 ~ the null ctx>0, the residual rate is finite-sample
    noise, not contextuality. If empirical materially exceeds null, it is flagged.

    Returns one row per (theta_q, regime) with strict + relaxed (+ null) rates.
    """
    from collections import defaultdict
    d = load_data(data_dir)
    elig, ret = d["window_eligibility"], d["daily_returns"]
    window_ids = sorted(elig.window_id.unique())
    if sample:
        window_ids = window_ids[:sample]
    win_bounds = {wid: (g.win_start.iloc[0], g.win_end.iloc[0])
                  for wid, g in elig.groupby("window_id")}
    labels = load_crisis_labels(window_ids, win_bounds, source=crisis_source,
                                threshold=crisis_threshold)
    base_nmin = min(n_min, relaxed_n_min)
    relaxed_set = {float(q) for q in relaxed_quantiles}
    # tally[(q, regime)] = [strict_valid, strict_naive, strict_cbd,
    #                       relaxed_valid, relaxed_naive, relaxed_cbd]
    tally = defaultdict(lambda: np.zeros(6))
    # reservoir of relaxed-valid pairs at the relaxed quantiles, for the null pass
    null_samples = defaultdict(list)                      # (q, regime) -> list[rowdict]
    rng = np.random.default_rng(seed)
    log.info(f"threshold sweep over quantiles {tuple(quantiles)} "
             f"(strict N_min={n_min}; relaxed diagnostic N_min={relaxed_n_min} "
             f"at q in {tuple(relaxed_quantiles)}; null_relative={null_relative})")
    for wid, ws, we, permnos, ret_wide in _iter_window_panels(elig, ret, window_ids, win_bounds):
        reg = labels.get(wid, "calm")
        for q in quantiles:
            rows = run_window(wid, ws, we, permnos, ret_wide, base_nmin, theta_q=q)
            t = tally[(float(q), reg)]
            is_relaxed_q = float(q) in relaxed_set
            for r in rows:
                nmin = min(r["N00"], r["N01"], r["N10"], r["N11"])
                if nmin < relaxed_n_min or np.isnan(r["s_odd"]):
                    continue
                t[3] += 1; t[4] += (r["s_odd"] > 2); t[5] += (r["ctx"] > 0)
                if nmin >= n_min:
                    t[0] += 1; t[1] += (r["s_odd"] > 2); t[2] += (r["ctx"] > 0)
                if null_relative and is_relaxed_q:        # reservoir-sample for null
                    bucket = null_samples[(float(q), reg)]
                    rec = {"window_id": wid, "win_start": ws, "win_end": we,
                           "permno_a": r["permno_a"], "permno_b": r["permno_b"],
                           "regime": reg, "valid": True,
                           "N00": r["N00"], "N01": r["N01"],
                           "N10": r["N10"], "N11": r["N11"]}
                    if len(bucket) < null_cap:
                        bucket.append(rec)
                    else:
                        j = rng.integers(0, t[3])
                        if j < null_cap:
                            bucket[int(j)] = rec
        log.info(f"  swept window {wid} ({reg})")

    null_rate = _sweep_null_rates(null_samples, ret, relaxed_n_min, seed) \
        if null_relative else {}
    # cell-size-MATCHED finite-N ctx>0 floor: simulate iid signs at each empirical
    # relaxed-valid pair's actual (N00,N01,N10,N11) geometry (the proper noise floor;
    # the equal-cell floor understates it because only the binding cell is small).
    matched_floor = {}
    if null_relative:
        eqfloor = _finite_n_ctx_floor(relaxed_n_min, seed=seed)
        log.info(f"finite-N ctx>0 floor (equal cells at N={relaxed_n_min}): {eqfloor:.2%}")
        for (q, reg), recs in null_samples.items():
            tuples = [(int(r["N00"]), int(r["N01"]), int(r["N10"]), int(r["N11"]))
                      for r in recs]
            matched_floor[(q, reg)] = _matched_finite_n_floor(tuples, seed=seed)
            log.info(f"  matched finite-N floor @ theta_q={q:.2f} {reg}: "
                     f"{matched_floor[(q, reg)]:.2%} (from {len(tuples):,} cell geometries)")

    out = []
    for (q, reg), t in sorted(tally.items()):
        sv, rv = t[0], t[3]
        nr = null_rate.get((q, reg), {})
        is_relaxed_q = float(q) in relaxed_set
        floor_qr = matched_floor.get((q, reg), np.nan) if is_relaxed_q else np.nan
        out.append({
            "theta_q": q, "regime": reg, "n_min": n_min, "relaxed_n_min": relaxed_n_min,
            "n_valid": int(sv),
            "naive_rate": (t[1] / sv) if sv else np.nan,
            "cbd_rate": (t[2] / sv) if sv else np.nan,
            "n_valid_relaxed": int(rv),
            "naive_rate_relaxed": (t[4] / rv) if rv else np.nan,
            "cbd_rate_relaxed": (t[5] / rv) if rv else np.nan,
            "cbd_rate_null_relaxed": nr.get("cbd_rate", np.nan),
            "n_valid_null_relaxed": nr.get("n_valid", 0),
            "cbd_rate_smallN_floor": floor_qr,
            "is_relaxed_diag_q": is_relaxed_q,
        })
    df = pd.DataFrame(out)
    for _, r in df.iterrows():
        msg = (f"  theta_q={r.theta_q:.2f} {r.regime:>6}: "
               f"naive(s_odd>2)={r.naive_rate:.2%}, CbD(ctx>0)={r.cbd_rate:.2%}, "
               f"N_valid={r.n_valid:,}")
        if r.is_relaxed_diag_q:
            msg += (f"  [relaxed N_min={relaxed_n_min}: CbD={r.cbd_rate_relaxed:.2%} "
                    f"(N={r.n_valid_relaxed:,}); classical-null={r.cbd_rate_null_relaxed:.2%} "
                    f"(N={r.n_valid_null_relaxed:,}); finite-N floor={r.cbd_rate_smallN_floor:.2%}]")
        log.info(msg)
        if r.cbd_rate > 0.01:
            log.warning(f"  !! theta_q={r.theta_q:.2f} {r.regime}: strict CbD rate "
                        f"{r.cbd_rate:.2%} > 1% -- INSPECT (a result, not a bug).")
        # only a genuine concern if empirical clears BOTH the (possibly-collapsed)
        # classical null AND the finite-N noise floor
        if (r.is_relaxed_diag_q and np.isfinite(r.cbd_rate_smallN_floor)
                and r.cbd_rate_relaxed > r.cbd_rate_smallN_floor + 0.01
                and (not np.isfinite(r.cbd_rate_null_relaxed)
                     or r.n_valid_null_relaxed < 100
                     or r.cbd_rate_relaxed > r.cbd_rate_null_relaxed + 0.01)):
            log.warning(f"  !! theta_q={r.theta_q:.2f} {r.regime}: empirical relaxed CbD "
                        f"{r.cbd_rate_relaxed:.2%} exceeds the finite-N floor "
                        f"{r.cbd_rate_smallN_floor:.2%} -- NOT just small-N noise; INSPECT.")
    return df


def s_odd_split_half_reliability(elig, ret, *, n_min=10, theta_q=0.5, sample=None,
                                 seed=0, half_min_cell=2):
    """Per-window s_odd REGIME-PRESERVING split-half reliability.

    The pitfall of a naive day-halving split is that recomputing theta on each half
    changes which days are large/small, so the two halves no longer measure the same
    four-cell construct (the parallel-test assumption fails) and the reliability is
    biased low. Here we instead FIX each stock's per-stock threshold at its FULL-
    window theta_q-quantile (`thresholds=` injected into run_window), assign regimes
    once, and split the days odd/even. Both halves then share identical per-day
    regime labels, so they are genuine parallel sub-tests; the half-length Pearson r
    is corrected to full length by Spearman-Brown (r_sb = 2r/(1+r)).

    Reliability is computed over pairs that are valid at the full N_min in the full
    window; a half contributes a pair's s_odd as long as each of its four cells has
    >= `half_min_cell` days. Computed here (not in networks.py) because it needs the
    per-DAY series only the driver has. Returns one row per window.
    """
    window_ids = sorted(elig.window_id.unique())
    if sample:
        window_ids = window_ids[:sample]
    win_bounds = {wid: (g.win_start.iloc[0], g.win_end.iloc[0])
                  for wid, g in elig.groupby("window_id")}
    rows = []
    for wid, ws, we, permnos, ret_wide in _iter_window_panels(elig, ret, window_ids, win_bounds):
        rw = ret_wide.sort_index()
        cols = [p for p in permnos if p in rw.columns]
        if rw.shape[0] < 8 or len(cols) < 3:
            continue
        absR = rw[cols].abs()
        # full-window per-stock thresholds (identical regime definition to analysis)
        thr = {p: window_threshold(absR[p].dropna().values, theta_q)
               for p in cols if absR[p].notna().any()}
        # pairs valid at the full N_min in the full window (the reliability universe)
        full_valid = {(r["permno_a"], r["permno_b"])
                      for r in run_window(wid, ws, we, cols, rw, n_min,
                                          theta_q=theta_q, thresholds=thr)
                      if r["valid"]}
        if len(full_valid) < 3:
            continue
        ha, hb = rw.iloc[0::2], rw.iloc[1::2]              # parallel odd/even halves
        sa = {(r["permno_a"], r["permno_b"]): r["s_odd"]
              for r in run_window(wid, ws, we, cols, ha, half_min_cell,
                                  theta_q=theta_q, thresholds=thr)
              if r["valid"] and not np.isnan(r["s_odd"])}
        sb = {(r["permno_a"], r["permno_b"]): r["s_odd"]
              for r in run_window(wid, ws, we, cols, hb, half_min_cell,
                                  theta_q=theta_q, thresholds=thr)
              if r["valid"] and not np.isnan(r["s_odd"])}
        common = [k for k in full_valid if k in sa and k in sb]
        if len(common) < 3:
            continue
        va = np.array([sa[k] for k in common]); vb = np.array([sb[k] for k in common])
        if va.std() == 0 or vb.std() == 0:
            continue
        r = float(np.corrcoef(va, vb)[0, 1])
        r_sb = (2 * r / (1 + r)) if (1 + r) != 0 else np.nan
        rows.append({"window_id": wid, "win_start": ws, "n_pairs": len(common),
                     "reliability_halfsplit": r, "reliability_sb": r_sb})
        log.info(f"  reliability window {wid}: r={r:.3f} (SB={r_sb:.3f}), "
                 f"N_common={len(common):,}")
    df = pd.DataFrame(rows)
    if len(df):
        log.info(f"s_odd reliability (regime-preserving): mean half-split "
                 f"r={df.reliability_halfsplit.mean():.3f}, mean Spearman-Brown "
                 f"r={df.reliability_sb.mean():.3f}")
    return df


def _resolve_stats_path(stats_path):
    """Return a readable parquet path/dir for pair_window_stats (handles bare name)."""
    for cand in (stats_path, stats_path + ".parquet"):
        if os.path.isdir(cand) or os.path.exists(cand):
            return cand
    return stats_path


def emit_null_gate_stats(stats_path, returns, *, n_windows=6, n_nodes=60,
                         theta_q=0.5, n_min=10, seed=0):
    """Emit a per-window-DENSE classical-null pair_window_stats frame for the
    networks MR-QAP null baseline. For the first `n_windows` windows it takes the
    first `n_nodes` names (sorted) appearing in valid real pairs and simulates the
    classical null for ALL real valid pairs among them (same window x node support
    the gate uses, so real-vs-null R^2 is apples-to-apples). Schema is identical to
    pair_window_stats (tagged source='classical_null'); the null's math is unchanged.
    Output is meant to be gitignored.
    """
    path = _resolve_stats_path(stats_path)
    ids = sorted(pd.read_parquet(path, columns=["window_id"])["window_id"].unique())[:n_windows]
    cols = ["window_id", "win_start", "win_end", "permno_a", "permno_b", "valid", "regime"]
    df = pd.read_parquet(path, columns=cols, filters=[("window_id", "in", ids)])
    out = []
    for wid, g in df.groupby("window_id"):
        gv = g[g["valid"]] if "valid" in g.columns else g
        nodes = sorted(pd.unique(gv[["permno_a", "permno_b"]].to_numpy().ravel()).tolist())
        keep = set(nodes[:n_nodes])
        sub = gv[gv["permno_a"].isin(keep) & gv["permno_b"].isin(keep)]
        if sub.empty:
            continue
        nd = classical_null_reproduction(sub, returns, n_min=n_min, max_pairs=None,
                                         seed=seed, theta_q=theta_q)
        if len(nd):
            out.append(nd)
        log.info(f"  null-gate window {wid}: {len(keep)} nodes, "
                 f"{len(sub):,} pairs -> {len(nd) if len(nd) else 0:,} null rows")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _finite_n_ctx_floor(n_min, *, n_sims=8000, seed=0):
    """Finite-N ctx>0 false-positive floor: push iid fair +-1 signs through the FULL
    estimator (correlations AND marginals, so delta is included) at equal per-cell
    count `n_min`, and return the fraction with ctx>0. This is the proper noise
    baseline for the relaxed-N_min diagnostic: equal cells at the binding N_min are
    the worst case, so empirical <= this floor => the rate is finite-sample noise,
    not contextuality. (Complements the per-pair classical null, which cannot
    populate the four cells at high theta and collapses there.)"""
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(int(n_sims)):
        E = np.empty(4); am = np.empty(4); bm = np.empty(4)
        for k in range(4):
            a = rng.choice([-1.0, 1.0], n_min); b = rng.choice([-1.0, 1.0], n_min)
            E[k] = np.mean(a * b); am[k] = np.mean(a); bm[k] = np.mean(b)
        if ctx(E, am, bm) > 0:
            cnt += 1
    return cnt / n_sims


def _matched_finite_n_floor(cellsizes, *, seed=0, max_tuples=5000):
    """Cell-size-MATCHED finite-N ctx>0 floor. For each empirical relaxed-valid
    pair's actual (N00,N01,N10,N11), draw iid fair +-1 signs at exactly those cell
    sizes and push through the full estimator; return the fraction with ctx>0. This
    is the correctly-matched noise baseline (it reproduces the realistic geometry --
    typically one tiny binding cell and three large cells -- which the equal-cell
    floor does not). empirical ~ this floor => finite-sample noise."""
    if not cellsizes:
        return np.nan
    rng = np.random.default_rng(seed)
    if len(cellsizes) > max_tuples:
        idx = rng.choice(len(cellsizes), max_tuples, replace=False)
        cellsizes = [cellsizes[i] for i in idx]
    cnt = 0
    for sizes in cellsizes:
        E = np.empty(4); am = np.empty(4); bm = np.empty(4)
        for k, N in enumerate(sizes):
            N = max(int(N), 1)
            a = rng.choice([-1.0, 1.0], N); b = rng.choice([-1.0, 1.0], N)
            E[k] = np.mean(a * b); am[k] = np.mean(a); bm[k] = np.mean(b)
        if ctx(E, am, bm) > 0:
            cnt += 1
    return cnt / len(cellsizes)


def _sweep_null_rates(null_samples, ret, relaxed_n_min, seed):
    """Run the classical null on the reservoir-sampled relaxed-valid pairs at each
    relaxed (q, regime), under the SAME relaxed N_min and that theta_q. Returns
    {(q, regime): {'cbd_rate':.., 'n_valid':..}} so the sweep diagnostic is
    null-relative (empirical ~ null => finite-sample noise)."""
    out = {}
    for (q, reg), recs in null_samples.items():
        if not recs:
            continue
        sub = pd.DataFrame(recs)
        nd = classical_null_reproduction(sub, ret, n_min=relaxed_n_min, max_pairs=None,
                                         seed=seed, theta_q=float(q))
        if len(nd):
            v = nd[nd["valid"]]
            out[(q, reg)] = {"cbd_rate": float((v.ctx > 0).mean()) if len(v) else np.nan,
                             "n_valid": int(len(v))}
            log.info(f"  null @ theta_q={q:.2f} {reg}: CbD(ctx>0)="
                     f"{out[(q, reg)]['cbd_rate']:.2%} (N={out[(q, reg)]['n_valid']:,})")
    return out


# ==========================================================================
# TESTS  (run: python cbd_analysis.py --test)
# ==========================================================================
def _test_handchecked():
    """Spec's worked deflation example: s_odd=2.3, Delta=0.50, CTX=-0.20."""
    E = [0.6, 0.5, 0.5, -0.7]
    a_marg = [-0.20, -0.05, 0.05, 0.10]
    b_marg = [-0.15, 0.00, 0.05, 0.10]
    assert np.isclose(s_odd(E), 2.3), s_odd(E)
    assert np.isclose(delta(a_marg, b_marg), 0.50), delta(a_marg, b_marg)
    assert np.isclose(ctx(E, a_marg, b_marg), -0.20), ctx(E, a_marg, b_marg)


def _test_postselection():
    """Spec's classical post-selection construction: marginals 0 (Delta=0),
    correlations (+c,+c,+c,-c) -> s_odd = 4c, CTX = 4c-2."""
    zeros = [0.0, 0.0, 0.0, 0.0]
    for c in (0.25, 0.5, 0.9, 1.0):
        E = [c, c, c, -c]
        assert np.isclose(s_odd(E), 4 * c), (c, s_odd(E))
        assert np.isclose(delta(zeros, zeros), 0.0)
        assert np.isclose(ctx(E, zeros, zeros), 4 * c - 2), (c, ctx(E, zeros, zeros))


def _test_anchors():
    """KDL canonical checks: PR-box -> CTX=2; deterministic -> CTX=0 (boundary)."""
    assert np.isclose(ctx([1, 1, 1, -1], [0, 0, 0, 0], [0, 0, 0, 0]), 2.0)
    assert np.isclose(ctx([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]), 0.0)


def _test_pipeline_smoke():
    """Tiny synthetic panel exercises run_window end to end (plumbing only)."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2008-01-01", periods=80)
    permnos = [101, 102, 103]
    recs = []
    for p in permnos:
        for dte in dates:
            recs.append((p, dte, float(rng.normal(0, 0.02))))
    ret = pd.DataFrame(recs, columns=["permno", "date", "ret"])
    ret_wide = ret.pivot_table(index="date", columns="permno", values="ret")
    rows = run_window(0, dates[0], dates[-1], permnos, ret_wide, n_min=5)
    assert len(rows) == 3                      # C(3,2) pairs
    assert all(set(("s_odd", "delta", "ctx", "valid")) <= set(r) for r in rows)


def _test_theta_quantile():
    """window_threshold honors q; higher q shrinks the large-move regime (more
    days classified small, fewer large), and analyze/run_window thread it through."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert np.isclose(window_threshold(x, 0.5), np.median(x))
    assert np.isclose(window_threshold(x, 0.9), np.quantile(x, 0.9))
    # higher theta_q -> stricter 'large' cutoff -> fewer regime-0 (large) days
    n_large_med = int((assign_regime(x, window_threshold(x, 0.5)) == 0).sum())
    n_large_hi = int((assign_regime(x, window_threshold(x, 0.9)) == 0).sum())
    assert n_large_hi < n_large_med


def _test_threshold_sweep():
    """sweep_thresholds returns the expected schema and strict<=relaxed valid counts."""
    import tempfile
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2008-01-02", periods=200)
    permnos = [11, 22, 33, 44, 55, 66]
    recs = [(p, d, float(rng.normal(0, 0.02))) for p in permnos for d in dates]
    ret = pd.DataFrame(recs, columns=["permno", "date", "ret"])
    memb = pd.DataFrame({"permno": permnos, "mbrstartdt": pd.Timestamp("2000-01-01"),
                         "mbrenddt": pd.Timestamp("2025-12-31")})
    cal = pd.DataFrame({"date": dates})
    elig = pd.DataFrame([(0, dates[0], dates[-1], p) for p in permnos],
                        columns=["window_id", "win_start", "win_end", "permno"])
    with tempfile.TemporaryDirectory() as tmp:
        for nm, df in (("membership", memb), ("daily_returns", ret),
                       ("trading_calendar", cal), ("window_eligibility", elig)):
            df.to_parquet(os.path.join(tmp, nm + ".parquet"), index=False)
        sw = sweep_thresholds(tmp, quantiles=(0.5, 0.75), n_min=5, relaxed_n_min=2,
                              relaxed_quantiles=(0.75,), null_relative=True)
    assert {"theta_q", "regime", "cbd_rate", "n_valid", "cbd_rate_relaxed",
            "n_valid_relaxed", "cbd_rate_null_relaxed", "n_valid_null_relaxed",
            "is_relaxed_diag_q"} <= set(sw.columns)
    assert (sw["n_valid_relaxed"] >= sw["n_valid"]).all()
    assert sw["is_relaxed_diag_q"].any()
    # the null-relative diagnostic ran at the relaxed quantile (some null pairs valid)
    diag = sw[sw["is_relaxed_diag_q"]]
    assert (diag["n_valid_null_relaxed"] > 0).any()


def _test_null_theta_threading():
    """The classical null honors theta_q: a higher quantile shrinks the large-move
    regime so the four-cell structure has fewer large-cell days (smaller N00)."""
    rng = np.random.default_rng(11)
    params = {"scaleA": (3.0, 1.0), "scaleB": (3.0, 1.0),
              "cells": {(0, 0): (0.5, 0.5, 0.5), (0, 1): (0.5, 0.5, 0.5),
                        (1, 0): (0.5, 0.5, 0.5), (1, 1): (0.5, 0.5, -0.5)}}
    ra, rb = simulate_pair(params, 6000, rng, theta_q=0.9)
    lo = _pair_stats_from_returns(ra, rb, n_min=1, theta_q=0.5)
    hi = _pair_stats_from_returns(ra, rb, n_min=1, theta_q=0.9)
    assert hi["N"][0] < lo["N"][0], (hi["N"][0], lo["N"][0])   # N00 shrinks with theta


def _test_reliability_and_null_gate():
    """s_odd_split_half_reliability returns r in [-1,1] per window, and
    emit_null_gate_stats produces a same-schema null frame for the gate."""
    import tempfile
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2008-01-02", periods=120)
    permnos = list(range(8))
    recs = [(p, d, float(rng.normal(0, 0.02))) for p in permnos for d in dates]
    ret = pd.DataFrame(recs, columns=["permno", "date", "ret"])
    elig = pd.DataFrame([(0, dates[0], dates[-1], p) for p in permnos],
                        columns=["window_id", "win_start", "win_end", "permno"])
    rel = s_odd_split_half_reliability(elig, ret, n_min=3, theta_q=0.5)
    assert {"window_id", "reliability_halfsplit", "reliability_sb"} <= set(rel.columns)
    if len(rel):
        assert rel["reliability_halfsplit"].between(-1, 1).all()
    # build a tiny pair_window_stats and emit the null-gate frame
    rw = ret.pivot_table(index="date", columns="permno", values="ret")
    stats = pd.DataFrame(run_window(0, dates[0], dates[-1], permnos, rw, n_min=3))
    stats["regime"] = "calm"
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "pair_window_stats.parquet")
        stats.to_parquet(sp, index=False)
        nd = emit_null_gate_stats(sp, ret, n_windows=1, n_nodes=6, n_min=3)
    assert {"window_id", "permno_a", "permno_b", "s_odd", "ctx", "valid"} <= set(nd.columns)


def _close_nan(x, y, atol=1e-9):
    """True if both NaN, or finite and close."""
    if (x is None or (isinstance(x, float) and np.isnan(x))) and \
       (y is None or (isinstance(y, float) and np.isnan(y))):
        return True
    return bool(np.isclose(x, y, atol=atol))


def _test_vectorized_equivalence():
    """run_window (vectorized) must match _run_window_loop bit-for-bit, including
    NaN / exact-zero days and invalid pairs."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2008-01-01", periods=70)
    permnos = [201, 202, 203, 204, 205]
    R = rng.normal(0, 0.02, size=(len(dates), len(permnos)))
    R[5, 1] = np.nan; R[10, 2] = np.nan; R[:3, 4] = np.nan      # missing days
    R[12, 0] = 0.0; R[20, 3] = 0.0; R[33, 2] = 0.0             # exact-zero (sgn 0)
    ret_wide = pd.DataFrame(R, index=dates, columns=permnos)
    int_keys = ("permno_a", "permno_b", "N00", "N01", "N10", "N11", "valid")
    flt_keys = ("E00", "E01", "E10", "E11", "a00", "a01", "a10", "a11",
                "b00", "b01", "b10", "b11", "s_odd", "delta", "ctx")
    for tq in (0.5, 0.75):                                      # default + a swept q
        loop = _run_window_loop(0, dates[0], dates[-1], permnos, ret_wide, 5, theta_q=tq)
        vec = run_window(0, dates[0], dates[-1], permnos, ret_wide, 5, theta_q=tq)
        assert len(loop) == len(vec) == 10                      # C(5,2)
        for L, V in zip(loop, vec):
            for k in int_keys:
                assert L[k] == V[k], (tq, k, L[k], V[k])
            for k in flt_keys:
                assert _close_nan(L[k], V[k]), (tq, k, L[k], V[k])


def _test_crisis_labels():
    """load_crisis_labels: VIX file, NBER built-in/file, and the None default."""
    import tempfile
    wb = {0: (pd.Timestamp("2005-01-03"), pd.Timestamp("2005-03-31")),   # calm
          1: (pd.Timestamp("2008-09-01"), pd.Timestamp("2008-11-28")),   # crisis
          2: (pd.Timestamp("2013-01-02"), pd.Timestamp("2013-03-29"))}   # calm
    wids = list(wb)
    # default -> all calm
    assert load_crisis_labels(wids, wb, source=None) == {0: "calm", 1: "calm", 2: "calm"}
    # NBER built-in: only the 2008 window overlaps a recession span
    nber = load_crisis_labels(wids, wb, source="nber")
    assert nber == {0: "calm", 1: "crisis", 2: "calm"}, nber
    with tempfile.TemporaryDirectory() as tmp:
        # VIX file: high VIX over the 2008 window, low elsewhere
        days = pd.bdate_range("2004-01-01", "2014-01-01")
        vix = pd.Series(15.0, index=days)
        vix[(vix.index >= wb[1][0]) & (vix.index <= wb[1][1])] = 45.0
        vpath = os.path.join(tmp, "vix.csv")
        pd.DataFrame({"date": days, "vix": vix.values}).to_csv(vpath, index=False)
        vlab = load_crisis_labels(wids, wb, source=vpath)
        assert vlab[1] == "crisis" and vlab[0] == "calm" and vlab[2] == "calm", vlab
        # recession-span file
        spath = os.path.join(tmp, "rec.csv")
        pd.DataFrame({"start": ["2008-01-01"], "end": ["2009-06-30"]}).to_csv(spath, index=False)
        assert load_crisis_labels(wids, wb, source=spath)[1] == "crisis"


def _test_crisis_taxonomy():
    """The named taxonomy loads, parses dates, and overlays windows WITHOUT
    touching the binary labeler. A 2008-food window overlaps named crises yet the
    NBER binary label is independent."""
    tax = load_crisis_taxonomy()
    assert len(tax) >= 10, len(tax)
    assert {"name", "start", "end", "type"} <= set(tax.columns)
    assert tax["start"].notna().all() and tax["end"].notna().all()
    wb = {0: (pd.Timestamp("2008-03-01"), pd.Timestamp("2008-05-31")),   # GFC + food
          1: (pd.Timestamp("2003-01-01"), pd.Timestamp("2003-03-31"))}   # quiet
    tags = tag_windows_with_crises([0, 1], wb, taxonomy=tax)
    assert any("Financial" in n or "food" in n.lower() for n in tags[0]), tags[0]
    assert tags[1] == [], tags[1]
    # overlay must not change the binary labeler
    assert load_crisis_labels([0, 1], wb, source=None) == {0: "calm", 1: "calm"}


def _test_classical_generator_reproduces_box():
    """A purely-classical generator with symmetric marginals and per-cell
    correlations (+c,+c,+c,-c) must, pushed through the SAME estimator, reproduce
    the spec's post-selection box: delta~0, s_odd~4c, ctx~4c-2 (no contextual
    ingredient)."""
    rng = np.random.default_rng(7)
    c = 0.8
    params = {"scaleA": (2.0, 1.0), "scaleB": (2.0, 1.0),
              "cells": {(0, 0): (0.5, 0.5, c), (0, 1): (0.5, 0.5, c),
                        (1, 0): (0.5, 0.5, c), (1, 1): (0.5, 0.5, -c)}}
    ra, rb = simulate_pair(params, 20000, rng)
    res = _pair_stats_from_returns(ra, rb, n_min=10)
    assert res["valid"]
    assert abs(res["delta"]) < 0.08, res["delta"]
    assert abs(res["s_odd"] - 4 * c) < 0.15, res["s_odd"]
    assert abs(res["ctx"] - (4 * c - 2)) < 0.15, res["ctx"]


def _synthetic_panel_and_stats(seed=0, n=120):
    """Helper: build a tiny (permno, date, ret) panel + matching pair_window_stats."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", periods=n)
    permnos = [11, 22, 33]
    recs = [(p, d, float(rng.normal(0, 0.02))) for p in permnos for d in dates]
    ret = pd.DataFrame(recs, columns=["permno", "date", "ret"])
    rw = ret.pivot_table(index="date", columns="permno", values="ret")
    rows = run_window(0, dates[0], dates[-1], permnos, rw, n_min=10)
    for r in rows:
        r["regime"] = "calm"
    return ret, pd.DataFrame(rows)


def _test_classical_null_runs():
    """classical_null_reproduction returns the expected schema on a tiny panel."""
    ret, stats = _synthetic_panel_and_stats(seed=1)
    out = classical_null_reproduction(stats, ret, n_min=10, seed=0)
    assert len(out) >= 1
    for col in ("s_odd", "delta", "ctx", "valid", "source"):
        assert col in out.columns
    assert (out["source"] == "classical_null").all()


def _test_exogenous_contrast():
    """Shared exogenous context empties the off-diagonal cells (only N00/N11)."""
    ret, stats = _synthetic_panel_and_stats(seed=2)
    dates = sorted(ret["date"].unique())
    exo = {pd.Timestamp(d): (0 if i % 2 == 0 else 1) for i, d in enumerate(dates)}
    out = exogenous_context_contrast(stats, ret, exo, n_min=5)
    assert {"E00", "E11", "N00", "N11", "dcmov", "valid_diag"} <= set(out.columns)
    assert (out["N00"] + out["N11"] > 0).all()


def _test_controls():
    """Placebo collapses CTX vs the contextual box; small-sample s_odd shrinks
    with N; max-stat + sector helpers run."""
    rng = np.random.default_rng(9)
    # placebo on a contextual box: empirical ctx>0, permuted ctx should drop
    params = {"scaleA": (2.0, 1.0), "scaleB": (2.0, 1.0),
              "cells": {(0, 0): (0.5, 0.5, 0.8), (0, 1): (0.5, 0.5, 0.8),
                        (1, 0): (0.5, 0.5, 0.8), (1, 1): (0.5, 0.5, -0.8)}}
    ra, rb = simulate_pair(params, 4000, rng)
    dates = pd.bdate_range("2008-01-01", periods=len(ra))
    ret = pd.concat([
        pd.DataFrame({"permno": 1, "date": dates, "ret": ra}),
        pd.DataFrame({"permno": 2, "date": dates, "ret": rb})])
    emp = _pair_stats_from_returns(ra, rb, n_min=10)
    stats = pd.DataFrame([_pack_row(0, dates[0], dates[-1], 1, 2, emp["E"],
                                    emp["a_marg"], emp["b_marg"], emp["N"],
                                    emp["s_odd"], emp["delta"], emp["ctx"], emp["valid"])])
    placebo = permutation_placebo(stats, ret, n_min=10, n_perm=8, seed=0)
    assert placebo["ctx"].mean() < emp["ctx"] - 0.3, (placebo["ctx"].mean(), emp["ctx"])

    ss = small_sample_null([10, 200], n_sims=300, seed=0)
    assert ss.loc[ss.N == 10, "s_odd_mean"].iloc[0] > ss.loc[ss.N == 200, "s_odd_mean"].iloc[0]

    mx = max_stat_null(stats, ret, n_min=10, n_perm=20, seed=0)
    assert "ctx_threshold" in mx.columns and len(mx) == 1
    sec = sector_stratified_rates(stats, {1: "Tech", 2: "Tech"})
    assert "cbd_rate" in sec.columns and (sec["pair_sector"] == "within:Tech").any()


_TESTS = (_test_handchecked, _test_postselection, _test_anchors, _test_pipeline_smoke,
          _test_theta_quantile, _test_threshold_sweep, _test_null_theta_threading,
          _test_reliability_and_null_gate,
          _test_vectorized_equivalence, _test_crisis_labels, _test_crisis_taxonomy,
          _test_classical_generator_reproduces_box, _test_classical_null_runs,
          _test_exogenous_contrast, _test_controls)


def run_tests():
    for t in _TESTS:
        t(); log.info(f"PASS {t.__name__}")
    log.info("ALL TESTS PASSED")


def parse_args():
    ap = argparse.ArgumentParser(description="CbD analysis over extractor parquet outputs.")
    ap.add_argument("--data-dir", default="wrds_sp500_data")
    ap.add_argument("--out", default="wrds_sp500_data/pair_window_stats.parquet")
    ap.add_argument("--partition", action="store_true",
                    help="memory-safe full-span mode: write per-window shards + compact "
                         "aggregates into a DIRECTORY (default wrds_sp500_data/pair_window_stats) "
                         "instead of one monolithic frame. Use for the full 1990-2025 run.")
    ap.add_argument("--partition-out", default="wrds_sp500_data/pair_window_stats",
                    help="output directory for --partition shards + aggregates")
    ap.add_argument("--null-pairs-per-window", type=int, default=400,
                    help="per-window cap on the streamed deflation-null sample (--partition)")
    ap.add_argument("--no-null-stream", action="store_true",
                    help="disable the streamed deflation null in --partition mode")
    ap.add_argument("--n-min", type=int, default=10, help="minimum count per regime cell")
    ap.add_argument("--sample", type=int, default=None, help="process only first N windows")
    ap.add_argument("--crisis-source", default=None,
                    help="VIX/NBER labels: 'nber', or a parquet/csv path "
                         "(VIX: columns date,vix; recession spans: columns start,end)")
    ap.add_argument("--crisis-threshold", type=float, default=None,
                    help="fixed VIX level for crisis (default: sample-median aggregate)")
    ap.add_argument("--null", action="store_true",
                    help="also run the classical-null deflation and save it")
    ap.add_argument("--null-out", default=None, help="path for classical-null parquet")
    ap.add_argument("--null-max-pairs", type=int, default=2000,
                    help="cap on simulated pairs for the null (distribution, not all pairs)")
    ap.add_argument("--theta-quantile", type=float, default=0.5,
                    help="per-stock |R| quantile for the large/small regime split (spec default 0.5)")
    ap.add_argument("--sweep", nargs="?", const="default", default=None,
                    help="run the theta robustness sweep -> threshold_sweep.parquet and "
                         "exit. Bare --sweep uses {0.25,0.40,0.50,0.75,0.90,0.95}; or pass "
                         "a comma list e.g. --sweep '0.25,0.5,0.9'")
    ap.add_argument("--sweep-out", default=None, help="path for the threshold-sweep parquet")
    ap.add_argument("--relaxed-n-min", type=int, default=3,
                    help="small-sample diagnostic N_min at high-theta sweep points")
    ap.add_argument("--no-sweep-null", action="store_true",
                    help="disable the null-relative diagnostic in the sweep")
    ap.add_argument("--reliability", action="store_true",
                    help="compute s_odd split-half reliability -> s_odd_reliability.parquet and exit")
    ap.add_argument("--null-gate-stats", action="store_true",
                    help="emit a per-window-dense classical-null pair_window_stats frame "
                         "(for the networks MR-QAP null baseline) -> "
                         "classical_null_gate_stats.parquet and exit")
    ap.add_argument("--gate-windows", type=int, default=6,
                    help="windows for --null-gate-stats / reliability emission (cost control)")
    ap.add_argument("--gate-nodes", type=int, default=60,
                    help="first-N nodes per window for the null-gate emission (cost control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="run unit tests and exit")
    return ap.parse_args()


_DEFAULT_SWEEP = (0.25, 0.40, 0.42, 0.45, 0.48, 0.50, 0.75, 0.90, 0.95)


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        run_tests()
    elif args.sweep:
        quantiles = (_DEFAULT_SWEEP if args.sweep == "default"
                     else tuple(float(q) for q in args.sweep.split(",")))
        relaxed = tuple(q for q in quantiles if q >= 0.90)
        sw = sweep_thresholds(args.data_dir, quantiles=quantiles, n_min=args.n_min,
                              crisis_source=args.crisis_source,
                              crisis_threshold=args.crisis_threshold, sample=args.sample,
                              relaxed_n_min=args.relaxed_n_min, relaxed_quantiles=relaxed,
                              null_relative=not args.no_sweep_null, seed=args.seed)
        sweep_out = args.sweep_out or os.path.join(args.data_dir, "threshold_sweep.parquet")
        try:
            sw.to_parquet(sweep_out, index=False)
        except Exception:                                  # noqa: BLE001
            sweep_out = sweep_out.replace(".parquet", ".csv"); sw.to_csv(sweep_out, index=False)
        log.info(f"wrote threshold sweep -> {sweep_out}")
    elif args.reliability:
        d = load_data(args.data_dir)
        rel = s_odd_split_half_reliability(
            d["window_eligibility"], d["daily_returns"], n_min=args.n_min,
            theta_q=args.theta_quantile, sample=args.sample, seed=args.seed)
        out = os.path.join(args.data_dir, "s_odd_reliability.parquet")
        try:
            rel.to_parquet(out, index=False)
        except Exception:                                  # noqa: BLE001
            out = out.replace(".parquet", ".csv"); rel.to_csv(out, index=False)
        log.info(f"wrote s_odd reliability -> {out}")
    elif args.null_gate_stats:
        d = load_data(args.data_dir)
        stats_path = os.path.join(args.data_dir, "pair_window_stats")
        nd = emit_null_gate_stats(stats_path, d["daily_returns"],
                                  n_windows=args.gate_windows, n_nodes=args.gate_nodes,
                                  theta_q=args.theta_quantile, n_min=args.n_min,
                                  seed=args.seed)
        out = os.path.join(args.data_dir, "classical_null_gate_stats.parquet")
        try:
            nd.to_parquet(out, index=False)
        except Exception:                                  # noqa: BLE001
            out = out.replace(".parquet", ".csv"); nd.to_csv(out, index=False)
        log.info(f"wrote {len(nd):,} null-gate rows -> {out}")
    elif args.partition:
        analyze_streaming(args.data_dir, args.partition_out, args.n_min,
                          sample=args.sample, crisis_source=args.crisis_source,
                          crisis_threshold=args.crisis_threshold,
                          theta_q=args.theta_quantile, run_null=not args.no_null_stream,
                          null_pairs_per_window=args.null_pairs_per_window, seed=args.seed)
    else:
        analyze(args.data_dir, args.out, args.n_min,
                sample=args.sample, crisis_source=args.crisis_source,
                crisis_threshold=args.crisis_threshold, run_null=args.null,
                null_out=args.null_out, null_max_pairs=args.null_max_pairs,
                seed=args.seed, theta_q=args.theta_quantile)
