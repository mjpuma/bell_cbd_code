#!/usr/bin/env python3
"""
plots.py
========
High-quality figures for the CbD S&P 500 crisis analysis. Reads ONLY the parquet
files produced by the two-stage pipeline (the extractor's panel + the analysis
output) -- it never re-pulls from WRDS -- and writes each figure as both PNG
(300 dpi) and PDF (vector) into the output directory.

USAGE
-----
    python src/plots.py --data-dir wrds_sp500_data --out figures/
    python src/plots.py --test          # smoke test (no real data needed)

DESIGN
------
Every plotting function takes a DataFrame (or two) and RETURNS a matplotlib
Figure, so it is unit-testable and never calls plt.show(). Figures degrade
gracefully: if a required column is missing the function logs a warning and
returns a valid (placeholder) Figure instead of crashing. All figures share a
clean, minimal rcParams block with a colorblind-safe (Okabe-Ito) palette.

INPUTS (parquet in --data-dir)
------------------------------
  pair_window_stats.parquet   analysis output (s_odd, delta, ctx, N.., regime)
  window_eligibility.parquet  window_id, win_start, win_end, permno
  daily_returns.parquet       permno, date, ret
  identifiers.parquet         optional; permno + 'sector' enables Fig (g-sector)
  classical_null_stats.parquet  optional; enables the empirical-vs-null overlay
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                       # headless; never opens a window
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Reuse the locked threshold definition so Fig (e) cannot diverge from the math.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from cbd_analysis import window_threshold
except Exception:                            # noqa: BLE001
    def window_threshold(abs_ret):           # fallback mirrors the spec default
        return float(np.median(abs_ret))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("plots")

# Okabe-Ito colorblind-safe palette.
OKABE = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
}
REGIME_COLORS = {"crisis": OKABE["vermillion"], "calm": OKABE["blue"]}


def set_style() -> None:
    """Apply a consistent, minimal, readable rcParams block (fonts >= 11)."""
    plt.rcParams.update({
        "figure.figsize": (8.0, 5.0),
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.8,
    })


def save_figure(fig: Figure, outdir: str, name: str, dpi: int = 300) -> list[str]:
    """Save a Figure as both PNG (raster, `dpi`) and PDF (vector). Returns paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    log.info(f"  saved {name}: {', '.join(os.path.basename(p) for p in paths)}")
    return paths


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _missing(df: Optional[pd.DataFrame], cols, name: str) -> bool:
    """Return True (and log) if df is None/empty or lacks any required column."""
    if df is None or len(df) == 0:
        log.warning(f"{name}: no data; skipping.")
        return True
    miss = [c for c in cols if c not in df.columns]
    if miss:
        log.warning(f"{name}: missing column(s) {miss}; skipping.")
        return True
    return False


def _placeholder(title: str, msg: str) -> Figure:
    """A valid Figure that explains why a panel is empty (graceful degradation)."""
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True,
            color=OKABE["grey"], fontsize=12, transform=ax.transAxes)
    return fig


def _caption(fig: Figure, text: str) -> None:
    """One-line caption/subtitle at the bottom of the figure."""
    fig.text(0.005, 0.005, text, ha="left", va="bottom",
             fontsize=9, color=OKABE["grey"])


def _valid(stats: pd.DataFrame) -> pd.DataFrame:
    """Restrict to valid pair-windows when the flag is present."""
    return stats[stats["valid"]] if "valid" in stats.columns else stats


def window_regime_table(stats: pd.DataFrame) -> pd.DataFrame:
    """One row per window_id: win_start + its crisis/calm label (from the stats)."""
    cols = [c for c in ("window_id", "win_start", "regime") if c in stats.columns]
    g = stats[cols].drop_duplicates("window_id")
    if "win_start" in g.columns:
        g = g.assign(win_start=pd.to_datetime(g["win_start"])).sort_values("win_start")
    return g.reset_index(drop=True)


def _shade_crisis(ax, reg_tbl: pd.DataFrame) -> None:
    """Shade contiguous crisis windows on a time axis."""
    if "regime" not in reg_tbl.columns or "win_start" not in reg_tbl.columns:
        return
    crisis = reg_tbl["regime"].eq("crisis").to_numpy()
    x = reg_tbl["win_start"].to_numpy()
    i = 0
    labeled = False
    while i < len(crisis):
        if crisis[i]:
            j = i
            while j + 1 < len(crisis) and crisis[j + 1]:
                j += 1
            ax.axvspan(x[i], x[j], color=OKABE["vermillion"], alpha=0.12,
                       label="crisis" if not labeled else None)
            labeled = True
            i = j + 1
        else:
            i += 1


def _data_window_caption(stats: pd.DataFrame) -> str:
    """'<start> .. <end>, N=<valid pairs>' summary for subtitles."""
    if "win_start" not in stats.columns:
        return f"N={len(stats):,} rows"
    ws = pd.to_datetime(stats["win_start"]).min()
    we = pd.to_datetime(stats.get("win_end", stats["win_start"])).max()
    n = int(_valid(stats).shape[0])
    return f"{ws:%Y-%m-%d} .. {we:%Y-%m-%d}   |   N={n:,} valid pair-windows"


# ==========================================================================
# (a) Eligible constituents per window over time, crisis-shaded
# ==========================================================================
def plot_eligible_over_time(elig: pd.DataFrame, stats: pd.DataFrame) -> Figure:
    name = "fig_a_eligible_constituents"
    if _missing(elig, ("window_id", "win_start", "permno"), name):
        return _placeholder("Eligible constituents per window", "window_eligibility missing")
    counts = (elig.assign(win_start=pd.to_datetime(elig["win_start"]))
              .groupby(["window_id", "win_start"])["permno"].nunique()
              .reset_index(name="n").sort_values("win_start"))
    reg = window_regime_table(stats) if stats is not None else pd.DataFrame()
    if not reg.empty:
        counts = counts.merge(reg[["window_id", "regime"]], on="window_id", how="left")

    fig, ax = plt.subplots()
    ax.plot(counts["win_start"], counts["n"], color=OKABE["blue"], marker="o", ms=3)
    if "regime" in counts.columns:
        _shade_crisis(ax, counts.rename(columns={"regime": "regime"}))
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
    ax.set_title("Eligible S&P 500 constituents per window")
    ax.set_xlabel("Window start date")
    ax.set_ylabel("Number of eligible names")
    ax.set_ylim(bottom=0)
    fig.autofmt_xdate()
    _caption(fig, _data_window_caption(stats if stats is not None else elig))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (b) Coverage / missingness by year
# ==========================================================================
def plot_coverage_missingness(returns: pd.DataFrame) -> Figure:
    name = "fig_b_coverage_missingness"
    if _missing(returns, ("permno", "date", "ret"), name):
        return _placeholder("Coverage / missingness", "daily_returns missing")
    df = returns.assign(year=pd.to_datetime(returns["date"]).dt.year)
    by = df.groupby("year")["ret"].agg(
        avail=lambda s: float(s.notna().mean()),
        zero=lambda s: float((s.fillna(np.nan) == 0).mean())).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(by["year"], 100 * by["avail"], color=OKABE["green"])
    ax1.set_title("Return availability")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Non-missing return-days (%)")
    ax1.set_ylim(0, 100)
    ax2.bar(by["year"], 100 * by["zero"], color=OKABE["orange"])
    ax2.set_title("Exact-zero returns")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Zero-return days (%)")
    _caption(fig, "sgn(0)=0 days are dropped from the +-1 series; thin names inflate them.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (c) Cross-sectional distribution of returns and |returns|
# ==========================================================================
def plot_return_distributions(returns: pd.DataFrame) -> Figure:
    name = "fig_c_return_distributions"
    if _missing(returns, ("ret",), name):
        return _placeholder("Return distributions", "daily_returns missing")
    r = pd.to_numeric(returns["ret"], errors="coerce").dropna().to_numpy()
    r = r[np.abs(r) < np.nanquantile(np.abs(r), 0.999)]   # trim tails for display
    absr = np.abs(r)
    theta = float(np.median(absr))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.hist(r, bins=80, color=OKABE["blue"], alpha=0.85)
    ax1.axvline(0, color=OKABE["black"], lw=0.8)
    ax1.set_title("Daily returns")
    ax1.set_xlabel("Return"); ax1.set_ylabel("Count")
    ax2.hist(absr, bins=80, color=OKABE["skyblue"], alpha=0.9)
    ax2.axvline(theta, color=OKABE["vermillion"], lw=1.5, ls="--",
                label=f"median |R| = {theta:.3f}")
    ax2.set_title("|Return| (magnitude regimes)")
    ax2.set_xlabel("|Return|"); ax2.set_ylabel("Count")
    ax2.legend()
    _caption(fig, "The median |R| is the per-stock regime threshold theta (large vs small move).")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (d) Four-cell count distribution across pairs
# ==========================================================================
def plot_cell_counts(stats: pd.DataFrame) -> Figure:
    name = "fig_d_cell_counts"
    cells = ["N00", "N01", "N10", "N11"]
    if _missing(stats, cells, name):
        return _placeholder("Four-cell counts", "N00..N11 missing")
    data = [pd.to_numeric(stats[c], errors="coerce").dropna().to_numpy() for c in cells]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=["N00\n(lg,lg)", "N01\n(lg,sm)", "N10\n(sm,lg)", "N11\n(sm,sm)"],
                    showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(OKABE["skyblue"]); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color(OKABE["black"])
    ax.set_title("Per-cell sample sizes across pairs")
    ax.set_ylabel("Count per cell (days)")
    ax.set_xlabel("Magnitude cell (A,B)")
    _caption(fig, "Confirms the N_min rule (>=10/cell) is not discarding most pairs.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (e) Threshold theta over time (recomputed from the panel)
# ==========================================================================
def plot_threshold_over_time(returns: pd.DataFrame, elig: pd.DataFrame) -> Figure:
    name = "fig_e_threshold_over_time"
    if _missing(returns, ("permno", "date", "ret"), name) or \
       _missing(elig, ("window_id", "win_start", "win_end", "permno"), name):
        return _placeholder("Threshold theta over time", "panel/eligibility missing")
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    by_permno = {int(p): g.set_index("date")["ret"] for p, g in ret.groupby("permno")}
    rows = []
    for wid, g in elig.assign(win_start=pd.to_datetime(elig["win_start"]),
                              win_end=pd.to_datetime(elig["win_end"])).groupby("window_id"):
        ws, we = g["win_start"].iloc[0], g["win_end"].iloc[0]
        ths = []
        for p in g["permno"].astype(int):
            if p not in by_permno:
                continue
            s = by_permno[p]
            a = np.abs(s[(s.index >= ws) & (s.index <= we)].dropna().to_numpy())
            if a.size:
                ths.append(window_threshold(a))
        if ths:
            rows.append((ws, np.median(ths), np.quantile(ths, 0.1), np.quantile(ths, 0.9)))
    if not rows:
        return _placeholder("Threshold theta over time", "no overlap between returns and windows")
    t = pd.DataFrame(rows, columns=["win_start", "med", "lo", "hi"]).sort_values("win_start")

    fig, ax = plt.subplots()
    ax.fill_between(t["win_start"], t["lo"], t["hi"], color=OKABE["skyblue"], alpha=0.3,
                    label="10-90th pct across names")
    ax.plot(t["win_start"], t["med"], color=OKABE["blue"], label="median name theta")
    ax.set_title(r"Per-stock regime threshold $\theta$ (median |R|) over time")
    ax.set_xlabel("Window start date"); ax.set_ylabel(r"$\theta$  (|return|)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    _caption(fig, "theta rises in volatile periods; recomputed from the daily panel per window.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (f) Distributions of s_odd, delta, ctx split by regime
# ==========================================================================
def plot_statistic_distributions(stats: pd.DataFrame) -> Figure:
    name = "fig_f_statistic_distributions"
    if _missing(stats, ("s_odd", "delta", "ctx"), name):
        return _placeholder("s_odd / delta / ctx distributions", "statistics missing")
    v = _valid(stats)
    has_reg = "regime" in v.columns
    quantities = [("s_odd", r"$s_{\mathrm{odd}}$", 2.0),
                  ("delta", r"$\Delta$", None),
                  ("ctx", r"CTX $=s_{\mathrm{odd}}-\Delta-2$", 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (col, lab, ref) in zip(axes, quantities):
        groups = ([("crisis", v[v.regime == "crisis"]), ("calm", v[v.regime == "calm"])]
                  if has_reg else [("all", v)])
        for gname, gdf in groups:
            x = pd.to_numeric(gdf[col], errors="coerce").dropna().to_numpy()
            if x.size == 0:
                continue
            ax.hist(x, bins=50, density=True, alpha=0.55,
                    color=REGIME_COLORS.get(gname, OKABE["blue"]), label=gname)
        if ref is not None:
            ax.axvline(ref, color=OKABE["black"], ls="--", lw=1.2)
        ax.set_title(lab); ax.set_xlabel(lab); ax.set_ylabel("density")
        if has_reg:
            ax.legend()
    fig.suptitle("Cross-sectional CbD statistics by regime", fontweight="bold")
    _caption(fig, _data_window_caption(stats))
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


# ==========================================================================
# (g) Headline exhibit: naive vs CbD-corrected violation rates, crisis vs calm
# ==========================================================================
def _rates_by_regime(v: pd.DataFrame) -> pd.DataFrame:
    has_reg = "regime" in v.columns
    keys = ["crisis", "calm"] if has_reg else ["all"]
    out = []
    for k in keys:
        d = v[v.regime == k] if has_reg else v
        if len(d) == 0:
            continue
        out.append({"regime": k, "n": len(d),
                    "naive": float((pd.to_numeric(d["s_odd"], errors="coerce") > 2).mean()),
                    "cbd": float((pd.to_numeric(d["ctx"], errors="coerce") > 0).mean())})
    return pd.DataFrame(out)


def plot_violation_rates(stats: pd.DataFrame) -> Figure:
    name = "fig_g_violation_rates"
    if _missing(stats, ("s_odd", "ctx"), name):
        return _placeholder("Violation rates", "s_odd/ctx missing")
    r = _rates_by_regime(_valid(stats))
    if r.empty:
        return _placeholder("Violation rates", "no valid pairs")
    x = np.arange(len(r)); w = 0.38
    fig, ax = plt.subplots()
    ax.bar(x - w / 2, 100 * r["naive"], w, color=OKABE["vermillion"],
           label=r"naive  ($s_{\mathrm{odd}}>2$)")
    ax.bar(x + w / 2, 100 * r["cbd"], w, color=OKABE["blue"],
           label=r"CbD-corrected  (CTX$>0$)")
    for i, row in r.iterrows():
        ax.text(i - w / 2, 100 * row["naive"], f"{100*row['naive']:.1f}%",
                ha="center", va="bottom", fontsize=10)
        ax.text(i + w / 2, 100 * row["cbd"], f"{100*row['cbd']:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(N={n:,})" for g, n in zip(r["regime"], r["n"])])
    ax.set_ylabel("Share of pairs (%)")
    ax.set_title("Deflation: naive violation rate vs CbD-corrected rate")
    ax.legend(loc="upper right")
    _caption(fig, "The naive rate inflates (esp. in crisis); CTX>0 collapses it -- the deflation.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (h) Scatter s_odd vs delta colored by regime, with CTX=0 boundary
# ==========================================================================
def plot_sodd_vs_delta(stats: pd.DataFrame, max_points: int = 20000) -> Figure:
    name = "fig_h_sodd_vs_delta"
    if _missing(stats, ("s_odd", "delta"), name):
        return _placeholder("s_odd vs delta", "s_odd/delta missing")
    v = _valid(stats).copy()
    v["s_odd"] = pd.to_numeric(v["s_odd"], errors="coerce")
    v["delta"] = pd.to_numeric(v["delta"], errors="coerce")
    v = v.dropna(subset=["s_odd", "delta"])
    if len(v) > max_points:
        v = v.sample(max_points, random_state=0)
    fig, ax = plt.subplots()
    has_reg = "regime" in v.columns
    for gname in (["crisis", "calm"] if has_reg else ["all"]):
        d = v[v.regime == gname] if has_reg else v
        if len(d) == 0:
            continue
        ax.scatter(d["delta"], d["s_odd"], s=6, alpha=0.35,
                   color=REGIME_COLORS.get(gname, OKABE["blue"]), label=gname, edgecolors="none")
    dmax = float(np.nanmax(v["delta"])) if len(v) else 1.0
    xs = np.linspace(0, max(dmax, 0.1), 100)
    ax.plot(xs, xs + 2, color=OKABE["black"], lw=1.5, ls="--", label=r"CTX$=0$: $s_{\mathrm{odd}}=\Delta+2$")
    ax.set_xlabel(r"$\Delta$ (inconsistency / direct influence)")
    ax.set_ylabel(r"$s_{\mathrm{odd}}$")
    ax.set_title(r"$s_{\mathrm{odd}}$ vs $\Delta$ — points above the line are CTX$>0$")
    ax.legend(loc="upper left")
    _caption(fig, _data_window_caption(stats))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (i) Empirical vs classical-null CTX overlay
# ==========================================================================
def plot_ctx_overlay(stats: pd.DataFrame, null_stats: Optional[pd.DataFrame]) -> Figure:
    name = "fig_i_ctx_overlay"
    if _missing(stats, ("ctx",), name):
        return _placeholder("Empirical vs classical-null CTX", "ctx missing")
    if null_stats is None or "ctx" not in getattr(null_stats, "columns", []):
        log.warning(f"{name}: classical-null stats unavailable; plotting empirical only.")
    emp = pd.to_numeric(_valid(stats)["ctx"], errors="coerce").dropna().to_numpy()
    fig, ax = plt.subplots()
    bins = np.linspace(min(emp.min(), -2), max(emp.max(), 1), 60) if emp.size else 60
    ax.hist(emp, bins=bins, density=True, alpha=0.55, color=OKABE["blue"], label="empirical")
    if null_stats is not None and "ctx" in getattr(null_stats, "columns", []):
        nul = pd.to_numeric(_valid(null_stats)["ctx"], errors="coerce").dropna().to_numpy()
        if nul.size:
            ax.hist(nul, bins=bins, density=True, alpha=0.55, color=OKABE["orange"],
                    label="classical null")
    ax.axvline(0, color=OKABE["black"], ls="--", lw=1.2, label="CTX=0")
    ax.set_title("CTX: empirical vs classical-null (deflation reproduction)")
    ax.set_xlabel(r"CTX $=s_{\mathrm{odd}}-\Delta-2$"); ax.set_ylabel("density")
    ax.legend(loc="upper right")
    _caption(fig, "Overlap => a purely classical generator reproduces the apparent contextuality.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (j) threshold robustness sweep: CbD rate vs theta-percentile
# ==========================================================================
def plot_threshold_sweep(sweep: pd.DataFrame) -> Figure:
    name = "fig_j_threshold_sweep"
    if _missing(sweep, ("theta_q", "regime", "cbd_rate", "n_valid"), name):
        return _placeholder("Threshold sweep", "threshold_sweep parquet missing")
    n_min = int(sweep["n_min"].iloc[0]) if "n_min" in sweep.columns else None
    rlx = int(sweep["relaxed_n_min"].iloc[0]) if "relaxed_n_min" in sweep.columns else None
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for reg in ("crisis", "calm"):
        d = sweep[sweep["regime"] == reg].sort_values("theta_q")
        if d.empty:
            continue
        col = REGIME_COLORS.get(reg, OKABE["blue"])
        ax.plot(100 * d["theta_q"], 100 * d["cbd_rate"], marker="o", color=col,
                label=f"{reg}: CbD (CTX>0), N_min={n_min}")
        ax2.plot(100 * d["theta_q"], d["n_valid"].clip(lower=1), color=col, ls=":",
                 lw=1.2, alpha=0.5)
        if "is_relaxed_diag_q" in d.columns and "cbd_rate_relaxed" in d.columns:
            dr = d[d["is_relaxed_diag_q"]]
            if len(dr):
                ax.scatter(100 * dr["theta_q"], 100 * dr["cbd_rate_relaxed"],
                           facecolors="none", edgecolors=col, marker="s", s=90, zorder=5,
                           label=f"{reg}: relaxed N_min={rlx} (small-N diag)")
                if "cbd_rate_null_relaxed" in dr.columns and dr["cbd_rate_null_relaxed"].notna().any():
                    ax.scatter(100 * dr["theta_q"], 100 * dr["cbd_rate_null_relaxed"],
                               color=col, marker="x", s=70, zorder=6, linewidths=1.8,
                               label=f"{reg}: classical-null (relaxed)")
    # finite-N ctx>0 noise floor (regime-independent) at the relaxed quantiles
    if "cbd_rate_smallN_floor" in sweep.columns:
        fl = sweep.dropna(subset=["cbd_rate_smallN_floor"]).drop_duplicates("theta_q")
        if len(fl):
            ax.scatter(100 * fl["theta_q"], 100 * fl["cbd_rate_smallN_floor"],
                       color=OKABE["green"], marker="_", s=260, zorder=7, linewidths=2.2,
                       label="finite-N ctx>0 floor")
    ax.set_xlabel(r"$\theta$ percentile of $|R|$  (regime 0 = large-move, $|R|\geq\theta$)")
    ax.set_ylabel("CbD-corrected rate, CTX>0 (%)")
    ax.set_ylim(bottom=0)
    ax2.set_ylabel("valid pair-windows (dotted, log)")
    ax2.set_yscale("log")
    ax2.grid(False)
    ax.set_title("Deflation stability across the magnitude threshold")
    h1, l1 = ax.get_legend_handles_labels()
    if h1:
        ax.legend(loc="upper left", fontsize=9)
    _caption(fig, "Solid = strict-N_min CbD rate; squares = relaxed-N_min diagnostic, "
                  "compared to x = classical-null and \u2014 = cell-size-matched finite-N "
                  "floor (same condition); dotted = valid denominator (log) = "
                  "well-posedness boundary. Strict rate \u2248 0 across the well-posed band.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# (optional) sector-stratified headline exhibit
# ==========================================================================
def plot_violation_rates_by_sector(stats: pd.DataFrame, sector_map: dict) -> Figure:
    name = "fig_g_sector_violation_rates"
    if _missing(stats, ("s_odd", "ctx", "permno_a", "permno_b"), name):
        return _placeholder("Sector violation rates", "stats missing")
    if not sector_map:
        log.warning(f"{name}: no sector map (identifiers lack a 'sector' column); skipping.")
        return _placeholder("Sector violation rates", "no sector identifiers available")
    v = _valid(stats).copy()
    sa = v["permno_a"].astype(int).map(sector_map)
    sb = v["permno_b"].astype(int).map(sector_map)
    v["pair_sector"] = np.where(sa.eq(sb) & sa.notna(), sa.astype(str), "cross")
    rows = []
    for sect, d in v.groupby("pair_sector"):
        rows.append({"sector": sect, "n": len(d),
                     "naive": float((pd.to_numeric(d["s_odd"], errors="coerce") > 2).mean()),
                     "cbd": float((pd.to_numeric(d["ctx"], errors="coerce") > 0).mean())})
    r = pd.DataFrame(rows).sort_values("naive", ascending=False)
    x = np.arange(len(r)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(r)), 5))
    ax.bar(x - w / 2, 100 * r["naive"], w, color=OKABE["vermillion"], label="naive (s_odd>2)")
    ax.bar(x + w / 2, 100 * r["cbd"], w, color=OKABE["blue"], label="CbD (CTX>0)")
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(N={n:,})" for s, n in zip(r["sector"], r["n"])],
                                         rotation=30, ha="right")
    ax.set_ylabel("Share of pairs (%)")
    ax.set_title("Violation rates by pair sector (within-sector vs cross)")
    ax.legend(loc="upper right")
    _caption(fig, _data_window_caption(stats))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# I/O + driver
# ==========================================================================
def _read_optional(path: str) -> Optional[pd.DataFrame]:
    for ext in (".parquet", ".csv"):
        p = path if path.endswith(ext) else path + ext
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    return None


def load_sector_map(data_dir: str) -> dict:
    """permno -> sector from identifiers.parquet, if a sector-like column exists."""
    idf = _read_optional(os.path.join(data_dir, "identifiers"))
    if idf is None or "permno" not in idf.columns:
        return {}
    cols = {c.lower(): c for c in idf.columns}
    for key in ("sector", "gsector", "gics_sector", "gicsector"):
        if key in cols:
            return dict(zip(idf["permno"].astype(int), idf[cols[key]].astype(str)))
    return {}


def build_all(data_dir: str, out_dir: str, stats_file: Optional[str] = None,
              null_file: Optional[str] = None) -> None:
    """Load the parquet inputs and write every figure to `out_dir`."""
    set_style()
    stats = _read_optional(stats_file or os.path.join(data_dir, "pair_window_stats"))
    if stats is None:
        log.error("pair_window_stats not found; run cbd_analysis.py first. Aborting.")
        return
    elig = _read_optional(os.path.join(data_dir, "window_eligibility"))
    returns = _read_optional(os.path.join(data_dir, "daily_returns"))
    null_stats = _read_optional(null_file or os.path.join(data_dir, "classical_null_stats"))
    sweep = _read_optional(os.path.join(data_dir, "threshold_sweep"))
    sector_map = load_sector_map(data_dir)

    figs = {
        "fig_a_eligible_constituents": lambda: plot_eligible_over_time(elig, stats),
        "fig_b_coverage_missingness": lambda: plot_coverage_missingness(returns),
        "fig_c_return_distributions": lambda: plot_return_distributions(returns),
        "fig_d_cell_counts": lambda: plot_cell_counts(stats),
        "fig_e_threshold_over_time": lambda: plot_threshold_over_time(returns, elig),
        "fig_f_statistic_distributions": lambda: plot_statistic_distributions(stats),
        "fig_g_violation_rates": lambda: plot_violation_rates(stats),
        "fig_h_sodd_vs_delta": lambda: plot_sodd_vs_delta(stats),
        "fig_i_ctx_overlay": lambda: plot_ctx_overlay(stats, null_stats),
        "fig_j_threshold_sweep": lambda: plot_threshold_sweep(sweep),
        "fig_g_sector_violation_rates": lambda: plot_violation_rates_by_sector(stats, sector_map),
    }
    for name, fn in figs.items():
        try:
            fig = fn()
            save_figure(fig, out_dir, name)
            plt.close(fig)
        except Exception as e:                       # noqa: BLE001  (never crash the batch)
            log.warning(f"{name}: failed to render ({e}); skipping.")
    log.info(f"done -> {out_dir}")


# ==========================================================================
# SMOKE TEST  (python src/plots.py --test)
# ==========================================================================
def _synthetic_stats(n_pairs: int = 200, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic pair_window_stats covering both regimes."""
    rng = np.random.default_rng(seed)
    starts = pd.bdate_range("2008-01-01", periods=8, freq="21D")
    rows = []
    for wid, ws in enumerate(starts):
        regime = "crisis" if 2 <= wid <= 4 else "calm"
        for k in range(n_pairs // len(starts)):
            s = float(np.clip(rng.normal(2.0 if regime == "crisis" else 1.6, 0.4), 0, 4))
            d = float(np.abs(rng.normal(0.3, 0.2)))
            rows.append({"window_id": wid, "win_start": ws,
                         "win_end": ws + pd.Timedelta(days=84),
                         "permno_a": 100 + (k % 5), "permno_b": 200 + (k % 4),
                         "E00": 0.4, "E01": 0.3, "E10": 0.3, "E11": -0.2,
                         "a00": -0.1, "a01": 0.0, "a10": 0.05, "a11": 0.1,
                         "b00": -0.1, "b01": 0.0, "b10": 0.05, "b11": 0.1,
                         "N00": int(rng.integers(12, 25)), "N01": int(rng.integers(12, 25)),
                         "N10": int(rng.integers(12, 25)), "N11": int(rng.integers(12, 25)),
                         "s_odd": s, "delta": d, "ctx": s - d - 2, "valid": True,
                         "regime": regime})
    return pd.DataFrame(rows)


def _synthetic_panel():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2008-01-01", periods=120)
    permnos = [100, 101, 102, 103, 104, 200, 201, 202, 203]
    rec = [(p, d, float(rng.normal(0, 0.02))) for p in permnos for d in dates]
    ret = pd.DataFrame(rec, columns=["permno", "date", "ret"])
    elig = pd.DataFrame([(wid, dates[0], dates[-1], p) for wid in range(3) for p in permnos],
                        columns=["window_id", "win_start", "win_end", "permno"])
    return ret, elig


def _synthetic_sweep() -> pd.DataFrame:
    """Tiny synthetic threshold_sweep frame for the (j) smoke test."""
    rows = []
    for q in (0.25, 0.40, 0.50, 0.75, 0.90, 0.95):
        diag = q >= 0.90
        for reg, base in (("crisis", 0.0), ("calm", 0.0)):
            rows.append({"theta_q": q, "regime": reg, "n_min": 10, "relaxed_n_min": 3,
                         "n_valid": int(5000 * (1 - q)), "naive_rate": 0.05,
                         "cbd_rate": base, "n_valid_relaxed": int(9000 * (1 - q)),
                         "naive_rate_relaxed": 0.06,
                         "cbd_rate_relaxed": 0.017 if diag else 0.0,
                         "cbd_rate_null_relaxed": 0.016 if diag else np.nan,
                         "n_valid_null_relaxed": int(2000 * (1 - q)) if diag else 0,
                         "cbd_rate_smallN_floor": 0.018 if diag else np.nan,
                         "is_relaxed_diag_q": diag})
    return pd.DataFrame(rows)


def run_tests() -> None:
    """Each plotting function must return a valid Figure on synthetic input."""
    set_style()
    stats = _synthetic_stats()
    returns, elig = _synthetic_panel()
    null_stats = stats.assign(ctx=stats["ctx"] - 0.1)
    checks = [
        ("a", plot_eligible_over_time(elig, stats)),
        ("b", plot_coverage_missingness(returns)),
        ("c", plot_return_distributions(returns)),
        ("d", plot_cell_counts(stats)),
        ("e", plot_threshold_over_time(returns, elig)),
        ("f", plot_statistic_distributions(stats)),
        ("g", plot_violation_rates(stats)),
        ("h", plot_sodd_vs_delta(stats)),
        ("i", plot_ctx_overlay(stats, null_stats)),
        ("j", plot_threshold_sweep(_synthetic_sweep())),
        ("g-sector", plot_violation_rates_by_sector(stats, {100: "Tech", 101: "Tech",
                                                            102: "Energy", 103: "Energy",
                                                            104: "Tech", 200: "Health",
                                                            201: "Health", 202: "Energy",
                                                            203: "Tech"})),
    ]
    for tag, fig in checks:
        assert isinstance(fig, Figure), tag
        plt.close(fig)
        log.info(f"PASS fig ({tag}) returns a Figure")
    # graceful degradation: missing columns must not crash, still return a Figure
    empty = pd.DataFrame({"window_id": [0]})
    for fn in (lambda: plot_violation_rates(empty),
               lambda: plot_sodd_vs_delta(empty),
               lambda: plot_cell_counts(empty),
               lambda: plot_ctx_overlay(empty, None)):
        fig = fn()
        assert isinstance(fig, Figure)
        plt.close(fig)
    log.info("ALL PLOT TESTS PASSED")


def parse_args():
    ap = argparse.ArgumentParser(description="Figures for the CbD S&P 500 analysis.")
    ap.add_argument("--data-dir", default="wrds_sp500_data",
                    help="directory with the pipeline parquet files")
    ap.add_argument("--out", default="figures/", help="output directory for figures")
    ap.add_argument("--stats-file", default=None,
                    help="path to pair_window_stats (default: <data-dir>/pair_window_stats.parquet)")
    ap.add_argument("--null-file", default=None,
                    help="path to classical_null_stats (default: <data-dir>/classical_null_stats.parquet)")
    ap.add_argument("--test", action="store_true", help="run the smoke test and exit")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        run_tests()
    else:
        build_all(args.data_dir, args.out, stats_file=args.stats_file, null_file=args.null_file)
