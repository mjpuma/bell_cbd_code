#!/usr/bin/env python3
"""
networks.py
===========
Network-topology layer for the CbD S&P 500 deflation study. Reads ONLY
``pair_window_stats.parquet`` (the Stage-2 analysis output) -- it never re-pulls
from WRDS -- and builds, per window, THREE parallel weighted graphs over the
eligible names, then compares their topology crisis vs calm and tests whether
each tier adds structure beyond the one below it.

THE THREE-TIER GRAPH HIERARCHY (per window)
-------------------------------------------
  pooled : N-weighted mean sign-correlation across all four CHSH cells
           (= ordinary co-movement / correlation network)        -- BASELINE
  e00    : both-large-move co-movement E00                        -- CANONICAL (a)
           (tail-coupling / contagion; the non-trivial object)
  s_odd  : the CHSH combination s_odd                             -- graph (b)

The MR-QAP hierarchy answers "does s_odd earn its place?":
  pooled -> e00 : does conditioning on large moves add structure beyond correlation?
  e00 -> s_odd  : does the CHSH combination add structure beyond tail coupling?
If s_odd is structurally ~= e00, CHSH is decoration (publish the tail-coupling
network paper); if s_odd adds structure beyond e00, that is the case for CHSH.

EDGE THRESHOLDS
---------------
  density-controlled top-q% (default 5%): primary for ALL topology metrics and
    MR-QAP -- comparing graphs of unequal density confounds structure with density.
  absolute s_odd >= 2 graph: a DESCRIPTIVE exhibit only ("amount of coupling",
    the network analog of the naive violation rate), reported as density/count.

USAGE
-----
    python src/networks.py --data-dir wrds_sp500_data --out figures/
    python src/networks.py --test     # smoke test (no real data needed)

Outputs network_metrics.parquet and network_qap.parquet into --data-dir, plus
figures into --out. Designed to STREAM window-by-window so it scales to the full
panel (it never concatenates all pair-windows into memory); pass a partitioned
directory of per-window parquet shards and it consumes them one at a time.
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import logging
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plots import (set_style, save_figure, OKABE, REGIME_COLORS,  # noqa: E402
                   _caption, _missing, _placeholder, _shade_crisis)
from cbd_analysis import load_crisis_taxonomy, tag_windows_with_crises  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("networks")

# Columns the network layer needs (projection keeps the streaming read light).
NEEDED_COLS = ["window_id", "win_start", "win_end", "permno_a", "permno_b",
               "E00", "E01", "E10", "E11", "N00", "N01", "N10", "N11",
               "s_odd", "valid", "regime"]

GRAPH_KINDS = ("pooled", "e00", "s_odd")          # density-matched topology tiers


# ==========================================================================
# STREAMING INPUT
# ==========================================================================
def iter_window_stats(path: str) -> Iterator[pd.DataFrame]:
    """Yield one valid-pair DataFrame per window_id.

    Streams so it never holds the whole panel in memory:
      * directory  -> treated as a partitioned dataset (one parquet shard per
                      window); each shard is read and yielded on its own.
      * .parquet/.csv file -> read with a column projection, then grouped by
                      window_id (the monolithic smoke-span path).
    """
    if os.path.isdir(path):
        shards = sorted(glob.glob(os.path.join(path, "*.parquet")))
        for sh in shards:
            df = pd.read_parquet(sh, columns=_avail_cols(sh))
            for _, g in df.groupby("window_id"):
                yield _prep_window(g)
        return
    file = path if os.path.exists(path) else path + ".parquet"
    if file.endswith(".parquet"):
        df = pd.read_parquet(file, columns=_avail_cols(file))
    else:
        df = pd.read_csv(file)
    for _, g in df.groupby("window_id"):
        yield _prep_window(g)


def _avail_cols(file: str) -> Optional[list]:
    """Intersection of NEEDED_COLS with the file schema (projection pushdown)."""
    try:
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(file).schema.names)
        return [c for c in NEEDED_COLS if c in have]
    except Exception:                                  # noqa: BLE001
        return None


def _prep_window(g: pd.DataFrame) -> pd.DataFrame:
    """Restrict to valid pairs and attach the three edge weights."""
    if "valid" in g.columns:
        g = g[g["valid"]]
    g = g.copy()
    g["w_pooled"] = pooled_weight(g)
    g["w_e00"] = g["E00"].astype(float) if "E00" in g.columns else np.nan
    g["w_s_odd"] = g["s_odd"].astype(float) if "s_odd" in g.columns else np.nan
    return g


def pooled_weight(df: pd.DataFrame) -> pd.Series:
    """N-weighted mean sign-correlation across the four CHSH cells == ordinary
    co-movement (the baseline correlation network)."""
    cells = [("E00", "N00"), ("E01", "N01"), ("E10", "N10"), ("E11", "N11")]
    if not all(e in df.columns and n in df.columns for e, n in cells):
        return pd.Series(np.nan, index=df.index)
    num = sum(df[e].fillna(0.0) * df[n].fillna(0.0) for e, n in cells)
    den = sum(df[n].fillna(0.0) for _, n in cells)
    return num / den.replace(0, np.nan)


_WEIGHT_COL = {"pooled": "w_pooled", "e00": "w_e00", "s_odd": "w_s_odd"}


# ==========================================================================
# GRAPH CONSTRUCTION
# ==========================================================================
def build_graph(df: pd.DataFrame, kind: str, edge_quantile: float = 0.05,
                abs_threshold: Optional[float] = None,
                sector_map: Optional[dict] = None) -> nx.Graph:
    """Build one weighted graph for a window.

    kind          : 'pooled' | 'e00' | 's_odd' (selects the edge weight).
    edge_quantile : keep the top `edge_quantile` fraction of pairs by |weight|
                    (density-controlled; the SAME count is applied to every kind
                    so the three graphs are density-matched). Ignored when
                    `abs_threshold` is given.
    abs_threshold : if set, keep edges with weight >= abs_threshold instead
                    (used for the descriptive absolute s_odd>=2 graph).
    sector_map    : optional permno -> sector; stored as a node attribute.
                    NOTE: this is the hook for a later agricultural node-filter;
                    NO sector filtering is performed here.
    """
    wcol = _WEIGHT_COL[kind]
    sub = df[["permno_a", "permno_b", wcol]].dropna()
    sub = sub[sub["permno_a"] != sub["permno_b"]]
    G = nx.Graph()
    if sub.empty:
        return G
    nodes = pd.unique(sub[["permno_a", "permno_b"]].to_numpy().ravel())
    for p in nodes:
        attrs = {"permno": int(p)}
        if sector_map:
            attrs["sector"] = sector_map.get(int(p))      # ag-filter hook (unused)
        G.add_node(int(p), **attrs)

    w = sub[wcol].to_numpy(dtype=float)
    if abs_threshold is not None:
        keep = w >= abs_threshold
    else:
        n_keep = max(1, int(round(edge_quantile * len(sub))))
        order = np.argsort(-np.abs(w))                     # strongest |weight| first
        keep = np.zeros(len(sub), dtype=bool)
        keep[order[:n_keep]] = True
    kept = sub.loc[keep]
    for a, b, wt in zip(kept["permno_a"].astype(int), kept["permno_b"].astype(int),
                        kept[wcol].astype(float)):
        G.add_edge(a, b, weight=float(wt))
    return G


def graph_metrics(G: nx.Graph) -> dict:
    """Edge density, average clustering, giant-component fraction, modularity and
    community count for a (thresholded) graph. Degrades to NaN/zeros if empty."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    out = {"n_nodes": n, "n_edges": m, "edge_density": np.nan,
           "avg_clustering": np.nan, "giant_frac": np.nan,
           "modularity": np.nan, "n_communities": np.nan}
    if n == 0:
        return out
    out["edge_density"] = nx.density(G)
    out["avg_clustering"] = nx.average_clustering(G) if m else 0.0
    if m:
        giant = max(nx.connected_components(G), key=len)
        out["giant_frac"] = len(giant) / n
        try:
            comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
            out["n_communities"] = len(comms)
            out["modularity"] = nx.community.modularity(G, comms, weight="weight")
        except Exception:                                  # noqa: BLE001
            pass
    else:
        out["giant_frac"] = 1.0 / n
        out["n_communities"] = n
        out["modularity"] = 0.0
    return out


def window_network_metrics(stats_path: str, edge_quantile: float = 0.05,
                           abs_threshold: float = 2.0,
                           sector_map: Optional[dict] = None,
                           taxonomy: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Stream the windows and compute topology metrics for every graph kind plus
    the descriptive absolute s_odd>=`abs_threshold` graph. One row per
    (window, graph). If `taxonomy` is given, each window is also tagged with the
    named crises and crisis-TYPES (financial/food/energy/mixed) it overlaps, for
    the cross-crisis breakdown (Step-5 reporting requirement)."""
    rows = []
    win_bounds = {}
    for g in iter_window_stats(stats_path):
        if g.empty:
            continue
        wid = int(g["window_id"].iloc[0])
        ws = g["win_start"].iloc[0]
        win_bounds[wid] = (ws, g["win_end"].iloc[0] if "win_end" in g.columns else ws)
        regime = g["regime"].iloc[0] if "regime" in g.columns else "calm"
        for kind in GRAPH_KINDS:
            G = build_graph(g, kind, edge_quantile=edge_quantile, sector_map=sector_map)
            rows.append({"window_id": wid, "win_start": ws, "regime": regime,
                         "graph": kind, "threshold": f"top-{edge_quantile:.0%}",
                         **graph_metrics(G)})
        # descriptive "amount of coupling": absolute s_odd >= abs_threshold
        Gabs = build_graph(g, "s_odd", abs_threshold=abs_threshold, sector_map=sector_map)
        rows.append({"window_id": wid, "win_start": ws, "regime": regime,
                     "graph": "s_odd_abs", "threshold": f">={abs_threshold:g}",
                     **graph_metrics(Gabs)})
        log.info(f"  window {wid} ({regime}): metrics for "
                 f"{', '.join(GRAPH_KINDS)} + s_odd_abs")
    df = pd.DataFrame(rows)
    if len(df) and taxonomy is not None and len(taxonomy):
        df = _attach_taxonomy(df, win_bounds, taxonomy)
    return df


def _attach_taxonomy(metrics: pd.DataFrame, win_bounds: dict,
                     taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Add `crisis_names` and `crisis_types` (semicolon-joined) per window from the
    named-crisis taxonomy overlay. 'none' when a window overlaps no named episode."""
    type_by_name = dict(zip(taxonomy["name"], taxonomy["type"]))
    names = tag_windows_with_crises(list(win_bounds), win_bounds, taxonomy=taxonomy)
    name_map, type_map = {}, {}
    for wid, hits in names.items():
        name_map[wid] = ";".join(hits) if hits else "none"
        types = sorted({t for n in hits for t in str(type_by_name.get(n, "")).split("/")})
        type_map[wid] = ";".join(types) if types else "none"
    metrics = metrics.copy()
    metrics["crisis_names"] = metrics["window_id"].map(name_map)
    metrics["crisis_types"] = metrics["window_id"].map(type_map)
    return metrics


# ==========================================================================
# MR-QAP  (the "does s_odd earn its place" gate)
# ==========================================================================
def pair_matrix(df: pd.DataFrame, kind: str, nodes: list) -> np.ndarray:
    """Symmetric (n x n) weight matrix over `nodes` for one graph kind; absent
    pairs and the diagonal are NaN (structural)."""
    idx = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)
    M = np.full((n, n), np.nan)
    wcol = _WEIGHT_COL[kind]
    for a, b, w in zip(df["permno_a"].astype(int), df["permno_b"].astype(int),
                       df[wcol].astype(float)):
        if a in idx and b in idx and np.isfinite(w):
            M[idx[a], idx[b]] = M[idx[b], idx[a]] = w
    return M


def _upper(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def qap_correlation(M1: np.ndarray, M2: np.ndarray, n_perm: int = 199,
                    seed: int = 0) -> dict:
    """QAP (Mantel-style) correlation between two weight matrices: Pearson r over
    off-diagonal entries finite in both, with a node-permutation p-value (relabel
    M2's nodes, recompute r)."""
    rng = np.random.default_rng(seed)
    n = M1.shape[0]
    base1 = _upper(M1)
    m2u = _upper(M2)
    mask = np.isfinite(base1) & np.isfinite(m2u)
    if mask.sum() < 3:
        return {"r": np.nan, "p": np.nan, "n_pairs": int(mask.sum())}
    r_obs = float(np.corrcoef(base1[mask], m2u[mask])[0, 1])
    ge = 1                                                  # +1 includes observed
    for _ in range(n_perm):
        perm = rng.permutation(n)
        M2p = M2[np.ix_(perm, perm)]
        m2p = _upper(M2p)
        mk = np.isfinite(base1) & np.isfinite(m2p)
        if mk.sum() < 3:
            continue
        rp = float(np.corrcoef(base1[mk], m2p[mk])[0, 1])
        if abs(rp) >= abs(r_obs):
            ge += 1
    return {"r": r_obs, "p": ge / (n_perm + 1), "n_pairs": int(mask.sum())}


def mrqap(Y: np.ndarray, Xs: list, n_perm: int = 199, seed: int = 0) -> dict:
    """Baseline MR-QAP regression of edge weights: OLS of Y on [1, X1, X2, ...]
    over off-diagonal entries finite in Y and every X, with Y-permutation
    p-values (relabel Y's nodes). Returns coefficients, R^2 and p-values.

    This is the Y-permutation QAP regression; Dekker's double semi-partialling
    (DSP) is the stricter alternative -- left as a clearly-marked hook for when
    collinearity among predictors becomes a concern.
    """
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    yu = _upper(Y)
    xus = [_upper(X) for X in Xs]
    mask = np.isfinite(yu)
    for xu in xus:
        mask &= np.isfinite(xu)
    if mask.sum() < (len(Xs) + 2):
        return {"coef": [np.nan] * (len(Xs) + 1), "r2": np.nan,
                "p": [np.nan] * (len(Xs) + 1), "n_pairs": int(mask.sum())}

    def fit(yvec):
        A = np.column_stack([np.ones(mask.sum())] + [xu[mask] for xu in xus])
        beta, *_ = np.linalg.lstsq(A, yvec, rcond=None)
        resid = yvec - A @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((yvec - yvec.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return beta, r2

    beta_obs, r2_obs = fit(yu[mask])
    ge = np.ones(len(beta_obs))
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        yp = _upper(Y[np.ix_(perm, perm)])
        mk = np.isfinite(yp)
        for xu in xus:
            mk &= np.isfinite(xu)
        if mk.sum() < (len(Xs) + 2):
            continue
        A = np.column_stack([np.ones(mk.sum())] + [xu[mk] for xu in xus])
        bp, *_ = np.linalg.lstsq(A, yp[mk], rcond=None)
        ge += (np.abs(bp) >= np.abs(beta_obs)).astype(float)
        cnt += 1
    p = (ge / (cnt + 1)).tolist()
    return {"coef": beta_obs.tolist(), "r2": r2_obs, "p": p,
            "n_pairs": int(mask.sum())}


def qap_hierarchy(stats_path: str, n_perm: int = 199, max_windows: Optional[int] = 8,
                  min_nodes: int = 20, seed: int = 0,
                  node_subsample: Optional[int] = None) -> pd.DataFrame:
    """Run the three-tier comparison per window (subsampled for cost) and return a
    long table: QAP correlations pooled~e00, pooled~s_odd, e00~s_odd, plus the
    MR-QAP gate R^2 of s_odd ~ pooled + e00 (how much of s_odd structure is already
    explained by correlation + tail coupling).

    node_subsample : if set, restrict each window to its first-N sorted nodes (and
                     the pairs among them). Used to align the REAL gate with the
                     classical-null gate frame emitted on the same window x node
                     support (see cbd_analysis.emit_null_gate_stats)."""
    rows = []
    used = 0
    for g in iter_window_stats(stats_path):
        if g.empty:
            continue
        wid = int(g["window_id"].iloc[0])
        regime = g["regime"].iloc[0] if "regime" in g.columns else "calm"
        nodes = sorted(pd.unique(g[["permno_a", "permno_b"]].to_numpy().ravel()).tolist())
        if node_subsample and len(nodes) > node_subsample:
            nodes = nodes[:node_subsample]
            keep = set(nodes)
            g = g[g["permno_a"].isin(keep) & g["permno_b"].isin(keep)]
        if len(nodes) < min_nodes:
            continue
        Mp = pair_matrix(g, "pooled", nodes)
        Me = pair_matrix(g, "e00", nodes)
        Ms = pair_matrix(g, "s_odd", nodes)
        qpe = qap_correlation(Mp, Me, n_perm=n_perm, seed=seed)
        qps = qap_correlation(Mp, Ms, n_perm=n_perm, seed=seed)
        qes = qap_correlation(Me, Ms, n_perm=n_perm, seed=seed)
        gate = mrqap(Ms, [Mp, Me], n_perm=n_perm, seed=seed)   # s_odd ~ pooled+e00
        rows.append({
            "window_id": wid, "regime": regime, "n_nodes": len(nodes),
            "r_pooled_e00": qpe["r"], "p_pooled_e00": qpe["p"],
            "r_pooled_s_odd": qps["r"], "p_pooled_s_odd": qps["p"],
            "r_e00_s_odd": qes["r"], "p_e00_s_odd": qes["p"],
            "gate_r2_s_odd_on_pooled_e00": gate["r2"],
            "gate_coef_e00": gate["coef"][2] if len(gate["coef"]) > 2 else np.nan,
            "gate_p_e00": gate["p"][2] if len(gate["p"]) > 2 else np.nan,
        })
        log.info(f"  QAP window {wid} ({regime}): r(e00,s_odd)={qes['r']:.3f} "
                 f"p={qes['p']:.3f}; gate R^2(s_odd|pooled,e00)={gate['r2']:.3f}")
        used += 1
        if max_windows and used >= max_windows:
            break
    return pd.DataFrame(rows)


def gate_null_relative_summary(real_qap: pd.DataFrame,
                               null_qap: Optional[pd.DataFrame] = None,
                               reliability: Optional[pd.DataFrame] = None) -> dict:
    """Collapse the per-window gate into a NULL- and NOISE-RELATIVE headline.

    The gate R^2 is how much of s_odd is explained by pooled + E00 (tail coupling).
    Read on its own, (1 - R^2) overstates novelty because part of the residual is
    just sampling noise. So we compare real R^2 to the classical-null R^2 (no
    contextual ingredient) and bound it by the s_odd reliability ceiling:
      * real R^2 ~= null R^2  => s_odd's relation to tail coupling is the same as
        under the null => the CHSH combination is decoration (tail-coupling paper).
      * real R^2 << null R^2  => real s_odd carries MORE unexplained structure than
        the null (candidate genuine signal), to be judged against the reliability
        ceiling (a network cannot encode structure beyond its edge reliability).
    `excess_residual = null_R2 - real_R2` is the residual structure in real beyond
    the null; values within (1 - reliability^2) are inside the noise floor.
    """
    def _m(df, c):
        return float(df[c].mean()) if (df is not None and len(df) and c in df.columns) else np.nan
    real_r2 = _m(real_qap, "gate_r2_s_odd_on_pooled_e00")
    null_r2 = _m(null_qap, "gate_r2_s_odd_on_pooled_e00")
    rel_sb = _m(reliability, "reliability_sb")
    excess = (null_r2 - real_r2) if (np.isfinite(real_r2) and np.isfinite(null_r2)) else np.nan
    noise_floor = (1 - rel_sb ** 2) if np.isfinite(rel_sb) else np.nan
    summ = {"real_gate_r2": real_r2, "null_gate_r2": null_r2,
            "reliability_sb": rel_sb, "noise_floor_1_minus_rel2": noise_floor,
            "excess_residual": excess,
            "r_e00_s_odd_real": _m(real_qap, "r_e00_s_odd"),
            "r_e00_s_odd_null": _m(null_qap, "r_e00_s_odd")}
    if np.isfinite(real_r2) and np.isfinite(null_r2):
        if abs(real_r2 - null_r2) < 0.05:
            log.info(f"GATE VERDICT: real R^2={real_r2:.3f} ~= null R^2={null_r2:.3f} "
                     f"=> s_odd structure ~ tail-coupling artifact (CHSH is decoration; "
                     f"points to the tail-coupling network paper).")
        elif real_r2 < null_r2 - 0.05:
            log.info(f"GATE VERDICT: real R^2={real_r2:.3f} < null R^2={null_r2:.3f} "
                     f"(excess residual {excess:.3f}; s_odd reliability SB={rel_sb:.3f}, "
                     f"noise floor {noise_floor:.3f}) => real s_odd carries structure "
                     f"beyond tail coupling above the null; judge vs the ceiling.")
        else:
            log.info(f"GATE VERDICT: real R^2={real_r2:.3f} > null R^2={null_r2:.3f} "
                     f"=> s_odd is MORE explained by tail coupling than the null; no "
                     f"evidence of structure beyond E00.")
    else:
        log.info(f"GATE: real R^2={real_r2:.3f} (no null baseline supplied).")
    return summ


# ==========================================================================
# FIGURES
# ==========================================================================
_METRICS = [("edge_density", "edge density"), ("avg_clustering", "avg clustering"),
            ("giant_frac", "giant-component frac"), ("modularity", "modularity")]


def plot_network_metrics_crisis_calm(metrics: pd.DataFrame) -> Figure:
    name = "fig_k_network_metrics_crisis_calm"
    if _missing(metrics, ("graph", "regime"), name):
        return _placeholder("Network metrics", "network_metrics missing")
    dens = metrics[metrics["graph"].isin(GRAPH_KINDS)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (col, lab) in zip(axes.ravel(), _METRICS):
        if col not in dens.columns:
            ax.set_visible(False)
            continue
        means = (dens.groupby(["graph", "regime"])[col].mean()
                 .reindex(GRAPH_KINDS, level=0))
        kinds = GRAPH_KINDS
        x = np.arange(len(kinds))
        wbar = 0.38
        for i, reg in enumerate(("calm", "crisis")):
            vals = [means.get((k, reg), np.nan) for k in kinds]
            ax.bar(x + (i - 0.5) * wbar, vals, wbar, label=reg,
                   color=REGIME_COLORS.get(reg))
        ax.set_xticks(x); ax.set_xticklabels(kinds)
        ax.set_title(lab); ax.set_ylabel(lab)
    axes.ravel()[0].legend(title="regime", fontsize=9)
    fig.suptitle("Density-matched network topology: crisis vs calm",
                 fontweight="bold")
    _caption(fig, "Top-q% density-matched graphs (pooled correlation, E00 tail "
                  "coupling, s_odd); bars = mean over windows.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_sodd_abs_amount(metrics: pd.DataFrame) -> Figure:
    """Descriptive 'amount of coupling': absolute s_odd>=2 graph edge density,
    crisis vs calm -- the network analog of the naive violation rate."""
    name = "fig_l_sodd_abs_amount"
    if _missing(metrics, ("graph", "regime", "edge_density"), name):
        return _placeholder("Absolute s_odd coupling", "network_metrics missing")
    d = metrics[metrics["graph"] == "s_odd_abs"]
    if d.empty:
        return _placeholder("Absolute s_odd coupling", "no s_odd_abs rows")
    fig, ax = plt.subplots(figsize=(6, 5))
    regs = ["calm", "crisis"]
    means = [d[d["regime"] == r]["edge_density"].mean() for r in regs]
    ax.bar(regs, [100 * m for m in means],
           color=[REGIME_COLORS.get(r) for r in regs], width=0.6)
    for i, m in enumerate(means):
        if np.isfinite(m):
            ax.text(i, 100 * m, f"{100*m:.2f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("edge density of the s_odd \u2265 2 graph (%)")
    ax.set_title("Amount of CHSH coupling (absolute s_odd \u2265 2)")
    _caption(fig, "Absolute (NOT density-matched) graph: the network analog of the "
                  "naive s_odd>2 violation rate; mean over windows.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_network_metrics_over_time(metrics: pd.DataFrame, metric: str = "giant_frac"
                                   ) -> Figure:
    name = "fig_m_network_metric_over_time"
    if _missing(metrics, ("graph", "win_start", metric), name):
        return _placeholder("Network metric over time", "network_metrics missing")
    dens = metrics[metrics["graph"].isin(GRAPH_KINDS)].copy()
    dens["win_start"] = pd.to_datetime(dens["win_start"])
    reg_tbl = (dens[["window_id", "win_start", "regime"]].drop_duplicates("window_id")
               .sort_values("win_start"))
    fig, ax = plt.subplots(figsize=(9, 5))
    _shade_crisis(ax, reg_tbl)
    palette = {"pooled": OKABE["grey"], "e00": OKABE["blue"], "s_odd": OKABE["vermillion"]}
    for kind in GRAPH_KINDS:
        dk = dens[dens["graph"] == kind].sort_values("win_start")
        if dk.empty:
            continue
        ax.plot(dk["win_start"], dk[metric], marker="o", ms=3, color=palette[kind],
                label=kind)
    ax.set_ylabel(dict(_METRICS).get(metric, metric))
    ax.set_xlabel("window start")
    ax.set_title(f"{dict(_METRICS).get(metric, metric)} over time (crisis shaded)")
    ax.legend(fontsize=9)
    _caption(fig, "Density-matched graphs; shaded spans = crisis windows (VIX/NBER label).")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_metrics_by_crisis_type(metrics: pd.DataFrame, metric: str = "modularity"
                                ) -> Figure:
    """Step-5 cross-crisis breakdown: a density-matched metric for the E00 and
    s_odd graphs, broken out across crisis-taxonomy types (financial/food/energy/
    mixed/none). One window can contribute to several types (it is exploded)."""
    name = "fig_o_metrics_by_crisis_type"
    if _missing(metrics, ("graph", "crisis_types", metric), name):
        return _placeholder("Metrics by crisis type",
                            "network_metrics lacks crisis_types (taxonomy overlay)")
    d = metrics[metrics["graph"].isin(("e00", "s_odd"))].copy()
    d["crisis_types"] = d["crisis_types"].fillna("none")
    d = d.assign(ctype=d["crisis_types"].str.split(";")).explode("ctype")
    order = [t for t in ("financial", "food", "energy", "commodity", "mixed", "none")
             if t in set(d["ctype"])]
    if not order:
        return _placeholder("Metrics by crisis type", "no taxonomy types present")
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"e00": OKABE["blue"], "s_odd": OKABE["vermillion"]}
    x = np.arange(len(order))
    wbar = 0.38
    for i, kind in enumerate(("e00", "s_odd")):
        vals = [d[(d["ctype"] == t) & (d["graph"] == kind)][metric].mean() for t in order]
        ax.bar(x + (i - 0.5) * wbar, vals, wbar, label=kind, color=palette[kind])
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel(dict(_METRICS).get(metric, metric))
    ax.set_title(f"{dict(_METRICS).get(metric, metric)} by crisis-taxonomy type")
    ax.legend(title="graph", fontsize=9)
    _caption(fig, "Density-matched graphs; windows exploded across overlapping crisis "
                  "types (config/crises.csv overlay, NOT the binary VIX/NBER label).")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_gate_null_relative(summary: dict) -> Figure:
    """Null- and noise-relative gate headline: real vs classical-null R^2 of
    s_odd ~ pooled + E00, with the s_odd reliability ceiling annotated."""
    name = "fig_p_gate_null_relative"
    real_r2 = summary.get("real_gate_r2", np.nan)
    if not np.isfinite(real_r2):
        return _placeholder("Gate (null-relative)", "no real gate R^2")
    null_r2 = summary.get("null_gate_r2", np.nan)
    rel = summary.get("reliability_sb", np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = {"real": real_r2}
    if np.isfinite(null_r2):
        bars["classical-null"] = null_r2
    ax.bar(list(bars), list(bars.values()),
           color=[OKABE["vermillion"], OKABE["grey"]][:len(bars)], width=0.55)
    for i, v in enumerate(bars.values()):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    if np.isfinite(rel):
        ax.axhline(rel, ls="--", color=OKABE["green"],
                   label=f"s_odd reliability ceiling (SB r={rel:.2f})")
        ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$R^2$ of  s_odd ~ pooled + E00")
    ax.set_title("Does s_odd add structure beyond tail coupling?")
    excess = summary.get("excess_residual", np.nan)
    sub = ("real \u2248 null \u21d2 tail-coupling artifact"
           if (np.isfinite(null_r2) and abs(real_r2 - null_r2) < 0.05)
           else f"excess residual (null\u2212real R\u00b2) = {excess:.2f}")
    _caption(fig, f"Null-relative gate: {sub}. Structure beyond E00 must clear both "
                  f"the null and the reliability ceiling.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_qap_hierarchy(qap: pd.DataFrame) -> Figure:
    name = "fig_n_qap_hierarchy"
    if _missing(qap, ("r_pooled_e00", "r_e00_s_odd"), name):
        return _placeholder("MR-QAP hierarchy", "network_qap missing")
    tiers = [("r_pooled_e00", "pooled \u2192 E00"),
             ("r_pooled_s_odd", "pooled \u2192 s_odd"),
             ("r_e00_s_odd", "E00 \u2192 s_odd")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    labels = [lab for _, lab in tiers]
    means = [qap[c].mean() for c, _ in tiers]
    sds = [qap[c].std() for c, _ in tiers]
    ax.bar(labels, means, yerr=sds, capsize=4, color=OKABE["blue"], width=0.6)
    ax.set_ylabel("QAP correlation r (mean \u00b1 sd over windows)")
    ax.set_title("Does each tier add structure?")
    ax.tick_params(axis="x", labelrotation=15)
    if "gate_r2_s_odd_on_pooled_e00" in qap.columns:
        ax2.hist(qap["gate_r2_s_odd_on_pooled_e00"].dropna(), bins=10,
                 color=OKABE["vermillion"], alpha=0.8)
        ax2.set_xlabel(r"$R^2$ of  s_odd ~ pooled + E00")
        ax2.set_ylabel("windows")
        ax2.set_title("How much of s_odd is already explained")
    fig.suptitle("MR-QAP gate: 'does s_odd earn its place?'", fontweight="bold")
    _caption(fig, "High r(E00,s_odd) and high gate-R^2 => s_odd ~= tail coupling "
                  "(CHSH is decoration); low => s_odd adds structure.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


# ==========================================================================
# DRIVER
# ==========================================================================
def build_all(data_dir: str, out_dir: str, stats_file: Optional[str] = None,
              edge_quantile: float = 0.05, abs_threshold: float = 2.0,
              run_mrqap: bool = True, mrqap_nperm: int = 199,
              mrqap_windows: Optional[int] = 8, null_stats_file: Optional[str] = None,
              reliability_file: Optional[str] = None,
              gate_node_subsample: Optional[int] = 60) -> None:
    set_style()
    stats_path = stats_file or os.path.join(data_dir, "pair_window_stats")
    if not (os.path.isdir(stats_path) or os.path.exists(stats_path)
            or os.path.exists(stats_path + ".parquet")):
        log.error(f"{stats_path}: pair_window_stats not found; run cbd_analysis.py first.")
        return

    taxonomy = load_crisis_taxonomy()
    metrics = window_network_metrics(stats_path, edge_quantile=edge_quantile,
                                     abs_threshold=abs_threshold, taxonomy=taxonomy)
    mpath = os.path.join(data_dir, "network_metrics.parquet")
    _save(metrics, mpath)
    log.info(f"wrote {len(metrics):,} network-metric rows -> {mpath}")

    qap = pd.DataFrame()
    summary: dict = {}
    if run_mrqap:
        qap = qap_hierarchy(stats_path, n_perm=mrqap_nperm, max_windows=mrqap_windows,
                            node_subsample=gate_node_subsample)
        qpath = os.path.join(data_dir, "network_qap.parquet")
        _save(qap, qpath)
        log.info(f"wrote {len(qap):,} QAP rows -> {qpath}")
        # null baseline (PRIMARY) + reliability ceiling (SUPPORTING)
        null_path = null_stats_file or os.path.join(data_dir, "classical_null_gate_stats")
        null_qap = pd.DataFrame()
        if os.path.isdir(null_path) or os.path.exists(null_path) \
                or os.path.exists(null_path + ".parquet"):
            null_qap = qap_hierarchy(null_path, n_perm=mrqap_nperm,
                                     max_windows=mrqap_windows,
                                     node_subsample=gate_node_subsample)
        else:
            log.warning("no classical_null_gate_stats found; gate has no null baseline. "
                        "Emit it via `cbd_analysis.py --null-gate-stats`.")
        reliability = _read_optional(reliability_file
                                     or os.path.join(data_dir, "s_odd_reliability"))
        summary = gate_null_relative_summary(qap, null_qap, reliability)
        _save(pd.DataFrame([summary]), os.path.join(data_dir, "network_gate_summary.parquet"))

    figs = {
        "fig_k_network_metrics_crisis_calm": lambda: plot_network_metrics_crisis_calm(metrics),
        "fig_l_sodd_abs_amount": lambda: plot_sodd_abs_amount(metrics),
        "fig_m_network_metric_over_time": lambda: plot_network_metrics_over_time(metrics),
        "fig_n_qap_hierarchy": lambda: plot_qap_hierarchy(qap),
        "fig_o_metrics_by_crisis_type": lambda: plot_metrics_by_crisis_type(metrics),
        "fig_p_gate_null_relative": lambda: plot_gate_null_relative(summary),
    }
    for nm, fn in figs.items():
        try:
            fig = fn(); save_figure(fig, out_dir, nm); plt.close(fig)
        except Exception as e:                             # noqa: BLE001
            log.warning(f"{nm}: failed to render ({e}); skipping.")
    log.info(f"done -> {out_dir}")


def _read_optional(path: str) -> Optional[pd.DataFrame]:
    for ext in (".parquet", ".csv"):
        p = path if path.endswith(ext) else path + ext
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    return None


def _save(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:                                      # noqa: BLE001
        df.to_csv(path.replace(".parquet", ".csv"), index=False)


# ==========================================================================
# SMOKE TEST  (python src/networks.py --test)
# ==========================================================================
def _synthetic_stats(seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic pair_window_stats with a planted block structure so the
    graphs and QAP have something non-degenerate to find."""
    rng = np.random.default_rng(seed)
    rows = []
    starts = pd.bdate_range("2008-01-01", periods=4, freq="30D")
    permnos = list(range(30))
    block = {p: (p // 10) for p in permnos}                # 3 communities
    for wid, ws in enumerate(starts):
        regime = "crisis" if wid in (1, 2) else "calm"
        for i in range(len(permnos)):
            for j in range(i + 1, len(permnos)):
                same = block[permnos[i]] == block[permnos[j]]
                base = 0.6 if same else 0.05
                e00 = float(np.clip(base + rng.normal(0, 0.1), -1, 1))
                pooled_like = float(np.clip(base * 0.8 + rng.normal(0, 0.1), -1, 1))
                s = float(np.clip(2 * abs(e00) + rng.normal(0, 0.2), 0, 4))
                rows.append({"window_id": wid, "win_start": ws,
                             "win_end": ws + pd.Timedelta(days=30),
                             "permno_a": permnos[i], "permno_b": permnos[j],
                             "E00": e00, "E01": pooled_like, "E10": pooled_like,
                             "E11": -0.1,
                             "N00": 15, "N01": 15, "N10": 15, "N11": 15,
                             "s_odd": s, "valid": True, "regime": regime})
    return pd.DataFrame(rows)


def run_tests() -> None:
    set_style()
    import tempfile
    stats = _synthetic_stats()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pair_window_stats.parquet")
        stats.to_parquet(path, index=False)

        # streaming reader yields one frame per window
        wins = list(iter_window_stats(path))
        assert len(wins) == 4, len(wins)
        g0 = wins[0]
        assert {"w_pooled", "w_e00", "w_s_odd"} <= set(g0.columns)

        # graph builders + metrics
        for kind in GRAPH_KINDS:
            G = build_graph(g0, kind, edge_quantile=0.1)
            assert isinstance(G, nx.Graph) and G.number_of_nodes() > 0
            md = graph_metrics(G)
            assert set(("edge_density", "avg_clustering", "giant_frac",
                        "modularity", "n_communities")) <= set(md)
        Gabs = build_graph(g0, "s_odd", abs_threshold=2.0)
        assert isinstance(Gabs, nx.Graph)

        # density-matched graphs have equal edge counts across kinds
        counts = {k: build_graph(g0, k, edge_quantile=0.1).number_of_edges()
                  for k in GRAPH_KINDS}
        assert len(set(counts.values())) == 1, counts

        # metrics table (+ taxonomy overlay tagging)
        metrics = window_network_metrics(path, edge_quantile=0.1,
                                         taxonomy=load_crisis_taxonomy())
        assert set(("graph", "regime", "modularity")) <= set(metrics.columns)
        assert (metrics["graph"] == "s_odd_abs").any()
        assert "crisis_types" in metrics.columns

        # QAP correlation + MR-QAP
        nodes = sorted(pd.unique(g0[["permno_a", "permno_b"]].to_numpy().ravel()).tolist())
        Mp = pair_matrix(g0, "pooled", nodes)
        Me = pair_matrix(g0, "e00", nodes)
        Ms = pair_matrix(g0, "s_odd", nodes)
        q = qap_correlation(Me, Ms, n_perm=49)
        assert set(("r", "p", "n_pairs")) <= set(q) and -1 <= q["r"] <= 1
        mr = mrqap(Ms, [Mp, Me], n_perm=49)
        assert set(("coef", "r2", "p")) <= set(mr) and len(mr["coef"]) == 3

        qdf = qap_hierarchy(path, n_perm=49, max_windows=2, min_nodes=5,
                            node_subsample=20)
        assert {"r_e00_s_odd", "gate_r2_s_odd_on_pooled_e00"} <= set(qdf.columns)

        # null-relative gate summary (null frame ~ shifted real frame here)
        null_like = qdf.assign(gate_r2_s_odd_on_pooled_e00=qdf["gate_r2_s_odd_on_pooled_e00"])
        rel = pd.DataFrame({"window_id": [0, 1], "reliability_sb": [0.7, 0.72]})
        summ = gate_null_relative_summary(qdf, null_like, rel)
        assert {"real_gate_r2", "null_gate_r2", "reliability_sb", "excess_residual"} <= set(summ)

        # every figure returns a Figure
        for fig in (plot_network_metrics_crisis_calm(metrics),
                    plot_sodd_abs_amount(metrics),
                    plot_network_metrics_over_time(metrics),
                    plot_qap_hierarchy(qdf),
                    plot_metrics_by_crisis_type(metrics),
                    plot_gate_null_relative(summ)):
            assert isinstance(fig, Figure)
            plt.close(fig)

        # graceful degradation on missing columns
        empty = pd.DataFrame({"window_id": [0]})
        for fn in (lambda: plot_network_metrics_crisis_calm(empty),
                   lambda: plot_sodd_abs_amount(empty),
                   lambda: plot_qap_hierarchy(empty),
                   lambda: plot_metrics_by_crisis_type(empty),
                   lambda: plot_gate_null_relative({})):
            assert isinstance(fn(), Figure)
    log.info("ALL NETWORK TESTS PASSED")


def parse_args():
    ap = argparse.ArgumentParser(description="Network topology over pair_window_stats.")
    ap.add_argument("--data-dir", default="wrds_sp500_data")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--stats-file", default=None,
                    help="pair_window_stats parquet/csv, or a partitioned directory of shards")
    ap.add_argument("--edge-quantile", type=float, default=0.05,
                    help="density-matched top-q fraction of edges (default 0.05)")
    ap.add_argument("--abs-threshold", type=float, default=2.0,
                    help="absolute s_odd cutoff for the descriptive 'amount' graph")
    ap.add_argument("--no-mrqap", action="store_true", help="skip the MR-QAP hierarchy")
    ap.add_argument("--mrqap-nperm", type=int, default=199)
    ap.add_argument("--mrqap-windows", type=int, default=8,
                    help="cap windows used for MR-QAP (cost control; 0 = all)")
    ap.add_argument("--null-stats-file", default=None,
                    help="classical-null gate stats (from cbd_analysis --null-gate-stats); "
                         "default looks for classical_null_gate_stats in --data-dir")
    ap.add_argument("--reliability-file", default=None,
                    help="s_odd reliability (from cbd_analysis --reliability); "
                         "default looks for s_odd_reliability in --data-dir")
    ap.add_argument("--gate-nodes", type=int, default=60,
                    help="first-N nodes per window for the MR-QAP gate (align with the "
                         "null-gate emission; 0 = all nodes)")
    ap.add_argument("--test", action="store_true", help="run unit tests and exit")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        run_tests()
    else:
        build_all(args.data_dir, args.out, stats_file=args.stats_file,
                  edge_quantile=args.edge_quantile, abs_threshold=args.abs_threshold,
                  run_mrqap=not args.no_mrqap, mrqap_nperm=args.mrqap_nperm,
                  mrqap_windows=(args.mrqap_windows or None),
                  null_stats_file=args.null_stats_file,
                  reliability_file=args.reliability_file,
                  gate_node_subsample=(args.gate_nodes or None))
