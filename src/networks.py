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
# R1 TOPOLOGY RELIABILITY  (edge-set Jaccard + half-graph metric reliability)
# ==========================================================================
RELIABILITY_METRICS = ("modularity", "giant_frac", "avg_clustering")


def _edge_set(G: nx.Graph) -> set:
    """Undirected edge set as frozensets (order-independent)."""
    return {frozenset((int(a), int(b))) for a, b in G.edges()}


def topology_reliability(edges_path: str, edge_quantile: float = 0.05,
                         kinds=("e00", "pooled", "s_odd")) -> pd.DataFrame:
    """R1b + R1c. Reads the per-window x half edge frame emitted by
    cbd_analysis.s_odd_split_half_reliability(emit_edges_path=...) and, per window,
    for each graph kind:
      * builds the density-matched (top-`edge_quantile`) graph on EACH day-half over
        the common pair set, and reports the Jaccard overlap of the two top-q edge
        SETS (R1b: edge-set stability), and
      * the modularity / giant-frac / clustering of each half graph (R1c inputs).
    One row per (window, kind). The across-window half-1-vs-half-2 metric correlation
    (the decisive R1c number) is computed by `reliability_summary`.
    """
    def _iter_edge_windows(path):
        """Yield one per-window edge frame, streaming a shard directory or grouping
        a monolithic file (so the full-span frame is never held in memory)."""
        if os.path.isdir(path):
            for sh in sorted(glob.glob(os.path.join(path, "*.parquet"))):
                d = pd.read_parquet(sh)
                for _, g in d.groupby("window_id"):
                    yield g
            return
        d = _read_optional(path)
        if d is None or d.empty:
            return
        for _, g in d.groupby("window_id"):
            yield g

    wcol = {"e00": "w_e00", "pooled": "w_pooled", "s_odd": "w_s_odd"}
    rows = []
    for g in _iter_edge_windows(edges_path):
        wid = int(g["window_id"].iloc[0])
        ga, gb = g[g["half"] == "a"], g[g["half"] == "b"]
        key = ["permno_a", "permno_b"]
        common = ga.merge(gb[key], on=key)               # pairs present in both halves
        if len(common) < 10:
            continue
        a = ga.merge(common[key], on=key)
        b = gb.merge(common[key], on=key)
        for kind in kinds:
            wc = wcol[kind]
            # rename to the build_graph weight column so we can reuse it as-is
            da = a.rename(columns={wc: _WEIGHT_COL[kind]})
            db = b.rename(columns={wc: _WEIGHT_COL[kind]})
            Ga = build_graph(da, kind, edge_quantile=edge_quantile)
            Gb = build_graph(db, kind, edge_quantile=edge_quantile)
            ea, eb = _edge_set(Ga), _edge_set(Gb)
            union = ea | eb
            jacc = len(ea & eb) / len(union) if union else np.nan
            ma, mb = graph_metrics(Ga), graph_metrics(Gb)
            rec = {"window_id": int(wid), "graph": kind, "n_common": len(common),
                   "jaccard_topq": jacc}
            for mt in RELIABILITY_METRICS:
                rec[f"{mt}_a"] = ma[mt]
                rec[f"{mt}_b"] = mb[mt]
            rows.append(rec)
        log.info(f"  topo-reliability window {wid}: "
                 + ", ".join(f"{k} J={r['jaccard_topq']:.2f}"
                             for k, r in [(kk, next(x for x in rows[-3:] if x['graph'] == kk))
                                          for kk in kinds] if r))
    return pd.DataFrame(rows)


def reliability_summary(edge_rel: pd.DataFrame,
                        weight_rel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Collapse the per-window topology reliability into one row per graph kind:
    mean top-q edge Jaccard (R1b) and the across-window half-1-vs-half-2 correlation
    of each metric (R1c). If `weight_rel` (the cbd_analysis edge-weight SB-r table) is
    given, its mean SB r per kind (R1a) is merged in for the single checkpoint table.
    """
    if edge_rel is None or edge_rel.empty:
        return pd.DataFrame()
    sb_col = {"s_odd": "reliability_sb", "e00": "reliability_sb_e00",
              "pooled": "reliability_sb_pooled"}
    out = []
    for kind, g in edge_rel.groupby("graph"):
        rec = {"graph": kind, "n_windows": len(g),
               "mean_jaccard_topq": float(g["jaccard_topq"].mean())}
        for mt in RELIABILITY_METRICS:
            a, b = g[f"{mt}_a"].to_numpy(float), g[f"{mt}_b"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            rec[f"{mt}_halfcorr"] = (float(np.corrcoef(a[ok], b[ok])[0, 1])
                                     if ok.sum() >= 3 and a[ok].std() > 0
                                     and b[ok].std() > 0 else np.nan)
        if weight_rel is not None and not weight_rel.empty and kind in sb_col \
                and sb_col[kind] in weight_rel.columns:
            rec["mean_edge_sb_r"] = float(weight_rel[sb_col[kind]].mean())
        out.append(rec)
    df = pd.DataFrame(out)
    log.info("R1 RELIABILITY CHECKPOINT (per graph kind):")
    for _, r in df.iterrows():
        sb = r.get("mean_edge_sb_r", np.nan)
        log.info(f"  {r['graph']:>6}: edge SB r={sb:.3f} | top-q Jaccard="
                 f"{r['mean_jaccard_topq']:.3f} | half-corr "
                 f"mod={r['modularity_halfcorr']:.3f} giant={r['giant_frac_halfcorr']:.3f} "
                 f"clust={r['avg_clustering_halfcorr']:.3f}")
    return df


def plot_topology_reliability(summary: pd.DataFrame, e00_cleared: bool) -> Figure:
    """R1 checkpoint exhibit: edge SB r, top-q Jaccard, and half-to-half metric
    correlations per graph kind, with the noise-floor guides drawn."""
    name = "fig_q_topology_reliability"
    if _missing(summary, ("graph", "mean_jaccard_topq"), name):
        return _placeholder("Topology reliability", "summary missing")
    order = [k for k in ("pooled", "e00", "s_odd") if k in set(summary["graph"])]
    s = summary.set_index("graph").loc[order]
    series = [("edge SB r", s.get("mean_edge_sb_r")),
              ("top-q Jaccard", s["mean_jaccard_topq"]),
              ("modularity half-corr", s["modularity_halfcorr"]),
              ("giant-frac half-corr", s["giant_frac_halfcorr"])]
    series = [(lab, v) for lab, v in series if v is not None]
    x = np.arange(len(order)); w = 0.8 / len(series)
    fig, ax = plt.subplots()
    palette = [OKABE["blue"], OKABE["orange"], OKABE["green"], OKABE["vermillion"]]
    for k, (lab, v) in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, np.asarray(v, float), w,
               label=lab, color=palette[k % len(palette)])
    ax.axhline(0.3, color=OKABE["black"], ls="--", lw=1.0, label="SB r=0.3 bar")
    ax.axhline(0.2, color=OKABE["black"], ls=":", lw=1.0, label="Jaccard=0.2 bar")
    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_LABELS.get(k, k) for k in order])
    ax.set_ylabel("reliability (r / Jaccard)")
    verdict = "E00 CLEARS the bar" if e00_cleared else "E00 NOISE-DOMINATED (deflation-only)"
    ax.set_title(f"R1 reliability checkpoint -- {verdict}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    _caption(fig, "Regime-preserving odd/even split. Bars below the dashed/dotted "
                  "guides indicate selection-on-noise.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ==========================================================================
# R2  CONFIGURATION-MODEL NULL  (N/era-controlled excess-over-null metrics)
# ==========================================================================
def _config_null_metrics(G: nx.Graph, n_rewire: int, seed: int) -> dict:
    """Observed vs degree-preserving (double-edge-swap) null for one graph.
    Returns {metric: (obs, null_mean, null_sd)} for modularity/giant-frac/clustering.
    Edge density is invariant under degree-preserving rewiring, so it is omitted."""
    obs = graph_metrics(G)
    out = {m: (obs[m], np.nan, np.nan) for m in RELIABILITY_METRICS}
    m_edges = G.number_of_edges()
    if m_edges < 2:
        return out
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in RELIABILITY_METRICS}
    for _ in range(n_rewire):
        H = G.copy()
        try:
            nx.double_edge_swap(H, nswap=m_edges, max_tries=10 * m_edges,
                                seed=int(rng.integers(1_000_000_000)))
        except Exception:                                  # noqa: BLE001
            continue
        md = graph_metrics(H)
        for m in RELIABILITY_METRICS:
            samples[m].append(md[m])
    for m in RELIABILITY_METRICS:
        arr = np.array([x for x in samples[m] if np.isfinite(x)], dtype=float)
        out[m] = (obs[m], float(arr.mean()) if len(arr) else np.nan,
                  float(arr.std(ddof=1)) if len(arr) > 1 else np.nan)
    return out


def config_model_excess(stats_path: str, edge_quantile: float = 0.05,
                        n_rewire: int = 20, seed: int = 0,
                        taxonomy: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """R2. Per window x graph kind, report each topology metric as excess over a
    degree-preserving configuration-model null (obs - null_mean), plus a z-score
    (obs - null_mean)/null_sd. This controls for the node-count / composition shifts
    that confound modularity & giant-frac across eras (food/crisis windows cluster in
    specific eras with different name counts). One row per (window, graph)."""
    rows, win_bounds = [], {}
    for g in iter_window_stats(stats_path):
        if g.empty:
            continue
        wid = int(g["window_id"].iloc[0]); ws = g["win_start"].iloc[0]
        win_bounds[wid] = (ws, g["win_end"].iloc[0] if "win_end" in g.columns else ws)
        regime = g["regime"].iloc[0] if "regime" in g.columns else "calm"
        for kind in GRAPH_KINDS:
            G = build_graph(g, kind, edge_quantile=edge_quantile)
            cm = _config_null_metrics(G, n_rewire, seed + wid)
            rec = {"window_id": wid, "win_start": ws, "regime": regime, "graph": kind}
            for m, (obs, mu, sd) in cm.items():
                rec[m] = obs
                rec[f"{m}_null_mean"] = mu
                rec[f"{m}_excess"] = obs - mu if np.isfinite(mu) else np.nan
                rec[f"{m}_z"] = ((obs - mu) / sd) if (np.isfinite(mu) and sd and sd > 0) else np.nan
            rows.append(rec)
        log.info(f"  config-null window {wid} ({regime}) done")
    df = pd.DataFrame(rows)
    if len(df) and taxonomy is not None and len(taxonomy):
        df = _attach_taxonomy(df, win_bounds, taxonomy)
    return df


# ==========================================================================
# R3  WINDOW-LEVEL INFERENCE  (windows as the unit; pair-level p dropped)
# ==========================================================================
def _perm_diff(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
               seed: int = 0) -> tuple:
    """Two-sided permutation test of the mean difference (a - b). Returns
    (diff, p, n_a, n_b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan, len(a), len(b))
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a); ge = 1
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= abs(obs):
            ge += 1
    return (float(obs), ge / (n_perm + 1), len(a), len(b))


def window_level_inference(excess: pd.DataFrame, kind: str = "e00",
                           metrics=("modularity", "giant_frac", "avg_clustering"),
                           value_suffix: str = "_excess", n_perm: int = 20000,
                           seed: int = 0) -> pd.DataFrame:
    """R3. Crisis-vs-calm and food-vs-financial permutation tests at the WINDOW level
    (windows as the unit, ~50 crisis vs ~379 calm; ~58 food vs ~114 financial), on the
    excess-over-null metrics for `kind` (E00 by default). Pair-level p-values are
    intentionally not produced -- with 5e7 pairs they are meaningless."""
    if excess is None or excess.empty or "graph" not in excess.columns:
        return pd.DataFrame()
    d = excess[excess["graph"] == kind].copy()             # one row per window
    rows = []
    for m in metrics:
        col = m + value_suffix
        if col not in d.columns:
            continue
        cr = d[d["regime"] == "crisis"][col].to_numpy(float)
        ca = d[d["regime"] == "calm"][col].to_numpy(float)
        diff, p, na, nb = _perm_diff(cr, ca, n_perm, seed)
        rows.append({"contrast": "crisis_vs_calm", "metric": m, "graph": kind,
                     "mean_group1": float(np.nanmean(cr)) if len(cr) else np.nan,
                     "mean_group2": float(np.nanmean(ca)) if len(ca) else np.nan,
                     "n_group1": na, "n_group2": nb, "diff": diff, "perm_p": p})
    if "crisis_types" in d.columns:
        dt = d.assign(_t=d["crisis_types"].fillna("none").str.split(";"))
        food = dt[dt["_t"].apply(lambda L: "food" in L)]
        fin = dt[dt["_t"].apply(lambda L: "financial" in L)]
        for m in metrics:
            col = m + value_suffix
            if col not in d.columns:
                continue
            diff, p, na, nb = _perm_diff(food[col].to_numpy(float),
                                         fin[col].to_numpy(float), n_perm, seed)
            rows.append({"contrast": "food_vs_financial", "metric": m, "graph": kind,
                         "mean_group1": float(np.nanmean(food[col])) if len(food) else np.nan,
                         "mean_group2": float(np.nanmean(fin[col])) if len(fin) else np.nan,
                         "n_group1": na, "n_group2": nb, "diff": diff, "perm_p": p})
    out = pd.DataFrame(rows)
    if len(out):
        log.info("R3 WINDOW-LEVEL INFERENCE (excess-over-null, E00):")
        for _, r in out.iterrows():
            log.info(f"  {r['contrast']:>18} {r['metric']:>14}: "
                     f"diff={r['diff']:+.4f} (n={r['n_group1']} vs {r['n_group2']}) "
                     f"perm_p={r['perm_p']:.4f}")
    return out


def robustness_report(data_dir: str, out_dir: str, edge_quantile: float = 0.05,
                      n_rewire: int = 20, seed: int = 0,
                      stats_file: Optional[str] = None) -> None:
    """Run R2 (config-model excess) + R3 (window-level inference), write artifacts,
    and render the excess-over-null small-multiples figure."""
    set_style()
    stats_path = stats_file or _resolve_stats(data_dir)
    tax = _load_taxonomy_safe()
    excess = config_model_excess(stats_path, edge_quantile=edge_quantile,
                                 n_rewire=n_rewire, seed=seed, taxonomy=tax)
    if excess.empty:
        log.error("robustness_report: no windows; aborting.")
        return
    _save(excess, os.path.join(data_dir, "network_metrics_excess.parquet"))
    wli = window_level_inference(excess, kind="e00")
    _save(wli, os.path.join(data_dir, "window_level_inference.parquet"))
    try:
        fig = plot_all_metrics_over_time(excess, value_suffix="_excess")
        save_figure(fig, out_dir, "fig_u_excess_metrics_over_time"); plt.close(fig)
    except Exception as e:                                  # noqa: BLE001
        log.warning(f"fig_u: failed to render ({e}); skipping.")


def _resolve_stats(data_dir: str) -> str:
    for cand in ("pair_window_stats", "pair_window_stats.parquet"):
        p = os.path.join(data_dir, cand)
        if os.path.isdir(p) or os.path.exists(p):
            return p
    return os.path.join(data_dir, "pair_window_stats")


def _load_taxonomy_safe() -> Optional[pd.DataFrame]:
    try:
        return load_crisis_taxonomy()
    except Exception:                                      # noqa: BLE001
        return None


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
                               reliability: Optional[pd.DataFrame] = None,
                               n_boot: int = 2000, seed: int = 0) -> dict:
    """Collapse the per-window gate into a NULL- and NOISE-RELATIVE headline.

    The gate R^2 is how much of s_odd is explained by pooled + E00 (tail coupling).
    Read on its own, (1 - R^2) overstates novelty because pooled/E00/s_odd share
    estimation noise, so the predictors fit noise too and inflate R^2. The verdict
    therefore rests on the NULL-RELATIVE difference d = real_R2 - null_R2 (the
    classical null has no contextual ingredient; only the difference cancels the
    shared noise). We pair d per-window and bootstrap a CI over windows:
      * CI for d straddling 0 (and small) => s_odd ~ tail-coupling artifact (CHSH is
        decoration; points to the tail-coupling network paper).
      * d significantly < 0 => real s_odd has MORE residual structure than the null
        (candidate signal), to be judged against the reliability ceiling.
    Reliability (regime-preserving split-half, Spearman-Brown) is reported as the
    ceiling on attainable R^2; it is shown on the figure only when the bars do not
    exceed it (otherwise reported qualitatively).
    """
    def _m(df, c):
        return float(df[c].mean()) if (df is not None and len(df) and c in df.columns) else np.nan
    col = "gate_r2_s_odd_on_pooled_e00"
    real_r2 = _m(real_qap, col)
    null_r2 = _m(null_qap, col)
    rel_sb = _m(reliability, "reliability_sb")
    summ = {"real_gate_r2": real_r2, "null_gate_r2": null_r2, "reliability_sb": rel_sb,
            "r_e00_s_odd_real": _m(real_qap, "r_e00_s_odd"),
            "r_e00_s_odd_null": _m(null_qap, "r_e00_s_odd"),
            "diff_real_minus_null": np.nan, "diff_ci_lo": np.nan, "diff_ci_hi": np.nan,
            "n_gate_windows": 0}
    # paired per-window difference + bootstrap CI over windows
    if (real_qap is not None and null_qap is not None and len(real_qap) and len(null_qap)
            and col in real_qap.columns and col in null_qap.columns):
        merged = real_qap[["window_id", col]].merge(
            null_qap[["window_id", col]], on="window_id", suffixes=("_real", "_null"))
        d = (merged[col + "_real"] - merged[col + "_null"]).dropna().to_numpy()
        if len(d):
            rng = np.random.default_rng(seed)
            boot = np.array([rng.choice(d, len(d), replace=True).mean()
                             for _ in range(int(n_boot))])
            summ.update(diff_real_minus_null=float(d.mean()),
                        diff_ci_lo=float(np.percentile(boot, 2.5)),
                        diff_ci_hi=float(np.percentile(boot, 97.5)),
                        n_gate_windows=int(len(d)))
    d_, lo, hi = summ["diff_real_minus_null"], summ["diff_ci_lo"], summ["diff_ci_hi"]
    if np.isfinite(d_):
        straddles = (lo <= 0 <= hi)
        verdict = ("s_odd ~ tail-coupling artifact (CHSH decoration)" if (straddles or d_ >= 0)
                   else "real s_odd has residual structure beyond the null; judge vs ceiling")
        log.info(f"GATE VERDICT: real R^2={real_r2:.3f}, null R^2={null_r2:.3f}, "
                 f"d=real-null={d_:+.3f} [95% CI {lo:+.3f},{hi:+.3f}] over "
                 f"{summ['n_gate_windows']} windows; reliability SB={rel_sb:.3f} "
                 f"=> {verdict}.")
    else:
        log.info(f"GATE: real R^2={real_r2:.3f} (no null baseline supplied).")
    return summ


# ==========================================================================
# FIGURES
# ==========================================================================
_METRICS = [("edge_density", "edge density"), ("avg_clustering", "avg clustering"),
            ("giant_frac", "giant-component frac"), ("modularity", "modularity")]

# H3: lead with pooled (baseline) and E00 (canonical tail coupling); s_odd is the
# CONTROL the MR-QAP gate rules out (no structure beyond E00).
GRAPH_LABELS = {"pooled": "pooled\n(baseline)", "e00": "E00\n(canonical)",
                "s_odd": "s_odd\n(control)"}


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
        ax.set_xticks(x); ax.set_xticklabels([GRAPH_LABELS.get(k, k) for k in kinds],
                                             fontsize=9)
        ax.set_title(lab); ax.set_ylabel(lab)
    axes.ravel()[0].legend(title="regime", fontsize=9)
    fig.suptitle("Density-matched network topology: crisis vs calm",
                 fontweight="bold")
    _caption(fig, "Lead exhibits are pooled (correlation baseline) and E00 (canonical "
                  "tail coupling); s_odd is the CONTROL the MR-QAP gate rules out (no "
                  "structure beyond E00). Top-q% density-matched; bars = mean over windows.")
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
    styles = {"pooled": "--", "e00": "-", "s_odd": ":"}     # E00 solid = canonical
    labels = {"pooled": "pooled (baseline)", "e00": "E00 (canonical)",
              "s_odd": "s_odd (control)"}
    for kind in GRAPH_KINDS:
        dk = dens[dens["graph"] == kind].sort_values("win_start")
        if dk.empty:
            continue
        ax.plot(dk["win_start"], dk[metric], marker="o", ms=3, color=palette[kind],
                ls=styles.get(kind, "-"), label=labels.get(kind, kind))
    ax.set_ylabel(dict(_METRICS).get(metric, metric))
    ax.set_xlabel("window start")
    ax.set_title(f"{dict(_METRICS).get(metric, metric)} over time (crisis shaded)")
    ax.legend(fontsize=9)
    _caption(fig, "E00 (canonical tail coupling) solid; pooled baseline dashed; s_odd "
                  "control dotted. Shaded spans = crisis windows (VIX/NBER label).")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_all_metrics_over_time(metrics: pd.DataFrame, value_suffix: str = "") -> Figure:
    """P1c. Small-multiples of ALL FOUR topology metrics over time, pooled (baseline)
    + E00 (canonical), s_odd as control, crisis-shaded. Generalizes fig_m (giant-frac
    only). With `value_suffix='_excess'` it plots the configuration-model
    excess-over-null versions (R2) instead of the raw metrics."""
    name = "fig_t_all_metrics_over_time"
    base_cols = [m for m, _ in _METRICS]
    cols = [c + value_suffix for c in base_cols]
    if _missing(metrics, ("graph", "win_start"), name) or \
            not any(c in metrics.columns for c in cols):
        return _placeholder("All metrics over time", "network_metrics missing")
    dens = metrics[metrics["graph"].isin(GRAPH_KINDS)].copy()
    dens["win_start"] = pd.to_datetime(dens["win_start"])
    reg_tbl = (dens[["window_id", "win_start", "regime"]].drop_duplicates("window_id")
               .sort_values("win_start"))
    palette = {"pooled": OKABE["grey"], "e00": OKABE["blue"], "s_odd": OKABE["vermillion"]}
    styles = {"pooled": "--", "e00": "-", "s_odd": ":"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (mcol, mlab) in zip(axes.ravel(), _METRICS):
        col = mcol + value_suffix
        if col not in dens.columns:
            ax.set_visible(False); continue
        _shade_crisis(ax, reg_tbl)
        for kind in GRAPH_KINDS:
            dk = dens[dens["graph"] == kind].sort_values("win_start")
            if dk.empty or col not in dk:
                continue
            ax.plot(dk["win_start"], dk[col], color=palette[kind],
                    ls=styles.get(kind, "-"), lw=1.1, label=GRAPH_LABELS.get(kind, kind))
        if value_suffix == "_excess":
            ax.axhline(0, color=OKABE["black"], lw=0.8)
        ax.set_title((mlab + (" (excess over config-null)" if value_suffix else "")))
        ax.set_ylabel(mlab)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("window start")
    fig.autofmt_xdate()
    kind_note = ("configuration-model excess-over-null" if value_suffix
                 else "raw metrics")
    fig.suptitle(f"Network topology over time ({kind_note})", fontweight="bold")
    _caption(fig, "E00 (canonical) solid, pooled (baseline) dashed, s_odd (control) "
                  "dotted; shaded = crisis windows (VIX/NBER).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
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
    """Null- and noise-relative gate headline for the s_odd CONTROL: real vs
    classical-null R^2 of s_odd ~ pooled + E00, with the bootstrapped (real-null)
    R^2 difference and CI. The reliability ceiling is drawn only when the bars do
    not exceed it (a ceiling the bars clear would be misleading)."""
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
    bar_max = max(bars.values())
    # draw the reliability ceiling ONLY if the bars do not exceed it
    if np.isfinite(rel) and rel >= bar_max:
        ax.axhline(rel, ls="--", color=OKABE["green"],
                   label=f"s_odd reliability ceiling (SB r={rel:.2f})")
        ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$R^2$ of  s_odd ~ pooled + E00")
    ax.set_title("s_odd control: no structure beyond E00 tail coupling")
    d = summary.get("diff_real_minus_null", np.nan)
    lo, hi = summary.get("diff_ci_lo", np.nan), summary.get("diff_ci_hi", np.nan)
    if np.isfinite(d):
        rel_txt = (f"reliability SB r={rel:.2f}" if np.isfinite(rel) else "")
        rel_note = ("" if (np.isfinite(rel) and rel >= bar_max)
                    else f" (R\u00b2 exceeds {rel_txt}; ceiling shown qualitatively only)")
        if np.isfinite(lo) and lo <= 0 <= hi:
            verdict = "CI straddles 0 \u21d2 s_odd \u2248 tail coupling"
        elif d >= 0:
            verdict = "real \u2265 null \u21d2 s_odd even more tied to tail coupling (decoration)"
        else:
            verdict = "real < null \u21d2 residual structure beyond E00"
        sub = (f"d = real\u2212null R\u00b2 = {d:+.2f} [95% CI {lo:+.2f}, {hi:+.2f}]; "
               + verdict + rel_note)
    else:
        sub = "no null baseline supplied"
    _caption(fig, f"Null-relative gate: {sub}.")
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
        "fig_t_all_metrics_over_time": lambda: plot_all_metrics_over_time(metrics),
    }
    excess = _read_optional(os.path.join(data_dir, "network_metrics_excess"))
    if excess is not None and not excess.empty:
        figs["fig_u_excess_metrics_over_time"] = \
            lambda: plot_all_metrics_over_time(excess, value_suffix="_excess")
    for nm, fn in figs.items():
        try:
            fig = fn(); save_figure(fig, out_dir, nm); plt.close(fig)
        except Exception as e:                             # noqa: BLE001
            log.warning(f"{nm}: failed to render ({e}); skipping.")
    log.info(f"done -> {out_dir}")


def topo_reliability_report(data_dir: str, out_dir: str, edge_quantile: float = 0.05,
                            edges_file: Optional[str] = None) -> pd.DataFrame:
    """R1 checkpoint driver: read the split-half edge frame + edge-weight SB-r table,
    compute the topology reliability (Jaccard + half-graph metric correlation), write
    artifacts, render fig (q), and return the per-kind summary."""
    set_style()
    edges = edges_file or os.path.join(data_dir, "split_half_edge_weights")
    edge_rel = topology_reliability(edges, edge_quantile=edge_quantile)
    if edge_rel.empty:
        log.error("topo_reliability_report: no reliability rows; run "
                  "`cbd_analysis.py --reliability` first to emit split_half_edge_weights.")
        return pd.DataFrame()
    _save(edge_rel, os.path.join(data_dir, "topology_reliability.parquet"))
    weight_rel = _read_optional(os.path.join(data_dir, "s_odd_reliability"))
    summary = reliability_summary(edge_rel, weight_rel)
    _save(summary, os.path.join(data_dir, "reliability_checkpoint.parquet"))
    e00 = summary[summary["graph"] == "e00"]
    cleared = bool(len(e00) and (
        (e00["mean_edge_sb_r"].iloc[0] if "mean_edge_sb_r" in e00 else 0) >= 0.3
        or e00["mean_jaccard_topq"].iloc[0] >= 0.2
        or e00["modularity_halfcorr"].iloc[0] >= 0.3))
    try:
        fig = plot_topology_reliability(summary, cleared)
        save_figure(fig, out_dir, "fig_q_topology_reliability"); plt.close(fig)
    except Exception as e:                                  # noqa: BLE001
        log.warning(f"fig_q: failed to render ({e}); skipping.")
    log.info(f"R1 VERDICT: E00 {'CLEARS' if cleared else 'FAILS'} the reliability bar.")
    return summary


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
        summ = gate_null_relative_summary(qdf, null_like, rel, n_boot=200)
        assert {"real_gate_r2", "null_gate_r2", "reliability_sb",
                "diff_real_minus_null", "diff_ci_lo", "diff_ci_hi"} <= set(summ)

        # R1 topology reliability (synthetic per-window x half edge frame)
        erows = []
        for wid in range(4):
            gw = stats[stats.window_id == wid]
            for half in ("a", "b"):
                noise = np.random.default_rng(wid * 10 + (half == "b")).normal(
                    0, 0.05, len(gw))
                erows.append(pd.DataFrame({
                    "window_id": wid, "permno_a": gw.permno_a.values,
                    "permno_b": gw.permno_b.values, "half": half,
                    "w_s_odd": gw.s_odd.values + noise,
                    "w_e00": gw.E00.values + noise,
                    "w_pooled": gw.E00.values * 0.8 + noise}))
        epath = os.path.join(tmp, "split_half_edge_weights.parquet")
        pd.concat(erows, ignore_index=True).to_parquet(epath, index=False)
        erel = topology_reliability(epath, edge_quantile=0.1)
        assert {"graph", "jaccard_topq", "modularity_a", "modularity_b"} <= set(erel.columns)
        rsumm = reliability_summary(erel, rel.assign(reliability_sb_e00=0.6,
                                                     reliability_sb_pooled=0.6))
        assert {"graph", "mean_jaccard_topq", "modularity_halfcorr"} <= set(rsumm.columns)

        # R2 configuration-model excess + R3 window-level inference
        excess = config_model_excess(path, edge_quantile=0.1, n_rewire=5,
                                     taxonomy=load_crisis_taxonomy())
        assert {"graph", "modularity_excess", "modularity_z"} <= set(excess.columns)
        wli = window_level_inference(excess, kind="e00", n_perm=200)
        assert {"contrast", "metric", "perm_p", "n_group1"} <= set(wli.columns)
        assert (wli["contrast"] == "crisis_vs_calm").any()

        # every figure returns a Figure
        for fig in (plot_network_metrics_crisis_calm(metrics),
                    plot_sodd_abs_amount(metrics),
                    plot_network_metrics_over_time(metrics),
                    plot_all_metrics_over_time(metrics),
                    plot_all_metrics_over_time(excess, value_suffix="_excess"),
                    plot_qap_hierarchy(qdf),
                    plot_metrics_by_crisis_type(metrics),
                    plot_gate_null_relative(summ),
                    plot_topology_reliability(rsumm, True)):
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
    ap.add_argument("--topo-reliability", action="store_true",
                    help="R1 checkpoint: read split_half_edge_weights + s_odd_reliability, "
                         "report edge-set Jaccard + half-graph metric reliability and exit")
    ap.add_argument("--robustness", action="store_true",
                    help="R2+R3: configuration-model excess-over-null metrics + window-level "
                         "permutation inference (crisis/calm, food/financial) and exit")
    ap.add_argument("--n-rewire", type=int, default=20,
                    help="degree-preserving rewired replicates per graph for --robustness")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (config-model null)")
    ap.add_argument("--test", action="store_true", help="run unit tests and exit")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        run_tests()
    elif args.topo_reliability:
        topo_reliability_report(args.data_dir, args.out, edge_quantile=args.edge_quantile)
    elif args.robustness:
        robustness_report(args.data_dir, args.out, edge_quantile=args.edge_quantile,
                          n_rewire=args.n_rewire, seed=args.seed,
                          stats_file=args.stats_file)
    else:
        build_all(args.data_dir, args.out, stats_file=args.stats_file,
                  edge_quantile=args.edge_quantile, abs_threshold=args.abs_threshold,
                  run_mrqap=not args.no_mrqap, mrqap_nperm=args.mrqap_nperm,
                  mrqap_windows=(args.mrqap_windows or None),
                  null_stats_file=args.null_stats_file,
                  reliability_file=args.reliability_file,
                  gate_node_subsample=(args.gate_nodes or None))
