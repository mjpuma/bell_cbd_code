# Bell / CbD — S&P 500 crisis analysis (code)

Two-stage pipeline for the Contextuality-by-Default (CbD) analysis of S&P 500
crisis periods. `docs/cbd_sp500_analysis.(pdf|tex)` is the source of truth for
all definitions and notation.

## Layout
```
bell_cbd_code/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── extract_wrds_sp500.py   # Stage 1: pull data from WRDS (CRSP CIZ/v2)
│   ├── cbd_analysis.py         # Stage 2: CbD statistics per pair/window + theta sweep
│   ├── plots.py                # Stage 3: figures from the parquet (offline)
│   └── networks.py             # Stage 3: network topology + MR-QAP (offline)
├── config/
│   └── crises.csv              # named crisis taxonomy (overlay, NOT the binary label)
├── wrds_sp500_data/            # pipeline parquet outputs land here (gitignored)
├── figures/                    # generated figures (gitignored)
└── docs/
    ├── cbd_sp500_analysis.pdf  # the spec (read first)
    ├── cbd_sp500_analysis.tex
    ├── crises.md               # crisis taxonomy notes + VIX/NBER reconciliation
    └── research_program_plan.md
```

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Stage 1 — extract (one-time, network-bound)
First run prompts for your WRDS username/password and offers to create
`~/.pgpass`; nothing is stored in the code. Leave `SAMPLE_TEST = True` in the
script for a short laptop smoke test, then set it to `False` for the full pull.
```
python src/extract_wrds_sp500.py
```
Writes to `data/`: membership, daily_returns, trading_calendar,
window_eligibility (parquet). See the HPC / Empire AI notes at the bottom of the
script (run extraction on a login node; compute nodes usually have no egress).

## Stage 2 — analyze (offline, iterative)
```
python src/cbd_analysis.py --test           # run unit tests (math must stay green)
python src/cbd_analysis.py --sample 3        # first 3 windows, quick check
python src/cbd_analysis.py                   # full run -> wrds_sp500_data/pair_window_stats.parquet

# crisis/calm labels from an EXTERNAL indicator (never from the pairs' returns):
python src/cbd_analysis.py --crisis-source nber                 # built-in NBER recessions
python src/cbd_analysis.py --crisis-source path/to/vix.parquet  # columns: date, vix
python src/cbd_analysis.py --crisis-source path/to/recessions.csv  # columns: start, end

# the deflation null (classical generator -> same pipeline -> compare distributions):
python src/cbd_analysis.py --null --null-max-pairs 5000

# magnitude-threshold robustness sweep (theta = per-stock |R| quantile):
python src/cbd_analysis.py --sweep --crisis-source nber   # bare --sweep = default set
#   default {0.25,0.40,0.42,0.45,0.48,0.50,0.75,0.90,0.95} — a fine in-band grid
#   (0.40–0.50) plus boundary points; or pass --sweep "0.25,0.5,0.9".
#   -> wrds_sp500_data/threshold_sweep.parquet (strict N_min headline; relaxed-N_min
#      DIAGNOSTIC at high quantiles, reported NULL-RELATIVE: classical-null ctx>0 and
#      a cell-size-matched finite-N ctx>0 floor next to the empirical relaxed rate).

# s_odd split-half reliability (the MR-QAP gate's reliability ceiling):
python src/cbd_analysis.py --reliability      # -> s_odd_reliability.parquet

# classical-null per-window-dense stats for the networks MR-QAP null baseline:
python src/cbd_analysis.py --null-gate-stats --gate-windows 6 --gate-nodes 60
#   -> classical_null_gate_stats.parquet (same schema, source='classical_null').

# FULL 1990-2025 run (memory-safe; the monolithic frame is ~5e7 rows / ~13GB):
python src/cbd_analysis.py --partition          # streams one shard per window
#   -> wrds_sp500_data/pair_window_stats/shard_NNNNN.parquet  (partition-by-window)
#   plus compact aggregates beside it (the per-pair frame is NEVER held in memory):
#     headline_rates.parquet      per-regime naive(s_odd>2) / CbD(ctx>0) running tallies
#     stat_hist.parquet           per-(var,regime) histograms of s_odd/delta/ctx
#     cell_summary.parquet        mean N00..N11 per regime (the N_min health check)
#     scatter_subsample.parquet   bounded s_odd-vs-delta subsample for fig (h)
#     classical_null_hist.parquet streamed deflation-null ctx histogram for fig (i)
```
`--crisis-source` auto-detects file type: a `vix` column → window labeled crisis
when its mean VIX exceeds the sample-median aggregate (override with
`--crisis-threshold`); `start`/`end` columns (or the literal `nber`) → crisis
when the window overlaps a recession span.

`--theta-quantile` (default `0.5`, the spec's median) sets the per-stock,
within-window |R| quantile that splits large- vs small-move regimes. The sweep has
**two roles**: the sub-median points `{0.25, 0.40, 0.50}` keep the both-large (E00)
cell populated and test the **robustness of the deflation**; `{0.75, 0.90, 0.95}`
document the **well-posedness boundary**. The boundary is **two-sided**: at low
theta the both-*small* cell (N11) collapses and at high theta the both-*large* cell
(N00) collapses, so under strict N_min=10 only `θ ≈ 0.40–0.50` is well-posed (the
valid denominator collapses to 0 outside that band over a ~63-day window). Every
"relaxed-N_min, noise not contextuality" claim is reported **null-relative**: next
to the empirical relaxed rate the sweep logs the classical-null ctx>0 and a
cell-size-matched finite-N floor; if the empirical rate exceeds both, it is flagged
loudly (it is then a boundary artifact to characterize, not a clean noise claim).

## Stage 3 — figures (offline, reads parquet only)
```
python src/plots.py --test                          # smoke test (no real data)
python src/plots.py --data-dir wrds_sp500_data --out figures/
```
Writes each figure as PNG (300 dpi) + PDF (vector). Figures degrade gracefully
if an input column is missing (logged warning, no crash). The empirical-vs-null
overlay (fig i) appears once `classical_null_stats.parquet` exists; the
threshold-sweep figure (fig j) once `threshold_sweep.parquet` exists; the
sector-stratified exhibit appears if `identifiers.parquet` carries a `sector`
column. For the full-span `--partition` run, `plots.py` auto-detects the
streaming aggregates (`headline_rates.parquet` etc.) and renders figs (d,f,g,h,i)
from them, so it likewise never materializes the full pair-window frame.

## Stage 3 — network topology (offline, reads pair_window_stats only)
```
python src/networks.py --test                          # smoke test (no real data)
python src/networks.py --data-dir wrds_sp500_data --out figures/
```
Per window it builds three density-matched (top-5% by default, `--edge-quantile`)
weighted graphs over the eligible names — `pooled` (correlation baseline), `e00`
(both-large-move tail coupling, the canonical graph), `s_odd` (the CHSH
combination) — plus a descriptive absolute `s_odd ≥ 2` graph (`--abs-threshold`,
the network analog of the naïve violation rate). It reports edge density,
clustering, giant-component fraction and modularity (crisis vs calm and over
time), and runs the three-tier **MR-QAP** gate `pooled → e00 → s_odd` to test
whether `s_odd` adds structure beyond tail coupling (writes `network_metrics.parquet`
and `network_qap.parquet`). The gate is reported **null- and noise-relative**: if a
`classical_null_gate_stats` frame (from `cbd_analysis.py --null-gate-stats`) and an
`s_odd_reliability` frame (`--reliability`, a **regime-preserving** split-half
estimate) are present, it also runs the same gate on the classical null and reports
the bootstrapped paired difference `d = real R² − null R²` with a 95% CI over
windows (writes `network_gate_summary.parquet`, fig p). The verdict rests on `d`
(the shared estimation noise cancels in the difference), not on `1 − R²`; the
reliability ceiling is drawn only when the bars don't exceed it. s_odd leads as the
**control** the gate rules out. The crisis-taxonomy overlay
is broken out as `network_metrics.crisis_types` and fig o (metrics by crisis type).
`--gate-nodes` (default 60) aligns the real gate with the null-gate node support.
The reader streams window-by-window, so pointing `--stats-file` at a partitioned
directory of per-window shards scales to the full panel. The `sector_map` argument
is a carried hook for a later agricultural node-filter; no filtering is done here.

## Robustness pass (R1–R4) and descriptive figures (P1)
These hardening steps gate the network findings and add recognizable context.

- **R1 — reliability checkpoint** (`cbd_analysis.py --reliability` then
  `networks.py --topo-reliability`). The regime-preserving odd/even split-half now
  reports Spearman-Brown reliability for **s_odd, E00, and pooled** edge weights and
  emits per-window edge shards into `split_half_edge_weights/`. `--topo-reliability`
  consumes them for (a) the top-5% **edge-set Jaccard** overlap (is the *identity* of
  the strongest edges reproducible?) and (b) **half-graph metric correlations** across
  windows (are the *aggregate* modularity/giant-frac/clustering reproducible?). Writes
  `topology_reliability.parquet`, `reliability_checkpoint.parquet`, fig q. Full-span
  verdict: E00 edge SB r = 0.34 (marginal), Jaccard ≈ 0.05 (**fails** — no edge-level
  claims), but aggregate half-corr is high (mod 0.79, giant 0.93, clust 0.90) → E00
  **aggregate-metric** claims are supported; specific-edge claims are not.
- **R2 — configuration-model null** (`networks.py --robustness`, `--n-rewire`). Each
  topology metric is reported as **excess over a degree-preserving (double-edge-swap)
  null** plus a z-score in null-SD units, controlling for the node-count/composition
  shifts that confound modularity & giant-frac across eras. Writes
  `network_metrics_excess.parquet` and fig u (excess over time).
- **R3 — window-level inference** (runs inside `--robustness`). Crisis-vs-calm and
  food-vs-financial permutation tests at the **window level** (windows as the unit) on
  the excess metrics; pair-level p-values are intentionally dropped (with ~5·10⁷ pairs
  they are meaningless). Writes `window_level_inference.parquet`. Full-span E00 result:
  crisis windows fragment (giant-frac excess p=0.0015) and de-cluster (clustering excess
  p=0.0002), but the *modularity* crisis effect washes out after the degree-null control
  (p=0.20) — it was largely a composition effect. The **food-vs-financial** modularity
  gap survives the strongest controls (excess modularity p=0.008, clustering p=0.021):
  food windows are structurally distinct beyond generic-crisis effects.
- **R4 — window-length sensitivity** (`rebuild_windows.py --window-days 120`). Re-derives
  `window_eligibility` at 120 td from the on-disk panel (no WRDS re-pull; reuses the
  extractor's `build_windows`/`compute_eligibility`) into a sibling data dir with the
  large files symlinked, then re-runs the pipeline there. Confirms thin cells were the
  reliability bottleneck (E00 edge SB r 0.34 → 0.49) and checks that the giant-component
  crisis-dissolution, the food-vs-financial modularity gap, and the COVID-vs-Lehman
  fragmentation contrast survive the longer window.
- **P1 — descriptive context** (`plots.py`): the real published **S&P 500 index level**
  (CRSP `crsp.dsp500_v2.spindx`, saved once to `sp500_index.parquet`) crisis-shaded with
  an equal-weighted-constituent overlay (fig r), and the cross-sectional **|daily return|
  box-whisker** by month (fig s, `freq` parameter) showing crises are the high-|R|
  periods. The all-four-metrics small-multiples over time is fig t (raw) / fig u (excess).

## Crisis taxonomy (overlay)
`config/crises.csv` lists named crisis sub-periods (1990–2025) with a type label
(financial/food/energy/mixed) for cross-crisis comparison. It is loaded via the
shared util `load_crisis_taxonomy()` / `tag_windows_with_crises()` and is kept
deliberately separate from the binary VIX/NBER `load_crisis_labels` driver. See
`docs/crises.md` for boundaries and the food-vs-financial reconciliation (food
episodes that do NOT register as equity-VIX/NBER stress are themselves a finding).

## Status of cbd_analysis.py
- DONE & tested: core estimators (s_odd, Delta, CTX), regime classification,
  per-pair stats with N_min, parquet I/O, and the per-window driver (now
  vectorized — `run_window` is asserted bit-for-bit against the reference
  `_run_window_loop` in `--test`).
- Implemented (modeling choices; interfaces pinned):
  - `load_crisis_labels` — VIX/NBER labels, external to the pairs.
  - `classical_null_reproduction` — the deflation null (the central result).
- Also added (spec §pipeline Steps 4–5), each unit-tested:
  - `exogenous_context_contrast` (+ `make_vix_regime`) — shared exogenous context.
  - `permutation_placebo`, `small_sample_null`, `max_stat_null`,
    `sector_stratified_rates` — the controls.
  - `sweep_thresholds` — the magnitude-threshold robustness sweep (§Step 2).
  - `load_crisis_taxonomy` / `tag_windows_with_crises` — the named-crisis overlay.

## Note
The math is locked behind unit tests; keep all three suites green after every
change: `python src/cbd_analysis.py --test`, `python src/plots.py --test`, and
`python src/networks.py --test`.
