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
python src/cbd_analysis.py --sweep "0.5,0.75,0.9,0.95" --crisis-source nber
#   -> wrds_sp500_data/threshold_sweep.parquet (strict N_min headline + a
#      relaxed-N_min small-sample DIAGNOSTIC at the high quantiles).
```
`--crisis-source` auto-detects file type: a `vix` column → window labeled crisis
when its mean VIX exceeds the sample-median aggregate (override with
`--crisis-threshold`); `start`/`end` columns (or the literal `nber`) → crisis
when the window overlaps a recession span.

`--theta-quantile` (default `0.5`, the spec's median) sets the per-stock,
within-window |R| quantile that splits large- vs small-move regimes. At high
theta the large-move regime gets rare and the four-cell CHSH structure loses
support (strict N_min=10 → the valid denominator collapses): that is the method's
well-posedness boundary, not a failure of deflation. The relaxed-N_min sweep
points are a small-sample diagnostic only — never a result.

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
column.

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
and `network_qap.parquet`). The reader streams window-by-window, so pointing
`--stats-file` at a partitioned directory of per-window shards scales to the full
panel. The `sector_map` argument is a carried hook for a later agricultural
node-filter; no filtering is done here.

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
