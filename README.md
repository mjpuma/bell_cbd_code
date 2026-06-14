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
│   ├── cbd_analysis.py         # Stage 2: compute CbD statistics per pair/window
│   └── plots.py                # Stage 3: figures from the parquet (offline)
├── wrds_sp500_data/            # pipeline parquet outputs land here (gitignored)
├── figures/                    # generated figures (gitignored)
└── docs/
    ├── cbd_sp500_analysis.pdf  # the spec (read first)
    ├── cbd_sp500_analysis.tex
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
```
`--crisis-source` auto-detects file type: a `vix` column → window labeled crisis
when its mean VIX exceeds the sample-median aggregate (override with
`--crisis-threshold`); `start`/`end` columns (or the literal `nber`) → crisis
when the window overlaps a recession span.

## Stage 3 — figures (offline, reads parquet only)
```
python src/plots.py --test                          # smoke test (no real data)
python src/plots.py --data-dir wrds_sp500_data --out figures/
```
Writes each figure as PNG (300 dpi) + PDF (vector). Figures degrade gracefully
if an input column is missing (logged warning, no crash). The empirical-vs-null
overlay (fig i) appears once `classical_null_stats.parquet` exists; the
sector-stratified exhibit appears if `identifiers.parquet` carries a `sector`
column.

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

## Note
The math is locked behind unit tests; keep `python src/cbd_analysis.py --test`
and `python src/plots.py --test` green after every change.
