# Figure Manifest — CbD S&P 500 deflation analysis

Verification-grade manifest of every figure in `figures/`. For each figure: identity,
provenance (source parquet + exact statistic), visual content, the underlying numbers read
from the **source data** (not eyeballed), and the manuscript result it maps to.

## Global context (applies to all figures)
- **Data window:** 1990-01-02 … 2025-12-31 (CRSP CIZ v2). `daily_returns`: 7,156,235 rows,
  1,326 distinct permnos. **429 rolling windows** (60 trading days, step 21 td).
- **Pair-windows:** 52,633,010 total (calm `n_total` 46,544,054 + crisis 6,088,956). After
  the N_min ≥ 10/cell rule: **valid** = calm 37,156,946 + crisis 4,621,363 = 41,778,309.
- **Canonical magnitude threshold:** θ = per-stock, within-window **median |R|** (θ-quantile 0.50).
- **Crisis label:** built-in NBER recession spans → **50 crisis windows, 379 calm** (the binary
  driver; separate from the `config/crises.csv` taxonomy overlay used in fig o).
- **Formats:** every stem below has **both** `.png` (300 dpi raster) and `.pdf` (vector) on disk.
- **Render path:** figs a–j, r, s came from `plots.py build_all_streaming` (full-span streaming;
  d/f/g/i use aggregate-fed renderers). figs k–p from `networks.py build_all` (prior canonical run);
  q/t/u rendered in the robustness pass. All read parquet only; no WRDS re-pull.
- **Palette (Okabe–Ito):** blue `#0072B2`, vermillion `#D55E00`, orange `#E69F00`, skyblue
  `#56B4E9`, green `#009E73`, grey `#999999`. Convention: **crisis = vermillion, calm = blue**.

---

## fig_a_eligible_constituents — FINAL (data-quality)
1. **Identity:** `fig_a_eligible_constituents` (.png + .pdf). Final descriptive panel.
2. **Provenance:** `window_eligibility.parquet`; per window, `nunique(permno)`. Crisis shading
   from the per-window `regime` (shards).
3. **Visual:** single panel. x = window start date (1990–2025); y = "Number of eligible names"
   (linear, from 0). Blue line+markers = eligible count per window. Vermillion shaded spans
   (alpha 0.12) = contiguous crisis windows; legend "crisis". Title "Eligible S&P 500
   constituents per window". Caption = data window + N valid pair-windows.
4. **Numbers:**

| quantity | value |
|---|---|
| windows | 429 |
| eligible names/window: min / median / max | 478 / 497 / 504 |

5. **Mapping:** data-quality / coverage (constituent count over time).

## fig_b_coverage_missingness — FINAL (data-quality)
1. **Identity:** `fig_b_coverage_missingness` (.png + .pdf). Final descriptive panel.
2. **Provenance:** `daily_returns.parquet`, grouped by calendar year: `avail = ret.notna().mean()`,
   `zero = (ret == 0).mean()`.
3. **Visual:** two panels side by side. Left: green bars, "Return availability", y = "Non-missing
   return-days (%)" 0–100. Right: orange bars, "Exact-zero returns", y = "Zero-return days (%)".
   x = Year. Caption notes sgn(0)=0 days are dropped from the ±1 series.
4. **Numbers:**

| quantity | range across years |
|---|---|
| return availability (%) | 99.4 – 100.0 |
| exact-zero returns (%) | 0.44 – 15.02 |

5. **Mapping:** data-quality / missingness (no manuscript headline; supports panel reliability).

## fig_c_return_distributions — FINAL (data-quality / regime separation)
1. **Identity:** `fig_c_return_distributions` (.png + .pdf). Final descriptive panel.
2. **Provenance:** `daily_returns.parquet`; `ret` trimmed at the 99.9th |R| pctile for display;
   θ = `median(|R|)`.
3. **Visual:** two panels. Left: blue histogram of signed daily returns (80 bins), black vertical
   line at 0, title "Daily returns". Right: skyblue histogram of |return| (80 bins), vermillion
   dashed line "median |R| = …" = θ; title "|Return| (magnitude regimes)".
4. **Numbers:**

| quantity | value |
|---|---|
| median |R| (θ, displayed) | ≈ 0.0106 (1.06%) |
| mean |R| | 0.0164 |

5. **Mapping:** data-quality — shows the magnitude regimes the θ split separates.

## fig_d_cell_counts — FINAL (aggregate-fed)
1. **Identity:** `fig_d_cell_counts` (.png + .pdf). Final (uses `plot_cell_counts_agg`, the
   streaming variant — bars of **mean** per-cell size, not the boxplot test variant).
2. **Provenance:** `cell_summary.parquet`; mean N00/N01/N10/N11 per regime over valid pairs.
3. **Visual:** grouped bars per cell {N00 (lg,lg), N01 (lg,sm), N10 (sm,lg), N11 (sm,sm)}; one bar
   per regime (calm blue / crisis vermillion); black dashed line "N_min=10". y = "Mean count per
   cell (days)". Confirms all four cells exceed N_min at the canonical θ.
4. **Numbers (mean days/cell):**

| regime | N00 | N01 | N10 | N11 |
|---|---|---|---|---|
| calm | 16.10 | 13.49 | 13.48 | 15.16 |
| crisis | 16.52 | 13.13 | 13.12 | 15.69 |

   All ≥ 13 ≫ N_min=10; N00 (both-large) is the **best**-populated cell → E00-as-canonical is safe.
5. **Mapping:** regime/context diagnostic (N_min not discarding structure).

## fig_e_threshold_over_time — FINAL (diagnostic)
1. **Identity:** `fig_e_threshold_over_time` (.png + .pdf). Final.
2. **Provenance:** `daily_returns` + `window_eligibility`; per window, per eligible name θ =
   `window_threshold(|R|)` (median |R|); plots median and 10–90th pctile band across names.
3. **Visual:** skyblue band (10–90th pctile across names) + blue line (median name θ). x = window
   start; y = "θ (|return|)". Title "Per-stock regime threshold θ (median |R|) over time".
4. **Numbers (read from the rendered series):** median-name θ baseline ≈ 0.008–0.010; peaks
   ≈ **0.039** (2008-09 GFC), ≈ **0.028** (2020 COVID); 10–90 band reaches ≈ 0.066 in 2008-09.
5. **Mapping:** diagnostic — θ tracks volatility (rises in crises).

## fig_f_statistic_distributions — FINAL (aggregate-fed)
1. **Identity:** `fig_f_statistic_distributions` (.png + .pdf). Final (`..._agg`, streamed histograms).
2. **Provenance:** `stat_hist.parquet` (per var × regime fixed-bin counts).
3. **Visual:** three panels: s_odd, Δ, CTX. Step-density histograms overlaid by regime
   (crisis vermillion / calm blue). Black dashed reference at s_odd=2 (panel 1) and CTX=0 (panel 3).
   y = density. Suptitle "Cross-sectional CbD statistics by regime".
4. **Numbers (histogram support / totals):**

| var | bin range | total count |
|---|---|---|
| s_odd | [0, 4] | 41,778,309 |
| Δ | [0, 2] | 39,586,652 |
| CTX | [−4, 2] | 41,726,931 |

5. **Mapping:** core results — distribution of the CbD statistics by regime.

## fig_g_violation_rates — FINAL HEADLINE (aggregate-fed)
1. **Identity:** `fig_g_violation_rates` (.png + .pdf). **The deflation headline** (`..._agg`).
2. **Provenance:** `headline_rates.parquet`; `naive_rate = mean(s_odd>2)`, `cbd_rate = mean(CTX>0)`
   over valid pairs, per regime. Printed % labels on bars.
3. **Visual:** grouped bars per regime (crisis, calm): vermillion = "naive (s_odd>2)", blue =
   "CbD-corrected (CTX>0)", with printed `%` labels (2 dp). x tick = "regime (N=…)". y = "Share of
   pairs (%)". Title "Deflation: naive violation rate vs CbD-corrected rate".
4. **Numbers:**

| regime | N valid | naive s_odd>2 | n_naive | CbD CTX>0 | n_cbd |
|---|---|---|---|---|---|
| calm | 37,156,946 | **2.468%** | 916,953 | **0.0290%** | 10,793 |
| crisis | 4,621,363 | **4.575%** | 211,424 | **0.0649%** | 2,999 |

   The naive rate (higher in crisis) collapses ~85–70× under CTX>0 → the deflation.
5. **Mapping:** **deflation headline** (naïve vs CbD, crisis vs calm).

## fig_g_sector_violation_rates — PLACEHOLDER (not usable)
1. **Identity:** `fig_g_sector_violation_rates` (.png + .pdf). **Graceful placeholder, not a result.**
2. **Provenance:** would be `pair_window_stats` + a `sector` map from `identifiers.parquet`.
   `identifiers.parquet` columns = {permno, permco, ticker, securitynm, issuernm, primaryexch,
   sharetype, securitytype} — **no sector/GICS column**, so the function returns a placeholder.
3. **Visual:** blank panel, title "Sector violation rates", grey centered text **"no sector
   identifiers available"**.
4. **Numbers:** none.
5. **Mapping:** optional sector-stratified headline — **not available** (no sector identifiers).
   Verification flag: do not cite; needs a GICS/sector merge to populate.

## fig_h_sodd_vs_delta — FINAL (scatter)
1. **Identity:** `fig_h_sodd_vs_delta` (.png + .pdf). Final.
2. **Provenance:** `scatter_subsample.parquet` (per-window-quota subsample of valid pairs;
   not the full 41.8M).
3. **Visual:** scatter of Δ (x) vs s_odd (y), colored by regime (crisis vermillion / calm blue).
   Black dashed line **CTX=0: s_odd = Δ + 2**. Points **above** the line are CTX>0. Title
   "s_odd vs Δ — points above the line are CTX>0".
4. **Numbers (subsample):**

| quantity | value |
|---|---|
| points plotted | 19,734 (calm 17,434 / crisis 2,300) |
| fraction above CTX=0 line | 0.020% (≈ 2 of 19,734) |
| s_odd range | [0.0, 3.1] |
| Δ range | [0.06, 3.9] |

   Visually almost the entire cloud sits below s_odd=Δ+2 → CTX>0 is rare (the deflation made visible).
5. **Mapping:** s_odd-vs-Δ / CTX=0 boundary scatter.

## fig_i_ctx_overlay — FINAL (aggregate-fed)
1. **Identity:** `fig_i_ctx_overlay` (.png + .pdf). Final (`..._agg`).
2. **Provenance:** empirical CTX from `stat_hist` (var=ctx, summed over regimes);
   classical-null CTX from `classical_null_hist.parquet` (the streamed `classical_null_reproduction`).
3. **Visual:** two step-density histograms — blue "empirical" vs orange "classical null" — over
   CTX; black dashed "CTX=0". Title "CTX: empirical vs classical-null (deflation reproduction)".
4. **Numbers:**

| series | support | mass at CTX>0 (≈) |
|---|---|---|
| empirical | [−4, 2], total 41.7M | calm 0.029% / crisis 0.065% (see fig g) |
| classical null | [−4, 2], total 126,176 sims | ≈ **0.12%** (147 sims with CTX≥0) |

   Both distributions sit overwhelmingly below 0 and overlap → a purely classical generator
   reproduces the (near-absent) apparent contextuality. The tiny null CTX>0 mass (~0.12%) is the
   finite-sample floor, comparable to/above the empirical rate.
5. **Mapping:** empirical-vs-null CTX overlay (the deflation null result).

## fig_j_threshold_sweep — FINAL (well-posedness map)
1. **Identity:** `fig_j_threshold_sweep` (.png + .pdf). Final.
2. **Provenance:** `threshold_sweep.parquet`. Strict-N_min `cbd_rate` (N_min=10), relaxed
   diagnostic `cbd_rate_relaxed` (N_min=3), classical-null `cbd_rate_null_relaxed`, finite-N floor
   `cbd_rate_smallN_floor`, and `n_valid` (denominator).
3. **Visual:** left y = "CbD-corrected rate CTX>0 (%)"; right y (log) = "valid pair-windows"
   (dotted). x = θ percentile of |R|. Lines per regime (crisis vermillion / calm blue) for strict
   CbD; dotted = valid denominator; open squares = relaxed-N_min diagnostic; "x" = classical-null
   (relaxed); green "—" = cell-size-matched finite-N floor. Title "Well-posedness map".
4. **Numbers (CbD CTX>0, strict N_min=10; relaxed where strict denominator collapses):**

| θq | calm CbD (strict) | crisis CbD (strict) | calm N_valid | crisis N_valid | calm CbD (relaxed) | crisis CbD (relaxed) |
|---|---|---|---|---|---|---|
| 0.25 | — (N=0) | — | 0 | 0 | 0.044% | 0.072% |
| 0.40 | 0.030% | 0.059% | 19,891,118 | 2,870,522 | 0.028% | 0.059% |
| 0.42 | 0.028% | 0.056% | 25,120,960 | 3,445,428 | 0.029% | 0.063% |
| 0.45 | 0.026% | 0.057% | 32,470,754 | 4,184,276 | 0.032% | 0.070% |
| 0.48 | 0.028% | 0.063% | 36,088,227 | 4,520,251 | 0.037% | 0.083% |
| **0.50** | **0.029%** | **0.065%** | 37,156,946 | 4,621,363 | 0.040% | 0.088% |
| 0.75 | — | — | 0 | 0 | 0.167% | 0.310% |
| 0.90 | — | — | 0 | 0 | 1.314% | 1.572% |
| 0.95 | — | — | 0 | 0 | — (N=0) | — |

   Out-of-band relaxed diagnostics at θq=0.90: classical-null CbD calm 0.000% / crisis 4.167%;
   finite-N floor calm 0.2% / crisis 0.3%. In-band (0.40–0.50) CbD stays ≈ 0.03%/0.06% (flat
   deflation); the strict denominator is 0 outside 0.40–0.50 (N11 starves <0.40, N00 >0.50).
5. **Mapping:** θ well-posedness map / deflation robustness.

## fig_k_network_metrics_crisis_calm — FINAL (crisis-vs-calm topology)
1. **Identity:** `fig_k_network_metrics_crisis_calm` (.png + .pdf). Final.
2. **Provenance:** `network_metrics.parquet`; mean over windows of each metric by graph × regime.
   Density-matched at top-5% (so edge_density is identical across graph kinds within a regime).
3. **Visual:** 2×2 small-multiples (edge density, avg clustering, giant-frac, modularity). Per
   panel, grouped bars over {pooled, E00, s_odd} with calm (blue) / crisis (vermillion). s_odd is
   labeled the **control**. Suptitle "Density-matched network topology: crisis vs calm".
4. **Numbers (mean over windows):**

| graph | regime | edge_density | avg_clustering | giant_frac | modularity |
|---|---|---|---|---|---|
| pooled | calm | 0.0416 | 0.2403 | 0.9404 | 0.3017 |
| pooled | crisis | 0.0396 | 0.2345 | 0.9021 | 0.2937 |
| E00 | calm | 0.0416 | 0.2059 | 0.9294 | 0.2862 |
| E00 | crisis | 0.0396 | 0.1881 | 0.8836 | 0.2662 |
| s_odd | calm | 0.0416 | 0.1033 | 0.9808 | 0.2501 |
| s_odd | crisis | 0.0396 | 0.0943 | 0.9704 | 0.2409 |

   E00 in crisis: giant-frac 0.929→0.884, clustering 0.206→0.188, modularity 0.286→0.266 (all drop).
5. **Mapping:** crisis-vs-calm network topology.

## fig_l_sodd_abs_amount — FINAL (s_odd "amount")
1. **Identity:** `fig_l_sodd_abs_amount` (.png + .pdf). Final descriptive.
2. **Provenance:** `network_metrics.parquet`, graph = `s_odd_abs` (absolute s_odd ≥ 2 graph,
   NOT density-matched); mean `edge_density` by regime.
3. **Visual:** two bars (calm blue / crisis vermillion), y = "edge density of the s_odd≥2 graph (%)",
   printed % labels. Title "Amount of CHSH coupling (absolute s_odd ≥ 2)".
4. **Numbers:**

| regime | s_odd≥2 edge density | windows |
|---|---|---|
| calm | **2.14%** | 379 |
| crisis | **3.86%** | 50 |

   The network analog of the naïve violation rate (more "coupling" in crisis), before deflation.
5. **Mapping:** s_odd amount (absolute coupling exhibit).

## fig_m_network_metric_over_time — INTERMEDIATE (single-metric subset of fig_t)
1. **Identity:** `fig_m_network_metric_over_time` (.png + .pdf). **Superseded by fig_t** (which shows
   all four metrics). fig_m plots only the default metric, `giant_frac`.
2. **Provenance:** `network_metrics.parquet`, giant-frac vs win_start by graph kind; crisis-shaded.
3. **Visual:** one panel: giant-component frac over time; E00 solid (blue), pooled dashed (grey),
   s_odd dotted (vermillion); vermillion crisis spans.
4. **Numbers:** giant-frac means (calm/crisis): E00 0.929/0.884, pooled 0.940/0.902, s_odd 0.981/0.970
   (same series summarized in fig_k); sharp dips in 2008-09 and 2020.
5. **Mapping:** metrics-over-time (single metric) — use fig_t/fig_u for the full version.

## fig_n_qap_hierarchy — FINAL (descriptive gate) 
1. **Identity:** `fig_n_qap_hierarchy` (.png + .pdf). Final descriptive (companion to the
   null-relative gate fig_p).
2. **Provenance:** `network_qap.parquet` (8 gate windows, node-subsampled to 60). QAP correlations
   and the MR-QAP gate R².
3. **Visual:** left: bars (mean ± sd over windows) of QAP r for tiers pooled→E00, pooled→s_odd,
   E00→s_odd (blue). Right: histogram of gate R²(s_odd ~ pooled+E00) over windows (vermillion).
   Suptitle "MR-QAP gate: 'does s_odd earn its place?'".
4. **Numbers (mean ± sd, n=8 windows):**

| quantity | mean | sd |
|---|---|---|
| r(pooled, E00) | 0.665 | 0.084 |
| r(pooled, s_odd) | 0.627 | 0.025 |
| r(E00, s_odd) | 0.595 | 0.037 |
| gate R²(s_odd ~ pooled+E00) | 0.452 | 0.031 |

5. **Mapping:** QAP/gate (descriptive); the verdict is in fig_p.

## fig_o_metrics_by_crisis_type — FINAL (by-crisis-type)
1. **Identity:** `fig_o_metrics_by_crisis_type` (.png + .pdf). Final.
2. **Provenance:** `network_metrics.parquet` + `config/crises.csv` taxonomy overlay
   (`crisis_types`); windows exploded across overlapping types; mean modularity by type × graph.
3. **Visual:** grouped bars over crisis types {financial, food, energy, commodity, mixed, none}, two
   bars each (E00 blue / s_odd vermillion). y = modularity. Title "modularity by crisis-taxonomy type".
4. **Numbers (mean E00 / s_odd modularity by type):**

| type | E00 | s_odd |
|---|---|---|
| financial | 0.2669 | 0.2324 |
| food | **0.2225** | 0.2057 |
| energy | 0.2387 | 0.2267 |
| commodity | 0.2431 | 0.2178 |
| mixed | 0.2300 | 0.2313 |
| none (calm) | 0.3035 | 0.2655 |

   Food windows have the **lowest** E00 modularity (0.2225) — below financial (0.2669); see fig_u/R3
   for the window-level significance.
5. **Mapping:** modularity/clustering by crisis-taxonomy type.

## fig_p_gate_null_relative — FINAL (gate verdict)
1. **Identity:** `fig_p_gate_null_relative` (.png + .pdf). **The s_odd gate verdict** (null-relative).
2. **Provenance:** `network_gate_summary.parquet` (real vs classical-null gate, bootstrap CI over
   8 gate windows). Reliability annotation from the s_odd_reliability available at render time.
3. **Visual:** bars real R² (vermillion) vs classical-null R² (grey); printed values; subtitle with
   d = real−null R², its 95% CI, and a verdict. Reliability ceiling line drawn **only if** it
   exceeds the bars (here it does not, so it is reported qualitatively, not drawn). Title "s_odd
   control: no structure beyond E00 tail coupling".
4. **Numbers:**

| quantity | value |
|---|---|
| real gate R²(s_odd ~ pooled+E00) | **0.452** |
| classical-null gate R² | **0.363** |
| d = real − null R² | **+0.089**  (95% CI **[+0.070, +0.105]**) |
| r(E00, s_odd) real / null | 0.595 / 0.534 |
| s_odd reliability SB r (annotated) | 0.045 |
| n gate windows | 8 |

   **Verification flag:** the annotated reliability (0.045) is from an *earlier* s_odd reliability
   estimate; the current R1 checkpoint (fig_q) gives s_odd **edge SB r = 0.163**. The gate verdict
   rests on `d` (CI excludes 0 but |d| is small → s_odd ≈ E00 tail coupling, CHSH adds little),
   which is independent of that reliability number.
5. **Mapping:** QAP/gate — null-relative verdict (s_odd is the ruled-out control).

## fig_q_topology_reliability — FINAL (R1 checkpoint)
1. **Identity:** `fig_q_topology_reliability` (.png + .pdf). Final R1 reliability checkpoint.
2. **Provenance:** `reliability_checkpoint.parquet` (built from `split_half_edge_weights/` shards
   + `s_odd_reliability.parquet`); regime-preserving odd/even split, 429 windows.
3. **Visual:** grouped bars per graph {pooled, E00, s_odd}: edge SB r (blue), top-q Jaccard (orange),
   modularity half-corr (green), giant-frac half-corr (vermillion). Black dashed "SB r=0.3 bar",
   dotted "Jaccard=0.2 bar". Title "R1 reliability checkpoint — E00 CLEARS the bar".
4. **Numbers (n=429 windows):**

| graph | edge SB r | top-q Jaccard | modularity half-corr | giant-frac half-corr | clustering half-corr |
|---|---|---|---|---|---|
| pooled | 0.297 | 0.058 | 0.661 | 0.645 | 0.911 |
| **E00** | **0.338** | **0.050** | **0.789** | **0.932** | 0.897 |
| s_odd | 0.163 | 0.038 | 0.919 | 0.956 | 0.954 |

   E00 **aggregate** metrics are reliable (half-corr 0.79–0.93); **edge-set identity is not**
   (Jaccard 0.05 < 0.2); edge SB r 0.338 marginally clears 0.3 → aggregate-metric claims only.
5. **Mapping:** reliability (R1 checkpoint) — gates the network claims.

## fig_r_sp500_index — FINAL (descriptive context)
1. **Identity:** `fig_r_sp500_index` (.png + .pdf). Final (P1).
2. **Provenance:** `sp500_index.parquet` (CRSP `crsp.dsp500_v2`): `spindx` level; equal-weight
   overlay = `cumprod(1+ewretd)` rebased to the spindx start. Crisis shading from window regime.
3. **Visual:** log-y index level. Blue solid = "S&P 500 index (CRSP spindx)"; orange dashed =
   "equal-weighted constituents (rebased)". Vermillion crisis spans. Title "S&P 500 index level,
   1990-2025".
4. **Numbers:**

| quantity | value |
|---|---|
| trading days | 9,067 (1990-01-02 … 2025-12-31) |
| spindx start → end | 359.7 → 6845.5 |
| equal-weighted (rebased to 359.7) end | ≈ 19,951 |

   EW ends far above cap-weight (rebased) — small/equal-weight premium over the full span.
5. **Mapping:** S&P 500 index level (context figure).

## fig_s_abs_return_box — FINAL (descriptive context)
1. **Identity:** `fig_s_abs_return_box` (.png + .pdf). Final (P1), monthly binning.
2. **Provenance:** `daily_returns.parquet`; cross-sectional |daily return| (×100) grouped by month.
3. **Visual:** skyblue box-and-whisker per month (whiskers 1.5 IQR, fliers hidden, black medians),
   crisis-shaded. y = "|daily return| (%)" (from 0); x = Date. Title "Cross-sectional |daily return|
   by month".
4. **Numbers:** **432 monthly boxes**. Whisker peaks visible at GFC 2008-09 (≈ 16%) and COVID
   2020-03 (≈ 18%); dot-com 2001-02 and 2011 elevated — face validity that crises = high-|R|.
5. **Mapping:** |R| box-whisker (volatility environment behind the θ regimes).

## fig_t_all_metrics_over_time — FINAL (raw metrics over time)
1. **Identity:** `fig_t_all_metrics_over_time` (.png + .pdf). Final RAW version (companion to the
   config-null-controlled fig_u).
2. **Provenance:** `network_metrics.parquet`; the four metrics over window-start, by graph kind.
3. **Visual:** 2×2 small-multiples (edge density, avg clustering, giant-frac, modularity). E00 solid
   (blue), pooled dashed (grey), s_odd dotted (vermillion); crisis-shaded. Suptitle "Network topology
   over time (raw metrics)".
4. **Numbers:** same series summarized in fig_k (means by graph×regime); time-resolved, with crisis
   dips most pronounced in giant-frac/clustering at 2008-09, 2011, 2020.
5. **Mapping:** metrics-over-time (raw).

## fig_u_excess_metrics_over_time — FINAL (config-null excess over time)
1. **Identity:** `fig_u_excess_metrics_over_time` (.png + .pdf). Final R2 version (the controlled
   metric — preferred over raw fig_t for inference).
2. **Provenance:** `network_metrics_excess.parquet`; each metric as **excess over a degree-preserving
   (double-edge-swap) configuration-model null** (12 rewires). Edge-density excess ≡ 0 (invariant
   under rewiring) → that quadrant is intentionally blank.
3. **Visual:** 2×2 small-multiples (top-left blank); giant-frac, clustering, modularity **excess over
   config-null** over time; E00 solid (blue), pooled dashed (grey), s_odd dotted (vermillion); black
   y=0 line; crisis-shaded. Suptitle "Network topology over time (configuration-model excess-over-null)".
4. **Numbers (mean excess over windows; with R3 window-level permutation p, E00):**

| graph | regime | modularity excess | giant-frac excess | clustering excess |
|---|---|---|---|---|
| E00 | calm | 0.1172 | −0.00211 | 0.0688 |
| E00 | crisis | 0.1035 | −0.00442 | 0.0362 |
| pooled | calm | 0.1244 | −0.00276 | 0.1063 |
| pooled | crisis | 0.1215 | −0.00499 | 0.0985 |
| s_odd | calm | 0.0449 | −0.00017 | 0.0198 |
| s_odd | crisis | 0.0302 | −0.00041 | 0.0103 |

   **R3 window-level tests (E00, excess; from `window_level_inference.parquet`):**

| contrast | metric | mean (grp1) | mean (grp2) | n (grp1/grp2) | diff | perm p |
|---|---|---|---|---|---|---|
| crisis vs calm | giant_frac | −0.00442 | −0.00211 | 50 / 379 | −0.00231 | **0.0015** |
| crisis vs calm | avg_clustering | 0.0362 | 0.0688 | 50 / 379 | −0.0326 | **0.0002** |
| crisis vs calm | modularity | 0.1035 | 0.1172 | 50 / 379 | −0.0137 | 0.200 (n.s.) |
| food vs financial | modularity | 0.0807 | 0.1146 | 58 / 114 | −0.0339 | **0.0084** |
| food vs financial | avg_clustering | 0.0159 | 0.0453 | 58 / 114 | −0.0294 | **0.021** |
| food vs financial | giant_frac | −0.00218 | −0.00371 | 58 / 114 | +0.00153 | 0.070 (n.s.) |

   Survives controls: E00 **crisis fragmentation** (giant-frac p=0.0015) + **de-clustering**
   (p=0.0002), and the **food<financial modularity** gap (p=0.0084). Crisis modularity effect washes
   out (p=0.20) after the degree-null control.
5. **Mapping:** metrics-over-time (excess) + crisis-vs-calm / food-vs-financial topology inference.

---

### Summary: which figures are the manuscript results
| manuscript result | figure(s) |
|---|---|
| deflation headline | **fig_g** (sector variant fig_g_sector = placeholder, not usable) |
| s_odd-vs-Δ / CTX=0 scatter | fig_h |
| empirical-vs-null CTX overlay | fig_i |
| θ well-posedness map | fig_j |
| s_odd amount | fig_l |
| QAP/gate | fig_n (descriptive), **fig_p** (null-relative verdict) |
| crisis-vs-calm topology | **fig_k**; over time fig_t (raw) / **fig_u** (excess, controlled) |
| by-crisis-type | fig_o |
| reliability checkpoint | **fig_q** |
| S&P index level | fig_r |
| |R| box-whisker | fig_s |
| data-quality / diagnostics | fig_a, fig_b, fig_c, fig_d, fig_e, fig_f |
| superseded | fig_m (single-metric subset of fig_t) |

**Verification flags:** (1) `fig_g_sector` is an empty placeholder — `identifiers.parquet` has no
sector column. (2) `fig_p`'s annotated s_odd reliability (0.045) predates the R1 recomputation
(current s_odd edge SB r = 0.163, fig_q); the gate verdict rests on d = real−null R² = +0.089
[CI +0.070, +0.105], which is unaffected.
