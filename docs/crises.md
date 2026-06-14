# Crisis taxonomy (1990–2025)

`config/crises.csv` is a **named-crisis overlay** for cross-crisis comparison. It
is intentionally **separate from the binary crisis/calm driver**: the binary
label used throughout the analysis comes from `load_crisis_labels` (VIX level or
NBER recession dates) and is *never* derived from this table. The taxonomy exists
only so results can be sliced and compared across episodes, and so we can flag
episodes whose character (e.g. food) does **not** register as VIX/NBER stress.

Each row has: `name, start, end, type, notes`. `type ∈ {financial, food, energy,
commodity, mixed}` (slash-separated when an episode spans categories). Boundaries
are a modeling choice — refine as needed; they are deliberately a bit generous so
a 63-trading-day window that *touches* an episode is tagged with it.

Load it with the shared util:

```python
from cbd_analysis import load_crisis_taxonomy, tag_windows_with_crises
tax = load_crisis_taxonomy()                       # config/crises.csv
tags = tag_windows_with_crises(window_ids, win_bounds, taxonomy=tax)
```

## The episodes

| Episode | Span | Type | Registers as VIX/NBER stress? |
|---|---|---|---|
| Gulf War oil shock | 1990-08 – 1991-02 | energy/financial | **Yes** — NBER recession Jul 1990 – Mar 1991. |
| Asian crisis / LTCM | 1997-07 – 1998-12 | financial | **Partly** — sharp VIX spike late 1998; *no* US recession. |
| Dot-com bust | 2000-03 – 2002-10 | financial | **Yes** — NBER recession Mar–Nov 2001. |
| Global food price crisis | 2007-01 – 2008-08 | food | **Mostly no** — a food/commodity episode; only its tail overlaps the GFC equity-stress window. **Key mismatch.** |
| Global Financial Crisis | 2007-12 – 2009-06 | financial | **Yes** — the canonical crisis: NBER recession + record VIX (Lehman, Sep 2008). |
| Russian wheat ban / Arab Spring food spike | 2010-06 – 2011-12 | food | **Mostly no** — food-price shock with little US equity stress; partial overlap with the 2011 euro-debt VIX spike only. **Key mismatch.** |
| European sovereign debt | 2011-05 – 2012-09 | financial | **Partly** — VIX spike Aug 2011; *no* US recession. |
| Oil / commodity collapse | 2014-06 – 2016-02 | energy/commodity | **Partly** — energy-sector stress and a Q1-2016 risk-off; *no* US recession. |
| COVID-19 | 2020-02 – 2020-06 | mixed/financial | **Yes** — record VIX Mar 2020; NBER recession Feb–Apr 2020. |
| Russia–Ukraine food–energy | 2022-02 – 2022-12 | food/energy | **Partly** — equity drawdown and an inflation/commodity shock; *no* NBER recession. |

## Reconciliation with the binary labeler (why this matters)

The binary VIX/NBER label captures **financial-market stress**. Several episodes
in the taxonomy are primarily **food or energy** shocks that leave the equity VIX
and the NBER recession indicator largely unmoved:

- **2007–08 global food price crisis** and **2010–11 Russian wheat ban / Arab
  Spring** are the clearest cases: real-economy/commodity stress that a stock-
  market-based crisis label will read as "calm" (except where they happen to
  overlap the GFC or the euro-debt VIX spike).
- **2014–16 commodity collapse** and **2022 Russia–Ukraine** are intermediate:
  real VIX wobbles but no NBER recession.

This food-vs-financial mismatch is **a finding, not a nuisance**. It is also the
hook for the later agricultural extension: the same network machinery, restricted
to a food/ag node set (the `sector_map` hook already carried in `networks.py`),
can ask whether food episodes that are invisible to the equity VIX label leave a
signature in the ag sub-network — without re-architecting anything here.
