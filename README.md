# Governance-Grade Validation of Machine-Learning VaR Models

Replication code and data for:

> **When Backtests Lie: A Governance-Grade Validation Framework for
> Machine-Learning Value-at-Risk Models under Regime Shifts**
> Niraj Neupane, CA (ICAI) — Korvane / Calderyn Institute
> SSRN working paper, 2026.

This repository reproduces every table and figure in the paper from public data,
with fixed random seeds.

---

## The result in one line

VaR models that **pass** the standard unconditional backtest (Kupiec) breach
**2–3× the target rate in stress at 95%**, and **5–9× at the 99% level** that
underpins regulatory capital — across equities, fixed income, FX, and crypto. A
formal logistic test confirms the regime effect is overwhelming (odds ratios
3.5–4.8, _p_ < 10⁻²⁷). Standard validation gives **false comfort**; the paper's
governance-grade scorecard catches what it misses.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── src/
│   ├── download_data.py     # pull public data → data/*.csv
│   ├── pipeline.py          # models + validation battery + scorecard
│   ├── run_backtests.py     # full grid: 6 assets × {95%, 99%} → results/
│   ├── regime_test.py       # formal logistic test of the regime effect
│   └── make_figures.py      # publication figures → figures/
├── data/                    # downloaded returns + regimes (reproducible)
├── results/                 # scorecards (CSV) produced by the pipeline
├── figures/                 # PDF/PNG figures used in the paper
└── paper/                   # manuscript PDF
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt

python src/download_data.py        # 1. fetch public data  → data/
python src/run_backtests.py        # 2. run the full grid  → results/
python src/regime_test.py          # 3. formal regime test (prints odds ratios)
python src/make_figures.py         # 4. figures            → figures/
```

Steps 2–4 read only from `data/` and are fully offline. Step 1 needs internet.

## Data

All data is **public and free**. Equity indices, the Treasury ETF, and Bitcoin
come from Yahoo Finance (with a Stooq fallback); FX and the financial-stress
indices come from FRED. Daily log returns, 2000–2026 where available. Asset start
dates differ (Treasuries from 2002, Bitcoin from 2014). No proprietary or
employer data is used.

| Asset | Ticker | Class |
|---|---|---|
| S&P 500 | `^GSPC` | Equity (primary) |
| Nasdaq Composite | `^IXIC` | Equity |
| Long Treasuries | `TLT` | Fixed income |
| EUR/USD | `DEXUSEU` (FRED) | FX |
| USD/JPY | `DEXJPUS` (FRED) | FX |
| Bitcoin | `BTC-USD` | Crypto (high-vol) |

## Method

Four one-step-ahead, walk-forward VaR forecasters — historical simulation,
GARCH-_t_, gradient-boosted quantile regression, linear quantile regression —
each run through a five-part battery mapped to the three SR 11-7 governance
pillars:

1. **Unconditional coverage** — Kupiec LR
2. **Conditional coverage** — Christoffersen LR
3. **Regime-conditional coverage** — stress amplification Λ = stress breach rate ÷ target
4. **Overfitting control** — probability of backtest overfitting (combinatorial CV)
5. **Calibration** — split-conformal recalibration

A model passes overall only if it passes every pillar.

## Reproducibility

All randomness is seeded. `run_backtests.py` writes one scorecard CSV per
asset/level plus a combined `grid_ALL.csv`. The numbers in the paper's tables
correspond exactly to these files (monthly re-estimation, `REFIT = 21`).

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).

## License

MIT — see `LICENSE`. The data is redistributed under the terms of its original
public providers; this repository stores derived daily returns for convenience
and reproducibility.
