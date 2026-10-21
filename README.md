# When Backtests Lie

**A governance-grade validation framework for machine-learning Value-at-Risk models under regime shifts.**

Niraj Neupane, CA (ICAI) — Quantitative Researcher, Korvane; Financial Economist, Calderyn Institute · ORCID: 0009-0003-7026-7026

![Status](https://img.shields.io/badge/Status-Working%20Paper-e05d44)
![Domain](https://img.shields.io/badge/Domain-Model%20Risk-1f6feb)
![Code](https://img.shields.io/badge/Code-Python-2b3137)
![Data](https://img.shields.io/badge/Data-Public%20%7C%202000--2026-6f42c1)
![License](https://img.shields.io/badge/License-MIT-3fb950)

---

## What this is

This repository accompanies the working paper **_When Backtests Lie: A
Governance-Grade Validation Framework for Machine-Learning Value-at-Risk Models
under Regime Shifts._** It reproduces every table and figure in the paper from
public data, with fixed random seeds.

It sits alongside the companion tail-risk paper _Conditional Tail Episode Risk_
([repo](https://github.com/nirajneupane17/conditional-tail-episode-risk)): that
work asks how to _measure_ the worst sustained tail episode; this one asks whether
the tools used to _validate_ risk models can be trusted once those models are
built with machine learning.

> **The result in one line.** VaR models that **pass** the standard unconditional
> backtest (Kupiec) breach **2–3× the target rate in stress at 95%**, and
> **5–9× at the 99% level** that underpins regulatory capital — across equities,
> fixed income, FX, and crypto. A formal logistic test confirms the regime effect
> is overwhelming (odds ratios 3.5–4.8, _p_ < 10⁻²⁷). Standard validation gives
> **false comfort**; the paper's governance-grade scorecard catches what it misses.

---

## The validation scorecard

Four one-step-ahead, walk-forward VaR forecasters — historical simulation,
GARCH-_t_, gradient-boosted quantile regression, linear quantile regression — are
run through a five-part battery mapped to the three SR 11-7 governance pillars. A
model passes overall **only if it passes every pillar.**

| Test | Governance pillar | What it catches |
|---|---|---|
| Kupiec `LR_uc` | Outcomes analysis | Wrong average breach rate |
| Christoffersen `LR_cc` | Outcomes analysis | Clustered breaches |
| **Stress amplification Λ** | Outcomes analysis | **Regime fragility (the blind spot)** |
| Probability of backtest overfitting | Ongoing monitoring | Selection artifacts |
| Conformal coverage gap | Calibration | Miscalibration |

The novel diagnostic is **Λ = stress breach rate ÷ target** — the one that standard
backtesting averages away.

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt

python src/download_data.py     # 1. fetch public data   → data/
python src/run_backtests.py     # 2. full grid, 6 assets → results/
python src/regime_test.py       # 3. formal regime test  (prints odds ratios)
python src/make_figures.py      # 4. publication figures → figures/
```

Steps 2–4 read only from `data/` and run fully offline. Step 1 needs internet.
The shipped `data/` and `results/` already contain the exact inputs and outputs
behind the paper, so you can reproduce the figures without re-downloading.

---

## Repository layout

```
.
├── README.md
├── LICENSE                   # MIT
├── CITATION.cff
├── requirements.txt
├── src/
│   ├── download_data.py      # public data → data/*.csv
│   ├── pipeline.py           # models + validation battery + scorecard
│   ├── run_backtests.py      # full grid: 6 assets × {95%, 99%} → results/
│   └── make_figures.py       # publication figures → figures/
├── data/                     # downloaded returns + regime labels
├── results/                  # scorecards (CSV) produced by the pipeline
├── figures/                  # PDF/PNG figures used in the paper
└── paper/                    # manuscript PDF
```

---

## Data

All data is **public and free** — no proprietary or employer data is used. Daily
log returns, 2000–2026 where available (asset start dates differ).

| Asset | Source | Class |
|---|---|---|
| S&P 500 (`^GSPC`) | Yahoo Finance | Equity (primary) |
| Nasdaq Composite (`^IXIC`) | Yahoo Finance | Equity |
| Long Treasuries (`TLT`) | Yahoo Finance | Fixed income |
| EUR/USD (`DEXUSEU`) | FRED | FX |
| USD/JPY (`DEXJPUS`) | FRED | FX |
| Bitcoin (`BTC-USD`) | Yahoo Finance | Crypto (high-vol) |

Regimes are labelled **point-in-time**: a day is _stress_ if its trailing 21-day
realized volatility exceeds the expanding 80th percentile, else _calm_. An
exogenous crisis calendar (2008, 2011, 2020, 2022) serves as a cross-check.

---

## Reproducibility

All randomness is seeded. `run_backtests.py` writes one scorecard CSV per
asset/level plus a combined `grid_ALL.csv`; the paper's tables correspond exactly
to these files (500-day window, monthly re-estimation). Full estimator and feature
specifications are in Appendices B–C of the paper.

## Citation

If you use this code or data, please cite the paper — see [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE). Data is redistributed under the terms of its original public
providers; this repository stores derived daily returns for convenience and
reproducibility.
