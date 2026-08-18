# Wind Energy Forecasting — Hackathon Bolzano 2026

> **Day-ahead wind power forecasting using a two-stage tree ensemble pipeline.**  
> Submitted to Hackathon Bolzano 2026 organised by Key to Energy / unibz.

---

## Overview

This project builds a machine learning pipeline to predict hourly wind farm power output from day-ahead meteorological forecasts. Operators in the Day-Ahead electricity market must submit production forecasts 24 hours in advance; deviations from actual generation incur financial penalties. The goal is a regression model that minimises a custom market-aware error metric.

The approach is a **two-stage pipeline**:

- **Stage 1 — Reliability Assessment**: an independent module evaluates the trustworthiness of each weather timestamp by measuring cross-provider agreement, missing data fraction, and wind direction consensus across 235 weather columns.
- **Stage 2 — Tree Ensemble Forecasting**: five tree-based models are trained and compared on 37 engineered features derived from the weather data, physics-informed transformations, temporal encodings, and the Stage 1 reliability scores.

---

## Dataset

| Split | Period | Rows |
|-------|--------|------|
| Training | Jan 2024 – Jun 2025 | 13,126 hours |
| Test | Jul 2025 – Dec 2025 | 4,416 hours |

**Weather forecasts:** 235 columns from 5 meteorological models × 6 sites × 4 providers, covering wind speed, wind gust, wind direction, temperature, and solar irradiation.

**Target:** hourly wind power production in MWh (range 0 – ~10 MWh).

---

## Data Cleaning

Two operational variables distort the observed production and must be handled before training:

| Variable | Description | Rule |
|----------|-------------|------|
| **Availability** | Fraction of plant capacity online | Correct: `P_corr = P_obs / Availability` |
| **ODD** | Grid-operator curtailment order | If ODD < 1 → exclude row |

Additional filters: rows where `Availability = 0` (plant offline) or production exceeds the 99th-percentile capacity by more than 10% are also excluded. After cleaning: **11,697 training** and **4,066 test** samples remain.

---

## Feature Engineering

235 raw weather columns → **37 engineered features**:

| Group | Features | Rationale |
|-------|----------|-----------|
| Wind speed | mean, std, min, max, range, top-provider mean | Aggregate + provider spread |
| Wind gust | mean, std, max | Peak energy; highest feature importance |
| Wind direction | sin, cos (circular encoding) | Avoids 359°→0° discontinuity |
| Temperature, irradiation | mean | Air density and weather-state signals |
| Physics | `ws³`, `ws²`, gust factor, `wind_u`, `wind_v` | Power ∝ v³; directional components |
| Temporal | hour, month, day-of-year + sin/cos encodings | Cyclical time structure |
| Reliability | provider agreement, missing fraction, direction consensus | Trust signal per timestamp |

Missing values are imputed with **training-set medians** (applied to both train and test, no leakage).

---

## Models

Five tree-based models are compared. All are scale-invariant (no `StandardScaler` needed). Boosting models use early stopping on an internal 3-month validation slice.

| Model | Family |
|-------|--------|
| Random Forest | Bagging |
| Extra Trees | Bagging |
| XGBoost | Boosting |
| LightGBM | Boosting |
| CatBoost | Boosting |

---

## Scoring Metric

The hackathon uses a custom metric **E_f** (Errore di Forecasting):

```
E_f = Σ_d ( Σ_{h∈d} |ŷ_h - y_h| ) + | Σ_d Σ_{h∈d} (ŷ_h - y_h) |
```

- **First term**: total daily absolute error
- **Second term**: magnitude of cumulative bias (systematic drift)

A model that makes random, self-cancelling errors is penalised less than one that consistently over- or under-predicts. **Lower E_f is better.**

---

## Results

Test-set performance (Jul – Dec 2025):

| Model | MAE | RMSE | R² | E_f | Bias |
|-------|-----|------|----|-----|------|
| **XGBoost** | **0.761** | 1.178 | **0.667** | **3754** | +660 |
| LightGBM | 0.761 | 1.188 | 0.661 | 3780 | +684 |
| Extra Trees | 0.757 | 1.187 | 0.662 | 3845 | +767 |
| Random Forest | 0.761 | 1.197 | 0.656 | 3853 | +757 |
| CatBoost | 0.769 | 1.211 | 0.648 | 3929 | +800 |

**XGBoost ranks first** (E_f = 3,754). All five models achieve MAE ≈ 0.76 MWh on a 0–10 MWh target range — the differences are driven by systematic bias, not raw accuracy.

---

## Key Findings

1. **All models systematically over-predict** (+660 to +800 MWh bias). Training mean production (1.734 MWh) > test mean (1.588 MWh) — the models learned from a windier period and carried that expectation into a calmer test period. This **distribution shift** is the dominant source of E_f error.

2. **Boosting edges ahead of bagging**, mainly through lower bias. The gap is small — all five tree models are closely matched on raw accuracy.

3. **Validation winner ≠ test winner.** On the 3-month internal validation window, CatBoost looked best (bias ≈ +7 MWh). On the held-out 6-month test set, CatBoost has the highest bias (+800 MWh) and ranks last. Model selection on a single short window can mislead.

4. **Data quality sets the ceiling.** With 235 noisy, partially-disagreeing weather columns and a distribution shift between train and test, the best R² is 0.667. Bias correction or recalibration would likely help more than further model tuning.

---

## Project Structure

```
.
├── hackathon_final.ipynb   # Main notebook (EDA → cleaning → features → training → results)
├── README.md
└── requirements.txt
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Data

The dataset is hosted on Kaggle:  
[`faroukfadelbrachemi/data4econ`](https://www.kaggle.com/datasets/faroukfadelbrachemi/data4econ)

Set `DATA_DIR` in the notebook to point to your local copy or Kaggle input path.

---

## Author

**Farouk Fadel Brachemi**  
M.Sc. Computing for Data Science — Free University of Bozen-Bolzano  
AI Engineer Intern — Volt LOGIQ, Bolzano
