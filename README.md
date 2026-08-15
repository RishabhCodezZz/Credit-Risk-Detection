# Credit Score Classification — Leakage-Audited Pipeline

A 3-class credit-score classifier (Poor / Standard / Good) built around one central claim: **most
public notebooks on this dataset report an inflated score because of customer-level data leakage,
and this pipeline measures, then eliminates, that inflation.**

| | |
|---|---|
| **Dataset** | [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) (Kaggle, ParisRohan) — 100,000 rows, 12,500 customers × 8 monthly snapshots |
| **Task** | Multiclass classification — `Credit_Score` ∈ {Poor, Standard, Good} |
| **Final holdout** | **macro-F1 0.70 · accuracy 71% · ROC-AUC (OvR) 0.87** |
| **Leakage found & removed** | **+0.12 macro-F1** — the gap between a naive random split and a customer-grouped split, measured directly in the notebook |
| **Stack** | LightGBM, XGBoost, CatBoost, Optuna, SHAP, PyTorch, scikit-learn |

> **This is a methodology project, not a production risk model.** The dataset is synthetic — its
> messy values are deliberately injected and its labels are rule-generated, not observed from real
> borrowers. The point of this repo is the validation discipline, not the raw score.

---

## The problem this solves

Each customer in this dataset appears ~8 times (one row per month), and a customer's credit score
barely changes month to month. A plain `train_test_split` on rows lets the same customer's data land
on both sides of the split — the model partly memorizes the customer instead of learning the pattern,
and the reported accuracy is inflated as a result.

This notebook proves the effect rather than just avoiding it. It trains the same model both ways and
reports both numbers:

```
Random row split   (leaky)    macro-F1: 0.8171   <- what a naive split reports
Grouped customer split (honest) macro-F1: 0.6974
Inflation from leakage:        +0.1197
```

Everything downstream — splitting, imputation, feature engineering, model selection, ensembling,
threshold tuning — is built to never let that leakage back in.

## Methodology

**1. Split before preprocessing.** `GroupShuffleSplit` on `Customer_ID` runs *first*, before any
statistic is computed. Every quantile cap, median, and mode used for imputation is fit on the
training customers only and applied to holdout — not computed on the full dataset and then split.

**2. Four-tier imputation**, cheapest and most-informative source first:
   - exact reconstruction where the panel structure allows it (a customer's credit history age
     increments by exactly one month per row, so it's recoverable losslessly, not just imputed)
   - the customer's own observed history (forward/backward fill)
   - interpolation along the customer's own timeline for genuinely time-varying columns
   - a train-fitted median/mode, only as a last resort

**3. Causal feature engineering.** Rolling statistics use `shift(1)` before the expanding window, so
no row ever sees its own current or future values. A customer's first month has no history — it's
filled with a structural zero, not a population average pretending they have average history.

**4. Model selection on out-of-fold predictions only.** Five models (Logistic Regression, Random
Forest, LightGBM, XGBoost, CatBoost) are trained under `StratifiedGroupKFold`, tuned with Optuna
(30 trials/model, GPU-parallel, wall-clock bounded), and ranked purely on OOF macro-F1. **The holdout
set is touched exactly once, at the very end**, by whichever model won on OOF — never used to pick
the winner.

**5. Ensembling and threshold tuning, fit on OOF, verified on holdout.** A greedy-selected weighted
blend of the trained models beats every single model. A per-class decision-threshold search is also
attempted — and rejected by its own noise floor (see below), a deliberate self-check rather than a
result quietly discarded.

**6. Two tested hypotheses that failed, kept in the notebook on purpose:**
   - *Customer-level prediction averaging* — plausible, since customers repeat across months. Tested
     and **rejected**: only 41.7% of customers hold a single label across their 8 months, so
     averaging smooths away real, moving signal instead of noise.
   - *Per-class threshold tuning* — moved OOF score by +0.0003, indistinguishable from noise, and a
     `MIN_THRESHOLD_GAIN` floor now auto-rejects any search result that doesn't clear a real margin,
     so the pipeline stops proposing a correction it can't back up.

## Results

**Out-of-fold model comparison** (5-fold `StratifiedGroupKFold`, before ensembling):

| Model | macro-F1 | ROC-AUC (OvR) |
|---|---|---|
| LightGBM (tuned) | 0.684 | 0.861 |
| Random Forest | 0.684 | 0.860 |
| LightGBM | 0.683 | 0.862 |
| XGBoost | 0.683 | 0.863 |
| CatBoost | 0.683 | 0.865 |
| PyTorch MLP | 0.664 | 0.848 |
| Logistic Regression | 0.649 | 0.811 |

**Final holdout** (touched once, after all selection):

| Configuration | macro-F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| Best single model | 0.701 | 70.6% | 0.869 |
| **Weighted ensemble (final)** | **0.705** | **71.2%** | **0.872** |
| Customer aggregation (rejected) | 0.699 | 70.8% | 0.856 |

**Diagnostics, run three times independently — all agree:**
- OOF vs. holdout gap is *negative* (holdout scores higher) → **not overfitting**
- Learning curve is flat from 25% to 100% of training data (0.672 → 0.694 → 0.697) → the model is
  **feature-limited, not data-limited**; more rows would not move the score
- A saved model bundle (blend weights, fold models, fitted preprocessing stats) round-trips
  raw CSV rows to predictions with **100% match** against the in-notebook results

**Top features by SHAP importance:** `months_of_history`, `Credit_Mix`, `Num_Credit_Card`,
`Num_Credit_Inquiries`, `Outstanding_Debt` (customer-level mean), `Interest_Rate`.

## Repository contents

| File | What it is |
|---|---|
| `credit_score_model.ipynb` | Source notebook — clean, unexecuted, ready to run on Kaggle |
| `credit_score_model_executed.ipynb` | Full run with all outputs, on Kaggle GPU T4×2 (~23 min) |

## Running it

1. Open `credit_score_model.ipynb` on [Kaggle](https://www.kaggle.com/code).
2. **+ Add Input** → search "Credit Score Classification" (ParisRohan) → Add.
3. **Settings → Accelerator → GPU T4 ×2.**
4. Run All. `FAST_MODE = True` in the config cell gives a ~30–45 min smoke run; `False` runs the
   full search (~1.5–2.5 h, wall-clock bounded at every tuning stage so it cannot hang).

## Honest limitations

- The dataset is synthetic; results here do not transfer to real credit-risk decisioning.
- Class weighting trades probability calibration for balanced recall — the model's probabilities
  rank well (hence the ROC-AUC) but are not literally P(class | x) without post-hoc recalibration.
- `months_of_history` is the single strongest feature. It's legitimate — the month is known at
  prediction time — but it means part of the model's skill is "where in the sequence is this row,"
  not pure financial signal.
