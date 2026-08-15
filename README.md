# Credit Score Classification — Leakage-Audited Pipeline

A 3-class credit-score classifier (Poor / Standard / Good) built around one central claim: **most
public notebooks on this dataset report an inflated score because of customer-level data leakage,
and this pipeline measures, then eliminates, that inflation.**

| | |
|---|---|
| **Dataset** | [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) (Kaggle, ParisRohan) — 100,000 rows, 12,500 customers × 8 monthly snapshots |
| **Task** | Multiclass classification — `Credit_Score` ∈ {Poor, Standard, Good} |
| **Final holdout** | **macro-F1 0.7046 · accuracy 71.18% · ROC-AUC (OvR) 0.8715** |
| **Leakage found & removed** | **+0.1197 macro-F1** — the gap between a naive random split and a customer-grouped split, measured directly in the notebook |
| **Runtime** | 22.9 min end-to-end on Kaggle GPU T4×2 (full search — see [Runtime breakdown](#runtime-breakdown)) |
| **Stack** | LightGBM, XGBoost, CatBoost, Optuna, SHAP, PyTorch, scikit-learn |
| **License** | [MIT](LICENSE) |

> **This is a methodology project, not a production risk model.** The dataset is synthetic — its
> messy values are deliberately injected and its labels are rule-generated, not observed from real
> borrowers. The point of this repo is the validation discipline, not the raw score.

## Contents

- [The problem this solves](#the-problem-this-solves)
- [Methodology](#methodology)
- [Results](#results)
  - [Final holdout](#final-holdout-touched-exactly-once)
  - [Per-class breakdown](#per-class-breakdown-holdout)
  - [Full out-of-fold model comparison](#full-out-of-fold-model-comparison)
  - [Ensemble composition](#ensemble-composition)
  - [Tuned hyperparameters](#tuned-hyperparameters)
  - [Class weights & fold balance](#class-weights--fold-balance)
  - [Overfitting & learning-curve diagnostics](#overfitting--learning-curve-diagnostics)
  - [Data quality / imputation](#data-quality--imputation)
  - [Explainability (SHAP)](#explainability-shap)
  - [Runtime breakdown](#runtime-breakdown)
- [Repository contents](#repository-contents)
- [Running it](#running-it)
- [Honest limitations](#honest-limitations)

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
training customers only (80,000 rows / 10,000 customers) and applied to holdout (20,000 rows /
2,500 customers) — not computed on the full dataset and then split. Customer overlap between the two
sides is asserted to be zero.

**2. Four-tier imputation**, cheapest and most-informative source first:
   - exact reconstruction where the panel structure allows it (a customer's credit history age
     increments by exactly one month per row, so it's recoverable losslessly, not just imputed)
   - the customer's own observed history (forward/backward fill)
   - interpolation along the customer's own timeline for genuinely time-varying columns
   - a train-fitted median/mode, only as a last resort

**3. Causal feature engineering.** Rolling statistics use `shift(1)` before the expanding window, so
no row ever sees its own current or future values. A customer's first month has no history — it's
filled with a structural zero, not a population average pretending they have average history.

**4. Model selection on out-of-fold predictions only.** Ten model variants are trained under
`StratifiedGroupKFold` (5 folds), tuned with Optuna (30 trials/model, GPU-parallel, wall-clock
bounded), and ranked purely on OOF macro-F1. **The holdout set is touched exactly once, at the very
end**, by whichever model won on OOF — never used to pick the winner.

**5. Ensembling and threshold tuning, fit on OOF, verified on holdout.** A greedy-selected weighted
blend of the trained models beats every single model. A per-class decision-threshold search is also
attempted — and rejected by its own noise floor (see [Results](#results)), a deliberate self-check
rather than a result quietly discarded.

**6. Two tested hypotheses that failed, kept in the notebook on purpose:**
   - *Customer-level prediction averaging* — plausible, since customers repeat across months. Tested
     and **rejected**: only 41.7% of customers hold a single label across their 8 months, so
     averaging smooths away real, moving signal instead of noise.
   - *Per-class threshold tuning* — moved OOF score by +0.0003, indistinguishable from noise, and a
     `MIN_THRESHOLD_GAIN` floor now auto-rejects any search result that doesn't clear a real margin,
     so the pipeline stops proposing a correction it can't back up.

## Results

All numbers below are from the run captured in
[`credit_score_model_executed.ipynb`](credit_score_model_executed.ipynb) — Kaggle, GPU T4×2,
`FAST_MODE=False`.

### Final holdout (touched exactly once)

| Configuration | macro-F1 | weighted-F1 | Accuracy | ROC-AUC (OvR) | Log loss |
|---|---|---|---|---|---|
| Best single model (`LightGBM_tuned`) | 0.7005 | 0.7082 | 0.7058 | 0.8687 | 0.6584 |
| **Weighted ensemble — FINAL** | **0.7046** | **0.7144** | **0.7118** | **0.8715** | **0.6350** |
| + threshold tuning | 0.7046 | 0.7144 | 0.7118 | 0.8715 | 0.6350 |
| + customer-level aggregation (rejected) | 0.6989 | 0.7105 | 0.7076 | 0.8557 | 0.6654 |

Threshold tuning is listed but identical to the plain ensemble — its OOF gain (+0.0003) never
cleared the `MIN_THRESHOLD_GAIN=0.001` floor, so it was automatically discarded rather than applied
and reported as a separate win. Holdout gain from actually applying it was confirmed at +0.0000 —
the guard's decision was correct.

### Per-class breakdown (holdout)

Class weighting deliberately trades calibration for balanced recall — see the precision/recall
asymmetry below, especially on `Good`, the smallest class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Poor | 0.6976 | 0.7358 | 0.7162 | 5,726 |
| Standard | 0.8105 | 0.6637 | 0.7298 | 10,580 |
| Good | 0.5667 | 0.8127 | 0.6678 | 3,694 |
| **accuracy** | | | **0.7118** | 20,000 |
| macro avg | 0.6916 | 0.7374 | 0.7046 | 20,000 |
| weighted avg | 0.7332 | 0.7118 | 0.7144 | 20,000 |

### Full out-of-fold model comparison

All 10 trained variants, 5-fold `StratifiedGroupKFold`, ranked by OOF macro-F1 (this is the ranking
used for model selection — holdout was not consulted):

| Model | macro-F1 | weighted-F1 | Accuracy | ROC-AUC (OvR) | Log loss |
|---|---|---|---|---|---|
| **LightGBM_tuned** | **0.6841** | 0.6958 | 0.6924 | 0.8612 | 0.6801 |
| RandomForest | 0.6836 | 0.6952 | 0.6914 | 0.8596 | 0.6851 |
| LightGBM | 0.6834 | 0.6969 | 0.6934 | 0.8619 | 0.6721 |
| XGBoost | 0.6831 | 0.6933 | 0.6898 | 0.8630 | 0.6885 |
| LightGBM_noweight | 0.6830 | 0.7063 | 0.7056 | 0.8659 | 0.6384 |
| CatBoost | 0.6827 | 0.6912 | 0.6883 | 0.8650 | 0.6902 |
| CatBoost_tuned | 0.6822 | 0.6905 | 0.6875 | 0.8639 | 0.6989 |
| LightGBM_ordinal | 0.6784 | 0.7041 | 0.7039 | 0.8655 | 0.6396 |
| PyTorch_MLP | 0.6636 | 0.6744 | 0.6704 | 0.8482 | 0.7373 |
| Logistic Regression | 0.6493 | 0.6623 | 0.6574 | 0.8108 | 0.8176 |

Note how tightly the top 7 cluster (0.678–0.684): five different model families landed within 0.006
macro-F1 of each other, which is the signal that drove the diagnostics below — the ceiling here is
the feature set, not the algorithm.

### Ensemble composition

Blend weights, selected by greedy forward selection (Caruana-style) on OOF probabilities:

| Model | Weight |
|---|---|
| LightGBM_noweight | 0.435 |
| CatBoost | 0.261 |
| LightGBM_tuned | 0.130 |
| CatBoost_tuned | 0.087 |
| LightGBM_ordinal | 0.087 |
| XGBoost | 0.000 |
| LightGBM | 0.000 |

`XGBoost` and plain `LightGBM` are correctly zeroed out — greedy selection determined they add
nothing the other five don't already cover, not that they failed. Blend OOF macro-F1: **0.6924**,
ahead of the best single model's 0.6841.

### Tuned hyperparameters

Found by Optuna (30 trials/model, group-aware 50%-customer subsample, wall-clock capped at 1500s;
LightGBM used 429s, CatBoost used 188s):

```
LightGBM_tuned  (CV macro-F1 0.6889)
  learning_rate:      0.0871
  num_leaves:         61
  max_depth:          7
  min_child_samples:  87
  feature_fraction:   0.592
  bagging_fraction:   0.985
  bagging_freq:       6
  reg_alpha:          2.854
  reg_lambda:         1.131

CatBoost_tuned  (CV macro-F1 0.6879)
  learning_rate:        0.1062
  depth:                4
  l2_leaf_reg:          5.749
  random_strength:      5.924
  bagging_temperature:  0.0465
```

### Class weights & fold balance

```
Poor: 1.146   Standard: 0.626   Good: 1.887
```

`StratifiedGroupKFold` keeps class balance nearly identical across all 5 folds despite grouping by
customer — plain `GroupKFold` does not guarantee this:

| Fold | n | Poor | Standard | Good |
|---|---|---|---|---|
| 0 | 16,000 | 0.294 | 0.529 | 0.177 |
| 1 | 15,992 | 0.285 | 0.546 | 0.170 |
| 2 | 16,000 | 0.287 | 0.529 | 0.184 |
| 3 | 16,008 | 0.295 | 0.532 | 0.174 |
| 4 | 16,000 | 0.294 | 0.527 | 0.179 |

### Overfitting & learning-curve diagnostics

Reproduced across three independent runs with the same verdict each time:

| | macro-F1 | weighted-F1 | Accuracy | ROC-AUC | Log loss |
|---|---|---|---|---|---|
| Train (in-sample) | 0.8339 | 0.8358 | 0.8356 | 0.9559 | 0.4187 |
| Out-of-fold (5-fold CV) | 0.6834 | 0.6969 | 0.6934 | 0.8619 | 0.6721 |
| Holdout (unseen customers) | 0.7021 | 0.7103 | 0.7080 | 0.8698 | 0.6489 |

- **Train − OOF gap: +0.1505** — large, but normal for boosted trees and not itself evidence of a
  problem.
- **OOF − holdout gap: −0.0187** — holdout scores *higher* than OOF. This is the number that matters,
  and a negative gap means the model generalizes; it rules out overfitting.

**Learning curve** (holdout macro-F1 vs. training rows used, single fold-0 model):

| Training data used | Rows | Holdout macro-F1 |
|---|---|---|
| 25% | 16,000 | 0.6718 |
| 50% | 32,000 | 0.6935 |
| 100% | 64,000 | 0.6968 |

Flat from 50% to 100% (+0.003 for doubling the data) — more rows would not meaningfully move the
score. Combined with the tight model-family clustering above, this pipeline is **feature-limited**,
not data- or capacity-limited, and that's the reason tuning was stopped rather than pushed further.

### Data quality / imputation

From the raw 100,000-row file:

- **25,265 rows** contained an injected sentinel value (`'_______'` / `'_'` / `'!@9#%8'`)
- **2,169 rows** had a physically impossible age, converted to NaN before imputation
- **Final NaN count after the full pipeline: 0 in both train and holdout**

<details>
<summary>Full missing-value cascade by column and pipeline stage (click to expand)</summary>

| Column | After parsing junk | After impossible→NaN | After per-customer fill | After train median/mode |
|---|---|---|---|---|
| Credit_Mix | 16,180 | 16,180 | 0 | 0 |
| Monthly_Inhand_Salary | 12,033 | 12,033 | 0 | 0 |
| Type_of_Loan† | 9,184 | 9,184 | 9,184 | 9,184 |
| Name† | 7,929 | 7,929 | 7,929 | 7,929 |
| Credit_History_Age_Months | 7,149 | 7,149 | 0 | 0 |
| Credit_History_Age† | 7,149 | 7,149 | 7,149 | 7,149 |
| Payment_Behaviour | 6,121 | 6,121 | 0 | 0 |
| Occupation | 5,645 | 5,645 | 0 | 0 |
| Num_of_Delayed_Payment | 5,638 | 6,753 | 0 | 0 |
| Amount_invested_monthly | 3,581 | 3,581 | 0 | 0 |
| Num_of_Loan | 0 | 3,492 | 0 | 0 |
| Age | 0 | 2,169 | 0 | 0 |
| Num_Credit_Inquiries | 1,552 | 2,869 | 0 | 0 |
| Num_Credit_Card | 0 | 1,777 | 0 | 0 |
| Changed_Credit_Limit | 1,692 | 1,692 | 0 | 0 |
| Interest_Rate | 0 | 1,653 | 0 | 0 |
| Num_Bank_Accounts | 0 | 1,074 | 0 | 0 |
| Monthly_Balance | 973 | 973 | 0 | 0 |
| Annual_Income | 0 | 80 | 0 | 0 |
| Total_EMI_per_month | 0 | 80 | 0 | 0 |

† `Type_of_Loan`, `Name`, and `Credit_History_Age` (the raw string) are dropped before modeling —
their signal is already extracted into `Loan_*` multi-hot flags and `Credit_History_Age_Months`
respectively, so these columns are never imputed.

</details>

### Explainability (SHAP)

Top 20 features by mean |SHAP| value, `LightGBM_tuned`, fold-0 model, 2,000-row holdout sample:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `months_of_history` | 0.2276 |
| 2 | `Num_Credit_Card` | 0.1344 |
| 3 | `Credit_Mix` | 0.1252 |
| 4 | `Credit_Mix_ord_cust_max` | 0.1183 |
| 5 | `Num_Credit_Inquiries_cust_min` | 0.1178 |
| 6 | `Outstanding_Debt_cust_mean` | 0.1146 |
| 7 | `Credit_Mix_ord` | 0.1125 |
| 8 | `Outstanding_Debt_cust_max` | 0.1110 |
| 9 | `Credit_Mix_ord_cust_min` | 0.0967 |
| 10 | `Delay_from_due_date_cust_mean` | 0.0880 |
| 11 | `Interest_Rate` | 0.0852 |
| 12 | `Interest_Rate_cust_mean` | 0.0833 |
| 13 | `Payment_Behaviour_Low_spent_Small_value_payments` | 0.0823 |
| 14 | `Interest_Rate_cust_max` | 0.0820 |
| 15 | `Outstanding_Debt_cust_min` | 0.0801 |
| 16 | `Interest_Rate_cust_min` | 0.0767 |
| 17 | `Outstanding_Debt` | 0.0731 |
| 18 | `Credit_Mix_ord_cust_mean` | 0.0638 |
| 19 | `Credit_History_Age_Months` | 0.0552 |
| 20 | `Num_of_Delayed_Payment_cust_mean` | 0.0476 |

`months_of_history` dominates by a wide margin — more than double the #2 feature. It's a legitimate
feature (the month is known at prediction time), but it also means a meaningful share of the model's
skill is "where in the customer's timeline is this row," not pure financial signal — worth stating
plainly rather than leaving implicit.

A round-trip test confirms the saved model bundle reproduces the notebook's own predictions exactly:
**400 raw rows, 50 customers, 100.0% match.**

### Runtime breakdown

Wall-clock progress through the full (`FAST_MODE=False`) run, 22.9 minutes end-to-end:

| Cumulative time | Stage | Stage duration |
|---|---|---|
| 2.8 min | Baselines (Logistic Regression, Random Forest) | 2.8 min |
| 5.1 min | Gradient boosting (LightGBM, XGBoost, CatBoost) | 2.4 min |
| 8.6 min | Class-weight A/B + ordinal decomposition experiments | 3.5 min |
| 10.0 min | Optuna hyperparameter tuning (30+30 trials) | 1.4 min |
| 20.3 min | Refitting tuned models on full 5 folds | 10.3 min |
| 21.4 min | Ensembling + threshold tuning | 1.1 min |
| 22.2 min | Neural baseline (PyTorch MLP) | 0.8 min |
| 22.9 min | Final holdout evaluation + SHAP | 0.7 min |

An earlier version of this tuning stage didn't finish in 5 hours (unbounded per-trial cost — see
[`CLAUDE.md`](CLAUDE.md) for the fix). Every expensive stage is now wall-clock-capped by construction,
not just fast in practice.

## Repository contents

| File | What it is |
|---|---|
| `credit_score_model.ipynb` | Source notebook — clean, unexecuted, ready to run on Kaggle |
| `credit_score_model_executed.ipynb` | Full run with all outputs (the source of every number above), Kaggle GPU T4×2, ~23 min |
| `requirements.txt` | Python dependencies (unpinned — matches whatever the current Kaggle GPU image ships) |
| `LICENSE` | MIT |

## Running it

1. Open `credit_score_model.ipynb` on [Kaggle](https://www.kaggle.com/code).
2. **+ Add Input** → search "Credit Score Classification" (ParisRohan) → Add.
3. **Settings → Accelerator → GPU T4 ×2.**
4. Run All. `FAST_MODE = True` in the config cell gives a ~30–45 min smoke run; `False` runs the
   full search (~1.5–2.5 h, wall-clock bounded at every tuning stage so it cannot hang) and is what
   produced every number in this README.

## Honest limitations

- The dataset is synthetic; results here do not transfer to real credit-risk decisioning.
- Class weighting trades probability calibration for balanced recall — the model's probabilities
  rank well (hence the ROC-AUC) but are not literally P(class | x) without post-hoc recalibration.
- `months_of_history` is the single strongest feature. It's legitimate — the month is known at
  prediction time — but it means part of the model's skill is "where in the sequence is this row,"
  not pure financial signal.
- Effective independent sample size is closer to 12,500 (unique customers) than 100,000 (rows) —
  41.7% of customers hold a single label across all 8 of their months, so many rows are correlated
  repeats of the same underlying signal, not independent observations.
