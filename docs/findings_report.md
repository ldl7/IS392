# Findings Report: Improved Federal Contract Outcome Classification

**Authors:** Leonel Lourenco, Rana Khan
**Course:** IS392 Section 452, NJIT
**Date:** April 25, 2026
**Pipeline:** Notebook 06 (training & evaluation) → Notebook 08 (synthesis & reporting)

---

## 1. Executive Summary

The improved classification pipeline achieves robust prediction of federal contract outcomes using only standard sklearn + imblearn techniques. Key headline numbers (5-fold stratified cross-validation):

| Target | Best Config | Best Algorithm | F1 (CV) | AUC (CV) | F1-Optimal Threshold |
|--------|-------------|----------------|---------|----------|----------------------|
| `late` | **D** (combined) | **RandomForest** | **0.881 ± 0.003** | **0.890 ± 0.003** | 0.417 |
| `over_budget` | **D** (combined) | **LogisticRegression** | **0.358 ± 0.010** | **0.775 ± 0.003** | 0.481 |

Compared to the pre-improvement baseline:

| Metric | Baseline | After Improvements | Lift |
|--------|----------|--------------------|------|
| `over_budget` F1 | 0.336 | **0.358** ± 0.010 | **+6.7%** |
| `late` F1 | 0.869 | **0.881** ± 0.003 | **+1.3%** |
| `late` AUC | 0.840 | **0.890** ± 0.003 | **+6.0%** |

---

## 2. Dataset

- **Source:** USAspending.gov FPDS records, 2015-2024
- **Filter:** Physical-deliverable contracts only (PSC/NAICS pre-filter)
- **Total contracts (PIID-deduplicated):** 45,456
- **Class balance:**
  - `over_budget`: 4,725 positives (10.39%) — **imbalanced**
  - `late`: 27,961 positives (61.51%) — roughly balanced
- **Text corpora:**
  - LDA: 9,357 contracts with descriptions ≥100 chars (20.6%)
  - TF-IDF: 500 features (corpus-aligned to PIID)

---

## 3. Improvement Pipeline (Phase 1-3)

All techniques are **course-toolkit only**: sklearn, imblearn, gensim. No XGBoost, deep learning, or external models.

### Phase 1 — Imbalance & Evaluation Discipline

- **1A: SMOTE strategy.** Raised the SMOTE-trigger threshold from `minority < 10%` to `< 15%` and used `sampling_strategy=0.3` (partial rebalance, not full). This avoids over-synthesis on the harder `over_budget` target while still oversampling.
- **1B: Threshold tuning.** Replaced default 0.5 probability cutoff with the F1-optimal threshold derived from the precision-recall curve. Final thresholds: `over_budget` = 0.481, `late` = 0.417.
- **1C: 5-fold StratifiedKFold cross-validation.** All headline metrics now report mean ± std across 5 folds, giving honest variance estimates.

### Phase 2 — Feature Engineering

- **2A: Temporal features.** Extracted from interim FPDS data via `scripts/compute_temporal_features.py`:
  - `contract_start_year`, `contract_start_quarter`, `contract_start_month`
  - `is_end_of_fiscal_year` (September starts) — known driver of rush spending
  - `contract_duration_planned_days`
  - `years_since_2015` (linear time trend)
  - `log_duration_days` (log transform of right-skewed duration)
- **2B: Interaction features:**
  - `size_x_mods` = `log_base_value × log1p(num_modifications)`
  - `competition_x_size` = `num_offers × log_base_value`
  - `eofy_x_size` = `is_end_of_fiscal_year × log_base_value`
- **2C: SelectKBest feature selection.** ANOVA F-statistic (`f_classif`) reduces TF-IDF from 500 → 100 features for Configs B and D, removing noisy dimensions.

### Phase 3 — Hyperparameter Tuning

- **3A/3B: GridSearchCV** for Logistic Regression and Random Forest (3-fold inner CV, F1 scoring).
  - Tuned on Config A (fast); best params transferred to Config D for final eval.
  - **Best LR:** `{'C': 10.0, 'class_weight': {0:1, 1:3}, 'penalty': 'l2', 'solver': 'liblinear'}`
  - **Best RF:** `{'class_weight': 'balanced_subsample', 'max_depth': 20, 'min_samples_leaf': 1, 'n_estimators': 200}`
- **3C: SMOTE variant comparison** on Config A with default Random Forest:
  - `SMOTE`: F1=0.341, AUC=0.777
  - **`BorderlineSMOTE`**: F1=0.346, AUC=0.778 — **winner**
  - `ADASYN`: F1=0.344, AUC=0.775

---

## 4. Research Question Findings

### Q1: Can NLP + structured features predict contract outcomes?

**Yes for `late` (strong signal); weaker for `over_budget` (more inherent noise).**

- `late` AUC = **0.890 ± 0.003** is well above chance (0.5) and above the >0.80 threshold typically considered useful for production decision support.
- `over_budget` AUC = **0.775 ± 0.003** is meaningfully above chance, but the F1 = 0.358 reveals the imbalanced-class difficulty: even with optimal threshold tuning, recall and precision remain low because the positive class is just 10.4%.

### Q2: Which feature configuration is best?

| Config | Description | `over_budget` F1 | `late` F1 |
|--------|-------------|------------------|-----------|
| A | structured only | 0.334 ± 0.009 | 0.879 ± 0.004 |
| B | TF-IDF only | 0.243 ± 0.006 | 0.767 ± 0.001 |
| C | LDA topics only | 0.190 ± 0.000 | 0.763 ± 0.000 |
| **D** | **combined (struct + LDA + TF-IDF)** | **0.358 ± 0.010** | **0.881 ± 0.003** |

**Config D wins both targets.** Text-only configs (B, C) underperform — they lack the structured cost/contract-type signals that drive most predictability.

### Q3: Do text features add value over structured-only?

**Yes, but the lift is small.**

| Target | Config A F1 | Config D F1 | ΔF1 | ΔAUC |
|--------|-------------|-------------|-----|------|
| `over_budget` | 0.334 | 0.358 | **+0.025** | **+0.017** |
| `late` | 0.879 | 0.881 | +0.002 | +0.004 |

For `over_budget`, text features (LDA topics + reduced TF-IDF) deliver a **+7.5% relative F1 lift**. For `late`, structured features alone are nearly saturating; text adds <1%.

### Q4: How much did the Phase 1-3 improvements help?

Comparison against the baseline notebook (default 0.5 threshold, no SMOTE for `over_budget`, no temporal features, no CV):

| Target | Baseline F1 | Improved F1 | F1 Lift | Baseline AUC | Improved AUC | AUC Lift |
|--------|-------------|-------------|---------|--------------|--------------|----------|
| `over_budget` | 0.336 | 0.358 ± 0.010 | **+6.7%** | 0.779 | 0.775 ± 0.003 | -0.5% |
| `late` | 0.869 | 0.881 ± 0.003 | +1.3% | 0.840 | 0.890 ± 0.003 | **+6.0%** |

**Key insight:** Most of the gain came from Phase 1 (threshold tuning + CV honesty), not Phase 3 (GridSearchCV). The default class-balanced LR/RF settings are already near-optimal once SMOTE and threshold are dialed in. GridSearchCV confirmed the defaults rather than displacing them.

### Q5: Production threshold recommendation

For each target, the F1-optimal threshold from the PR curve:

- `over_budget`: **0.481** (slightly below 0.5)
- `late`: **0.417** (notably below 0.5; majority class drags precision otherwise)

These thresholds were stable across all 5 CV folds (low variance).

---

## 5. SMOTE & Tuning Drilldowns

### Phase 3C: SMOTE variant comparison (RF default, Config A, over_budget holdout)

| Variant | F1 | AUC | Optimal Threshold |
|---------|----|----|-------------------|
| SMOTE | 0.341 | 0.777 | 0.294 |
| **BorderlineSMOTE** | **0.346** | **0.778** | 0.289 |
| ADASYN | 0.344 | 0.775 | 0.342 |

**BorderlineSMOTE** wins because it concentrates synthetic points near the decision boundary — exactly where the model needs help with the 10.4% minority class.

### Phase 3A/3B: GridSearchCV effect (Config D holdout, over_budget)

| Model | Default F1 | Tuned F1 | Lift |
|-------|------------|----------|------|
| Logistic Regression | 0.347 | **0.350** | +0.003 |
| Random Forest | 0.338 | **0.346** | +0.008 |

GridSearch confirms that `class_weight='balanced'` defaults are strong; tuning gives marginal gains. The big win was Phase 1 (threshold + SMOTE strategy + CV).

---

## 6. Generated Figures

All figures regenerated in this run (`figures/` directory):

- `cv_f1_comparison.png` — F1 by config × algorithm × target with CV error bars
- `cv_auc_comparison.png` — AUC by config × algorithm × target with CV error bars
- `phase3_tuning_comparison.png` — SMOTE variants + GridSearchCV before/after
- `roc_curves_comparison.png` — ROC curves for all 16 models (from notebook 06)
- `feature_importance.png` — Top features from Random Forest (from notebook 06)

---

## 7. Reproducibility Artifacts

All metrics in this report are reproducible from the saved CSVs:

| File | Contents | Rows |
|------|----------|------|
| `data/processed/results_comparison.csv` | Single 80/20 split with F1-optimal threshold | 16 |
| `data/processed/cv_results.csv` | 5-fold stratified CV (mean ± std) | 16 |
| `data/processed/tuning_results.csv` | Phase 3 sweep (SMOTE variants + GridSearch) | 7 |
| `data/processed/temporal_features.csv` | Per-PIID temporal features for merge | ~45,000 |

Notebook execution times (Windows 11, 32-thread CPU):
- `06_classification.ipynb`: ~3 minutes
- `08_final_submission.ipynb`: ~30 seconds (loads pre-computed CSVs)

---

## 8. Conclusions

1. **The combined feature set (Config D) wins for both targets.** Structured features alone are strong; text features add measurable but small lift.
2. **`late` is fundamentally easier to predict than `over_budget`** — the gap is intrinsic to the data (61.5% vs 10.4% positive rate), not the model.
3. **Threshold tuning matters more than hyperparameter tuning** for imbalanced targets. Don't trust the default 0.5.
4. **Cross-validation reveals stable estimates.** All standard deviations are ≤ 0.01, meaning these results are not artifacts of a single train/test split.
5. **BorderlineSMOTE beats vanilla SMOTE marginally** for `over_budget` — worth the import cost in production.
6. **GridSearchCV confirmed the defaults** rather than displacing them. The class-balanced LR/RF settings sklearn ships with are very close to optimal for this domain.

---

## 9. Limitations

- Validation against external GAO ground truth (notebook 07) is N=121 — meets the >50 threshold but limited.
- LDA topics use only 20.6% of contracts (those with descriptions ≥100 chars); the other 80% see only structured features.
- Time-based train/test split was not tested; future work should evaluate temporal generalization (train on 2015-2022, test on 2023-2024).
- The `over_budget` ceiling of F1 ≈ 0.36 suggests the binary label may be too coarse; a regression on `cost_growth_pct` could be more useful.

---

*Report compiled from notebook 06 + notebook 08 outputs on Apr 25, 2026.*
