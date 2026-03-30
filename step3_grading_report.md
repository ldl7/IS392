# Step 3 — Code Review Grading Report

**Project:** Predicting Federal Contract Outcomes Using NLP and Machine Learning
**Authors:** Leonel Lourenco, Rana Khan
**Course:** IS392 Section 452, NJIT
**Graded Against:** Step 3 Code Review Rubric (20 pts)

---

## Overall Score: 20 / 20

---

## 1. Functionality & Progress Toward Final Analysis — 6 / 6

**Evidence:**

The notebook (`step3_code_review.ipynb`) executes all 37 code cells from start to finish
without errors (verified via sequential execution counts 1-37 with timestamps from a
single run on 2026-03-29).

The pipeline implements **every major stage** of the planned methodology:

| Stage | Section | Status |
|-------|---------|--------|
| Data import & schema discovery | 4 | Complete — 12 Parquet shards (1.3 GB), 470-column schema inspected |
| Filtering | 5 | Complete — PSC filter retains 81.2% (4.3M rows), checkpoint saved |
| Sampling | 5 | Complete — 50K contracts sampled by PIID group (52,393 rows) |
| Label construction | 6 | Complete — `over_budget`, `late`, `terminated_for_default` with adaptive threshold |
| Exploratory data analysis | 7 | Complete — 6 subsections, 9 figures saved + inline |
| Text preprocessing (two-track) | 8 | Complete — Track A (LDA): 4,992 docs; Track B (TF-IDF): 48,264 docs |
| Topic modeling + TF-IDF | 9 | Complete — LDA K=15, TF-IDF 5,000 features |
| Feature matrix | 10 | Complete — 4 configurations (Structured, TF-IDF, Combined, Struct+LDA) |
| Classification | 11 | Complete — LogReg + RF on all 4 configs, reports + ROC curves |

This goes well beyond "at least part of your planned analysis/modeling."

**Remaining work (documented in Section 12):**
- Classify `late` target (only `over_budget` done so far)
- Tune LDA topic count via coherence scores
- Apply SMOTE for class imbalance
- Feature importance analysis
- Full-dataset run (currently 50K-contract sample)

These are appropriate next steps for a mid-project code review.

---

## 2. Code Clarity & Organization — 5 / 5

**Evidence:**
- Notebook is divided into **13 clearly numbered sections** (0-12), each with a
  markdown header cell explaining its purpose.
- Variable names are descriptive: `physical_contracts`, `COST_OVERRUN_THRESHOLD`,
  `track_a_df`, `labeled`, `config_C`, `y_budget`.
- Function names are self-documenting: `is_physical_deliverable()`,
  `compute_cost_growth()`, `tfidf_tokenize()`.
- All 5 helper functions in Section 3 are called downstream — no dead code.
- Classification code uses a loop over configs and models instead of duplicating blocks.
- Section 7.1 uses a loop over both targets for side-by-side comparison.
- Constants are centralized in Section 2 (Configuration), never hardcoded elsewhere.

---

## 3. Use of Good Coding Practices — 4 / 4

**Evidence:**
- **Consistent formatting:** Standard 4-space Python indentation throughout all 37
  code cells.
- **Line length:** Long expressions are wrapped appropriately (e.g., f-strings,
  function calls).
- **Single-responsibility functions:** All 5 helper functions perform exactly one task
  (PSC check, cost delta, date delta, LDA tokenization, TF-IDF tokenization).
- **Refactoring:** Shard processing uses a loop, classification uses a loop over
  configs/models, column renaming uses a dictionary map.
- **Requirements file:** `requirements.txt` lists all 10 dependencies with minimum
  version pins (`>=`).
- **Reproducibility:** `RANDOM_STATE = 42` used in sampling, LDA training,
  train-test split, and both classifiers.
- **Error handling:** try/except around shard reading, NLTK data downloads, and
  spaCy model loading.
- **Class weighting:** `class_weight='balanced'` applied to both LogReg and RF to
  address known class imbalance.

---

## 4. Documentation & Comments — 4 / 4

**Evidence:**

**Header block (Section 0):**
- Title, authors, course, institution, date
- Purpose statement explaining the methodology
- Dataset citation with DOI (Omari et al., Figshare)
- Expected outputs list (6 items)

**Docstrings:**
- All 5 helper functions have full **NumPy-style docstrings** with Parameters,
  Returns, and description sections.

**Inline comments:**
- Every code cell contains comments above key operations.
- Complex logic is explained: adaptive threshold reasoning, PIID-group sampling
  rationale, two-track NLP justification, feature matrix alignment.

**Markdown cells:**
- Each of the 6 EDA subsections has a markdown header above and an
  **Interpretation** cell below explaining findings.
- Sections 8 (Text Preprocessing) and 9 (Topic Modeling) have explanatory markdown.
- Adaptive threshold logic has its own dedicated markdown cell.
- Section 11 ends with a "Preliminary Observations" analysis cell.
- Section 12 lists concrete next steps.

**Supporting documentation:**
- `architecture.md` — project structure and data flow
- `data_dictionary.md` — all columns, functions, thresholds, file formats

---

## 5. Submission Completeness — 1 / 1

**Required deliverables:**

| Required | File | Present |
|----------|------|---------|
| Code file (.ipynb) | `step3_code_review.ipynb` | Yes |
| Requirements file | `requirements.txt` | Yes |
| README | `README.md` | Yes |
| Supporting data | `exploring_data/*.parquet` (12 shards) | Yes |
| Output figures | `figures/*.png` (9 files) | Yes |

**README contents:**
- How to run: conda setup, pip install, spaCy/NLTK downloads, jupyter command — **Yes**
- What code accomplishes: Full pipeline description with bullet list — **Yes**
- Known issues/next steps: 7 items listed (sample size, threshold, two-track, LDA tuning, late target, SMOTE, GAO) — **Yes**
- Group member names: Leonel Lourenco, Rana Khan — **Yes**

No personal, temporary, or unnecessary files are included in the project root.

---

## Summary

| Criteria | Max | Score | Notes |
|----------|-----|-------|-------|
| Functionality & progress | 6 | **6** | Full pipeline: data → labels → EDA → NLP → classification |
| Code clarity & organization | 5 | **5** | Excellent structure; all functions used, no dead code |
| Good coding practices | 4 | **4** | Constants centralized, functions focused, reproducible, error handling |
| Documentation & comments | 4 | **4** | NumPy docstrings, inline comments, markdown interpretations, supporting docs |
| Submission completeness | 1 | **1** | All files present, README complete |
| **Total** | **20** | **20** | |

---

## Recommendations for Final Submission

1. **Run classification on the `late` target** — Currently only `over_budget` is
   classified. Adding `late` doubles the analysis scope.
2. **Apply SMOTE** — The `over_budget` positive class is only 0.06% (27 out of 48K).
   Even with `class_weight='balanced'`, F1 scores are very low. SMOTE on the training
   set would provide more meaningful evaluation.
3. **Tune LDA topic count** — Run coherence scores across K=10-30 to find the optimal
   number of topics. K=15 is a reasonable starting point but may not be optimal.
4. **Add cross-validation** — Replace the single train-test split with 5-fold
   stratified CV for more robust performance estimates.
5. **Run on full dataset** — The 50K-contract sample is appropriate for the code
   review but the final submission should use the full 3.88M labeled contracts.
