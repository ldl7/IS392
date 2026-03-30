# Step 3 — Code Review Self-Check Checklist

**Project:** Predicting Federal Contract Outcomes Using NLP and Machine Learning
**Authors:** Leonel Lourenco, Rana Khan
**Course:** IS392 Section 452, NJIT
**Date:** 3/28/2026

---

## 1. Functionality

- [x] Code runs from start to finish without errors.
  - All 37 cells execute sequentially (execution counts 1-37) with no exceptions.
- [x] All required datasets/files are included or linked.
  - Raw Parquet shards in `exploring_data/` (12 files, 1.3 GB).
  - Interim checkpoint saved to `data/interim/filtered_physical_deliverables.parquet`.
  - Final labeled dataset saved to `data/processed/labeled_contracts.parquet`.
- [x] Outputs (tables, plots, metrics) appear as expected.
  - 9 figures saved to `figures/` and rendered inline.
  - Classification reports printed for 8 model-config combinations.
  - Performance comparison table (F1, AUC-ROC) displayed.
  - ROC curves plotted and saved.
- [x] Core parts of your planned analysis/modeling are implemented.
  - Data loading and schema discovery (Section 4).
  - PSC filtering to physical deliverables (Section 5).
  - Outcome label construction with adaptive threshold (Section 6).
  - Exploratory data analysis with 6 subsections (Section 7).
  - Two-track text preprocessing: LDA + TF-IDF (Section 8).
  - Topic modeling and TF-IDF feature extraction (Section 9).
  - Feature matrix construction with 4 configurations (Section 10).
  - Preliminary classification with Logistic Regression and Random Forest (Section 11).

## 2. Code Organization

- [x] Code is logically divided into sections (e.g., Data Loading, Cleaning, EDA, Modeling, Evaluation).
  - Notebook uses numbered sections 0-12: Header, Imports, Configuration, Helpers, Data Loading, Filtering, Labels, EDA, Text Preprocessing, Topic Modeling, Feature Matrix, Classification, Next Steps.
- [x] Variables and functions have clear, descriptive names.
  - Constants: `COST_OVERRUN_THRESHOLD`, `SAMPLE_CONTRACTS`, `LDA_NUM_TOPICS`, etc.
  - Functions: `is_physical_deliverable()`, `compute_cost_growth()`, `clean_text()`, etc.
  - DataFrames: `physical_contracts`, `sample_df`, `labeled`, `track_a_df`, etc.
- [x] No unused or redundant code remains.
  - All cells contribute to the pipeline. No orphaned variables or dead code paths.

## 3. Good Coding Practices

- [x] Consistent indentation and spacing.
  - Standard 4-space Python indentation throughout.
- [x] No excessively long lines of code.
  - Lines stay within reasonable length; long expressions are wrapped.
- [x] Functions are focused on one task each (if functions are used).
  - `is_physical_deliverable()` — PSC classification only.
  - `compute_cost_growth()` — cost delta only.
  - `compute_delay()` — date delta only.
  - `clean_text()` — LDA text preprocessing only.
  - `tfidf_tokenize()` — TF-IDF text cleaning only.
- [x] Repeated code has been refactored or put into functions.
  - Helper functions in Section 3 are reused across Sections 5-8.
  - Classification loop iterates over configs and models instead of duplicating code.
  - Label distribution plots in Section 7.1 use a loop over both targets for side-by-side comparison.

## 4. Documentation

- [x] Inline comments explain complex logic or unusual steps.
  - Every code cell has comments above key operations.
  - Adaptive threshold logic, PIID-group sampling, and feature matrix alignment are all explained inline.
- [x] A brief docstring or comment block at the top describes:
  - [x] Purpose of the script/notebook — Section 0 header cell.
  - [x] Dataset(s) used — Omari et al. FPDS dataset with Figshare DOI.
  - [x] Expected outputs — Listed in Section 0 (filtered dataset, labels, EDA, NLP features, classification results).
- [x] All five helper functions have full NumPy-style docstrings with Parameters, Returns, and description.

## 5. Submission Readiness

- [x] A short README file is included with:
  - [x] How to run the code — conda/pip setup instructions, `jupyter notebook` command.
  - [x] What the code currently accomplishes — Full pipeline from data loading through classification.
  - [x] Known issues or next steps — Sample size, LDA tuning, SMOTE, `late` target, GAO validation.
- [x] All group member names are listed in the README.
  - Leonel Lourenco and Rana Khan listed under "Authors".
- [x] The submission includes only necessary files (no personal or temporary files).
  - Submission files: `step3_code_review.ipynb`, `requirements.txt`, `README.md`, `architecture.md`, `data_dictionary.md`, `figures/` directory.

---

## Submission File Inventory

| File | Purpose |
|------|---------|
| `step3_code_review.ipynb` | Main notebook (67 cells, sections 0-12) |
| `requirements.txt` | 10 Python dependencies with minimum versions |
| `README.md` | Setup instructions, accomplishments, known issues |
| `architecture.md` | Project structure and data flow documentation |
| `data_dictionary.md` | Column definitions, functions, thresholds |
| `figures/*.png` | 9 saved visualizations from EDA and classification |
| `exploring_data/*.parquet` | 12 raw data shards (1.3 GB, from Figshare) |
| `data/processed/labeled_contracts.parquet` | Final labeled dataset output |
| `data/interim/filtered_physical_deliverables.parquet` | Filtering checkpoint |
| `scripts/fpds_filter_and_label.py` | Standalone pipeline script (reference) |
