# Predicting Federal Contract Outcomes

## Authors
- Leonel Lourenco
- Rana Khan

## Course
IS392 Section 452, New Jersey Institute of Technology

## Overview
This project predicts whether U.S. federal government contracts for physical deliverables (construction, defense, manufacturing) will experience cost overruns or schedule delays. It uses NLP topic modeling (LDA) on contract description text combined with structured contract attributes to train binary classifiers (logistic regression, random forest).

## Dataset

USAspending.gov Federal Contract Data
- Source: https://www.usaspending.gov/download_center/custom_award_data
- Format: CSV files (transaction-level, one row per contract modification)
- Pre-filtered: Contracts ≥$500K with PSC codes indicating physical deliverables

**Alternative Dataset (Original):**
Omari et al. Comprehensive Federal Procurement Dataset (1979-2023)
- Source: https://doi.org/10.6084/m9.figshare.28057043
- Format: Parquet shards (for reference, not currently used)

## How to Run

### Prerequisites
1. Install Anaconda: https://www.anaconda.com/download
2. Create the environment:
   ```bash
   conda create -n contracts python=3.11 -y
   conda activate contracts
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

### Data Setup
1. Go to https://www.usaspending.gov/download_center/custom_award_data
2. Select filters:
   - **Award Type:** Contracts
   - **Date Range:** 2015-2024
   - **Agency:** All
   - **Location:** US states only
   - **Recipient:** All
3. Click "Download" to get CSV zip file
4. Extract CSVs to `data/raw/usaspending/`

**Pre-filtering Recommended:**
- Contract value ≥ $500,000
- PSC codes: Y (construction), Z (maintenance), or numeric 10-99 (supplies)

**For GAO Validation:**
- Download GAO Weapon Systems Annual Assessment PDFs (2003-2025)
- Place in `data/raw/gao_reports/`

### Running the Pipeline
```bash
conda activate contracts

# Option 1: Run notebooks individually
jupyter notebook notebooks/04_scaleup_label.ipynb       # Label construction
jupyter notebook notebooks/05_text_preprocessing_lda.ipynb  # NLP features
jupyter notebook notebooks/06_classification.ipynb       # Train models
jupyter notebook notebooks/07_gao_validation.ipynb      # Validate vs GAO

# Option 2: Run all via command line
jupyter nbconvert --execute notebooks/04_scaleup_label.ipynb
jupyter nbconvert --execute notebooks/05_text_preprocessing_lda.ipynb
jupyter nbconvert --execute notebooks/06_classification.ipynb
```

## What the Code Accomplishes

**Dataset:** 45,456 federal contracts (physical deliverables only)
- Cost overrun rate: 10.4% (4,725 contracts)
- Schedule delay rate: 61.5% (27,961 contracts)

**Pipeline:**
1. **Label Construction** (Notebook 04): PIID-grouped cost growth and delay labels with 5% adaptive threshold
2. **Text Preprocessing** (Notebook 05): Two-track approach
   - **Track A (LDA):** 9,498 descriptions ≥100 chars → 12 LDA topics
   - **Track B (TF-IDF):** All 45,456 descriptions → 5,000 features
3. **Classification** (Notebook 06): 4 configs × 2 algorithms × 2 targets = 16 models
   - **Best for `late`:** Config D (RF) — F1=0.871, AUC=0.852
   - **Best for `over_budget`:** Config A (LR) — F1=0.298, AUC=0.730
4. **GAO Validation** (Notebook 07): Independent ground truth check
   - 17 GAO reports processed (2003-2025)
   - 1,059 programs extracted, 140 linked to FPDS
   - Validation sample: N=121 contracts

## Technical Notes

- **Cost overrun threshold:** 5% primary, adaptive fallback to 1% if positive class < 5%
- **Description cutoff:** LDA requires ≥100 characters (20.6% of data); TF-IDF has no minimum
- **Class imbalance:** `over_budget` 10.4% (no SMOTE needed); `late` 61.5% (no SMOTE needed)
- **Validation:** N=121 (expanded from 17 via GAO 2003-2025 expansion) meets >50 threshold for headline metric
- **LDA:** K=12 topics selected via coherence tuning (C_v metric)
- **Text features:** TF-IDF adds minimal value; structured features dominate performance

## Project Structure
See `architecture.md` for full directory layout and `MANIFEST.md` for data file inventory.
See `data_dictionary.md` for column definitions and feature descriptions.
