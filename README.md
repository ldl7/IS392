# Predicting Federal Contract Outcomes

## Authors
- Leonel Lourenco
- Rana Khan

## Course
IS392 Section 452, New Jersey Institute of Technology

## Overview
This project predicts whether U.S. federal government contracts for physical deliverables (construction, defense, manufacturing) will experience cost overruns or schedule delays. It uses NLP topic modeling (LDA) on contract description text combined with structured contract attributes to train binary classifiers (logistic regression, random forest).

## Dataset
Omari et al. Comprehensive Federal Procurement Dataset (1979-2023)
- Source: https://doi.org/10.6084/m9.figshare.28057043
- License: CC0 (public domain)
- Format: Parquet shards

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
1. Download Parquet shards from the Figshare link above
2. Place them in `exploring_data/`

### Running the Notebook
```bash
conda activate contracts
jupyter notebook
```
Open `step3_code_review.ipynb` and run all cells (Cell > Run All).

## What the Code Currently Accomplishes
- Loads and explores FPDS Parquet data schema
- Filters to completed physical-deliverable contracts using PSC codes
- Constructs binary outcome labels (over_budget, late) by comparing initial vs final contract values and dates
- Performs exploratory data analysis with 6+ visualizations
- Preprocesses contract description text for NLP using a two-track approach:
  - **Track A (LDA):** Full tokenization, stop-word removal, and lemmatization on descriptions > 100 characters
  - **Track B (TF-IDF):** Lightweight text cleaning on all descriptions
- Trains a preliminary LDA topic model and extracts topic features
- Builds four feature configurations (structured-only, TF-IDF-only, combined structured+TF-IDF, and structured+LDA on the LDA subset)
- Trains and compares logistic regression and random forest classifiers
- Produces a baseline performance comparison table and ROC curves

## Known Issues and Next Steps
- **Sample size:** The notebook runs on a 50,000-contract sample (complete PIID groups) for speed. Full dataset run is pending for the final submission.
- **Adaptive threshold:** The cost overrun threshold starts at 5% and drops to 1% if the minority class is too small. This is documented in the notebook.
- **Two-track text approach:** Because 78% of descriptions are under 50 characters, LDA is applied only to longer descriptions (Track A) while TF-IDF covers all records (Track B).
- LDA topic count (K=15) is a preliminary choice; coherence score tuning across 10-30 topics is pending
- Classification has only been run on the over_budget target; the late target is pending
- SMOTE has not been applied yet; will be added if class imbalance is confirmed during evaluation
- GAO validation dataset (Gemini extraction) is in progress and not yet integrated
- Feature importance analysis and final interpretation are pending

## Project Structure
See `architecture.md` for the full directory layout and data flow diagram.
See `data_dictionary.md` for definitions of all variables, columns, and functions.
