# Project Architecture

## Overview

This project predicts whether U.S. federal government contracts for physical deliverables will experience cost overruns or schedule delays. It uses NLP (topic modeling via LDA) on contract description text combined with structured contract attributes to train binary classifiers (logistic regression, random forest).

The pipeline has four stages: data acquisition and filtering, exploratory data analysis, feature engineering and modeling, and evaluation. Each stage produces outputs that feed into the next.

---

## Directory Structure

```
federal-contract-prediction/
│
├── README.md                          # Project overview, setup instructions, how to run
├── requirements.txt                   # Python dependencies with minimum versions
├── architecture.md                    # This file. Describes project structure and data flow.
├── data_dictionary.md                 # Definitions of all variables, functions, columns, labels
│
├── step3_code_review.ipynb            # Main notebook: end-to-end pipeline for code review
│                                      #   (Sections 0-12: setup → filtering → labels → EDA →
│                                      #    text preprocessing → LDA/TF-IDF → classification)
│                                      #   NOTE: Replaces the planned notebooks/ directory.
│                                      #   All pipeline stages are consolidated into this single
│                                      #   notebook for the Step 3 code review submission.
│
├── docs/
│   ├── step1_topic_approval.docx      # Step 1 submission (topic and dataset approval)
│   ├── step2_writeup.docx             # Step 2 submission (preliminary write-up)
│   ├── setup_guide.docx               # Environment setup guide for team members
│   ├── gemini_batch_checklist.md       # Checklist for GAO PDF extraction via Gemini
│   └── ai_agent_task_plan.md          # Task plan for AI coding agent (FPDS pipeline)
│
├── exploring_data/                    # Parquet shards downloaded from Figshare (YYYYMM.parquet)
│   ├── 202212.parquet                 # ~105 MB per shard, 12 shards total (2022-2024)
│   ├── 202301.parquet
│   └── ...
│
├── data/
│   ├── raw/                           # Untouched downloads. Never modify files here.
│   │   └── gao_reports/               # GAO Weapon Systems Assessment PDFs (for Gemini task)
│   │       ├── gao_weapons_2025.pdf
│   │       ├── gao_weapons_2024.pdf
│   │       └── ...
│   │
│   ├── interim/                       # Intermediate outputs. Checkpoints between pipeline stages.
│   │   ├── filtered_physical_deliverables.parquet   # After PSC/NAICS filtering, before labels
│   │   ├── gao_chunks/                              # Chunked GAO PDFs for Gemini batch
│   │   └── gao_results/                             # Raw JSON responses from Gemini
│   │
│   └── processed/                     # Final clean datasets ready for modeling
│       ├── labeled_contracts.parquet   # Filtered + labeled contracts (primary dataset)
│       ├── labeled_contracts.csv       # Same data in CSV for quick inspection
│       ├── gao_validation_set.csv      # Extracted GAO cost/schedule data (1,059 programs)
│       ├── gao_fpds_linked.csv         # GAO-FPDS linked (140 programs matched to FPDS)
│       ├── gao_fpds_matches_detail.csv # Detailed match records
│       ├── validation_report.txt       # Cross-validation metrics (N=121 contracts)
│       └── dataset_summary.txt         # Metadata: row counts, class balance, quality stats
│
├── scripts/
│   ├── fpds_filter_and_label.py       # Standalone script version of filtering + label pipeline
│   ├── chunk_gao_pdfs_v2.py           # Chunks GAO PDFs into 80-page overlapping sections
│   ├── claude_batch_extract.py        # Sends GAO PDF chunks to Claude API, saves JSON
│   ├── test_claude_extract.py         # Smoke test for Claude API connectivity
│   ├── parse_gao_results.py           # Parses and merges Claude JSON into gao_validation_set.csv
│   ├── link_gao_to_fpds_v2.py         # Strict token matching for GAO-FPDS linking
│   └── gao_validation_analysis.py     # Standalone validation analysis with cross-validation metrics
│
├── figures/
│   ├── class_balance.png              # Bar chart of over_budget and late distributions
│   ├── cost_growth_distribution.png   # Histogram of cost_growth_pct
│   ├── delay_distribution.png         # Histogram of delay_days
│   ├── description_length_distribution.png  # Histogram of contract description text lengths
│   ├── overrun_rates_by_category.png  # Over-budget rate by PSC, agency, contract type
│   ├── correlation_heatmap.png        # Feature correlations with outcomes
│   ├── roc_curves_comparison.png      # ROC curves for all 4 feature configurations
│   ├── lda_coherence_scores.png       # Coherence by topic count (pending for final)
│   └── feature_importance.png         # Top features from random forest (pending for final)
│
├── models/                            # Saved trained models (optional, for reproducibility)
│   ├── lda_model.gensim              # Trained LDA model
│   ├── logreg_combined.pkl           # Logistic regression (combined features)
│   └── rf_combined.pkl               # Random forest (combined features)
│
├── column_names.txt                   # Full Parquet schema dump from Phase 2
│
└── .gitignore                         # Excludes data/raw/, models/, and large files from Git
```

---

## Data Flow

```
Figshare Parquet Shards (99M records, 470 columns)
        │
        ▼
[Phase 2: Schema Discovery]
        │  Identify exact column names
        ▼
[Phase 3: PSC/NAICS Filtering]
        │  Keep only physical deliverables (construction, defense, manufacturing)
        │  Output: data/interim/filtered_physical_deliverables.parquet
        ▼
[Phase 4: Label Construction]
        │  Group by PIID, compare initial vs final cost/dates
        │  Produce: over_budget (0/1), late (0/1), terminated_for_default (0/1)
        │  Output: data/processed/labeled_contracts.parquet
        ▼
[EDA — Section 7]                      [Text Preprocessing — Section 8]
        │  Descriptive stats                    │  Two-track approach:
        │  Distributions, correlations          │    Track A: Full NLP (>100 chars) → LDA
        │  Class balance check                  │    Track B: Lightweight clean (all) → TF-IDF
        ▼                                       ▼
[Topic Modeling — Section 9]
        │  Track A: LDA with Gensim (K=15, tuning pending)
        │  Track B: TF-IDF (max 5000 features)
        │  Extract topic proportion vectors + sparse TF-IDF matrix
        ▼
[Feature Matrix — Section 10]
        │  Merge text features + structured features
        │  Four configurations:
        │    Config A: structured only
        │    Config B: TF-IDF text features only
        │    Config C: combined (structured + TF-IDF)
        │    Config D: structured + LDA topics (Track A subset)
        ▼
[Classification — Notebook 06]
        │  Feature engineering:
        │    - Phase 2A: Temporal features (year, quarter, month, EOFY flag, duration, years_since_2015)
        │    - Phase 2B: Interaction features (size×mods, competition×size, EOFY×size)
        │    - Phase 2C: SelectKBest (f_classif) reduces TF-IDF 5000 → 100 for configs B/D
        │  Training:
        │    - Logistic regression + random forest on each (target, config)
        │    - Phase 1A: SMOTE with sampling_strategy=0.3 (partial balance)
        │    - class_weight='balanced' throughout
        │    - Stratified 80/20 train-test split
        ▼
[Evaluation — Notebook 06]
        │  Phase 1B: F1-optimal probability threshold from PR curve
        │  Phase 1C: 5-fold stratified cross-validation (mean±std)
        │  Phase 3A/3B: GridSearchCV hyperparameter tuning (LR + RF)
        │  Phase 3C: SMOTE variants comparison (SMOTE, BorderlineSMOTE, ADASYN)
        │  Outputs: results_comparison.csv, cv_results.csv, tuning_results.csv
        ▼
[GAO Validation — Notebook 07]
        │  1,059 programs extracted from 17 GAO reports (2003-2025)
        │  140 programs linked to FPDS; N=121 validation sample
        ▼
[Final Submission — Notebook 08]
        │  Research question summary + ROC curves + feature importance
        ▼
Final Report
```

---

## Supplementary Pipeline (Gemini/GAO Validation)

This runs in parallel with the main pipeline during Weeks 1-2 and feeds into evaluation in Weeks 7-8.

```
GAO Weapon Systems Annual Assessment PDFs (17 reports, 2003-2025)
        │
        ▼
[Chunk PDFs into 80-page overlapping sections]
        │  Output: data/interim/gao_chunks/
        ▼
[Claude API: Extract structured data]
        │  Send chunks with JSON extraction prompt
        │  Model: claude-3-haiku-20240307 (cost-optimized)
        │  Output: data/interim/gao_results/ (raw JSON)
        ▼
[Parse JSON into DataFrame]
        │  Output: data/processed/gao_validation_set.csv
        │  Records: 1,059 programs extracted
        ▼
[Token-match GAO programs to FPDS contracts]
        │  Strategy: distinctive token matching (acronyms + model numbers)
        │  Output: data/processed/gao_fpds_linked.csv
        │  Matches: 140 programs linked to FPDS
        ▼
[Validate classifier predictions against GAO ground truth]
        │  Sample: N=121 contracts with both FPDS labels and GAO extractions
        │  Metrics: cost agreement, schedule agreement
        ▼
[Feed into Evaluation - Notebook 07]
```

**GAO Validation Metrics:**
- **Reports processed:** 17 (2003-2025, all Weapon Systems Annual Assessments)
- **Programs extracted:** 1,059
- **FPDS linkages:** 140 programs matched
- **Validation sample:** N=121 contracts with GAO↔FPDS agreement
- **Cost agreement rate:** ~60% (within threshold tolerance)
- **Schedule agreement rate:** ~70%
- **Status:** Headline validation metric (N > 50 threshold met)

---

## Role Mapping

| Component | Owner | Weeks |
|-----------|-------|-------|
| Parquet download and schema discovery | Member 1 | 1-2 |
| PSC/NAICS filtering script | Member 1 | 1-2 |
| Label construction (cost/schedule) | Member 2 | 1-2 |
| GAO/Gemini extraction (supplementary) | Member 1 | 1-2 |
| Structured data EDA + visualizations | Member 1 | 3-4 |
| Text preprocessing pipeline | Member 2 | 3-4 |
| LDA topic modeling + tuning | Member 2 | 5-6 |
| Feature matrix construction | Member 2 | 5-6 |
| Classification training | Both | 5-6 |
| Evaluation metrics + comparison | Member 1 | 7-8 |
| Feature importance + interpretation | Member 2 | 7-8 |
| Final report | Both | 7-8 |

---

## Key Conventions

**Naming:** The consolidated notebook is `step3_code_review.ipynb` (sections numbered 0-12 internally). Scripts in `scripts/` are named by function. Data files use snake_case.

**Data immutability:** Files in `data/raw/` are never modified. All transformations produce new files in `data/interim/` or `data/processed/`.

**Checkpoints:** The pipeline saves intermediate outputs at each stage so any step can be re-run without reprocessing everything from scratch.

**Git:** The `.gitignore` should exclude `data/raw/`, `data/interim/`, `models/`, and any file over 50 MB. Only notebooks, scripts, docs, and processed CSVs (if small enough) go into version control.

**Reproducibility:** Each notebook starts with a cell that loads its input file and prints the shape and column list, so anyone can verify they have the right data before running.