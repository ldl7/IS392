# Data Pipeline Manifest

Generated: 2026-04-24

## Pipeline Overview

| Stage | Script/Notebook | Input | Output | Rows |
|-------|-----------------|-------|--------|------|
| 1. Filter & Label | `scripts/fpds_filter_and_label.py` | USAspending FPDS CSV | `labeled_contracts.parquet` | 45,456 |
| 2. Scale-up Label | `notebooks/04_scaleup_label.ipynb` | FPDS raw | `labeled_contracts.csv` | 45,456 |
| 3. Text Preprocessing | `notebooks/05_text_preprocessing_lda.ipynb` | `labeled_contracts.csv` | LDA + TF-IDF features | 9,498 (LDA), 45,456 (TF-IDF) |
| 4. Classification | `notebooks/06_classification.ipynb` | All above | Models + results | N=45,456 |
| 5. GAO Validation | `scripts/gao_validation_analysis.py` | GAO extracts + linked FPDS | Validation metrics | N=121 |

## Key Metrics

- **Total contracts:** 45,456
- **Cost overrun positive:** 4,725 (10.4%)
- **Delay positive:** 27,961 (61.5%)
- **Validation programs (GAO):** 121
- **LDA corpus (≥100 chars):** 9,498 (20.6%)

## File Inventory

### Processed Data (`data/processed/`)
| File | Size | Description |
|------|------|-------------|
| `labeled_contracts.csv` | 11.8 MB | Main labeled dataset |
| `doc_topic_matrix.parquet` | 1.2 MB | LDA topic vectors |
| `tfidf_matrix.npz` | 1.9 MB | Sparse TF-IDF matrix |
| `results_comparison.csv` | 1.3 KB | Classification metrics |
| `validation_report.txt` | 26.7 KB | GAO validation analysis |

### Models (`models/`)
| File | Size | Description |
|------|------|-------------|
| `lda_model.gensim` | 12.7 KB | Trained LDA (12 topics) |
| `*_RandomForest.pkl` | Various | RF models per config |
| `*_LogisticRegression.pkl` | Various | LR models per config |

## Reproducibility

To rebuild from scratch:

```bash
# 1. Download FPDS data (manual via USAspending)
# Place in data/raw/fpds/

# 2. Run filtering and labeling
python scripts/fpds_filter_and_label.py

# 3. Run notebooks in order
jupyter nbconvert --execute notebooks/04_scaleup_label.ipynb
jupyter nbconvert --execute notebooks/05_text_preprocessing_lda.ipynb
jupyter nbconvert --execute notebooks/06_classification.ipynb

# 4. Run GAO validation (after GAO extraction)
python scripts/gao_validation_analysis.py
```

## Known Limitations

1. **Description cutoff:** 100-character minimum for LDA (20.6% of data)
2. **Cost threshold:** 5% with adaptive fallback to 1% if positive class < 5%
3. **Class imbalance:** over_budget 10.4%, late 61.5%
4. **Validation N:** 121 (expanded from 17 via GAO 2003-2025)
