# Master Completion Plan: Step 3 Through Final Submission

## Overview

This document lays out every remaining piece of work needed to finish the project, from the current Step 3 code review checkpoint through the final deliverable. It consolidates all earlier plans (agent task plan, Gemini GAO extraction, Member 2's text preprocessing and LDA work, final modeling, and report writing) into one reference so nothing gets lost.

The project runs on an 8-week timeline starting from the end of Week 2. Current status is roughly 35-40% complete, with most planning done but the majority of code execution still ahead.

---

## Table of Contents

1. Step 3: Code Review Checkpoint (Weeks 3-4)
2. Member 2's Work: Text Preprocessing and LDA (Weeks 3-5)
3. Supplementary Track: Gemini GAO Extraction (Weeks 3-6, parallel)
4. Classification and Evaluation (Weeks 6-7)
5. Final Submission: Report and Deliverables (Week 8)
6. Quality Checks and Final Review

---

## 1. Step 3: Code Review Checkpoint (Weeks 3-4)

This is the 20% code review deliverable. The grader is checking that the notebook runs cleanly, the methodology is sound, the code is well-organized, and there's meaningful progress. The goal is a single `.ipynb` that runs start to finish without errors, plus a `README.md` and `requirements.txt`.

### 1.1 Fix the Sampling Bug (Critical, Day 1)

The agent discovered the 100K sample was drawn from raw modification rows before label construction, which broke multi-modification contracts and produced only 36 over_budget and 40 late positives instead of the expected ~92 and ~1,200.

- Implement Option C: sample complete PIID groups before label construction
- Set `SAMPLE_CONTRACTS = 50_000` (accounting for multiple modifications per contract)
- Use `np.random.RandomState(RANDOM_STATE).choice()` to pick PIIDs, then filter all rows matching those PIIDs
- Re-run the full pipeline with the fix and verify label counts look reasonable

### 1.2 Notebook Structure (12 Sections)

The notebook must follow this exact section order with a markdown header cell above each:

1. **Header block:** Project title, authors, course, date, purpose, dataset citation, expected outputs
2. **Environment setup and imports:** Comment every import group, print package versions
3. **Configuration:** All constants at the top (SHARD_FOLDER, MIN_CONTRACT_VALUE=$500K, MIN_DESCRIPTION_LENGTH=100, COST_OVERRUN_THRESHOLD=0.05, LDA_NUM_TOPICS=15, TEST_SIZE=0.20, RANDOM_STATE=42)
4. **Helper functions:** `is_physical_deliverable()`, `compute_cost_growth()`, `compute_delay()`, `clean_text()`, `plot_label_distribution()`. Each with full docstring.
5. **Data loading and schema discovery:** Print column mapping, load 500-row sample, verify fields
6. **Filtering to physical deliverables:** PSC filter, $500K floor, save checkpoint
7. **Outcome label construction:** Vectorized groupby, derive over_budget and late, print dropout at each filter step
8. **EDA:** 6+ visualizations with interpretation cells below each (label distributions, cost growth distribution, delay distribution, description text quality, overrun rates by PSC/agency/contract type, correlation heatmap)
9. **Text preprocessing:** Tokenize, stop words including domain terms (FAR, DFARS, IAW, SOW, PWS, CLIN, FFP, CPFF, MOD), lemmatize with spaCy, print corpus stats
10. **Topic modeling (LDA):** Build dictionary, filter extremes (under 15 docs or over 50% of docs), train LDA with K=15, print top 10 words per topic, extract topic vectors
11. **Feature matrix construction:** Build four configurations (structured only, TF-IDF only, LDA only, combined)
12. **Preliminary classification:** Train logistic regression and random forest on each config for `over_budget` target, comparison table, ROC curves, next steps cell

### 1.3 Critical Rules for Full Points

- Notebook must run from first cell to last without any errors
- If the full dataset is too slow, use a 50K PIID sample and note it in the README under "Known Issues"
- Every section has a markdown explanation cell above it
- Every function has a docstring
- Every non-obvious line has an inline comment
- Variable names are descriptive (no `df2`, no `temp`)
- No commented-out code, no dead imports
- Consistent indentation (4 spaces)
- No lines over 100 characters

### 1.4 Supporting Files

**requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
nltk>=3.8.0
spacy>=3.6.0
gensim>=4.3.0
imbalanced-learn>=0.11.0
```

**README.md must include:**
- Author names (both team members)
- Course name and section
- Dataset source and license
- Setup instructions (Anaconda, conda env, pip install, spaCy model download, NLTK data)
- Data download instructions (Figshare URL, where to place files)
- How to run (activate env, open notebook, Run All)
- What the code accomplishes (bullet list)
- Known issues and next steps

### 1.5 Pre-Submission Checklist

- [ ] Notebook runs from top to bottom with Run All
- [ ] All cells have output (no empty cells that were never run)
- [ ] All visualizations render inline
- [ ] Classification report prints correctly
- [ ] ROC curve saved to figures/
- [ ] requirements.txt and README.md are in the submission folder
- [ ] No `__pycache__`, `.DS_Store`, or checkpoint files included
- [ ] Both team member names are in the README

---

## 2. Member 2's Work: Text Preprocessing and LDA (Weeks 3-5)

If Member 2 returns to the project, these are their primary responsibilities. If not, the agent handles this as part of Step 3. Either way, this work must be done.

### 2.1 Text Preprocessing Pipeline

**Objective:** Turn the raw `description` column into a cleaned token list suitable for LDA.

**Steps:**
1. Install and test NLTK and spaCy in the conda environment. Confirm `en_core_web_sm` downloads successfully.
2. Build custom stop words list: standard English stop words plus government procurement terms (FAR, DFARS, IAW, SOW, PWS, CLIN, FFP, CPFF, MOD, PIID, FY, TBD, NIIN, NSN, CAGE, UEI, DUNS)
3. Write the `clean_text()` function:
   - Lowercase
   - Tokenize using `nltk.word_tokenize()`
   - Remove stop words (combined list)
   - Lemmatize using spaCy
   - Remove tokens under 3 characters
   - Remove pure digit tokens
   - Return list of cleaned tokens
4. Apply to the full description corpus. Use batch processing for spaCy (`nlp.pipe()` with batch_size=100) to speed up lemmatization.
5. Store results as a new `tokens` column in the DataFrame.
6. Print preprocessing stats: total documents, average tokens per doc, top 20 tokens overall, documents with zero tokens after cleaning.
7. Drop rows with empty token lists.

### 2.2 LDA Topic Modeling

**Objective:** Discover latent topics in the contract description corpus and generate topic proportion vectors to feed into the classifier.

**Steps:**
1. Build Gensim Dictionary from the token lists
2. Filter extremes: `dictionary.filter_extremes(no_below=15, no_above=0.5)` (remove tokens in fewer than 15 docs or more than 50% of docs)
3. Build bag-of-words corpus: `[dictionary.doc2bow(tokens) for tokens in df['tokens']]`
4. Train initial LDA model with K=15 topics, passes=10, random_state=42
5. Print top 10 words per topic with clean formatting
6. Tune K using coherence scores:
   - Train LDA for K in [10, 12, 15, 18, 20, 25, 30]
   - Compute C_v coherence using `gensim.models.CoherenceModel`
   - Plot coherence vs. K
   - Select K at the highest coherence value before the curve plateaus
7. Retrain with optimal K
8. Provisionally label topics based on top words (e.g., "Topic 3: construction materials," "Topic 7: maintenance logistics")
9. Extract topic proportion vectors: one row per document, K columns, each row summing to ~1.0
10. Save trained model to `models/lda_model.gensim` for reproducibility
11. Save coherence curve to `figures/lda_coherence_scores.png`

### 2.3 TF-IDF Feature Extraction (Supplementary)

**Objective:** Build a parallel text feature set that handles shorter descriptions better than LDA.

**Steps:**
1. Use `sklearn.feature_extraction.text.TfidfVectorizer` with parameters: `max_features=500`, `min_df=10`, `max_df=0.5`, `ngram_range=(1,2)`
2. Fit on the token-joined descriptions (reconstruct cleaned text as space-separated tokens)
3. Transform the corpus into a sparse TF-IDF matrix
4. Convert to DataFrame with meaningful column names (`tfidf_[term]`)
5. Save as part of the feature matrix

### 2.4 Deliverables from This Section

- `notebooks/05_text_preprocessing.ipynb`
- `notebooks/06_topic_modeling.ipynb`
- `models/lda_model.gensim`
- Document-topic matrix saved as Parquet for downstream use
- TF-IDF matrix saved as sparse .npz file or Parquet

---

## 3. Supplementary Track: Gemini GAO Extraction (Weeks 3-6, Parallel)

This runs alongside the main pipeline. Budget is $5 in API credits. Output is a validation dataset used during final evaluation in Week 7-8.

### 3.1 Setup (Once)

- Create Google Cloud account and enable Gemini API
- Generate API key from https://aistudio.google.com/apikey
- Add $5 billing budget alert in Google Cloud Console
- Install: `pip install google-genai PyMuPDF`

### 3.2 PDF Collection

- Create `data/raw/gao_reports/` folder
- Download all 22 GAO Weapon Systems Annual Assessments (2003-2025) from gao.gov
- Name them consistently: `gao_weapons_2025.pdf`, `gao_weapons_2024.pdf`, etc.

### 3.3 PDF Chunking

- Write `scripts/chunk_gao_pdfs.py`
- Use PyMuPDF (`fitz.open()`) to identify the appendix/program-assessment section in each PDF (usually the back third)
- Split each appendix into 5-10 page chunks
- Save to `data/interim/gao_chunks/` with naming like `gao_2025_chunk_01.pdf`
- Target: 150-300 chunks total across all 22 reports

### 3.4 Prompt Design

Save the extraction prompt as `prompts/gao_extraction_prompt.txt`. Key requirements:
- Return ONLY valid JSON, no additional text
- Each object in the returned array must have exact keys: `program_name`, `report_year`, `service_branch`, `baseline_cost_millions`, `current_cost_millions`, `cost_growth_percent`, `original_ioc_date`, `current_ioc_date`, `schedule_delay_months`, `nunn_mccurdy_breach`, `primary_challenge`
- Use `null` for any value not explicitly stated in the document (never guess or estimate)
- Test on 2-3 chunks in Google AI Studio (free) before running the full batch

### 3.5 Batch Extraction

- Write `scripts/gemini_batch_extract.py`
- Submit all chunks as a batch job using Gemini 2.0 Flash for cost savings
- Include retry logic with exponential backoff for rate limits
- Save each raw JSON response to `data/interim/gao_results/`
- Log any failures for later retry

### 3.6 Parse and Clean

- Write `scripts/gemini_parse_results.py`
- Read all JSON response files
- Concatenate into a single pandas DataFrame
- Deduplicate by (program_name, report_year)
- Spot-check 10-15 entries against original PDFs for accuracy
- Clean obvious errors (wrong units, date format issues)
- Save to `data/processed/gao_validation_set.csv`

### 3.7 Link to FPDS Data

- Write `scripts/link_gao_to_fpds.py`
- Use `rapidfuzz` for fuzzy matching GAO program names to FPDS vendor/contract descriptions
- Start with exact substring matching, fall back to similarity threshold of 85%
- Document match rate (expect 20-40% of GAO programs to find FPDS matches)
- Save linked dataset to `data/processed/gao_fpds_linked.csv`

### 3.8 Validation Use

When the classifier predicts which FPDS contracts will go over budget, cross-reference the predictions against the GAO-documented outcomes for the matched subset. Use this as an independent validation check and report accuracy on this "verified" subset separately.

---

## 4. Classification and Evaluation (Weeks 6-7)

Once text features are built (both LDA topics and TF-IDF) and the feature matrix is complete, the classification and evaluation phase starts.

### 4.1 Feature Matrix Assembly

Build four feature configurations:
- **Config A (structured only):** log_initial_cost, num_modifications, num_offers, encoded contract_type, encoded competition, encoded agency (top 10 + other), PSC category
- **Config B (TF-IDF only):** 500 TF-IDF features from cleaned text
- **Config C (LDA topics only):** K topic proportion features from LDA
- **Config D (combined):** all of the above

### 4.2 Train-Test Split

- Stratified 80/20 split using `train_test_split(stratify=y, random_state=42)`
- Apply SMOTE from `imbalanced-learn` to the training set only if minority class is under 10%
- Never apply SMOTE to the test set

### 4.3 Model Training

For each of the two target variables (`over_budget`, `late`):
- Train Logistic Regression on all four configs (class_weight='balanced')
- Train Random Forest on all four configs (n_estimators=200, max_depth=20, class_weight='balanced')
- Store all 16 trained models (2 targets × 4 configs × 2 models = 16)
- Save models to `models/` folder with descriptive names

### 4.4 Evaluation

For each model:
- Confusion matrix
- Precision, recall, F1-score per class
- AUC-ROC
- For random forest: feature importance ranking

Build a master comparison table:

| Target | Config | Model | Precision (pos) | Recall (pos) | F1 | AUC |
|--------|--------|-------|-----------------|--------------|----|----|
| over_budget | A | LogReg | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

### 4.5 Visualizations for the Report

- ROC curve comparison chart (one per target variable, all four configs on same axes)
- Feature importance bar chart (top 20 features from combined-config random forest)
- Confusion matrix heatmaps for best-performing model
- Coherence score curve for LDA K selection
- Topic word cloud or bar chart for top topics

### 4.6 GAO Cross-Validation

Take the FPDS contracts matched to GAO programs. Run the best classifier on them. Report accuracy on this validation subset separately from the full test set. This serves as an independent check on the derived label quality.

### 4.7 Interpretation

Write a markdown section answering the three research questions:
1. Can contract language predict outcomes? (Cite F1 and AUC numbers)
2. Which topics are most associated with negative outcomes? (Cite specific LDA topics)
3. Do text features improve over structured-only? (Cite Config A vs. Config D numbers)

---

## 5. Final Submission: Report and Deliverables (Week 8)

The final deliverable likely includes a written report, the complete notebook, and a presentation. Confirm requirements with your professor.

### 5.1 Final Report Structure

Assume a 10-15 page written report (check syllabus for exact requirements):

1. **Abstract** (150 words): problem, approach, key finding, implication
2. **Introduction** (1 page): why government contract overruns matter, research questions
3. **Related work** (1-2 pages): IDA study, academic NLP on procurement, cite all sources from the Step 2 write-up
4. **Dataset** (1-2 pages): Omari et al. description, filtering decisions including the work-order discovery, final sample size, label construction methodology
5. **Methodology** (2-3 pages): text preprocessing, LDA with coherence tuning, TF-IDF, classifier choices, four-configuration experimental design, evaluation metrics
6. **Results** (3-4 pages): EDA findings with visualizations, LDA topic interpretation with examples, classification comparison table, feature importance analysis, GAO validation check
7. **Discussion** (1-2 pages): interpretation of findings, answer each research question directly, limitations, future work (full contract documents via SAM.gov, embeddings, BERT)
8. **References** (1 page): complete citation list

### 5.2 Presentation (If Required)

10-15 slide deck covering:
- Problem and research questions (2 slides)
- Dataset and filtering decisions (2 slides)
- Methodology pipeline diagram (1 slide)
- EDA highlights (1-2 slides)
- Top LDA topics (1 slide)
- Classification results comparison (1-2 slides)
- Feature importance (1 slide)
- Limitations and future work (1 slide)
- Conclusion (1 slide)

### 5.3 Final Code Deliverable

- Single master notebook `final_submission.ipynb` that runs the entire pipeline end-to-end on the sample dataset
- All individual phase notebooks (01-09) in the `notebooks/` folder
- `scripts/` folder with reusable standalone scripts
- `data/processed/` with final labeled contracts and GAO validation set (if small enough to include)
- `figures/` with all visualizations
- `models/` with trained LDA, logistic regression, and random forest models
- `requirements.txt`
- Comprehensive `README.md`
- `architecture.md` and `data_dictionary.md` (already created)

### 5.4 Pre-Submission Final Checklist

- [ ] All notebooks run end-to-end without errors
- [ ] README explains the full project and how to reproduce results
- [ ] All figures have titles, axis labels, and are referenced in the report
- [ ] All research questions are answered explicitly in the discussion
- [ ] Limitations are acknowledged honestly (CPARS inaccessibility, derived label noise, sample size)
- [ ] Future work section mentions SAM.gov full-document approach and transformer embeddings
- [ ] Both team members are credited
- [ ] Bibliography is complete with all sources cited
- [ ] No em-dashes in any written document
- [ ] Grammar and flow passes a final read-through

---

## 6. Quality Checks and Final Review

These apply throughout but especially before each major submission.

### 6.1 Code Quality

- No hardcoded file paths (use constants at the top)
- No commented-out code
- No `print()` statements left in for debugging
- Functions do one thing each
- No duplicated code blocks (anything repeated goes in a function)
- All imports are used

### 6.2 Documentation Quality

- Every notebook has a header cell explaining purpose and inputs/outputs
- Every function has a docstring with parameters and return value
- Complex logic has a comment block explaining the approach
- All visualizations have interpretation cells below them

### 6.3 Reproducibility

- `RANDOM_STATE = 42` is used consistently
- All file paths are relative, not absolute
- Anyone with the README and the Figshare data can reproduce the results
- `requirements.txt` has pinned minimum versions

### 6.4 Honest Limitations

The final report should explicitly acknowledge:
- FPDS description field is not the full contract document
- CPARS ground-truth performance ratings are not accessible
- Derived labels from modification histories are imperfect proxies
- Sample size may not fully represent 99M record population
- Work-order filtering may have removed some legitimate smaller contracts
- LDA topic counts selected by coherence are one of several valid methodological choices

---

## Final Notes

Everything above is structured to fit into the remaining 5-6 weeks of the project. The critical path runs through:

1. Get Step 3 notebook running cleanly (this week)
2. Finish text preprocessing and LDA (agent or Member 2)
3. Run Gemini extraction in parallel (does not block anything)
4. Build feature matrix and run full classification
5. Write the final report

If anything slips, the order of priority is: core pipeline (1, 2, 4) > final report (5) > GAO extraction (3). The GAO work is a value-add. If time runs out, drop it and move it to "future work" in the report. The core methodology is what gets you the grade.