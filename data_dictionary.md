# Data Dictionary

## Overview

This document defines every column, variable, label, function, and key term used in the project. It is organized by where each item appears in the pipeline.

---

## 1. Raw FPDS Parquet Columns

These are the columns as they appear in the Figshare Parquet shards (confirmed via Phase 2 schema discovery on 2022-2024 shards). Column names use the FPDS nested XML structure converted to dot notation. Many columns have a parent node (struct) and a `.#text` child that holds the actual value — the `.#text` variant is what the pipeline reads.

### Contract Identification

| Column | Type | Description |
|--------|------|-------------|
| `content.ID.ContractID.PIID` | string | Procurement Instrument Identifier. The unique ID for each contract. All modifications to the same contract share the same PIID. This is the primary grouping key for label construction. |
| `content.ID.ContractID.modNumber` | string | Modification number. "0" or blank indicates the original award. Subsequent numbers (1, 2, 3... or P00001, P00002...) indicate amendments. Used to sort modifications chronologically within a PIID group. |
| `content.ID.referencedID.PIID` | string | Parent Indefinite Delivery Vehicle PIID, if this contract is an order under a larger IDV. May be null for standalone contracts. |

### Contract Description (NLP Corpus)

| Column | Type | Description |
|--------|------|-------------|
| `content.contractData.descriptionOfContractRequirement` | string | Free-text description of what the contract purchases. Up to 4,000 characters. This is the primary text field for NLP/topic modeling. Quality varies widely: some entries are detailed procurement narratives, others are just part numbers or administrative codes. Median length is 35 chars; 78% are under 50 chars. Taken from the INITIAL award only (mod 0), not subsequent modifications. |

### Product/Service Classification (Filtering)

| Column | Type | Description |
|--------|------|-------------|
| `content.productOrServiceInformation.productOrServiceCode.#text` | string | Product Service Code (PSC). A 4-character code classifying what was purchased. Used to filter to physical deliverables. Y-series = construction, Z-series = real property maintenance, two-digit numeric 10-99 = supplies and equipment. |
| `content.productOrServiceInformation.principalNAICSCode.#text` | string | North American Industry Classification System code. 6-digit code classifying the contractor's industry. Used as a secondary filter and as a structured feature for modeling. |

### Dollar Fields (Cost Overrun Labels)

| Column | Type | Description |
|--------|------|-------------|
| `content.dollarValues.baseAndAllOptionsValue` | string (numeric) | Total estimated ceiling value of the contract including all options at the time of this action. Stored as string in parquet (e.g., "1829311.00"), converted to float64 during pipeline. On the initial award (mod 0), this represents the original cost estimate. This is the "initial_cost" for label construction. |
| `content.dollarValues.baseAndExercisedOptionsValue` | string (numeric) | Current committed value of the contract including exercised options at the time of this action. On the final modification, this represents the actual committed cost. This is the "final_cost" for label construction. |
| `content.dollarValues.obligatedAmount` | string (numeric) | Dollar amount obligated (legally committed) by this specific contract action. The sum across all modifications gives total obligations. |

### Date Fields (Schedule Delay Labels)

| Column | Type | Description |
|--------|------|-------------|
| `content.relevantContractDates.currentCompletionDate` | string (datetime) | The current expected completion date as of this contract action. Stored as string (e.g., "2022-12-31 12:12:00"), converted to datetime during pipeline. Compared between mod 0 and final mod to measure schedule slippage. |
| `content.relevantContractDates.ultimateCompletionDate` | string (datetime) | The ultimate completion date including all option periods. Used as a fallback if currentCompletionDate is null. |
| `content.relevantContractDates.effectiveDate` | string (datetime) | The date the contract or modification takes effect. On mod 0, this is the contract start date. |
| `content.relevantContractDates.signedDate` | string (datetime) | The date the contract action was signed. |

### Modification Information

| Column | Type | Description |
|--------|------|-------------|
| `content.contractData.reasonForModification.#text` | string | Single-letter code indicating why this modification was made. Values include: "A" (Additional Work), "B" (Supplemental Agreement), "C" (Funding Only Action), etc. Used to distinguish genuine overruns from planned changes and to flag terminated contracts (checked for "DEFAULT" or "TERMINAT" strings). |

### Structured Features (For Classifier)

| Column | Type | Description |
|--------|------|-------------|
| `content.contractData.typeOfContractPricing.#text` | string | Single-letter contract pricing type code. Key values: J (Fixed Price Redetermination), A (Firm Fixed Price), S (Cost Plus Fixed Fee), U (Cost Plus Award Fee), Y (Time and Materials). Top 5 by volume: J, A, K, U, Y. Fixed-price contracts shift cost risk to the contractor; cost-reimbursement shifts it to the government. |
| `content.competition.extentCompeted.#text` | string | Single-letter competition code. Values: A (Full and Open), D (Full and Open after exclusion), C (Not Available for Competition), G (Not Competed), etc. |
| `content.competition.numberOfOffersReceived` | string (numeric) | Number of bids/offers the government received. Stored as string, converted to numeric. Higher competition may correlate with better outcomes. |
| `content.purchaserInformation.contractingOfficeAgencyID.#text` | string | 4-digit agency code identifying which federal agency awarded the contract. Examples: 2100 = Army, 5700 = Air Force, 1700 = Navy, 97AS = DLA. |
| `content.vendor.vendorHeader.vendorName` | string | Legal name of the contractor/vendor. Used for linking to FAPIIS records and GAO program data. |
| `content.placeOfPerformance.principalPlaceOfPerformance.stateCode.#text` | string | Two-letter state code where the contract work is performed. |

---

## 2. Processed Dataset Columns (labeled_contracts.parquet)

These are the columns in the final labeled dataset produced by the filtering and label construction pipeline. Each row represents one completed contract.

### Identification

| Column | Type | Description |
|--------|------|-------------|
| `piid` | string | Contract PIID. Carried through from the raw data. Primary key. |

### NLP Text

| Column | Type | Description |
|--------|------|-------------|
| `description` | string | Contract description text from the initial award (mod 0). This is the input for text preprocessing and LDA topic modeling. Not cleaned or tokenized yet at this stage. |
| `tokens` | list of str | Lemmatized token list produced by `clean_text()` for Track A (LDA). Only populated for descriptions >= `MIN_DESCRIPTION_LENGTH` (100 chars). Each element is a lowercase, lemmatized, non-stopword token of length >= 3. |
| `tfidf_text` | string | Lightweight-cleaned text produced by `tfidf_tokenize()` for Track B (TF-IDF). Populated for all records. Lowercase, non-alphanumeric characters removed, spaces collapsed. Empty string if original description is null. |

### Outcome Labels (Target Variables)

| Column | Type | Description |
|--------|------|-------------|
| `over_budget` | int (0/1) or null | Binary label. 1 if cost_growth_pct exceeds the adaptive threshold. The notebook starts at 5% and reduces to 1% if the minority class is under 5% of the dataset. Null if cost data is unusable. This is one of the two primary target variables for classification. |
| `late` | int (0/1) or null | Binary label. 1 if delay_days is greater than 0 (final completion date is later than initial). 0 otherwise. Null if date data is unusable. This is the other primary target variable. |
| `terminated_for_default` | int (0/1) | Binary flag. 1 if any modification in the contract's history has a reason code containing "DEFAULT" or "TERMINAT". Indicates the contractor was formally found in breach. Rare but strong signal. |

### Derived Numeric Outcomes

| Column | Type | Description |
|--------|------|-------------|
| `initial_cost` | float | baseAndAllOptionsValue from the initial award. The government's original cost estimate at contract signing. |
| `final_cost` | float | baseAndExercisedOptionsValue from the final modification. The actual committed value at contract closeout. Falls back to baseAndAllOptionsValue if exercised value is null. |
| `cost_growth_pct` | float | ((final_cost - initial_cost) / initial_cost) * 100. Positive values indicate cost overruns. Negative values indicate the contract came in under budget. Null if initial_cost is zero, negative, or missing. |
| `initial_completion` | datetime | currentCompletionDate from the initial award. The original expected finish date. |
| `final_completion` | datetime | currentCompletionDate from the final modification. The actual (or last projected) finish date. |
| `delay_days` | int | (final_completion - initial_completion).days. Positive values indicate delays. Negative values indicate early completion. Null if either date is missing. |

### Structured Features

| Column | Type | Description |
|--------|------|-------------|
| `num_modifications` | int | Total number of contract actions (modifications) recorded for this PIID. Higher counts may indicate troubled contracts. |
| `contract_type` | string | Pricing type from initial award. See typeOfContractPricing above. |
| `competition` | string | Competition level from initial award. See extentCompeted above. |
| `num_offers` | numeric | Number of bids received. From initial award. |
| `agency` | string | Awarding agency code. From initial award. |
| `vendor` | string | Contractor name. From initial award. |
| `state` | string | State of performance. From initial award. |
| `psc` | string | Product Service Code. From initial award. Indicates contract category. |
| `naics` | string | NAICS code. From initial award. Indicates contractor industry. |
| `all_mod_reasons` | string | Comma-separated list of all unique reasonForModification values across the contract's history. Used for the terminated_for_default flag and for understanding how the contract evolved. |

---

## 3. Feature Matrix Columns (Built in Section 10 of step3_code_review.ipynb)

These columns exist in the feature matrices used for classification. Four configurations are defined:
- **Config A:** Structured features only
- **Config B:** TF-IDF text features only
- **Config C:** Combined (structured + TF-IDF)
- **Config D:** Structured + LDA topics (Track A subset only)

### LDA Topic Features (Config D only)

| Column | Type | Description |
|--------|------|-------------|
| `topic_0` through `topic_K` | float (0 to 1) | Topic proportion for each of K topics discovered by LDA. Each value represents how much of the contract description relates to that topic. All topic columns for a given row sum to approximately 1.0. K is currently 15 (tuning via coherence scores is pending). Only available for Track A records (descriptions >= 100 characters). |

### TF-IDF Features (Configs B and C)

| Column | Type | Description |
|--------|------|-------------|
| TF-IDF term columns | float (0 to 1) | Up to `TFIDF_MAX_FEATURES` (5,000) columns, one per vocabulary term. Values are TF-IDF weights representing term importance within each document relative to the corpus. Available for all records via Track B. |

### Encoded Structured Features (Configs A, C, D)

| Column | Type | Description |
|--------|------|-------------|
| `log_base_value` | float | Signed log transform of base_value: `sign(x) * log1p(abs(x))`. Handles negative dollar values safely while reducing skewness. |
| `log_final_value` | float | Signed log transform of final_value: `sign(x) * log1p(abs(x))`. Same safe transform as log_base_value. |
| `num_modifications` | int | Carried through from processed dataset. |
| `num_offers_imputed` | float | Number of offers received, with nulls imputed to the median value. |
| `contract_type_*` | int (0/1) | One-hot encoded columns from `pd.get_dummies` on contract_type. |
| `competition_*` | int (0/1) | One-hot encoded columns from `pd.get_dummies` on competition level. |
| `psc_category_*` | int (0/1) | One-hot encoded columns from `pd.get_dummies` on PSC category (first character: Y, Z, or numeric grouping). |

---

## 4. GAO Validation Dataset Columns (gao_validation_set.csv)

These columns are extracted from GAO Weapon Systems Annual Assessment PDFs via Gemini batch processing.

| Column | Type | Description |
|--------|------|-------------|
| `program_name` | string | Name of the Major Defense Acquisition Program as it appears in the GAO report. Examples: "F-35 Lightning II," "DDG 51 Destroyer," "CH-53K King Stallion." |
| `report_year` | int | Year of the GAO report this data was extracted from (2003-2025). |
| `service_branch` | string | Military service managing the program: Army, Navy, Air Force, Marine Corps, DoD-wide, or Space Force. |
| `baseline_cost_millions` | float or null | Total acquisition cost at the original program baseline, in millions of dollars. |
| `current_cost_millions` | float or null | Total acquisition cost at the current estimate as of the report year, in millions of dollars. |
| `cost_growth_percent` | float or null | Percentage change from baseline to current cost. Comparable to cost_growth_pct in the FPDS dataset but measured at the program level rather than contract level. |
| `original_ioc_date` | string (YYYY-MM) or null | Originally planned date for Initial Operational Capability. |
| `current_ioc_date` | string (YYYY-MM) or null | Current estimated IOC date as of the report year. |
| `schedule_delay_months` | int or null | Difference in months between current and original IOC dates. Comparable to delay_days in the FPDS dataset but measured in months at the program level. |
| `nunn_mccurdy_breach` | boolean | Whether the program triggered a Nunn-McCurdy unit cost breach (statutory threshold: 15%/25%/30%/50% growth). A strong signal of severe cost problems. |
| `primary_challenge` | string | One-sentence summary of the main challenge or risk factor described in the GAO narrative for this program. Extracted for potential supplementary NLP analysis. |

---

## 5. Key Functions (fpds_filter_and_label.py)

### explore_shard()
- **Purpose:** Load one Parquet shard and print all column names matching relevant keywords. Used for Phase 2 schema discovery.
- **Inputs:** Reads the first .parquet file found in SHARD_FOLDER.
- **Outputs:** Prints column names to console. Returns a list of relevant column name strings.
- **Side effects:** None. Read-only.

### is_physical_deliverable(psc_code)
- **Purpose:** Determine whether a Product Service Code indicates a physical deliverable.
- **Input:** psc_code (string) - a single PSC code value.
- **Output:** Boolean. True if the code starts with Y, Z, or is a two-digit numeric code between 10-99.
- **Logic:** Y-series = construction. Z-series = real property maintenance/repair. 10-99 = supplies and equipment (defense and civilian hardware).

### tfidf_tokenize(text)
- **Purpose:** Lightweight text cleaning for TF-IDF Track B. Applies basic lowercase and character filtering without spaCy lemmatization, fast enough for all records regardless of description length.
- **Input:** text (string) - raw contract description text.
- **Output:** Cleaned string suitable for TfidfVectorizer input. Lowercase, non-alphanumeric characters replaced with spaces, multiple spaces collapsed.
- **Logic:** If text is null or non-string, returns empty string. Otherwise: lowercase → regex remove non-alphanumeric (keep spaces) → collapse whitespace → strip.

### filter_shards()
- **Purpose:** Read all Parquet shards in SHARD_FOLDER, keeping only rows where the PSC code passes is_physical_deliverable(). Processes one shard at a time to manage memory.
- **Inputs:** Reads all .parquet files in SHARD_FOLDER. Uses COLUMNS_TO_KEEP to select columns.
- **Outputs:** Returns a pandas DataFrame of filtered records. Prints progress and retention stats.
- **Error handling:** Skips unreadable shards and logs errors. Warns if expected columns are missing.

### construct_labels(df)
- **Purpose:** Take the filtered DataFrame, group by PIID, compare initial vs final modification records, and produce one row per contract with outcome labels.
- **Input:** df (DataFrame) - the filtered physical-deliverable records with all modifications.
- **Output:** Returns a new DataFrame with one row per unique contract, containing: piid, description, initial_cost, final_cost, cost_growth_pct, over_budget, initial_completion, final_completion, delay_days, late, terminated_for_default, and all structured features.
- **Label logic:**
  - over_budget = 1 if cost_growth_pct > threshold (starts at 5%, adapts to 1% if minority class < 5% of dataset), else 0
  - late = 1 if delay_days > 0, else 0
  - terminated_for_default = 1 if any modification reason contains "DEFAULT" or "TERMINAT"

---

## 7. Label Thresholds and Decision Points

| Parameter | Default Value | Adaptive Logic |
|-----------|--------------|----------------|
| Cost overrun threshold | 5% growth | Starts at 5%. If minority class < 5% of dataset, drops to 1%. Documented in notebook Section 6.4b. |
| Schedule delay threshold | Any delay (> 0 days) | Increase to 30+ days if late class is severely imbalanced |
| Minimum description length | 100 characters (Track A) | Track B (TF-IDF) has no minimum. Lower Track A to 50 if filtering at 100 drops too many records. |
| LDA topic count | K=15 (preliminary) | Tune via coherence scores across K=10-30 (pending for final submission) |
| LDA coherence metric | C_v | Switch to U_mass if C_v is unstable |
| TF-IDF max features | 5,000 | Increase if vocabulary is too constrained; decrease for speed |
| Train-test split | 80/20 stratified | Switch to 5-fold cross-validation if dataset is under 5,000 records |
| Class weighting | `class_weight='balanced'` | Apply SMOTE to training set only if balanced weighting is insufficient |

---

## 8. File Formats

| Format | Used For | Why |
|--------|----------|-----|
| Parquet | Raw shards, interim filtered data, final labeled dataset | Columnar format, fast reads, supports selective column loading, handles large datasets efficiently |
| CSV | Final labeled dataset (duplicate), GAO validation set | Human-readable, opens in Excel/Google Sheets for quick inspection |
| PNG | All figures and visualizations | Standard image format, embeds in reports and notebooks |
| JSON | Gemini API responses | Native output format of the Gemini API |
| PKL (pickle) | Saved scikit-learn models | Standard Python serialization for trained model objects |
| Gensim native | Saved LDA model | Gensim's own format preserves model state for reloading |
| TXT | Column names dump, dataset summary | Simple text for quick reference |

---

## 9. External Identifiers and Codes

| Code System | Description | Used For |
|-------------|-------------|----------|
| PIID | Procurement Instrument Identifier. Unique contract ID assigned by the contracting office. Format varies by agency. | Grouping modifications, linking across datasets |
| PSC | Product Service Code. 4-character alphanumeric. First character indicates broad category (Y=construction, R=professional services, D=IT, etc.). | Filtering to physical deliverables |
| NAICS | North American Industry Classification System. 6-digit numeric. Classifies contractor industry. | Secondary filtering, structured feature |
| UEI | Unique Entity Identifier. Replaced DUNS in 2022. Identifies contractors in SAM.gov. | Linking to FAPIIS termination records |
| CAGE | Commercial and Government Entity code. 5-character alphanumeric. Identifies contractor facilities. | Secondary contractor linking |