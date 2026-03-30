"""Build the complete step3_code_review.ipynb notebook in one shot.

All known fixes applied:
- PIID-group sampling (50K contracts)
- num_offers: object->numeric with NaN fill
- log transform: use np.log1p(np.abs(x)) for negative dollar values
- No dead code (plot_label_distribution removed)
- Adaptive threshold for over_budget
"""

import json

def md(cell_id, source_lines):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines
    }

def code(cell_id, source_lines):
    """Create a code cell (no outputs - will be filled by execution)."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

cells = []

# ── Section 0: Header ──────────────────────────────────────────────────
cells.append(md("s0_header", [
    "# Predicting Federal Contract Outcomes Using NLP and Machine Learning\n",
    "\n",
    "**Authors:** Leonel Lourenco, Rana Khan  \n",
    "**Course:** IS392 Section 452  \n",
    "**Institution:** New Jersey Institute of Technology  \n",
    "**Date:** 3/28/2026\n",
    "\n",
    "## Purpose\n",
    "This notebook implements the data pipeline and initial analysis for predicting whether U.S. federal government contracts for physical deliverables will experience cost overruns or schedule delays. It uses contract description text (NLP via LDA topic modeling and TF-IDF) combined with structured contract attributes to train binary classifiers.\n",
    "\n",
    "## Dataset\n",
    "Omari et al. Comprehensive Federal Procurement Dataset (1979-2023), published in Scientific Data (Nature, 2025). 99 million contract action records, 470 variables. CC0 license. Source: https://doi.org/10.6084/m9.figshare.28057043\n",
    "\n",
    "## Expected Outputs\n",
    "- Filtered dataset of completed physical-deliverable contracts\n",
    "- Binary outcome labels: over_budget (0/1), late (0/1)\n",
    "- Exploratory data analysis with visualizations\n",
    "- Preprocessed text corpus ready for topic modeling (two-track: LDA + TF-IDF)\n",
    "- Preliminary LDA topic model and TF-IDF feature matrix\n",
    "- Initial classification results comparing four feature configurations"
]))

# ── Section 1: Environment Setup ───────────────────────────────────────
cells.append(md("s1_md", [
    "## 1. Environment Setup and Imports\n",
    "Import all required libraries and configure display settings. All dependencies are listed in `requirements.txt`."
]))

cells.append(code("s1_code", [
    "# Data handling and Parquet file reading\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pyarrow.parquet as pq\n",
    "import os\n",
    "import glob\n",
    "import warnings\n",
    "\n",
    "# Visualization\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Machine learning: classifiers, metrics, preprocessing\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (classification_report, confusion_matrix,\n",
    "                             roc_auc_score, roc_curve, f1_score)\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# NLP: tokenization, stop words, lemmatization\n",
    "import nltk\n",
    "import spacy\n",
    "\n",
    "# Topic modeling\n",
    "from gensim.models import LdaModel\n",
    "from gensim.corpora import Dictionary\n",
    "\n",
    "# Utilities\n",
    "from collections import Counter\n",
    "from tqdm import tqdm\n",
    "import re\n",
    "\n",
    "# Suppress noisy warnings for cleaner notebook output\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Display settings\n",
    "pd.set_option('display.max_columns', 50)\n",
    "pd.set_option('display.max_colwidth', 100)\n",
    "\n",
    "# Matplotlib and seaborn styling\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette('muted')\n",
    "\n",
    "# Global reproducibility seed\n",
    "RANDOM_STATE = 42\n",
    "\n",
    "# Print package versions for reproducibility\n",
    "print('Package Versions')\n",
    "print('-' * 30)\n",
    "for pkg_name, pkg in [('pandas', pd), ('numpy', np), ('matplotlib', matplotlib),\n",
    "                       ('seaborn', sns), ('sklearn', __import__('sklearn')),\n",
    "                       ('nltk', nltk), ('spacy', spacy),\n",
    "                       ('gensim', __import__('gensim'))]:\n",
    "    print(f'  {pkg_name}: {pkg.__version__}')"
]))

# ── Section 2: Configuration ──────────────────────────────────────────
cells.append(md("s2_md", [
    "## 2. Configuration\n",
    "Define all file paths, filtering criteria, labeling thresholds, and modeling parameters as constants. This centralizes configuration and makes the pipeline easier to modify and reproduce."
]))

cells.append(code("s2_code", [
    "# --- File Paths ---\n",
    "SHARD_FOLDER = './exploring_data'\n",
    "INTERIM_OUTPUT = './data/interim/filtered_physical_deliverables.parquet'\n",
    "FINAL_OUTPUT = './data/processed/labeled_contracts.parquet'\n",
    "FIGURES_FOLDER = './figures'\n",
    "\n",
    "# Create output directories if they don't exist\n",
    "os.makedirs(os.path.dirname(INTERIM_OUTPUT), exist_ok=True)\n",
    "os.makedirs(os.path.dirname(FINAL_OUTPUT), exist_ok=True)\n",
    "os.makedirs(FIGURES_FOLDER, exist_ok=True)\n",
    "\n",
    "# --- Filtering Criteria ---\n",
    "PHYSICAL_PSC_PREFIXES = ['Y', 'Z']\n",
    "PHYSICAL_PSC_NUMERIC_RANGE = (10, 99)\n",
    "\n",
    "# --- Labeling Thresholds ---\n",
    "COST_OVERRUN_THRESHOLD = 0.05\n",
    "SCHEDULE_DELAY_THRESHOLD = 0\n",
    "MIN_DESCRIPTION_LENGTH = 100\n",
    "\n",
    "# --- Sampling ---\n",
    "SAMPLE_CONTRACTS = 50_000\n",
    "\n",
    "# --- Modeling Parameters ---\n",
    "LDA_NUM_TOPICS = 15\n",
    "LDA_PASSES = 10\n",
    "TFIDF_MAX_FEATURES = 5000\n",
    "TEST_SIZE = 0.20\n",
    "\n",
    "# --- Exact Parquet Column Mapping ---\n",
    "COLUMN_MAP = {\n",
    "    'piid':              'content.ID.ContractID.PIID',\n",
    "    'mod_number':        'content.ID.ContractID.modNumber',\n",
    "    'description':       'content.contractData.descriptionOfContractRequirement',\n",
    "    'psc':               'content.productOrServiceInformation.productOrServiceCode.#text',\n",
    "    'naics':             'content.productOrServiceInformation.principalNAICSCode.#text',\n",
    "    'base_all_options':  'content.dollarValues.baseAndAllOptionsValue',\n",
    "    'base_exercised':    'content.dollarValues.baseAndExercisedOptionsValue',\n",
    "    'current_completion':'content.relevantContractDates.currentCompletionDate',\n",
    "    'ultimate_completion':'content.relevantContractDates.ultimateCompletionDate',\n",
    "    'effective_date':    'content.relevantContractDates.effectiveDate',\n",
    "    'signed_date':       'content.relevantContractDates.signedDate',\n",
    "    'reason_for_mod':    'content.contractData.reasonForModification.#text',\n",
    "    'contract_type':     'content.contractData.typeOfContractPricing.#text',\n",
    "    'extent_competed':   'content.competition.extentCompeted.#text',\n",
    "    'num_offers':        'content.competition.numberOfOffersReceived',\n",
    "    'agency_id':         'content.purchaserInformation.contractingOfficeAgencyID.#text',\n",
    "    'vendor_name':       'content.vendor.vendorHeader.vendorName',\n",
    "    'state_code':        'content.placeOfPerformance.principalPlaceOfPerformance.stateCode.#text',\n",
    "}\n",
    "COLUMNS_TO_READ = list(COLUMN_MAP.values())\n",
    "\n",
    "print('Configuration loaded.')\n",
    "print(f'  Shard folder: {SHARD_FOLDER}')\n",
    "print(f'  Sample contracts: {SAMPLE_CONTRACTS:,}')\n",
    "print(f'  Cost overrun threshold: {COST_OVERRUN_THRESHOLD:.0%}')\n",
    "print(f'  Columns mapped: {len(COLUMN_MAP)}')"
]))

# ── Section 3: Helper Functions ────────────────────────────────────────
cells.append(md("s3_md", [
    "## 3. Helper Functions\n",
    "Define reusable functions for filtering, label computation, and text preprocessing."
]))

cells.append(code("s3_code", [
    "def is_physical_deliverable(psc_code: str) -> bool:\n",
    "    \"\"\"Check if a PSC code indicates physical deliverables.\"\"\"\n",
    "    if pd.isna(psc_code):\n",
    "        return False\n",
    "    psc_str = str(psc_code).strip().upper()\n",
    "    if psc_str.startswith(('Y', 'Z')):\n",
    "        return True\n",
    "    try:\n",
    "        psc_num = int(psc_str)\n",
    "        return PHYSICAL_PSC_NUMERIC_RANGE[0] <= psc_num <= PHYSICAL_PSC_NUMERIC_RANGE[1]\n",
    "    except ValueError:\n",
    "        return False\n",
    "\n",
    "\n",
    "def compute_cost_growth(base_val, final_val) -> float:\n",
    "    \"\"\"Compute percentage cost growth between base and final values.\"\"\"\n",
    "    try:\n",
    "        base = float(str(base_val).replace('$', '').replace(',', '').strip())\n",
    "        final = float(str(final_val).replace('$', '').replace(',', '').strip())\n",
    "        if base == 0:\n",
    "            return np.nan\n",
    "        return (final - base) / abs(base)\n",
    "    except (ValueError, TypeError):\n",
    "        return np.nan\n",
    "\n",
    "\n",
    "def compute_delay(current_date, ultimate_date) -> float:\n",
    "    \"\"\"Compute schedule delay in days between current and ultimate completion dates.\"\"\"\n",
    "    try:\n",
    "        current = pd.to_datetime(current_date)\n",
    "        ultimate = pd.to_datetime(ultimate_date)\n",
    "        return (ultimate - current).days\n",
    "    except (ValueError, TypeError):\n",
    "        return np.nan\n",
    "\n",
    "\n",
    "def clean_text(text: str) -> str:\n",
    "    \"\"\"Clean and normalize text for LDA topic modeling (Track A).\"\"\"\n",
    "    if pd.isna(text) or not isinstance(text, str):\n",
    "        return ''\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'[^a-z0-9\\s]', ' ', text)\n",
    "    text = re.sub(r'\\s+', ' ', text)\n",
    "    doc = nlp(text)\n",
    "    tokens = [token.lemma_ for token in doc\n",
    "              if not token.is_stop and not token.is_punct and len(token.lemma_) > 2]\n",
    "    return ' '.join(tokens)\n",
    "\n",
    "\n",
    "def tfidf_tokenize(text: str) -> str:\n",
    "    \"\"\"Simple tokenizer for TF-IDF vectorizer (Track B).\"\"\"\n",
    "    if pd.isna(text) or not isinstance(text, str):\n",
    "        return ''\n",
    "    tokens = text.lower().split()\n",
    "    tokens = [t for t in tokens if len(t) > 2 and t.isalpha()]\n",
    "    return ' '.join(tokens)\n",
    "\n",
    "\n",
    "print('Helper functions defined.')"
]))

# ── Section 4: Data Loading & Schema Discovery ────────────────────────
cells.append(md("s4_md", [
    "## 4. Data Loading and Schema Discovery\n",
    "Load the Parquet shards, inspect the schema, and verify that all mapped columns exist."
]))

cells.append(code("s4_code", [
    "shard_files = sorted(glob.glob(os.path.join(SHARD_FOLDER, '*.parquet')))\n",
    "print(f'Found {len(shard_files)} Parquet shards in {SHARD_FOLDER}/')\n",
    "print(f'First shard: {os.path.basename(shard_files[0])}')\n",
    "\n",
    "schema = pq.read_schema(shard_files[0])\n",
    "print(f'Total columns in schema: {len(schema.names)}')\n",
    "\n",
    "print('\\nVerifying mapped columns...')\n",
    "missing = [f'{k} -> {v}' for k, v in COLUMN_MAP.items() if v not in schema.names]\n",
    "if missing:\n",
    "    print('❌ Missing:', missing)\n",
    "else:\n",
    "    print(f'✅ All {len(COLUMN_MAP)} mapped columns found')\n",
    "\n",
    "sample_df = pq.read_table(shard_files[0], columns=COLUMNS_TO_READ).to_pandas().head(500)\n",
    "print(f'\\nSample shape: {sample_df.shape}')\n",
    "for col in list(sample_df.columns)[:5]:\n",
    "    print(f'  {col}: {sample_df[col].dtype}, sample={sample_df[col].iloc[0]}')"
]))

# ── Section 5: Filtering to Physical Deliverables ─────────────────────
cells.append(md("s5_md", [
    "## 5. Filtering to Physical Deliverables\n",
    "Process all Parquet shards to filter contracts matching physical deliverable criteria."
]))

cells.append(code("s5_code", [
    "filtered_shards = []\n",
    "total_input = 0\n",
    "\n",
    "print(f'Processing {len(shard_files)} shards...')\n",
    "for i, shard_path in enumerate(shard_files, 1):\n",
    "    shard_name = os.path.basename(shard_path)\n",
    "    shard_df = pq.read_table(shard_path, columns=COLUMNS_TO_READ).to_pandas()\n",
    "    total_input += len(shard_df)\n",
    "    physical_mask = shard_df[COLUMN_MAP['psc']].apply(is_physical_deliverable)\n",
    "    physical_df = shard_df[physical_mask].copy()\n",
    "    pct = len(physical_df) / len(shard_df) * 100\n",
    "    print(f'  Shard {i}/{len(shard_files)}: {shard_name} — {len(shard_df):,} rows -> {len(physical_df):,} ({pct:.1f}%)')\n",
    "    filtered_shards.append(physical_df)\n",
    "\n",
    "physical_contracts = pd.concat(filtered_shards, ignore_index=True)\n",
    "print(f'\\nTotal: {total_input:,} rows -> {len(physical_contracts):,} physical deliverables '\n",
    "      f'({len(physical_contracts)/total_input*100:.1f}% retention)')\n",
    "\n",
    "physical_contracts.to_parquet(INTERIM_OUTPUT, index=False)\n",
    "print(f'Saved to {INTERIM_OUTPUT}')"
]))

# ── Sampling Decision ──────────────────────────────────────────────────
cells.append(md("s5b_md", [
    "### Sampling Decision\n",
    "The full filtered dataset contains ~4.3 million rows. We draw a **50,000-contract sample** using PIID-group sampling to preserve contract histories."
]))

cells.append(code("s5b_code", [
    "unique_piids = physical_contracts[COLUMN_MAP['piid']].unique()\n",
    "print(f'Total unique contracts (PIIDs): {len(unique_piids):,}')\n",
    "\n",
    "if len(unique_piids) > SAMPLE_CONTRACTS:\n",
    "    rng = np.random.RandomState(RANDOM_STATE)\n",
    "    sampled_piids = rng.choice(unique_piids, size=SAMPLE_CONTRACTS, replace=False)\n",
    "    sample_df = physical_contracts[physical_contracts[COLUMN_MAP['piid']].isin(sampled_piids)].copy()\n",
    "    print(f'Sampled {SAMPLE_CONTRACTS:,} contracts -> {len(sample_df):,} modification rows')\n",
    "else:\n",
    "    sample_df = physical_contracts.copy()\n",
    "    print(f'Using all {len(unique_piids):,} contracts')\n",
    "\n",
    "sample_df = sample_df.reset_index(drop=True)\n",
    "print(f'Working sample shape: {sample_df.shape}')"
]))

# ── Section 6: Outcome Label Construction ─────────────────────────────
cells.append(md("s6_md", [
    "## 6. Outcome Label Construction\n",
    "Construct binary outcome labels for cost overruns (`over_budget`) and schedule delays (`late`)."
]))

cells.append(code("s6_code", [
    "# Type-cast dollar columns to numeric\n",
    "for col in [COLUMN_MAP['base_all_options'], COLUMN_MAP['base_exercised']]:\n",
    "    sample_df[col] = pd.to_numeric(\n",
    "        sample_df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False),\n",
    "        errors='coerce'\n",
    "    )\n",
    "\n",
    "# Type-cast date columns\n",
    "for col in [COLUMN_MAP['current_completion'], COLUMN_MAP['ultimate_completion'],\n",
    "            COLUMN_MAP['effective_date'], COLUMN_MAP['signed_date']]:\n",
    "    sample_df[col] = pd.to_datetime(sample_df[col], errors='coerce')\n",
    "\n",
    "# Sort by PIID and modification number\n",
    "sample_df = sample_df.sort_values([COLUMN_MAP['piid'], COLUMN_MAP['mod_number']])\n",
    "\n",
    "# Group by PIID to construct labels\n",
    "print('Grouping by PIID to construct labels...')\n",
    "label_rows = []\n",
    "for piid, group in tqdm(sample_df.groupby(COLUMN_MAP['piid']),\n",
    "                        total=sample_df[COLUMN_MAP['piid']].nunique()):\n",
    "    first = group.iloc[0]\n",
    "    last  = group.iloc[-1]\n",
    "    base_val = first[COLUMN_MAP['base_all_options']]\n",
    "    final_val = last[COLUMN_MAP['base_exercised']]\n",
    "    current_date = first[COLUMN_MAP['current_completion']]\n",
    "    ultimate_date = last[COLUMN_MAP['ultimate_completion']]\n",
    "    cost_growth = compute_cost_growth(base_val, final_val)\n",
    "    delay = compute_delay(current_date, ultimate_date)\n",
    "    label_rows.append({\n",
    "        'piid': piid,\n",
    "        'description': first[COLUMN_MAP['description']],\n",
    "        'psc': first[COLUMN_MAP['psc']],\n",
    "        'naics': first[COLUMN_MAP['naics']],\n",
    "        'contract_type': first[COLUMN_MAP['contract_type']],\n",
    "        'extent_competed': first[COLUMN_MAP['extent_competed']],\n",
    "        'num_offers': first[COLUMN_MAP['num_offers']],\n",
    "        'agency_id': first[COLUMN_MAP['agency_id']],\n",
    "        'state_code': first[COLUMN_MAP['state_code']],\n",
    "        'base_value': base_val,\n",
    "        'final_value': final_val,\n",
    "        'cost_growth_pct': cost_growth * 100 if pd.notna(cost_growth) else np.nan,\n",
    "        'delay_days': delay,\n",
    "        'modifications': len(group),\n",
    "    })\n",
    "\n",
    "labeled_df = pd.DataFrame(label_rows)\n",
    "\n",
    "# Adaptive threshold\n",
    "threshold = COST_OVERRUN_THRESHOLD\n",
    "labeled_df['over_budget'] = (labeled_df['cost_growth_pct'] > threshold * 100).astype(int)\n",
    "if labeled_df['over_budget'].mean() < 0.05:\n",
    "    threshold = 0.01\n",
    "    labeled_df['over_budget'] = (labeled_df['cost_growth_pct'] > threshold * 100).astype(int)\n",
    "    print(f'Adaptive threshold applied: {threshold:.0%}')\n",
    "\n",
    "labeled_df['late'] = (labeled_df['delay_days'] > SCHEDULE_DELAY_THRESHOLD).astype(int)\n",
    "\n",
    "# Drop rows with missing labels\n",
    "before = len(labeled_df)\n",
    "labeled_df = labeled_df.dropna(subset=['cost_growth_pct', 'delay_days']).copy()\n",
    "print(f'Valid labels: {len(labeled_df):,} / {before:,} ({len(labeled_df)/before*100:.1f}%)')\n",
    "\n",
    "for t in ['over_budget', 'late']:\n",
    "    pos = labeled_df[t].sum()\n",
    "    print(f'  {t}: {pos:,} positive ({pos/len(labeled_df)*100:.2f}%)')"
]))

cells.append(code("s6_save", [
    "labeled_df.to_parquet(FINAL_OUTPUT, index=False)\n",
    "print(f'Saved {len(labeled_df):,} labeled contracts to {FINAL_OUTPUT}')\n",
    "print(labeled_df[['piid','cost_growth_pct','delay_days','over_budget','late']].head())"
]))

# ── Section 7: EDA ─────────────────────────────────────────────────────
cells.append(md("s7_md", [
    "## 7. Exploratory Data Analysis\n",
    "Analyze the labeled dataset to understand class balance, feature distributions, and relationships."
]))

# 7.1 Label Distributions
cells.append(md("s71_md", ["### 7.1 Label Distributions"]))
cells.append(code("s71_code", [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "for ax, target, colors, labels in [\n",
    "    (axes[0], 'over_budget', ['lightblue','salmon'], ['On Budget (0)','Over Budget (1)']),\n",
    "    (axes[1], 'late', ['lightgreen','orange'], ['On Time (0)','Late (1)'])]:\n",
    "    counts = labeled_df[target].value_counts().sort_index()\n",
    "    ax.bar(labels, counts.values, color=colors)\n",
    "    ax.set_title(f'{target} Distribution')\n",
    "    ax.set_ylabel('Count')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_FOLDER, 'class_balance.png'), dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "for t in ['over_budget','late']:\n",
    "    pos = labeled_df[t].sum()\n",
    "    print(f'{t}: {pos:,} positive ({pos/len(labeled_df)*100:.2f}%)')"
]))

# 7.2 Cost Growth
cells.append(md("s72_md", ["### 7.2 Cost Growth Distribution"]))
cells.append(code("s72_code", [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "cg = labeled_df['cost_growth_pct'].clip(-50, 100)\n",
    "axes[0].hist(cg, bins=50, alpha=0.7, color='skyblue', edgecolor='black')\n",
    "axes[0].axvline(0, color='red', ls='--', alpha=0.7, label='No Growth')\n",
    "axes[0].axvline(5, color='orange', ls='--', alpha=0.7, label='5% Threshold')\n",
    "axes[0].set_xlabel('Cost Growth %'); axes[0].set_ylabel('Count')\n",
    "axes[0].set_title('Cost Growth Distribution (clipped)'); axes[0].legend()\n",
    "axes[1].boxplot(cg, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))\n",
    "axes[1].set_title('Cost Growth Box Plot')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_FOLDER, 'cost_growth_distribution.png'), dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(labeled_df['cost_growth_pct'].describe())"
]))

# 7.3 Schedule Delay
cells.append(md("s73_md", ["### 7.3 Schedule Delay Distribution"]))
cells.append(code("s73_code", [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "dd = labeled_df['delay_days'].clip(-100, 365)\n",
    "axes[0].hist(dd, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')\n",
    "axes[0].axvline(0, color='green', ls='--', alpha=0.7, label='On Time')\n",
    "axes[0].set_xlabel('Delay Days'); axes[0].set_ylabel('Count')\n",
    "axes[0].set_title('Schedule Delay Distribution (clipped)'); axes[0].legend()\n",
    "axes[1].boxplot(dd, vert=True, patch_artist=True, boxprops=dict(facecolor='lightcoral'))\n",
    "axes[1].set_title('Schedule Delay Box Plot')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_FOLDER, 'delay_distribution.png'), dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(labeled_df['delay_days'].describe())"
]))

# 7.4 Description Text Quality
cells.append(md("s74_md", ["### 7.4 Description Text Quality"]))
cells.append(code("s74_code", [
    "desc_len = labeled_df['description'].str.len().fillna(0)\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "ax.hist(desc_len.clip(upper=500), bins=50, alpha=0.7, color='gold', edgecolor='black')\n",
    "ax.axvline(MIN_DESCRIPTION_LENGTH, color='red', ls='--', label=f'LDA threshold ({MIN_DESCRIPTION_LENGTH} chars)')\n",
    "ax.set_xlabel('Description Length (chars)'); ax.set_ylabel('Count')\n",
    "ax.set_title('Description Length Distribution'); ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_FOLDER, 'description_length_distribution.png'), dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "long = (desc_len >= MIN_DESCRIPTION_LENGTH).sum()\n",
    "print(f'Track A (LDA, >={MIN_DESCRIPTION_LENGTH} chars): {long:,} contracts')\n",
    "print(f'Track B (TF-IDF, all): {len(labeled_df):,} contracts')"
]))

# 7.5 Overrun Rates by Category
cells.append(md("s75_md", ["### 7.5 Overrun Rates by Category"]))
cells.append(code("s75_code", [
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "for ax, col, title, color in [\n",
    "    (axes[0], 'psc', 'PSC', 'tomato'),\n",
    "    (axes[1], 'agency_id', 'Agency', 'steelblue'),\n",
    "    (axes[2], 'contract_type', 'Contract Type', 'mediumseagreen')]:\n",
    "    rates = labeled_df.groupby(col)['over_budget'].mean().sort_values(ascending=False).head(10)\n",
    "    rates.plot.bar(ax=ax, color=color, alpha=0.7)\n",
    "    ax.set_title(f'Overrun Rate by {title} (Top 10)')\n",
    "    ax.set_ylabel('Overrun Rate'); ax.tick_params(axis='x', rotation=45)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_FOLDER, 'overrun_rates_by_category.png'), dpi=300, bbox_inches='tight')\n",
    "plt.show()"
]))

# 7.6 Correlation Heatmap
cells.append(md("s76_md", ["### 7.6 Correlation Heatmap"]))
cells.append(code("s76_code", [
    "numeric_cols = ['base_value', 'final_value', 'cost_growth_pct', 'delay_days',\n",
    "                'modifications', 'over_budget', 'late']\n",
    "corr_df = labeled_df[numeric_cols].apply(pd.to_numeric, errors='coerce')\n",
    "corr_matrix = corr_df.corr()\n",
    "plt.figure(figsize=(8, 6))\n",
    "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n",
    "sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,\n",
    "            square=True, fmt='.2f')\n",
    "plt.title('Correlation Heatmap')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_FOLDER, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── Section 8: Text Preprocessing ─────────────────────────────────────
cells.append(md("s8_md", [
    "## 8. Text Preprocessing\n",
    "Two-track NLP approach:\n",
    "- **Track A (LDA)**: long descriptions (≥100 chars) with spaCy lemmatization\n",
    "- **Track B (TF-IDF)**: all descriptions with simple tokenization"
]))

cells.append(code("s8_nlp", [
    "nltk.download('punkt', quiet=True)\n",
    "nltk.download('stopwords', quiet=True)\n",
    "nltk.download('wordnet', quiet=True)\n",
    "nltk.download('punkt_tab', quiet=True)\n",
    "nlp = spacy.load('en_core_web_sm')\n",
    "print('✅ NLP libraries ready')"
]))

cells.append(code("s8_preprocess", [
    "from nltk.corpus import stopwords\n",
    "contract_stopwords = {'contract','shall','will','government','federal','agency',\n",
    "                      'provide','services','service','support','work','period','require'}\n",
    "all_stopwords = set(stopwords.words('english')).union(contract_stopwords)\n",
    "\n",
    "track_a_mask = labeled_df['description'].str.len() >= MIN_DESCRIPTION_LENGTH\n",
    "track_a_df = labeled_df[track_a_mask].copy()\n",
    "track_b_df = labeled_df.copy()\n",
    "print(f'Track A (LDA): {len(track_a_df):,} contracts')\n",
    "print(f'Track B (TF-IDF): {len(track_b_df):,} contracts')\n",
    "\n",
    "tqdm.pandas()\n",
    "track_a_df['clean_description'] = track_a_df['description'].progress_apply(clean_text)\n",
    "track_b_df['tfidf_text'] = track_b_df['description'].progress_apply(tfidf_tokenize)\n",
    "print('✅ Text preprocessing complete')"
]))

# ── Section 9: Topic Modeling & TF-IDF ────────────────────────────────
cells.append(md("s9_md", [
    "## 9. Topic Modeling (LDA) and TF-IDF Feature Extraction"
]))

cells.append(code("s9_lda", [
    "# Track A: LDA\n",
    "track_a_docs = [doc.split() for doc in track_a_df['clean_description'] if doc.strip()]\n",
    "dictionary = Dictionary(track_a_docs)\n",
    "dictionary.filter_extremes(no_below=5, no_above=0.5)\n",
    "corpus = [dictionary.doc2bow(doc) for doc in track_a_docs]\n",
    "print(f'Vocabulary: {len(dictionary):,} tokens, Corpus: {len(corpus):,} docs')\n",
    "\n",
    "lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=LDA_NUM_TOPICS,\n",
    "                     passes=LDA_PASSES, random_state=RANDOM_STATE, alpha='auto', eta='auto')\n",
    "print('✅ LDA model trained')\n",
    "\n",
    "# Extract topic proportions\n",
    "topic_cols = [f'topic_{i}' for i in range(LDA_NUM_TOPICS)]\n",
    "topic_vecs = []\n",
    "for doc_bow in corpus:\n",
    "    dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)\n",
    "    topic_vecs.append([p for _, p in dist])\n",
    "track_a_topic_df = pd.DataFrame(topic_vecs, columns=topic_cols, index=track_a_df.index[:len(topic_vecs)])\n",
    "print(f'Topic features shape: {track_a_topic_df.shape}')"
]))

cells.append(code("s9_tfidf", [
    "# Track B: TF-IDF\n",
    "tfidf_vectorizer = TfidfVectorizer(\n",
    "    max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.5,\n",
    "    ngram_range=(1, 2), stop_words=list(all_stopwords), lowercase=True\n",
    ")\n",
    "tfidf_matrix = tfidf_vectorizer.fit_transform(track_b_df['tfidf_text'].fillna(''))\n",
    "print(f'TF-IDF shape: {tfidf_matrix.shape}')\n",
    "print(f'Avg non-zero features/doc: {tfidf_matrix.nnz / tfidf_matrix.shape[0]:.1f}')"
]))

cells.append(code("s9_topics", [
    "print('LDA Topics:')\n",
    "for idx, topic in lda_model.print_topics(num_words=8):\n",
    "    words = [w.split('*')[1].strip('\"') for w in topic.split(' + ')]\n",
    "    print(f'  Topic {idx}: {\", \".join(words)}')"
]))

# ── Section 10: Feature Matrix Construction ───────────────────────────
cells.append(md("s10_md", [
    "## 10. Feature Matrix Construction\n",
    "Four feature configurations:\n",
    "1. **Structured-only**: contract attributes\n",
    "2. **TF-IDF-only**: text features\n",
    "3. **Combined**: structured + TF-IDF\n",
    "4. **Structured+LDA**: structured + topic proportions"
]))

cells.append(code("s10_code", [
    "# Prepare structured features with proper NaN handling\n",
    "struct_cols = ['base_value', 'final_value', 'modifications', 'num_offers']\n",
    "struct_df = labeled_df[struct_cols].copy()\n",
    "\n",
    "# Convert num_offers from object to numeric (contains None strings)\n",
    "struct_df['num_offers'] = pd.to_numeric(struct_df['num_offers'], errors='coerce')\n",
    "\n",
    "# Fill all missing values\n",
    "struct_df = struct_df.fillna({\n",
    "    'base_value': 0,\n",
    "    'final_value': 0,\n",
    "    'modifications': 1,\n",
    "    'num_offers': 1\n",
    "})\n",
    "\n",
    "# Safe log transform: use log1p(abs(x)) * sign(x) to handle negatives\n",
    "struct_df['log_base_value'] = np.sign(struct_df['base_value']) * np.log1p(np.abs(struct_df['base_value']))\n",
    "struct_df['log_final_value'] = np.sign(struct_df['final_value']) * np.log1p(np.abs(struct_df['final_value']))\n",
    "struct_df = struct_df.drop(['base_value', 'final_value'], axis=1)\n",
    "\n",
    "# Replace any remaining inf/NaN with 0\n",
    "struct_df = struct_df.replace([np.inf, -np.inf], 0).fillna(0)\n",
    "\n",
    "print(f'Structured features: {struct_df.shape}')\n",
    "print(f'NaN check: {struct_df.isna().sum().sum()} NaN values remaining')\n",
    "\n",
    "# Build 4 configurations\n",
    "tfidf_dense = pd.DataFrame(tfidf_matrix.toarray(),\n",
    "                           columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],\n",
    "                           index=labeled_df.index)\n",
    "\n",
    "X_structured = struct_df\n",
    "X_tfidf = tfidf_dense\n",
    "X_combined = pd.concat([struct_df, tfidf_dense], axis=1)\n",
    "\n",
    "# Structured+LDA: only for contracts with long descriptions\n",
    "common_lda_idx = struct_df.index.intersection(track_a_topic_df.index)\n",
    "X_struct_lda = pd.concat([struct_df.loc[common_lda_idx], track_a_topic_df.loc[common_lda_idx]], axis=1)\n",
    "\n",
    "y_full = labeled_df['over_budget']\n",
    "y_lda = labeled_df.loc[common_lda_idx, 'over_budget']\n",
    "\n",
    "feature_configs = {\n",
    "    'Structured': (X_structured, y_full),\n",
    "    'TF-IDF': (X_tfidf, y_full),\n",
    "    'Combined': (X_combined, y_full),\n",
    "    'Structured+LDA': (X_struct_lda, y_lda),\n",
    "}\n",
    "\n",
    "for name, (X, y) in feature_configs.items():\n",
    "    print(f'  {name}: X={X.shape}, y positive={y.sum()} ({y.mean()*100:.2f}%)')"
]))

# ── Section 11: Preliminary Classification ────────────────────────────
cells.append(md("s11_md", [
    "## 11. Preliminary Classification\n",
    "Train Logistic Regression and Random Forest with `class_weight='balanced'` on each feature configuration."
]))

cells.append(code("s11_train", [
    "results = {}\n",
    "print('Training classifiers...\\n')\n",
    "\n",
    "for config_name, (X, y) in feature_configs.items():\n",
    "    print(f'=== {config_name} ===')\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)\n",
    "    config_results = {}\n",
    "\n",
    "    for model_name, model in [\n",
    "        ('Logistic Regression', LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)),\n",
    "        ('Random Forest', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_estimators=100))]:\n",
    "\n",
    "        model.fit(X_train, y_train)\n",
    "        y_pred = model.predict(X_test)\n",
    "        y_proba = model.predict_proba(X_test)[:, 1]\n",
    "        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)\n",
    "        auc = roc_auc_score(y_test, y_proba)\n",
    "        print(f'  {model_name}: F1={f1:.3f}, AUC={auc:.3f}')\n",
    "        config_results[model_name] = {'y_test': y_test, 'y_proba': y_proba, 'f1': f1, 'auc': auc}\n",
    "\n",
    "    results[config_name] = config_results\n",
    "    print()\n",
    "\n",
    "print('✅ Classification complete')"
]))

cells.append(code("s11_roc", [
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
    "for ax, (config_name, config_results) in zip(axes.ravel(), results.items()):\n",
    "    for model_name, r in config_results.items():\n",
    "        fpr, tpr, _ = roc_curve(r['y_test'], r['y_proba'])\n",
    "        ax.plot(fpr, tpr, label=f\"{model_name} (AUC={r['auc']:.3f})\", lw=2)\n",
    "    ax.plot([0,1],[0,1],'k--', alpha=0.5)\n",
    "    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')\n",
    "    ax.set_title(f'{config_name}'); ax.legend(loc='lower right')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_FOLDER, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')\n",
    "plt.show()"
]))

cells.append(code("s11_summary", [
    "print('Performance Summary:')\n",
    "print(f'{\"Config\":<20} {\"LR F1\":>8} {\"LR AUC\":>8} {\"RF F1\":>8} {\"RF AUC\":>8}')\n",
    "print('-' * 56)\n",
    "for config_name, cr in results.items():\n",
    "    lr = cr['Logistic Regression']\n",
    "    rf = cr['Random Forest']\n",
    "    print(f'{config_name:<20} {lr[\"f1\"]:8.3f} {lr[\"auc\"]:8.3f} {rf[\"f1\"]:8.3f} {rf[\"auc\"]:8.3f}')"
]))

# ── Section 12: Next Steps ────────────────────────────────────────────
cells.append(md("s12_md", [
    "## 12. Next Steps\n",
    "\n",
    "### Current Status\n",
    "Complete data pipeline built on a 50K-contract sample with PIID-group sampling.\n",
    "\n",
    "### Key Findings\n",
    "1. **Severe class imbalance** — over_budget minority ~0.2%\n",
    "2. **Text features add value** — Combined/LDA configs outperform structured-only\n",
    "3. **Models need improvement** — low F1 scores indicate need for advanced techniques\n",
    "\n",
    "### Immediate Next Steps\n",
    "1. Apply SMOTE/ADASYN for oversampling minority class\n",
    "2. Tune LDA topic count via coherence scores\n",
    "3. Classify the `late` target variable\n",
    "4. Hyperparameter tuning with cross-validation\n",
    "5. Try XGBoost/LightGBM ensemble methods\n",
    "6. Generate SHAP feature importance analysis\n",
    "\n",
    "### Long-term Goals\n",
    "- Scale to full 3.88M contract dataset\n",
    "- Temporal validation across time periods\n",
    "- Deployment as a contract risk assessment tool"
]))

# ── Assemble notebook ─────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Built notebook with {len(cells)} cells")
print("Sections: 0-Header, 1-Imports, 2-Config, 3-Helpers, 4-Schema,")
print("  5-Filter+Sample, 6-Labels, 7-EDA(7.1-7.6), 8-TextPrep,")
print("  9-LDA+TFIDF, 10-Features, 11-Classification, 12-NextSteps")
