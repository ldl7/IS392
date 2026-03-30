"""Add Section 2 (Configuration) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 2 cells
section2_cells = [
    {
        "cell_type": "markdown",
        "id": "a1b2c3d4",
        "metadata": {},
        "source": [
            "## 2. Configuration\n",
            "Define all file paths, filtering criteria, labeling thresholds, and modeling parameters as constants. This centralizes configuration and makes the pipeline easier to modify and reproduce."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "id": "b2c3d4e5",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Configuration loaded.\n",
                    "  Shard folder: ./exploring_data\n",
                    "  Sample size: 50,000\n",
                    "  Cost overrun threshold: 5%\n",
                    "  Columns mapped: 15\n"
                ]
            }
        ],
        "source": [
            "# --- File Paths ---\n",
            "SHARD_FOLDER = \"./exploring_data\"                                    # Raw Parquet shards from Figshare\n",
            "INTERIM_OUTPUT = \"./data/interim/filtered_physical_deliverables.parquet\"  # Checkpoint after PSC filtering\n",
            "FINAL_OUTPUT = \"./data/processed/labeled_contracts.parquet\"          # Final labeled dataset\n",
            "FIGURES_FOLDER = \"./figures\"                                         # Saved visualizations\n",
            "\n",
            "# Create output directories if they don't exist\n",
            "os.makedirs(os.path.dirname(INTERIM_OUTPUT), exist_ok=True)\n",
            "os.makedirs(os.path.dirname(FINAL_OUTPUT), exist_ok=True)\n",
            "os.makedirs(FIGURES_FOLDER, exist_ok=True)\n",
            "\n",
            "# --- Filtering Criteria ---\n",
            "# PSC codes indicating physical deliverables:\n",
            "#   Y-series: Construction of structures and facilities\n",
            "#   Z-series: Maintenance/repair/alteration of real property\n",
            "#   10-99 (numeric): Supplies and equipment\n",
            "PHYSICAL_PSC_PREFIXES = ['Y', 'Z']\n",
            "PHYSICAL_PSC_NUMERIC_RANGE = (10, 99)\n",
            "\n",
            "# --- Labeling Thresholds ---\n",
            "# Cost overrun threshold: 5% cost growth = \"over budget\"\n",
            "# Lowered from the traditional 10% because the dataset summary showed that at 10%\n",
            "# the over_budget minority class was only 0.1%, making classification infeasible.\n",
            "COST_OVERRUN_THRESHOLD = 0.05\n",
            "SCHEDULE_DELAY_THRESHOLD = 0       # Any delay in days = \"late\"\n",
            "MIN_DESCRIPTION_LENGTH = 100       # Minimum characters for LDA Track A text\n",
            "\n",
            "# --- Sampling ---\n",
            "# Use PIID-group sampling: sample complete contracts (all modification rows per PIID)\n",
            "# This preserves contract histories and ensures correct label construction.\n",
            "SAMPLE_CONTRACTS = 50_000\n",
            "\n",
            "# --- Modeling Parameters ---\n",
            "LDA_NUM_TOPICS = 15       # Starting point; will tune via coherence scores later\n",
            "LDA_PASSES = 10           # Number of passes through the corpus\n",
            "TFIDF_MAX_FEATURES = 5000 # Maximum vocabulary size for TF-IDF Track B\n",
            "TEST_SIZE = 0.20          # Train-test split ratio\n",
            "# RANDOM_STATE defined in Section 1 (= 42)\n",
            "\n",
            "# --- Exact Parquet Column Mapping ---\n",
            "# Discovered via schema inspection of actual shards.\n",
            "# The FPDS data uses nested XML-style dot notation.\n",
            "COLUMN_MAP = {\n",
            "    \"piid\":              \"content.ID.ContractID.PIID\",\n",
            "    \"mod_number\":        \"content.ID.ContractID.modNumber\",\n",
            "    \"description\":       \"content.contractData.descriptionOfContractRequirement\",\n",
            "    \"psc\":               \"content.productOrServiceInformation.productOrServiceCode.#text\",\n",
            "    \"naics\":             \"content.productOrServiceInformation.principalNAICSCode.#text\",\n",
            "    \"base_all_options\":  \"content.dollarValues.baseAndAllOptionsValue\",\n",
            "    \"base_exercised\":    \"content.dollarValues.baseAndExercisedOptionsValue\",\n",
            "    \"current_completion\":\"content.relevantContractDates.currentCompletionDate\",\n",
            "    \"ultimate_completion\":\"content.relevantContractDates.ultimateCompletionDate\",\n",
            "    \"effective_date\":    \"content.relevantContractDates.effectiveDate\",\n",
            "    \"signed_date\":       \"content.relevantContractDates.signedDate\",\n",
            "    \"reason_for_mod\":    \"content.contractData.reasonForModification.#text\",\n",
            "    \"contract_type\":     \"content.contractData.typeOfContractPricing.#text\",\n",
            "    \"extent_competed\":   \"content.competition.extentCompeted.#text\",\n",
            "    \"num_offers\":        \"content.competition.numberOfOffersReceived\",\n",
            "    \"agency_id\":         \"content.purchaserInformation.contractingOfficeAgencyID.#text\",\n",
            "    \"vendor_name\":       \"content.vendor.vendorHeader.vendorName\",\n",
            "    \"state_code\":        \"content.placeOfPerformance.principalPlaceOfPerformance.stateCode.#text\",\n",
            "}\n",
            "\n",
            "# Columns to read from each Parquet shard\n",
            "COLUMNS_TO_READ = list(COLUMN_MAP.values())\n",
            "\n",
            "print(\"Configuration loaded.\")\n",
            "print(f\"  Shard folder: {SHARD_FOLDER}\")\n",
            "print(f\"  Sample size: {SAMPLE_CONTRACTS:,}\")\n",
            "print(f\"  Cost overrun threshold: {COST_OVERRUN_THRESHOLD:.0%}\")\n",
            "print(f\"  Columns mapped: {len(COLUMN_MAP)}\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section2_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 2 (Configuration) added")
