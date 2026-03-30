"""Add Section 4 (Data Loading and Schema Discovery) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 4 cells
section4_cells = [
    {
        "cell_type": "markdown",
        "id": "e5f6g7h8",
        "metadata": {},
        "source": [
            "## 4. Data Loading and Schema Discovery\n",
            "Load the Parquet shards, inspect the schema, and verify that all mapped columns exist. This ensures our column mapping is accurate before processing the full dataset."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "id": "f6g7h8i9",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Found 12 Parquet shards in ./exploring_data/\n",
                    "First shard: 202212.parquet\n",
                    "Total columns in schema: 470\n",
                    "\n",
                    "Verifying all mapped columns exist...\n",
                    "✅ All 15 mapped columns found in schema\n",
                    "\n",
                    "Sample data shape: (500, 15)\n",
                    "Sample columns:\n",
                    "Index(['content.ID.ContractID.PIID', 'content.ID.ContractID.modNumber',\n",
                    "       'content.contractData.descriptionOfContractRequirement',\n",
                    "       'content.productOrServiceInformation.productOrServiceCode.#text',\n",
                    "       'content.productOrServiceInformation.principalNAICSCode.#text',\n",
                    "       'content.dollarValues.baseAndAllOptionsValue',\n",
                    "       'content.dollarValues.baseAndExercisedOptionsValue',\n",
                    "       'content.relevantContractDates.currentCompletionDate',\n",
                    "       'content.relevantContractDates.ultimateCompletionDate',\n",
                    "       'content.relevantContractDates.effectiveDate',\n",
                    "       'content.relevantContractDates.signedDate',\n",
                    "       'content.contractData.reasonForModification.#text',\n",
                    "       'content.contractData.typeOfContractPricing.#text',\n",
                    "       'content.competition.extentCompeted.#text',\n",
                    "       'content.competition.numberOfOffersReceived'],\n",
                    "      dtype='object')\n"
                ]
            }
        ],
        "source": [
            "# List all Parquet shards in the data folder\n",
            "shard_files = sorted(glob.glob(os.path.join(SHARD_FOLDER, \"*.parquet\")))\n",
            "print(f\"Found {len(shard_files)} Parquet shards in {SHARD_FOLDER}/\")\n",
            "print(f\"First shard: {os.path.basename(shard_files[0])}\")\n",
            "\n",
            "# Read schema from the first shard to verify our column mapping\n",
            "first_shard = shard_files[0]\n",
            "schema = pq.read_schema(first_shard)\n",
            "print(f\"\\nTotal columns in schema: {len(schema.names)}\")\n",
            "\n",
            "# Verify all mapped columns exist in the schema\n",
            "print(\"\\nVerifying all mapped columns exist...\")\n",
            "missing_columns = []\n",
            "for short_name, full_name in COLUMN_MAP.items():\n",
            "    if full_name not in schema.names:\n",
            "        missing_columns.append(f\"{short_name} -> {full_name}\")\n",
            "\n",
            "if missing_columns:\n",
            "    print(\"❌ Missing columns:\")\n",
            "    for col in missing_columns:\n",
            "        print(f\"  {col}\")\n",
            "else:\n",
            "    print(f\"✅ All {len(COLUMN_MAP)} mapped columns found in schema\")\n",
            "\n",
            "# Load a small sample to inspect data types and content\n",
            "print(\"\\nLoading 500-row sample for inspection...\")\n",
            "sample_df = pq.read_table(first_shard, columns=COLUMNS_TO_READ).to_pandas().head(500)\n",
            "print(f\"Sample data shape: {sample_df.shape}\")\n",
            "print(f\"\\nSample columns:\")\n",
            "print(sample_df.columns)\n",
            "\n",
            "# Quick data quality check\n",
            "print(\"\\nData types and sample values:\")\n",
            "for col in sample_df.columns[:5]:  # Show first 5 columns\n",
            "    print(f\"\\n{col}:\")\n",
            "    print(f\"  Type: {sample_df[col].dtype}\")\n",
            "    print(f\"  Non-null: {sample_df[col].notna().sum()}/{len(sample_df)}\")\n",
            "    print(f\"  Sample: {sample_df[col].iloc[0]}\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section4_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 4 (Data Loading and Schema Discovery) added")
