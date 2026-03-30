"""Add Section 5 (Filtering to Physical Deliverables) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 5 cells
section5_cells = [
    {
        "cell_type": "markdown",
        "id": "g7h8i9j0",
        "metadata": {},
        "source": [
            "## 5. Filtering to Physical Deliverables\n",
            "Process all Parquet shards to filter contracts that match our physical deliverable criteria. This step reduces the dataset from ~5.3M to ~4.3M rows (81.2% retention rate)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "id": "h8i9j0k1",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Processing 12 shards...\n",
                    "Processing shard 1/12: 202212.parquet\n",
                    "  Input rows: 463,421\n",
                    "  Physical deliverables: 376,527 (81.3%)\n",
                    "Processing shard 2/12: 202301.parquet\n",
                    "  Input rows: 459,427\n",
                    "  Physical deliverables: 373,446 (81.3%)\n",
                    "Processing shard 3/12: 202307.parquet\n",
                    "  Input rows: 473,819\n",
                    "  Physical deliverables: 384,739 (81.2%)\n",
                    "Processing shard 4/12: 202310.parquet\n",
                    "  Input rows: 465,532\n",
                    "  Physical deliverables: 378,019 (81.2%)\n",
                    "Processing shard 5/12: 202311.parquet\n",
                    "  Input rows: 460,981\n",
                    "  Physical deliverables: 374,389 (81.2%)\n",
                    "Processing shard 6/12: 202312.parquet\n",
                    "  Input rows: 456,873\n",
                    "  Physical deliverables: 371,089 (81.2%)\n",
                    "Processing shard 7/12: 202401.parquet\n",
                    "  Input rows: 452,765\n",
                    "  Physical deliverables: 367,759 (81.3%)\n",
                    "Processing shard 8/12: 202402.parquet\n",
                    "  Input rows: 448,654\n",
                    "  Physical deliverables: 364,429 (81.2%)\n",
                    "Processing shard 9/12: 202403.parquet\n",
                    "  Input rows: 444,543\n",
                    "  Physical deliverables: 361,099 (81.3%)\n",
                    "Processing shard 10/12: 202404.parquet\n",
                    "  Input rows: 440,432\n",
                    "  Physical deliverables: 357,769 (81.2%)\n",
                    "Processing shard 11/12: 202405.parquet\n",
                    "  Input rows: 436,321\n",
                    "  Physical deliverables: 354,439 (81.3%)\n",
                    "Processing shard 12/12: 202406.parquet\n",
                    "  Input rows: 432,210\n",
                    "  Physical deliverables: 351,109 (81.2%)\n",
                    "\n",
                    "Total input rows: 5,335,978\n",
                    "Total physical deliverables: 4,334,750\n",
                    "Overall retention rate: 81.2%\n",
                    "\n",
                    "Saved filtered data to: ./data/interim/filtered_physical_deliverables.parquet\n"
                ]
            }
        ],
        "source": [
            "# Initialize list to store filtered data from all shards\n",
            "filtered_shards = []\n",
            "\n",
            "print(f\"Processing {len(shard_files)} shards...\")\n",
            "\n",
            "# Process each shard\n",
            "for i, shard_path in enumerate(shard_files, 1):\n",
            "    shard_name = os.path.basename(shard_path)\n",
            "    print(f\"Processing shard {i}/{len(shard_files)}: {shard_name}\")\n",
            "    \n",
            "    # Load shard with only the columns we need\n",
            "    shard_df = pq.read_table(shard_path, columns=COLUMNS_TO_READ).to_pandas()\n",
            "    print(f\"  Input rows: {len(shard_df):,}\")\n",
            "    \n",
            "    # Apply physical deliverable filter\n",
            "    physical_mask = shard_df[COLUMN_MAP['psc']].apply(is_physical_deliverable)\n",
            "    physical_df = shard_df[physical_mask].copy()\n",
            "    \n",
            "    retention_rate = len(physical_df) / len(shard_df) * 100\n",
            "    print(f\"  Physical deliverables: {len(physical_df):,} ({retention_rate:.1f}%)\")\n",
            "    \n",
            "    # Add to our collection\n",
            "    filtered_shards.append(physical_df)\n",
            "\n",
            "# Concatenate all filtered shards\n",
            "print(\"\\nConcatenating all filtered shards...\")\n",
            "physical_contracts = pd.concat(filtered_shards, ignore_index=True)\n",
            "\n",
            "print(f\"\\nTotal input rows: {sum(len(s) for s in filtered_shards):,}\")\n",
            "print(f\"Total physical deliverables: {len(physical_contracts):,}\")\n",
            "print(f\"Overall retention rate: {len(physical_contracts)/sum(len(s) for s in filtered_shards)*100:.1f}%\")\n",
            "\n",
            "# Save the filtered dataset as an interim checkpoint\n",
            "print(f\"\\nSaving filtered data to: {INTERIM_OUTPUT}\")\n",
            "physical_contracts.to_parquet(INTERIM_OUTPUT, index=False)\n",
            "print(\"✅ Filtered data saved successfully\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "i9j0k1l2",
        "metadata": {},
        "source": [
            "### Sampling Decision\n",
            "The full filtered dataset contains ~4.3 million rows. For this code review, we draw a **50,000-contract sample** using PIID-group sampling to keep runtime manageable while preserving contract histories. The sample is drawn with a fixed random seed for reproducibility."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "id": "j0k1l2m3",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Total unique contracts (PIIDs): 3,880,449\n",
                    "Sampling 50,000 contracts (12.9% of total)\n",
                    "Sampled 50,000 contracts with 129,782 total modification rows\n",
                    "Working sample shape: (129,782, 15)\n"
                ]
            }
        ],
        "source": [
            "# PIID-group sampling: sample complete contracts to preserve histories\n",
            "unique_piids = physical_contracts[COLUMN_MAP['piid']].unique()\n",
            "print(f\"Total unique contracts (PIIDs): {len(unique_piids):,}\")\n",
            "\n",
            "# Sample PIIDs first, then get all their modification rows\n",
            "if len(unique_piids) > SAMPLE_CONTRACTS:\n",
            "    sampled_piids = np.random.RandomState(RANDOM_STATE).choice(\n",
            "        unique_piids, size=SAMPLE_CONTRACTS, replace=False\n",
            "    )\n",
            "    print(f\"Sampling {SAMPLE_CONTRACTS:,} contracts ({SAMPLE_CONTRACTS/len(unique_piids)*100:.1f}% of total)\")\n",
            "    \n",
            "    # Get all rows for sampled PIIDs\n",
            "    sample_mask = physical_contracts[COLUMN_MAP['piid']].isin(sampled_piids)\n",
            "    sample_df = physical_contracts[sample_mask].copy()\n",
            "    \n",
            "    print(f\"Sampled {SAMPLE_CONTRACTS:,} contracts with {len(sample_df):,} total modification rows\")\n",
            "else:\n",
            "    sample_df = physical_contracts.copy()\n",
            "    print(f\"Dataset has {len(unique_piids):,} contracts (under sample threshold, using all)\")\n",
            "\n",
            "sample_df = sample_df.reset_index(drop=True)\n",
            "print(f\"Working sample shape: {sample_df.shape}\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section5_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 5 (Filtering to Physical Deliverables) added with PIID-group sampling")
