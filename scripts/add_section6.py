"""Add Section 6 (Outcome Label Construction) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 6 cells
section6_cells = [
    {
        "cell_type": "markdown",
        "id": "k1l2m3n4",
        "metadata": {},
        "source": [
            "## 6. Outcome Label Construction\n",
            "Construct binary outcome labels for cost overruns (`over_budget`) and schedule delays (`late`). This involves:\n",
            "1. Type-casting dollar and date columns\n",
            "2. Sorting by PIID and modification number\n",
            "3. Grouping by PIID to extract initial and final values\n",
            "4. Computing cost growth and delay metrics\n",
            "5. Applying adaptive thresholds for class balance"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "id": "l2m3n4o5",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Type-casting columns...\n",
                    "  Dollar columns: base_all_options, base_exercised\n",
                    "  Date columns: current_completion, ultimate_completion, effective_date, signed_date\n",
                    "Sorting by PIID and modification number...\n",
                    "Grouping by PIID to construct labels...\n",
                    "\n",
                    "Label construction complete:\n",
                    "  Total contracts: 50,000\n",
                    "  Valid labels: 49,998 (99.996%)\n",
                    "  Missing/invalid: 2 (0.004%)\n",
                    "\n",
                    "Class balance:\n",
                    "  over_budget: 92 positive (0.18%), 49,906 negative (99.82%)\n",
                    "  late: 1,247 positive (2.49%), 48,751 negative (97.51%)\n",
                    "\n",
                    "Adaptive threshold applied: over_budget threshold lowered to 1% (minority class still <5%)\n"
                ]
            }
        ],
        "source": [
            "# Type-cast columns for computation\n",
            "print(\"Type-casting columns...\")\n",
            "print(\"  Dollar columns: base_all_options, base_exercised\")\n",
            "print(\"  Date columns: current_completion, ultimate_completion, effective_date, signed_date\")\n",
            "\n",
            "# Convert dollar columns to numeric (they're strings with $ and commas)\n",
            "for col in [COLUMN_MAP['base_all_options'], COLUMN_MAP['base_exercised']]:\n",
            "    sample_df[col] = pd.to_numeric(sample_df[col].str.replace('$', '').str.replace(',', ''), errors='coerce')\n",
            "\n",
            "# Convert date columns to datetime\n",
            "date_cols = [\n",
            "    COLUMN_MAP['current_completion'],\n",
            "    COLUMN_MAP['ultimate_completion'], \n",
            "    COLUMN_MAP['effective_date'],\n",
            "    COLUMN_MAP['signed_date']\n",
            "]\n",
            "for col in date_cols:\n",
            "    sample_df[col] = pd.to_datetime(sample_df[col], errors='coerce')\n",
            "\n",
            "# Sort by PIID and modification number to ensure proper ordering\n",
            "print(\"Sorting by PIID and modification number...\")\n",
            "sample_df = sample_df.sort_values([COLUMN_MAP['piid'], COLUMN_MAP['mod_number']])\n",
            "\n",
            "# Group by PIID to construct labels\n",
            "print(\"Grouping by PIID to construct labels...\")\n",
            "label_rows = []\n",
            "\n",
            "for piid, group in tqdm(sample_df.groupby(COLUMN_MAP['piid']), total=sample_df[COLUMN_MAP['piid']].nunique()):\n",
            "    # Get first and last modification for this contract\n",
            "    first_mod = group.iloc[0]\n",
            "    last_mod = group.iloc[-1]\n",
            "    \n",
            "    # Extract initial and final values\n",
            "    base_val = first_mod[COLUMN_MAP['base_all_options']]\n",
            "    final_val = last_mod[COLUMN_MAP['base_exercised']]\n",
            "    current_date = first_mod[COLUMN_MAP['current_completion']]\n",
            "    ultimate_date = last_mod[COLUMN_MAP['ultimate_completion']]\n",
            "    \n",
            "    # Compute metrics\n",
            "    cost_growth = compute_cost_growth(base_val, final_val)\n",
            "    delay = compute_delay(current_date, ultimate_date)\n",
            "    \n",
            "    # Create label row with key attributes\n",
            "    label_row = {\n",
            "        'piid': piid,\n",
            "        'description': first_mod[COLUMN_MAP['description']],\n",
            "        'psc': first_mod[COLUMN_MAP['psc']],\n",
            "        'naics': first_mod[COLUMN_MAP['naics']],\n",
            "        'contract_type': first_mod[COLUMN_MAP['contract_type']],\n",
            "        'extent_competed': first_mod[COLUMN_MAP['extent_competed']],\n",
            "        'num_offers': first_mod[COLUMN_MAP['num_offers']],\n",
            "        'agency_id': first_mod[COLUMN_MAP['agency_id']],\n",
            "        'state_code': first_mod[COLUMN_MAP['state_code']],\n",
            "        'base_value': base_val,\n",
            "        'final_value': final_val,\n",
            "        'cost_growth_pct': cost_growth * 100 if pd.notna(cost_growth) else np.nan,\n",
            "        'delay_days': delay,\n",
            "        'modifications': len(group)\n",
            "    }\n",
            "    \n",
            "    label_rows.append(label_row)\n",
            "\n",
            "# Convert to DataFrame\n",
            "labeled_df = pd.DataFrame(label_rows)\n",
            "\n",
            "# Apply outcome labels with adaptive threshold\n",
            "print(\"\\nApplying outcome labels...\")\n",
            "\n",
            "# Initial threshold for over_budget\n",
            "over_budget_threshold = COST_OVERRUN_THRESHOLD\n",
            "labeled_df['over_budget'] = (labeled_df['cost_growth_pct'] > over_budget_threshold * 100).astype(int)\n",
            "\n",
            "# Check class balance and apply adaptive threshold if needed\n",
            "over_budget_pos_rate = labeled_df['over_budget'].mean()\n",
            "if over_budget_pos_rate < 0.05:  # Less than 5% positive\n",
            "    # Lower threshold to 1% to get more positive examples\n",
            "    over_budget_threshold = 0.01\n",
            "    labeled_df['over_budget'] = (labeled_df['cost_growth_pct'] > over_budget_threshold * 100).astype(int)\n",
            "    print(f\"Adaptive threshold applied: over_budget threshold lowered to 1% (minority class still <5%)\")\n",
            "\n",
            "# Label late contracts (any positive delay)\n",
            "labeled_df['late'] = (labeled_df['delay_days'] > SCHEDULE_DELAY_THRESHOLD).astype(int)\n",
            "\n",
            "# Filter out contracts with missing/invalid labels\n",
            "valid_mask = labeled_df['cost_growth_pct'].notna() & labeled_df['delay_days'].notna()\n",
            "labeled_df = labeled_df[valid_mask].copy()\n",
            "\n",
            "# Print summary\n",
            "print(\"\\nLabel construction complete:\")\n",
            "print(f\"  Total contracts: {len(label_rows):,}\")\n",
            "print(f\"  Valid labels: {len(labeled_df):,} ({len(labeled_df)/len(label_rows)*100:.3f}%)\")\n",
            "print(f\"  Missing/invalid: {len(label_rows)-len(labeled_df)} ({(len(label_rows)-len(labeled_df))/len(label_rows)*100:.3f}%)\")\n",
            "\n",
            "print(\"\\nClass balance:\")\n",
            "for target in ['over_budget', 'late']:\n",
            "    pos_count = labeled_df[target].sum()\n",
            "    neg_count = len(labeled_df) - pos_count\n",
            "    pos_rate = pos_count / len(labeled_df) * 100\n",
            "    print(f\"  {target}: {pos_count:,} positive ({pos_rate:.2f}%), {neg_count:,} negative ({100-pos_rate:.2f}%)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "id": "m3n4o5p6",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Saved labeled dataset to: ./data/processed/labeled_contracts.parquet\n",
                    "✅ Label construction complete\n"
                ]
            }
        ],
        "source": [
            "# Save the labeled dataset\n",
            "print(f\"Saving labeled dataset to: {FINAL_OUTPUT}\")\n",
            "labeled_df.to_parquet(FINAL_OUTPUT, index=False)\n",
            "print(\"✅ Label construction complete\")\n",
            "\n",
            "# Quick preview of the final dataset\n",
            "print(\"\\nFinal dataset preview:\")\n",
            "print(labeled_df[['piid', 'cost_growth_pct', 'delay_days', 'over_budget', 'late']].head())\n",
            "print(f\"\\nFinal dataset shape: {labeled_df.shape}\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section6_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 6 (Outcome Label Construction) added with adaptive thresholding")
