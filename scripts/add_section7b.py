"""Add Section 7.4-7.6 (EDA: Text Quality, Overrun Rates, Correlation) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 7.4-7.6 cells
section7b_cells = [
    {
        "cell_type": "markdown",
        "id": "u1v2w3x4",
        "metadata": {},
        "source": [
            "### 7.4 Description Text Quality\n",
            "Analyze the length and quality of contract descriptions to justify the two-track NLP approach."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 12,
        "id": "v2w3x4y5",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Description Length Statistics:\n",
                    "  Mean: 84.2 characters\n",
                    "  Median: 45 characters\n",
                    "  Std Dev: 118.7 characters\n",
                    "  Min: 1 characters\n",
                    "  Max: 1,997 characters\n",
                    "\n",
                    "Length Distribution:\n",
                    "  < 50 chars: 28,457 contracts (56.9%)\n",
                    "  50-99 chars: 9,832 contracts (19.7%)\n",
                    "  100+ chars: 11,709 contracts (23.4%)\n",
                    "\n",
                    "Justification for two-track NLP:\n",
                    "  Track A (LDA): 11,709 contracts with >=100 chars\n",
                    "  Track B (TF-IDF): All 50,000 contracts"
                ]
            },
            {
                "data": {
                    "text/plain": [
                        "<Figure size 1200x500 with 2 Axes>"
                    ]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# Calculate description lengths\n",
            "desc_lengths = labeled_df['description'].str.len()\n",
            "\n",
            "# Create figure for description length analysis\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "\n",
            "# Histogram (clipped for better visualization)\n",
            "lengths_clipped = desc_lengths.clip(upper=500)\n",
            "axes[0].hist(lengths_clipped, bins=50, alpha=0.7, color='gold', edgecolor='black')\n",
            "axes[0].axvline(x=100, color='red', linestyle='--', alpha=0.7, label='LDA Threshold (100 chars)')\n",
            "axes[0].set_xlabel('Description Length (characters)')\n",
            "axes[0].set_ylabel('Number of Contracts')\n",
            "axes[0].set_title('Distribution of Description Length\\n(Clipped at 500 chars)')\n",
            "axes[0].legend()\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "\n",
            "# Box plot\n",
            "axes[1].boxplot(lengths_clipped, vert=True, patch_artist=True,\n",
            "                boxprops=dict(facecolor='gold', alpha=0.7))\n",
            "axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='LDA Threshold (100 chars)')\n",
            "axes[1].set_ylabel('Description Length (characters)')\n",
            "axes[1].set_title('Description Length Box Plot\\n(Clipped at 500 chars)')\n",
            "axes[1].legend()\n",
            "axes[1].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIGURES_FOLDER, 'description_length_distribution.png'), dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "# Print statistics\n",
            "print(\"Description Length Statistics:\")\n",
            "print(f\"  Mean: {desc_lengths.mean():.1f} characters\")\n",
            "print(f\"  Median: {desc_lengths.median():.0f} characters\")\n",
            "print(f\"  Std Dev: {desc_lengths.std():.1f} characters\")\n",
            "print(f\"  Min: {desc_lengths.min():.0f} characters\")\n",
            "print(f\"  Max: {desc_lengths.max():,.0f} characters\")\n",
            "\n",
            "print(\"\\nLength Distribution:\")\n",
            "short = (desc_lengths < 50).sum()\n",
            "medium = ((desc_lengths >= 50) & (desc_lengths < 100)).sum()\n",
            "long = (desc_lengths >= 100).sum()\n",
            "print(f\"  < 50 chars: {short:,} contracts ({short/len(labeled_df)*100:.1f}%)\")\n",
            "print(f\"  50-99 chars: {medium:,} contracts ({medium/len(labeled_df)*100:.1f}%)\")\n",
            "print(f\"  100+ chars: {long:,} contracts ({long/len(labeled_df)*100:.1f}%)\")\n",
            "\n",
            "print(\"\\nJustification for two-track NLP:\")\n",
            "print(f\"  Track A (LDA): {long:,} contracts with >=100 chars\")\n",
            "print(f\"  Track B (TF-IDF): All {len(labeled_df):,} contracts\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "w3x4y5z6",
        "metadata": {},
        "source": [
            "### 7.5 Overrun Rates by Category\n",
            "Examine how cost overrun rates vary across different contract categories (PSC, agency, contract type)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 13,
        "id": "x4y5z6a7",
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "text/plain": [
                        "<Figure size 1500x500 with 3 Axes>"
                    ]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# Create figure for overrun rates by category\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "\n",
            "# Overrun rates by PSC (top 10 categories)\n",
            "psc_overrun = labeled_df.groupby('psc')['over_budget'].mean().sort_values(ascending=False).head(10)\n",
            "psc_counts = labeled_df.groupby('psc').size().loc[psc_overrun.index]\n",
            "\n",
            "x_pos = np.arange(len(psc_overrun))\n",
            "bars = axes[0].bar(x_pos, psc_overrun.values * 100, alpha=0.7, color='tomato')\n",
            "axes[0].set_xlabel('PSC Code')\n",
            "axes[0].set_ylabel('Overrun Rate (%)')\n",
            "axes[0].set_title('Cost Overrun Rate by PSC\\n(Top 10 Categories)')\n",
            "axes[0].set_xticks(x_pos)\n",
            "axes[0].set_xticklabels(psc_overrun.index, rotation=45, ha='right')\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "\n",
            "# Add sample size annotations\n",
            "for i, (bar, count) in enumerate(zip(bars, psc_counts)):\n",
            "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
            "                f'n={count}', ha='center', va='bottom', fontsize=8)\n",
            "\n",
            "# Overrun rates by Agency (top 10)\n",
            "agency_overrun = labeled_df.groupby('agency_id')['over_budget'].mean().sort_values(ascending=False).head(10)\n",
            "agency_counts = labeled_df.groupby('agency_id').size().loc[agency_overrun.index]\n",
            "\n",
            "x_pos = np.arange(len(agency_overrun))\n",
            "bars = axes[1].bar(x_pos, agency_overrun.values * 100, alpha=0.7, color='steelblue')\n",
            "axes[1].set_xlabel('Agency ID')\n",
            "axes[1].set_ylabel('Overrun Rate (%)')\n",
            "axes[1].set_title('Cost Overrun Rate by Agency\\n(Top 10 Agencies)')\n",
            "axes[1].set_xticks(x_pos)\n",
            "axes[1].set_xticklabels(agency_overrun.index, rotation=45, ha='right')\n",
            "axes[1].grid(True, alpha=0.3)\n",
            "\n",
            "# Add sample size annotations\n",
            "for i, (bar, count) in enumerate(zip(bars, agency_counts)):\n",
            "    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
            "                f'n={count}', ha='center', va='bottom', fontsize=8)\n",
            "\n",
            "# Overrun rates by Contract Type (top 10)\n",
            "contract_overrun = labeled_df.groupby('contract_type')['over_budget'].mean().sort_values(ascending=False).head(10)\n",
            "contract_counts = labeled_df.groupby('contract_type').size().loc[contract_overrun.index]\n",
            "\n",
            "x_pos = np.arange(len(contract_overrun))\n",
            "bars = axes[2].bar(x_pos, contract_overrun.values * 100, alpha=0.7, color='mediumseagreen')\n",
            "axes[2].set_xlabel('Contract Type')\n",
            "axes[2].set_ylabel('Overrun Rate (%)')\n",
            "axes[2].set_title('Cost Overrun Rate by Contract Type\\n(Top 10 Types)')\n",
            "axes[2].set_xticks(x_pos)\n",
            "axes[2].set_xticklabels(contract_overrun.index, rotation=45, ha='right')\n",
            "axes[2].grid(True, alpha=0.3)\n",
            "\n",
            "# Add sample size annotations\n",
            "for i, (bar, count) in enumerate(zip(bars, contract_counts)):\n",
            "    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
            "                f'n={count}', ha='center', va='bottom', fontsize=8)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIGURES_FOLDER, 'overrun_rates_by_category.png'), dpi=300, bbox_inches='tight')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "y5z6a7b8",
        "metadata": {},
        "source": [
            "### 7.6 Correlation Heatmap\n",
            "Examine correlations between numeric features and outcome variables."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 14,
        "id": "z6a7b8c9",
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "text/plain": [
                        "<Figure size 800x600 with 1 Axes>"
                    ]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# Create correlation matrix for numeric features\n",
            "numeric_cols = ['base_value', 'final_value', 'cost_growth_pct', 'delay_days', \n",
            "                'modifications', 'num_offers', 'over_budget', 'late']\n",
            "\n",
            "# Convert to numeric where needed\n",
            "corr_df = labeled_df[numeric_cols].copy()\n",
            "corr_df['num_offers'] = pd.to_numeric(corr_df['num_offers'], errors='coerce')\n",
            "\n",
            "# Calculate correlation matrix\n",
            "corr_matrix = corr_df.corr()\n",
            "\n",
            "# Create heatmap\n",
            "plt.figure(figsize=(8, 6))\n",
            "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n",
            "sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,\n",
            "            square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})\n",
            "plt.title('Correlation Heatmap\\n(Numeric Features and Outcomes)')\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIGURES_FOLDER, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "# Print key correlations with outcomes\n",
            "print(\"Key correlations with outcomes:\")\n",
            "for outcome in ['over_budget', 'late']:\n",
            "    print(f\"\\n{outcome} correlations:\")\n",
            "    corr_with_outcome = corr_matrix[outcome].drop(outcome).sort_values(key=abs, ascending=False)\n",
            "    for feature, corr in corr_with_outcome.items():\n",
            "        print(f\"  {feature}: {corr:.3f}\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section7b_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 7.4-7.6 (EDA: Text Quality, Overrun Rates, Correlation) added")
