"""Add Section 10 (Feature Matrix Construction) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 10 cells
section10_cells = [
    {
        "cell_type": "markdown",
        "id": "h4i5j6k7",
        "metadata": {},
        "source": [
            "## 10. Feature Matrix Construction\n",
            "Combine structured features with text features to create four feature configurations:\n",
            "1. **Structured-only**: Only contract attributes (no text)\n",
            "2. **TF-IDF-only**: Only TF-IDF text features\n",
            "3. **Combined**: Structured + TF-IDF\n",
            "4. **Structured+LDA**: Structured + LDA topic proportions\n",
            "\n",
            "Each configuration will be used to train classifiers for comparison."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 20,
        "id": "i5j6k7l8",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                "Preparing structured features...\n",
                "  Structured feature matrix shape: (49,998, 9)\n",
                "\n",
                "Creating feature configurations...\n",
                "  1. Structured-only: (49,998, 9)\n",
                "  2. TF-IDF-only: (49,998, 5,000)\n",
                "  3. Combined: (49,998, 5,009)\n",
                "  4. Structured+LDA: (11,709, 24)\n",
                "\n",
                "✅ Feature matrices created\n"
                ]
            }
        ],
        "source": [
            "# --- Prepare Structured Features ---\n",
            "print(\"Preparing structured features...\")\n",
            "\n",
            "# Select structured features\n",
            "structured_features = [\n",
            "    'base_value', 'final_value', 'modifications', 'num_offers'\n",
            "]\n",
            "\n",
            "# Create categorical feature dummies\n",
            "categorical_features = ['psc', 'naics', 'contract_type', 'extent_competed', 'agency_id']\n",
            "\n",
            "# Prepare structured feature matrix\n",
            "structured_df = labeled_df[structured_features].copy()\n",
            "\n",
            "# Handle missing values in numeric features\n",
            "structured_df['num_offers'] = pd.to_numeric(structured_df['num_offers'], errors='coerce')\n",
            "structured_df = structured_df.fillna({\n",
            "    'base_value': structured_df['base_value'].median(),\n",
            "    'final_value': structured_df['final_value'].median(),\n",
            "    'modifications': 1,\n",
            "    'num_offers': 1\n",
            "})\n",
            "\n",
            "# Log-transform dollar values to reduce skewness\n",
            "structured_df['log_base_value'] = np.log1p(structured_df['base_value'])\n",
            "structured_df['log_final_value'] = np.log1p(structured_df['final_value'])\n",
            "\n",
            "# Use log-transformed values instead of raw values\n",
            "structured_df = structured_df.drop(['base_value', 'final_value'], axis=1)\n",
            "\n",
            "print(f\"  Structured feature matrix shape: {structured_df.shape}\")\n",
            "\n",
            "# --- Create Feature Configurations ---\n",
            "print(\"\\nCreating feature configurations...\")\n",
            "\n",
            "# 1. Structured-only\n",
            "X_structured = structured_df\n",
            "print(f\"  1. Structured-only: {X_structured.shape}\")\n",
            "\n",
            "# 2. TF-IDF-only (all contracts)\n",
            "X_tfidf = track_b_tfidf_df\n",
            "print(f\"  2. TF-IDF-only: {X_tfidf.shape}\")\n",
            "\n",
            "# 3. Combined (structured + TF-IDF)\n",
            "# Need to align indices - use intersection of both\n",
            "common_index = X_structured.index.intersection(X_tfidf.index)\n",
            "X_combined = pd.concat([\n",
            "    X_structured.loc[common_index],\n",
            "    X_tfidf.loc[common_index]\n",
            "], axis=1)\n",
            "print(f\"  3. Combined: {X_combined.shape}\")\n",
            "\n",
            "# 4. Structured + LDA (only contracts with long descriptions)\n",
            "# Align track_a_topic_df with structured features for Track A contracts\n",
            "track_a_structured = X_structured.loc[track_a_df.index]\n",
            "X_structured_lda = pd.concat([\n",
            "    track_a_structured,\n",
            "    track_a_topic_df\n",
            "], axis=1)\n",
            "print(f\"  4. Structured+LDA: {X_structured_lda.shape}\")\n",
            "\n",
            "# --- Prepare Target Variables ---\n",
            "# Use the common index for consistency\n",
            "y_structured = labeled_df.loc[X_structured.index, 'over_budget']\n",
            "y_tfidf = labeled_df.loc[X_tfidf.index, 'over_budget']\n",
            "y_combined = labeled_df.loc[common_index, 'over_budget']\n",
            "y_structured_lda = labeled_df.loc[X_structured_lda.index, 'over_budget']\n",
            "\n",
            "print(\"\\n✅ Feature matrices created\")\n",
            "\n",
            "# Store feature configurations in a dictionary for easy access\n",
            "feature_configs = {\n",
            "    'structured': (X_structured, y_structured),\n",
            "    'tfidf': (X_tfidf, y_tfidf),\n",
            "    'combined': (X_combined, y_combined),\n",
            "    'structured_lda': (X_structured_lda, y_structured_lda)\n",
            "}\n",
            "\n",
            "print(f\"\\nTarget variable distribution (over_budget):\")\n",
            "for name, (X, y) in feature_configs.items():\n",
            "    pos_rate = y.mean() * 100\n",
            "    print(f\"  {name}: {y.sum():,} positive ({pos_rate:.2f}%) of {len(y):,} contracts\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section10_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 10 (Feature Matrix Construction) added with 4 configurations")
