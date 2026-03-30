"""Fix the num_offers handling in the feature construction section."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find and update the feature construction cell (cell 20)
cells = notebook["cells"]
for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code" and i == 20:  # Section 10 code cell
        # Get the source and update it to properly handle num_offers
        source = cell["source"]
        
        # Replace the source with fixed version
        new_source = [
            "# --- Prepare Structured Features ---\n",
            "print(\"Preparing structured features...\")\n",
            "\n",
            "# Select structured features\n",
            "structured_features = [\n",
            "    'base_value', 'final_value', 'modifications', 'num_offers'\n",
            "]\n",
            "\n",
            "# Prepare structured feature matrix\n",
            "structured_df = labeled_df[structured_features].copy()\n",
            "\n",
            "# Handle missing values in numeric features\n",
            "# Convert num_offers to numeric, handling 'None' strings\n",
            "structured_df['num_offers'] = structured_df['num_offers'].replace('None', np.nan)\n",
            "structured_df['num_offers'] = pd.to_numeric(structured_df['num_offers'], errors='coerce')\n",
            "\n",
            "# Fill missing values with appropriate defaults\n",
            "structured_df = structured_df.fillna({\n",
            "    'base_value': structured_df['base_value'].median(),\n",
            "    'final_value': structured_df['final_value'].median(),\n",
            "    'modifications': 1,\n",
            "    'num_offers': 1  # Default to 1 offer when missing\n",
            "})\n",
            "\n",
            "# Check for any remaining NaN values\n",
            "print(f\"Missing values after imputation:\")\n",
            "for col in structured_df.columns:\n",
            "    nan_count = structured_df[col].isna().sum()\n",
            "    if nan_count > 0:\n",
            "        print(f\"  {col}: {nan_count} NaN values\")\n",
            "    else:\n",
            "        print(f\"  {col}: No NaN values\")\n",
            "\n",
            "# Log-transform dollar values to reduce skewness\n",
            "structured_df['log_base_value'] = np.log1p(structured_df['base_value'])\n",
            "structured_df['log_final_value'] = np.log1p(structured_df['final_value'])\n",
            "\n",
            "# Use log-transformed values instead of raw values\n",
            "structured_df = structured_df.drop(['base_value', 'final_value'], axis=1)\n",
            "\n",
            "print(f\"\\nStructured feature matrix shape: {structured_df.shape}\")\n"
        ]
        
        # Add the rest of the original source (from feature configurations onward)
        continue_source = False
        for line in source:
            if "# --- Create Feature Configurations ---" in line:
                continue_source = True
            if continue_source:
                new_source.append(line)
        
        cell["source"] = new_source
        break

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Fixed num_offers handling in feature construction")
