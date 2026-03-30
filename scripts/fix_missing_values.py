"""Fix missing values in the feature construction section."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find and update the feature construction cell (cell 20)
cells = notebook["cells"]
for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code" and i == 20:  # Section 10 code cell
        # Get the source and update it to handle missing values
        source = cell["source"]
        
        # Find the structured features preparation section and add missing value handling
        new_source = []
        for line in source:
            new_source.append(line)
            # Add missing value handling after the structured_df creation
            if "structured_df = labeled_df[structured_features].copy()" in line:
                new_source.extend([
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
                    "# Check for any remaining NaN values\n",
                    "print(f\"Missing values after imputation:\")\n",
                    "for col in structured_df.columns:\n",
                    "    nan_count = structured_df[col].isna().sum()\n",
                    "    if nan_count > 0:\n",
                    "        print(f\"  {col}: {nan_count} NaN values\")\n",
                    "\n"
                ])
        
        cell["source"] = new_source
        break

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Fixed missing values handling in feature construction")
