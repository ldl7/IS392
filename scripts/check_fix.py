"""Check if the fix was applied correctly and identify remaining NaN issues."""

import json

# Read the notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find cell 20 (feature construction)
cells = notebook["cells"]
if len(cells) > 20:
    cell = cells[20]
    if cell.get("cell_type") == "code":
        source = cell["source"]
        # Check if our fix is present
        fix_present = any("structured_df['num_offers'] = structured_df['num_offers'].replace('None', np.nan)" in line for line in source)
        print(f"Fix applied: {fix_present}")
        
        # Show relevant lines around num_offers handling
        print("\nRelevant code section:")
        for i, line in enumerate(source):
            if "num_offers" in line.lower():
                print(f"Line {i}: {line.strip()}")
else:
    print("Notebook doesn't have enough cells")
