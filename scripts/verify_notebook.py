"""Verify the executed notebook has no errors and all outputs."""
import json

with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")

code_cells = [c for c in cells if c["cell_type"] == "code"]
md_cells = [c for c in cells if c["cell_type"] == "markdown"]
print(f"Code cells: {len(code_cells)}")
print(f"Markdown cells: {len(md_cells)}")

# Check for execution counts
exec_counts = [c.get("execution_count") for c in code_cells]
print(f"Execution counts: {exec_counts}")

# Check for errors
error_cells = []
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                error_cells.append(i)

if error_cells:
    print(f"ERROR: Cells with errors: {error_cells}")
else:
    print("No errors found in any cell")

# Check for output in each code cell
empty_output = []
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        if not c.get("outputs"):
            empty_output.append(i)

if empty_output:
    print(f"Cells with no output: {empty_output}")
else:
    print("All code cells have output")

# Check figures exist
import os
expected_figures = [
    "class_balance.png",
    "cost_growth_distribution.png",
    "delay_distribution.png",
    "description_length_distribution.png",
    "overrun_rates_by_category.png",
    "correlation_heatmap.png",
    "roc_curves_comparison.png",
]
print("\nFigure check:")
for fig in expected_figures:
    path = os.path.join("figures", fig)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = f"OK ({size:,} bytes)" if exists else "MISSING"
    print(f"  {fig}: {status}")
