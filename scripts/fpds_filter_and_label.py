"""
FPDS Filtering & Label Construction Pipeline
=============================================
Reads Parquet shards from the Omari et al. Federal Procurement Dataset,
filters to physical deliverables (construction, defense, manufacturing),
constructs cost-overrun and schedule-delay labels, and produces a final
labeled dataset for downstream EDA and modeling.

Outputs:
    data/interim/filtered_physical_deliverables.parquet
    data/processed/labeled_contracts.parquet
    data/processed/labeled_contracts.csv
    data/processed/dataset_summary.txt
    figures/*.png
    column_names.txt
"""

import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARD_FOLDER = PROJECT_ROOT / "exploring_data"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
COLUMN_NAMES_FILE = PROJECT_ROOT / "column_names.txt"

for d in [DATA_INTERIM, DATA_PROCESSED, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Exact column name mapping (discovered via schema inspection of actual shards)
COLUMN_MAP = {
    "piid": "content.ID.ContractID.PIID",
    "mod_number": "content.ID.ContractID.modNumber",
    "description": "content.contractData.descriptionOfContractRequirement",
    "psc": "content.productOrServiceInformation.productOrServiceCode.#text",
    "naics": "content.productOrServiceInformation.principalNAICSCode.#text",
    "base_all_options": "content.dollarValues.baseAndAllOptionsValue",
    "base_exercised_options": "content.dollarValues.baseAndExercisedOptionsValue",
    "obligated_amount": "content.dollarValues.obligatedAmount",
    "current_completion_date": "content.relevantContractDates.currentCompletionDate",
    "ultimate_completion_date": "content.relevantContractDates.ultimateCompletionDate",
    "effective_date": "content.relevantContractDates.effectiveDate",
    "signed_date": "content.relevantContractDates.signedDate",
    "reason_for_mod": "content.contractData.reasonForModification.#text",
    "contract_type": "content.contractData.typeOfContractPricing.#text",
    "extent_competed": "content.competition.extentCompeted.#text",
    "num_offers": "content.competition.numberOfOffersReceived",
    "agency_id": "content.purchaserInformation.contractingOfficeAgencyID.#text",
    "vendor_name": "content.vendor.vendorHeader.vendorName",
    "state_code": "content.placeOfPerformance.principalPlaceOfPerformance.stateCode.#text",
}

# Columns we want to read from parquet (values of COLUMN_MAP)
COLUMNS_TO_READ = list(COLUMN_MAP.values())

# Cost overrun threshold (%)
COST_OVERRUN_THRESHOLD = 10.0

# Pipeline metadata collector
META = {
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "shards_processed": 0,
    "shards_failed": [],
    "total_rows_read": 0,
    "total_rows_after_psc_filter": 0,
    "total_rows_after_labels": 0,
    "total_rows_final": 0,
    "warnings": [],
}


# ===================================================================
# Phase 2: Schema Discovery
# ===================================================================

def phase2_schema_discovery():
    """Read schema from first shard, save column names, print mapping."""
    print("\n" + "=" * 70)
    print("PHASE 2: SCHEMA DISCOVERY")
    print("=" * 70)

    shards = sorted(SHARD_FOLDER.glob("*.parquet"))
    if not shards:
        print("ERROR: No .parquet files found in", SHARD_FOLDER)
        sys.exit(1)

    schema = pq.read_schema(str(shards[0]))
    all_names = schema.names
    print(f"\nTotal columns in schema: {len(all_names)}")

    # Save full column list
    with open(COLUMN_NAMES_FILE, "w", encoding="utf-8") as f:
        for i, name in enumerate(all_names):
            f.write(f"{i+1}. {name} ({schema.field(name).type})\n")
    print(f"Saved full column list to {COLUMN_NAMES_FILE}")

    # Check which columns we need are present
    print("\nColumn mapping (concept -> actual column):")
    missing_critical = []
    for concept, col_name in COLUMN_MAP.items():
        found = col_name in all_names
        status = "OK" if found else "MISSING"
        print(f"  {concept:30s} -> {col_name:70s} [{status}]")
        if not found and concept in ("piid", "description", "base_all_options",
                                      "current_completion_date"):
            missing_critical.append(concept)

    if missing_critical:
        print(f"\n*** CRITICAL: Missing columns: {missing_critical}")
        print("    Check whether you downloaded the full Parquet version")
        print("    (not the reduced CSV version which omits most fields).")
        META["warnings"].append(f"Missing critical columns: {missing_critical}")

    return all_names


# ===================================================================
# Phase 2.3: Sample Inspection
# ===================================================================

def phase2_sample_inspection():
    """Load a 500-row sample and inspect data quality."""
    print("\n" + "-" * 70)
    print("PHASE 2.3: 500-ROW SAMPLE INSPECTION")
    print("-" * 70)

    shards = sorted(SHARD_FOLDER.glob("*.parquet"))
    first_shard = str(shards[0])

    # Determine which columns are actually available
    schema = pq.read_schema(first_shard)
    available = [c for c in COLUMNS_TO_READ if c in schema.names]
    missing = [c for c in COLUMNS_TO_READ if c not in schema.names]
    if missing:
        print(f"  Columns missing from shard (will be null): {missing}")

    table = pq.read_table(first_shard, columns=available)
    df = table.to_pandas().head(500)

    # Rename for convenience
    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns={c: reverse_map[c] for c in df.columns if c in reverse_map})

    print(f"\nSample shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")

    # 5 sample values per column
    print("\n--- Sample values (5 rows) ---")
    for col in df.columns:
        vals = df[col].dropna().head(5).tolist()
        print(f"  {col}: {vals}")

    # Description field check
    desc_col = "description"
    if desc_col in df.columns:
        non_null = df[desc_col].notna().sum()
        total = len(df)
        print(f"\n  Descriptions: {non_null}/{total} non-null ({non_null/total*100:.1f}%)")
    else:
        print("\n  WARNING: description column not in sample")

    # Dollar fields numeric check
    for dc in ["base_all_options", "base_exercised_options", "obligated_amount"]:
        if dc in df.columns:
            print(f"  {dc} dtype: {df[dc].dtype}")

    # Date fields check
    for dc in ["current_completion_date", "effective_date", "signed_date"]:
        if dc in df.columns:
            print(f"  {dc} dtype: {df[dc].dtype}, sample: {df[dc].dropna().head(2).tolist()}")

    return df


# ===================================================================
# Phase 3: Filtering to Physical Deliverables
# ===================================================================

def is_physical_deliverable(psc_code):
    """Return True if PSC code indicates a physical deliverable.

    Physical deliverable codes:
        Y-series: Construction of structures and facilities
        Z-series: Maintenance, repair, alteration of real property
        Two-digit numeric 10-99: Supplies and equipment
    """
    if pd.isna(psc_code) or not isinstance(psc_code, str) or len(psc_code) == 0:
        return False
    first_char = psc_code[0].upper()
    if first_char in ("Y", "Z"):
        return True
    # Check for two-digit numeric codes 10-99
    try:
        numeric_prefix = int(psc_code[:2])
        if 10 <= numeric_prefix <= 99:
            return True
    except (ValueError, IndexError):
        pass
    return False


def phase3_filter_shards():
    """Read all shards, filter to physical deliverables, save checkpoint."""
    print("\n" + "=" * 70)
    print("PHASE 3: FILTERING TO PHYSICAL DELIVERABLES")
    print("=" * 70)

    shards = sorted(SHARD_FOLDER.glob("*.parquet"))
    filtered_chunks = []
    total_read = 0
    total_kept = 0

    for i, shard_path in enumerate(shards):
        shard_name = shard_path.name
        try:
            schema = pq.read_schema(str(shard_path))
            available = [c for c in COLUMNS_TO_READ if c in schema.names]

            table = pq.read_table(str(shard_path), columns=available)
            df = table.to_pandas()
            rows_in = len(df)
            total_read += rows_in

            # Add missing columns as NaN
            for c in COLUMNS_TO_READ:
                if c not in df.columns:
                    df[c] = np.nan

            # Apply PSC filter
            psc_col = COLUMN_MAP["psc"]
            mask = df[psc_col].apply(is_physical_deliverable)
            df_filtered = df[mask].copy()
            rows_kept = len(df_filtered)
            total_kept += rows_kept

            filtered_chunks.append(df_filtered)

            if (i + 1) % 5 == 0 or i == 0 or i == len(shards) - 1:
                pct = (total_kept / total_read * 100) if total_read > 0 else 0
                print(f"  Shard {i+1}/{len(shards)} ({shard_name}): "
                      f"read={rows_in:,}, kept={rows_kept:,} | "
                      f"Running total: {total_read:,} read, {total_kept:,} kept ({pct:.1f}%)")

        except Exception as e:
            print(f"  ERROR reading {shard_name}: {e}")
            META["shards_failed"].append(shard_name)
            continue

    META["shards_processed"] = len(shards) - len(META["shards_failed"])
    META["total_rows_read"] = total_read
    META["total_rows_after_psc_filter"] = total_kept

    if not filtered_chunks:
        print("ERROR: No data survived filtering. Check PSC column mapping.")
        sys.exit(1)

    df_all = pd.concat(filtered_chunks, ignore_index=True)

    # Rename columns to short names
    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df_all = df_all.rename(columns={c: reverse_map[c] for c in df_all.columns if c in reverse_map})

    print(f"\n  TOTAL: {total_read:,} rows read, {total_kept:,} kept "
          f"({total_kept/total_read*100:.1f}% retention)")

    # Save checkpoint
    checkpoint_path = DATA_INTERIM / "filtered_physical_deliverables.parquet"
    df_all.to_parquet(str(checkpoint_path), index=False)
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  Saved checkpoint: {checkpoint_path} ({size_mb:.1f} MB)")

    return df_all


# ===================================================================
# Phase 4: Outcome Label Construction
# ===================================================================

def _to_float(series):
    """Convert a column to float64, handling strings with $, commas."""
    if series.dtype == "object":
        series = series.str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _to_datetime(series):
    """Convert a column to datetime, handling mixed formats."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def phase4_construct_labels(df):
    """Clean, group by PIID, extract initial/final values, compute labels."""
    print("\n" + "=" * 70)
    print("PHASE 4: LABEL CONSTRUCTION")
    print("=" * 70)

    # --- 4.1: Clean and type-cast ---
    print("\n--- 4.1: Cleaning and type-casting ---")

    for col in ["base_all_options", "base_exercised_options", "obligated_amount"]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    for col in ["current_completion_date", "ultimate_completion_date",
                 "effective_date", "signed_date"]:
        if col in df.columns:
            df[col] = _to_datetime(df[col])

    if "num_offers" in df.columns:
        df["num_offers"] = pd.to_numeric(df["num_offers"], errors="coerce")

    # Mod number: try numeric sort key
    if "mod_number" in df.columns:
        df["mod_number_sort"] = pd.to_numeric(
            df["mod_number"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        ).fillna(0).astype(int)

    # Null report
    critical_cols = ["piid", "base_all_options", "base_exercised_options",
                     "current_completion_date", "description"]
    print("\n  Null counts in critical columns:")
    for c in critical_cols:
        if c in df.columns:
            n = df[c].isna().sum()
            print(f"    {c}: {n:,} nulls ({n/len(df)*100:.1f}%)")

    dollar_date_nulls = (
        df["base_all_options"].isna() & df["base_exercised_options"].isna()
        & df["current_completion_date"].isna()
    )
    print(f"\n  Rows with ALL dollar AND date columns null: {dollar_date_nulls.sum():,}")

    # --- 4.2: Sort and group by PIID ---
    print("\n--- 4.2: Grouping by contract PIID ---")
    sort_col = "mod_number_sort" if "mod_number_sort" in df.columns else "mod_number"
    df = df.sort_values(["piid", sort_col]).reset_index(drop=True)

    grouped = df.groupby("piid", sort=False)
    n_contracts = grouped.ngroups
    mod_counts = grouped.size()
    print(f"  Unique contracts (PIIDs): {n_contracts:,}")
    print(f"  Median modifications per contract: {mod_counts.median():.0f}")
    print(f"  Max modifications: {mod_counts.max()}")

    # --- 4.3: Extract initial vs final per contract (vectorized) ---
    print("\n--- 4.3: Extracting initial vs final values (vectorized) ---")

    # Drop rows with null PIID before grouping
    df = df.dropna(subset=["piid"]).copy()
    print(f"  Rows after dropping null PIIDs: {len(df):,}")

    g = df.groupby("piid", sort=False)

    # Initial row values (first row per group = lowest mod number)
    initial_cols = {
        "description": "description",
        "base_all_options": "initial_cost",
        "current_completion_date": "initial_completion_date",
        "effective_date": "effective_date",
        "contract_type": "contract_type",
        "extent_competed": "competition",
        "num_offers": "num_offers",
        "agency_id": "agency",
        "vendor_name": "vendor",
        "state_code": "state",
        "psc": "psc",
        "naics": "naics",
    }
    initial = g.first()[[c for c in initial_cols.keys() if c in df.columns]]
    initial = initial.rename(columns=initial_cols)

    # Final row values (last row per group = highest mod number)
    final = g.last()[["base_exercised_options", "base_all_options",
                       "current_completion_date"]].rename(columns={
        "base_exercised_options": "final_cost_exercised",
        "base_all_options": "final_cost_fallback",
        "current_completion_date": "final_completion_date",
    })

    # Modification counts
    mod_count = g.size().rename("num_modifications")

    # All modification reasons (optimized: deduplicate first, then aggregate)
    if "reason_for_mod" in df.columns:
        print("  Aggregating modification reasons...")
        reason_df = df[["piid", "reason_for_mod"]].dropna(subset=["reason_for_mod"])
        reason_df = reason_df.drop_duplicates()
        mod_reasons = reason_df.groupby("piid")["reason_for_mod"].agg(
            lambda x: ", ".join(x.astype(str))
        ).rename("all_mod_reasons")
    else:
        mod_reasons = pd.Series(dtype=str, name="all_mod_reasons")

    # Combine into single DataFrame
    labeled = initial.join(final).join(mod_count).join(mod_reasons)
    labeled.index.name = "piid"
    labeled = labeled.reset_index()

    # Final cost: prefer exercised, fallback to all-options
    labeled["final_cost"] = labeled["final_cost_exercised"].fillna(labeled["final_cost_fallback"])
    labeled = labeled.drop(columns=["final_cost_exercised", "final_cost_fallback"])

    print(f"  Constructed {len(labeled):,} contract-level rows")

    # --- 4.4: Compute outcome labels ---
    print("\n--- 4.4: Computing outcome labels ---")

    # cost_growth_pct
    labeled["cost_growth_pct"] = np.where(
        (labeled["initial_cost"] > 0) & labeled["initial_cost"].notna() & labeled["final_cost"].notna(),
        ((labeled["final_cost"] - labeled["initial_cost"]) / labeled["initial_cost"]) * 100,
        np.nan
    )

    # over_budget
    labeled["over_budget"] = np.where(
        labeled["cost_growth_pct"].notna(),
        (labeled["cost_growth_pct"] > COST_OVERRUN_THRESHOLD).astype(int),
        np.nan
    )

    # delay_days
    labeled["delay_days"] = np.where(
        labeled["initial_completion_date"].notna() & labeled["final_completion_date"].notna(),
        (labeled["final_completion_date"] - labeled["initial_completion_date"]).dt.days,
        np.nan
    )

    # late
    labeled["late"] = np.where(
        labeled["delay_days"].notna(),
        (labeled["delay_days"].astype(float) > 0).astype(int),
        np.nan
    )

    # terminated_for_default
    labeled["terminated_for_default"] = labeled["all_mod_reasons"].str.upper().str.contains(
        r"DEFAULT|TERMINAT", na=False
    ).astype(int)

    print(f"  cost_growth_pct non-null: {labeled['cost_growth_pct'].notna().sum():,}")
    print(f"  delay_days non-null: {labeled['delay_days'].notna().sum():,}")

    # --- 4.5: Filter unusable rows ---
    print("\n--- 4.5: Filtering unusable rows ---")
    n_before = len(labeled)

    # Drop where BOTH over_budget AND late are null
    mask1 = labeled["over_budget"].isna() & labeled["late"].isna()
    n_drop1 = mask1.sum()
    labeled = labeled[~mask1].copy()
    print(f"  Dropped {n_drop1:,} rows: both over_budget and late are null")

    # Drop where initial_cost <= 0
    mask2 = labeled["initial_cost"].fillna(0) <= 0
    n_drop2 = mask2.sum()
    labeled = labeled[~mask2].copy()
    print(f"  Dropped {n_drop2:,} rows: initial_cost <= 0 or null")

    # Drop where description is null/empty
    mask3 = labeled["description"].isna() | (labeled["description"].astype(str).str.strip() == "")
    n_drop3 = mask3.sum()
    labeled = labeled[~mask3].copy()
    print(f"  Dropped {n_drop3:,} rows: description null or empty")

    print(f"\n  Final dataset: {len(labeled):,} contracts (dropped {n_before - len(labeled):,} total)")

    META["total_rows_after_labels"] = n_before
    META["total_rows_final"] = len(labeled)

    if len(labeled) < 1000:
        print("\n  *** WARNING: Fewer than 1,000 contracts. Check column mappings. ***")
        META["warnings"].append(f"Only {len(labeled)} contracts in final dataset (< 1000)")

    return labeled


# ===================================================================
# Phase 5: Quality Checks
# ===================================================================

def phase5_quality_checks(df):
    """Run all quality checks and decision point analysis."""
    print("\n" + "=" * 70)
    print("PHASE 5: QUALITY CHECKS AND DECISION POINTS")
    print("=" * 70)

    # --- 5.1: Class balance ---
    print("\n--- 5.1: Class Balance Report ---")
    for label_col in ["over_budget", "late", "terminated_for_default"]:
        if label_col in df.columns:
            valid = df[label_col].dropna()
            if len(valid) > 0:
                counts = valid.value_counts().sort_index()
                total_v = len(valid)
                print(f"\n  {label_col}:")
                for val, cnt in counts.items():
                    print(f"    {int(val)}: {cnt:,} ({cnt/total_v*100:.1f}%)")

                # Check minority class
                minority_pct = counts.min() / total_v * 100
                if minority_pct < 10:
                    warn = (f"DECISION POINT: {label_col} minority class is only "
                            f"{minority_pct:.1f}%. Consider lowering the threshold "
                            f"or applying SMOTE during modeling.")
                    print(f"    *** WARNING: {warn}")
                    META["warnings"].append(warn)
            else:
                print(f"\n  {label_col}: no valid values")

    # --- 5.2: Text quality ---
    print("\n--- 5.2: Text Quality Report ---")
    desc = df["description"].astype(str)
    desc_lens = desc.str.len()
    non_null = df["description"].notna().sum()
    print(f"  Non-null descriptions: {non_null:,}")
    print(f"  Median length: {desc_lens.median():.0f} chars")
    print(f"  Mean length: {desc_lens.mean():.0f} chars")
    print(f"  Under 50 chars: {(desc_lens < 50).sum():,}")
    print(f"  Under 100 chars: {(desc_lens < 100).sum():,}")
    print(f"  Over 500 chars: {(desc_lens > 500).sum():,}")

    short = df[desc_lens < 50]["description"].head(5).tolist()
    print(f"\n  Short descriptions (<50 chars):")
    for s in short:
        print(f"    '{s}'")

    long_descs = df[desc_lens > 500]["description"].head(5).tolist()
    print(f"\n  Long descriptions (>500 chars):")
    for s in long_descs:
        print(f"    '{s[:120]}...'")

    if (desc_lens < 50).sum() > len(df) * 0.5:
        warn = ("DECISION POINT: Most descriptions are very short. "
                "LDA may not work well. Consider falling back to TF-IDF.")
        print(f"\n  *** WARNING: {warn}")
        META["warnings"].append(warn)

    # --- 5.3: Cost and schedule distributions ---
    print("\n--- 5.3: Cost and Schedule Distributions ---")
    for metric, col in [("cost_growth_pct", "cost_growth_pct"), ("delay_days", "delay_days")]:
        valid = df[col].dropna()
        if len(valid) > 0:
            print(f"\n  {metric}:")
            print(f"    Count: {len(valid):,}")
            print(f"    Min: {valid.min():.1f}")
            print(f"    25th pctl: {valid.quantile(0.25):.1f}")
            print(f"    Median: {valid.median():.1f}")
            print(f"    75th pctl: {valid.quantile(0.75):.1f}")
            print(f"    Max: {valid.max():.1f}")
            print(f"    Mean: {valid.mean():.1f}")
            print(f"    Std: {valid.std():.1f}")

    # Top 5 extreme overruns
    top_cost = df.nlargest(5, "cost_growth_pct")[["piid", "agency", "psc", "cost_growth_pct"]]
    print(f"\n  Top 5 cost overruns:")
    print(top_cost.to_string(index=False))

    top_delay = df.nlargest(5, "delay_days")[["piid", "agency", "psc", "delay_days"]]
    print(f"\n  Top 5 schedule delays:")
    print(top_delay.to_string(index=False))

    # --- 5.4: Breakdown by category ---
    print("\n--- 5.4: Breakdown by Category ---")

    # PSC category
    df["psc_category"] = df["psc"].astype(str).str[0].str.upper()
    for label_col in ["over_budget", "late"]:
        valid = df[df[label_col].notna()].copy()
        if len(valid) > 0:
            rates = valid.groupby("psc_category")[label_col].mean() * 100
            print(f"\n  {label_col} rate by PSC category:")
            for cat, rate in rates.sort_values(ascending=False).items():
                cnt = (valid["psc_category"] == cat).sum()
                print(f"    {cat}: {rate:.1f}% (n={cnt:,})")

    # Contract type (top 5)
    for label_col in ["over_budget", "late"]:
        valid = df[df[label_col].notna() & df["contract_type"].notna()].copy()
        if len(valid) > 0:
            top_types = valid["contract_type"].value_counts().head(5).index
            rates = valid[valid["contract_type"].isin(top_types)].groupby("contract_type")[label_col].mean() * 100
            print(f"\n  {label_col} rate by contract type (top 5):")
            for ct, rate in rates.sort_values(ascending=False).items():
                print(f"    {ct}: {rate:.1f}%")

    # Agency (top 10)
    for label_col in ["over_budget", "late"]:
        valid = df[df[label_col].notna() & df["agency"].notna()].copy()
        if len(valid) > 0:
            top_agencies = valid["agency"].value_counts().head(10).index
            rates = valid[valid["agency"].isin(top_agencies)].groupby("agency")[label_col].mean() * 100
            print(f"\n  {label_col} rate by agency (top 10):")
            for ag, rate in rates.sort_values(ascending=False).items():
                print(f"    {ag}: {rate:.1f}%")

    return df


# ===================================================================
# Phase 6: Outputs
# ===================================================================

def phase6_save_outputs(df):
    """Save final dataset, metadata, and visualizations."""
    print("\n" + "=" * 70)
    print("PHASE 6: SAVING OUTPUTS")
    print("=" * 70)

    # --- 6.1: Save datasets ---
    pq_path = DATA_PROCESSED / "labeled_contracts.parquet"
    csv_path = DATA_PROCESSED / "labeled_contracts.csv"

    df.to_parquet(str(pq_path), index=False)
    df.to_csv(str(csv_path), index=False)

    pq_size = os.path.getsize(pq_path) / (1024 * 1024)
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  Saved: {pq_path} ({pq_size:.1f} MB)")
    print(f"  Saved: {csv_path} ({csv_size:.1f} MB)")

    # --- 6.2: Metadata summary ---
    summary_path = DATA_PROCESSED / "dataset_summary.txt"
    _write_summary(df, summary_path)
    print(f"  Saved: {summary_path}")

    # --- 6.3: Visualizations ---
    print("\n  Generating visualizations...")
    sns.set_theme(style="whitegrid")

    _plot_class_balance(df)
    _plot_cost_growth(df)
    _plot_delay_distribution(df)
    _plot_description_lengths(df)
    _plot_overrun_by_psc(df)
    _plot_overrun_by_agency(df)

    print("  All figures saved to", FIGURES_DIR)

    # --- 6.4: Final console summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    valid_ob = df["over_budget"].dropna()
    valid_late = df["late"].dropna()
    desc_coverage = df["description"].notna().sum() / len(df) * 100

    print(f"  Total contracts: {len(df):,}")
    if len(valid_ob) > 0:
        print(f"  Over-budget rate: {valid_ob.mean()*100:.1f}%")
    if len(valid_late) > 0:
        print(f"  Late rate: {valid_late.mean()*100:.1f}%")
    cg = df["cost_growth_pct"].dropna()
    if len(cg) > 0:
        print(f"  Median cost growth: {cg.median():.1f}%")
    dd = df["delay_days"].dropna()
    if len(dd) > 0:
        print(f"  Median delay: {dd.median():.0f} days")
    print(f"  Description coverage: {desc_coverage:.1f}%")

    if META["warnings"]:
        print("\n  Decision points triggered:")
        for w in META["warnings"]:
            print(f"    - {w}")
    else:
        print("\n  No warnings triggered.")

    if len(df) >= 1000:
        print("\n  >>> Dataset is ready for EDA and modeling. <<<")
    else:
        print("\n  >>> WARNING: Dataset is small. Review filtering before proceeding. <<<")


def _write_summary(df, path):
    """Write dataset_summary.txt."""
    valid_ob = df["over_budget"].dropna()
    valid_late = df["late"].dropna()
    desc = df["description"].astype(str)

    with open(path, "w", encoding="utf-8") as f:
        f.write("FPDS Filter & Label Pipeline Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run date: {META['run_date']}\n")
        f.write(f"Shards processed: {META['shards_processed']}\n")
        if META["shards_failed"]:
            f.write(f"Shards failed: {META['shards_failed']}\n")
        _fmt = lambda v: f"{v:,}" if isinstance(v, (int, float)) else str(v)
        f.write(f"Total rows read from shards: {_fmt(META['total_rows_read'])}\n")
        f.write(f"Total rows after PSC filtering: {_fmt(META['total_rows_after_psc_filter'])}\n")
        f.write(f"Total rows after label construction: {_fmt(META['total_rows_after_labels'])}\n")
        f.write(f"Total rows in final dataset: {_fmt(META['total_rows_final'])}\n\n")

        f.write("Class Balance\n")
        f.write("-" * 30 + "\n")
        if len(valid_ob) > 0:
            f.write(f"over_budget=1: {(valid_ob==1).sum():,} ({(valid_ob==1).mean()*100:.1f}%)\n")
            f.write(f"over_budget=0: {(valid_ob==0).sum():,} ({(valid_ob==0).mean()*100:.1f}%)\n")
        if len(valid_late) > 0:
            f.write(f"late=1: {(valid_late==1).sum():,} ({(valid_late==1).mean()*100:.1f}%)\n")
            f.write(f"late=0: {(valid_late==0).sum():,} ({(valid_late==0).mean()*100:.1f}%)\n")
        td = df["terminated_for_default"]
        f.write(f"terminated_for_default=1: {(td==1).sum():,} ({(td==1).mean()*100:.1f}%)\n\n")

        f.write("Text Quality\n")
        f.write("-" * 30 + "\n")
        f.write(f"Non-null descriptions: {df['description'].notna().sum():,}\n")
        f.write(f"Median length: {desc.str.len().median():.0f} chars\n")
        f.write(f"Mean length: {desc.str.len().mean():.0f} chars\n\n")

        f.write("Column List\n")
        f.write("-" * 30 + "\n")
        for col in df.columns:
            f.write(f"  {col} ({df[col].dtype})\n")

        if META["warnings"]:
            f.write("\nWarnings / Decision Points\n")
            f.write("-" * 30 + "\n")
            for w in META["warnings"]:
                f.write(f"  - {w}\n")


def _plot_class_balance(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, col, title in zip(axes,
                               ["over_budget", "late"],
                               ["Over-Budget (>10% cost growth)", "Late (any delay)"]):
        valid = df[col].dropna()
        if len(valid) > 0:
            counts = valid.value_counts().sort_index()
            counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
            ax.set_title(title)
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
            ax.set_xticklabels(["0 (No)", "1 (Yes)"], rotation=0)
            for p in ax.patches:
                ax.annotate(f"{int(p.get_height()):,}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "class_balance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cost_growth(df):
    cg = df["cost_growth_pct"].dropna()
    if len(cg) == 0:
        return
    cg_clipped = cg.clip(-100, 200)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cg_clipped, bins=80, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(COST_OVERRUN_THRESHOLD, color="red", linestyle="--",
               label=f"Threshold ({COST_OVERRUN_THRESHOLD}%)")
    ax.set_title("Cost Growth Distribution (clipped to [-100%, 200%])")
    ax.set_xlabel("Cost Growth (%)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "cost_growth_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_delay_distribution(df):
    dd = df["delay_days"].dropna()
    if len(dd) == 0:
        return
    dd_clipped = dd.clip(-365, 1000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dd_clipped, bins=80, color="#9b59b6", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", label="On-time threshold")
    ax.set_title("Schedule Delay Distribution (clipped to [-365, 1000] days)")
    ax.set_xlabel("Delay (days)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "delay_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_description_lengths(df):
    desc_lens = df["description"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(desc_lens.clip(0, 2000), bins=80, color="#e67e22", edgecolor="white", alpha=0.8)
    ax.set_title("Contract Description Length Distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "description_length_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_overrun_by_psc(df):
    df_temp = df.copy()
    df_temp["psc_category"] = df_temp["psc"].astype(str).str[0].str.upper()
    valid = df_temp[df_temp["over_budget"].notna()]
    if len(valid) == 0:
        return
    rates = valid.groupby("psc_category")["over_budget"].agg(["mean", "count"])
    rates = rates[rates["count"] >= 10].sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(rates.index, rates["mean"] * 100, color="#2ecc71", edgecolor="white")
    ax.set_title("Over-Budget Rate by PSC Category")
    ax.set_xlabel("PSC First Character")
    ax.set_ylabel("Over-Budget Rate (%)")
    for bar, (_, row) in zip(bars, rates.iterrows()):
        ax.annotate(f"n={int(row['count']):,}",
                    (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "overrun_rate_by_psc.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_overrun_by_agency(df):
    valid = df[df["over_budget"].notna() & df["agency"].notna()]
    if len(valid) == 0:
        return
    top10 = valid["agency"].value_counts().head(10).index
    subset = valid[valid["agency"].isin(top10)]
    rates = subset.groupby("agency")["over_budget"].agg(["mean", "count"])
    rates = rates.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(rates.index.astype(str), rates["mean"] * 100, color="#3498db", edgecolor="white")
    ax.set_title("Over-Budget Rate by Top 10 Agencies")
    ax.set_xlabel("Over-Budget Rate (%)")
    ax.set_ylabel("Agency ID")
    for bar, (_, row) in zip(bars, rates.iterrows()):
        ax.annotate(f"n={int(row['count']):,}",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2.),
                    ha="left", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "overrun_rate_by_agency.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ===================================================================
# Main
# ===================================================================

def main():
    resume = "--resume" in sys.argv
    checkpoint = DATA_INTERIM / "filtered_physical_deliverables.parquet"

    print("FPDS Filter & Label Pipeline")
    print(f"Started: {META['run_date']}")
    print(f"Shard folder: {SHARD_FOLDER}")
    print(f"Shards found: {len(list(SHARD_FOLDER.glob('*.parquet')))}")

    if resume and checkpoint.exists():
        print(f"\n  --resume: Loading checkpoint from {checkpoint}")
        df_filtered = pd.read_parquet(str(checkpoint))
        META["shards_processed"] = "loaded from checkpoint"
        META["total_rows_read"] = "loaded from checkpoint"
        META["total_rows_after_psc_filter"] = len(df_filtered)
        print(f"  Loaded {len(df_filtered):,} rows")
    else:
        # Phase 2
        phase2_schema_discovery()
        phase2_sample_inspection()

        # Phase 3
        df_filtered = phase3_filter_shards()

    # Phase 4
    df_labeled = phase4_construct_labels(df_filtered)

    # Phase 5
    df_labeled = phase5_quality_checks(df_labeled)

    # Phase 6
    phase6_save_outputs(df_labeled)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
