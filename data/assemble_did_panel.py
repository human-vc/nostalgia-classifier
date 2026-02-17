"""
Assemble the final difference-in-differences panel dataset.

Merges:
1. Ad-level nostalgia scores aggregated to DMA level
2. DMA-county crosswalk
3. County-level turnout (2020 vs 2024)
4. County-level demographics (controls)

Output: processed/did_panel.csv

This is the dataset consumed by data_preparation_did.py for the
actual regression analysis.
"""

import os
import argparse

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_or_die(path, name):
    if not os.path.exists(path):
        print(f"ERROR: {name} not found at {path}")
        print(f"Run the appropriate data script first.")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {name}: {len(df)} rows")
    return df


def aggregate_nostalgia_by_dma(ads_df):
    """Aggregate ad-level nostalgia scores to DMA level."""
    # Group by DMA and year
    if "dma" not in ads_df.columns:
        print("WARNING: No DMA column in ad data. Cannot aggregate spatially.")
        return None

    agg = (
        ads_df
        .groupby(["dma", "year"])
        .agg(
            nostalgic_count=("Nostalgia_Binary", "sum"),
            total_ads=("Nostalgia_Binary", "count"),
            mean_score=("nostalgia_score", "mean"),
        )
        .reset_index()
    )
    agg["nostalgia_pct"] = (agg["nostalgic_count"] / agg["total_ads"] * 100).round(2)

    print(f"\nDMA-level nostalgia aggregation:")
    for year in sorted(agg["year"].unique()):
        yr = agg[agg["year"] == year]
        print(f"  {year}: {len(yr)} DMAs, mean nostalgia% = {yr['nostalgia_pct'].mean():.1f}%")

    return agg


def build_panel(ads_path, dma_path, turnout_path, demographics_path):
    """Build the full county-level panel dataset."""
    # Load all inputs
    ads = load_or_die(ads_path, "Ad corpus")
    dma_map = load_or_die(dma_path, "DMA-county map")
    turnout = load_or_die(turnout_path, "County turnout")
    demographics = load_or_die(demographics_path, "Demographics")

    if any(x is None for x in [dma_map, turnout, demographics]):
        return None

    # If we have ad data, aggregate to DMA level
    if ads is not None and "dma" in ads.columns:
        dma_nostalgia = aggregate_nostalgia_by_dma(ads)
    else:
        print("\nNo ad data with DMA info. Creating panel without nostalgia treatment.")
        dma_nostalgia = None

    # Start with the DMA-county crosswalk
    panel = dma_map[["dma", "dma_code", "state", "county_fips", "county_name"]].copy()

    # Merge turnout for each year
    # Ensure county_fips is string and zero-padded
    panel["county_fips"] = panel["county_fips"].astype(str).str.zfill(5)
    turnout["county_fips"] = turnout["county_fips"].astype(str).str.zfill(5)
    demographics["county_fips"] = demographics["county_fips"].astype(str).str.zfill(5)

    # Get 2020 turnout
    t2020 = turnout[turnout["year"] == 2020][["county_fips", "turnout_pct", "total_votes"]].copy()
    t2020 = t2020.rename(columns={"turnout_pct": "turnout_2020", "total_votes": "votes_2020"})
    panel = panel.merge(t2020, on="county_fips", how="left")

    # Get 2016 turnout (as pre-period for 2020 analysis) or 2024
    for year in [2016, 2024]:
        t_yr = turnout[turnout["year"] == year][["county_fips", "turnout_pct", "total_votes"]].copy()
        t_yr = t_yr.rename(columns={
            "turnout_pct": f"turnout_{year}",
            "total_votes": f"votes_{year}"
        })
        panel = panel.merge(t_yr, on="county_fips", how="left")

    # Merge demographics
    demo_cols = ["county_fips", "pct_white", "pct_black", "pct_hispanic",
                 "pct_college", "median_income", "median_age", "total_population"]
    available_cols = [c for c in demo_cols if c in demographics.columns]
    panel = panel.merge(demographics[available_cols], on="county_fips", how="left")

    # Merge nostalgia treatment (if available)
    if dma_nostalgia is not None:
        for year in sorted(dma_nostalgia["year"].unique()):
            nost_yr = dma_nostalgia[dma_nostalgia["year"] == year][
                ["dma", "nostalgia_pct", "mean_score", "total_ads"]
            ].copy()
            nost_yr = nost_yr.rename(columns={
                "nostalgia_pct": f"nostalgia_{year}",
                "mean_score": f"nostalgia_score_{year}",
                "total_ads": f"n_ads_{year}",
            })
            panel = panel.merge(nost_yr, on="dma", how="left")

    # Calculate differences
    if "turnout_2020" in panel.columns:
        if "turnout_2016" in panel.columns:
            panel["delta_turnout_2016_2020"] = panel["turnout_2020"] - panel["turnout_2016"]
        if "turnout_2024" in panel.columns:
            panel["delta_turnout_2020_2024"] = panel["turnout_2024"] - panel["turnout_2020"]

    # Drop rows with no turnout data
    turnout_cols = [c for c in panel.columns if c.startswith("turnout_")]
    panel = panel.dropna(subset=turnout_cols, how="all")
    panel = panel.drop_duplicates(subset="county_fips")

    print(f"\nFinal panel: {len(panel)} counties")
    print(f"Columns: {list(panel.columns)}")

    return panel


def main():
    parser = argparse.ArgumentParser(description="Assemble DiD panel dataset")
    parser.add_argument("--ads", default=os.path.join(PROCESSED_DIR, "ad_corpus_labeled.csv"))
    parser.add_argument("--dma-map", default=os.path.join(PROCESSED_DIR, "dma_county_map.csv"))
    parser.add_argument("--turnout", default=os.path.join(PROCESSED_DIR, "county_turnout.csv"))
    parser.add_argument("--demographics", default=os.path.join(PROCESSED_DIR, "county_demographics.csv"))
    parser.add_argument("--output", default=os.path.join(PROCESSED_DIR, "did_panel.csv"))
    args = parser.parse_args()

    print("=" * 60)
    print("DiD PANEL DATASET ASSEMBLY")
    print("=" * 60)

    panel = build_panel(args.ads, args.dma_map, args.turnout, args.demographics)

    if panel is not None:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        panel.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")

        # Summary statistics
        print(f"\n{'='*60}")
        print("PANEL SUMMARY")
        print(f"{'='*60}")
        print(panel.describe().round(2).to_string())


if __name__ == "__main__":
    main()
