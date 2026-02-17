"""
Fetch county-level presidential election returns from MIT Election Data Lab.

Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ
Direct CSV: https://dataverse.harvard.edu/api/access/datafile/8082609

This provides county-level vote totals for presidential elections 2000-2020.
For 2024 data, we fall back to the NYT/AP results or Dave Leip's atlas.

Output: processed/county_turnout.csv
"""

import os
import argparse

import pandas as pd
import requests

# MIT Election Lab county presidential returns
MIT_COUNTY_PRES_URL = "https://dataverse.harvard.edu/api/access/datafile/8082609"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

# Census county population estimates (for turnout denominator)
# Using 2020 Decennial Census P1 table (total population by county)
CENSUS_POP_URL = "https://api.census.gov/data/2020/dec/pl?get=P1_001N,NAME&for=county:*&in=state:*"

BATTLEGROUND_STATES = {
    "ARIZONA", "GEORGIA", "MICHIGAN", "NEVADA",
    "NORTH CAROLINA", "PENNSYLVANIA", "WISCONSIN"
}

BATTLEGROUND_FIPS = {
    "04", "13", "26", "32", "37", "42", "55"  # State FIPS codes
}


def download_mit_data(force=False):
    """Download the MIT county presidential dataset."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, "countypres_2000-2020.csv")

    if os.path.exists(path) and not force:
        print(f"Using cached: {path}")
        return path

    print("Downloading MIT Election Lab county presidential returns...")
    resp = requests.get(MIT_COUNTY_PRES_URL, timeout=60,
                        headers={"User-Agent": "NostalgiaClassifier/1.0"})
    resp.raise_for_status()

    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"Saved: {path} ({len(resp.content)/1024:.0f} KB)")
    return path


def download_population(api_key=None, force=False):
    """Download county population from Census API."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, "county_population_2020.csv")

    if os.path.exists(path) and not force:
        print(f"Using cached: {path}")
        return path

    print("Downloading Census 2020 county populations...")
    url = CENSUS_POP_URL
    if api_key:
        url += f"&key={api_key}"

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # First row is headers
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df)} counties)")
    return path


def build_turnout(election_path, population_path, years=None):
    """
    Build county-level turnout dataset.

    Turnout = total_votes / voting_age_population
    Since we use total Census population as denominator, this gives a
    lower-bound turnout estimate. For the DiD, what matters is the
    *change* in turnout, not the absolute level.
    """
    # Load election data
    elections = pd.read_csv(election_path)
    print(f"Loaded {len(elections)} rows from MIT dataset")

    if years:
        elections = elections[elections["year"].isin(years)]

    # Aggregate to county-year level (sum across candidates)
    county_votes = (
        elections
        .groupby(["year", "state", "county_fips", "county_name"])
        .agg(total_votes=("candidatevotes", "sum"))
        .reset_index()
    )

    # Remove duplicate counting (totalvotes column is already the sum)
    # Actually MIT data has one row per candidate per county, so we need
    # to take the max of totalvotes (which is the same for all candidates in a county)
    county_totals = (
        elections
        .groupby(["year", "state", "county_fips", "county_name"])
        .agg(total_votes=("totalvotes", "first"))
        .reset_index()
    )

    # Load population
    pop = pd.read_csv(population_path)
    pop["county_fips"] = pop["state"].str.zfill(2) + pop["county"].str.zfill(3)
    pop["population"] = pd.to_numeric(pop["P1_001N"], errors="coerce")
    pop = pop[["county_fips", "population"]].copy()

    # Merge
    merged = county_totals.merge(pop, on="county_fips", how="left")
    merged["total_votes"] = pd.to_numeric(merged["total_votes"], errors="coerce")

    # Calculate turnout
    merged["turnout_pct"] = (merged["total_votes"] / merged["population"]) * 100

    # Clean up
    merged = merged.dropna(subset=["total_votes", "population"])
    merged = merged[merged["population"] > 0]

    # Cap turnout at 100% (some counties have registration > Census pop)
    merged.loc[merged["turnout_pct"] > 100, "turnout_pct"] = 100.0

    print(f"\nTurnout summary by year:")
    for year in sorted(merged["year"].unique()):
        yr = merged[merged["year"] == year]
        print(f"  {year}: {len(yr)} counties, median turnout {yr['turnout_pct'].median():.1f}%")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch county election data")
    parser.add_argument("--census-key", default=None,
                        help="Census API key (optional, higher rate limit)")
    parser.add_argument("--years", nargs="+", type=int, default=[2016, 2020],
                        help="Election years to include")
    parser.add_argument("--battleground-only", action="store_true",
                        help="Filter to battleground states only")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    args = parser.parse_args()

    print("=" * 60)
    print("ELECTION DATA FETCHER")
    print("=" * 60)

    election_path = download_mit_data(force=args.force)
    pop_path = download_population(api_key=args.census_key, force=args.force)
    turnout = build_turnout(election_path, pop_path, years=args.years)

    if args.battleground_only:
        turnout = turnout[turnout["state"].str.upper().isin(BATTLEGROUND_STATES)]
        print(f"\nFiltered to battleground states: {len(turnout)} rows")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "county_turnout.csv")
    turnout.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
