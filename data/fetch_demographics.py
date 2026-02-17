"""
Fetch county-level demographics from the US Census ACS 5-Year Estimates.

Variables pulled:
- % White alone (B02001_002E / B02001_001E)
- % Black alone (B02001_003E / B02001_001E)
- % Hispanic (B03003_003E / B03003_001E)
- % Bachelor's degree or higher (B15003_022E+ / B15003_001E)
- Median household income (B19013_001E)
- Total population (B01003_001E)
- Median age (B01002_001E)

Source: https://api.census.gov/data/2022/acs/acs5

Output: processed/county_demographics.csv
"""

import os
import argparse

import pandas as pd
import requests

BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

# ACS variable codes
VARIABLES = {
    "B01003_001E": "total_population",
    "B01002_001E": "median_age",
    "B02001_001E": "race_total",
    "B02001_002E": "white_alone",
    "B02001_003E": "black_alone",
    "B03003_001E": "hispanic_total",
    "B03003_003E": "hispanic_alone",
    "B15003_001E": "edu_total",
    "B15003_022E": "bachelors",
    "B15003_023E": "masters",
    "B15003_024E": "professional",
    "B15003_025E": "doctorate",
    "B19013_001E": "median_income",
}


def fetch_acs(year=2022, api_key=None):
    """Fetch ACS 5-year estimates for all counties."""
    url = BASE_URL.format(year=year)
    var_str = ",".join(VARIABLES.keys())
    params = {
        "get": f"NAME,{var_str}",
        "for": "county:*",
        "in": "state:*",
    }
    if api_key:
        params["key"] = api_key

    print(f"Fetching ACS {year} 5-year estimates...")
    resp = requests.get(url, params=params, timeout=120)

    if resp.status_code != 200:
        print(f"Error: HTTP {resp.status_code}")
        print(resp.text[:500])
        return None

    data = resp.json()
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)
    print(f"Received {len(df)} counties")
    return df


def process_demographics(df):
    """Convert raw ACS data into clean demographic variables."""
    # Build FIPS code
    df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)

    # Convert numeric columns
    for var_code in VARIABLES:
        df[var_code] = pd.to_numeric(df[var_code], errors="coerce")

    # Calculate percentages
    result = pd.DataFrame()
    result["county_fips"] = df["county_fips"]
    result["county_name"] = df["NAME"]
    result["state_fips"] = df["state"]
    result["total_population"] = df["B01003_001E"]
    result["median_age"] = df["B01002_001E"]
    result["median_income"] = df["B19013_001E"]

    # Race percentages
    race_total = df["B02001_001E"].replace(0, float("nan"))
    result["pct_white"] = (df["B02001_002E"] / race_total * 100).round(2)
    result["pct_black"] = (df["B02001_003E"] / race_total * 100).round(2)

    # Hispanic percentage
    hisp_total = df["B03003_001E"].replace(0, float("nan"))
    result["pct_hispanic"] = (df["B03003_003E"] / hisp_total * 100).round(2)

    # Education: % with bachelor's or higher
    edu_total = df["B15003_001E"].replace(0, float("nan"))
    college_plus = df["B15003_022E"] + df["B15003_023E"] + df["B15003_024E"] + df["B15003_025E"]
    result["pct_college"] = (college_plus / edu_total * 100).round(2)

    # Drop rows with no population
    result = result[result["total_population"] > 0].copy()

    print(f"\nDemographic summary ({len(result)} counties):")
    for col in ["pct_white", "pct_black", "pct_hispanic", "pct_college", "median_income"]:
        print(f"  {col}: median={result[col].median():.1f}, "
              f"mean={result[col].mean():.1f}, "
              f"std={result[col].std():.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch Census demographics")
    parser.add_argument("--year", type=int, default=2022,
                        help="ACS 5-year estimate year (default: 2022)")
    parser.add_argument("--api-key", default=None,
                        help="Census API key (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("CENSUS DEMOGRAPHICS FETCHER")
    print("=" * 60)

    raw = fetch_acs(year=args.year, api_key=args.api_key)
    if raw is None:
        print("Failed to fetch data")
        return

    demographics = process_demographics(raw)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "county_demographics.csv")
    demographics.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
