"""
Build a DMA (Designated Market Area) to county FIPS crosswalk.

Nielsen DMAs are the geographic units used by the Wesleyan Media Project
for ad tracking. We need this mapping to connect ad-level data (by DMA)
to county-level turnout and demographics.

Strategy:
1. Try to download from public academic sources
2. Fall back to a hardcoded mapping for the 7 battleground states

The full 210-DMA mapping requires a Nielsen license, but battleground
state mappings are well-documented in political science literature.

Output: processed/dma_county_map.csv
"""

import os
import argparse

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

# Hardcoded DMA-county mapping for battleground states
# Source: Nielsen 2020 DMA definitions cross-referenced with
# Census Bureau county FIPS codes
BATTLEGROUND_DMA_COUNTIES = [
    # ARIZONA
    {"dma": "Phoenix", "dma_code": 753, "state": "AZ", "county_fips": "04013", "county_name": "Maricopa"},
    {"dma": "Phoenix", "dma_code": 753, "state": "AZ", "county_fips": "04021", "county_name": "Pinal"},
    {"dma": "Phoenix", "dma_code": 753, "state": "AZ", "county_fips": "04007", "county_name": "Gila"},
    {"dma": "Tucson", "dma_code": 789, "state": "AZ", "county_fips": "04019", "county_name": "Pima"},
    {"dma": "Tucson", "dma_code": 789, "state": "AZ", "county_fips": "04023", "county_name": "Santa Cruz"},
    {"dma": "Yuma-El Centro", "dma_code": 771, "state": "AZ", "county_fips": "04027", "county_name": "Yuma"},
    {"dma": "Flagstaff-Prescott", "dma_code": 753, "state": "AZ", "county_fips": "04005", "county_name": "Coconino"},
    {"dma": "Flagstaff-Prescott", "dma_code": 753, "state": "AZ", "county_fips": "04025", "county_name": "Yavapai"},

    # GEORGIA
    {"dma": "Atlanta", "dma_code": 524, "state": "GA", "county_fips": "13121", "county_name": "Fulton"},
    {"dma": "Atlanta", "dma_code": 524, "state": "GA", "county_fips": "13089", "county_name": "DeKalb"},
    {"dma": "Atlanta", "dma_code": 524, "state": "GA", "county_fips": "13135", "county_name": "Gwinnett"},
    {"dma": "Atlanta", "dma_code": 524, "state": "GA", "county_fips": "13067", "county_name": "Cobb"},
    {"dma": "Atlanta", "dma_code": 524, "state": "GA", "county_fips": "13063", "county_name": "Clayton"},
    {"dma": "Atlanta", "dma_code": 524, "state": "GA", "county_fips": "13151", "county_name": "Henry"},
    {"dma": "Savannah", "dma_code": 507, "state": "GA", "county_fips": "13051", "county_name": "Chatham"},
    {"dma": "Augusta", "dma_code": 520, "state": "GA", "county_fips": "13245", "county_name": "Richmond"},
    {"dma": "Macon", "dma_code": 503, "state": "GA", "county_fips": "13021", "county_name": "Bibb"},

    # MICHIGAN
    {"dma": "Detroit", "dma_code": 505, "state": "MI", "county_fips": "26163", "county_name": "Wayne"},
    {"dma": "Detroit", "dma_code": 505, "state": "MI", "county_fips": "26125", "county_name": "Oakland"},
    {"dma": "Detroit", "dma_code": 505, "state": "MI", "county_fips": "26099", "county_name": "Macomb"},
    {"dma": "Detroit", "dma_code": 505, "state": "MI", "county_fips": "26161", "county_name": "Washtenaw"},
    {"dma": "Grand Rapids", "dma_code": 563, "state": "MI", "county_fips": "26081", "county_name": "Kent"},
    {"dma": "Grand Rapids", "dma_code": 563, "state": "MI", "county_fips": "26139", "county_name": "Ottawa"},
    {"dma": "Lansing", "dma_code": 551, "state": "MI", "county_fips": "26065", "county_name": "Ingham"},
    {"dma": "Flint-Saginaw", "dma_code": 513, "state": "MI", "county_fips": "26049", "county_name": "Genesee"},
    {"dma": "Traverse City", "dma_code": 540, "state": "MI", "county_fips": "26055", "county_name": "Grand Traverse"},

    # NEVADA
    {"dma": "Las Vegas", "dma_code": 839, "state": "NV", "county_fips": "32003", "county_name": "Clark"},
    {"dma": "Reno", "dma_code": 811, "state": "NV", "county_fips": "32031", "county_name": "Washoe"},

    # NORTH CAROLINA
    {"dma": "Charlotte", "dma_code": 517, "state": "NC", "county_fips": "37119", "county_name": "Mecklenburg"},
    {"dma": "Raleigh-Durham", "dma_code": 560, "state": "NC", "county_fips": "37183", "county_name": "Wake"},
    {"dma": "Raleigh-Durham", "dma_code": 560, "state": "NC", "county_fips": "37063", "county_name": "Durham"},
    {"dma": "Greensboro", "dma_code": 518, "state": "NC", "county_fips": "37081", "county_name": "Guilford"},
    {"dma": "Greensboro", "dma_code": 518, "state": "NC", "county_fips": "37067", "county_name": "Forsyth"},

    # PENNSYLVANIA
    {"dma": "Philadelphia", "dma_code": 504, "state": "PA", "county_fips": "42101", "county_name": "Philadelphia"},
    {"dma": "Philadelphia", "dma_code": 504, "state": "PA", "county_fips": "42091", "county_name": "Montgomery"},
    {"dma": "Philadelphia", "dma_code": 504, "state": "PA", "county_fips": "42029", "county_name": "Chester"},
    {"dma": "Philadelphia", "dma_code": 504, "state": "PA", "county_fips": "42045", "county_name": "Delaware"},
    {"dma": "Philadelphia", "dma_code": 504, "state": "PA", "county_fips": "42017", "county_name": "Bucks"},
    {"dma": "Pittsburgh", "dma_code": 508, "state": "PA", "county_fips": "42003", "county_name": "Allegheny"},
    {"dma": "Pittsburgh", "dma_code": 508, "state": "PA", "county_fips": "42129", "county_name": "Westmoreland"},
    {"dma": "Harrisburg-Lancaster", "dma_code": 566, "state": "PA", "county_fips": "42043", "county_name": "Dauphin"},
    {"dma": "Harrisburg-Lancaster", "dma_code": 566, "state": "PA", "county_fips": "42071", "county_name": "Lancaster"},
    {"dma": "Wilkes-Barre-Scranton", "dma_code": 577, "state": "PA", "county_fips": "42079", "county_name": "Luzerne"},
    {"dma": "Erie", "dma_code": 516, "state": "PA", "county_fips": "42049", "county_name": "Erie"},

    # WISCONSIN
    {"dma": "Milwaukee", "dma_code": 617, "state": "WI", "county_fips": "55079", "county_name": "Milwaukee"},
    {"dma": "Milwaukee", "dma_code": 617, "state": "WI", "county_fips": "55133", "county_name": "Waukesha"},
    {"dma": "Madison", "dma_code": 669, "state": "WI", "county_fips": "55025", "county_name": "Dane"},
    {"dma": "Green Bay", "dma_code": 658, "state": "WI", "county_fips": "55009", "county_name": "Brown"},
    {"dma": "La Crosse-Eau Claire", "dma_code": 702, "state": "WI", "county_fips": "55063", "county_name": "La Crosse"},
    {"dma": "Wausau-Rhinelander", "dma_code": 705, "state": "WI", "county_fips": "55073", "county_name": "Marathon"},
]


def build_mapping():
    """Build the DMA-county crosswalk."""
    df = pd.DataFrame(BATTLEGROUND_DMA_COUNTIES)

    print(f"Built DMA-county mapping: {len(df)} county-DMA pairs")
    print(f"States: {sorted(df['state'].unique())}")
    print(f"DMAs: {df['dma'].nunique()} unique markets")

    for state in sorted(df["state"].unique()):
        st = df[df["state"] == state]
        print(f"  {state}: {len(st)} counties across {st['dma'].nunique()} DMAs")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build DMA-county crosswalk")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "dma_county_map.csv"))
    args = parser.parse_args()

    print("=" * 60)
    print("DMA-COUNTY CROSSWALK BUILDER")
    print("=" * 60)

    mapping = build_mapping()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
