"""
Process Wesleyan Media Project (WMP) ad data for Stage 2 training.

This script takes raw WMP data files (.dta or .csv) and produces a
clean labeled corpus for fine-tuning the nostalgia classifier.

WMP data must be purchased separately ($20/year from mediaproject.wesleyan.edu).
Place files in data/raw/ as:
  - wmp_2020.dta (or .csv)
  - wmp_2024.dta (or .csv)

The script:
1. Loads and standardizes WMP data across election cycles
2. Extracts ad transcripts (the 'creative' or 'transcript' field)
3. Deduplicates by creative (same ad aired many times)
4. Applies the nostalgia dictionary for initial labeling
5. Exports for manual review and Stage 2 training

Output: processed/ad_corpus_labeled.csv
"""

import os
import json
import argparse
from collections import Counter

import pandas as pd
import numpy as np

# Try to import Stata reader
try:
    from pandas.io.stata import read_stata
except ImportError:
    read_stata = None

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def load_wmp_file(path):
    """Load WMP data from Stata (.dta) or CSV format."""
    if path.endswith(".dta"):
        return pd.read_stata(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown format: {path}")


def standardize_wmp(df, year):
    """
    Standardize WMP column names across election cycles.

    WMP variable names change slightly between years. This maps them
    to a consistent schema.
    """
    # Common column name mappings (WMP name → our name)
    col_maps = {
        # Transcript/creative text
        "creative": "transcript",
        "ad_creative": "transcript",
        "Transcript": "transcript",

        # DMA
        "dma": "dma",
        "dma_name": "dma",
        "market": "dma",

        # Party
        "party": "party",
        "ad_party": "party",

        # Tone
        "ad_tone": "tone",
        "tone": "tone",

        # Race level
        "race": "race_level",
        "office": "race_level",

        # Sponsor
        "sponsor": "sponsor",
        "ad_sponsor": "sponsor",
    }

    renamed = {}
    for old_name, new_name in col_maps.items():
        if old_name in df.columns and new_name not in renamed.values():
            renamed[old_name] = new_name

    df = df.rename(columns=renamed)
    df["year"] = year

    # Keep only columns we need
    keep_cols = ["transcript", "dma", "party", "tone", "race_level",
                 "sponsor", "year"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    return df


def deduplicate_creatives(df):
    """
    Deduplicate by ad creative text.

    The same ad airs thousands of times. For training the classifier,
    we only need unique transcripts. We keep metadata from the first
    occurrence and add an airing count.
    """
    if "transcript" not in df.columns:
        print("WARNING: No transcript column found!")
        return df

    # Count airings per unique transcript
    airing_counts = df.groupby("transcript").size().reset_index(name="n_airings")

    # Keep first occurrence of each transcript
    deduped = df.drop_duplicates(subset=["transcript"], keep="first")
    deduped = deduped.merge(airing_counts, on="transcript", how="left")

    print(f"  Deduplicated: {len(df)} airings → {len(deduped)} unique creatives")
    return deduped


def apply_nostalgia_labels(df, dict_path=None):
    """
    Apply rule-based nostalgia labeling to ad transcripts.

    This provides initial labels that should be manually reviewed.
    The label_nostalgia.py module handles the actual scoring.
    """
    if dict_path is None:
        dict_path = os.path.join(os.path.dirname(__file__), "nostalgia_dictionary.json")

    # Import the labeler
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from label_nostalgia import load_dictionary, score_text

    categories, rules = load_dictionary(dict_path)
    threshold = rules.get("positive_threshold", 2.5)
    min_markers = rules.get("min_terms_for_positive", 2)

    scores = []
    labels = []
    marker_counts = []

    for text in df["transcript"].fillna(""):
        if len(str(text).split()) < 10:
            scores.append(0.0)
            labels.append(0)
            marker_counts.append(0)
            continue

        score, n_markers, _ = score_text(str(text), categories)
        scores.append(round(score, 3))
        marker_counts.append(n_markers)

        if score >= threshold and n_markers >= min_markers:
            labels.append(1)
        else:
            labels.append(0)

    df = df.copy()
    df["nostalgia_score"] = scores
    df["Nostalgia_Binary"] = labels
    df["n_markers"] = marker_counts

    return df


def main():
    parser = argparse.ArgumentParser(description="Process WMP ad data")
    parser.add_argument("--wmp-2020", default=None,
                        help="Path to WMP 2020 data file")
    parser.add_argument("--wmp-2024", default=None,
                        help="Path to WMP 2024 data file")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "ad_corpus_labeled.csv"))
    args = parser.parse_args()

    print("=" * 60)
    print("WMP AD CORPUS PROCESSOR")
    print("=" * 60)

    # Auto-detect WMP files in raw/
    dfs = []
    for year, path_arg in [(2020, args.wmp_2020), (2024, args.wmp_2024)]:
        if path_arg:
            path = path_arg
        else:
            # Try common filenames
            for ext in [".dta", ".csv"]:
                candidate = os.path.join(RAW_DIR, f"wmp_{year}{ext}")
                if os.path.exists(candidate):
                    path = candidate
                    break
            else:
                print(f"  No WMP {year} data found (looked in {RAW_DIR}/)")
                continue

        print(f"\nLoading {year} data from {path}...")
        df = load_wmp_file(path)
        print(f"  Raw: {len(df)} rows, {len(df.columns)} columns")

        df = standardize_wmp(df, year)
        df = deduplicate_creatives(df)
        dfs.append(df)

    if not dfs:
        print("\nNo WMP data found. To use this script:")
        print("  1. Purchase data from https://mediaproject.wesleyan.edu/dataaccess/")
        print("  2. Place files in data/raw/ as wmp_2020.dta and wmp_2024.dta")
        print("  3. Re-run this script")

        # Create a sample/template file so Jacob knows the expected format
        sample = pd.DataFrame({
            "Transcript": [
                "We need to bring back the prosperity of four years ago. Remember when gas was $2 a gallon?",
                "My opponent voted against healthcare three times. That's his record.",
                "Our founding fathers built something special. We've lost our way. Time to take it back.",
                "I'm fighting for a new future. More jobs, better schools, real opportunity.",
            ],
            "Nostalgia_Binary": [1, 0, 1, 0],
        })
        sample_path = os.path.join(OUTPUT_DIR, "ad_corpus_sample.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sample.to_csv(sample_path, index=False)
        print(f"\nCreated sample file: {sample_path}")
        print("You can test train.py with this sample data.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} unique ads across {combined['year'].nunique()} cycles")

    # Apply nostalgia labels
    print("\nApplying nostalgia labeling...")
    labeled = apply_nostalgia_labels(combined)

    # Summary
    print(f"\nLabeling summary:")
    print(f"  Total ads: {len(labeled)}")
    print(f"  Nostalgic: {(labeled['Nostalgia_Binary']==1).sum()} "
          f"({(labeled['Nostalgia_Binary']==1).mean()*100:.1f}%)")
    print(f"  Non-nostalgic: {(labeled['Nostalgia_Binary']==0).sum()} "
          f"({(labeled['Nostalgia_Binary']==0).mean()*100:.1f}%)")

    if "party" in labeled.columns:
        print(f"\n  By party:")
        for party, group in labeled.groupby("party"):
            n_nost = (group["Nostalgia_Binary"] == 1).sum()
            print(f"    {party}: {n_nost}/{len(group)} nostalgic ({n_nost/len(group)*100:.1f}%)")

    if "year" in labeled.columns:
        print(f"\n  By year:")
        for year, group in labeled.groupby("year"):
            n_nost = (group["Nostalgia_Binary"] == 1).sum()
            print(f"    {year}: {n_nost}/{len(group)} nostalgic ({n_nost/len(group)*100:.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the Transcript + Nostalgia_Binary format expected by train.py
    train_cols = labeled.rename(columns={"transcript": "Transcript"})
    train_cols = train_cols[["Transcript", "Nostalgia_Binary"] +
                            [c for c in train_cols.columns
                             if c not in ("Transcript", "Nostalgia_Binary")]]
    train_cols.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Also save a review file with high-confidence and borderline cases
    borderline = labeled[
        (labeled["nostalgia_score"] > 1.0) & (labeled["nostalgia_score"] < 4.0)
    ].copy()
    if len(borderline) > 0:
        review_path = os.path.join(OUTPUT_DIR, "ad_corpus_review.csv")
        borderline.to_csv(review_path, index=False)
        print(f"Borderline cases for manual review: {review_path} ({len(borderline)} ads)")


if __name__ == "__main__":
    main()
