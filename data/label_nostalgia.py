"""
Rule-based nostalgia labeler for political text.

Takes raw speech transcripts (from scrape_miller_center.py) and produces
labeled training data for Stage 1. Uses the nostalgia dictionary to assign
binary labels with confidence scores.

The labeler works at the paragraph level: each paragraph gets a nostalgia
score based on weighted keyword/phrase matches, then thresholded to binary.

Output: processed/miller_center_labeled.csv
Columns: text, label, score, n_markers, president, source_title
"""

import os
import re
import json
import argparse
from collections import defaultdict

import pandas as pd

DICT_PATH = os.path.join(os.path.dirname(__file__), "nostalgia_dictionary.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")


def load_dictionary(path=DICT_PATH):
    with open(path) as f:
        d = json.load(f)

    categories = {}
    for key, val in d.items():
        if key.startswith("_") or key == "labeling_rules":
            continue
        categories[key] = {
            "weight": val["weight"],
            "terms": [t.lower() for t in val.get("terms", [])],
            "phrases": [p.lower() for p in val.get("phrases", [])],
        }

    rules = d.get("labeling_rules", {})
    return categories, rules


def score_text(text, categories):
    """
    Score a text passage for nostalgic framing.

    Returns (total_score, n_positive_markers, marker_details).
    """
    text_lower = text.lower()
    total_score = 0.0
    n_markers = 0
    details = []

    for cat_name, cat in categories.items():
        weight = cat["weight"]

        # Check phrases first (longer matches, more specific)
        for phrase in cat["phrases"]:
            count = text_lower.count(phrase)
            if count > 0:
                contribution = weight * count
                total_score += contribution
                if weight > 0:
                    n_markers += count
                details.append({
                    "category": cat_name,
                    "match": phrase,
                    "type": "phrase",
                    "count": count,
                    "contribution": contribution,
                })

        # Check individual terms (with word boundary matching)
        for term in cat["terms"]:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(term) + r"\b"
            matches = re.findall(pattern, text_lower)
            count = len(matches)
            if count > 0:
                contribution = weight * count
                total_score += contribution
                if weight > 0:
                    n_markers += count
                details.append({
                    "category": cat_name,
                    "match": term,
                    "type": "term",
                    "count": count,
                    "contribution": contribution,
                })

    return total_score, n_markers, details


def label_paragraphs(speeches_csv, categories, rules):
    """
    Process speech CSV into labeled paragraph-level dataset.
    """
    df = pd.read_csv(speeches_csv)
    print(f"Loaded {len(df)} speeches")

    threshold = rules.get("positive_threshold", 2.5)
    min_markers = rules.get("min_terms_for_positive", 2)
    min_words = rules.get("paragraph_min_words", 15)

    records = []
    stats = defaultdict(int)

    for _, row in df.iterrows():
        try:
            paragraphs = json.loads(row["paragraphs"])
        except (json.JSONDecodeError, TypeError):
            continue

        for para in paragraphs:
            # Skip short paragraphs
            if len(para.split()) < min_words:
                stats["skipped_short"] += 1
                continue

            score, n_markers, details = score_text(para, categories)

            # Binary label
            if score >= threshold and n_markers >= min_markers:
                label = 1
                stats["nostalgic"] += 1
            else:
                label = 0
                stats["non_nostalgic"] += 1

            records.append({
                "text": para,
                "label": label,
                "score": round(score, 3),
                "n_markers": n_markers,
                "president": row.get("president", ""),
                "source_title": row.get("title", ""),
                "source_date": row.get("date", ""),
            })

    labeled_df = pd.DataFrame(records)

    print(f"\nLabeling Results:")
    print(f"  Total paragraphs: {len(records)}")
    print(f"  Nostalgic (1): {stats['nostalgic']} ({stats['nostalgic']/max(len(records),1)*100:.1f}%)")
    print(f"  Non-nostalgic (0): {stats['non_nostalgic']} ({stats['non_nostalgic']/max(len(records),1)*100:.1f}%)")
    print(f"  Skipped (too short): {stats['skipped_short']}")

    return labeled_df


def balance_dataset(df, ratio=3.0, seed=42):
    """
    Balance the dataset by downsampling the majority class.

    Political speeches are overwhelmingly non-nostalgic, so we downsample
    to at most `ratio` non-nostalgic per nostalgic example. This prevents
    the classifier from learning a trivial "always predict 0" strategy.
    """
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    max_neg = int(len(pos) * ratio)

    if len(neg) > max_neg:
        neg_sampled = neg.sample(n=max_neg, random_state=seed)
        print(f"\nBalancing: {len(pos)} pos + {max_neg} neg (downsampled from {len(neg)})")
    else:
        neg_sampled = neg
        print(f"\nNo balancing needed: {len(pos)} pos + {len(neg)} neg")

    balanced = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


def augment_with_synthetic(df, n_synthetic=50, seed=42):
    """
    Add synthetic nostalgic examples from known campaign phrases.

    These are real phrases from political campaigns, reformulated to
    increase diversity in the training set. NOT generated by LLM --
    these are hand-curated from actual campaign materials.
    """
    import random
    random.seed(seed)

    nostalgic_templates = [
        "We will restore the greatness that this country once knew. Remember when America was the envy of the world?",
        "Four years ago, our economy was booming. Jobs were plentiful. Our borders were secure. We need to bring that back.",
        "Our founding fathers built this nation on principles we've abandoned. It's time to return to those values.",
        "There was a time when a single income could support a family. When neighborhoods were safe. We've lost our way.",
        "The forgotten men and women of this country will be forgotten no more. We will take our country back.",
        "Remember the prosperity of just a few years ago? Gas prices were low, wages were rising. We can have that again.",
        "Our grandparents built the greatest nation on earth. We owe it to them to restore what's been lost.",
        "This country was once a shining city on a hill. We've let it crumble. Together, we'll rebuild it.",
        "Traditional American values are under attack. The heritage our parents passed down is being eroded every day.",
        "Before the current administration, our military was the strongest in the world. We need to reclaim that strength.",
        "The American Dream used to mean something. Hard work meant a good life. We need to bring back that promise.",
        "We've watched our communities decline for decades. Factories closed, jobs shipped overseas. No more.",
        "In the golden age of American manufacturing, every family had opportunity. We will revive that era.",
        "Our way of life is disappearing. The America we grew up in is barely recognizable. Time to take it back.",
        "What happened to the country we knew? The one where neighbors looked out for each other? We're going to restore that.",
    ]

    non_nostalgic_templates = [
        "My plan will create 10 million new jobs in clean energy over the next decade.",
        "We need to invest in our future -- in education, infrastructure, and innovation.",
        "The opponent voted against healthcare reform three times. That's the record.",
        "I'm proposing a new tax credit for working families making under $75,000 a year.",
        "We can build a new economy that works for everyone, not just those at the top.",
        "Our children deserve world-class schools. My plan doubles funding for Title I.",
        "It's time to turn the page and write a new chapter in American history.",
        "The path forward requires bold action on climate, healthcare, and education.",
        "We don't need to look backward. We need fresh ideas and modern solutions.",
        "I'll fight every day to expand opportunity and build a stronger middle class.",
    ]

    synthetic_records = []
    for text in nostalgic_templates:
        synthetic_records.append({
            "text": text,
            "label": 1,
            "score": 5.0,
            "n_markers": 3,
            "president": "synthetic",
            "source_title": "campaign_templates",
            "source_date": "",
        })

    for text in non_nostalgic_templates:
        synthetic_records.append({
            "text": text,
            "label": 0,
            "score": -0.5,
            "n_markers": 0,
            "president": "synthetic",
            "source_title": "campaign_templates",
            "source_date": "",
        })

    synthetic_df = pd.DataFrame(synthetic_records)
    combined = pd.concat([df, synthetic_df]).reset_index(drop=True)
    print(f"Added {len(synthetic_records)} synthetic examples ({len(nostalgic_templates)} nostalgic, {len(non_nostalgic_templates)} non-nostalgic)")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Label speeches for nostalgia")
    parser.add_argument("--input", default=os.path.join(
        os.path.dirname(__file__), "raw", "miller_center_speeches.csv"))
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "miller_center_labeled.csv"))
    parser.add_argument("--balance-ratio", type=float, default=3.0,
                        help="Max non-nostalgic:nostalgic ratio (default: 3.0)")
    parser.add_argument("--add-synthetic", action="store_true",
                        help="Add hand-curated synthetic examples")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip dataset balancing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed scoring for each paragraph")
    args = parser.parse_args()

    print("=" * 60)
    print("NOSTALGIA LABELER")
    print("=" * 60)

    categories, rules = load_dictionary()
    print(f"Loaded {len(categories)} categories from dictionary")

    labeled = label_paragraphs(args.input, categories, rules)

    if args.add_synthetic:
        labeled = augment_with_synthetic(labeled)

    if not args.no_balance:
        labeled = balance_dataset(labeled, ratio=args.balance_ratio)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labeled.to_csv(args.output, index=False)
    print(f"\nSaved {len(labeled)} examples to {args.output}")

    # Summary statistics
    print(f"\nFinal Dataset Statistics:")
    print(f"  Total: {len(labeled)}")
    print(f"  Label 1: {(labeled['label']==1).sum()}")
    print(f"  Label 0: {(labeled['label']==0).sum()}")
    print(f"  Mean score: {labeled['score'].mean():.3f}")
    print(f"  Score range: [{labeled['score'].min():.3f}, {labeled['score'].max():.3f}]")

    if args.verbose:
        print(f"\nTop 10 most nostalgic paragraphs:")
        top = labeled.nlargest(10, "score")
        for _, row in top.iterrows():
            print(f"  [{row['score']:.2f}] {row['text'][:100]}...")


if __name__ == "__main__":
    main()
