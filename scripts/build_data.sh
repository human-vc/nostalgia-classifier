#!/usr/bin/env bash
# =============================================================
# Nostalgia Classifier — Full Data Pipeline
# =============================================================
# Builds the complete dataset from public sources.
# WMP ad data must be downloaded separately (see data/README.md).
#
# Usage:
#   ./scripts/build_data.sh              # Run everything
#   ./scripts/build_data.sh --skip-scrape # Skip Miller Center (if cached)
#   ./scripts/build_data.sh --census-key YOUR_KEY  # Use Census API key
# =============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_SCRAPE=false
CENSUS_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-scrape) SKIP_SCRAPE=true; shift ;;
        --census-key) CENSUS_KEY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================="
echo " NOSTALGIA CLASSIFIER — DATA PIPELINE"
echo "============================================================="
echo ""

# Create directories
mkdir -p data/raw data/processed

# Step 1: Scrape Miller Center speeches
if [ "$SKIP_SCRAPE" = false ]; then
    echo ">>> Step 1: Scraping Miller Center speeches..."
    python data/scrape_miller_center.py --start-year 2000
    echo ""
else
    echo ">>> Step 1: SKIPPED (--skip-scrape)"
    echo ""
fi

# Step 2: Label speeches for nostalgia
if [ -f "data/raw/miller_center_speeches.csv" ]; then
    echo ">>> Step 2: Labeling speeches..."
    python data/label_nostalgia.py --add-synthetic
    echo ""
else
    echo ">>> Step 2: SKIPPED (no speech data yet)"
    echo ""
fi

# Step 3: Fetch election data
echo ">>> Step 3: Fetching election data..."
ELECTION_ARGS="--years 2016 2020"
python data/fetch_election_data.py $ELECTION_ARGS
echo ""

# Step 4: Fetch demographics
echo ">>> Step 4: Fetching Census demographics..."
DEMO_ARGS=""
if [ -n "$CENSUS_KEY" ]; then
    DEMO_ARGS="--api-key $CENSUS_KEY"
fi
python data/fetch_demographics.py $DEMO_ARGS
echo ""

# Step 5: Build DMA mapping
echo ">>> Step 5: Building DMA-county crosswalk..."
python data/build_dma_mapping.py
echo ""

# Step 6: Process WMP data (if available)
if ls data/raw/wmp_20*.{dta,csv} 1>/dev/null 2>&1; then
    echo ">>> Step 6: Processing WMP ad data..."
    python data/prepare_ad_corpus.py
    echo ""
else
    echo ">>> Step 6: SKIPPED (no WMP data in data/raw/)"
    echo "    Purchase from: https://mediaproject.wesleyan.edu/dataaccess/"
    echo "    Creating sample data for testing..."
    python data/prepare_ad_corpus.py
    echo ""
fi

# Step 7: Assemble DiD panel
echo ">>> Step 7: Assembling DiD panel dataset..."
python data/assemble_did_panel.py
echo ""

echo "============================================================="
echo " PIPELINE COMPLETE"
echo "============================================================="
echo ""
echo "Data files:"
ls -lh data/processed/*.csv 2>/dev/null || echo "  (none yet)"
echo ""
echo "Next steps:"
echo "  1. Run Stage 1: python pretrain.py --data_path data/processed/miller_center_labeled.csv"
echo "  2. Run Stage 2: python train.py --data_path data/processed/ad_corpus_labeled.csv"
echo "  3. Run DiD:     python data_preparation_did.py"
