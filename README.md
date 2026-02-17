# Nostalgia Classifier

DistilBERT-based classifier to detect nostalgic framing in political advertisements.

This model accompanies the paper: **"Nostalgic Messaging-Driven Turnout Analysis: Transformer-Based Detection and Temporal Causal Modeling of Political Ad Effects in Battleground States"**

## Project Structure

```
nostalgia-classifier/
├── data/
│   ├── README.md                   # Data source documentation
│   ├── nostalgia_dictionary.json   # Nostalgia lexicon (terms, phrases, weights)
│   ├── scrape_miller_center.py     # Presidential speech transcript scraper
│   ├── label_nostalgia.py          # Rule-based nostalgia labeler
│   ├── prepare_ad_corpus.py        # WMP political ad data processor
│   ├── fetch_election_data.py      # MIT Election Lab county returns
│   ├── fetch_demographics.py       # Census ACS county demographics
│   ├── build_dma_mapping.py        # DMA-to-county FIPS crosswalk
│   └── assemble_did_panel.py       # Final DiD panel assembly
├── scripts/
│   └── build_data.sh              # End-to-end data pipeline runner
├── pretrain.py                     # Stage 1: Miller Center domain adaptation
├── train.py                        # Stage 2: 5-fold CV + production model
├── inference.py                    # Classify new ads (single or batch)
├── data_preparation_did.py         # DiD regression analysis
└── models/
    └── nostalgia_classifier/       # Pre-trained model weights
```

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Build the Dataset

```bash
# Run the full data pipeline (scrapes speeches, fetches election data + demographics)
./scripts/build_data.sh

# Or with a Census API key (optional, for higher rate limits):
./scripts/build_data.sh --census-key YOUR_KEY
```

This will:
- Scrape ~200 presidential speeches from the Miller Center (2000–2024)
- Label speech paragraphs for nostalgic framing using the nostalgia dictionary
- Download county-level election returns from MIT Election Data Lab
- Fetch county demographics from the Census ACS API
- Build the DMA-county crosswalk for battleground states
- Assemble the DiD panel dataset

**Note:** The Wesleyan Media Project ad data must be purchased separately ($20/year, academic access only). See `data/README.md` for details.

### 3. Stage 1: Pre-fine-tune on Political Speeches

```bash
python pretrain.py --data_path data/processed/miller_center_labeled.csv --output_dir models/pretrained
```

Domain-adaptive pre-fine-tuning on ~400+ labeled political speech excerpts. This teaches the model political language patterns before it sees actual ads.

### 4. Stage 2: Fine-tune on Ad Corpus

```bash
python train.py --data_path data/processed/ad_corpus_labeled.csv --pretrained_dir models/pretrained
```

5-fold stratified cross-validation followed by full-dataset retraining. Results saved to `models/nostalgia_classifier/results.json`.

### 5. Run Inference

**Single text:**
```bash
python inference.py --model_dir models/nostalgia_classifier \
    --text "We will make America great again. Return to the prosperity of four years ago."
```

**Batch CSV:**
```bash
python inference.py --model_dir models/nostalgia_classifier \
    --csv_path ads.csv --text_column Transcript
```

## Nostalgia Dictionary

The nostalgia lexicon (`data/nostalgia_dictionary.json`) defines six categories of nostalgic and anti-nostalgic framing, each with weighted terms and phrases:

| Category | Weight | Example Terms |
|---|---|---|
| Restoration | 1.0 | "restore", "bring back", "take our country back" |
| Temporal Past | 0.8 | "remember when", "four years ago", "used to" |
| Golden Age | 0.9 | "greatest economy", "golden age", "shining city" |
| Values/Tradition | 0.7 | "traditional values", "founding fathers", "heritage" |
| Decline Framing | 0.6 | "forgotten", "lost our way", "crumbling" |
| Anti-Nostalgia | -0.5 | "forward", "future", "build back better" |

A paragraph is labeled **nostalgic** when its weighted score ≥ 2.5 and it contains ≥ 2 distinct nostalgic markers.

## Data Sources

| Source | Cost | Use |
|---|---|---|
| [Miller Center](https://millercenter.org/the-presidency/presidential-speeches) | Free | Stage 1 training corpus |
| [Wesleyan Media Project](https://mediaproject.wesleyan.edu/dataaccess/) | $20/yr | Stage 2 ad transcripts |
| [MIT Election Lab](https://electionlab.mit.edu/data) | Free | County-level turnout |
| [Census ACS](https://api.census.gov/) | Free | County demographics |

## Model Architecture

- Base: `distilbert-base-uncased`
- Classification head: 2-class softmax
- Max sequence length: 512 tokens
- Optimizer: AdamW with linear warmup
- Two-stage training: domain adaptation → task fine-tuning
- Evaluation: 5-fold stratified cross-validation
- Production model: retrained on full dataset after CV

## DiD Analysis

The difference-in-differences analysis examines whether changes in nostalgic ad exposure between election cycles predict changes in county-level voter turnout, controlling for demographics. The analysis focuses on seven battleground states: Arizona, Georgia, Michigan, Nevada, North Carolina, Pennsylvania, and Wisconsin.

```bash
# After building the panel dataset:
python data_preparation_did.py
```

## License

MIT
