# Nostalgia Classifier

DistilBERT-based classifier to detect nostalgic framing in political advertisements.

This model accompanies the paper: **"When Nostalgia Backfires: Context-Dependent Effects of Nostalgic Political Advertising in Electoral Flux"**

## Quick Start

### 1. Setup
```bash
cd nostalgia_classifier
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py --data_path your_data.csv --epochs 5 --batch_size 16
```

Your CSV should have columns: `Transcript` (text) and `Nostalgia_Binary` (0 or 1).

### 3. Run Inference

**Single text:**
```bash
python inference.py --model_dir models/nostalgia_classifier \
    --text "We will make America great again. Return to the prosperity of four years ago."
```

**CSV file:**
```bash
python inference.py --model_dir models/nostalgia_classifier \
    --csv_path new_ads.csv --output_path predictions.csv
```

## What Makes an Ad "Nostalgic"?

Based on the dictionary from the paper:

### Nostalgic (Label = 1)
- **Restoration language:** "restore", "return", "bring back", "again"
- **Temporal references:** "remember when", "four years ago", "before", "once"
- **Values framing:** "traditional", "our values", "the way things were"
- **Golden age rhetoric:** "greatest economy", "golden age", "prosperity"

### Non-Nostalgic (Label = 0)
- **Forward-looking:** "new way forward", "build", "future", "opportunity"
- **Attack ads:** Criticism without past-focused longing
- **Policy-specific:** Concrete proposals without temporal framing

## Files

- `train.py` - Main training script
- `inference.py` - Classify new ads
- `requirements.txt` - Dependencies
- `models/nostalgia_classifier/` - Pre-trained model weights

## Model Architecture

- Base: `distilbert-base-uncased`
- Classification head: 2-class softmax
- Max sequence length: 256 tokens
- Training: AdamW with linear warmup

## Citation

If using this classifier, please cite:

> Crainic, J. (2025). When Nostalgia Backfires: Context-Dependent Effects of Nostalgic Political Advertising in Electoral Flux.
