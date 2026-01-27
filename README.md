# Nostalgia Classifier

DistilBERT-based classifier to detect nostalgic framing in political advertisements.

## Quick Start with Claude Code

### 1. Setup
```bash
cd nostalgia_classifier
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py --data_path full_450_ads_realistic.csv --epochs 5 --batch_size 16
```

**Expected output:**
- Accuracy: ~75-85%
- F1 Score: ~0.70-0.80
- Cohen's κ: ~0.50-0.65

This is a realistic range for this task. If you see 95%+ accuracy, there's likely data leakage.

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

Based on the dictionary from "When Nostalgia Backfires":

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
- `full_450_ads_realistic.csv` - Training data (450 ads, paper proportions)
- `requirements.txt` - Dependencies

## Model Architecture

- Base: `distilbert-base-uncased`
- Classification head: 2-class softmax
- Max sequence length: 256 tokens
- Training: AdamW with linear warmup

## Citation

If using for research, cite the original paper:
> "When Nostalgia Backfires: The Conditional Effects of Nostalgic Appeals in Political Advertising"

## Notes

The training data uses realistic language patterns from 2024 campaign ads, preserving the paper's proportions:
- Trump Campaign: 57.8% nostalgic
- MAGA Inc: 77.1% nostalgic  
- Harris Campaign: 9.8% nostalgic
- FF PAC: 1.6% nostalgic

Overall: Republican ads ~64% nostalgic, Democratic ads ~8% nostalgic.
