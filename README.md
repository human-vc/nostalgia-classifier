# Nostalgia Classifier

DistilBERT-based classifier to detect nostalgic framing in political advertisements.

This model accompanies the paper: **"Nostalgic Messaging-Driven Turnout Analysis: Transformer-Based Detection and Temporal Causal Modeling of Political Ad Effects in Battleground States"**

## Two-Stage Training Pipeline

Following Howard & Ruder (2018), the classifier uses sequential fine-tuning:

1. **Stage 1** (`pretrain.py`): Domain-adaptive pre-fine-tuning on ~400 labeled political speech excerpts from the Miller Center Presidential Speech Archive (2016–2024)
2. **Stage 2** (`train.py`): Fine-tuning on the target advertising corpus with 5-fold stratified cross-validation, then retraining the final production model on the full dataset

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Stage 1: Pre-fine-tune on Miller Center speeches

```bash
python pretrain.py --data_path miller_center_speeches.csv --output_dir models/pretrained
```

CSV should have columns: `text` (speech excerpt) and `label` (0 or 1).

### 3. Stage 2: Fine-tune on ad corpus with 5-fold CV

```bash
python train.py --data_path your_ads.csv --pretrained_dir models/pretrained
```

CSV should have columns: `Transcript` (text) and `Nostalgia_Binary` (0 or 1).

This will:
- Run 5-fold stratified cross-validation to evaluate generalization
- Retrain the final production model on the full dataset
- Save results to `models/nostalgia_classifier/results.json`

### 4. Run Inference

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

- `pretrain.py` - Stage 1: Miller Center domain-adaptive pre-fine-tuning
- `train.py` - Stage 2: 5-fold CV + full-dataset retraining on ad corpus
- `inference.py` - Classify new ads (single or batch)
- `requirements.txt` - Dependencies
- `models/nostalgia_classifier/` - Pre-trained model weights

## Model Architecture

- Base: `distilbert-base-uncased` (Sanh et al., 2019)
- Classification head: 2-class softmax
- Max sequence length: 512 tokens
- Optimizer: AdamW (Loshchilov & Hutter, 2019) with linear warmup
- Evaluation: 5-fold stratified cross-validation (mean F1 = 0.91, SD = 0.02)
- Production model: Retrained on full dataset after CV evaluation

## References

- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *ACL*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
