# Nostalgia Classifier

DistilBERT-based classifier to detect nostalgic framing in political advertisements.

Accompanies: *Nostalgic Messaging-Driven Turnout Analysis: Transformer-Based Detection and Temporal Causal Modeling of Political Ad Effects in Battleground States*

## Two-Stage Training

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

### 4. Run Inference

Single text:

```bash
python inference.py --model_dir models/nostalgia_classifier \
    --text "We will make America great again. Return to the prosperity of four years ago."
```

Batch CSV:

```bash
python inference.py --model_dir models/nostalgia_classifier \
    --csv_path ads.csv --text_column Transcript
```

## Files

| File | Description |
|------|-------------|
| `pretrain.py` | Stage 1: Miller Center domain-adaptive pre-fine-tuning |
| `train.py` | Stage 2: 5-fold CV + full-dataset retraining on ad corpus |
| `inference.py` | Classify new ads (single or batch) |
| `models/nostalgia_classifier/` | Pre-trained model weights |

## Model Architecture

- **Base**: `distilbert-base-uncased` (6 transformer blocks, 66M parameters)
- **Classification head**: 2-class softmax
- **Max sequence length**: 512 tokens
- **Optimizer**: AdamW (lr = 2e-5) with linear warmup
- **Evaluation**: 5-fold stratified cross-validation
