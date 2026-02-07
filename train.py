"""
Stage 2: Fine-tune DistilBERT on political advertisement corpus.

Supports two-stage transfer learning (Howard & Ruder, 2018):
  1. Pre-fine-tune on Miller Center speeches (pretrain.py)
  2. Fine-tune on target ad corpus with 5-fold stratified CV (this script)

The final production model is retrained on the full dataset after CV evaluation.

Usage:
    # From scratch (no pre-training)
    python train.py --data_path ads.csv

    # Two-stage (after running pretrain.py)
    python train.py --data_path ads.csv --pretrained_dir models/pretrained

CSV format: columns 'Transcript' (text) and 'Nostalgia_Binary' (0 or 1)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
import warnings
warnings.filterwarnings('ignore')


class Config:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    SEED = 42
    CV_FOLDS = 5
    PATIENCE = 3


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PoliticalAdDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

    for batch in tqdm(dataloader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(true_labels, predictions)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions, true_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'auc': roc_auc_score(true_labels, all_probs) if len(set(true_labels)) > 1 else 0.0
    }, predictions, true_labels


def load_base_model(pretrained_dir, device):
    """Load model from pretrained checkpoint (Stage 1) or from scratch."""
    if pretrained_dir and os.path.exists(pretrained_dir):
        print(f"Loading pre-trained model from: {pretrained_dir}")
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_dir)
        model = DistilBertForSequenceClassification.from_pretrained(pretrained_dir)
    else:
        if pretrained_dir:
            print(f"Warning: {pretrained_dir} not found, loading base DistilBERT")
        print(f"Loading base DistilBERT: {Config.MODEL_NAME}")
        tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(
            Config.MODEL_NAME, num_labels=2
        )
    model.to(device)
    return model, tokenizer


def train_single_fold(model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience):
    """Train one fold with early stopping. Returns best model state and val metrics."""
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate with best weights
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    val_metrics, val_preds, val_true = evaluate(model, val_loader, device)
    return val_metrics, val_preds, val_true, best_model_state


def cross_validate(texts, labels, tokenizer, device, args):
    """5-fold stratified cross-validation to evaluate generalization performance."""
    skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"{Config.CV_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")

    texts_arr = np.array(texts)
    labels_arr = np.array(labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        print(f"\n--- Fold {fold + 1}/{Config.CV_FOLDS} ---")

        fold_train_texts = texts_arr[train_idx].tolist()
        fold_train_labels = labels_arr[train_idx].tolist()
        fold_val_texts = texts_arr[val_idx].tolist()
        fold_val_labels = labels_arr[val_idx].tolist()

        print(f"  Train: {len(fold_train_texts)} | Val: {len(fold_val_texts)}")

        # Fresh model for each fold
        model, _ = load_base_model(args.pretrained_dir, device)

        train_dataset = PoliticalAdDataset(fold_train_texts, fold_train_labels, tokenizer, Config.MAX_LENGTH)
        val_dataset = PoliticalAdDataset(fold_val_texts, fold_val_labels, tokenizer, Config.MAX_LENGTH)

        batch_size = args.batch_size or Config.BATCH_SIZE
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        epochs = args.epochs or Config.EPOCHS
        total_steps = len(train_loader) * epochs
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * Config.WARMUP_RATIO),
            num_training_steps=total_steps
        )

        val_metrics, _, _, _ = train_single_fold(
            model, train_loader, val_loader, optimizer, scheduler,
            device, epochs, Config.PATIENCE
        )

        fold_metrics.append(val_metrics)
        print(f"  F1: {val_metrics['f1']:.4f}  AUC: {val_metrics['auc']:.4f}  "
              f"Acc: {val_metrics['accuracy']:.4f}")

    # Aggregate CV results
    metric_names = ['f1', 'auc', 'accuracy', 'precision', 'recall']
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    cv_summary = {}
    for m in metric_names:
        values = [fm[m] for fm in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv_summary[m] = {'mean': mean_val, 'std': std_val}
        print(f"  {m.upper():>10s}: {mean_val:.4f} (SD = {std_val:.4f})")

    return cv_summary, fold_metrics


def train_final_model(texts, labels, tokenizer, device, args):
    """Retrain on the full dataset for production use."""
    print(f"\n{'='*60}")
    print("FINAL MODEL: RETRAINING ON FULL DATASET")
    print(f"{'='*60}")
    print(f"Training on all {len(texts)} advertisements")

    model, _ = load_base_model(args.pretrained_dir, device)

    full_dataset = PoliticalAdDataset(texts, labels, tokenizer, Config.MAX_LENGTH)
    batch_size = args.batch_size or Config.BATCH_SIZE
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    epochs = args.epochs or Config.EPOCHS
    total_steps = len(full_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * Config.WARMUP_RATIO),
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, full_loader, optimizer, scheduler, device)
        print(f"  Epoch {epoch + 1}/{epochs}  Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")

    # Generate confusion matrix on full corpus (production model self-evaluation)
    eval_loader = DataLoader(full_dataset, batch_size=batch_size)
    eval_metrics, eval_preds, eval_true = evaluate(model, eval_loader, device)

    cm = confusion_matrix(eval_true, eval_preds)
    print(f"\nFull-corpus confusion matrix (production model):")
    print(f"                Pred Non-Nost  Pred Nost")
    print(f"  Actual Non-Nost    {cm[0,0]:4d}         {cm[0,1]:4d}")
    print(f"  Actual Nost        {cm[1,0]:4d}         {cm[1,1]:4d}")
    print(f"\n{classification_report(eval_true, eval_preds, target_names=['Non-Nostalgic', 'Nostalgic'])}")

    return model, tokenizer, eval_metrics, eval_preds, eval_true


def main(args):
    print("=" * 60)
    print("NOSTALGIA CLASSIFIER - DistilBERT")
    print("=" * 60)

    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Total: {len(df)} | Nostalgic: {df['Nostalgia_Binary'].sum()} | "
          f"Non-nostalgic: {(df['Nostalgia_Binary']==0).sum()}")

    texts = df['Transcript'].tolist()
    labels = df['Nostalgia_Binary'].tolist()

    # Load tokenizer (from pretrained or base)
    if args.pretrained_dir and os.path.exists(args.pretrained_dir):
        tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_dir)
        print(f"Using tokenizer from: {args.pretrained_dir}")
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)

    # Phase 1: Cross-validation for evaluation
    cv_summary, fold_metrics = cross_validate(texts, labels, tokenizer, device, args)

    # Phase 2: Retrain on full dataset for production
    model, tokenizer, final_metrics, final_preds, final_true = train_final_model(
        texts, labels, tokenizer, device, args
    )

    # Save model and results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = {
        'cross_validation': {
            'n_folds': Config.CV_FOLDS,
            'summary': {k: {'mean': float(v['mean']), 'std': float(v['std'])}
                       for k, v in cv_summary.items()},
            'fold_metrics': [{k: float(v) for k, v in fm.items()} for fm in fold_metrics]
        },
        'final_model': {k: float(v) for k, v in final_metrics.items()},
        'config': {
            'max_length': Config.MAX_LENGTH,
            'batch_size': args.batch_size or Config.BATCH_SIZE,
            'epochs': args.epochs or Config.EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'pretrained_dir': args.pretrained_dir
        }
    }
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame([final_metrics]).to_csv(f'{output_dir}/metrics.csv', index=False)
    print(f"\nModel saved to: {output_dir}")
    print(f"Results saved to: {output_dir}/results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: Fine-tune on political ad corpus')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ad corpus CSV (columns: Transcript, Nostalgia_Binary)')
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='Path to Stage 1 pre-trained model (from pretrain.py)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='models/nostalgia_classifier')
    args = parser.parse_args()
    main(args)
