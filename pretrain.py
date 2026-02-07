"""
Stage 1: Domain-adaptive pre-fine-tuning on Miller Center Presidential Speech Archive.

Following Howard & Ruder (2018), this script fine-tunes DistilBERT on ~400 labeled
political speech excerpts before the target advertising corpus, exposing the model
to broader nostalgic political language patterns.

Usage:
    python pretrain.py --data_path miller_center_speeches.csv --output_dir models/pretrained

CSV format: columns 'text' (speech excerpt) and 'label' (0 = non-nostalgic, 1 = nostalgic)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')


class Config:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    SEED = 42
    PATIENCE = 3


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpeechDataset(Dataset):
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

    for batch in tqdm(dataloader, desc='Pre-training'):
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

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions, zero_division=0)
    }


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions, zero_division=0)
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Miller Center pre-fine-tuning')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Miller Center speech excerpts CSV (columns: text, label)')
    parser.add_argument('--output_dir', type=str, default='models/pretrained',
                        help='Directory to save pre-trained model')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE)
    args = parser.parse_args()

    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("STAGE 1: MILLER CENTER PRE-FINE-TUNING")
    print("=" * 60)
    print(f"Device: {device}")

    df = pd.read_csv(args.data_path)
    print(f"\nLoaded {len(df)} speech excerpts")
    print(f"Distribution: {df['label'].value_counts().to_dict()}")

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=Config.SEED, stratify=labels
    )

    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2
    ).to(device)

    train_dataset = SpeechDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
    val_dataset = SpeechDataset(val_texts, val_labels, tokenizer, Config.MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\nTraining on {len(train_texts)} excerpts for {args.epochs} epochs")
    print(f"Validation: {len(val_texts)} excerpts")

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train  Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}  F1: {train_metrics['f1']:.4f}")
        print(f"  Val    Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}  F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> Saved best model (val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"\nPre-training complete. Best validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"\nProceed to Stage 2:")
    print(f"  python train.py --data_path your_ads.csv --pretrained_dir {args.output_dir}")


if __name__ == '__main__':
    main()
