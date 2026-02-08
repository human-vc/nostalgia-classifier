"""
Stage 1 of the nostalgia classifier pipeline.
Pre-fine-tunes DistilBERT on Miller Center presidential speech excerpts
so the model picks up political language patterns before we hit it with
the actual ad corpus in train.py.
"""

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

SEED = 42
LR = 2e-5
WARMUP_FRACTION = 0.1
GRAD_CLIP = 1.0


def seed_everything(s=SEED):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class SpeechDataset(Dataset):
    """Wraps speech excerpts for the DataLoader."""
    def __init__(self, texts, labels, tok, maxlen=512):
        self.texts, self.labels = texts, labels
        self.tok = tok
        self.maxlen = maxlen

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), truncation=True,
                       padding='max_length', max_length=self.maxlen,
                       return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[i], dtype=torch.long),
        }


def run_train_epoch(model, loader, optim, sched, dev):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc='Pre-training'):
        ids = batch['input_ids'].to(dev)
        mask = batch['attention_mask'].to(dev)
        labs = batch['label'].to(dev)

        optim.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labs)
        out.loss.backward()
        running_loss += out.loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()
        sched.step()

        all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        all_labels.extend(labs.cpu().numpy())

    n = len(loader)
    return running_loss / n, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, zero_division=0)


def run_eval(model, loader, dev):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(dev)
            mask = batch['attention_mask'].to(dev)
            labs = batch['label'].to(dev)

            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            running_loss += out.loss.item()
            all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    n = len(loader)
    return running_loss / n, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, zero_division=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Miller Center pre-fine-tuning')
    parser.add_argument('--data_path', required=True, help='CSV with columns: text, label')
    parser.add_argument('--output_dir', default='models/pretrained')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=LR)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    seed_everything()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("STAGE 1: MILLER CENTER PRE-FINE-TUNING")
    print("=" * 60)
    print(f"Device: {dev}")

    # load data
    df = pd.read_csv(args.data_path)
    print(f"\nLoaded {len(df)} speech excerpts")
    print(f"Distribution: {df['label'].value_counts().to_dict()}")

    texts, labels = df['text'].tolist(), df['label'].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=SEED, stratify=labels)

    # model setup
    base = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(base)
    model = DistilBertForSequenceClassification.from_pretrained(base, num_labels=2).to(dev)

    train_ds = SpeechDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_ds = SpeechDataset(val_texts, val_labels, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_FRACTION),
        num_training_steps=total_steps)

    print(f"\nTraining on {len(train_texts)} excerpts for {args.epochs} epochs")
    print(f"Validation: {len(val_texts)} excerpts")

    best_f1 = 0
    no_improve = 0
    patience = 3

    for ep in range(args.epochs):
        tr_loss, tr_acc, tr_f1 = run_train_epoch(model, train_loader, optimizer, scheduler, dev)
        v_loss, v_acc, v_f1 = run_eval(model, val_loader, dev)

        print(f"\nEpoch {ep+1}/{args.epochs}")
        print(f"  Train  Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  F1: {tr_f1:.4f}")
        print(f"  Val    Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  F1: {v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            no_improve = 0
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> Saved best model (val F1: {best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep+1}")
                break

    print(f"\nPre-training complete. Best validation F1: {best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"\nProceed to Stage 2:")
    print(f"  python train.py --data_path your_ads.csv --pretrained_dir {args.output_dir}")
