import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
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
import warnings
warnings.filterwarnings('ignore')


class Config:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    SEED = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15


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

    for batch in tqdm(dataloader, desc='Training'):
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
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
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
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'cohen_kappa': cohen_kappa_score(true_labels, predictions)
    }, predictions, true_labels


def main(args):
    print("=" * 60)
    print("NOSTALGIA CLASSIFIER - DistilBERT")
    print("=" * 60)

    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print(f"\nLoading: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Total: {len(df)} | Nostalgic: {df['Nostalgia_Binary'].sum()} | Non-nostalgic: {(df['Nostalgia_Binary']==0).sum()}")

    texts = df['Transcript'].tolist()
    labels = df['Nostalgia_Binary'].tolist()

    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=Config.TEST_SIZE, stratify=labels, random_state=Config.SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_SIZE/(1-Config.TEST_SIZE), stratify=y_temp, random_state=Config.SEED
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print(f"\nLoading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=2)
    model.to(device)

    train_dataset = PoliticalAdDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_dataset = PoliticalAdDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = PoliticalAdDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)

    batch_size = args.batch_size or Config.BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    epochs = args.epochs or Config.EPOCHS
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * Config.WARMUP_RATIO), num_training_steps=total_steps
    )

    print(f"\nTraining: {epochs} epochs, batch size {batch_size}")

    best_val_f1 = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            print(f"New best (F1: {best_val_f1:.4f})")

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    model.load_state_dict(best_model_state)
    test_metrics, test_preds, test_true = evaluate(model, test_loader, device)

    print(f"\nAccuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"Cohen's k: {test_metrics['cohen_kappa']:.4f}")

    cm = confusion_matrix(test_true, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"             Pred 0   Pred 1")
    print(f"Actual 0     {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"Actual 1     {cm[1,0]:4d}     {cm[1,1]:4d}")

    print(f"\n{classification_report(test_true, test_preds, target_names=['Non-Nostalgic', 'Nostalgic'])}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    pd.DataFrame([test_metrics]).to_csv(f'{output_dir}/metrics.csv', index=False)

    print(f"\nModel saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='models/nostalgia_classifier')

    args = parser.parse_args()
    main(args)
