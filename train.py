"""
Stage 2: fine-tune on the political ad corpus.

Runs 5-fold stratified CV to get generalization estimates, then retrains
on the full dataset and saves the production model. Expects the pretrained
checkpoint from pretrain.py (Stage 1) but will fall back to base DistilBERT.
"""

import os, json, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
from tqdm import tqdm

# -- hyperparams --
BASE_MODEL = 'distilbert-base-uncased'
MAXLEN = 512
SEED = 42
N_FOLDS = 5
PATIENCE = 3
WARMUP_FRAC = 0.1


def seed_everything(s=SEED):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class AdDataset(Dataset):
    def __init__(self, texts, labels, tok):
        self.texts, self.labels, self.tok = texts, labels, tok

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), truncation=True, padding='max_length',
                       max_length=MAXLEN, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[i], dtype=torch.long),
        }


def _get_model(pretrained_dir, device):
    """Try loading the Stage 1 checkpoint; fall back to vanilla DistilBERT."""
    if pretrained_dir and os.path.exists(pretrained_dir):
        print(f"  Loading pre-trained model from: {pretrained_dir}")
        tok = DistilBertTokenizer.from_pretrained(pretrained_dir)
        mdl = DistilBertForSequenceClassification.from_pretrained(pretrained_dir)
    else:
        if pretrained_dir:
            print(f"  Warning: {pretrained_dir} not found, using base DistilBERT")
        tok = DistilBertTokenizer.from_pretrained(BASE_MODEL)
        mdl = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    return mdl.to(device), tok


def _train_one_epoch(model, loader, optim, sched, dev):
    model.train()
    total_loss = 0.0
    preds, truth = [], []
    for batch in tqdm(loader, desc='Training', leave=False):
        ids = batch['input_ids'].to(dev)
        mask = batch['attention_mask'].to(dev)
        labs = batch['label'].to(dev)

        optim.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labs)
        out.loss.backward()
        total_loss += out.loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        truth.extend(labs.cpu().numpy())

    return total_loss / len(loader), accuracy_score(truth, preds)


def _evaluate(model, loader, dev):
    model.eval()
    total_loss = 0.0
    preds, truth, probs_pos = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(dev)
            mask = batch['attention_mask'].to(dev)
            labs = batch['label'].to(dev)
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            total_loss += out.loss.item()

            p = torch.softmax(out.logits, dim=1)
            preds.extend(torch.argmax(p, dim=1).cpu().numpy())
            truth.extend(labs.cpu().numpy())
            probs_pos.extend(p[:, 1].cpu().numpy())

    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(truth, preds),
        'precision': precision_score(truth, preds, zero_division=0),
        'recall': recall_score(truth, preds, zero_division=0),
        'f1': f1_score(truth, preds, zero_division=0),
        'auc': roc_auc_score(truth, probs_pos) if len(set(truth)) > 1 else 0.0,
    }
    return metrics, preds, truth


def _train_with_early_stop(model, train_ldr, val_ldr, optim, sched, dev, n_epochs):
    """Train loop with early stopping on val F1. Returns best state dict + val metrics."""
    best_f1 = 0
    best_state = None
    stale = 0

    for _ in range(n_epochs):
        _train_one_epoch(model, train_ldr, optim, sched, dev)
        val_m, _, _ = _evaluate(model, val_ldr, dev)
        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    # reload best weights and get final metrics
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    return _evaluate(model, val_ldr, dev)


# ── cross-validation ────────────────────────────────────────────

def do_cv(texts, labels, tokenizer, dev, args):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    texts_arr, labels_arr = np.array(texts), np.array(labels)
    bs = args.batch_size
    ep = args.epochs
    fold_results = []

    print(f"\n{'='*60}")
    print(f"{N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---")
        tr_t, tr_l = texts_arr[tr_idx].tolist(), labels_arr[tr_idx].tolist()
        va_t, va_l = texts_arr[va_idx].tolist(), labels_arr[va_idx].tolist()
        print(f"  Train: {len(tr_t)} | Val: {len(va_t)}")

        model, _ = _get_model(args.pretrained_dir, dev)
        tr_ldr = DataLoader(AdDataset(tr_t, tr_l, tokenizer), batch_size=bs, shuffle=True)
        va_ldr = DataLoader(AdDataset(va_t, va_l, tokenizer), batch_size=bs)

        n_steps = len(tr_ldr) * ep
        optim = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=int(n_steps * WARMUP_FRAC), num_training_steps=n_steps)

        val_m, _, _ = _train_with_early_stop(model, tr_ldr, va_ldr, optim, sched, dev, ep)
        fold_results.append(val_m)
        print(f"  F1: {val_m['f1']:.4f}  AUC: {val_m['auc']:.4f}  Acc: {val_m['accuracy']:.4f}")

    # summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    cv_summary = {}
    for key in ['f1', 'auc', 'accuracy', 'precision', 'recall']:
        vals = [r[key] for r in fold_results]
        mu, sd = np.mean(vals), np.std(vals)
        cv_summary[key] = {'mean': float(mu), 'std': float(sd)}
        print(f"  {key.upper():>10s}: {mu:.4f} (SD = {sd:.4f})")

    return cv_summary, fold_results


# ── final model ─────────────────────────────────────────────────

def train_final(texts, labels, tokenizer, dev, args):
    print(f"\n{'='*60}")
    print("FINAL MODEL: RETRAINING ON FULL DATASET")
    print(f"{'='*60}")
    print(f"Training on all {len(texts)} advertisements")

    model, _ = _get_model(args.pretrained_dir, dev)
    bs = args.batch_size
    ep = args.epochs

    ds = AdDataset(texts, labels, tokenizer)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    n_steps = len(loader) * ep
    optim = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(n_steps * WARMUP_FRAC), num_training_steps=n_steps)

    for e in range(ep):
        loss, acc = _train_one_epoch(model, loader, optim, sched, dev)
        print(f"  Epoch {e+1}/{ep}  Loss: {loss:.4f}  Acc: {acc:.4f}")

    # self-eval on full corpus
    eval_ldr = DataLoader(ds, batch_size=bs)
    metrics, preds, truth = _evaluate(model, eval_ldr, dev)

    cm = confusion_matrix(truth, preds)
    print(f"\nFull-corpus confusion matrix (production model):")
    print(f"                Pred Non-Nost  Pred Nost")
    print(f"  Actual Non-Nost    {cm[0,0]:4d}         {cm[0,1]:4d}")
    print(f"  Actual Nost        {cm[1,0]:4d}         {cm[1,1]:4d}")
    print(f"\n{classification_report(truth, preds, target_names=['Non-Nostalgic', 'Nostalgic'])}")

    return model, tokenizer, metrics


# ── main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: Fine-tune on political ad corpus')
    parser.add_argument('--data_path', required=True,
                        help='CSV with Transcript and Nostalgia_Binary columns')
    parser.add_argument('--pretrained_dir', default=None,
                        help='Stage 1 checkpoint directory (from pretrain.py)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', default='models/nostalgia_classifier')
    args = parser.parse_args()

    print("=" * 60)
    print("NOSTALGIA CLASSIFIER - DistilBERT")
    print("=" * 60)

    seed_everything()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    df = pd.read_csv(args.data_path)
    print(f"\nLoading: {args.data_path}")
    print(f"Total: {len(df)} | Nostalgic: {df['Nostalgia_Binary'].sum()} | "
          f"Non-nostalgic: {(df['Nostalgia_Binary']==0).sum()}")

    texts = df['Transcript'].tolist()
    labels = df['Nostalgia_Binary'].tolist()

    # tokenizer
    if args.pretrained_dir and os.path.exists(args.pretrained_dir):
        tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_dir)
        print(f"Using tokenizer from: {args.pretrained_dir}")
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)

    cv_summary, fold_metrics = do_cv(texts, labels, tokenizer, dev, args)
    model, tokenizer, final_metrics = train_final(texts, labels, tokenizer, dev, args)

    # save everything
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = {
        'cross_validation': {
            'n_folds': N_FOLDS,
            'summary': cv_summary,
            'fold_metrics': [{k: float(v) for k, v in fm.items()} for fm in fold_metrics],
        },
        'final_model': {k: float(v) for k, v in final_metrics.items()},
        'config': {
            'max_length': MAXLEN, 'batch_size': args.batch_size,
            'epochs': args.epochs, 'learning_rate': 2e-5,
            'pretrained_dir': args.pretrained_dir,
        },
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(args.output_dir, 'metrics.csv'), index=False)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Results saved to: {args.output_dir}/results.json")
