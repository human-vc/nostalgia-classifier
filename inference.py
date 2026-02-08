"""
Inference script for the nostalgia classifier.
Supports single-text predictions and batch CSV processing.
"""

import argparse
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MAXLEN = 512
LABELS = {0: 'Non-Nostalgic', 1: 'Nostalgic'}


def load_model(model_dir):
    tok = DistilBertTokenizer.from_pretrained(model_dir)
    mdl = DistilBertForSequenceClassification.from_pretrained(model_dir)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl.to(dev).eval()
    return mdl, tok, dev


def classify_one(text, model, tok, dev):
    enc = tok(text, truncation=True, padding='max_length',
              max_length=MAXLEN, return_tensors='pt')
    with torch.no_grad():
        ids = enc['input_ids'].to(dev)
        mask = enc['attention_mask'].to(dev)
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=1)
        cls = torch.argmax(probs, dim=1).item()
    return {
        'prediction': cls,
        'label': LABELS[cls],
        'confidence': probs[0, cls].item(),
        'nostalgia_probability': probs[0, 1].item(),
    }


def classify_batch(texts, model, tok, dev, bs=32):
    out = []
    for start in range(0, len(texts), bs):
        chunk = texts[start:start + bs]
        enc = tok(chunk, truncation=True, padding='max_length',
                  max_length=MAXLEN, return_tensors='pt')
        with torch.no_grad():
            ids = enc['input_ids'].to(dev)
            mask = enc['attention_mask'].to(dev)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits, dim=1)
            classes = torch.argmax(probs, dim=1).cpu().numpy()
            nost_probs = probs[:, 1].cpu().numpy()

        for j, (c, p) in enumerate(zip(classes, nost_probs)):
            snippet = chunk[j][:100] + '...' if len(chunk[j]) > 100 else chunk[j]
            out.append({
                'text': snippet,
                'prediction': int(c),
                'label': LABELS[int(c)],
                'nostalgia_probability': float(p),
            })
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify political ads for nostalgic framing')
    parser.add_argument('--model_dir', required=True, help='Trained model directory')
    parser.add_argument('--text', help='Single text to classify')
    parser.add_argument('--csv_path', help='CSV file for batch classification')
    parser.add_argument('--text_column', default='Transcript',
                        help='Column containing ad transcripts (default: Transcript)')
    parser.add_argument('--output_path', help='Where to save predictions CSV')
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer, device = load_model(args.model_dir)
    print(f"Model loaded. Device: {device}")

    if args.text:
        res = classify_one(args.text, model, tokenizer, device)
        print(f"\n{'='*50}")
        print("PREDICTION")
        print(f"{'='*50}")
        preview = args.text[:100] + '...' if len(args.text) > 100 else args.text
        print(f"Text: {preview}")
        print(f"\nLabel:    {res['label']}")
        print(f"Confidence: {res['confidence']:.2%}")
        print(f"Nostalgia probability: {res['nostalgia_probability']:.2%}")

    elif args.csv_path:
        print(f"\nLoading CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        col = args.text_column
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

        texts = df[col].tolist()
        print(f"Processing {len(texts)} texts...")
        results = classify_batch(texts, model, tokenizer, device)

        df['Nostalgia_Prediction'] = [r['prediction'] for r in results]
        df['Nostalgia_Label'] = [r['label'] for r in results]
        df['Nostalgia_Probability'] = [r['nostalgia_probability'] for r in results]

        n_nost = (df['Nostalgia_Prediction'] == 1).sum()
        n_non = (df['Nostalgia_Prediction'] == 0).sum()
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Total ads:      {len(df)}")
        print(f"Nostalgic:      {n_nost} ({n_nost/len(df):.1%})")
        print(f"Non-nostalgic:  {n_non} ({n_non/len(df):.1%})")

        out_path = args.output_path or args.csv_path.replace('.csv', '_predictions.csv')
        df.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")
    else:
        print("Please provide --text or --csv_path")
