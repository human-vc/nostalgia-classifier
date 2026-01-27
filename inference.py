import argparse
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np


def load_model(model_dir):
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_single(text, model, tokenizer, device, max_length=256):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        nostalgia_prob = probs[0, 1].item()

    return {
        'prediction': pred_class,
        'label': 'Nostalgic' if pred_class == 1 else 'Non-Nostalgic',
        'confidence': confidence,
        'nostalgia_probability': nostalgia_prob
    }


def predict_batch(texts, model, tokenizer, device, batch_size=32, max_length=256):
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

            pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
            nostalgia_probs = probs[:, 1].cpu().numpy()

        for j, (pred, prob) in enumerate(zip(pred_classes, nostalgia_probs)):
            results.append({
                'text': batch_texts[j][:100] + '...' if len(batch_texts[j]) > 100 else batch_texts[j],
                'prediction': int(pred),
                'label': 'Nostalgic' if pred == 1 else 'Non-Nostalgic',
                'nostalgia_probability': float(prob)
            })

    return results


def main(args):
    print("Loading model...")
    model, tokenizer, device = load_model(args.model_dir)
    print(f"Model loaded. Device: {device}")

    if args.text:
        result = predict_single(args.text, model, tokenizer, device)

        print(f"\n{'='*50}")
        print("PREDICTION")
        print(f"{'='*50}")
        print(f"Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
        print(f"\nLabel: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Nostalgia probability: {result['nostalgia_probability']:.2%}")

    elif args.csv_path:
        print(f"\nLoading CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)

        text_col = args.text_column or 'Transcript'
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Available: {df.columns.tolist()}")

        texts = df[text_col].tolist()
        print(f"Processing {len(texts)} texts...")

        results = predict_batch(texts, model, tokenizer, device)

        df['Nostalgia_Prediction'] = [r['prediction'] for r in results]
        df['Nostalgia_Label'] = [r['label'] for r in results]
        df['Nostalgia_Probability'] = [r['nostalgia_probability'] for r in results]

        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Total ads: {len(df)}")
        print(f"Nostalgic: {(df['Nostalgia_Prediction']==1).sum()} ({(df['Nostalgia_Prediction']==1).mean():.1%})")
        print(f"Non-nostalgic: {(df['Nostalgia_Prediction']==0).sum()} ({(df['Nostalgia_Prediction']==0).mean():.1%})")

        output_path = args.output_path or args.csv_path.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    else:
        print("Please provide --text or --csv_path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--text', type=str)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--text_column', type=str, default='Transcript')
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    main(args)
