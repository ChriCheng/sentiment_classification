import argparse
import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .bert_dataset import BertSentimentDataset


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    dataset = BertSentimentDataset(
        csv_path=args.test_path,
        tokenizer=tokenizer,
        max_len=args.max_len,
        has_label=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test acc: {acc:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="checkpoints/bert_best")
    parser.add_argument("--test_path", type=str, default="preprocessed_file/test.csv")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())