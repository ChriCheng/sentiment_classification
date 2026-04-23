import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .bert_dataset import BertSentimentDataset, detokenize, parse_sentence_field
from .dataset import LABEL_NAMES
from .evaluate import (
    build_confusion_matrix,
    format_confusion_matrix,
    print_class_accuracy,
    print_classification_report,
    print_prediction_distribution,
)


def print_error_examples(dataset, labels, preds, confidences, label_names, max_examples):
    if max_examples <= 0:
        return

    error_indices = [idx for idx, (label, pred) in enumerate(zip(labels, preds)) if label != pred]
    if not error_indices:
        print("Error Examples: none")
        return

    print(f"Error Examples: showing {min(max_examples, len(error_indices))}/{len(error_indices)}")
    for idx in error_indices[:max_examples]:
        row = dataset.df.iloc[idx]
        tokens = parse_sentence_field(row["sentences"])
        text = detokenize(tokens)
        if len(text) > 140:
            text = text[:137] + "..."

        label = labels[idx]
        pred = preds[idx]
        confidence = confidences[idx]
        print(
            f"  #{idx:<5} true={label}:{label_names[label]} "
            f"pred={pred}:{label_names[pred]} conf={confidence:.4f} | {text}"
        )


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_confidences = []
    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        probs = torch.softmax(outputs.logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        total_loss += outputs.loss.item() * labels.size(0)
        total_count += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())

    total_correct = sum(int(pred == label) for pred, label in zip(all_preds, all_labels))

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "labels": all_labels,
        "preds": all_preds,
        "confidences": all_confidences,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

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

    metrics = evaluate(model, loader, device)

    label_names = LABEL_NAMES[: model.config.num_labels]

    print("Evaluation Summary:")
    print(f"  device      : {device}")
    print(f"  model dir   : {args.model_dir}")
    print(f"  test set    : {args.test_path}")
    print(f"  test samples: {len(dataset)}")
    print(f"  test loss   : {metrics['loss']:.4f}")
    print(f"  test acc    : {metrics['acc']:.4f}")
    print()

    print_prediction_distribution(metrics["preds"], label_names)
    print()

    print_class_accuracy(metrics["labels"], metrics["preds"], label_names)
    print()

    print("Classification Report:")
    print_classification_report(metrics["labels"], metrics["preds"], label_names)
    print()

    print("Confusion Matrix:")
    matrix = build_confusion_matrix(metrics["labels"], metrics["preds"], model.config.num_labels)
    print(format_confusion_matrix(matrix, label_names))
    print()

    print_error_examples(
        dataset,
        metrics["labels"],
        metrics["preds"],
        metrics["confidences"],
        label_names,
        args.show_errors,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="checkpoints/bert_best")
    parser.add_argument("--test_path", type=str, default="preprocessed_file/test.csv")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--show_errors",
        type=int,
        default=0,
        help="Number of misclassified examples to print.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
