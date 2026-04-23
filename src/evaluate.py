import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import LABEL_NAMES, SentimentDataset, parse_sentence_field
from .model import TextCNN


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_labels = []
    all_preds = []
    all_confidences = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "labels": all_labels,
        "preds": all_preds,
        "confidences": all_confidences,
    }


def format_confusion_matrix(matrix, label_names):
    headers = ["true\\pred"] + [str(i) for i in range(len(label_names))]
    rows = []

    for idx, row in enumerate(matrix):
        rows.append([f"{idx}:{label_names[idx]}", *[str(value) for value in row]])

    col_widths = [
        max(len(str(row[col_idx])) for row in [headers, *rows])
        for col_idx in range(len(headers))
    ]

    lines = [
        "  ".join(str(value).ljust(col_widths[idx]) for idx, value in enumerate(headers)),
        "  ".join("-" * width for width in col_widths),
    ]
    lines.extend(
        "  ".join(str(value).ljust(col_widths[idx]) for idx, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def build_confusion_matrix(labels, preds, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, pred in zip(labels, preds):
        matrix[label][pred] += 1
    return matrix


def print_classification_report(labels, preds, label_names):
    rows = []
    total = len(labels)
    total_correct = 0

    for class_idx, label_name in enumerate(label_names):
        tp = sum(1 for label, pred in zip(labels, preds) if label == class_idx and pred == class_idx)
        fp = sum(1 for label, pred in zip(labels, preds) if label != class_idx and pred == class_idx)
        fn = sum(1 for label, pred in zip(labels, preds) if label == class_idx and pred != class_idx)
        support = sum(1 for label in labels if label == class_idx)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((label_name, precision, recall, f1, support))
        total_correct += tp

    macro_precision = sum(row[1] for row in rows) / len(rows)
    macro_recall = sum(row[2] for row in rows) / len(rows)
    macro_f1 = sum(row[3] for row in rows) / len(rows)

    weighted_precision = sum(row[1] * row[4] for row in rows) / total if total else 0.0
    weighted_recall = sum(row[2] * row[4] for row in rows) / total if total else 0.0
    weighted_f1 = sum(row[3] * row[4] for row in rows) / total if total else 0.0
    accuracy = total_correct / total if total else 0.0

    name_width = max(len("label"), *(len(row[0]) for row in rows), len("weighted avg"))
    print(f"{'label'.ljust(name_width)}  precision  recall  f1-score  support")
    print(f"{'-' * name_width}  ---------  ------  --------  -------")
    for label_name, precision, recall, f1, support in rows:
        print(f"{label_name.ljust(name_width)}  {precision:>9.4f}  {recall:>6.4f}  {f1:>8.4f}  {support:>7}")
    print()
    print(f"{'accuracy'.ljust(name_width)}  {'':>9}  {'':>6}  {accuracy:>8.4f}  {total:>7}")
    print(f"{'macro avg'.ljust(name_width)}  {macro_precision:>9.4f}  {macro_recall:>6.4f}  {macro_f1:>8.4f}  {total:>7}")
    print(
        f"{'weighted avg'.ljust(name_width)}  "
        f"{weighted_precision:>9.4f}  {weighted_recall:>6.4f}  {weighted_f1:>8.4f}  {total:>7}"
    )


def print_class_accuracy(labels, preds, label_names):
    print("Per-class Accuracy:")
    for class_idx, label_name in enumerate(label_names):
        total = sum(1 for label in labels if label == class_idx)
        correct = sum(1 for label, pred in zip(labels, preds) if label == class_idx and pred == class_idx)
        acc = correct / total if total else 0.0
        print(f"  {class_idx}:{label_name:<14} {acc:.4f} ({correct}/{total})")


def print_prediction_distribution(preds, label_names):
    print("Prediction Distribution:")
    total = len(preds)
    for class_idx, label_name in enumerate(label_names):
        count = sum(1 for pred in preds if pred == class_idx)
        ratio = count / total if total else 0.0
        print(f"  {class_idx}:{label_name:<14} {count:>5} ({ratio:.2%})")


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
        tokens = parse_sentence_field(row[dataset.sentence_col])
        text = " ".join(tokens)
        if len(text) > 140:
            text = text[:137] + "..."

        label = labels[idx]
        pred = preds[idx]
        confidence = confidences[idx]
        print(
            f"  #{idx:<5} true={label}:{label_names[label]} "
            f"pred={pred}:{label_names[pred]} conf={confidence:.4f} | {text}"
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    config = checkpoint["config"]
    token2id = checkpoint["token2id"]

    model = TextCNN(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_classes=config["num_classes"],
        num_filters=config["num_filters"],
        kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"],
        pad_idx=config["pad_idx"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = SentimentDataset(
        csv_path=args.test_path,
        token2id=token2id,
        max_len=config["max_len"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)

    label_names = checkpoint.get("label_names", LABEL_NAMES)
    num_classes = config["num_classes"]
    label_names = label_names[:num_classes]

    print("Evaluation Summary:")
    print(f"  device      : {device}")
    print(f"  checkpoint  : {args.ckpt_path}")
    print(f"  test set    : {args.test_path}")
    print(f"  test samples: {len(test_dataset)}")
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
    matrix = build_confusion_matrix(metrics["labels"], metrics["preds"], num_classes)
    print(format_confusion_matrix(matrix, label_names))
    print()

    print_error_examples(
        test_dataset,
        metrics["labels"],
        metrics["preds"],
        metrics["confidences"],
        label_names,
        args.show_errors,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/textcnn_best.pt")
    parser.add_argument("--test_path", type=str, default="preprocessed_file/test.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--show_errors",
        type=int,
        default=0,
        help="Number of misclassified examples to print.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
