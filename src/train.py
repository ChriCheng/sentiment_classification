import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LABEL_NAMES, SentimentDataset, load_vocab
from .model import TextCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token2id = load_vocab(os.path.join(args.data_dir, "tokens2id.csv"))

    train_dataset = SentimentDataset(
        csv_path=os.path.join(args.data_dir, "train.csv"),
        token2id=token2id,
        max_len=args.max_len,
    )
    dev_dataset = SentimentDataset(
        csv_path=os.path.join(args.data_dir, "dev.csv"),
        token2id=token2id,
        max_len=args.max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = TextCNN(
        vocab_size=len(token2id),
        embed_dim=args.embed_dim,
        num_classes=5,
        num_filters=args.num_filters,
        kernel_sizes=args.kernel_sizes,
        dropout=args.dropout,
        pad_idx=0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dev_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total_count += batch_size

            progress_bar.set_postfix(
                train_loss=f"{total_loss / total_count:.4f}",
                train_acc=f"{total_correct / total_count:.4f}",
            )

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "token2id": token2id,
                    "label_names": LABEL_NAMES,
                    "config": {
                        "vocab_size": len(token2id),
                        "embed_dim": args.embed_dim,
                        "num_classes": 5,
                        "num_filters": args.num_filters,
                        "kernel_sizes": args.kernel_sizes,
                        "dropout": args.dropout,
                        "pad_idx": 0,
                        "max_len": args.max_len,
                    },
                    "best_dev_acc": best_dev_acc,
                },
                args.save_path,
            )
            print(f"Best model saved to: {args.save_path}")

    print(f"Training finished. Best dev acc = {best_dev_acc:.4f}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="preprocessed_file")
    parser.add_argument("--save_path", type=str, default="checkpoints/textcnn_best.pt")

    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)