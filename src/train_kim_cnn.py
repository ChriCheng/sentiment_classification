# src/train_kim_cnn.py
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LABEL_NAMES, SentimentDataset, load_vocab
from .model import TextCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_word2vec_matrix(
    word2vec_path: str,
    token2id: dict,
    embed_dim: int = 300,
    binary: bool = True,
):
    kv = KeyedVectors.load_word2vec_format(
        word2vec_path,
        binary=binary,
        unicode_errors="ignore",
    )

    matrix = np.random.uniform(-0.25, 0.25, (len(token2id), embed_dim)).astype(np.float32)
    matrix[token2id["<PAD>"]] = np.zeros(embed_dim, dtype=np.float32)

    matched = 0
    for token, idx in token2id.items():
        if token in ("<PAD>", "<UNK>"):
            continue

        candidates = [token, token.lower()]
        for cand in candidates:
            if cand in kv:
                matrix[idx] = kv[cand]
                matched += 1
                break

    coverage = matched / max(len(token2id) - 2, 1)
    print(f"word2vec matched tokens: {matched}/{max(len(token2id)-2,1)}")
    print(f"word2vec coverage     : {coverage:.4%}")
    return torch.tensor(matrix, dtype=torch.float32), coverage


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

    return total_loss / total_count, total_correct / total_count


def apply_max_norm(model, max_norm: float = 3.0):
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name and param.dim() > 1:
            param.data.renorm_(p=2, dim=0, maxnorm=max_norm)


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

    pretrained_matrix, coverage = load_word2vec_matrix(
        word2vec_path=args.word2vec_path,
        token2id=token2id,
        embed_dim=args.embed_dim,
        binary=args.word2vec_binary,
    )
    model.embedding.weight.data.copy_(pretrained_matrix)
    model.embedding.weight.requires_grad = True  # non-static

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=args.lr,
        rho=args.rho,
        weight_decay=args.weight_decay,
    )

    best_dev_acc = 0.0
    best_epoch = 0
    patience_count = 0
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
            apply_max_norm(model, args.max_norm)

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
            best_epoch = epoch
            patience_count = 0

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
                        "word2vec_coverage": coverage,
                        "model_variant": "kim_cnn_non_static",
                    },
                    "best_dev_acc": best_dev_acc,
                },
                args.save_path,
            )
            print(f"Best model saved to: {args.save_path}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch = {best_epoch}")
                break

    print(f"Training finished. Best dev acc = {best_dev_acc:.4f} at epoch {best_epoch}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/kim_sst")
    parser.add_argument("--save_path", type=str, default="checkpoints/kim_cnn_non_static.pt")

    parser.add_argument(
        "--word2vec_path",
        type=str,
        required=True,
        help="Path to GoogleNews-vectors-negative300.bin or another word2vec-format file",
    )
    parser.add_argument("--word2vec_binary", action="store_true")

    parser.add_argument("--max_len", type=int, default=56)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_norm", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=4)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())