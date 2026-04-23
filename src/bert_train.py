import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .bert_dataset import BertSentimentDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
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

        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item() * labels.size(0)
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = BertSentimentDataset(
        csv_path=os.path.join(args.data_dir, "train.csv"),
        tokenizer=tokenizer,
        max_len=args.max_len,
        has_label=True,
    )
    dev_dataset = BertSentimentDataset(
        csv_path=os.path.join(args.data_dir, "dev.csv"),
        tokenizer=tokenizer,
        max_len=args.max_len,
        has_label=True,
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

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        progress_bar = tqdm(
    train_loader,
    desc=f"Epoch {epoch}/{args.epochs}",
    ascii=True,
    dynamic_ncols=True,
)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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

        dev_loss, dev_acc = evaluate(model, dev_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            torch.save(
                {
                    "best_dev_acc": best_dev_acc,
                    "model_name": args.model_name,
                    "max_len": args.max_len,
                },
                os.path.join(args.save_dir, "training_meta.pt"),
            )
            print(f"Best model saved to: {args.save_dir}")

    print(f"Training finished. Best dev acc = {best_dev_acc:.4f}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="preprocessed_file")
    parser.add_argument("--save_dir", type=str, default="checkpoints/bert_best")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")

    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
