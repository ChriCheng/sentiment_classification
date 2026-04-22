import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SentimentDataset
from .model import TextCNN


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
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/textcnn_best.pt")
    parser.add_argument("--test_path", type=str, default="preprocessed_file/test.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())