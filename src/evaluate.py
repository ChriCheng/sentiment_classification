import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SentimentDataset
from .model import TextCNN


def build_model_from_checkpoint(checkpoint):
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    embedding_key = next(
        key for key in ("embedding.weight", "embedding_trainable.weight") if key in state_dict
    )
    embedding_name = embedding_key.rsplit(".", 1)[0]

    conv_weight_keys = sorted(
        (key for key in state_dict if key.startswith("convs.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not conv_weight_keys:
        raise ValueError("Checkpoint does not contain TextCNN convolution weights")

    conv_shapes = [state_dict[key].shape for key in conv_weight_keys]
    num_filters = conv_shapes[0][0]
    kernel_sizes = [shape[2] for shape in conv_shapes]
    embed_dim = state_dict[embedding_key].shape[1]
    vocab_size = state_dict[embedding_key].shape[0]
    num_classes = state_dict["fc.bias"].shape[0]

    use_batch_norm = any(key.startswith("batch_norms.") for key in state_dict)
    conv_feature_dim = num_filters * len(kernel_sizes)
    fc_in_features = state_dict["fc.weight"].shape[1]
    if fc_in_features == conv_feature_dim:
        pooling = "max"
    elif fc_in_features == conv_feature_dim * 2:
        pooling = "max_avg"
    else:
        raise ValueError(
            "Cannot infer TextCNN pooling from checkpoint: "
            f"fc_in_features={fc_in_features}, conv_feature_dim={conv_feature_dim}"
        )

    return TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=config.get("dropout", 0.5),
        pad_idx=config.get("pad_idx", 0),
        embedding_name=embedding_name,
        use_batch_norm=use_batch_norm,
        pooling=pooling,
    )


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

    model = build_model_from_checkpoint(checkpoint).to(device)
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
