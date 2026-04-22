import argparse

import torch

from .dataset import LABEL_NAMES, simple_tokenize
from .model import TextCNN


@torch.no_grad()
def predict_one(model, text, token2id, max_len, device):
    model.eval()

    pad_id = token2id["<PAD>"]
    unk_id = token2id["<UNK>"]

    tokens = simple_tokenize(text)
    ids = [token2id.get(tok, unk_id) for tok in tokens][:max_len]

    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))

    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    logits = model(input_ids)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_id = torch.argmax(probs).item()
    return pred_id, probs.cpu().tolist(), tokens


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

    pred_id, probs, tokens = predict_one(
        model=model,
        text=args.text,
        token2id=token2id,
        max_len=config["max_len"],
        device=device,
    )

    print("Tokens:", tokens)
    print("Predicted label:", pred_id, "-", LABEL_NAMES[pred_id])
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {i} ({LABEL_NAMES[i]}): {p:.4f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/textcnn_best.pt")
    parser.add_argument("--text", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())