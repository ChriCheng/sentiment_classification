import ast
import re
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

LABEL_NAMES = [
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
]


def load_vocab(vocab_path: str) -> dict:
    """
    tokens2id.csv 视为两列: token, id
    为了给 PAD/UNK 预留位置，这里把原始 id 全部 +2
    """
    vocab_df = pd.read_csv(vocab_path, header=None, names=["token", "id"])

    token2id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for _, row in vocab_df.iterrows():
        token = str(row["token"])
        idx = int(row["id"]) + 2
        token2id[token] = idx

    return token2id


def parse_sentence_field(text) -> List[str]:
    """
    兼容两种格式：
    1. "['the', 'movie', 'is', 'good']"
    2. "the movie is good"
    """
    if isinstance(text, list):
        return [str(x) for x in text]

    if not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass

    return text.split()


def simple_tokenize(text: str) -> List[str]:
    """
    给 predict.py 用的简单英文分词器
    """
    text = text.lower().strip()
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)


class SentimentDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        token2id: dict,
        max_len: int = 48,
        sentence_col: str = "sentences",
        label_col: str = "label",
        has_label: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.token2id = token2id
        self.max_len = max_len
        self.sentence_col = sentence_col
        self.label_col = label_col
        self.has_label = has_label

        self.pad_id = token2id[PAD_TOKEN]
        self.unk_id = token2id[UNK_TOKEN]

        if self.sentence_col not in self.df.columns:
            raise ValueError(f"Column '{self.sentence_col}' not found in {csv_path}")

        if self.has_label and self.label_col not in self.df.columns:
            raise ValueError(f"Column '{self.label_col}' not found in {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def encode_tokens(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        ids = ids[: self.max_len]

        attention_mask = [1] * len(ids)

        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids += [self.pad_id] * pad_len
            attention_mask += [0] * pad_len

        return torch.tensor(ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tokens = parse_sentence_field(row[self.sentence_col])
        input_ids, attention_mask = self.encode_tokens(tokens)

        if self.has_label:
            label = int(row[self.label_col])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": torch.tensor(label, dtype=torch.long),
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }