import ast
from typing import List
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
def detokenize(tokens):
        text = " ".join(tokens)

        text = text.replace(" n't", "n't")
        text = text.replace(" 'm", "'m")
        text = text.replace(" 're", "'re")
        text = text.replace(" 's", "'s")
        text = text.replace(" 've", "'ve")
        text = text.replace(" 'd", "'d")
        text = text.replace(" 'll", "'ll")

        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ;", ";")
        text = text.replace(" :", ":")
        text = text.replace(" )", ")")
        text = text.replace("( ", "(")

        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

def parse_sentence_field(text) -> List[str]:
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


class BertSentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128, has_label=True):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_label = has_label

        if "sentences" not in self.df.columns:
            raise ValueError(f"'sentences' column not found in {csv_path}")

        if has_label and "label" not in self.df.columns:
            raise ValueError(f"'label' column not found in {csv_path}")

    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tokens = parse_sentence_field(row["sentences"])
        text = detokenize(tokens)

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.has_label:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)

        return item