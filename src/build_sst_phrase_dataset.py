# tools/build_sst_phrase_dataset.py
import argparse
import csv
import os
import re
from collections import Counter, defaultdict
def normalize_quotes(text: str) -> str:
    text = text.replace("`` ", "\"")
    text = text.replace(" ''", "\"")
    text = text.replace("``", "\"")
    text = text.replace("''", "\"")
    return text

def normalize_token(tok: str) -> str:
    mapping = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    return mapping.get(tok, tok)


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


def sentiment_value_to_label(value: float) -> int:
    if value <= 0.2:
        return 0
    if value <= 0.4:
        return 1
    if value <= 0.6:
        return 2
    if value <= 0.8:
        return 3
    return 4


def load_dictionary(path: str):
    phrase_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            phrase, phrase_id = line.rsplit("|", 1)
            phrase_to_id[phrase] = int(phrase_id)
    return phrase_to_id


def load_sentiment_labels(path: str):
    id_to_value = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            phrase_id, value = line.split("|")
            id_to_value[int(phrase_id)] = float(value)
    return id_to_value


def load_sentence_splits(dataset_sentences_path: str, dataset_split_path: str):
    sent_id_to_text = {}
    with open(dataset_sentences_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sent_id, sentence = line.split("\t", 1)
            sent_id_to_text[int(sent_id)] = sentence

    sent_id_to_split = {}
    with open(dataset_split_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            sent_id, split_id = line.split(",")
            sent_id_to_split[int(sent_id)] = int(split_id)

    return sent_id_to_text, sent_id_to_split


def load_sostr(path: str):
    all_tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("|")
            all_tokens.append(tokens)
    return all_tokens


def load_stree(path: str):
    all_parents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parents = [int(x) for x in line.strip().split("|")]
            all_parents.append(parents)
    return all_parents


def lookup_label(raw_tokens, norm_tokens, phrase_to_id, id_to_value):
    candidates = []

    raw_phrase = " ".join(raw_tokens)
    norm_phrase = " ".join(norm_tokens)
    detok_raw = detokenize(raw_tokens)
    detok_norm = detokenize(norm_tokens)

    candidates.extend([
        raw_phrase,
        norm_phrase,
        detok_raw,
        detok_norm,
        normalize_quotes(raw_phrase),
        normalize_quotes(norm_phrase),
        normalize_quotes(detok_raw),
        normalize_quotes(detok_norm),
    ])

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand in phrase_to_id:
            phrase_id = phrase_to_id[cand]
            if phrase_id in id_to_value:
                return sentiment_value_to_label(id_to_value[phrase_id])

    return None


def build_tree_children(parents):
    children = defaultdict(list)
    root = None
    for child_idx, parent_idx in enumerate(parents, start=1):
        if parent_idx == 0:
            root = child_idx
        else:
            children[parent_idx].append(child_idx)
    if root is None:
        raise ValueError("Root not found in STree.")
    return children, root


def extract_node_tokens(tokens, parents):
    n = len(tokens)
    children, root = build_tree_children(parents)
    cache = {}

    def visit(node_idx):
        if node_idx in cache:
            return cache[node_idx]

        if node_idx <= n:
            raw_tokens = [tokens[node_idx - 1]]
            norm_tokens = [normalize_token(tokens[node_idx - 1])]
            first_leaf = node_idx
            cache[node_idx] = (first_leaf, raw_tokens, norm_tokens)
            return cache[node_idx]

        child_infos = [visit(child) for child in children[node_idx]]
        child_infos.sort(key=lambda x: x[0])

        raw_tokens = []
        norm_tokens = []
        first_leaf = child_infos[0][0]

        for _, child_raw, child_norm in child_infos:
            raw_tokens.extend(child_raw)
            norm_tokens.extend(child_norm)

        cache[node_idx] = (first_leaf, raw_tokens, norm_tokens)
        return cache[node_idx]

    for node_idx in range(1, len(parents) + 1):
        visit(node_idx)

    return cache, root


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["sentences", "label", "source", "tree_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_vocab(path, rows):
    counter = Counter()
    for row in rows:
        tokens = eval(row["sentences"])
        counter.update(tokens)

    vocab = sorted(counter.keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for idx, token in enumerate(vocab):
            writer.writerow([token, idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="stanfordSentimentTreebank")
    parser.add_argument("--out_dir", type=str, default="data/kim_sst")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    phrase_to_id = load_dictionary(os.path.join(raw_dir, "dictionary.txt"))
    id_to_value = load_sentiment_labels(os.path.join(raw_dir, "sentiment_labels.txt"))
    sent_id_to_text, sent_id_to_split = load_sentence_splits(
    os.path.join(raw_dir, "datasetSentences.txt"),
    os.path.join(raw_dir, "datasetSplit.txt"),
)
    all_tokens = load_sostr(os.path.join(raw_dir, "SOStr.txt"))
    all_parents = load_stree(os.path.join(raw_dir, "STree.txt"))

    if len(all_tokens) != len(all_parents):
        raise ValueError("SOStr.txt and STree.txt line count mismatch.")

    train_rows = []
    dev_rows = []
    test_rows = []

    for tree_id, (tokens, parents) in enumerate(zip(all_tokens, all_parents)):
        node_info, root = extract_node_tokens(tokens, parents)
        _, root_raw, root_norm = node_info[root]

        sentence_id = tree_id + 1
        split_id = sent_id_to_split.get(sentence_id)

        if split_id is None:
            raise ValueError(f"Cannot find split for sentence_id={sentence_id}")

        # 这两行只保留作调试信息，不再用于 split 匹配
        root_sentence = detokenize(root_norm)
        original_sentence = sent_id_to_text.get(sentence_id, "")

        if split_id == 1:
            # train: 用整个句子的所有子树（包括 root sentence）
            for node_idx in range(1, len(parents) + 1):
                _, raw_tokens, norm_tokens = node_info[node_idx]
                label = lookup_label(raw_tokens, norm_tokens, phrase_to_id, id_to_value)
                if label is None:
                    continue

                train_rows.append(
                    {
                        "sentences": repr(norm_tokens),
                        "label": label,
                        "source": "sentence" if node_idx == root else "phrase",
                        "tree_id": tree_id,
                    }
                )

        elif split_id in (2, 3):
            # dev/test: 只保留整句
            label = lookup_label(root_raw, root_norm, phrase_to_id, id_to_value)
            if label is None:
                continue

            row = {
                "sentences": repr(root_norm),
                "label": label,
                "source": "sentence",
                "tree_id": tree_id,
            }

            if split_id == 3:
                dev_rows.append(row)
            else:
                test_rows.append(row)
        else:
            raise ValueError(f"Unexpected split id: {split_id}")

    save_rows(os.path.join(out_dir, "train.csv"), train_rows)
    save_rows(os.path.join(out_dir, "dev.csv"), dev_rows)
    save_rows(os.path.join(out_dir, "test.csv"), test_rows)
    save_vocab(os.path.join(out_dir, "tokens2id.csv"), train_rows)

    print("Build finished.")
    print(f"  raw dir           : {raw_dir}")
    print(f"  output dir        : {out_dir}")
    print(f"  train rows        : {len(train_rows)}")
    print(f"  dev sentences     : {len(dev_rows)}")
    print(f"  test sentences    : {len(test_rows)}")
    print(f"  train sentences   : {sum(1 for r in train_rows if r['source'] == 'sentence')}")
    print(f"  train phrases     : {sum(1 for r in train_rows if r['source'] == 'phrase')}")


if __name__ == "__main__":
    main()