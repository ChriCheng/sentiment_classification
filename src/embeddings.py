import numpy as np
import torch


def load_glove_embeddings(glove_path: str, token2id: dict, embed_dim: int):
    """
    根据 token2id 构建 embedding matrix:
    - 命中 GloVe 的词：使用预训练向量
    - 未命中的词：随机初始化
    - PAD: 全 0
    """
    vocab_size = len(token2id)

    embedding_matrix = np.random.normal(
        loc=0.0, scale=0.05, size=(vocab_size, embed_dim)
    ).astype(np.float32)

    pad_id = token2id["<PAD>"]
    embedding_matrix[pad_id] = np.zeros(embed_dim, dtype=np.float32)

    found = 0
    matched_ids = set()

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != embed_dim + 1:
                continue

            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)

            # 先精确匹配，再 lower 匹配
            if word in token2id:
                idx = token2id[word]
                if idx not in matched_ids:
                    embedding_matrix[idx] = vector
                    matched_ids.add(idx)
                    found += 1
            else:
                lw = word.lower()
                if lw in token2id:
                    idx = token2id[lw]
                    if idx not in matched_ids:
                        embedding_matrix[idx] = vector
                        matched_ids.add(idx)
                        found += 1

    total_real_tokens = max(vocab_size - 2, 1)  # 去掉 PAD/UNK
    coverage = found / total_real_tokens

    print(f"GloVe matched tokens: {found}/{total_real_tokens}")
    print(f"GloVe coverage: {coverage:.4%}")

    return torch.tensor(embedding_matrix, dtype=torch.float), coverage