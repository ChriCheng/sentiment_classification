from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embedding: bool = False,
        embedding_name: str = "embedding",
        use_batch_norm: bool = False,
        pooling: str = "max",
    ):
        super().__init__()

        if embedding_name not in {"embedding", "embedding_trainable"}:
            raise ValueError(f"Unsupported embedding_name: {embedding_name}")
        if pooling not in {"max", "max_avg"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.embedding_name = embedding_name
        self.pooling = pooling

        embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        setattr(self, embedding_name, embedding)

        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != embedding.weight.data.shape:
                raise ValueError(
                    f"Shape mismatch: pretrained_embeddings={pretrained_embeddings.shape}, "
                    f"embedding_weight={embedding.weight.data.shape}"
                )
            embedding.weight.data.copy_(pretrained_embeddings)

        if freeze_embedding:
            embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes]
        )
        self.batch_norms = (
            nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
            if use_batch_norm
            else None
        )

        self.dropout = nn.Dropout(dropout)
        pooling_multiplier = 2 if pooling == "max_avg" else 1
        self.fc = nn.Linear(num_filters * len(kernel_sizes) * pooling_multiplier, num_classes)

    def conv_and_pool(
        self,
        x: torch.Tensor,
        conv: nn.Module,
        batch_norm: nn.Module = None,
    ) -> torch.Tensor:
        x = conv(x)                 # [B, F, L-k+1, 1]
        x = F.relu(x)
        x = x.squeeze(3)            # [B, F, L-k+1]
        if batch_norm is not None:
            x = batch_norm(x)

        max_pooled = F.max_pool1d(x, x.size(2)).squeeze(2)  # [B, F]
        if self.pooling == "max_avg":
            avg_pooled = F.avg_pool1d(x, x.size(2)).squeeze(2)
            x = torch.cat([max_pooled, avg_pooled], dim=1)
        else:
            x = max_pooled

        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = getattr(self, self.embedding_name)
        x = embedding(input_ids)        # [B, L, E]
        x = x.unsqueeze(1)              # [B, 1, L, E]

        if self.batch_norms is None:
            conv_outputs = [self.conv_and_pool(x, conv) for conv in self.convs]
        else:
            conv_outputs = [
                self.conv_and_pool(x, conv, batch_norm)
                for conv, batch_norm in zip(self.convs, self.batch_norms)
            ]
        x = torch.cat(conv_outputs, dim=1)

        x = self.dropout(x)
        logits = self.fc(x)
        return logits
