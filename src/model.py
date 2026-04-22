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
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != self.embedding.weight.data.shape:
                raise ValueError(
                    f"Shape mismatch: pretrained_embeddings={pretrained_embeddings.shape}, "
                    f"embedding_weight={self.embedding.weight.data.shape}"
                )
            self.embedding.weight.data.copy_(pretrained_embeddings)

        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def conv_and_pool(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        x = conv(x)                 # [B, F, L-k+1, 1]
        x = F.relu(x)
        x = x.squeeze(3)            # [B, F, L-k+1]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [B, F]
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)   # [B, L, E]
        x = x.unsqueeze(1)              # [B, 1, L, E]

        conv_outputs = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)

        x = self.dropout(x)
        logits = self.fc(x)
        return logits