import torch
import torch.nn as nn

import math
import einx

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        std = math.sqrt(2/(in_features+out_features))

        self.weight = nn.Parameter(
            torch.empty(
                out_features, 
                in_features,
                device=device,
                dtype=dtype
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            0,
            std,
            -3*std,
            3*std
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return einx.dot(
            "out_features in_features, ... in_features -> ... out_features",
            self.weight, x_in
        )


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings, 
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            0,
            1,
            -3,
            3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

