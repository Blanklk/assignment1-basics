import torch
import torch.nn as nn

import math
from einops import rearrange, einsum
import einx


class CE(nn.Module):
    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logit: [batch_size, vocab_size]
        # logit = logit - torch.max(logit, dim=-1, keepdim=True).values
        log_sum_exp = torch.logsumexp(logit, dim=-1) # already have stable compute
        batch_idx = torch.arange(logit.shape[0], device=logit.device)
        res = log_sum_exp - logit[batch_idx, target]
        return torch.mean(res)
