import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.model.n_embd % cfg.model.n_head == 0
        # key, query, value projections for all heads
        self.cfg = cfg
        self.key = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.query = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.value = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(cfg.model.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.model.resid_pdrop)

        # output projection
        self.proj = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.n_head = cfg.model.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.cfg.model.attn_pdrop if self.training else 0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.model.n_embd)
        self.ln2 = nn.LayerNorm(cfg.model.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.model.n_embd, 4 * cfg.model.n_embd),
            GELU(),
            nn.Linear(4 * cfg.model.n_embd, cfg.model.n_embd),
            nn.Dropout(cfg.model.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
