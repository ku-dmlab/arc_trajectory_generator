import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

from bclone.base import Block

logger = logging.getLogger(__name__)

class Policy(nn.Module):
    """(state, z) -> action, next_state (?)"""
    """State: spaces.Dict(
            {
                "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "grid_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),
                "selected": spaces.MultiBinary((self.H,self.W)),
                "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "clip_dim": spaces.Tuple((spaces.Discrete(self.H,start=0),spaces.Discrete(self.W,start=0))),
            }
        )
        Action: spaces.Dict(
            {
                "selection": spaces.MultiBinary((self.H,self.W)), # selection Mask
                "operation": spaces.Discrete(action_count)
            }
        )
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.H, self.W = cfg.env.grid_y, cfg.env.grid_x
        self.num_pixel = cfg.env.grid_x * cfg.env.grid_y
        self.num_tokens = self.num_pixel + 1
        self.loc2ind = np.arange(self.num_pixel).reshape(cfg.env.grid_y, cfg.env.grid_x)

        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_pixel, cfg.model.n_embd))

        self.goal_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.pred_emb = nn.Parameter(torch.zeros(1, 2, cfg.model.n_embd))
        self.grid_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.clip_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.valid_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.invalid_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.selected_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.selection_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.emb_list = [
            "goal_emb",
            "pred_emb",
            "pos_emb",
            "grid_emb", 
            "clip_emb", 
            "valid_emb", 
            "invalid_emb", 
            "selected_emb", 
            "selection_emb"]
        self.drop = nn.Dropout(cfg.model.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.model.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(cfg.model.n_embd)
        self.head_selected = nn.Linear(cfg.model.n_embd, 1)
        self.head_grid = nn.Linear(cfg.model.n_embd, cfg.env.num_colors)
        self.head_clip = nn.Linear(cfg.model.n_embd, cfg.env.num_colors)
        self.head_grid_dim = nn.Linear(cfg.model.n_embd, self.H + 1)
        self.head_clip_dim = nn.Linear(cfg.model.n_embd, self.H + 1) # currently assumes self.H = self.W !

        self.color_encoder = nn.Embedding(cfg.env.num_colors, cfg.model.n_embd)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                """
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                """
                if pn.endswith('bias'):
                    # 원래 모든 bias는 학습 안되는데 여기서는 학습 OK
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        for emb in self.emb_list:
            decay.add(emb)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.cfg.train.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.learning_rate, betas=(self.cfg.train.beta1, self.cfg.train.beta2))

        return optimizer

    def forward(self, selected, grid, grid_dim, clip, clip_dim, goal, goal_dim):
        """
        grid: B x H x W (int, 0 ~ num_colors)
        grid_dim: B x 2 (int, 1 ~ self.H/W)
        selected: B x H x W (int, 0/1)
        clip: B x H x W (int, 0 ~ num_colors)
        clip_dim: B x 2 (int, 0 ~ self.H/W)
        goal: B x H x W
        """
        B, H, W = grid.shape
        assert grid_dim.shape == (B, 2)
        assert clip.shape == (B, H, W)
        assert clip_dim.shape == (B, 2)
        assert selected.shape == (B, H, W)
        assert goal.shape == (B, H, W)
        assert goal_dim.shape == (B, 2)

        def gen_mask_from_dim(dim):
            mask = torch.zeros((B, H, W)).to(dim.device)
            for i, (h, w) in enumerate(dim):
                mask[i, :h, :w] = 1.0
            return mask.reshape(B, H * W, 1)

        grid = self.color_encoder(grid.reshape(B, H * W))
        grid_mask = gen_mask_from_dim(grid_dim)

        grid += self.pos_emb + self.grid_emb
        grid += grid_mask * self.valid_emb + (1 - grid_mask) * self.invalid_emb
        grid += selected.reshape(B, H * W, 1) * self.selected_emb

        clip = self.color_encoder(clip.reshape(B, H * W))
        clip_mask = gen_mask_from_dim(clip_dim)

        clip += self.pos_emb + self.clip_emb
        clip += clip_mask * self.valid_emb + (1 - clip_mask) * self.invalid_emb

        goal = self.color_encoder(goal.reshape(B, H * W))
        goal_mask = gen_mask_from_dim(goal_dim)
        
        goal += self.pos_emb + self.goal_emb
        goal += goal_mask * self.valid_emb + (1 - goal_mask) * self.invalid_emb

        input_tokens = torch.cat([grid, clip, torch.tile(self.pred_emb, [B, 1, 1]), goal], axis=1)
        x = self.drop(input_tokens)
        x = self.blocks(x)
        x = self.ln_f(x)

        ps = self.head_selected(x[:, :H * W]).squeeze(-1)
        pg = self.head_grid(x[:, :H * W])
        pgd = self.head_grid_dim(x[:, H * W * 2:H * W * 2 + 2])
        pc = self.head_grid(x[:, H * W:H * W * 2])
        pcd = self.head_clip_dim(x[:, H * W * 2:H * W * 2 + 2])
        return ps, pg, pgd, pc, pcd
