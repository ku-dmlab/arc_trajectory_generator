import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.model.n_embd)
        self.ln2 = nn.LayerNorm(cfg.model.n_embd)
        self.resid_drop = nn.Dropout(cfg.model.resid_pdrop)
        self.attn = nn.MultiheadAttention(cfg.model.n_embd, cfg.model.n_head, cfg.model.attn_pdrop, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.model.n_embd, 4 * cfg.model.n_embd),
            GELU(),
            nn.Linear(4 * cfg.model.n_embd, cfg.model.n_embd),
            nn.Dropout(cfg.model.resid_pdrop),
        )

    def forward(self, inp):
        x, mask = inp
        lnx = self.ln1(x)
        oup, _ = self.attn(lnx, lnx, lnx, key_padding_mask=mask)
        x = x + self.resid_drop(oup)
        x = x + self.mlp(self.ln2(x))
        return x, mask

class GPT_DQNCritic(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, cfg, env):
        super().__init__()
        self.cfg = cfg

        self.num_pixel = cfg.env.grid_x * cfg.env.grid_y
        self.num_tokens = self.num_pixel + 1

        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_pixel, cfg.model.n_embd))
        self.state_emb = nn.Parameter(torch.zeros(8, 1, cfg.model.n_embd))
        self.bbox_emb = nn.Parameter(torch.zeros(4, cfg.model.n_embd))
        self.drop = nn.Dropout(cfg.model.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.model.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(cfg.model.n_embd)
        self.head_bbox_action_mean = nn.Linear(cfg.model.n_embd, 35 * 4)
        self.head_bbox_action_std = nn.Linear(cfg.model.n_embd, 35 * 4)
        self.head_operation = nn.Linear(cfg.model.n_embd, 35)
        self.head_critic = nn.Linear(cfg.model.n_embd, 35)

        self.color_encoder = nn.Embedding(cfg.env.num_colors, cfg.model.n_embd)
        self.binary_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.term_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.trials_encoder = nn.Embedding(4, cfg.model.n_embd)
        self.active_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.rotation_encoder = nn.Embedding(4, cfg.model.n_embd)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
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

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding, torch.nn.MultiheadAttention)
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

        decay.add('pos_emb')
        decay.add('state_emb')
        decay.add('bbox_emb')

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

    def forward(self, grid, grid_dim, selected, clip, clip_dim, terminated,
                trials_remain, active, object, object_sel, object_dim, object_pos,
                background, rotation_parity, input, input_dim, answer, answer_dim,
                bbox=None):

        B = grid.shape[0]

        def compute_mask(base, end_dim, start_dim=None):
            active = torch.ones_like(base)
            if not (end_dim <=5).all() or not (end_dim >= 0).all():
                print(end_dim)
            for i in range(B):
                if start_dim is None:
                    active[i, :end_dim[i, 0], :end_dim[i, 1]] = 0
                else:
                    active[i, 
                    start_dim[i, 0]:start_dim[i, 0] + end_dim[i, 0],
                    start_dim[i, 1]:start_dim[i, 1] + end_dim[i, 1]] = 0
            return active.reshape(B, -1)
    
        def translate(base, pos):
            ret = torch.zeros_like(base)
            for i in range(B):
                target_x = slice(None) if pos[i, 0] == 0 else slice(None, -pos[i, 0])
                target_y = slice(None) if pos[i, 1] == 0 else slice(None, -pos[i, 1])
                ret[i, pos[i, 0]:, pos[i, 1]:] = base[i, target_x, target_y]
            return ret
        active_grid = compute_mask(grid, grid_dim)

        grid = self.color_encoder(grid.reshape(B, -1))
        grid = grid + self.pos_emb + self.state_emb[0]

        selected = self.binary_encoder(selected.reshape(B, -1)) # follows active_grid
        selected = selected + self.pos_emb + self.state_emb[1]

        active_clip = compute_mask(clip, clip_dim)

        clip = self.color_encoder(clip.reshape(B, -1))
        clip = clip + self.pos_emb + self.state_emb[2]

        active_obj = compute_mask(object, object_dim, object_pos)

        object = self.color_encoder(translate(object, object_pos).reshape(B, -1))
        object = object + self.pos_emb + self.state_emb[3]

        object_sel = self.binary_encoder(translate(object_sel, object_pos).reshape(B, -1)) # follows active_obj
        object_sel = object_sel + self.pos_emb + self.state_emb[4]

        background = self.color_encoder(background.reshape(B, -1)) # follows active_grid
        background = background + self.pos_emb + self.state_emb[5]

        active_inp = compute_mask(input, input_dim)

        input = self.color_encoder(input.reshape(B, -1))
        input = input + self.pos_emb + self.state_emb[6]

        active_ans = compute_mask(answer, answer_dim)

        answer = self.color_encoder(answer.reshape(B, -1))
        answer = answer + self.pos_emb + self.state_emb[7]

        cls_tkn = (
            self.term_encoder(terminated) + 
            self.trials_encoder(trials_remain) +
            self.active_encoder(active) + 
            self.rotation_encoder(rotation_parity)
            ).reshape(-1, 1, self.cfg.model.n_embd)
        
        inputs = torch.cat([
            grid, selected, clip, object, 
            object_sel, background, input, answer, cls_tkn], axis=1)

        if bbox is not None:
            inputs = inputs + (bbox @ self.bbox_emb)[:, None] # when both state, bbox input are given

        masks = torch.cat([
            active_grid, active_grid, active_clip, active_obj,
            active_obj, active_grid, active_inp, active_ans, torch.zeros_like(active_grid[:, :1])], axis=1)

        x = self.drop(inputs)
        x, _ = self.blocks((x, masks.bool()))
        x = self.ln_f(x)

        return x

    def act(self, deterministic=False, **kwargs):
        x = self.forward(**kwargs)[:, -1]
        operation_logits = self.head_operation(x)
        if deterministic:
            operation = torch.argmax(operation_logits, axis=1)
        else:
            operation = Categorical(logits=operation_logits).sample()
        
        bbox_action_mean = self.head_bbox_action_mean(x).reshape(-1, 35, 4)
        bbox_action_mean = torch.gather(bbox_action_mean, 1, operation[:, None, None].tile((1, 1, 4))).squeeze(1)

        if deterministic:
            return operation, bbox_action_mean

        bbox_action_std = self.head_bbox_action_std(x).reshape(-1, 35, 4)
        bbox_action_std = torch.gather(bbox_action_std, 1, operation[:, None, None].tile((1, 1, 4))).squeeze(1)
        bbox_action_std = torch.exp(torch.clip(bbox_action_std, -20, 2))
        bbox_action = Normal(bbox_action_mean, bbox_action_std).sample()
        return operation, bbox_action
    
    def action_and_log_pi(self, **kwargs):
        x = self.forward(**kwargs)[:, -1]
        operation_logits = self.head_operation(x)
        operation_dist = Categorical(logits=operation_logits)
        operation = operation_dist.sample()
        op_log_pi = operation_dist.log_prob(operation)
        
        bbox_action_mean = self.head_bbox_action_mean(x).reshape(-1, 35, 4)
        bbox_action_mean = torch.gather(bbox_action_mean, 1, operation[:, None, None].tile((1, 1, 4))).squeeze(1)

        bbox_action_std = self.head_bbox_action_std(x).reshape(-1, 35, 4)
        bbox_action_std = torch.gather(bbox_action_std, 1, operation[:, None, None].tile((1, 1, 4))).squeeze(1)
        bbox_action_std = torch.exp(torch.clip(bbox_action_std, -20, 2))
        bbox_action_dist = Normal(bbox_action_mean, bbox_action_std)
        bbox_action = bbox_action_dist.sample()

        bbox_log_pi = bbox_action_dist.log_prob(bbox_action).sum(1)

        return operation, bbox_action, op_log_pi, bbox_log_pi, nn.functional.softmax(operation_logits, -1)
    
    def value(self, operation, bbox, pi=None, **kwargs):
        x = self.forward(**kwargs, bbox=bbox)[:, -1]
        all_qvalues = self.head_critic(x)
        qvalue = torch.gather(all_qvalues, 1, operation[:, None])
        if pi is not None:
            value = (pi * all_qvalues).sum(1)
            return qvalue.squeeze(1), value
        return qvalue.squeeze(1)
    
