import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from utils.truncated_normal import TruncatedNormal

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
        self.key = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.query = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.value = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(cfg.model.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.model.resid_pdrop)

        # output projection
        self.proj = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.n_head = cfg.model.n_head

    def forward(self, x, key_padding_mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att[key_padding_mask[:, None, :, None].tile(1, self.n_head, 1, T)] = float('-inf')
        att[key_padding_mask[:, None, None, :].tile(1, self.n_head, T, 1)] = float('-inf')

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
        self.resid_drop = nn.Dropout(cfg.model.resid_pdrop)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.model.n_embd, 4 * cfg.model.n_embd),
            GELU(),
            nn.Linear(4 * cfg.model.n_embd, cfg.model.n_embd),
            nn.Dropout(cfg.model.resid_pdrop),
        )

    def forward(self, inp):
        x, mask = inp
        x = x + self.attn(self.ln1(x), key_padding_mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x, mask

class GPT_DQNCritic(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, cfg, env):
        super().__init__()
        self.cfg = cfg

        self.num_pixel = cfg.env.grid_x * cfg.env.grid_y
        self.num_tokens = self.num_pixel + 1

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_pixel, cfg.model.n_embd) * 0.02)
        self.global_pos_emb = nn.Parameter(torch.randn(1, self.num_pixel * 8 + 2, cfg.model.n_embd) * 0.02)
        self.state_emb = nn.Parameter(torch.randn(8, 1, cfg.model.n_embd) * 0.02)
        self.bbox_emb = nn.Linear(4, cfg.model.n_embd)
        self.drop = nn.Dropout(cfg.model.embd_pdrop)
        self.cls_tkn = nn.Parameter(torch.randn(1, 1, cfg.model.n_embd) * 0.02)

        # transformer
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.model.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(cfg.model.n_embd)
        self.head_critic = nn.Linear(cfg.model.n_embd, cfg.env.num_actions)

        self.color_encoder = nn.Embedding(cfg.env.num_colors, cfg.model.n_embd)
        self.binary_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.term_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.trials_encoder = nn.Embedding(4, cfg.model.n_embd)
        self.active_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.rotation_encoder = nn.Embedding(4, cfg.model.n_embd)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # rnn policy
        self.head_operation = nn.Linear(cfg.model.n_embd, cfg.env.num_actions)
        self.policy = nn.GRUCell(cfg.model.n_embd, cfg.model.n_embd)
        self.operation_encoder_hidden = nn.Linear(cfg.model.n_embd, cfg.model.n_embd * cfg.env.num_actions)
        self.operation_encoder_input = nn.Embedding(cfg.env.num_actions, cfg.model.n_embd)
        self.bbox_emb_weight = nn.Parameter(torch.randn(4, 1, cfg.model.n_embd) * 0.02)
        self.bbox_emb_bias = nn.Parameter(torch.randn(4, 1, cfg.model.n_embd) * 0.02)
        self.bbox_head = nn.Linear(cfg.model.n_embd, 2)
        self.head_aux = nn.Linear(cfg.model.n_embd, 1)

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

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, nn.GRUCell)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                """
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                """
                if 'bias' in pn:
                    # 원래 모든 bias는 학습 안되는데 여기서는 학습 OK
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif 'weight' in pn and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        no_decay.add('state_emb')
        no_decay.add('cls_tkn')
        no_decay.add('bbox_emb_weight')
        no_decay.add('bbox_emb_bias')

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
        active_ans = compute_mask(answer, answer_dim)

        dist =  (((grid == answer) ** 2).reshape(B, -1) * (1 - active_ans)).float().mean(dim=1, keepdims=True)

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

        answer = self.color_encoder(answer.reshape(B, -1))
        answer = answer + self.pos_emb + self.state_emb[7]

        info_tkn = (
            self.term_encoder(terminated) + 
            self.trials_encoder(trials_remain) +
            self.active_encoder(active) + 
            self.rotation_encoder(rotation_parity)
            ).reshape(-1, 1, self.cfg.model.n_embd)

        cls_tkn = self.cls_tkn.tile(len(active_grid), 1, 1)
        if bbox is not None:
            cls_tkn += self.bbox_emb(bbox)[:, None]

        inputs = torch.cat([
            grid, selected, clip, object, 
            object_sel, background, input, answer, info_tkn, cls_tkn], axis=1) + self.global_pos_emb

        masks = torch.cat([
            active_grid, active_grid, active_clip, active_obj,
            active_obj, active_grid + 1 - active.reshape(-1, 1), active_inp, active_ans, 
            torch.zeros((len(active_grid), 2), device=active_grid.device)], axis=1)

        x = self.drop(inputs)
        x, _ = self.blocks((x, masks.bool()))
        x = self.ln_f(x)

        return x, dist

    def _act(self, deterministic=False, return_log_pi=False, **kwargs):
        x, _ = self.forward(**kwargs)
        x = x[:, -1]
        operation_logits = self.head_operation(x)
        if deterministic:
            operation = torch.argmax(operation_logits, axis=1)
        else:
            operation_dist = Categorical(logits=operation_logits)
            operation = operation_dist.sample()
            op_log_pi = - operation_dist.entropy()
        hx = self.operation_encoder_hidden(x).reshape(-1, self.cfg.env.num_actions, self.cfg.model.n_embd)
        hx = torch.gather(hx, 1, operation[:, None, None].tile((1, 1, self.cfg.model.n_embd))).squeeze(1)
        inp = self.operation_encoder_input(operation)
        output = []
        bbox_log_pi = 0
        for i in range(4):
            hx = self.policy(inp, hx)
            oup_i = self.bbox_head(hx)
            mean = torch.nn.functional.sigmoid(oup_i[:, 0]) * self.cfg.env.grid_x
            if deterministic:
                oup_i = mean
            else:
                action_dist = TruncatedNormal(
                    mean,
                    torch.nn.functional.softplus(oup_i[:, 1]),
                    torch.zeros_like(oup_i[:, 0]),
                    torch.ones_like(oup_i[:, 0]) * self.cfg.env.grid_x
                )
                oup_i = action_dist.sample()
                bbox_log_pi = action_dist.log_prob(oup_i) + bbox_log_pi
            oup_i = oup_i[:, None]
            output.append(oup_i)
            inp = self.bbox_emb_weight[i] * oup_i + self.bbox_emb_bias[i]
        bbox_action = torch.concatenate(output, axis=1)

        if not return_log_pi:
            return operation, bbox_action
        else:
            return operation, bbox_action, op_log_pi, bbox_log_pi, nn.functional.softmax(operation_logits, -1)

    def act(self, deterministic=False, **kwargs):
        return self._act(deterministic=deterministic, **kwargs)

    def action_and_log_pi(self, **kwargs):
        return self._act(deterministic=False, return_log_pi=True, **kwargs)

    def value(self, operation, bbox, pi=None, **kwargs):
        x, dist = self.forward(**kwargs, bbox=bbox)
        x = x[:, -1]
        all_qvalues = self.head_critic(x)
        qvalue = torch.gather(all_qvalues, 1, operation[:, None])
        if pi is not None:
            value = (pi * all_qvalues).sum(1)
            return qvalue.squeeze(1), value, all_qvalues
        return qvalue.squeeze(1), self.head_aux(x), dist
    
