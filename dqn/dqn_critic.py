import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

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

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
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

class GPT_DQNCritic(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.num_pixel = cfg.env.grid_x * cfg.env.grid_y
        self.num_tokens = self.num_pixel + 1

        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_pixel, cfg.model.n_embd))
        self.goal_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.state_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.critic_emb = nn.Parameter(torch.zeros(1, 1, cfg.model.n_embd))
        self.drop = nn.Dropout(cfg.model.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.model.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(cfg.model.n_embd)
        self.head_action = nn.Linear(cfg.model.n_embd, cfg.env.num_actions)
        self.head_critic = nn.Linear(cfg.model.n_embd, 1)

        self.color_encoder = nn.Embedding(cfg.env.num_colors, cfg.model.n_embd)
        self.obj_encoder = nn.Embedding(self.num_pixel, cfg.model.n_embd)

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

        decay.add('pos_emb')
        decay.add('critic_emb')
        decay.add('goal_emb')
        decay.add('state_emb')

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

    # state, action, and return
    def forward(self, state_colors, state_objs, goal_colors, goal_objects):

        batch_size = state_colors.shape[0]
        color_embeddings = self.color_encoder(state_colors.type(torch.long)) # batch_size, num_pixels, n_embd
        obj_embeddings = self.obj_encoder(state_objs.type(torch.long)) # batch_size, num_pixels, n_embd
        state_embedding = color_embeddings + obj_embeddings + self.pos_emb + self.state_emb

        color_embeddings = self.color_encoder(goal_colors.type(torch.long))
        obj_embeddings = self.obj_encoder(goal_objects.type(torch.long)) # batch_size, num_pixels, n_embd
        goal_embedding = color_embeddings + self.pos_emb + self.goal_emb # + obj_embedding

        critic_embed = torch.tile(self.critic_emb, [batch_size, 1, 1])

        inputs = torch.cat([state_embedding, critic_embed, goal_embedding], axis=1)
        x = self.drop(inputs)
        x = self.blocks(x)
        x = self.ln_f(x)
        #values = self.head_action(torch.cat([x[:, :self.num_pixel], x[:, self.num_pixel:]], axis=2)).view(batch_size, -1)
        action_logits = self.head_action(x[:, :self.num_pixel]).view(batch_size, -1)
        value = self.head_critic(x[:, self.num_pixel]).view(batch_size)

        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)

        return action_logits, action, action_logprob, value
