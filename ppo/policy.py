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

    def __init__(self, cfg, num_fixed_tokens, num_all_tokens):
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
        self.num_fixed_tokens = num_fixed_tokens
        self.register_buffer("bias", torch.tril(torch.ones(
            num_all_tokens, num_all_tokens)).view(1, 1, num_all_tokens, num_all_tokens))
        self.bias[:, :, :, :num_fixed_tokens] = 1
        self.bias = (1 - self.bias).bool().tile(1, self.n_head, 1, 1)

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
        att[self.bias[:, :, :T, :T].tile(B, 1, 1, 1)] = float('-inf')

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, cfg, num_fixed_tokens, num_all_tokens):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.model.n_embd)
        self.ln2 = nn.LayerNorm(cfg.model.n_embd)
        self.resid_drop = nn.Dropout(cfg.model.resid_pdrop)
        self.attn = CausalSelfAttention(cfg, num_fixed_tokens, num_all_tokens)
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

class GPTPolicy(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.num_pixel = cfg.env.grid_x * cfg.env.grid_y
        self.grid_shape = torch.tensor([self.cfg.env.grid_x, self.cfg.env.grid_y], device=self.device)

        self.num_fixed_tokens = self.num_pixel * 2 + 2 # info_tkn, cls_tkn
        self.num_action_recur = 5
        self.num_all_tokens = self.num_fixed_tokens + self.num_action_recur

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_pixel, cfg.model.n_embd) * 0.02)
        self.global_pos_emb = nn.Parameter(
            torch.randn(1, self.num_all_tokens, cfg.model.n_embd) * 0.02)
        self.state_emb = nn.Parameter(torch.randn(8, 1, cfg.model.n_embd) * 0.02)
        self.bbox_emb = nn.Linear(4, cfg.model.n_embd)
        self.drop = nn.Dropout(cfg.model.embd_pdrop)
        self.cls_tkn = nn.Parameter(torch.randn(1, 1, cfg.model.n_embd) * 0.02)

        # transformer
        self.blocks = nn.Sequential(*[Block(
            cfg, self.num_fixed_tokens, self.num_all_tokens
            ) for _ in range(cfg.model.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(cfg.model.n_embd)
        self.head_critic = nn.Linear(cfg.model.n_embd, cfg.env.num_actions)

        self.color_encoder = nn.Embedding(cfg.env.num_colors, cfg.model.n_embd)
        self.binary_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.term_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.trials_encoder = nn.Embedding(4, cfg.model.n_embd)
        self.active_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.rotation_encoder = nn.Embedding(4, cfg.model.n_embd)

        self.operation_encoder = nn.Embedding(cfg.env.num_actions, cfg.model.n_embd)
        self.bbox_encoder_1 = nn.Sequential(nn.Linear(1, cfg.model.n_embd), GELU())
        self.bbox_encoder_2 = nn.Sequential(nn.Linear(1, cfg.model.n_embd), GELU())
        self.bbox_encoder_3 = nn.Sequential(nn.Linear(1, cfg.model.n_embd), GELU())
        self.bbox_encoder_4 = nn.Sequential(nn.Linear(1, cfg.model.n_embd), GELU())
        self.encoders = [
            self.operation_encoder, self.bbox_encoder_1,
            self.bbox_encoder_2, self.bbox_encoder_3, self.bbox_encoder_4
        ]

        self.apply(self._transformer_init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # rnn policy
        def linear_with_orthogonal_init(inp_dim, oup_dim, scale):
            linear = nn.Linear(inp_dim, oup_dim)
            torch.nn.init.orthogonal_(linear.weight, scale)
            torch.nn.init.zeros_(linear.bias)
            return linear
            
        head_factory = lambda last_dim, oup_init_scale: nn.Sequential(
            linear_with_orthogonal_init(cfg.model.n_embd, cfg.model.n_embd, math.sqrt(2)), GELU(), 
            linear_with_orthogonal_init(cfg.model.n_embd, cfg.model.n_embd, math.sqrt(2)), GELU(), 
            linear_with_orthogonal_init(cfg.model.n_embd, last_dim, oup_init_scale))
        self.head_operation = head_factory(cfg.env.num_actions, 0.01)
        self.head_bbox_1 = head_factory(2, 0.01)
        self.head_bbox_2 = head_factory(2, 0.01)
        self.head_bbox_3 = head_factory(2, 0.01)
        self.head_bbox_4 = head_factory(2, 0.01)
        self.head_aux_reward = head_factory(1, 1)
        self.head_bboxes = [
            self.head_bbox_1, self.head_bbox_2, self.head_bbox_3, self.head_bbox_4, self.head_aux_reward]
        self.head_critic = head_factory(1, 1)
        self.head_aux_rtm1 = head_factory(1, 1)

        assert len(self.encoders) == self.num_action_recur == len(self.head_bboxes)

    def _transformer_init_weights(self, module):
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

        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.base_lr, betas=(self.cfg.train.beta1, self.cfg.train.beta2))

        return optimizer

    def forward(self, grid, grid_dim, selected, clip, clip_dim, terminated,
                trials_remain, active, object, object_sel, object_dim, object_pos,
                background, rotation_parity, input, input_dim, answer, answer_dim,
                additional_tokens):

        B = grid.shape[0]
        def compute_mask(base, end_dim, start_dim=None):

            if start_dim is None:
                active = torch.ones_like(base)
                tran = torch.zeros_like(end_dim)
                tran[:, 0] = end_dim[:, 0] - self.cfg.env.grid_x
                tran[:, 1] = end_dim[:, 1] - self.cfg.env.grid_y
                active = core_translate(active, tran)
            else:
                active = torch.ones_like(base)
                tran = torch.zeros_like(end_dim)
                tran[:, 0] = torch.minimum(start_dim[:, 0] + end_dim[:, 0] - self.cfg.env.grid_x, torch.zeros_like(start_dim[:, 0]))
                tran[:, 1] = torch.minimum(start_dim[:, 1] + end_dim[:, 1] - self.cfg.env.grid_y, torch.zeros_like(start_dim[:, 0]))
                active = core_translate(active, tran)

                opposite = torch.ones_like(base)
                tran[:, 0] = - start_dim[:, 0]
                tran[:, 1] = - start_dim[:, 1]
                opposite = core_translate(opposite, tran)

                opposite = torch.flip(opposite, [1, 2])
                active = torch.logical_and(active, opposite)
            return (~active.bool()).reshape(B, -1)

        def core_translate(base, pos):
            translate = torch.eye(2, 2, device=self.device)[None].tile(B, 1, 1)
            rate = - torch.flip(pos * 2 / self.grid_shape, [1])[..., None]
            translate = torch.concat([translate, rate], axis=2)
            ff = torch.nn.functional.affine_grid(translate, [B, 1, self.cfg.env.grid_x, self.cfg.env.grid_y], align_corners=False)
            res = torch.nn.functional.grid_sample(
                base.reshape([B, 1, self.cfg.env.grid_x, self.cfg.env.grid_y]).float(), ff, align_corners=False).round().long().squeeze(1)
            return res

        def translate(base, pos):
            pos[:, 0] = torch.remainder(pos[:, 0] + self.cfg.env.grid_x, self.cfg.env.grid_x)
            pos[:, 1] = torch.remainder(pos[:, 1] + self.cfg.env.grid_y, self.cfg.env.grid_y)
            return core_translate(base, pos)
        

        active_grid = compute_mask(grid, grid_dim)
        active_ans = compute_mask(answer, answer_dim)

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

        additional_tokens = [
            each.reshape(-1, 1, self.cfg.model.n_embd) for each in additional_tokens]
        # inputs = torch.cat([
        #     grid, selected, clip, object, 
        #     object_sel, background, input, answer, info_tkn, 
        #     cls_tkn] + additional_tokens, axis=1)
        inputs = torch.cat([
            grid, answer, info_tkn, cls_tkn] + additional_tokens, axis=1)
        
        inputs = inputs + self.global_pos_emb[:, :inputs.shape[1]]

        # masks = torch.cat([
        #     active_grid, active_grid, active_clip, active_obj,
        #     active_obj, active_grid + 1 - active.reshape(-1, 1), active_inp, active_ans, 
        #     torch.zeros((len(active_grid), 2 + len(additional_tokens)),
        #                  device=self.device)], axis=1)
        masks = torch.cat([
            active_grid, active_ans, 
            torch.zeros((len(active_grid), 2 + len(additional_tokens)),
                         device=self.device)], axis=1)

        x = self.drop(inputs)
        x, _ = self.blocks((x, masks.bool()))
        x = self.ln_f(x)

        return x

    def act(self, **kwargs):
        
        x = self.forward(**kwargs, additional_tokens=[])
        value = self.head_critic(x[:, -1]).squeeze(1)
        rtm1_pred = self.head_aux_rtm1(x[:, -1]).squeeze(1)

        dist = Categorical(logits=self.head_operation(x[:, -1]))
        operation = sample = dist.sample()
        log_prob = dist.log_prob(sample)

        additional_tokens = []
        bbox = []
        for i, (encoder, head) in enumerate(zip(self.encoders, self.head_bboxes)):
            additional_tokens.append(encoder(sample.reshape(-1, 1)))
            x = self.forward(**kwargs, additional_tokens=additional_tokens)
            if i != self.num_action_recur - 1:
                output = head(x[:, -1])
                dist = TruncatedNormal(
                    torch.nn.functional.sigmoid(output[:, 0]), 
                    torch.exp(torch.clamp(output[:, 1], -20, 2)),
                    0, 1)
                sample = dist.sample()
                log_prob += dist.log_prob(sample)
                bbox.append(sample)
            else:
                r_pred = head(x[:, -1]).squeeze(1)
        return operation, torch.stack(bbox, dim=1), -log_prob, value, rtm1_pred, r_pred

    def evaluate(self, operation, bbox, **kwargs):
        additional_tokens = [
            self.operation_encoder(operation),
            self.bbox_encoder_1(bbox[:, 0].reshape(-1, 1)),
            self.bbox_encoder_2(bbox[:, 1].reshape(-1, 1)),
            self.bbox_encoder_3(bbox[:, 2].reshape(-1, 1)),
            self.bbox_encoder_4(bbox[:, 3].reshape(-1, 1))
        ]
        x = self.forward(**kwargs, additional_tokens=additional_tokens)
        vpred = self.head_critic(x[:, -1 - self.num_action_recur]).squeeze(1)
        rtm1_pred = self.head_aux_rtm1(x[:, -1 - self.num_action_recur]).squeeze(1)

        operation_dist = Categorical(logits=self.head_operation(x[:, -1 - self.num_action_recur]))
        log_prob = operation_dist.log_prob(operation.squeeze())
        entropy = operation_dist.entropy()

        bbox_logits = [
            self.head_bbox_1(x[:, -5]),
            self.head_bbox_2(x[:, -4]),
            self.head_bbox_3(x[:, -3]),
            self.head_bbox_4(x[:, -2]),
        ]
        bbox_dist = [TruncatedNormal(
                    torch.nn.functional.sigmoid(each[:, 0]), 
                    torch.exp(torch.clamp(each[:, 1], -20, 2)),
                    0, 1) for each in bbox_logits]

        for i, dist in enumerate(bbox_dist):
            log_prob += dist.log_prob(bbox[:, i])
            assert entropy.shape == dist.entropy.shape
            entropy += dist.entropy
        r_pred = self.head_aux_reward(x[:, -1]).squeeze(1)

        return -log_prob, vpred, entropy, rtm1_pred, r_pred