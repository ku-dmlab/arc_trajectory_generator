

import torch


B = 2


def for_compute_mask(base, end_dim, start_dim=None):
    active = torch.ones_like(base)
    if not (end_dim <=5).all() or not (end_dim >= 0).all():
        print(end_dim)
    for i in range(B):
        if start_dim is None:
            active[i, :end_dim[i, 0], :end_dim[i, 1]] = 0
        else:
            active[i, 
            torch.maximum(start_dim[i, 0], 0 * start_dim[i, 0]):torch.maximum(start_dim[i, 0]+ end_dim[i, 0], 0 * start_dim[i, 0]),
            torch.maximum(start_dim[i, 1], 0 * start_dim[i, 1]):torch.maximum(start_dim[i, 1]+ end_dim[i, 1], 0 * start_dim[i, 0])] = 0
    return active.reshape(B, -1)

def for_translate(base, pos):
    ret = torch.zeros_like(base)
    for i in range(B):
        target_x = slice(None) if pos[i, 0] == 0 else slice(None, -pos[i, 0])
        target_y = slice(None) if pos[i, 1] == 0 else slice(None, -pos[i, 1])
        ret[i, pos[i, 0]:, pos[i, 1]:] = base[i, target_x, target_y]
    return ret

def compute_mask(base, end_dim, start_dim=None):

    if start_dim is None:
        active = torch.ones_like(base)
        tran = torch.zeros_like(end_dim)
        tran[:, 0] = end_dim[:, 0] - base.shape[1]
        tran[:, 1] = end_dim[:, 1] - base.shape[2]
        active = core_translate(active, tran)
    else:
        active = torch.ones_like(base)
        tran = torch.zeros_like(end_dim)
        tran[:, 0] = torch.minimum(start_dim[:, 0] + end_dim[:, 0] - base.shape[1], torch.zeros_like(start_dim[:, 0]))
        tran[:, 1] = torch.minimum(start_dim[:, 1] + end_dim[:, 1] - base.shape[2], torch.zeros_like(start_dim[:, 0]))
        active = core_translate(active, tran)

        opposite = torch.ones_like(base)
        tran[:, 0] = - start_dim[:, 0]
        tran[:, 1] = - start_dim[:, 1]
        opposite = core_translate(opposite, tran)

        opposite = torch.flip(opposite, [1, 2])
        active = torch.logical_and(active, opposite)
    return (~active.bool()).reshape(B, -1)

def core_translate(base, pos):
    translate = torch.eye(2, 2, device=base.device)[None].tile(base.shape[0], 1, 1)
    rate = - torch.flip(pos * 2 / torch.tensor(base.shape[1:], device=pos.device), [1])[..., None]
    translate = torch.concat([translate, rate], axis=2)
    ff = torch.nn.functional.affine_grid(translate, [base.shape[0], 1, base.shape[1], base.shape[2]], align_corners=False)
    res = torch.nn.functional.grid_sample(base.float()[:, None], ff, align_corners=False).round().long().squeeze(1)
    return res

def translate(base, pos):
    pos[:, 0] = torch.remainder(pos[:, 0] + base.shape[1], base.shape[1])
    pos[:, 1] = torch.remainder(pos[:, 1] + base.shape[2], base.shape[2])
    return core_translate(base, pos)

from tqdm import trange
for _ in trange(100):
    base = torch.randint(B, [B, 5, 5])
    pos1 = torch.randint(5, [B, 2])
    pos2 = torch.randint(10, [B, 2]) - 5
    a = for_compute_mask(base, pos1, pos2).bool()
    b = compute_mask(base, pos1, pos2)
    if not (a == b).all():
        print(pos2)
        print(pos1)
        print(a.reshape(2, 5, 5))
        print(b.reshape(2, 5, 5))
        break
for _ in trange(100):
    base = torch.randint(10, [B, 5, 5])
    pos1 = torch.randint(10, [B, 2]) - 5
    a = for_translate(base, pos1)
    b = translate(base, pos1)
    if not (a == b).all():
        print(base)
        print(pos1)
        print(a.reshape(2, 5, 5))
        print(b.reshape(2, 5, 5))
        break