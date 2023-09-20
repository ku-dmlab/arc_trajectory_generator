import os
import pickle

from functools import reduce
import numpy as np
import torch
from bclone.model import Policy
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset, DataLoader
import copy

class Trainer:
    def __init__(self, cfg, trajs, logger, device):
        self.device = device
        self.learning_rate = cfg.train.learning_rate
        self.batch_size = cfg.train.batch_size

        self.policy = Policy(cfg).to(self.device)

        self.optimizer = self.policy.configure_optimizers()
        
        self.selected_loss = torch.nn.BCELoss()
        self.grid_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.grid_dim_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.clip_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.clip_dim_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        self.cfg = cfg
        self.logger = logger

        dataset = dict(selected=[], grid=[], grid_dim=[], clip=[], clip_dim=[])
        target_dataset = dict(selected=[], grid=[], grid_dim=[], clip=[], clip_dim=[])
        goals = []
        goal_dims = []
        for traj in trajs:
            for i in range(len(traj) - 1):
                for key in dataset:
                    dataset[key].append(traj[i][key])
                    target_dataset[key].append(traj[i + 1][key])
                goals.append(copy.deepcopy(traj[-1]["grid"]))
                goal_dims.append(copy.deepcopy(traj[-1]["grid_dim"]))
        
        for key in dataset:
            dataset[key] = torch.LongTensor(dataset[key])
            target_dataset[key] = torch.LongTensor(target_dataset[key])
        goals = torch.LongTensor(goals)
        goal_dims = torch.LongTensor(goal_dims)
        
        ds = TensorDataset(
            *[dataset[key].to(device) for key in ["selected", "grid", "grid_dim", "clip", "clip_dim"]],
            *[target_dataset[key].to(device) for key in ["selected", "grid", "grid_dim", "clip", "clip_dim"]],
            goals.to(device), goal_dims.to(device)
        )
        self.loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True)

    def _train(self, timestep):

        agg = [[] for _ in range(7)]
        for batches in self.loader:
            ret = self._train_step(*batches)
            if ret is not None:
                for i in range(len(agg)):
                    agg[i].append(ret[i])
        return timestep + len(self.loader), *[np.mean(each) for each in agg]

    def learn(self, checkpoint_dir):
        t = 0
        t_logged = 0
        pbar = tqdm(total=self.cfg.train.total_timesteps)
        while t < self.cfg.train.total_timesteps:
            len_episode, loss, s_acc, g_acc, gd_acc, c_acc, cd_acc, acc = self._train(t)
            self.logger.log(
                t, {"loss": loss, 
                    "s_acc": s_acc, "g_acc": g_acc, "gd_acc": gd_acc, "c_acc": c_acc, "cd_acc": cd_acc,
                    "acc": acc, "ep_len": len_episode - t}
            )
            t += len_episode
            pbar.update(len_episode)

            if t - t_logged >= self.cfg.logging.log_interval:
                t_logged = t
                if checkpoint_dir is not None:
                    self.save(os.path.join(checkpoint_dir, f"checkpoint_{t}"))
        pbar.close()

    def save(self, direc):
        os.makedirs(direc, exist_ok=True)
        OmegaConf.save(self.cfg, os.path.join(direc, "config.yaml"))
        torch.save(self.policy.state_dict(), os.path.join(direc, "policy.pt"))

    def _train_step(self, *args):
        s, g, gd, c, cd, ns, ng, ngd, nc, ncd, goal, goal_dim = args
        B, H, W, = s.shape
        flat_shape = (B, H * W)
        self.optimizer.zero_grad()

        ps, pg, pgd, pc, pcd = self.policy(s, g, gd, c, cd, goal, goal_dim)
        loss = (
            self.selected_loss(torch.sigmoid(ps), ns.reshape(flat_shape).float()) + 
            self.grid_loss(pg.transpose(1, 2), ng.reshape(flat_shape)) + 
            self.grid_dim_loss(pgd.transpose(1, 2), ngd) + 
            self.clip_loss(pc.transpose(1, 2), nc.reshape(flat_shape)) + 
            self.clip_dim_loss(pcd.transpose(1, 2), ncd))
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        s_acc = (torch.sigmoid(ps > 0.5) == ns.reshape(flat_shape)).all(-1)
        g_acc = (torch.argmax(pg, -1) == ng.reshape(flat_shape)).all(-1)
        gd_acc = (torch.argmax(pgd, -1) == ngd).all(-1)
        c_acc = (torch.argmax(pc, -1) == nc.reshape(flat_shape)).all(-1)
        cd_acc = (torch.argmax(pcd, -1) == ncd).all(-1)
        acc = reduce(torch.logical_and, [s_acc, g_acc, gd_acc, c_acc, cd_acc], torch.ones_like(s_acc))

        return loss.item(), *[each.float().mean().item() for each in [s_acc, g_acc, gd_acc, c_acc, cd_acc, acc]]
