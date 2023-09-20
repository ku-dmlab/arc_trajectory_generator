import os
from dataclasses import dataclass

import numpy as np
import torch
from embedder.model import Encoder, Decoder
from tqdm import tqdm
from omegaconf import OmegaConf


class ReplayBuffer:
    def __init__(self, device, memory_size=1e5):
        super().__init__()
        self.max_mem_size = int(memory_size)
        self.counter = 0
        self.grid = []
        self.grid_dim = []
        self.selected = []
        self.clip = []
        self.clip_dim = []
        self.selection = []
        self.operation = []
        self.device = device

    def add_experience(self, grid, grid_dim, selected, clip, clip_dim, selection, operation):
        if self.counter < self.max_mem_size:
            self.grid.append(grid)
            self.grid_dim.append(grid_dim)
            self.selected.append(selected)
            self.clip.append(clip)
            self.clip_dim.append(clip_dim)
            self.selection.append(selection)
            self.operation.append(operation)
        else:
            i = self.counter % self.max_mem_size
            self.grid[i] = grid
            self.grid_dim[i] = grid_dim
            self.selected[i] = selected
            self.clip[i] = clip
            self.clip_dim[i] = clip_dim
            self.selection[i] = selection
            self.operation[i] = operation
        self.counter += 1

    def get_random_experience(self, batch_size):
        Idx = np.random.choice(min(self.counter, self.max_mem_size), batch_size, replace=False)
        grid = np.stack([self.grid[i] for i in Idx])
        grid_dim = np.stack([self.grid_dim[i] for i in Idx])
        selected = np.stack([self.selected[i] for i in Idx])
        clip = np.stack([self.clip[i] for i in Idx])
        clip_dim = np.stack([self.clip_dim[i] for i in Idx])
        selection = np.stack([self.selection[i] for i in Idx])
        operation = np.stack([self.operation[i] for i in Idx])

        rets = (grid, grid_dim, selected, clip, clip_dim, selection, operation)
        types = [
            torch.cuda.LongTensor,
            torch.cuda.LongTensor,
            torch.cuda.FloatTensor,
            torch.cuda.LongTensor,
            torch.cuda.LongTensor,
            torch.cuda.FloatTensor,
            torch.cuda.LongTensor
        ]

        return [torch.tensor(each, device=self.device).type(tp) for each, tp in zip(rets, types)]

class Trainer:
    def __init__(self, env, cfg, logger, device):
        self.device = device
        self.learning_rate = cfg.train.learning_rate
        self.batch_size = cfg.train.batch_size

        self.encoder = Encoder(cfg, env).to(self.device)
        self.decoder = Decoder(cfg, env).to(self.device)
        self.buffer = ReplayBuffer(self.device)

        self.encoder_optimizer = self.encoder.configure_optimizers()
        self.decoder_optimizer = self.decoder.configure_optimizers()
        self.selected_recon_loss = torch.nn.BCELoss()
        self.action_recon_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        self.env = env
        self.cfg = cfg
        self.logger = logger

    def _rollout(self, timestep):

        obs, info = self.env.reset()

        losses, accs = [], []

        for t in range(self.cfg.train.max_ep_len):
            action = self.env.action_space.sample()
            action["selection"] = action["selection"].astype(bool)
            self.buffer.add_experience(
                obs["grid"], obs["grid_dim"], obs["selected"], obs["clip"], obs["clip_dim"], action["selection"], action["operation"])
            obs, reward, term, trunc, info = self.env.step(action)

            ret = self._train(timestep + t)
            if ret is not None:
                losses.append(ret[0])
                accs.append(ret[1])

            # if term or trunc:
            #     break
        return t + 1, np.mean(losses), np.mean(accs)


    def learn(self, checkpoint_dir):
        t = 0
        t_logged = 0
        pbar = tqdm(total=self.cfg.train.total_timesteps)
        while t < self.cfg.train.total_timesteps:
            len_episode, loss, acc = self._rollout(t)
            self.logger.log(
                t, {"train/loss": loss, "train/acc": acc, "train/ep_len": len_episode - t}
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
        torch.save(self.encoder.state_dict(), os.path.join(direc, "encoder.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(direc, "decoder.pt"))

    def _train(self, t):
        if self.buffer.counter < self.batch_size:
            return
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        (grid, grid_dim, selected, clip, clip_dim, selection, operation) = self.buffer.get_random_experience(self.batch_size)

        z = self.encoder(grid, grid_dim, selected, clip, clip_dim, selection, operation)
        selection_logit, action_logit = self.decoder(grid, grid_dim, selected, clip, clip_dim, z)

        reshaped_selection = selection.reshape(selection.shape[0], -1)
        loss = self.selected_recon_loss(torch.sigmoid(selection_logit), reshaped_selection) + self.action_recon_loss(action_logit, operation)
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad.clip_grad_norm_(self.decoder.parameters(), 1.0)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


        acc = torch.logical_and(((torch.sigmoid(selection_logit) > 0.5) == reshaped_selection).all(-1), torch.argmax(action_logit, dim=-1) == operation)
        return loss.item(), acc.float().mean().item()
    