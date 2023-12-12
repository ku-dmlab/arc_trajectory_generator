
import os
import datetime

import numpy as np
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import foarcle
from arcle.loaders import Loader
from dqn.dqn import DQN
from utils.util import get_device
import wandb

class TestLoader(Loader):

    def __init__(self, size_x, size_y, **kwargs):
        self.size_x = size_x
        self.size_y = size_y
        
        self.rng = np.random.default_rng(12345)
        super().__init__(**kwargs)

    def get_path(self, **kwargs):
        return ['']

    def parse(self, **kwargs):
        ti= np.zeros((self.size_x,self.size_y), dtype=np.uint8)
        to = np.zeros((self.size_x,self.size_y), dtype=np.uint8)
        ei = np.zeros((self.size_x,self.size_y), dtype=np.uint8)
        eo = np.zeros((self.size_x,self.size_y), dtype=np.uint8)

        ti[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        to[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        ei[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        eo[0:self.size_x, 0:self.size_y] = self.rng.integers(0,10, size=[self.size_x,self.size_y])
        return [([ti],[to],[ei],[eo], {'desc': "just for test"})]


@hydra.main(config_path="dqn", config_name="dqn_config")
def main(cfg: DictConfig) -> None:
    wandb.init(
        project="arc_traj_gen",
        config=OmegaConf.to_container(cfg)
    )

    def set_metric(ys, x):
        wandb.define_metric(x)
        for each in ys:
            wandb.define_metric(each, step_metric = x)

    train_step_metrics = [
        "online/rtgs_mean", "online/rtgs_min", "online/rtgs_max",
        "online/adv_mean", "online/adv_min", "online/adv_max",
        "online/old_logprobs_mean", "online/old_logprobs_min", "online/old_logprobs_max",
        "online/old_state_values_mean", "online/old_state_values_min", "online/old_state_values_max"
        "rollout/operation", "rollout/bbox_0", "rollout/bbox_1", "rollout/bbox_2", "rollout/bbox_3"]
    set_metric(train_step_metrics, "train_step")
    i_episode_metrics = ["rollout/done", "rollout/ep_ret", "rollout/ep_len"]
    set_metric(i_episode_metrics, "i_episode")
    grad_step_metrics = [
        "online/ratios_mean", "online/ratios_min", "online/ratios_max",
        "online/logprobs_mean", "online/logprobs_min", "online/logprobs_max",
        "online/state_values_mean", "online/state_values_min", "online/state_values_max",
        "loss/loss", "loss/actor_loss", "loss/critic_loss", "loss/entropy_loss"]
    set_metric(grad_step_metrics, "grad_step")


    ckpt_dir = os.path.join(cfg.logging.logging_dir, cfg.logging.exp_type)
    
    env = gym.make(
        'ARCLE/FOO2ARCv2Env-v0', 
        data_loader = TestLoader(cfg.env.grid_x, cfg.env.grid_y), 
        max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
        colors=10)

    agent = DQN(env, cfg)

    agent.learn(ckpt_dir)


if __name__ == "__main__":
    main()
