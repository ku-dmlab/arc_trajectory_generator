
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

        ti[0:self.size_x, 0:self.size_y] = self.rng.integers(0,2, size=[self.size_x,self.size_y])
        to[0:self.size_x, 0:self.size_y] = self.rng.integers(0,2, size=[self.size_x,self.size_y])
        ei[0:self.size_x, 0:self.size_y] = self.rng.integers(0,2, size=[self.size_x,self.size_y])
        eo[0:self.size_x, 0:self.size_y] = self.rng.integers(0,2, size=[self.size_x,self.size_y])
        return [([ti],[to],[ei],[eo], {'desc': "just for test"})]


@hydra.main(config_path="dqn", config_name="dqn_config")
def main(cfg: DictConfig) -> None:
    device = get_device(cfg.train.gpu_num)
    wandb.init(
        project="arc_traj_gen",
        config=OmegaConf.to_container(cfg)
    )
    wandb.define_metric("eval_step")
    wandb.define_metric("eval/operation", step_metric="eval_step")
    wandb.define_metric("eval/bbox_0", step_metric="eval_step")
    wandb.define_metric("eval/bbox_1", step_metric="eval_step")
    wandb.define_metric("eval/bbox_2", step_metric="eval_step")
    wandb.define_metric("eval/bbox_3", step_metric="eval_step")
    wandb.define_metric("eval/ep_len", step_metric="eval_step")
    wandb.define_metric("eval/ep_return", step_metric="eval_step")

    train_metrics = [
        "online/q_value_1_mean",
        "online/q_value_1_min",
        "online/q_value_1_max",
        "train/alpha",
        "train/op_entropy",
        "train/bbox_entropy",
        "train/entropy",
        "target/mean",
        "target/min",
        "target/max",
        "loss/loss",
        "loss/bbox_loss",
        "loss/op_loss",
        "loss/policy_loss",
        "loss/critic_loss",
        "loss/alpha_loss",
        "loss/aux_loss",
        "rollout/done",
        "rollout/epsilon", 
        "rollout/ep_ret",
        "rollout/ep_len",
        "rollout/operation",
        "rollout/bbox_0",
        "rollout/bbox_1",
        "rollout/bbox_2",
        "rollout/bbox_3",
        "rollout/rbbox_0",
        "rollout/rbbox_1",
        "rollout/rbbox_2",
        "rollout/rbbox_3"
    ]
    wandb.define_metric("train_step")
    for each in train_metrics:
        wandb.define_metric(each, step_metric="train_step")

    ckpt_dir = os.path.join(cfg.logging.logging_dir, cfg.logging.exp_type)
    
    env = gym.make(
        'ARCLE/FOO2ARCv2Env-v0', 
        data_loader = TestLoader(cfg.env.grid_x, cfg.env.grid_y), 
        max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
        colors=10)

    agent = DQN(env, cfg, device)

    agent.learn(ckpt_dir)


if __name__ == "__main__":
    main()
