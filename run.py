
import sys

import numpy as np
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import foarcle
from arcle.loaders import Loader
from loader import SizeConstrainedLoader
import wandb
from hydra import compose, initialize
from omegaconf import OmegaConf


from ppo.ppo import learn

class TestLoader(Loader):

    def __init__(self, size_x, size_y, **kwargs):
        self.size_x = size_x
        self.size_y = size_y
        
        self.rng = np.random.default_rng(12345)
        super().__init__(**kwargs)

    def get_path(self, **kwargs):
        return ['']

    def pick(self, **kwargs):
        return self.parse()[0]

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


#@hydra.main(config_path="ppo", config_name="ppo_config_random")
def main(gpu_num) -> None:
    with initialize(version_base=None, config_path="ppo", job_name="main"):
        cfg = compose(config_name=f"ppo_config_random{gpu_num}")

    wandb.init(
        project="arc_traj_gen",
        config=OmegaConf.to_container(cfg)
    )
    if cfg.env.use_arc:
        env = gym.make(
            'ARCLE/FOO2ARCv2Env-v0', 
            data_loader = SizeConstrainedLoader(cfg.env.grid_x), 
            max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
            colors=cfg.env.num_colors)
    else:
        env = gym.make(
            'ARCLE/FOO2ARCv2Env-v0', 
            data_loader = TestLoader(cfg.env.grid_x, cfg.env.grid_y), 
            max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
            colors=cfg.env.num_colors)

    learn(cfg, env)


def sweep():
    with initialize(version_base=None, config_path="ppo", job_name="main"):
        cfg = compose(config_name=f"ppo_config_random0")

    wandb.init(
        project="arc_traj_gen",
        config=OmegaConf.to_container(cfg)
    )
    cfg.train.nenvs = wandb.config.train["nenvs"]
    cfg.train.ent_coef = wandb.config.train["ent_coef"]
    cfg.train.base_lr = wandb.config.train["base_lr"]
    cfg.train.gamma = wandb.config.train["gamma"]
    cfg.train.nminibatches = wandb.config.train["nminibatches"]
    cfg.train.noptepochs = wandb.config.train["noptepochs"]
    cfg.train.cliprange = wandb.config.train["cliprange"]
    cfg.train.weight_decay = wandb.config.train["weight_decay"]
    cfg.train.first_cycle_steps = wandb.config.train["first_cycle_steps"]
    cfg.train.cycle_mult = wandb.config.train["cycle_mult"]
    cfg.train.warmup_steps = wandb.config.train["warmup_steps"]
    cfg.train.ender_reward = wandb.config.train["ender_reward"]
    cfg.model.n_layer = wandb.config.model["n_layer"]
    cfg.model.n_head = wandb.config.model["n_head"]
    cfg.model.n_embd = wandb.config.model["n_embd"]

    if cfg.env.use_arc:
        env = gym.make(
            'ARCLE/FOO2ARCv2Env-v0', 
            data_loader = SizeConstrainedLoader(cfg.env.grid_x), 
            max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
            colors=cfg.env.num_colors)
    else:
        env = gym.make(
            'ARCLE/FOO2ARCv2Env-v0', 
            data_loader = TestLoader(cfg.env.grid_x, cfg.env.grid_y), 
            max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
            colors=cfg.env.num_colors)

    learn(cfg, env)


if __name__ == "__main__":
    #main(int(sys.argv[1]))
    sweep()
