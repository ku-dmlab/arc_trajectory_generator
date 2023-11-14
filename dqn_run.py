
import os
import datetime

import numpy as np
import gymnasium as gym
import hydra
from omegaconf import DictConfig

import foarcle
from arcle.loaders import Loader
from dqn.dqn import DQN
from utils.util import get_device
from utils.logger import Logger

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
    device = get_device(cfg.train.gpu_num)
    logging_dir = os.path.join(cfg.logging.logging_dir, cfg.logging.exp_type)
    logger = Logger(
        logging_dir=logging_dir,
        run_name=f"{datetime.datetime.now().strftime('%m_%d_%H_%M_%S')}",
        print_data=cfg.logging.print_data,
        use_tb=cfg.logging.use_tb,
    )

    env = gym.make(
        'ARCLE/FOO2ARCv2Env-v0', 
        data_loader = TestLoader(cfg.env.grid_x, cfg.env.grid_y), 
        max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
        colors=10)

    agent = DQN(env, cfg, logger, device)

    agent.learn(logging_dir)


if __name__ == "__main__":
    main()
