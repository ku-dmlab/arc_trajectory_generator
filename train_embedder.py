
import arcle
import gymnasium as gym
import time
from embedder.model import Encoder, Decoder
from embedder.trainer import Trainer

import numpy as np
import hydra
from arcle.loaders import Loader

import torch
a = torch.zeros(5).cuda()

import os
import datetime

from utils.logger import Logger

def get_device(gpu_num):
    print(gpu_num)
    if gpu_num is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_num}")
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print("Device set to : cpu")
    return device


@hydra.main(config_path="embedder", config_name="embedder_config")
def main(cfg) -> None:
    device = get_device(cfg.train.gpu_num)
    logging_dir = os.path.join(cfg.logging.logging_dir, cfg.logging.exp_type)
    logger = Logger(
        logging_dir=logging_dir,
        run_name=f"{datetime.datetime.now().strftime('%m_%d_%H_%M_%S')}",
        print_data=cfg.logging.print_data,
        use_tb=cfg.logging.use_tb,
    )
    max_size = (cfg.env.grid_y, cfg.env.grid_x)
    class TestLoader(Loader):
        def get_path(self, **kwargs):
            return ['']

        def parse(self, **kwargs):
            ti, to, ei, eo= [np.zeros(max_size, dtype=np.uint8) for _ in range(4)]
            ti[:5, :5] = np.random.randint(0,10, size=[5, 5])
            return [([ti],[to],[ei],[eo], {'desc': "just for test"})]


    env = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=None, data_loader=TestLoader(), max_grid_size=(cfg.env.grid_y, cfg.env.grid_x), colors = cfg.env.num_colors, max_episode_steps=None)

    agent = Trainer(env, cfg, logger, device)
    agent.learn(logging_dir)


if __name__ == "__main__":
    main()