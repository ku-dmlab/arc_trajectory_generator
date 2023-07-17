import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

import hydra
from omegaconf import DictConfig
from dqn.dqn import DQN
from arc_env import MiniArcEnv, TestEnv1, DataBasedARCEnv
from utils.logger import Logger

import cProfile

MAX_ROW = 5
MAX_COL = 5
NUM_ACTIONS = 28

def print_state(state):
    for row in state.reshape(MAX_ROW, MAX_COL):
        print(" ".join([str(each) for each in row]))

def get_device(gpu_num):
    if gpu_num is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_num}')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")
    return device

@hydra.main(config_path="dqn", config_name="dqn_config")
def main(cfg: DictConfig) -> None:

    device = get_device(cfg.train.gpu_num)
    logging_dir = os.path.join(cfg.logging.logging_dir, cfg.logging.exp_type)
    logger = Logger(
        logging_dir = logging_dir,
        run_name=f"{datetime.datetime.now().strftime('%m_%d_%H_%M_%S')}",
        print_data=cfg.logging.print_data,
        use_tb=cfg.logging.use_tb)
    
    #env = MiniArcEnv(*TestEnv1.get_args())
    env = DataBasedARCEnv(cfg)

    agent = DQN(env, cfg, logger, device)

    agent.learn(logging_dir)


if __name__ == "__main__":
    main()