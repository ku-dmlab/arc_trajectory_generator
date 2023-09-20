
import pickle
from bclone.trainer import Trainer
import hydra
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


@hydra.main(config_path="bclone", config_name="config")
def main(cfg) -> None:
    device = get_device(cfg.train.gpu_num)
    logging_dir = os.path.join(cfg.logging.logging_dir, cfg.logging.exp_type)
    logger = Logger(
        logging_dir=logging_dir,
        run_name=f"{datetime.datetime.now().strftime('%m_%d_%H_%M_%S')}",
        print_data=cfg.logging.print_data,
        use_tb=cfg.logging.use_tb,
    )

    with open("/home/bjlee/arc_trajectory_generator/converted.pickle", "rb") as f:
        trajs = pickle.load(f)
    agent = Trainer(cfg, trajs, logger, device)
    agent.learn(logging_dir)


if __name__ == "__main__":
    main()