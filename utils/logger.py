import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(
        self,
        logging_dir: str,
        run_name: str,
        print_data: bool = True,
        use_tb: bool = True
    ) -> None:
        os.makedirs(logging_dir, exist_ok=True)

        self.print_data = print_data

        self.use_tb = use_tb
        if use_tb:
            log_dir = os.path.join(logging_dir, run_name)
            self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, step: int, data) -> None:
        for key in data:
            if self.print_data:
                print(f"{step}_{key}: {data[key]}")
            if self.use_tb:
                self.writer.add_scalar(key, data[key], step)

    def log_hp(self, hp) -> None:
        hp_str = "\n".join([f"|{key}|{value}" for key, value in hp])
        if self.print_data:
            print(hp_str)
        if self.use_tb:
            self.writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{hp_str}")
