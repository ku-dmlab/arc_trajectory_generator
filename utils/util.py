
import torch

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
