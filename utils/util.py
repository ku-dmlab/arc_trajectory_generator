
import torch
import numpy as np

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

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=0, shape=(), clip=10):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.clip = clip

    def update(self, x):
        batch_mean = np.mean(x, axis=0, dtype='float64')
        batch_var = np.var(x, axis=0, dtype='float64')
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
    
    def normalize(self, x, use_mean=False):
        if use_mean:
            return np.clip((x - self.mean) / np.sqrt(self.var + 1e-12), -self.clip, self.clip)
        else:
            return np.clip(x / np.sqrt(self.var + 1e-12), -self.clip, self.clip)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
