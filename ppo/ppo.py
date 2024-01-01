
import numpy as np

from .model import get_train_fn, get_act_fn
from .runner import Runner
from .policy import GPTPolicy
from utils.util import get_device
import torch
from tqdm import trange
import wandb
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def learn(cfg, env):

    tcfg = cfg.train
    device = get_device(tcfg.gpu_num)
    policy = GPTPolicy(cfg, device).to(device)
    policy = torch.compile(policy)

    nenvs = tcfg.nenvs
    nbatch = nenvs * tcfg.nsteps

    nbatch_train = nbatch // tcfg.nminibatches
    nupdates = tcfg.total_timesteps // nbatch

    optimizer = policy.configure_optimizers()
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, 
    #     lr_lambda=lambda step: 1.0 - (step - 1.0) / nupdates,
    #     verbose=False)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1000,
                                          cycle_mult=1.0,
                                          max_lr=5e-4,
                                          min_lr=5e-5,
                                          warmup_steps=500,
                                          gamma=1.0)
    train_fn = get_train_fn(policy, optimizer, tcfg)
    act_fn = get_act_fn(policy)

    # Instantiate the runner object
    runner = Runner(env, cfg, device, act_fn)
    all_ep_rets = []

    for update in trange(1, nupdates + 1):
        assert nbatch % tcfg.nminibatches == 0

        # Get minibatch
        print("Rollout:")
        (ob_acs, returns, values, neglogpacs, rtm1, rtm1_pred, norm_rew, rpred, gpred, gtp1,
         ep_rets, success_ts, ebbox) = runner.run()
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        all_ep_rets += ep_rets
        print("Done.")
        # Here what we're going to do is for each minibatch calculate the loss and append it.
        train_rets = []
        
        # Index of each element of batch_size
        # Create the indices array
        print("Training:")
        inds = np.arange(nbatch)
        train_actor = update >= tcfg.update_actor_after
        for _ in trange(tcfg.noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                ob_ac_slice = {key: ob_acs[key][mbinds] for key in ob_acs}
                slices = (arr[mbinds] for arr in (
                    returns, values, neglogpacs, advs, rtm1, norm_rew, gtp1))
                train_rets.append(torch.stack(train_fn(ob_ac_slice, *slices, train_actor)))
        print("Done.")
        scheduler.step()
        
        # Feedforward --> get losses --> update
        lossvals = npy(torch.mean(torch.stack(train_rets), axis=0))

        if update % tcfg.log_interval == 0:
            watched = {"old_state_val": values, 
                       "old_neg_logprobs": neglogpacs, 
                       "returns": returns, 
                       "rewards": rtm1, 
                       "rtm1_pred": rtm1_pred, 
                       "rpred": rpred}
            logged = {}
            for key, val in watched.items():
                logged[f"{key}/mean"] = safemean(npy(val))
                logged[f"{key}/min"] = np.min(npy(val))
                logged[f"{key}/max"] = np.max(npy(val))
                logged[f"{key}/std"] = np.std(npy(val))
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = max(1 - np.var(npy(returns) - npy(values))/np.var(npy(returns)), -1)
            ev_aux_rtm1 = max(1 - np.var(npy(rtm1) - npy(rtm1_pred))/np.var(npy(rtm1)), -1)
            ev_aux_rew = max(1 - np.var(npy(norm_rew) - npy(rpred))/np.var(npy(norm_rew)), -1)
            ev_aux_rew_on_rtm1 = max(1 - np.var(npy(norm_rew) - npy(rpred))/np.var(npy(norm_rew) - npy(rtm1_pred)), -1)
            acc_aux_gtp1 = (npy(gpred).argmax(2) == npy(gtp1)).mean()
            wandb.log(
                {
                    "rollout/operation": npy(ob_acs["operation"]),
                    "rollout/bbox_0": npy(ob_acs["bbox"][:, 0]),
                    "rollout/bbox_1": npy(ob_acs["bbox"][:, 1]),
                    "rollout/bbox_2": npy(ob_acs["bbox"][:, 2]),
                    "rollout/bbox_3": npy(ob_acs["bbox"][:, 3]),
                    "rollout/ebbox_0": ebbox[:, 0],
                    "rollout/ebbox_1": ebbox[:, 1],
                    "rollout/ebbox_2": ebbox[:, 2],
                    "rollout/ebbox_3": ebbox[:, 3],
                    "misc/serial_timesteps": update * tcfg.nsteps,
                    "misc/nupdates": update,
                    "misc/total_timesteps": update * nbatch, 
                    "explained_variance": float(ev),
                    "explained_variance_aux_rtm1": float(ev_aux_rtm1),
                    "explained_variance_aux_rew": float(ev_aux_rew),
                    "explained_variance_aux_rew_on_rtm1": float(ev_aux_rew_on_rtm1),
                    "acc_aux_gtp1": float(acc_aux_gtp1),
                    'eprewmean': safemean(all_ep_rets),
                    'success_ts': safemean(success_ts),
                    'success_rate': safemean(np.array(success_ts) > 0),
                    'lr': scheduler.get_lr()[0],
                    'loss/loss': lossvals[0],
                    'loss/pg_loss': lossvals[1],
                    'loss/vf_loss': lossvals[2],
                    'loss/entropy_loss': lossvals[3],
                    'loss/aux_loss_rtm1': lossvals[4],
                    'loss/aux_loss_rew': lossvals[5],
                    'loss/aux_loss_gtp1': lossvals[6],
                    'loss/approxkl': lossvals[7],
                    'loss/clipfrac': lossvals[8],
                } | logged
            )
            all_ep_rets = []

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def npy(tensor):
    return tensor.detach().cpu().numpy()