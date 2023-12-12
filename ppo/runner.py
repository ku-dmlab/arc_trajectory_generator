import torch
import numpy as np
from copy import deepcopy
from tqdm import trange
from utils.util import RunningMeanStd
from foarcle.actions.o2actions import batch_act

class Runner:
    GRIDS = [
        "input", "answer", "grid", "selected", "clip",
        "object", "object_sel","background"]
    TUPLES = [
        "input_dim", "answer_dim", "grid_dim", 
        "clip_dim", "object_dim", "object_pos",
        "bbox_pos", "bbox_dim"]
    NUMBERS = [
        "terminated", "trials_remain",
        "active", "rotation_parity",
        "operation", "reward", "rtm1", "rpred", "rtm1_pred",
        "neglogpac", "vpred"
    ]
    INFO_KEYS = ["input", "input_dim", "answer", "answer_dim"]
    STATE_KEYS = ["grid", "grid_dim", "selected", "clip", "clip_dim",
                  "terminated", "trials_remain", "active",
                  "object", "object_sel", "object_dim", "object_pos", 
                  "background", "rotation_parity"]
    ACTION_KEYS = ["operation", "bbox_pos", "bbox_dim"]

    def __init__(self, env, cfg, device, act_fn):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.att_set = set(self.GRIDS + self.TUPLES + self.NUMBERS)
        self.act_fn = act_fn
        self.rew_rms = RunningMeanStd(shape=(), clip=cfg.train.cliprew)
        self.ret_rms = RunningMeanStd(shape=(), clip=cfg.train.cliprew)
        self.reset()

    def reset(self):
        for att in self.att_set:
            setattr(self, att, [])

        state_infos = {key: [] for key in self.STATE_KEYS + self.INFO_KEYS}
        for _ in range(self.cfg.train.nenvs):
            raw_state, info = self.env.reset()
            state_info = flatten_and_copy(raw_state) | info
            for key in self.STATE_KEYS + self.INFO_KEYS:
                state_infos[key].append(state_info[key])

        for key in self.STATE_KEYS + self.INFO_KEYS:
            getattr(self, key).append(np.stack(state_infos[key]))
        self.timesteps = np.zeros(self.cfg.train.nenvs)
        self.sum_rewards = np.zeros(self.cfg.train.nenvs)
        self.disc_sum_rewards = np.zeros(self.cfg.train.nenvs)

        rewards = self._augmented_reward()
        self.rtm1.append(rewards)

    def _get_selection_from_bbox(self, bboxes):
        selection = np.zeros((len(bboxes), self.cfg.env.grid_x, self.cfg.env.grid_y)).astype(np.uint8)
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, self.cfg.env.grid_x)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, self.cfg.env.grid_y)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, self.cfg.env.grid_x)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, self.cfg.env.grid_y)
        bboxes = np.floor(bboxes).astype(np.uint8)
        for i, bbox in enumerate(bboxes):
            selection[i, bbox[0]:bbox[0] + bbox[2] + 1, bbox[1]:bbox[1] + bbox[3] + 1] = 1
        return selection

    def _augmented_reward(self):
        rewards = []
        for g, ad, a in zip(self.grid[-1], self.answer_dim[-1], self.answer[-1]):
            dist = np.mean(g[:ad[0].item(), :ad[1].item()] != a[:ad[0], :ad[1]])
            rewards.append((dist == 0) - dist)
        return np.array(rewards)

    def run(self):
        lten = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
        ften = lambda x: torch.tensor(x, dtype=torch.float, device=self.device)
        with torch.no_grad():
            for step in trange(self.cfg.train.nsteps):
                # Given observations, get action value and neglopacs
                # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
                policy_inp = {key: lten(getattr(self, key)[-1]) for key in self.STATE_KEYS + self.INFO_KEYS}
                operation, bbox, neglogpac, vpred, rtm1_pred, rpred = self.act_fn(**policy_inp)
                self.rtm1_pred.append(fnpy(rtm1_pred))
                self.rpred.append(fnpy(rpred))
                self.vpred.append(fnpy(vpred))
                self.neglogpac.append(fnpy(neglogpac))
                self.bbox_pos.append(npy(bbox[:, :2]))
                self.bbox_dim.append(npy(bbox[:, 2:]))
                self.operation.append(npy(operation))
                selection = self._get_selection_from_bbox(npy(bbox))

                # Take actions in env and look the results
                # Infos contains a ton of useful informations
                (grid, grid_dim, selected, clip, clip_dim, terminated, trials_remain, active,
                 object, object_sel, object_dim, object_pos, background, rotation_parity, env_reward) = batch_act(
                    self.input[-1].astype(np.uint8), self.input_dim[-1], self.answer[-1].astype(np.uint8), self.answer_dim[-1],
                    self.grid[-1].astype(np.uint8), self.grid_dim[-1], self.selected[-1].astype(bool), self.clip[-1].astype(np.uint8), 
                    self.clip_dim[-1], self.terminated[-1], self.trials_remain[-1], 
                    self.active[-1], self.object[-1].astype(np.uint8), self.object_sel[-1].astype(np.uint8), self.object_dim[-1], 
                    self.object_pos[-1], self.background[-1].astype(np.uint8), self.rotation_parity[-1], 
                    selection.astype(bool), operation
                )
                # append states
                self.grid.append(grid); self.grid_dim.append(grid_dim); self.selected.append(selected)
                self.clip.append(clip); self.clip_dim.append(clip_dim); self.terminated.append(terminated)
                self.trials_remain.append(trials_remain); self.active.append(active); self.object.append(object)
                self.object_sel.append(object_sel); self.object_dim.append(object_dim); self.object_pos.append(object_pos)
                self.background.append(background); self.rotation_parity.append(rotation_parity)
                # append infos
                self.input.append(deepcopy(self.input[-1])); self.input_dim.append(deepcopy(self.input_dim[-1]))
                self.answer.append(deepcopy(self.answer[-1])); self.answer_dim.append(deepcopy(self.answer_dim[-1]))

                rewards = self._augmented_reward()

                self.timesteps += 1
                self.sum_rewards += rewards
                self.disc_sum_rewards = self.cfg.train.gamma * self.disc_sum_rewards + rewards
                self.ret_rms.update(self.disc_sum_rewards)
                
                # normalize reward
                self.reward.append(rewards)
                self.rtm1.append(rewards)

        policy_inp = {key: lten(getattr(self, key)[-1]) for key in self.STATE_KEYS + self.INFO_KEYS}
        _, _, _, nextvalues, _, _ = self.act_fn(**policy_inp)
        nextvalues = fnpy(nextvalues)

        for att in self.STATE_KEYS + self.INFO_KEYS + self.ACTION_KEYS + ["vpred", "neglogpac", "reward", "rpred", "rtm1", "rtm1_pred"]:
            if att in self.STATE_KEYS + self.INFO_KEYS + ["rtm1"]:
                setattr(self, att, np.stack(getattr(self, att)[:-1]))
            else:
                setattr(self, att, np.stack(getattr(self, att)))
        
        # normalize reward
        self.rew_rms.update(self.rtm1.reshape(-1))
        self.rtm1 = self.rew_rms.normalize(self.rtm1, use_mean=True)
        norm_rew =  self.rew_rms.normalize(self.reward, use_mean=True)
        self.reward = self.ret_rms.normalize(self.reward)

        # discount/bootstrap off value fn
        self.returns = np.zeros_like(self.reward)
        advs = np.zeros_like(self.reward)
        lastgaelam = 0
        for t in reversed(range(self.cfg.train.nsteps)):
            if t != self.cfg.train.nsteps - 1:
                nextvalues = self.vpred[t + 1]

            delta = self.reward[t] + self.cfg.train.gamma * nextvalues - self.vpred[t]
            advs[t] = lastgaelam = delta + self.cfg.train.gamma * self.cfg.train.lam * lastgaelam
        self.returns = advs + self.vpred

        ret_ob_acs = {key: lten(sf01(getattr(self, key))) for key in self.STATE_KEYS + self.INFO_KEYS + self.ACTION_KEYS}
        ret = (
            ret_ob_acs, 
            ften(sf01(self.returns)), 
            ften(sf01(self.vpred)), 
            ften(sf01(self.neglogpac)), 
            ften(sf01(self.rtm1)), 
            ften(sf01(self.rtm1_pred)), 
            ften(sf01(norm_rew)),
            ften(sf01(self.rpred)),
            list(self.sum_rewards))
        self.reset()

        return ret

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    shape = (s[0] * s[1], int(np.prod(s[2:]))) if int(np.prod(s[2:])) != 1 else (s[0] * s[1],)
    return arr.swapaxes(0, 1).reshape(*shape)


def flatten_and_copy(state):
    new_state = deepcopy(state)
    object_state = new_state.pop("object_states")
    return new_state | object_state

def fnpy(tensor):
    return tensor.detach().cpu().numpy().astype(float)

def npy(tensor):
    return tensor.detach().cpu().numpy().astype(int)

def unpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.uint8)
