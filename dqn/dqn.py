import os
from copy import deepcopy
from dataclasses import dataclass

import wandb
import numpy as np
import torch
from .dqn_critic import GPT_DQNCritic
from tqdm import tqdm, trange
from omegaconf import OmegaConf

@dataclass
class GCSL_ITEM:
    start: int
    end: int
    answer: np.ndarray
    answer_dim: tuple

class PPOBuffer:
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
        "operation", "reward", "done", "truncated",
        "log_prob", "state_val"
    ]
    INFO_KEYS = ["input", "input_dim", "answer", "answer_dim"]
    STATE_KEYS = ["grid", "grid_dim", "selected", "clip", "clip_dim",
                  "terminated", "trials_remain", "active",
                  "object", "object_sel", "object_dim", "object_pos", 
                  "background", "rotation_parity"]
    ACTION_KEYS = ["operation", "bbox_pos", "bbox_dim"]

    def __init__(self, cfg, device, policy):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.att_set = set(self.GRIDS + self.TUPLES + self.NUMBERS)
        self.policy = policy
        self.reset()

    def reset(self):
        for att in self.GRIDS + self.TUPLES + self.NUMBERS:
            setattr(self, att, [])
        self.gcsl_items = []

    def add_experience(self, **kwargs):
        assert not self.att_set - set(kwargs.keys()), str(self.att_set - set(kwargs.keys()))
        assert not set(kwargs.keys()) - self.att_set, str(set(kwargs.keys()) - self.att_set)
        for key, value in kwargs.items():
            getattr(self, key).append(value)

    def add_gcsl(self, traj_length):
        if len(self.grid) == 0:
            return
        answer = self.grid[-1]
        answer_dim = self.grid_dim[-1]
        for i, grid in enumerate(self.grid[-traj_length:]):
            if (grid[:answer_dim[0], :answer_dim[1]] == answer[:answer_dim[0], :answer_dim[1]]).all():
                break
        self.gcsl_items.append(GCSL_ITEM(len(self.grid) - traj_length, len(self.grid) - traj_length + i, answer, answer_dim))

    def get_experiences(self):
        rets = {}
        for item in self.gcsl_items:
            print(item.start, item.end, item.answer)
        rets["gcsl_answer"] = torch.tensor(
            np.stack(sum([[item.answer] * (item.end - item.start) for item in self.gcsl_items], [])), device=self.device).type(torch.cuda.LongTensor)
        rets["gcsl_answer_dim"] = torch.tensor(
            np.stack(sum([[item.answer_dim] * (item.end - item.start) for item in self.gcsl_items], [])), device=self.device).type(torch.cuda.LongTensor)
        rets["gcsl_ind"] = torch.tensor(sum([list(range(item.start, item.end)) for item in self.gcsl_items], []))
        assert len(rets["gcsl_ind"]) == len(rets["gcsl_answer_dim"])

        for key in self.att_set:
            value = getattr(self, key)
            if isinstance(value[0], torch.Tensor):
                rets[key] = torch.stack(value)
            else:
                rets[key] = torch.tensor(np.stack(value), device=self.device).type(self._get_tensor_type(key))

        rtgs = []
        discounted_reward = 0
        for i, (r, d, trun) in enumerate(zip(reversed(self.reward), reversed(self.done), reversed(self.truncated))):
            if d:
                discounted_reward = 0
            elif trun:
                _, state_values, _ = self.policy.evaluate(**{key: rets[key][i:i+1] for key in self.STATE_KEYS + self.INFO_KEYS + self.ACTION_KEYS})
                discounted_reward = state_values.item()
            else:
                discounted_reward = r + (self.cfg.train.gamma * discounted_reward)
            rtgs.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rets["rtgs"] = torch.tensor(rtgs, dtype=torch.float32).to(self.device)

        advs = rets["rtgs"].squeeze() - rets["state_val"].squeeze()
        rets["advs"] = (advs - advs.mean()) / (advs.std() + 1e-7)
        self.reset()
        return rets

    def _get_tensor_type(self, att_name):
        return torch.cuda.FloatTensor if att_name in ["logprobs", "state_val"] else torch.cuda.LongTensor
    
    def get_tensor_dict(self, **kwargs):
        ret = {}
        for key, value in kwargs.items():
            if not isinstance(value, np.ndarray):
                value = np.array([value])
            ret[key] = torch.tensor(value, device=self.device).type(self._get_tensor_type(key))
            if key in self.GRIDS:
                ret[key] = ret[key].reshape(-1, self.cfg.env.grid_x, self.cfg.env.grid_y)
            elif key in self.TUPLES:
                ret[key] = ret[key].reshape(-1, 2)
            else:
                ret[key] = ret[key].reshape(-1, 1)
        return ret

class DQN:

    def __init__(self, env, cfg, device):
        self.device = device
        self.learning_rate = cfg.train.learning_rate
        self.tau = cfg.train.tau
        self.gamma = cfg.train.gamma
        self.epsilon = cfg.train.epsilon
        self.batch_size = cfg.train.batch_size

        self.policy = GPT_DQNCritic(cfg, env).to(self.device)
        self.policy_old = GPT_DQNCritic(cfg, env).to(self.device)

        self.policy = torch.compile(self.policy)
        self.policy_old = torch.compile(self.policy_old)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = PPOBuffer(cfg, self.device, self.policy)

        self.optimizer = self.policy.configure_optimizers()
        self.loss = torch.nn.MSELoss()

        self.env = env
        self.cfg = cfg

    def flatten_and_copy(self, state):
        new_state = deepcopy(state)
        object_state = new_state.pop("object_states")
        return new_state | object_state

    def _get_selection_from_bbox(self, bbox):
        selection = np.zeros((self.cfg.env.grid_x, self.cfg.env.grid_y))
        bbox[0] = np.clip(bbox[0], 0, self.cfg.env.grid_x)
        bbox[1] = np.clip(bbox[1], 0, self.cfg.env.grid_x)
        bbox[2] = np.clip(bbox[2], 0, self.cfg.env.grid_y)
        bbox[3] = np.clip(bbox[3], 0, self.cfg.env.grid_y)
        bbox = np.floor(bbox).astype(int)
        selection[bbox[0]:bbox[0] + bbox[2] + 1, bbox[1]:bbox[1] + bbox[3] + 1] = 1
        return selection

    def learn(self, checkpoint_dir):
        i_episode = 0

        # training loop
        pbar = tqdm(total=self.cfg.train.total_timesteps)
        total_step = 0
        while total_step <= self.cfg.train.total_timesteps:

            raw_state, info = self.env.reset()
            state = self.flatten_and_copy(raw_state)
            ep_ret = 0.0
            for t in range(1, self.cfg.train.max_ep_len + 1):
                total_step += 1
                pbar.update(1)

                q_input = self.buffer.get_tensor_dict(**(state | info))
                with torch.no_grad():
                    operation, bbox, log_prob, state_val = self.policy_old.act(**q_input)
                    cpu_op = operation[0].detach().cpu().numpy()
                    cpu_bbox = bbox.detach().cpu().numpy()
                if t % 10 == 0:
                    wandb.log(
                        {"rollout/operation": cpu_op,
                        "rollout/bbox_0": cpu_bbox[0],
                        "rollout/bbox_1": cpu_bbox[1],
                        "rollout/bbox_2": cpu_bbox[2],
                        "rollout/bbox_3": cpu_bbox[3],
                        "train_step": total_step}
                    )

                action = {"operation": cpu_op, "selection": self._get_selection_from_bbox(cpu_bbox)}
                raw_next_state, reward, done, _, _ = self.env.step(action)

                # if done and reward == 0:
                #     reward = reward - 1
                done = False
                truncated = (t == self.cfg.train.max_ep_len)
                reward = 0
                dist = np.mean(state["grid"][:info["answer_dim"][0], :info["answer_dim"][1]] != info["answer"][:info["answer_dim"][0], :info["answer_dim"][1]])
                reward = reward - 0.01 * dist
                reward = reward + (dist == 0)
                # saving reward and is_terminals
                action = {"operation": operation, "bbox_pos": bbox[:2], "bbox_dim": bbox[2:]}
                etc = {"log_prob": log_prob, "state_val": state_val, "reward": reward, "done": done, "truncated": truncated}
                self.buffer.add_experience(**state | info | action | etc)
                ep_ret += reward

                if truncated:
                    self.buffer.add_gcsl(t)
                # update PPO agent
                if total_step % self.cfg.train.update_timestep == 0:
                    self._train(total_step)

                # log in logging file
                if total_step % self.cfg.logging.log_interval == 0:
                    if checkpoint_dir is not None:
                        self.save(os.path.join(checkpoint_dir, f"checkpoint_{t}"))
                if done:
                    break
                state = self.flatten_and_copy(raw_next_state)


            i_episode += 1
            wandb.log(
                {"rollout/done": done,
                "rollout/ep_ret": ep_ret,
                "rollout/ep_len": t,
                "i_episode": i_episode})


    def save(self, direc):
        os.makedirs(direc, exist_ok=True)
        OmegaConf.save(self.cfg, os.path.join(direc, "config.yaml"))
        torch.save(self.policy.state_dict(), os.path.join(direc, "policy.pt"))

    def _train(self, t):

        exp = self.buffer.get_experiences()

        states = {key: exp[key] for key in self.buffer.STATE_KEYS}
        infos = {key: exp[key] for key in self.buffer.INFO_KEYS}
        actions = {key: exp[key] for key in self.buffer.ACTION_KEYS}

        n_steps = (self.cfg.train.update_timestep // self.cfg.train.batch_size) * self.cfg.train.nepochs
        for i in trange(n_steps):
            ind = torch.randint(self.cfg.train.update_timestep, size=(self.cfg.train.batch_size,))
            gcsl_ind_batch = torch.randint(len(exp["gcsl_ind"]), size=(self.cfg.train.batch_size,))
            gcsl_ind = exp["gcsl_ind"][gcsl_ind_batch]

            policy_inp = states | infos | actions
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                **{key: policy_inp[key][ind] for key in policy_inp})

            relabeled_inp = states | infos | actions
            relabeled_inp = {key: relabeled_inp[key][gcsl_ind] for key in relabeled_inp}
            relabeled_inp = relabeled_inp | {
                    "answer": exp["gcsl_answer"][gcsl_ind_batch],
                    "answer_dim": exp["gcsl_answer_dim"][gcsl_ind_batch]}

            relabeled_logprob, _, _ = self.policy.evaluate(**relabeled_inp)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - exp["log_prob"][ind].detach().squeeze())
            surr1 = ratios * exp["advs"][ind].detach()
            surr2 = torch.clamp(ratios, 0.9, 1.1) * exp["advs"][ind].detach()
            
            actor_loss = (-torch.min(surr1, surr2))


            vpred = state_values
            vpredclipped = exp["state_val"][ind].squeeze().detach() + torch.clamp(state_values - exp["state_val"][ind].squeeze().detach(), -0.1, 0.1)

            vf_loss1 = self.loss(vpred, exp["rtgs"][ind].detach())
            vf_loss2 = self.loss(vpredclipped, exp["rtgs"][ind].detach())

            #critic_loss = (0.5 * self.loss(state_values, exp["rtgs"][ind].detach()))
            critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2) * 10
            entropy_loss = - 0.01 * dist_entropy
            relabeled_loss = - relabeled_logprob * 0.1

            loss = actor_loss.mean() + critic_loss.mean() + entropy_loss.mean() #+ relabeled_loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            if i % 10 == 0:
                wandb.log(
                    {
                        "online/ratios_mean": torch.mean(ratios).item(),
                        "online/ratios_min": torch.min(ratios).item(),
                        "online/ratios_max": torch.max(ratios).item(),
                        "online/logprobs_mean": torch.mean(logprobs).item(),
                        "online/logprobs_min": torch.min(logprobs).item(),
                        "online/logprobs_max": torch.max(logprobs).item(),
                        "online/state_values_mean": torch.mean(state_values).item(),
                        "online/state_values_min": torch.min(state_values).item(),
                        "online/state_values_max": torch.max(state_values).item(),
                        "loss/loss": loss.item(),
                        "loss/actor_loss": actor_loss.mean().item(),
                        "loss/critic_loss": critic_loss.mean().item(),
                        "loss/entropy_loss": entropy_loss.mean().item(),
                        "loss/relabeled_loss": relabeled_loss.mean().item(),
                        "grad_step": t + i
                    }
                )

        self.policy_old.load_state_dict(self.policy.state_dict())

        wandb.log(
            {
                "online/rtgs_mean": torch.mean(exp["rtgs"]).item(),
                "online/rtgs_min": torch.min(exp["rtgs"]).item(),
                "online/rtgs_max": torch.max(exp["rtgs"]).item(),
                "online/adv_mean": torch.mean(exp["advs"]).item(),
                "online/adv_min": torch.min(exp["advs"]).item(),
                "online/adv_max": torch.max(exp["advs"]).item(),
                "online/old_logprobs_mean": torch.mean(exp["log_prob"]).item(),
                "online/old_logprobs_min": torch.min(exp["log_prob"]).item(),
                "online/old_logprobs_max": torch.max(exp["log_prob"]).item(),
                "online/old_state_values_mean": torch.mean(exp["state_val"]).item(),
                "online/old_state_values_min": torch.min(exp["state_val"]).item(),
                "online/old_state_values_max": torch.max(exp["state_val"]).item(),
                "train_step": t
            }
        )
