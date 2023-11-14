import os
from copy import deepcopy

import numpy as np
import torch
from .dqn_critic import GPT_DQNCritic
from tqdm import tqdm
from omegaconf import OmegaConf

class HindsightExperienceReplayBuffer:
    GRIDS = [
        "input", "answer", "next_input",
        "grid", "next_grid",
        "selected", "next_selected",
        "clip", "next_clip",
        "object", "next_object",
        "object_sel", "next_object_sel",
        "background", "next_background",
        "bbox"
        ]
    TUPLES = [
        "input_dim", "answer_dim", "next_input_dim",
        "grid_dim", "next_grid_dim",
        "clip_dim", "next_clip_dim",
        "object_dim", "next_object_dim",
        "object_pos", "next_object_pos"
    ]
    NUMBERS = [
        "terminated", "next_terminated",
        "trials_remain", "next_trials_remain",
        "active", "next_active",
        "rotation_parity", "next_rotation_parity",
        "operation", "reward", "done"
    ]
    INFO_KEYS = ["input", "input_dim", "answer", "answer_dim"]
    STATE_KEYS = ["grid", "grid_dim", "selected", "clip", "clip_dim",
                  "terminated", "trials_remain", "active",
                  "object", "object_sel", "object_dim", "object_pos", 
                  "background","rotation_parity"]
    NEXT_STATE_KEYS = [f"next_{key}" for key in STATE_KEYS]
    ACTION_KEYS = ["operation", "bbox"]

    def __init__(self, cfg, device, memory_size=1e5):
        super().__init__()
        self.cfg = cfg
        self.max_mem_size = int(memory_size)
        self.counter = 0
        self.device = device

        for att in self.GRIDS + self.TUPLES + self.NUMBERS:
            setattr(self, att, [])
        self.att_set = set(self.GRIDS + self.TUPLES + self.NUMBERS)


    def add_experience(self, **kwargs):
        assert not self.att_set - set(kwargs.keys())
        if self.counter < self.max_mem_size:
            for key, value in kwargs.items():
                getattr(self, key).append(value)
        else:
            i = self.counter % self.max_mem_size
            for key, value in kwargs.items():
                getattr(self, key)[i] = value
        self.counter += 1
        return self.counter % self.max_mem_size - 1

    def get_random_experience(self, batch_size):
        Idx = np.random.choice(min(self.counter, self.max_mem_size), batch_size, replace=False)
        rets = {}
        for key in self.att_set:
            value = getattr(self, key)
            rets[key] = np.stack([value[i] for i in Idx])
            rets[key] = torch.tensor(rets[key], device=self.device).type(self._get_tensor_type(key))
        return rets

    def _get_tensor_type(self, att_name):
        return torch.cuda.FloatTensor if att_name == "bbox" else torch.cuda.LongTensor
    
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

    NUM_OPERATIONS = 35

    def __init__(self, env, cfg, logger, device):
        self.device = device
        self.learning_rate = cfg.train.learning_rate
        self.tau = cfg.train.tau
        self.gamma = cfg.train.gamma
        self.epsilon = cfg.train.epsilon
        self.batch_size = cfg.train.batch_size
        self.dec_epsilon = cfg.train.dec_epsilon
        self.min_epsilon = cfg.train.min_epsilon

        self.q_online_1 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_online_2 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_target_1 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_target_2 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_target_1.load_state_dict(self.q_online_1.state_dict())
        self.q_target_2.load_state_dict(self.q_online_2.state_dict())
        self.buffer = HindsightExperienceReplayBuffer(cfg, self.device)

        self.optimizer_1 = self.q_online_1.configure_optimizers()
        self.optimizer_2 = self.q_online_2.configure_optimizers()
        self.loss = torch.nn.MSELoss()

        self.env = env
        self.cfg = cfg
        self.logger = logger

        self.target_entropy = -4
        self.log_alpha = torch.full([], -5.0, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon else self.min_epsilon
        )

    def soft_update_target(self):
        for target_param, param in zip(self.q_target_1.parameters(), self.q_online_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.q_target_2.parameters(), self.q_online_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def log_action(self, action, timestep, tag="train"):
        self.logger.log(
            timestep,
            {f"{tag}_action": action}
        )

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
        selection[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = 1
        return selection

    def _get_random_action(self):
        operation = np.random.randint(35)
        start_x = float(np.random.randint(self.cfg.env.grid_x))
        start_y = float(np.random.randint(self.cfg.env.grid_y))
        width = float(np.random.randint(self.cfg.env.grid_x - start_x))
        height = float(np.random.randint(self.cfg.env.grid_y - start_y))
        return operation, [start_x, start_y, width, height]

    def _rollout(self, timestep):
        raw_state, info = self.env.reset()
        state = self.flatten_and_copy(raw_state)
        done = False
        transitions = []
        ep_return = 0.0

        for t in range(self.cfg.train.max_ep_len):
            if np.random.random() > self.epsilon:
                q_input = self.buffer.get_tensor_dict(**(state | info))
                with torch.no_grad():
                    operation, bbox = self.q_online_1.act(**q_input, deterministic=False)
                    operation = operation[0].detach().cpu().numpy()
                    bbox = bbox[0].detach().cpu().numpy()
                self.log_action(operation, timestep + t, tag="train")
            else:
                operation, bbox = self._get_random_action()
            action = {"operation": operation, "selection": self._get_selection_from_bbox(bbox)}
            self.decrement_epsilon()

            raw_next_state, reward, done, _, _ = self.env.step(action)
            ep_return += float(reward)

            next_state = self.flatten_and_copy(raw_next_state)
            modified_next_state = {f"next_{key}":value for key, value in next_state.items()}
            exp_ind = self.buffer.add_experience(
                **{"reward": reward, "done": done, "operation": operation, "bbox": bbox}
                  | state | modified_next_state | info)
            transitions.append(exp_ind)

            self._train(timestep + t)
            state = next_state
            if done:
                break
        return t + 1, ep_return, transitions, done

    @torch.no_grad()
    def _eval(self, global_t):
        ep_lens = []
        ep_returns = []
        for _ in range(self.cfg.logging.num_evaluations):
            raw_state, info = self.env.reset()
            state = self.flatten_and_copy(raw_state)
            done = False
            ep_return = 0.0

            for t in range(self.cfg.train.max_ep_len):
                q_input = self.buffer.get_tensor_dict(**state | info)
                with torch.no_grad():
                    operation, bbox = self.q_online_1.act(**q_input, deterministic=True)
                    operation = operation[0].detach().cpu().numpy()
                    bbox = bbox[0].detach().cpu().numpy()
                self.log_action(operation, global_t + sum(ep_lens) + t, tag="train")
                action = {"operation": operation, "selection": self._get_selection_from_bbox(bbox)}

                raw_next_state, reward, done, _, _ = self.env.step(action)
                ep_return += float(reward)
                if done:
                    break
                state = self.flatten_and_copy(raw_next_state)
            ep_lens.append(t)
            ep_returns.append(ep_return)

        self.logger.log(
            global_t,
            {
                "eval/ep_len": np.mean(ep_lens),
                "eval/ep_return": np.mean(ep_returns),
            }
        )

    def _add_hindsight_experiences(self, transitions, global_t):
        answer = self.buffer.next_grid[transitions[-1]]
        answer_dim = self.buffer.next_grid_dim[transitions[-1]]
        for t, i in enumerate(transitions):
            experience = {key: getattr(self.buffer, key)[i] for key in self.buffer.att_set}
            experience["answer"] = answer
            experience["answer_dim"] = answer_dim
            
            self.buffer.add_experience(**experience)
            self._train(global_t + t)

            if experience["grid_dim"] == answer_dim and np.all(
                experience["grid"][:answer_dim[0], :answer_dim[1]] == 
                answer[:answer_dim[0], :answer_dim[1]]):

                next_state = {key: value for key, value in experience.items() if "next_" in key}
                state = {key[5:]: value for key, value in next_state.items()}
                new_exp = experience | state | next_state | {"reward":1, "done":True, "operation": 34}
                self.buffer.add_experience(**new_exp)
                self._train(global_t + t)
                break

        return t

    def learn(self, checkpoint_dir):
        t = 0
        t_logged = 0
        pbar = tqdm(total=self.cfg.train.total_timesteps)
        while t < self.cfg.train.total_timesteps:
            len_episode, ep_ret, transitions, done = self._rollout(t)
            self.logger.log(
                t, {"train/done": done, "train/epsilon": self.epsilon, "train/ep_ret": ep_ret}
            )
            t += len_episode
            pbar.update(len_episode)
            if not ep_ret:
                num_update = self._add_hindsight_experiences(transitions, t)
                t += num_update
                pbar.update(num_update)

            if t - t_logged >= self.cfg.logging.log_interval:
                t_logged = t
                self._eval(t)

                if checkpoint_dir is not None:
                    self.save(os.path.join(checkpoint_dir, f"checkpoint_{t}"))
        pbar.close()

    def save(self, direc):
        os.makedirs(direc, exist_ok=True)
        OmegaConf.save(self.cfg, os.path.join(direc, "config.yaml"))
        torch.save(self.q_online_1.state_dict(), os.path.join(direc, "qnet.pt"))

    def _repeat_first_dim_dict(self, dict, repeat_num):
        ret = {}
        for key in dict:
            ret[key] = dict[key].repeat(*[repeat_num] + [1] * (dict[key].dim - 1))

    def _train(self, t):
        if self.buffer.counter < self.batch_size:
            return
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()

        exp = self.buffer.get_random_experience(self.batch_size)

        alpha = self.log_alpha.exp()

        op, bbox, op_log_pi, bbox_log_pi, all_op_pi = self.q_online_1.action_and_log_pi(**{key: exp[key] for key in self.buffer.STATE_KEYS + self.buffer.INFO_KEYS})

        states = {key: exp[key] for key in self.buffer.STATE_KEYS}
        infos = {key: exp[key] for key in self.buffer.INFO_KEYS}
        next_states = {key[5:]: exp[key] for key in self.buffer.NEXT_STATE_KEYS}

        q_values, values = self.q_target_1.value(**states | infos, operation=op.detach(), bbox=bbox, pi=all_op_pi)

        policy_loss = op_log_pi * (values - q_values + alpha * (op_log_pi + bbox_log_pi)).detach()
        policy_loss += alpha * (op_log_pi + bbox_log_pi) - q_values
        policy_loss = policy_loss.mean()


        q_value_1 = self.q_online_1.value(**{key: exp[key] for key in self.buffer.STATE_KEYS + self.buffer.INFO_KEYS + self.buffer.ACTION_KEYS})
        q_value_2 = self.q_online_2.value(**{key: exp[key] for key in self.buffer.STATE_KEYS + self.buffer.INFO_KEYS + self.buffer.ACTION_KEYS})

        with torch.no_grad():
            next_op, next_bbox, next_op_log_pi, next_bbox_log_pi, _ = self.q_online_1.action_and_log_pi(
                **{key: exp[key] for key in self.buffer.INFO_KEYS}, **{key[5:]: exp[key] for key in self.buffer.NEXT_STATE_KEYS})
            
            next_q1 = self.q_target_1.value(** next_states | infos, operation=next_op, bbox=next_bbox)
            next_q2 = self.q_target_2.value(** next_states | infos, operation=next_op, bbox=next_bbox)
            
            target = exp["reward"] + self.gamma * (1 - exp["done"]) * (torch.minimum(next_q1, next_q2) - alpha * (next_op_log_pi + next_bbox_log_pi))
            
        critic_loss = self.loss(target.detach(), q_value_1) + self.loss(target.detach(), q_value_2)

        loss = policy_loss + critic_loss

        self.alpha_optim.zero_grad()
        alpha_loss = (self.log_alpha * (self.target_entropy - op_log_pi - bbox_log_pi).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.logger.log(
            t,
            {
                "policy/q_value_1": torch.mean(q_value_1).item(),
                "policy/q_value_1": torch.min(q_value_1).item(),
                "policy/q_value_1": torch.max(q_value_1).item(),
                "target/mean": torch.mean(target).item(),
                "target/min": torch.min(target).item(),
                "target/max": torch.max(target).item(),
                "online/alpha": alpha.item(),
                "loss/loss": loss.item(),
                "loss/policy_loss": policy_loss.item(),
                "loss/critic_loss": critic_loss.item(),
                "loss/alpha_loss": alpha_loss.item(),
            },
        )

        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.q_online_1.parameters(), 10.0)
        torch.nn.utils.clip_grad.clip_grad_norm_(self.q_online_2.parameters(), 10.0)
        self.optimizer_1.step()
        self.optimizer_2.step()
        self.soft_update_target()
