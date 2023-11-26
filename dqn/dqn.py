import os
from copy import deepcopy

import wandb
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

    def __init__(self, env, cfg, device):
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
        self.q_online_1 = torch.compile(self.q_online_1)
        self.q_online_2 = torch.compile(self.q_online_2)
        self.q_target_1 = torch.compile(self.q_target_1)
        self.q_target_2 = torch.compile(self.q_target_2)
        self.q_target_1.load_state_dict(self.q_online_1.state_dict())
        self.q_target_2.load_state_dict(self.q_online_2.state_dict())
        self.buffer = HindsightExperienceReplayBuffer(cfg, self.device)

        self.optimizer_1 = self.q_online_1.configure_optimizers()
        self.optimizer_2 = self.q_online_2.configure_optimizers()
        self.loss = torch.nn.MSELoss()
        self.aux_loss = torch.nn.MSELoss()

        self.env = env
        self.cfg = cfg

        self.bbox_target_entropy = -4
        self.op_target_entropy = 0.3
        self.log_alpha_bbox = torch.full([], -1.0, requires_grad=True, device=self.device)
        self.log_alpha_op = torch.full([], -1.0, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha_bbox, self.log_alpha_op], lr=self.learning_rate)

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon else self.min_epsilon
        )

    def soft_update_target(self):
        for target_param, param in zip(self.q_target_1.parameters(), self.q_online_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.q_target_2.parameters(), self.q_online_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

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

    def _get_random_action(self):
        operation = np.random.randint(self.cfg.env.num_actions)
        # start_x = float(np.random.randint(self.cfg.env.grid_x))
        # start_y = float(np.random.randint(self.cfg.env.grid_y))
        # width = float(np.random.randint(self.cfg.env.grid_x - start_x))
        # height = float(np.random.randint(self.cfg.env.grid_y - start_y))
        start_x = np.random.rand() * self.cfg.env.grid_x
        start_y = np.random.rand() * self.cfg.env.grid_y
        width = np.random.rand() * self.cfg.env.grid_x
        height = np.random.rand() * self.cfg.env.grid_y
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
                wandb.log(
                    {"rollout/operation": operation,
                    "rollout/bbox_0": bbox[0],
                    "rollout/bbox_1": bbox[1],
                    "rollout/bbox_2": bbox[2],
                    "rollout/bbox_3": bbox[3],
                    "rollout/rbbox_0": np.floor(np.clip(bbox[0], 0, self.cfg.env.grid_x)),
                    "rollout/rbbox_1": np.floor(np.clip(bbox[1], 0, self.cfg.env.grid_x)),
                    "rollout/rbbox_2": np.floor(np.clip(bbox[2], 0, self.cfg.env.grid_x)),
                    "rollout/rbbox_3": np.floor(np.clip(bbox[3], 0, self.cfg.env.grid_x)),
                    "train_step": timestep + t}
                )
            else:
                operation, bbox = self._get_random_action()
            action = {"operation": 34 if operation == 2 else operation, "selection": self._get_selection_from_bbox(bbox)}
            self.decrement_epsilon()

            raw_next_state, reward, done, _, _ = self.env.step(action)
            # reward = reward - ((raw_next_state["grid"] - info["answer"]) ** 2).mean() * 0.01
            # if done and reward == 0:
            #     reward = -1
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
        return t + 1, ep_return, transitions, done, t

    @torch.no_grad()
    def _eval(self, eval_step):
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
                wandb.log(
                    {"eval/operation": operation,
                    "eval/bbox_0": bbox[0],
                    "eval/bbox_1": bbox[1],
                    "eval/bbox_2": bbox[2],
                    "eval/bbox_3": bbox[3],
                    "eval_step": eval_step + sum(ep_lens) + t}
                )
                action = {"operation": 34 if operation == 2 else operation, "selection": self._get_selection_from_bbox(bbox)}

                raw_next_state, reward, done, _, _ = self.env.step(action)
                ep_return += float(reward)
                if done:
                    break
                state = self.flatten_and_copy(raw_next_state)
            ep_lens.append(t)
            ep_returns.append(ep_return)

        wandb.log(
            {
                "eval/ep_len": np.mean(ep_lens),
                "eval/ep_return": np.mean(ep_returns),
                "eval_step": eval_step + sum(ep_lens)
            }
        )
        return eval_step + sum(ep_lens)

    def _add_hindsight_experiences(self, transitions, global_t):
        answer = self.buffer.next_grid[transitions[-1]]
        answer_dim = self.buffer.next_grid_dim[transitions[-1]]
        for t, i in enumerate(transitions):
            experience = {key: getattr(self.buffer, key)[i] for key in self.buffer.att_set}
            experience["answer"] = answer
            experience["answer_dim"] = answer_dim
            
            self.buffer.add_experience(**experience)
            #self._train(global_t + t)

            if experience["next_grid_dim"] == answer_dim and np.all(
                experience["next_grid"][:answer_dim[0], :answer_dim[1]] == 
                answer[:answer_dim[0], :answer_dim[1]]):

                next_state = {key: value for key, value in experience.items() if "next_" in key}
                state = {key[5:]: value for key, value in next_state.items()}
                new_exp = experience | state | next_state | {"reward":1, "done":True, "operation": 2}
                self.buffer.add_experience(**new_exp)
                #self._train(global_t + t)
                break

        return t

    def learn(self, checkpoint_dir):
        eval_step = 0
        t = 0
        t_logged = 0
        pbar = tqdm(total=self.cfg.train.total_timesteps)
        while t < self.cfg.train.total_timesteps:
            len_episode, ep_ret, transitions, done, ep_len = self._rollout(t)
            wandb.log(
                {"rollout/done": done, 
                 "rollout/epsilon": self.epsilon, 
                 "rollout/ep_ret": ep_ret, 
                 "rollout/ep_len": ep_len,
                 "train_step": t + len_episode - 1})
            t += len_episode
            pbar.update(len_episode)
            if not ep_ret == 1:
                self._add_hindsight_experiences(transitions, t)

            if t - t_logged >= self.cfg.logging.log_interval:
                t_logged = t
                eval_step = self._eval(eval_step) + 100

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

        alpha_bbox = self.log_alpha_bbox.exp()
        alpha_op = self.log_alpha_op.exp()

        op, bbox, op_log_pi, bbox_log_pi, all_op_pi = self.q_online_1.action_and_log_pi(
            **{key: exp[key] for key in self.buffer.STATE_KEYS + self.buffer.INFO_KEYS})

        states = {key: exp[key] for key in self.buffer.STATE_KEYS}
        infos = {key: exp[key] for key in self.buffer.INFO_KEYS}
        actions = {key: exp[key] for key in self.buffer.ACTION_KEYS}
        next_states = {key[5:]: exp[key] for key in self.buffer.NEXT_STATE_KEYS}

        q_values, values, all_qvalues = self.q_online_1.value(**states | infos, operation=op.detach(), bbox=bbox, pi=all_op_pi)
        detached_q, _, _ = self.q_online_1.value(**states | infos, operation=op.detach(), bbox=bbox.detach(), pi=all_op_pi.detach())
        # To prevent gradient that directly increment Q

        op_loss = (op_log_pi * (values - q_values).detach()).mean()
        bbox_loss = (alpha_op * op_log_pi + alpha_bbox * bbox_log_pi - q_values + detached_q).mean()
        policy_loss = op_loss + bbox_loss


        q_value_1, cur_aux_1, cur_dist = self.q_online_1.value(**states | infos | actions)
        q_value_2, cur_aux_2, _ = self.q_online_2.value(**states | infos | actions)

        with torch.no_grad():
            next_op, next_bbox, next_op_log_pi, next_bbox_log_pi, _ = self.q_online_1.action_and_log_pi(** next_states | infos)
            
            next_q1, next_aux_1, next_dist = self.q_target_1.value(** next_states | infos, operation=next_op, bbox=next_bbox)
            next_q2, next_aux_2, _ = self.q_target_2.value(** next_states | infos, operation=next_op, bbox=next_bbox)
            
            target = exp["reward"] + self.gamma * (1 - exp["done"]) * (
                torch.minimum(next_q1, next_q2) - alpha_op * next_op_log_pi - alpha_bbox * next_bbox_log_pi)
            
        critic_loss = self.loss(target.detach(), q_value_1) + self.loss(target.detach(), q_value_2)

        aux_loss = (self.aux_loss(cur_aux_1, cur_dist) 
                    + self.aux_loss(cur_aux_2, cur_dist) 
                    + self.aux_loss(next_aux_1, next_dist) 
                    + self.aux_loss(next_aux_2, next_dist))

        loss = policy_loss + critic_loss + aux_loss

        self.alpha_optim.zero_grad()
        alpha_loss = -(self.log_alpha_op * (self.op_target_entropy + op_log_pi).detach() 
                      + self.log_alpha_bbox * (self.bbox_target_entropy + bbox_log_pi).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()

        wandb.log(
            {
                "online/q_value_1_act_var": torch.var(all_qvalues, dim=1).mean().item(),
                "online/q_value_1_mean": torch.mean(q_value_1).item(),
                "online/q_value_1_min": torch.min(q_value_1).item(),
                "online/q_value_1_max": torch.max(q_value_1).item(),
                "train/alpha_bbox": alpha_bbox.item(),
                "train/alpha_op": alpha_op.item(),
                "target/mean": torch.mean(target).item(),
                "target/min": torch.min(target).item(),
                "target/max": torch.max(target).item(),
                "train/op_entropy": torch.mean(-op_log_pi).item(),
                "train/bbox_entropy": torch.mean(-bbox_log_pi).item(),
                "train/entropy": -torch.mean(op_log_pi + bbox_log_pi).item(),
                "loss/loss": loss.item(),
                "loss/bbox_loss": bbox_loss.item(),
                "loss/op_loss": op_loss.item(),
                "loss/policy_loss": policy_loss.item(),
                "loss/critic_loss": critic_loss.item(),
                "loss/alpha_loss": alpha_loss.item(),
                "loss/aux_loss": aux_loss.item(),
                "train_step": t
            }
        )

        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.q_online_1.parameters(), 1.0)
        torch.nn.utils.clip_grad.clip_grad_norm_(self.q_online_2.parameters(), 1.0)
        self.optimizer_1.step()
        self.optimizer_2.step()
        self.soft_update_target()
