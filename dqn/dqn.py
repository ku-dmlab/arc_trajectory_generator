import os
from dataclasses import dataclass

import numpy as np
import torch
from .dqn_critic import GPT_DQNCritic
from tqdm import tqdm
from omegaconf import OmegaConf

class HindsightExperienceReplayBuffer:
    def __init__(self, device, memory_size=1e5):
        super().__init__()
        self.max_mem_size = int(memory_size)
        self.counter = 0
        self.state_colors = []
        self.state_objects = []
        self.next_state_colors = []
        self.next_state_objects = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.goal_colors = []
        self.goal_objects = []
        self.device = device

    def add_experience(self, state_color, state_object, action, reward, next_state_color, next_state_object, done, goal_color, goal_object):
        if self.counter < self.max_mem_size:
            self.state_colors.append(state_color)
            self.state_objects.append(state_object)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_state_colors.append(next_state_color)
            self.next_state_objects.append(next_state_object)
            self.dones.append(done)
            self.goal_colors.append(goal_color)
            self.goal_objects.append(goal_object)
        else:
            i = self.counter % self.max_mem_size
            self.state_colors[i] = state_color
            self.state_objects[i] = state_object
            self.actions[i] = action
            self.rewards[i] = reward
            self.next_state_colors[i] = next_state_color
            self.next_state_objects[i] = next_state_object
            self.dones[i] = done
            self.goal_colors[i] = goal_color
            self.goal_objects[i] = goal_object
        self.counter += 1

    def get_random_experience(self, batch_size):

        Idx = np.random.choice(min(self.counter, self.max_mem_size), batch_size, replace=False)
        state_color = np.stack([self.state_colors[i] for i in Idx])
        state_object = np.stack([self.state_objects[i] for i in Idx])
        action = np.stack([self.actions[i] for i in Idx])
        reward = np.stack([self.rewards[i] for i in Idx])
        next_state_color = np.stack([self.next_state_colors[i] for i in Idx])
        next_state_object = np.stack([self.next_state_objects[i] for i in Idx])
        done = np.stack([self.dones[i] for i in Idx])
        goal_color = np.stack([self.goal_colors[i] for i in Idx])
        goal_object = np.stack([self.goal_objects[i] for i in Idx])

        rets = state_color, state_object, action, reward, next_state_color, next_state_object, done, goal_color, goal_object
        types = [torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.LongTensor, torch.cuda.FloatTensor,
                 torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor, torch.cuda.FloatTensor]

        return [torch.tensor(each, device=self.device).type(tp) for each, tp in zip(rets, types)]
@dataclass
class Transition:
    state_color: np.ndarray
    state_object: np.ndarray
    action: int
    reward: float
    next_state_color: np.ndarray
    next_state_object: np.ndarray
    
    def get_goals(self):
        return np.copy(self.next_state_color), np.copy(self.next_state_object)

    def get_hindsight_experience(self, env, goal_color, goal_object):
        reward, done = env.reward_done(self.next_state_color, goal_color)
        return done, (self.state_color, self.state_object, self.action, reward, 
                      self.next_state_color, self.next_state_object, done, goal_color, goal_object)

class DQN:
    def __init__(self, env, cfg, logger, device):
        self.device = device
        self.learning_rate = cfg.train.learning_rate
        self.n_actions = env.num_actions * cfg.env.grid_x * cfg.env.grid_y
        self.tau = cfg.train.tau
        self.gamma = cfg.train.gamma
        self.epsilon = cfg.train.epsilon
        self.batch_size = cfg.train.batch_size
        self.dec_epsilon = cfg.train.dec_epsilon
        self.min_epsilon = cfg.train.min_epsilon
        self.action_indices = [i for i in range(self.n_actions)]
        self.learn_steps_count = 0

        self.q_online = GPT_DQNCritic(cfg).to(self.device)
        self.q_target = GPT_DQNCritic(cfg).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.buffer = HindsightExperienceReplayBuffer(self.device)

        self.optimizer = self.q_online.configure_optimizers()
        self.loss = torch.nn.MSELoss()

        self.env = env
        self.cfg = cfg
        self.logger = logger

    def decrement_epsilon(self):
        self.epsilon = (self.epsilon - self.dec_epsilon 
                        if self.epsilon > self.min_epsilon else self.min_epsilon)

    def soft_update_target(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_online.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def log_action(self, action, timestep):
        pixel = action // self.env.num_actions
        is_obj = action % self.env.num_actions // (self.env.num_actions // 2)
        row, col = pixel // self.cfg.env.grid_x, pixel % self.cfg.env.grid_y

        action_arg = action % (self.env.num_actions // 2)
        is_translate = action_arg < self.env.num_directions
        action_arg = action_arg if is_translate else action_arg - self.env.num_directions

        self.logger.log(
            timestep,
            {
                "greedy_action/pixel_row": row,
                "greedy_action/pixel_col": col,
                "greedy_action/is_obj": is_obj,
                "greedy_action/is_translate": is_translate,
                "greedy_action/action_arg": action_arg,
            },
        )

    @torch.no_grad()
    def greedy_action(self, state_color, state_object, goal_color, goal_object):
        state_color = torch.tensor(state_color[None], dtype=torch.float).to(self.device)
        state_object = torch.tensor(state_object[None], dtype=torch.float).to(self.device)
        goal_color = torch.tensor(goal_color[None], dtype=torch.float).to(self.device)
        goal_object = torch.tensor(goal_object[None], dtype=torch.float).to(self.device)
        actions = self.q_online.forward(state_color, state_object, goal_color, goal_object)
        return torch.argmax(actions).item()

    def _rollout(self, timestep):
        state_color, state_object = self.env.reset()
        goal_color = np.ravel(self.env.goal)
        goal_object = np.ravel(self.env.goal_obj)
        done = False
        transitions = []
        ep_return = 0.0

        for t in range(self.cfg.train.max_ep_len):
            if np.random.random() > self.epsilon:
                action = self.greedy_action(state_color, state_object, goal_color, goal_object)
                self.log_action(action, timestep + t)
            else:
                action = np.random.choice(self.n_actions)
            self.decrement_epsilon()

            next_state_color, next_state_object, reward, done = self.env.step(action)
            ep_return += float(reward)

            self.buffer.add_experience(
                state_color, state_object, action, reward,
                next_state_color, next_state_object, done, goal_color, goal_object)
            transitions.append(Transition(
                state_color, state_object, action, reward, 
                next_state_color, next_state_object))

            self._train(timestep + t)

            state_color, state_object = next_state_color, next_state_object

            if done:
                print("success!")
                break
        return t + 1, ep_return, transitions, done

    @torch.no_grad()
    def _eval(self, global_t):

        ep_lens = []
        ep_returns = []
        for _ in range(self.cfg.logging.num_evaluations):
            state_color, state_object = self.env.reset(is_train=False)
            goal_color = np.ravel(self.env.goal)
            goal_object = np.ravel(self.env.goal_obj)
            done = False
            ep_return = 0.0

            eval_actions = {}
            for t in range(self.cfg.train.max_ep_len):
                action = self.greedy_action(state_color, state_object, goal_color, goal_object)
                next_state_color, next_state_object, reward, done = self.env.step(action)
                ep_return += float(reward)
                eval_actions[f"eval/eval_action_{t}"] = action
                if done:
                    break
                state_color, state_object = next_state_color, next_state_object
            ep_lens.append(t)
            ep_returns.append(ep_return)

        self.logger.log(
            global_t,
            {
                "eval/ep_len": np.mean(ep_lens),
                "eval/ep_return": np.mean(ep_returns),
            } #| eval_actions
        )

    def _add_hindsight_experiences(self, transitions, global_t):
        goal_color, goal_object = transitions[-1].get_goals()
        assert not np.array_equal(goal_color, np.ravel(self.env.goal))
        for t, transition in enumerate(transitions):
            done, experience = transition.get_hindsight_experience(self.env, goal_color, goal_object)
            self.buffer.add_experience(*experience)
            self._train(global_t + t)
            if done:
                break
        return t

    def learn(self, checkpoint_dir):

        t = 0
        t_logged = 0
        pbar = tqdm(total=self.cfg.train.total_timesteps)
        while t < self.cfg.train.total_timesteps:
            len_episode, ep_ret, transitions, success = self._rollout(t)
            self.logger.log(
                t,
                {
                    "train/success": success,
                    "train/epsilon": self.epsilon,
                    "train/ep_ret": ep_ret
                }
            )
            t += len_episode
            pbar.update(len_episode)
            if not success:
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
        torch.save(self.q_online.state_dict(), os.path.join(direc, "qnet.pt"))


    def _train(self, t):
        if self.buffer.counter < self.batch_size:
            return
        self.optimizer.zero_grad()

        (state_color, state_object, action, reward, next_state_color, next_state_object, 
         done, goal_color, goal_object) = self.buffer.get_random_experience(self.batch_size)
        # Gets the evenly spaced batches

        q_pred = self.q_online.forward(state_color, state_object, goal_color, goal_object)
        q_pred = torch.gather(q_pred, 1, action.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_pred_next = self.q_online.forward(next_state_color, next_state_object, goal_color, goal_object)
            next_actions = torch.argmax(q_pred_next, dim=-1, keepdims=True)
            q_next = self.q_target.forward(next_state_color, next_state_object, goal_color, goal_object)
            q_next = torch.gather(q_next, 1, next_actions).squeeze()

        q_target = reward + self.gamma * q_next * (1 - done)

        # Computes loss and performs backpropagation
        loss = self.loss(q_target, q_pred)

        self.logger.log(
            t,
            {
                "target/mean": torch.mean(q_target).item(),
                "target/min": torch.min(q_target).item(),
                "target/max": torch.max(q_target).item(),
            },
        )
        self.logger.log(
            t,
            {
                "online/mean": torch.mean(q_pred).item(),
                "online/min": torch.min(q_pred).item(),
                "online/max": torch.max(q_pred).item(),
            },
        )
        self.logger.log(
            t,
            {
                "loss": loss.item(),
            },
        )

        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update_target()
