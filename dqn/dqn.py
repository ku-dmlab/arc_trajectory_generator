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
        self.device = device

    def add_experience(
        self,
        state_color,
        state_object,
        action,
        reward,
        next_state_color,
        next_state_object,
        done,
        goal_color
    ):
        if self.counter < self.max_mem_size:
            self.state_colors.append(state_color)
            self.state_objects.append(state_object)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_state_colors.append(next_state_color)
            self.next_state_objects.append(next_state_object)
            self.dones.append(done)
            self.goal_colors.append(goal_color)
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

        rets = (
            state_color,
            state_object,
            action,
            reward,
            next_state_color,
            next_state_object,
            done,
            goal_color
        )
        types = [
            torch.cuda.FloatTensor,
            torch.cuda.FloatTensor,
            torch.cuda.LongTensor,
            torch.cuda.FloatTensor,
            torch.cuda.FloatTensor,
            torch.cuda.FloatTensor,
            torch.cuda.FloatTensor,
            torch.cuda.FloatTensor
        ]

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
        reward, done = env.reward_done(self.state_color, self.action, self.next_state_color, goal_color)
        return done, (
            self.state_color,
            self.state_object,
            self.action,
            reward,
            self.next_state_color,
            self.next_state_object,
            done,
            goal_color
        )


class DQN:
    def __init__(self, env, cfg, logger, device):
        self.device = device
        self.learning_rate = cfg.train.learning_rate
        self.tau = cfg.train.tau
        self.gamma = cfg.train.gamma
        self.epsilon = cfg.train.epsilon
        self.batch_size = cfg.train.batch_size
        self.dec_epsilon = cfg.train.dec_epsilon
        self.min_epsilon = cfg.train.min_epsilon
        self.learn_steps_count = 0

        self.q_online_1 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_online_2 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_target_1 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_target_2 = GPT_DQNCritic(cfg, env).to(self.device)
        self.q_target_1.load_state_dict(self.q_online_1.state_dict())
        self.q_target_2.load_state_dict(self.q_online_2.state_dict())
        self.buffer = HindsightExperienceReplayBuffer(self.device)

        self.optimizer_1 = self.q_online_1.configure_optimizers()
        self.optimizer_2 = self.q_online_2.configure_optimizers()
        self.loss = torch.nn.MSELoss()

        self.env = env
        self.cfg = cfg
        self.logger = logger

        self.target_entropy = 0.3 * -np.log(1 / self.env.num_actions)
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
            self.env.get_action_log(action, tag)
        )

    @torch.no_grad()
    def select_action(self, state_color, state_object, goal_color, deterministic):
        state_color = torch.tensor(np.array(state_color)[None], dtype=torch.float).to(self.device)
        state_object = torch.tensor(np.array(state_object)[None], dtype=torch.float).to(self.device)
        goal_color = torch.tensor(np.array(goal_color)[None], dtype=torch.float).to(self.device)
        action_logits, action, _, _ = self.q_online_1.forward(
            state_color, state_object, goal_color
        )
        if deterministic:
            return torch.argmax(action_logits).item()
        else:
            return action.item()

    def _rollout(self, timestep):
        state_color, state_object = self.env.reset()
        for s, o, ns, no, r, a, d, g in zip(*self.env.get_traj()):
            self.buffer.add_experience(s, o, a.item(), r, ns, no, d, g)
        goal_color = np.ravel(self.env.goal)
        done = False
        transitions = []
        ep_return = 0.0

        for t in range(self.cfg.train.max_ep_len):
            if np.random.random() > self.epsilon:
                action = self.select_action(
                    state_color, state_object, goal_color, deterministic=False
                )
                self.log_action(action, timestep + t, tag="train")
            elif np.random.random() > 0.5:
                action = self.env.get_optimal_action(state_color, state_object, goal_color).item()
            else:
                action = np.random.choice(self.env.num_actions)
            self.decrement_epsilon()

            next_state_color, next_state_object, reward, done = self.env.step(action)
            ep_return += float(reward)

            self.buffer.add_experience(
                state_color,
                state_object,
                action,
                reward,
                next_state_color,
                next_state_object,
                done,
                goal_color
            )
            transitions.append(
                Transition(
                    state_color, state_object, action, reward, next_state_color, next_state_object
                )
            )

            self._train(timestep + t)

            state_color, state_object = next_state_color, next_state_object

            if done:
                break
        return t + 1, ep_return, transitions, done

    @torch.no_grad()
    def _eval(self, global_t):
        ep_lens = []
        ep_returns = []
        for _ in range(self.cfg.logging.num_evaluations):
            state_color, state_object = self.env.reset(is_train=False)
            goal_color = np.ravel(self.env.goal)
            done = False
            ep_return = 0.0

            eval_actions = {}
            for t in range(self.cfg.train.max_ep_len):
                action = self.select_action(
                    state_color, state_object, goal_color, deterministic=True
                )
                self.log_action(action, global_t + sum(ep_lens) + t, tag="eval")

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
            },  # | eval_actions
        )

    def _add_hindsight_experiences(self, transitions, global_t):
        goal_color, goal_object = transitions[-1].get_goals()
        assert not np.array_equal(goal_color, np.ravel(self.env.goal))
        for t, transition in enumerate(transitions):
            done, experience = transition.get_hindsight_experience(
                self.env, goal_color, goal_object
            )
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
                t, {"train/success": success, "train/epsilon": self.epsilon, "train/ep_ret": ep_ret}
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
        torch.save(self.q_online_1.state_dict(), os.path.join(direc, "qnet.pt"))

    def _train(self, t):
        if self.buffer.counter < self.batch_size:
            return
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()

        (
            state_color,
            state_object,
            _,
            reward,
            next_state_color,
            next_state_object,
            done,
            goal_color
        ) = self.buffer.get_random_experience(self.batch_size)
        # Gets the evenly spaced batches
        # afterstate value and next action probs

        alpha = self.log_alpha.exp()
        _, cur_action, cur_logprob, q_prev = self.q_online_1.forward(
            state_color, state_object, goal_color
        )
        with torch.no_grad():
            cur_after_color, cur_after_object = self.env.batch_step(
                state_color.cpu().numpy(), state_object.cpu().numpy(), cur_action.cpu().numpy()
            )
            _, _, _, cur_after_value = self.q_online_1.forward(
                torch.tensor(np.array(cur_after_color)).cuda(), 
                torch.tensor(np.array(cur_after_object)).cuda(), 
                goal_color
            )
            prev_reward = (
                (state_color == goal_color).float().mean(1)
                - 1
                + (state_color == goal_color).all(1).float()
            )
            cur_value = (q_prev - prev_reward) / self.gamma
        advantage = (cur_after_value - cur_value - alpha * cur_logprob - alpha).detach()
        # subtracting cur_val - is it ok?
        # if it is not ok - somehow get reward, and use v(s) = (q_prev - r_prev) / gamma
        # advantage = advantage - advantage.mean() # normalizing to help policy update
        policy_loss = -(advantage * cur_logprob).mean()

        _, next_action, next_logprob, next_value_1 = self.q_online_1.forward(
            next_state_color, next_state_object, goal_color
        )
        _, _, _, next_value_2 = self.q_online_2.forward(
            next_state_color, next_state_object, goal_color
        )

        with torch.no_grad():
            next_after_color, next_after_object = self.env.batch_step(
                next_state_color.cpu().numpy(), next_state_object.cpu().numpy(), next_action.cpu().numpy()
            )
            next_after_color, next_after_object = torch.tensor(np.array(next_after_color)).cuda(), torch.tensor(np.array(next_after_object)).cuda()
            _, _, _, next_after_value_1 = self.q_target_1.forward(
                next_after_color, next_after_object, goal_color
            )
            _, _, _, next_after_value_2 = self.q_target_2.forward(
                next_after_color, next_after_object, goal_color
            )
            target = reward + self.gamma * (1 - done) * (
                torch.min(next_after_value_1, next_after_value_2) - alpha * next_logprob
            )

        critic_loss = self.loss(target, next_value_1) + self.loss(target, next_value_2)

        loss = policy_loss + critic_loss

        self.alpha_optim.zero_grad()
        alpha_loss = -(self.log_alpha * (cur_logprob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.logger.log(
            t,
            {
                "policy/cur_aftval_mean": torch.mean(cur_after_value).item(),
                "policy/cur_aftval_min": torch.min(cur_after_value).item(),
                "policy/cur_aftval_max": torch.max(cur_after_value).item(),
                "policy/cur_val_mean": torch.mean(cur_value).item(),
                "policy/cur_val_min": torch.min(cur_value).item(),
                "policy/cur_val_max": torch.max(cur_value).item(),
                "policy/logprob_mean": torch.mean(cur_logprob).item(),
                "policy/logprob_min": torch.min(cur_logprob).item(),
                "policy/logprob_max": torch.max(cur_logprob).item(),
                "policy/adv_mean": torch.mean(advantage).item(),
                "policy/adv_min": torch.min(advantage).item(),
                "policy/adv_max": torch.max(advantage).item(),
                "target/mean": torch.mean(target).item(),
                "target/min": torch.min(target).item(),
                "target/max": torch.max(target).item(),
                "online/mean": torch.mean(next_value_1).item(),
                "online/min": torch.min(next_value_1).item(),
                "online/max": torch.max(next_value_1).item(),
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
