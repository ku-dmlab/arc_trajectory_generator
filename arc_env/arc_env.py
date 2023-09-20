import json
import os
import copy
import logging
from functools import partial
import arc_env.arc_actions as actions
from dataclasses import dataclass
from typing import Optional

import jax
from jax import jit
import jax.numpy as jnp
import torch
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ActionSpec:
    action_obj: actions.ArcAction
    action_arg: jnp.array
    is_global: bool
    is_object: bool
    pixel_row: Optional[int]
    pixel_col: Optional[int]

    def step(self, state, obj, selector):
        return self.action_obj.step(state, obj, self.action_arg, selector)


class MiniArcEnv:
    def __init__(self, cfg, initial, goal, initial_obj):
        self.cfg = cfg
        self.num_colors = cfg.env.num_colors
        self.default_color = cfg.env.default_color
        self.grid_x = cfg.env.grid_x
        self.grid_y = cfg.env.grid_y
        self.arc_local_actions = [getattr(actions, name)(cfg) for name in cfg.env.arc_local_actions]
        self.arc_global_actions = [
            getattr(actions, name)(cfg) for name in cfg.env.arc_global_actions
        ]
        self.list_actions = self.arc_local_actions + self.arc_global_actions

        self.num_pixels = self.grid_y * self.grid_x

        self.penalize_actions = [(list(cfg.env.arc_local_actions) + list(cfg.env.arc_global_actions)).index(name) for name in cfg.env.penalize_actions]
        (
            self.action_specs,
            self.action_types,
            self.action_args,
            self.action_rows,
            self.action_cols,
            self.action_is_obj,
            self.action_is_global,
            self.action_penalize
        ) = self.build_action_specs()
        self.num_local_actions = sum([action.num_actions for action in self.arc_local_actions])
        self.num_global_actions = sum([action.num_actions for action in self.arc_global_actions])
        self.num_actions = len(self.action_specs)
        print("actions penalized: ", self.penalize_actions)

        self.initial, self._goal, self.initial_obj = initial, goal, initial_obj
        self.state = None
        self.obj = None
        self.reset()

    def build_action_specs(self):
        action_specs = []
        for row in range(self.grid_y):
            for col in range(self.grid_x):
                for action in self.arc_local_actions:
                    action_specs.extend(
                        [
                            ActionSpec(action, jnp.array([i]), False, action.is_object(i), row, col)
                            for i in range(action.num_actions)
                        ]
                    )
        for action in self.arc_global_actions:
            action_specs.extend(
                [
                    ActionSpec(action, jnp.array([i]), True, False, 0, 0)
                    for i in range(action.num_actions)
                ]
            )
        types = jnp.array([self.list_actions.index(spec.action_obj) for spec in action_specs])
        args = jnp.concatenate([spec.action_arg for spec in action_specs])
        rows = jnp.array([spec.pixel_row for spec in action_specs])
        cols = jnp.array([spec.pixel_col for spec in action_specs])
        is_obj = jnp.array([spec.is_object for spec in action_specs])
        is_global = jnp.array([spec.is_global for spec in action_specs])
        penalize = jnp.array([self.list_actions.index(spec.action_obj) in self.penalize_actions for spec in action_specs])
        return action_specs, types, args, rows, cols, is_obj, is_global, penalize

    def get_action_log(self, action, tag):
        action_spec = self.action_specs[action]
        ret = {
            f"{tag}_action/action_arg": action_spec.action_arg.item(),
            f"{tag}_action/is_global": action_spec.is_global,
            f"{tag}_action/action_cls": self.list_actions.index(action_spec.action_obj),
        }
        if not action_spec.is_global:
            ret |= {
                f"{tag}_action/is_object": action_spec.is_object,
                f"{tag}_action/pixel_row": action_spec.pixel_row,
                f"{tag}_action/pixel_col": action_spec.pixel_col,
            }
        return ret

    def step(self, action):
        spec = self.action_specs[action]
        selector = self.get_selector(self.state, self.obj, jnp.array([action]))
        next_state, self.obj = spec.step(self.state, self.obj, selector)
        reward, done = self.reward_done(self.state, action, next_state)
        self.state = next_state
        return (
            np.array(self.state).ravel(),
            np.array(self.obj).ravel(),
            reward, done
        )

    def reward_done(self, state, action, next_state, goal=None):
        goal = goal if goal is not None else self._goal
        return (
            np.array(self.batch_reward(state, jnp.array([action]), next_state, goal)).ravel().item(),
            np.array(self.batch_done(next_state, goal)).ravel().item(),
        )

    @property
    def goal(self):
        return jnp.ravel(self._goal)

    @partial(jit, static_argnums=(0,))
    def get_selector(self, state, obj, action):
        batch_size = obj.shape[0]
        action_rows = self.action_rows[action]
        action_cols = self.action_cols[action]
        is_obj = self.action_is_obj[action][:, None, None]
        is_global = self.action_is_global[action][:, None, None]

        obj_index = obj[jnp.arange(batch_size), action_rows, action_cols]
        selector_obj = obj == obj_index[:, None, None]
        selector_pxl = (
            jnp.zeros_like(state).at[jnp.arange(batch_size), action_rows, action_cols].set(True)
        )
        selector_global = jnp.ones_like(state)
        return (
            selector_obj * is_obj * (1 - is_global)
            + selector_pxl * (1 - is_obj) * (1 - is_global)
            + selector_global * is_global
        )

    @partial(jit, static_argnums=(0,))
    def batch_step(self, state, obj, action):
        state = state.reshape(-1, self.grid_y, self.grid_x)
        obj = obj.reshape(-1, self.grid_y, self.grid_x)
        action_type = self.action_types[action]
        action_args = self.action_args[action]

        selector = self.get_selector(state, obj, action)

        ret_states = []
        ret_objs = []
        for i, action in enumerate(self.list_actions):
            cur_idx = (action_type == i)[:, None, None]
            tmp_state, tmp_obj = action.step(state, obj, action_args, selector)
            ret_states.append(tmp_state * cur_idx)
            ret_objs.append(tmp_obj * cur_idx)
        return sum(ret_states).reshape(-1, self.num_pixels), sum(ret_objs).reshape(
            -1, self.num_pixels
        )

    @partial(jit, static_argnums=(0,))
    def batch_reward(self, state, action, next_state, goal):
        penalty = self.action_penalize[action.ravel()]
        next_state = next_state.reshape(-1, self.grid_y, self.grid_x)
        goal = goal.reshape(-1, self.grid_y, self.grid_x)
        return (next_state == goal).mean([1, 2]) + (next_state == goal).all([1, 2]) * self.cfg.env.finish_reward - 1 - penalty * self.cfg.env.penality_reward

    @partial(jit, static_argnums=(0,))
    def batch_done(self, next_state, goal):
        next_state = next_state.reshape(-1, self.grid_y, self.grid_x)
        goal = goal.reshape(-1, self.grid_y, self.grid_x)
        return (next_state == goal).all([1, 2])

    def reset(self, is_train=True):
        self.state, self.obj = self.initial, self.initial_obj
        return jnp.ravel(self.state), jnp.ravel(self.obj)

    def batch_reset(self, batch_size, is_train=True):
        return (
            jnp.tile(self.state.reshape(1, -1), [batch_size, 1]),
            jnp.tile(self.obj.reshape(1, -1), [batch_size, 1]),
            jnp.tile(self.goal.reshape(1, -1), [batch_size, 1]),
        )

    def get_optimal_action(self, state, obj, goal):
        actions = jnp.arange(self.num_actions)
        states = jnp.tile(state, [self.num_actions, 1, 1])
        next_states, _ = self.batch_step(
            jnp.tile(state, [self.num_actions, 1, 1]),
            jnp.tile(obj, [self.num_actions, 1, 1]),
            actions,
        )
        rewards = self.batch_reward(states, actions, next_states, jnp.tile(goal, [self.num_actions, 1, 1]))
        return rewards.argmax(keepdims=True)


def get_json_file_list(project_dir, dirs):
    dir_list = []
    for dir in dirs:
        for root, _, files in os.walk(os.path.join(project_dir, dir)):
            for name in files:
                if name.endswith((".json")):
                    dir_list.append(os.path.join(root, name))
    return dir_list


class DataBasedARCEnv(MiniArcEnv):
    def __init__(self, cfg):
        self.train_list = get_json_file_list(cfg.env.project_dir, cfg.env.train_paths)
        self.test_list = get_json_file_list(cfg.env.project_dir, cfg.env.test_paths)
        logger.info(f"{len(self.train_list)} train trajectories")
        logger.info(f"{len(self.test_list)} test trajectories")
        super().__init__(cfg, None, None, None)

    def initialize_from_random_json(self, task_list):
        while True:
            task_num = np.random.randint(0, len(task_list))
            filename = task_list[task_num]
            with open(filename, "r") as f:
                traj = json.load(f)
            traj = traj["action_sequence"]["action_sequence"]
            initial = jnp.array(traj[0]["grid"])
            goal = jnp.array(traj[-1]["grid"])
            initial_obj = jnp.array(traj[0]["grid"])

            if initial.shape == goal.shape == initial_obj.shape == (self.grid_x, self.grid_y):
                break

        initial_aug, goal_aug = self.random_augment(initial, goal, self.num_colors)
        initial_obj_aug = self.random_augment(initial_obj, None, self.num_pixels)
        self.traj = traj
        return initial_aug[None], initial_obj_aug[None], goal_aug[None]

    def get_traj(self):
        sequence = copy.deepcopy(self.traj)
        invalid_action = ["none", "end", "start"]
        states = []
        objs = []
        next_states = []
        next_objs = []
        actions = []
        self.state_actions = None
        for i, step in enumerate(sequence):
            if step["action"]["tool"] not in invalid_action:
                states.append(jnp.array(sequence[i - 1]["grid"])[None])
                objs.append(jnp.array(sequence[i - 1]["pnp"])[None])
                next_states.append(jnp.array(sequence[i]["grid"])[None])
                next_objs.append(jnp.array(sequence[i]["pnp"])[None])
                actions.append(self.get_optimal_action(states[-1], objs[-1], next_states[-1]))
        states = jnp.stack(states)
        next_states = jnp.stack(next_states)
        objs = jnp.stack(objs)
        next_objs = jnp.stack(next_objs)
        actions = jnp.stack(actions)
        self.state_actions = np.concatenate([states.reshape(-1, self.num_pixels), actions.reshape(-1, 1)], axis=1)

        goals = jnp.tile(
            jnp.array(sequence[-1]["grid"]).reshape(-1, self.num_pixels), [len(states), 1]
        )
        rewards = self.batch_reward(states, actions, next_states, goals)
        dones = self.batch_done(next_states, goals)

        return (
            states.reshape(-1, self.num_pixels),
            objs.reshape(-1, self.num_pixels),
            next_states.reshape(-1, self.num_pixels),
            next_objs.reshape(-1, self.num_pixels),
            rewards,
            actions,
            dones,
            goals,
        )

    def random_augment(self, x1, x2, max_num):
        values = jnp.array(np.random.permutation(np.arange(max_num)))
        mapper = jnp.vectorize(lambda x: values[x])
        if x2 is None:
            return mapper(x1)
        return mapper(x1), mapper(x2)

    def reset(self, is_train=True):
        self.initial, self.initial_obj, self._goal = self.initialize_from_random_json(
            self.train_list if is_train else self.test_list
        )
        return super().reset()

    def batch_reset(self, batch_size, is_train=False):
        file_list = self.train_list if is_train else self.test_list
        rets = jnp.array(
            [self.initialize_from_random_json(file_list) for _ in range(batch_size)]
        ).squeeze(2)
        return (
            rets[:, 0].reshape(batch_size, -1),
            rets[:, 1].reshape(batch_size, -1),
            rets[:, 2].reshape(batch_size, -1),
        )

    def batch_reward(self, state, action, next_state, goal):
        reward = super().batch_reward(state, action, next_state, goal)
        return reward
        if self.state_actions is None:
            return reward
        traj_state_action = self.state_actions
        state_action = np.concatenate([state.reshape(-1, self.num_pixels), action.reshape(-1, 1)], axis=1)
        traj_state_action = np.tile(traj_state_action[:, None], [1, len(state_action), 1])
        state_action = np.tile(state_action[None], [len(traj_state_action), 1, 1])
        isin = np.any(np.all(traj_state_action == state_action, axis=2), axis=0)
        return reward + isin
        