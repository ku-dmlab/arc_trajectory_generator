import json
import os
import random
import numpy as np
from enum import Enum

NUM_COLORS = 10
DEFAULT_COLOR = 0

class ActionType(Enum):
    TRANSLATE = 0
    SELECT_FILL = 1

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class MiniArcEnv:
    def __init__(self, initial, goal, initial_obj, goal_obj, use_reward_shaping=True):
        self.initial = initial
        self.max_rows, self.max_cols = self.initial.shape
        self.goal = goal
        self.initial_obj = initial_obj
        self.goal_obj = goal_obj
        self.state = None
        self.obj = None
        self.use_reward_shaping = use_reward_shaping
        self.translate_x = {
            Direction.UP: 0, Direction.DOWN: 0, Direction.LEFT: -1, Direction.RIGHT: 1
        }
        self.translate_y = {
            Direction.UP: -1, Direction.DOWN: 1, Direction.LEFT: 0, Direction.RIGHT: 0
        }
        self.reset()

        self.num_actions = 2 * (len(Direction) + NUM_COLORS)
        self.num_directions = len(Direction)
        self.num_pixels = self.max_rows * self.max_cols

    def step(self, action):
        pixel = action // self.num_actions
        is_obj = action % self.num_actions // (len(Direction) + NUM_COLORS)
        row, col = pixel // self.max_rows, pixel % self.max_rows
        if is_obj:
            obj_index = self.obj[row, col]
            selected_rows, selected_cols = np.nonzero(self.obj == obj_index)
        else:
            selected_rows = np.array([row])
            selected_cols = np.array([col])
        action_arg = action % (self.num_actions // 2)
        is_translate = action_arg < len(Direction)

        if is_translate:
            return self.translate(selected_rows, selected_cols, action_arg)
        else:
            return self.select_fill(selected_rows, selected_cols, action_arg - len(Direction))

    def translate(self, selected_rows, selected_cols, direction):
        self.state = self.state.copy()
        self.obj = self.obj.copy()

        target_rows = selected_rows + self.translate_y[Direction(direction%4)]
        target_cols = selected_cols + self.translate_x[Direction(direction%4)]

        row_excluder = np.logical_and(target_rows >= 0, target_rows < self.max_rows)
        col_excluder = np.logical_and(target_cols >= 0, target_cols < self.max_cols)
        excluder = np.logical_and(row_excluder, col_excluder)
        target_rows = target_rows[excluder]
        moved_rows = selected_rows[excluder]
        target_cols = target_cols[excluder]
        moved_cols = selected_cols[excluder]

        temp = self.state[moved_rows, moved_cols]
        self.state[selected_rows, selected_cols] = DEFAULT_COLOR
        self.state[target_rows, target_cols] = temp

        temp = self.obj[moved_rows, moved_cols]
        self.obj[selected_rows, selected_cols] = DEFAULT_COLOR
        self.obj[target_rows, target_cols] = temp

        return np.ravel(self.state), np.ravel(self.obj), *self.reward_done(self.state)

    def select_fill(self, selected_rows, selected_cols, color):
        self.state = self.state.copy()
        self.obj = self.obj.copy()
        self.state[selected_rows, selected_cols] = color

        return np.ravel(self.state), np.ravel(self.obj), *self.reward_done(self.state)

    def reward_done(self, next_state, goal = None):
        goal = goal if goal is not None else self.goal
        assert next_state.shape == goal.shape
        is_done = int((next_state == goal).all())
        if not self.use_reward_shaping:
            return is_done - 1, is_done
        distance = self.num_pixels - (next_state == goal).sum()
        return - distance / self.num_pixels + is_done, is_done

    def reset(self, is_train=True):
        self.state = self.initial.copy()
        self.obj = self.initial_obj.copy()
        return np.ravel(self.state), np.ravel(self.obj)


class DataBasedARCEnv(MiniArcEnv):
    def __init__(self, cfg, use_reward_shaping=True):
        self.use_reward_shaping = use_reward_shaping
        self.translate_x = {
            Direction.UP: 0, Direction.DOWN: 0, Direction.LEFT: -1, Direction.RIGHT: 1
        }
        self.translate_y = {
            Direction.UP: -1, Direction.DOWN: 1, Direction.LEFT: 0, Direction.RIGHT: 0
        }
        self.num_actions = 2 * (len(Direction) + NUM_COLORS)
        self.num_directions = len(Direction)

        self.train_list = self.get_file_list(cfg.env.project_dir, cfg.env.train_paths)
        print(f"{len(self.train_list)} train_trajectories")
        self.test_list = self.get_file_list(cfg.env.project_dir, cfg.env.test_paths)
        print(f"{len(self.test_list)} train_trajectories")
        self.reset()
    
    def get_file_list(self, project_dir, dirs):
        dir_list = []
        for dir in dirs:
            for root, _, files in os.walk(os.path.join(project_dir, dir)):
                for name in files:
                    if name.endswith((".json")):
                        dir_list.append(os.path.join(root, name))
        return dir_list

    def initialize_from_tasks(self, task_list):
        found = False
        while not found:
            filename = random.choice(task_list)
            with open(filename, "r") as f:
                traj = json.load(f)
            traj = traj["action_sequence"]["action_sequence"]
            initial = np.array(traj[0]["grid"])
            goal = np.array(traj[-1]["grid"])
            initial_obj = np.array(traj[0]["grid"])
            goal_obj = np.array(traj[-1].get("pnp", initial_obj))
            if initial.shape == (5,5):
                found = True
        self.initial, self.goal = self.random_augment(initial, goal, NUM_COLORS)
        self.initial_obj, self.goal_obj = self.random_augment(initial_obj, goal_obj, 25)
        self.max_rows, self.max_cols = self.initial.shape
        self.num_pixels = self.max_rows * self.max_cols

    def random_augment(self, x1, x2, max_num):
        keys = np.arange(max_num)
        values = np.random.permutation(np.arange(max_num))
        mapper_dict = {k:v for k, v in zip(keys, values)}
        mapper = np.vectorize(lambda x: mapper_dict[x])
        return mapper(x1), mapper(x2)

    def reset(self, is_train=True):
        self.initialize_from_tasks(self.train_list if is_train else self.test_list)
        return super().reset()

class TestEnv1:
    INITIAL = np.array(
        [[0,4,4,0,0],
         [0,4,6,6,0],
         [0,4,4,6,0],
         [0,4,4,0,0],
         [0,6,4,0,0]])
    GOAL = np.array(
        [[0,6,6,0,0],
         [0,6,4,4,0],
         [0,6,6,4,0],
         [0,6,6,0,0],
         [0,4,6,0,0]])
    GOAL_OBJ = np.array(
        [[0,1,1,0,0],
         [0,1,2,2,0],
         [0,1,1,2,0],
         [0,1,1,0,0],
         [0,3,1,0,0]])
    OBJ_MAP = np.array(
        [[0,1,1,0,0],
         [0,1,2,2,0],
         [0,1,1,2,0],
         [0,1,1,0,0],
         [0,3,1,0,0]])

    @classmethod
    def get_args(cls):
        return cls.INITIAL.copy(), cls.GOAL.copy(), cls.OBJ_MAP.copy(), cls.GOAL_OBJ.copy()

class TestEnv2:
    INITIAL = np.array(
        [[0,4,4,0,0],
         [0,4,6,6,0],
         [0,4,4,6,0],
         [0,4,4,0,0],
         [0,6,4,0,0]])
    GOAL = np.array(
        [[0,4,4,0,0],
         [0,4,4,4,0],
         [0,4,4,4,0],
         [0,4,4,0,0],
         [0,6,4,0,0]])
    OBJ_MAP = np.array(
        [[0,1,1,0,0],
         [0,1,2,2,0],
         [0,1,1,2,0],
         [0,1,1,0,0],
         [0,3,1,0,0]])

    @classmethod
    def get_args(cls):
        return cls.INITIAL.copy(), cls.GOAL.copy(), cls.OBJ_MAP.copy()

