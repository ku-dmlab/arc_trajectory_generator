from abc import ABC, abstractmethod
from functools import partial
import jax
from jax import jit
import jax.numpy as jnp


class ArcAction(ABC):
    num_actions = None

    @abstractmethod
    def __init__(self, cfg):
        pass

    @abstractmethod
    def step(self, state, obj, action_arg, selector):
        """
        state: batch x row x col
        obj: batch x row x col
        action_arg: batch x 1
        selector: batch x row x col
        """
        pass

    def is_object(self, action_arg):
        return False


class ArcNonPixelAction(ABC):
    def is_object(self, action_arg):
        return True


class ArcObjectAction(ABC):
    def is_object(self, action_arg):
        return action_arg >= self.num_actions // 2

    def step(self, state, obj, action_arg, selector):
        action_arg = action_arg % (self.num_actions // 2)
        return self._step(state, obj, action_arg, selector)


class Translate(ArcObjectAction):
    translate_rows = jnp.array([-1, 1, 0, 0])
    translate_cols = jnp.array([0, 0, -1, 1])

    def __init__(self, cfg):
        assert len(self.translate_rows) == len(self.translate_cols)
        self.num_actions = len(self.translate_rows) * 2
        self.default_color = cfg.env.default_color
        self.default_obj = cfg.env.default_obj
        self.grid_x = cfg.env.grid_x
        self.grid_y = cfg.env.grid_y

    @partial(jit, static_argnums=(0,))
    def _step(self, state, obj, translate_direc, selector):
        zeros = jnp.zeros_like(state)
        direc = translate_direc[:, None, None]

        def get_target(initial):
            up = zeros.at[:, :-1].set(initial[:, 1:])
            down = zeros.at[:, 1:].set(initial[:, :-1])
            left = zeros.at[:, :, :-1].set(initial[:, :, 1:])
            right = zeros.at[:, :, 1:].set(initial[:, :, :-1])
            return (
                (direc == 0) * up + (direc == 1) * down + (direc == 2) * left + (direc == 3) * right
            )

        target_selector = get_target(selector)
        target_state = get_target(state)
        target_obj = get_target(obj)

        left_behind = jnp.logical_and(selector, ~jnp.logical_and(selector, target_selector))
        all_other = ~jnp.logical_or(selector, target_selector)

        state = (
            target_selector * target_state + left_behind * self.default_color + all_other * state
        )

        obj = target_selector * target_obj + left_behind * self.default_obj + all_other * obj

        return state, obj


class Fill(ArcObjectAction):
    def __init__(self, cfg):
        self.num_actions = cfg.env.num_colors * 2

    @partial(jit, static_argnums=(0,))
    def _step(self, state, obj, color, selector):
        state = selector * color[:, None, None] + (1 - selector) * state
        return state, obj


class Rotate(ArcNonPixelAction):
    def __init__(self, cfg):
        self.default_color = cfg.env.default_color
        self.default_obj = cfg.env.default_obj
        self.grid_x = cfg.env.grid_x
        self.grid_y = cfg.env.grid_y
        self.num_actions = 3

    @partial(jit, static_argnums=(0,))
    def step(self, state, obj, num_rot, selector):
        def single_step(state, obj, selector, num_rot):
            row_all = jax.tree_util.tree_reduce(
                jnp.logical_or, [jnp.roll(selector, i, 0) for i in range(self.grid_y)]
            )
            col_all = jax.tree_util.tree_reduce(
                jnp.logical_or, [jnp.roll(selector, i, 1) for i in range(self.grid_x)]
            )
            bounding_rectangle = jnp.logical_and(row_all, col_all)
            selected_row_center = (
                bounding_rectangle * jnp.arange(self.grid_y)[:, None]
            ).sum() / bounding_rectangle.sum()
            selected_col_center = (
                bounding_rectangle * jnp.arange(self.grid_x)
            ).sum() / bounding_rectangle.sum()
            row_center = (self.grid_y - 1) / 2
            col_center = (self.grid_x - 1) / 2
            row_dsp = row_center - selected_row_center
            col_dsp = col_center - selected_col_center
            dsp = jnp.stack([jnp.floor(row_dsp), jnp.floor(col_dsp)])
            cur_bias = dsp - jnp.stack([row_dsp, col_dsp])
            rotation = jnp.linalg.matrix_power(jnp.array([[0, -1], [1, 0]]), num_rot)
            final_bias = jnp.matmul(rotation, cur_bias[:, None]).squeeze()
            correction = jnp.floor(final_bias + jnp.array([0.5, 0.5]))

            def get_target(matrix):
                matrix = jax.image.scale_and_translate(
                    matrix, (self.grid_y, self.grid_x), (0, 1), jnp.ones(2), dsp, "linear"
                )
                matrix = jnp.rot90(matrix, k=num_rot)
                matrix = jax.image.scale_and_translate(
                    matrix,
                    (self.grid_y, self.grid_x),
                    (0, 1),
                    jnp.ones(2),
                    -dsp - correction,
                    "linear",
                )
                return matrix

            target_selector = get_target(selector)
            target_state = get_target(state)
            target_obj = get_target(obj)

            left_behind = jnp.logical_and(selector, ~jnp.logical_and(selector, target_selector))
            all_other = ~jnp.logical_or(selector, target_selector)

            state = (
                target_selector * target_state
                + left_behind * self.default_color
                + all_other * state
            )

            obj = target_selector * target_obj + left_behind * self.default_obj + all_other * obj

            return state, obj

        state_rot1, obj_rot1 = jax.vmap(partial(single_step, num_rot=1))(state, obj, selector)
        state_rot2, obj_rot2 = jax.vmap(partial(single_step, num_rot=2))(state, obj, selector)
        state_rot3, obj_rot3 = jax.vmap(partial(single_step, num_rot=3))(state, obj, selector)
        state = (
            state_rot1 * (num_rot == 0)[:, None, None]
            + state_rot2 * (num_rot == 1)[:, None, None]
            + state_rot3 * (num_rot == 2)[:, None, None]
        )
        obj = (
            obj_rot1 * (num_rot == 0)[:, None, None]
            + obj_rot2 * (num_rot == 1)[:, None, None]
            + obj_rot3 * (num_rot == 2)[:, None, None]
        )
        return state.astype(int), obj.astype(int)


class Flip(ArcNonPixelAction):
    def __init__(self, cfg):
        self.default_color = cfg.env.default_color
        self.default_obj = cfg.env.default_obj
        self.grid_x = cfg.env.grid_x
        self.grid_y = cfg.env.grid_y
        self.num_actions = 2

    @partial(jit, static_argnums=(0,))
    def step(self, state, obj, axis_flip, selector):
        def single_step(state, obj, selector, axis_flip):
            row_all = jax.tree_util.tree_reduce(
                jnp.logical_or, [jnp.roll(selector, i, 0) for i in range(self.grid_y)]
            )
            col_all = jax.tree_util.tree_reduce(
                jnp.logical_or, [jnp.roll(selector, i, 1) for i in range(self.grid_x)]
            )
            bounding_rectangle = jnp.logical_and(row_all, col_all)
            selected_row_center = (
                bounding_rectangle * jnp.arange(self.grid_y)[:, None]
            ).sum() / bounding_rectangle.sum()
            selected_col_center = (
                bounding_rectangle * jnp.arange(self.grid_x)
            ).sum() / bounding_rectangle.sum()
            row_center = (self.grid_y - 1) / 2
            col_center = (self.grid_x - 1) / 2
            row_dsp = row_center - selected_row_center
            col_dsp = col_center - selected_col_center
            dsp = jnp.stack([jnp.floor(row_dsp), jnp.floor(col_dsp)])
            cur_bias = dsp - jnp.stack([row_dsp, col_dsp])
            final_bias = (
                jnp.array([-1, 1]) * cur_bias * (1 - axis_flip)
                + jnp.array([1, -1]) * cur_bias * axis_flip
            )
            correction = jnp.floor(final_bias + jnp.array([0.5, 0.5]))

            def get_target(matrix):
                matrix = jax.image.scale_and_translate(
                    matrix, (self.grid_y, self.grid_x), (0, 1), jnp.ones(2), dsp, "linear"
                )
                matrix = jnp.flip(matrix, axis_flip)
                matrix = jax.image.scale_and_translate(
                    matrix,
                    (self.grid_y, self.grid_x),
                    (0, 1),
                    jnp.ones(2),
                    -dsp - correction,
                    "linear",
                )
                return matrix

            target_selector = get_target(selector)
            target_state = get_target(state)
            target_obj = get_target(obj)

            left_behind = jnp.logical_and(selector, ~jnp.logical_and(selector, target_selector))
            all_other = ~jnp.logical_or(selector, target_selector)

            state = (
                target_selector * target_state
                + left_behind * self.default_color
                + all_other * state
            )

            obj = target_selector * target_obj + left_behind * self.default_obj + all_other * obj

            return state, obj

        state_flip_0, obj_flip_0 = jax.vmap(partial(single_step, axis_flip=0))(state, obj, selector)
        state_flip_1, obj_flip_1 = jax.vmap(partial(single_step, axis_flip=1))(state, obj, selector)
        state = (
            state_flip_0 * (axis_flip == 0)[:, None, None]
            + state_flip_1 * (axis_flip == 1)[:, None, None]
        )
        obj = (
            obj_flip_0 * (axis_flip == 0)[:, None, None]
            + obj_flip_1 * (axis_flip == 1)[:, None, None]
        )
        return state.astype(int), obj.astype(int)
