import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from utils import load_dataset


class SudokuEnvironment(py_environment.PyEnvironment):
    def __init__(self, dataset):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype='int32',
            minimum=0,
            maximum=81 * 9 - 1,
            name="action",
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(81, 9),
            dtype='int32',
            minimum=0,
            maximum=1,
            name="observation",
        )
        self.ds = dataset
        q, s = next(iter(self.ds))
        self._state = q.numpy()
        self._solution = s.numpy()
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        q, s = next(iter(self.ds))
        self._state = q.numpy()
        self._solution = s.numpy()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        action_2d = (action // 9, action % 9)

        if self._episode_ended:
            return self.reset()

        if self.__correct_action(action_2d):
            self._state[action_2d] = 1

            if self.__is_solved():
                self._episode_ended = True
                return ts.termination(self._state, reward=1.0)
            else:
                return ts.transition(self._state, reward=0.1, discount=1.0)

        else:
            self._episode_ended = True
            return ts.termination(self._state, reward=-1.0)

    def __is_solved(self):
        return np.all(self._state == self._solution)

    # TODO: Write a allowed action instead
    def __correct_action(self, action):
        return self._solution[action] == 1 & self._state[action] == 0


if __name__ == '__main__':
    ds_train, ds_eval = load_dataset("data/sudoku.csv")
    env = SudokuEnvironment(ds_train)
    utils.validate_py_environment(env, episodes=50)
