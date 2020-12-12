import torch
import torch.nn as nn
import numpy as np

from gym.core import ObservationWrapper
from gym.spaces import Box

from mushroom_rl.environments import Environment

import warnings

from mushroom_rl.policy.td_policy import EpsGreedy


class MemoryNetwork(nn.Module):
    """
    This class is a base class for every network which is meant to be trained
    with the DRQN algorithm. It provides base functionality to handle the
    latent state correctly.
    """
    latent = None

    def reset_latent(self):
        pass

    def forward(self, state, action=None, **_):

        q = self._forward(self.memory_pass(state))

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(-1, action), -1)

            return q_acted

    def memory_pass(self, _):
        warnings.warn("Subclass should override memory_pass(self, state)",
                      RuntimeWarning)
        pass

    def _forward(self, hidden):
        raise NotImplementedError


class MemoryEpsGreedy(EpsGreedy):
    """
    This class extends the EpsGreedy class such that the latent state of the
    network is updated in every call of draw_action.
    """
    def draw_action(self, state):
        s = np.expand_dims(state, axis=(0, 1))
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(s)
            q = q.squeeze().squeeze()
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        # update latent memory anyway
        self._approximator.model.network.memory_pass(torch.from_numpy(s))

        return np.array([np.random.choice(self._approximator.n_actions)])


class FlagWrapper(ObservationWrapper):
    """
    This wrapper adds a flag to the observations to signalize that it's valid
    data. It is meant to be used with the DRQN for sequential updates.
    """
    def __init__(self, env, flag=True):
        super(FlagWrapper, self).__init__(env)
        self.flag = flag
        assert isinstance(env.observation_space, Box)
        low = np.append(self.observation_space.low, 0)
        high = np.append(self.observation_space.high, 1)
        self.observation_space = Box(low, high,
                                     dtype=env.observation_space.dtype)

    def observation(self, observation):
        """
        Adds a valid flag at the end of the observations.

        Returns:
            The updated observations.
        """
        return np.concatenate([observation, [self.flag]])


class POMDPWrapper(ObservationWrapper):
    """
    This wrapper makes MDP partially observable by masking the observations.
    """
    def __init__(self, env, mask=None):
        super(POMDPWrapper, self).__init__(env)
        if mask is not None:
            assert np.shape(mask) == self.observation_space.shape
            self.mask = mask
            low = self.observation_space.low[mask]
            high = self.observation_space.high[mask]
            self.observation_space = Box(low, high,
                                         dtype=env.observation_space.dtype)
        else:
            self.mask = np.ones(self.observation_space.shape, dtype=bool)

    def observation(self, observation):
        return observation[self.mask]


class MushroomWrapper(Environment):
    """
    Base class for mushroom environment wrappers.
    """
    def __init__(self, env: Environment):
        super().__init__(env.info)
        self.env = env

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def reset(self, state=None):
        return self.env.reset(state)


class MushroomPOMDPWrapper(MushroomWrapper):
    """
    This mushroom wrapper makes MDP partially observable by masking the
    observations.
    """
    def __init__(self, env, mask):
        super(MushroomPOMDPWrapper, self).__init__(env)
        assert np.shape(mask) == self.info.observation_space.shape
        self.mask = mask
        low = self.info.observation_space.low[mask]
        high = self.info.observation_space.high[mask]
        self.info.observation_space = Box(low, high)

    def step(self, action):
        obs, reward, absorbing, info = self.env.step(action)
        return obs[self.mask], reward, absorbing, info

    def reset(self, state=None):
        return self.env.reset(state)[self.mask]
