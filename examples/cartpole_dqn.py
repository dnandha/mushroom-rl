import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import episodes_length
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.algorithms.value import DQN


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **_):
        super().__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]

        h = [24, 24]

        self._h1 = nn.Linear(n_input, h[0])
        self._h2 = nn.Linear(h[0], h[1])
        self._h4 = nn.Linear(h[1], self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('linear'))

        self.float()

    def forward(self, state, action=None, **kwargs):

        q = state.float()
        q = self._h1(q)
        q = F.relu(q)
        q = F.relu(self._h2(q))
        q = self._h4(q)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.squeeze(1).gather(1, action))
            return q_acted


def experiment():
    np.random.seed(0)

    # MDP
    mdp = CartPole()

    # Policy
    epsilon_random = Parameter(.2)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    input_shape = mdp.info.observation_space.shape
    approximator_params = dict(network=Network,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.mse_loss,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               use_cuda=False)

    # Agent
    params = dict(batch_size=50,
                  initial_replay_size=200,
                  max_replay_size=1000,
                  target_update_frequency=100)

    agent = DQN(mdp.info, pi, TorchApproximator,
                approximator_params=approximator_params, **params)

    # Algorithm
    core = Core(agent, mdp)

    core.evaluate(n_episodes=3, render=True)

    # Train
    core.learn(n_episodes=1000, n_steps_per_fit=1)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    dataset = core.evaluate(n_episodes=3)

    core.evaluate(n_steps=10, render=True)

    return np.mean(episodes_length(dataset))


if __name__ == '__main__':
    n_experiment = 1

    steps = experiment()
    print('Final episode length: ', steps)
