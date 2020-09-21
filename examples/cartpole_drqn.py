import numpy as np
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import episodes_length
from mushroom_rl.utils.parameters import Parameter, LinearParameter
from mushroom_rl.approximators.parametric import RecurrentApproximator
from mushroom_rl.algorithms.value import DRQN


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dims, **kwargs):
        super().__init__()

        self._n_input = input_shape[-1]
        self._n_output = output_shape[0]

        self._h1 = nn.GRU(self._n_input, latent_dims, num_layers=2)
        self._h2 = nn.Linear(latent_dims, self._n_output)

        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('linear'))

        self.float()

    def forward(self, state, action=None, **kwargs):

        if "latent" in kwargs:
            latent = kwargs["latent"]
        else:
            latent = None

        # limit observability
        q = state[:, :, :self._n_input].float()

        q, latent = self._h1(q, latent)
        q = self._h2(q)

        if action is None:
            return q, latent
        else:
            action = action.long()
            q_acted = torch.squeeze(q.squeeze(1).gather(1, action))

            return q_acted, latent


def experiment(n_experiments):
    # MDP
    mdp = CartPole()

    # Policy
    epsilon_random = LinearParameter(value=1., threshold_value=.1, n=80000)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    approximator_params = dict(network=Network,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.mse_loss,
                               input_shape=[1],
                               latent_dims=10,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               use_cuda=False)

    # Agent
    params = dict(batch_size=10,
                  n_approximators=1,
                  initial_replay_size=120,
                  max_replay_size=10000,
                  target_update_frequency=10)

    agent = DRQN(mdp.info, pi, RecurrentApproximator,
                 approximator_params=approximator_params,
                 sequential_updates=False, **params)

    # reset latent state after every episode
    def callback(x):
        if x[0][-1]:
            agent.approximator.model.reset_latent()

    # Algorithm
    core = Core(agent, mdp, callback_step=callback)

    # core.evaluate(n_episodes=3, render=True)

    dataset = []
    for i in range(n_experiments):
        # Train
        core.learn(n_episodes=10000000, n_episodes_per_fit=20)

        # Test
        test_epsilon = Parameter(0.)
        agent.policy.set_epsilon(test_epsilon)

        d = np.mean(episodes_length(core.evaluate(n_episodes=3)))
        dataset.append(d)

    core.evaluate(n_steps=10, render=True)
    return np.mean(dataset)


if __name__ == '__main__':
    n_experiment = 1

    steps = experiment(n_experiment)
    print('Final episode length: ', steps)
