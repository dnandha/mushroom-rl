import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DRQN
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import Parameter


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dims, **kwargs):
        super().__init__()

        self._n_input = input_shape[-1]
        self._n_output = output_shape[0]

        self._h1 = nn.RNN(self._n_input, self._n_output,
                          nonlinearity='relu', bias=False,
                          batch_first=True)

        self.float()
        self.latent = None

    def forward(self, state, action=None, **kwargs):

        # limit observability
        q = state[:, :, :self._n_input].float()
        q, self.latent = self._h1(q, self.latent)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = q.gather(-1, action).squeeze(-1)

            return q_acted

    def reset_latent(self):
        self.latent = None


def learn(alg, alg_params):
    # MDP
    mdp = CartPole()
    np.random.seed(17)
    torch.manual_seed(1)

    # Policy
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    input_shape = mdp.info.observation_space.shape
    approximator_params = dict(network=Network,
                               optimizer={'class': optim.SGD,
                                          'params': {'lr': 1.}},
                               loss=F.mse_loss,
                               input_shape=[1],
                               latent_dims=0,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               use_cuda=False)

    # Agent
    agent = alg(mdp.info, pi, TorchApproximator,
                approximator_params=approximator_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=3, n_steps_per_fit=3, quiet=True)

    return agent


def test_drqn():
    params = dict(batch_size=1, n_approximators=1, initial_replay_size=2,
                  unroll_steps=2, max_replay_size=500,
                  target_update_frequency=50)
    approximator = learn(DRQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([0.29748732, -0.26594713, -0.11243464,
                       0.27099025, -0.5435388, 0.3462469,
                       -0.11877558, 0.29533836, 0.08097079,
                       -0.07069314, 0.16013438, 0.02848172])

    assert np.allclose(w, w_test)
