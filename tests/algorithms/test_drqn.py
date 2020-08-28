import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import shutil
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms import Agent
from mushroom_rl.algorithms.value import DRQN
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric.recurrent_torch_approximator import *
from mushroom_rl.utils.parameters import Parameter, LinearParameter


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]

        self._h1 = nn.LSTM(n_input, self.n_output)

        self.latent = (torch.zeros(self.n_output), torch.zeros(self.n_output))

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):

        q, self.latent = self._h1(torch.squeeze(state, 1).float(), self.latent)
        q = F.relu(q)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted

    def warm_up(self, states, actions):
        assert len(states) == len(actions), "Length of inputs does not match!"

        # reset latent state to zeros at the beginning of the episode
        self.latent = self.latent = (torch.zeros(self.n_output),
                                     torch.zeros(self.n_output))

        # go through all states to create initial latent state for training
        for i in range(0, len(states)):
            self.forward(states[i], actions[i])

        # restart gradient here
        self.latent[0].detach()
        self.latent[1].detach()


def learn(alg, alg_params):
    # MDP
    mdp = CartPole()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Policy
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    input_shape = mdp.info.observation_space.shape
    approximator_params = dict(network=Network,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               n_features=2, use_cuda=False)

    # Agent
    agent = alg(mdp.info, pi, RecurrentApproximator,
                approximator_params=approximator_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=500, n_steps_per_fit=5)

    return agent


def test_drqn():
    params = dict(batch_size=1, n_approximators=1, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50)
    approximator = learn(DRQN, params).approximator

    w = approximator.get_weights()
    # w_test = np.array([-0.15894288, 0.47257397, 0.05482405, 0.5442066,
    #                    -0.56469935, -0.07374532, -0.0706185, 0.40790945,
    #                    0.12486243])
    #
    # assert np.allclose(w, w_test)


if __name__ == '__main__':
    test_drqn()
