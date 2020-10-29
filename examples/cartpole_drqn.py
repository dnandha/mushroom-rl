import pathlib
import numpy as np
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

import matplotlib.pyplot as plt
from mushroom_rl.algorithms import Agent




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
        q = state.unsqueeze(1)[:, :, :self._n_input].float()
        q, latent = self._h1(q, latent)
        q = self._h2(q.squeeze(1))

        if action is None:
            return q, latent
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted, latent


def experiment():
    # Parameters
    folder_name = str(pathlib.Path.home()) + "/logs/DRQN/"
    training_episodes = 1#00
    evaluation_frequency = 1000
    max_steps = training_episodes * evaluation_frequency
    eval_episodes = 10
    np.random.seed(0)
    angle = np.random.uniform(-np.pi / 8., np.pi / 8., (eval_episodes, 1))
    states = np.concatenate((angle, np.zeros_like(angle)), axis=1)

    # MDP
    mdp = CartPole()

    # Policy
    epsilon = LinearParameter(value=1., threshold_value=.1, n=80000)
    epsilon_test = Parameter(value=0.0)
    pi = EpsGreedy(epsilon=epsilon)

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

    # Algorithm
    core = Core(agent, mdp)

    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    scores = []
    pi.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_episodes=10, quiet=True)
    scores.append(episodes_length(dataset))

    for n_epoch in range(1, max_steps // evaluation_frequency + 1):
        print('##############################################################')
        print('Epoch: ', n_epoch)
        print('--------------------------------------------------------------')
        print('- Learning:')
        # learning step
        pi.set_epsilon(epsilon)
        core.learn(n_episodes=evaluation_frequency, n_episodes_per_fit=20,
                   quiet=True)

        agent.save(folder_name + '/agent_' + str(n_epoch) + '.msh')

        print('- Evaluation:')
        # evaluation step
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(quiet=True, initial_states=states)
        scores.append(episodes_length(dataset))
        print("Mean:", np.mean(episodes_length(dataset)))

        np.save(folder_name + '/scores.npy', scores)

    # plot results
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    x = np.arange(0, evaluation_frequency * (len(mean)), evaluation_frequency)

    plt.figure()
    plt.grid(True)
    plt.title("DRQN Cartpole")
    plt.xlabel("Episodes", fontsize=7)
    plt.ylabel("Episode Length", fontsize=7)

    plt.plot(x, mean, '-', color='gray')

    plt.fill_between(x, mean - std, mean + std,
                     color='gray', alpha=0.2)

    plt.show()

    return scores[-1]


if __name__ == '__main__':
    steps = experiment()
    print('Final episode length: ', steps)
