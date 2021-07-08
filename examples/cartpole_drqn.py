import pathlib
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.utils.dataset import episodes_length
from mushroom_rl.utils.parameters import Parameter, LinearParameter
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.algorithms.value import DRQN
from mushroom_rl.utils.memory import *

import matplotlib.pyplot as plt


class Network(MemoryNetwork):
    def __init__(self, input_shape, output_shape, latent_dims, **_):
        super().__init__()

        self._n_input = input_shape[-1]
        self._n_output = output_shape[0]

        self._h1 = nn.GRU(self._n_input, latent_dims, num_layers=2,
                          batch_first=True)
        self._h2 = nn.Linear(latent_dims, self._n_output)

        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('linear'))

        self.float()

    def memory_pass(self, state):
        q = state[:, :, :self._n_input].float()
        q, self.latent = self._h1(q, self.latent)
        return q

    def _forward(self, hidden):
        return self._h2(hidden)

    def reset_latent(self):
        self.latent = None


def experiment():
    # Parameters
    folder_name = str(pathlib.Path.home()) + "/logs/DRQN/test2/"
    training_episodes = 100
    evaluation_frequency = 1000
    eval_episodes = 3
    initial_size = 200

    # MDP
    mdp = CartPole()

    # Policy
    epsilon = LinearParameter(value=1., threshold_value=.1, n=80000)
    epsilon_test = Parameter(value=0.0)
    epsilon_random = Parameter(value=1.0)
    pi = MemoryEpsGreedy(epsilon=epsilon)

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
                  unroll_steps=10,
                  initial_replay_size=initial_size,
                  max_replay_size=1000,
                  target_update_frequency=1000)

    agent = DRQN(mdp.info, pi, TorchApproximator,
                 approximator_params=approximator_params, **params)

    # Algorithm
    core = Core(agent, mdp)

    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    scores = []
    pi.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_episodes=eval_episodes, quiet=True)
    scores.append(episodes_length(dataset))

    pi.set_epsilon(epsilon_random)
    core.learn(n_steps=initial_size, n_steps_per_fit=initial_size, quiet=True)

    for n_epoch in range(1, training_episodes + 1):
        print('##############################################################')
        print('Epoch: ', n_epoch)
        print('--------------------------------------------------------------')
        print('- Learning:')
        # learning step
        pi.set_epsilon(epsilon)
        core.learn(n_episodes=evaluation_frequency, n_steps_per_fit=1,
                   quiet=True)

        agent.save(folder_name + '/agent_' + str(n_epoch) + '.msh')

        print('- Evaluation:')
        # evaluation step
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_episodes=eval_episodes, quiet=True)
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
