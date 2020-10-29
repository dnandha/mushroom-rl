import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DRQN
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric.recurrent_torch_approximator import *
from mushroom_rl.utils.parameters import Parameter


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dims, **kwargs):
        super().__init__()

        self._n_input = input_shape[-1]
        self._n_output = output_shape[0]

        self._h1 = nn.RNN(self._n_input, self._n_output)

        self.float()

    def forward(self, state, action=None, **kwargs):

        if "latent" in kwargs:
            latent = kwargs["latent"]
        else:
            latent = None

        # limit observability
        q = state.unsqueeze(1)[:, :, :self._n_input].float()
        q, latent = self._h1(q, latent)
        q = q.squeeze(1)

        if action is None:
            return q, latent
        else:
            action = action.long()
            q_acted = torch.squeeze(q.squeeze(1).gather(1, action))

            return q_acted, latent


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
                               input_shape=[1],
                               latent_dims=0,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               use_cuda=False)

    # Agent
    agent = alg(mdp.info, pi, RecurrentApproximator,
                approximator_params=approximator_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=1, n_steps_per_fit=1)

    return agent


def test_drqn():
    params = dict(batch_size=1, n_approximators=1, initial_replay_size=1,
                  max_replay_size=500, target_update_frequency=50)
    approximator = learn(DRQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([0.29748732, -0.25482982, -0.11192599, 0.27099025,
                       -0.5435388, 0.3462469, -0.11877558, 0.2937234,
                       0.08026147, -0.07069314, 0.16013438, 0.02848172,
                       0.2108646, -0.22499397, -0.04209387, -0.05197728,
                       0.08368343, -0.0023064])

    assert np.allclose(w, w_test)
