import torch
import numpy as np

from mushroom_rl.policy.td_policy import EpsGreedy


class MemoryNetwork(torch.nn.Module):
    """
    This class provides base functionality to handle the
    latent state correctly.
    """
    def __init__(self, zero_latent=None):
        super().__init__()
        self._zero_latent = zero_latent
        self.latent = zero_latent

        self.float()

    def reset_latent(self):
        self.latent = self._zero_latent


class MemoryQNetwork(MemoryNetwork):
    """
    Memory network class for Q-networks that work with the DRQN algorithm.
    """
    def forward(self, state, action=None, **_):
        q = self._forward(self.memory_pass(state))

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(-1, action), -1)

            return q_acted

    def memory_pass(self, state):
        raise NotImplementedError

    def _forward(self, hidden):
        raise NotImplementedError


class MemoryEpsGreedy(EpsGreedy):
    """
    This class extends the EpsGreedy class such that the latent state of the
    network is updated in every call of ´draw_action´.

    Approximator network should provide a public method ´memory_pass´ for best
    performance.
    """
    def draw_action(self, state):
        s = np.expand_dims(state, axis=(0, 1))
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(s, **self._predict_params)
            q = q.squeeze().squeeze()
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        # update latent memory anyway
        self.update_memory(s)

        return np.array([np.random.choice(self._approximator.n_actions)])

    def set_q(self, approximator):
        super(MemoryEpsGreedy, self).set_q(approximator)
        if hasattr(self._approximator.model.network, 'memory_pass'):
            self.update_memory = lambda s: \
                self._approximator.model.network.memory_pass(
                    torch.from_numpy(s))
        else:
            self.update_memory = self._approximator.predict

    update_memory = None
