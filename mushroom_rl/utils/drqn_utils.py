import torch
import torch.nn as nn
import numpy as np

import warnings

from mushroom_rl.policy.td_policy import EpsGreedy


class MemoryNetwork(nn.Module):
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
