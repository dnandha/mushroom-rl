import numpy as np

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.utils.replay_memory import SequentialReplayMemory


class DRQN(AbstractDQN):
    """
    Deep Recurrent Q-Network algorithm.
    "Deep Recurrent Q-Learning for Partially Observable MDPs".
    Hausknecht M. et al.. 2015.

    """
    def __init__(self, mdp_info, policy, approximator, approximator_params,
                 batch_size, target_update_frequency, unroll_steps,
                 replay_memory=None, initial_replay_size=500,
                 max_replay_size=5000, fit_params=None, clip_reward=True,
                 sequential_updates=False, dummy=None, double_dqn=False):
        """
        Constructor.

        Args:
            unroll_steps (int): number of serial elements per sample; also the
                minimum length for an episode to be stored.
            sequential_updates (bool): if True whole episodes are sampled,
                therefore, too short episodes will be padded with `dummy`s to
                always meet the length of `unroll_steps`.
            dummy (tuple): A dummy sample to be used to pad episodes when using
                 `sequential_updates`.
        """

        if replay_memory is not None:
            assert isinstance(replay_memory, SequentialReplayMemory),\
                "replay_buffer must be of type SequentialReplayMemory."
        else:
            replay_memory = SequentialReplayMemory(initial_replay_size,
                                                   max_replay_size,
                                                   unroll_steps,
                                                   sequential_updates,
                                                   dummy)

        super().__init__(mdp_info, policy, approximator,
                         approximator_params, batch_size,
                         target_update_frequency, replay_memory,
                         initial_replay_size, max_replay_size,
                         fit_params, clip_reward)

        # make sure the dummy matches the real data
        if sequential_updates and\
                (len(replay_memory.dummy) != 6 or
                 np.shape(dummy[0]) != mdp_info.observation_space.shape or
                 np.shape(dummy[3]) != mdp_info.observation_space.shape):
            raise ValueError('Padding dummy does not match requirements.')

        # double DQN
        if double_dqn:
            self._double = self._double_q
        else:
            self._double = lambda _, q: q

    def fit(self, dataset):
        # reset target latent
        self.target_approximator.model.network.reset_latent()

        # save latent state before resetting it for the fit
        latent = self.approximator.model.network.latent
        self.approximator.model.network.reset_latent()

        loss = super(DRQN, self).fit(dataset)

        # set latent back to old value
        self.approximator.model.network.latent = latent

        return loss

    def episode_start(self):
        super().episode_start()
        self.approximator.model.network.reset_latent()
        self._replay_memory.unfinished_episode = list()

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state)
        q = self._double(next_state, q)

        if np.any(absorbing):
            shape = list(q.shape)
            shape[-1] = 1
            q *= 1 - absorbing.reshape(shape)

        return np.max(q, axis=-1)

    def _double_q(self, next_state, q):
        max_a = np.argmax(q, axis=1)
        return self.target_approximator.predict(next_state, max_a)