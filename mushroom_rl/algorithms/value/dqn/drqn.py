from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.utils.replay_memory import SequentialReplayMemory


class DRQN(DQN):
    """
    Deep Recurrent Q-Network algorithm.
    "Deep Recurrent Q-Learning for Partially Observable MDPs".
    Hausknecht M. et al.. 2015.

    """
    def __init__(self, mdp_info, policy, approximator, approximator_params,
                 batch_size, target_update_frequency, unroll_steps,
                 replay_memory=None, initial_replay_size=500,
                 max_replay_size=5000, fit_params=None, n_approximators=1,
                 clip_reward=True, sequential_updates=False):

        if replay_memory is not None:
            assert isinstance(replay_memory, SequentialReplayMemory),\
                "replay_buffer must be of type SequentialReplayMemory."
        else:
            replay_memory = SequentialReplayMemory(initial_replay_size,
                                                   max_replay_size,
                                                   unroll_steps,
                                                   sequential_updates)

        super().__init__(mdp_info, policy, approximator,
                         approximator_params, batch_size,
                         target_update_frequency, replay_memory,
                         initial_replay_size, max_replay_size,
                         fit_params, n_approximators, clip_reward)

    def fit(self, dataset):
        # reset target latent
        self.target_approximator.model.network.reset_latent()

        # save latent state before resetting it for the fit
        latent = self.approximator.model.network.latent
        self.approximator.model.network.reset_latent()

        super(DRQN, self).fit(dataset)

        # set latent back to old value
        self.approximator.model.network.latent = latent

    def episode_start(self):
        super().episode_start()
        self.approximator.model.network.reset_latent()
        self._replay_memory.unfinished_episode = list()
