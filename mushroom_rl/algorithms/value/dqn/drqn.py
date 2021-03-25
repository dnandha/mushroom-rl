import numpy as np

from mushroom_rl.algorithms.value.dqn import AbstractDQN, DQN
from mushroom_rl.utils.replay_memory import SequentialReplayMemory,\
    ReplayMemory2


class AbstractDRQN(AbstractDQN):
    def __init__(self, mdp_info, policy, approximator, unroll_steps=4,
                 pad_episodes=False, dummy=None, replay_memory=None,
                 initial_replay_size=500, max_replay_size=5000, **params):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function. It must provide a public property ´latent´ and a
               public method ´reset_latent()´.
            unroll_steps (int): number of serial elements per sample; also the
                minimum length for an episode to be stored.
            sequential_updates (bool): if True whole episodes are sampled,
                therefore, too short episodes will be padded with `dummy`s to
                always meet the length of `unroll_steps`.
            dummy (tuple): A dummy sample to be used to pad episodes when using
                 `sequential_updates`.
        """

        # if replay_memory is not None:
        #     assert isinstance(replay_memory, SequentialReplayMemory),\
        #         "replay_buffer must be of type SequentialReplayMemory."
        # else:
        #     replay_memory = SequentialReplayMemory(initial_replay_size,
        #                                            max_replay_size,
        #                                            unroll_steps,
        #                                            pad_episodes,
        #                                            dummy)
        if replay_memory is None:
            replay_memory = ReplayMemory2(initial_replay_size,
                                          max_replay_size,
                                          unroll_steps)

        super().__init__(mdp_info, policy, approximator,
                         replay_memory=replay_memory, **params)

        # # make sure the _dummy matches the real data
        # if replay_memory.pad and not\
        #         np.shape(replay_memory.dummy[0]) ==\
        #         np.shape(replay_memory.dummy[3]) ==\
        #         mdp_info.observation_space.shape:
        #     raise ValueError('Dummy observations don\'t match the real ones.')

    def fit(self, dataset):
        # reset target latent
        self.target_approximator.model.network.reset_latent()

        # save latent state before resetting it for the fit
        latent = self.approximator.model.network.latent
        self.approximator.model.network.reset_latent()

        super().fit(dataset)

        # set latent back to old value
        self.approximator.model.network.latent = latent

    def episode_start(self):
        super().episode_start()
        self.approximator.model.network.reset_latent()
        self._replay_memory.unfinished_episode = list()


class DRQN(AbstractDRQN, DQN):
    """
    Deep Recurrent Q-Network algorithm.
    "Deep Recurrent Q-Learning for Partially Observable MDPs".
    Hausknecht M. et al.. 2015.

    """
    pass
