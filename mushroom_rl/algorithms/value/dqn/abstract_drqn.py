from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.utils.replay_memory import ReplayMemory2


class AbstractDRQN(AbstractDQN):
    def __init__(self, mdp_info, policy, approximator, unroll_steps=4,
                 replay_memory=None, initial_replay_size=500,
                 max_replay_size=5000, **params):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function. It must provide a public property ´latent´ and a
               public method ´reset_latent()´.
            unroll_steps (int): number of serial elements per sample; also the
                minimum length for an episode to be stored.
        """

        if replay_memory is None:
            replay_memory = ReplayMemory2(initial_replay_size,
                                          max_replay_size,
                                          unroll_steps)

        super().__init__(mdp_info, policy, approximator,
                         replay_memory=replay_memory, **params)

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
