from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DDPG
from mushroom_rl.utils.replay_memory import ReplayMemory2


class MemoryDDPG(DDPG):
    """

    """
    def __init__(self, mdp_info, policy_class, policy_params, actor_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, unroll_steps=4,
                 policy_delay=1, critic_fit_params=None,
                 actor_predict_params=None, critic_predict_params=None,
                 replay_memory=None):
        """
        Constructor.

        Args:
            unroll_steps (int): length of training sequences;
            replay_memory (ReplayMemory2, None): the replay memory;

        """

        super().__init__(mdp_info, policy_class, policy_params, actor_params,
                         actor_optimizer, critic_params, batch_size,
                         initial_replay_size, max_replay_size, tau,
                         policy_delay, critic_fit_params, actor_predict_params,
                         critic_predict_params)

        if replay_memory is None:
            self._replay_memory = ReplayMemory2(initial_replay_size,
                                                max_replay_size,
                                                unroll_steps)
        else:
            self._replay_memory = replay_memory

    def fit(self, dataset, **kwargs):
        # reset target latent
        self._target_actor_approximator.model.network.reset_latent()
        self._target_critic_approximator.model.network.reset_latent()

        # save latent state before resetting it for the fit
        latent_actor = self._actor_approximator.model.network.latent
        # latent_critic = self._critic_approximator.model.network.latent
        self._actor_approximator.model.network.reset_latent()
        self._critic_approximator.model.network.reset_latent()

        super().fit(dataset)

        # set latent back to old value
        self._actor_approximator.model.network.latent = latent_actor
        # self._critic_approximator.model.network.latent = latent_critic

    def _loss(self, state):
        self._actor_approximator.model.network.reset_latent()
        self._critic_approximator.model.network.reset_latent()

        action = self._actor_approximator(state, output_tensor=True,
                                          **self._actor_predict_params)
        q = self._critic_approximator(state, action, output_tensor=True,
                                      **self._critic_predict_params)

        return -q.mean()

    def episode_start(self):
        super().episode_start()
        self._actor_approximator.model.network.reset_latent()
        self._critic_approximator.model.network.reset_latent()
        self._replay_memory.unfinished_episode = list()
