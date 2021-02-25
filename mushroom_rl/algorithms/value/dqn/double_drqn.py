from mushroom_rl.algorithms.value.dqn import AbstractDRQN, DoubleDQN


class DoubleDRQN(AbstractDRQN, DoubleDQN):

    def _next_q(self, next_state, absorbing):
        q = super(DoubleDRQN, self)._next_q(next_state, absorbing)
        self.approximator.model.network.reset_latent()
        return q
