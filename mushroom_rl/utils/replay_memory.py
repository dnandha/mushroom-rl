import numpy as np

from mushroom_rl.core import Serializable
from mushroom_rl.utils.parameters import to_parameter


class ReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, initial_size, max_size):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory;
            max_size (int): maximum number of elements that the replay memory
                can contain.

        """
        self._initial_size = initial_size
        self._max_size = max_size

        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _idx='primitive!',
            _full='primitive!',
            _states='pickle!',
            _actions='pickle!',
            _rewards='pickle!',
            _next_states='pickle!',
            _absorbing='pickle!',
            _last='pickle!'
        )

    def add(self, dataset, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        assert n_steps_return > 0

        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset[i][2]
            j = 0
            while j < n_steps_return - 1:
                if dataset[i + j][5]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset[i + j][2]
            else:
                self._states[self._idx] = dataset[i][0]
                self._actions[self._idx] = dataset[i][1]
                self._rewards[self._idx] = reward

                self._next_states[self._idx] = dataset[i + j][3]
                self._absorbing[self._idx] = dataset[i + j][4]
                self._last[self._idx] = dataset[i + j][5]

                self._idx += 1
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0

                i += 1

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last)

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        self._states = [None for _ in range(self._max_size)]
        self._actions = [None for _ in range(self._max_size)]
        self._rewards = [None for _ in range(self._max_size)]
        self._next_states = [None for _ in range(self._max_size)]
        self._absorbing = [None for _ in range(self._max_size)]
        self._last = [None for _ in range(self._max_size)]

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    def _post_load(self):
        if self._full is None:
            self.reset()


class SumTree(object):
    """
    This class implements a sum tree data structure.
    This is used, for instance, by ``PrioritizedReplayMemory``.

    """
    def __init__(self, max_size):
        """
        Constructor.

        Args:
            max_size (int): maximum size of the tree.

        """
        self._max_size = max_size
        self._tree = np.zeros(2 * max_size - 1)
        self._data = [None for _ in range(max_size)]
        self._idx = 0
        self._full = False

    def add(self, dataset, priority, n_steps_return, gamma):
        """
        Add elements to the tree.

        Args:
            dataset (list): list of elements to add to the tree;
            priority (np.ndarray): priority of each sample in the dataset;
            n_steps_return (int): number of steps to consider for computing n-step return;
            gamma (float): discount factor for n-step return.

        """
        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset[i][2]

            j = 0
            while j < n_steps_return - 1:
                if dataset[i + j][5]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset[i + j][2]
            else:
                d = list(dataset[i])
                d[2] = reward
                d[3] = dataset[i + j][3]
                d[4] = dataset[i + j][4]
                d[5] = dataset[i + j][5]
                idx = self._idx + self._max_size - 1

                self._data[self._idx] = d
                self.update([idx], [priority[i]])

                self._idx += 1
                if self._idx == self._max_size:
                    self._idx = 0
                    self._full = True

                i += 1

    def get(self, s):
        """
        Returns the provided number of states from the replay memory.

        Args:
            s (float): the value of the samples to return.

        Returns:
            The requested sample.

        """
        idx = self._retrieve(s, 0)
        data_idx = idx - self._max_size + 1

        return idx, self._tree[idx], self._data[data_idx]

    def update(self, idx, priorities):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            idx (np.ndarray): indexes of the transitions in the dataset;
            priorities (np.ndarray): priorities of the transitions.

        """
        for i, p in zip(idx, priorities):
            delta = p - self._tree[i]

            self._tree[i] = p
            self._propagate(delta, i)

    def _propagate(self, delta, idx):
        parent_idx = (idx - 1) // 2

        self._tree[parent_idx] += delta

        if parent_idx != 0:
            self._propagate(delta, parent_idx)

    def _retrieve(self, s, idx):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if self._tree[left] == self._tree[right]:
            return self._retrieve(s, np.random.choice([left, right]))

        if s <= self._tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self._tree[left], right)

    @property
    def size(self):
        """
        Returns:
            The current size of the tree.

        """
        return self._idx if not self._full else self._max_size

    @property
    def max_p(self):
        """
        Returns:
            The maximum priority among the ones in the tree.

        """
        return self._tree[-self._max_size:].max()

    @property
    def total_p(self):
        """
        Returns:
            The sum of the priorities in the tree, i.e. the value of the root
            node.

        """
        return self._tree[0]


class PrioritizedReplayMemory(Serializable):
    """
    This class implements function to manage a prioritized replay memory as the
    one used in "Prioritized Experience Replay" by Schaul et al., 2015.

    """
    def __init__(self, initial_size, max_size, alpha, beta, epsilon=.01):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay
                memory;
            max_size (int): maximum number of elements that the replay memory
                can contain;
            alpha (float): prioritization coefficient;
            beta ([float, Parameter]): importance sampling coefficient;
            epsilon (float, .01): small value to avoid zero probabilities.

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon

        self._tree = SumTree(max_size)

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _alpha='primitive',
            _beta='primitive',
            _epsilon='primitive',
            _tree='pickle!'
        )

    def add(self, dataset, p, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            p (np.ndarray): priority of each sample in the dataset.
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        assert n_steps_return > 0

        self._tree.add(dataset, p, n_steps_return, gamma)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        states = [None for _ in range(n_samples)]
        actions = [None for _ in range(n_samples)]
        rewards = [None for _ in range(n_samples)]
        next_states = [None for _ in range(n_samples)]
        absorbing = [None for _ in range(n_samples)]
        last = [None for _ in range(n_samples)]

        idxs = np.zeros(n_samples, dtype=np.int)
        priorities = np.zeros(n_samples)

        total_p = self._tree.total_p
        segment = total_p / n_samples

        a = np.arange(n_samples) * segment
        b = np.arange(1, n_samples + 1) * segment
        samples = np.random.uniform(a, b)
        for i, s in enumerate(samples):
            idx, p, data = self._tree.get(s)

            idxs[i] = idx
            priorities[i] = p
            states[i], actions[i], rewards[i], next_states[i], absorbing[i],\
                last[i] = data
            states[i] = np.array(states[i])
            next_states[i] = np.array(next_states[i])

        sampling_probabilities = priorities / self._tree.total_p
        is_weight = (self._tree.size * sampling_probabilities) ** -self._beta()
        is_weight /= is_weight.max()

        return np.array(states), np.array(actions), np.array(rewards),\
            np.array(next_states), np.array(absorbing), np.array(last),\
            idxs, is_weight

    def update(self, error, idx):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            error (np.ndarray): errors to consider to compute the priorities;
            idx (np.ndarray): indexes of the transitions in the dataset.

        """
        p = self._get_priority(error)
        self._tree.update(idx, p)

    def _get_priority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self._tree.size > self._initial_size

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.

        """
        return self._tree.max_p if self.initialized else 1.

    def _post_load(self):
        if self._tree is None:
            self._tree = SumTree(self._max_size)


class SequentialReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Deep Recurrent Q-Learning for Partially Observable MDPs"
    by Hausknecht, M. et al..

    """

    def __init__(self, initial_size, max_size, unroll_steps=1,
                 pad=False, dummy=None):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay memory
            max_size (int): maximum number of elements that the replay memory
                can contain.
            unroll_steps (int, 1): number of serial elements per sample; also
                the minimum length for an episode to be stored.
            pad (bool, False): if True, too short episodes will be padded with
                padding states to always meet the length of `unroll_steps`.
            dummy (tuple, None): A dummy sample to be used to pad episodes when
                using `sequential_updates`.

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self._unroll_steps = unroll_steps

        if pad:
            assert dummy is not None and len(dummy) == 6,\
                "For the sequential updates a correct padding dummy must be" \
                "provided."

            self._process_ep = self._pad
            self._dummy = dummy
        else:
            self._process_ep = self._no_pad

        # length of each episode
        self._lengths = list()

        self._size = 0
        self._states = list()
        self._actions = list()
        self._rewards = list()
        self._next_states = list()
        self._absorbing = list()
        self._last = list()

        # save unfinished episode to collect the missing steps
        # in order to only store full episodes
        self.unfinished_episode = list()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _size='primitive',
            _unroll_steps='primitive',
            _full='pickle',
            _states='pickle',
            _actions='pickle',
            _rewards='pickle',
            _next_states='pickle',
            _absorbing='pickle',
            _last='pickle',
            _lengths='pickle'
        )

    @property
    def dummy(self):
        return self._dummy

    @property
    def pad(self):
        return self._process_ep == self._pad

    def add(self, dataset):
        """
        Add full episodes to the replay memory.

        Args:
            dataset (list): episode to add to the replay memory.

        Returns:
            Number of episodes that have been added

        """
        # split dataset into episodes
        episodes, self.unfinished_episode = \
            self._split_dataset(self.unfinished_episode + dataset)

        added = 0
        # add every episode one by one
        for episode in episodes:

            # only store if the episode is long enough but not too long
            if self._unroll_steps <= len(episode) < self._max_size:

                s = list()
                a = list()
                r = list()
                ss = list()
                ab = list()
                last = list()

                for i in range(len(episode)):
                    s.append(episode[i][0])
                    a.append(episode[i][1])
                    r.append(episode[i][2])
                    ss.append(episode[i][3])
                    ab.append(episode[i][4])
                    last.append(episode[i][5])

                # add episode to replay buffer
                self._states.append(s)
                self._actions.append(a)
                self._rewards.append(r)
                self._next_states.append(ss)
                self._absorbing.append(ab)
                self._last.append(last)

                self._size += len(episode)
                self._lengths.append(len(episode))
                while self._size > self._max_size:
                    # remove oldest episode
                    self._size -= len(self._states.pop(0))
                    self._actions.pop(0)
                    self._rewards.pop(0)
                    self._next_states.pop(0)
                    self._absorbing.pop(0)
                    self._last.pop(0)
                    self._lengths.pop(0)

                added += 1

        return added

    def get(self, batch_size):
        """
        Returns the provided number of samples from the replay memory.

        Args:
            batch_size (int): the number of samples to return.

        Returns:
            The samples as batch_size x unroll_steps x sample shape
            or a batch of episodes if self.sequential_updates is True.

        """
        assert batch_size <= self._initial_size, \
            "The batch size should be smaller than the initial size."

        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()

        # randomly selected episodes
        eps = np.random.randint(len(self._states), size=batch_size)

        # randomly selected start indices for EVERY episode
        idx = np.random.randint(0, np.array(self._lengths) -
                                self._unroll_steps + 1)

        for _ in range(self._unroll_steps):

            s_ep = list()
            a_ep = list()
            r_ep = list()
            ss_ep = list()
            ab_ep = list()
            last_ep = list()

            for ep in eps:
                s_ep.append(np.array(self._states[ep][idx[ep]]))
                a_ep.append(self._actions[ep][idx[ep]])
                r_ep.append(self._rewards[ep][idx[ep]])
                ss_ep.append(np.array(self._next_states[ep][idx[ep]]))
                ab_ep.append(self._absorbing[ep][idx[ep]])
                last_ep.append(self._last[ep][idx[ep]])

            s.append(np.array(s_ep))
            a.append(np.array(a_ep))
            r.append(np.array(r_ep))
            ss.append(np.array(ss_ep))
            ab.append(np.array(ab_ep))
            last.append(np.array(last_ep))

            idx += 1

        return np.array(s), np.array(a), np.array(r), np.array(ss),\
            np.array(ab), np.array(last)

    def reset(self):
        """
        Reset the replay memory.

        """
        self._size = 0
        self._states = list()
        self._actions = list()
        self._rewards = list()
        self._next_states = list()
        self._absorbing = list()
        self._last = list()
        self._lengths = list()

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self._size > self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._size

    def _split_dataset(self, dataset):
        """
        Splits the dataset into a list of full episodes and the rest.

        Args:
            dataset: The dataset to be split into distinct episodes.

        Returns:
            A list of full episodes in the dataset and the unfinished episode

        """
        # split data into episodes based on last flag
        indices = np.where(np.array(dataset, dtype=object)[:, -1])[0].tolist()

        # calculate start and end values from indices
        args = (0,) + tuple(data + 1 for data in indices)

        episodes = []
        end = 0
        for start, end in zip(args, args[1:]):
            episodes.append(self._process_ep(dataset, start, end))

        return episodes, dataset[end:len(dataset)]

    @staticmethod
    def _no_pad(dataset, start, end):
        return dataset[start:end]

    def _pad(self, dataset, start, end):
        padding = self._unroll_steps + start - end
        return [self._dummy] * padding + dataset[start:end]


class ReplayMemory2(ReplayMemory):
    """
    Replay memory, which samples sequences.

    The sequences may be parts of different rollouts but the sub-sequences
        are parts of real rollout.

    """

    def __init__(self, initial_size, max_size, unroll_steps=1):
        """
        Constructor.

        Args:
            unroll_steps (int, 1): number of serial elements per sample; also
                the minimum length for an episode to be stored.
        """
        super().__init__(initial_size, max_size)
        self._unroll_steps = unroll_steps
        self._add_save_attr(_unroll_steps='primitive')

    def get(self, n_samples):
        s = list()
        a = list()
        r = list()
        ss = list()
        ab = list()
        last = list()

        idx = np.random.randint(self.size - self._unroll_steps + 1,
                                size=n_samples)

        for _ in range(self._unroll_steps):
            s_ep = list()
            a_ep = list()
            r_ep = list()
            ss_ep = list()
            ab_ep = list()
            last_ep = list()

            for i in idx:
                s_ep.append(np.array(self._states[i]))
                a_ep.append(self._actions[i])
                r_ep.append(self._rewards[i])
                ss_ep.append(np.array(self._next_states[i]))
                ab_ep.append(self._absorbing[i])
                last_ep.append(self._last[i])

            s.append(np.array(s_ep))
            a.append(np.array(a_ep))
            r.append(np.array(r_ep))
            ss.append(np.array(ss_ep))
            ab.append(np.array(ab_ep))
            last.append(np.array(last_ep))

            idx += 1

        return np.array(s), np.array(a), np.array(r), np.array(ss), \
            np.array(ab), np.array(last)
