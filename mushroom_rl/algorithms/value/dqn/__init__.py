from .dqn import AbstractDQN, DQN
from .double_dqn import DoubleDQN
from .averaged_dqn import AveragedDQN
from .maxmin_dqn import MaxminDQN
from .categorical_dqn import CategoricalDQN
from .drqn import AbstractDRQN, DRQN
from .double_drqn import DoubleDRQN


__all__ = ['AbstractDQN', 'DQN', 'DoubleDQN', 'AveragedDQN', 'MaxminDQN',
           'CategoricalDQN', 'AbstractDRQN', 'DRQN', 'DoubleDRQN']
