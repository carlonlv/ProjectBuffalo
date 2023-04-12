""" This module contains the algorithm class for update rules to decide when to update the model.
"""

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset

from ..utility import NonnegativeInt, PositiveFlt, PositiveInt


class ReplayBuffer(object):
    """ Replay Buffer for Experience Replay."""

    def __init__(self, length: PositiveInt):
        """ Initialize the Replay Buffer.

        :param length: The maximum length of storage for the Replay Buffer.
        """
        self.experience_replay = deque(maxlen=length)

    def collect(self, experience: Any):
        """ Collect the experience into the Replay Buffer.
        """
        self.experience_replay.append(experience)

    def sample(self, sample_size: PositiveInt):
        """ Sample a batch of experiences from the Replay Buffer.

        :param sample_size: The size of the batch of experiences to be sampled. If the size is larger than the size of the Replay Buffer, the size of the Replay Buffer will be used instead.
        :return: A batch of experiences.
        """
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        return sample


class OnlineUpdateRule(ABC):
    """ Base class for online update rules.
    """

    @abstractmethod
    def clear_logs(self):
        """ Clear the logs.
        """
        raise NotImplementedError

    @abstractmethod
    def init(self, dataset: Dataset, model: nn.Module, optimizer: Any, loss_func: Any):
        """ Initialize the online update rule based on observed dataset, model, optimizer, and loss function.
        """
        raise NotImplementedError

    @abstractmethod
    def decide(self, t_index: NonnegativeInt) -> bool:
        """ Decide whether to update the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_epochs(self) -> PositiveInt:
        """ Get the number of epochs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_clip_grad(self) -> PositiveFlt:
        """ Get the clip gradient.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, obs, residuals):
        """ Update the online update rule.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_indices(self, t_index: NonnegativeInt) -> list:
        """ Get the indices of the training data.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_train_stats(self, t_index: NonnegativeInt, train_loss: PositiveFlt, train_resid: pd.DataFrame):
        """ Collect the training statistics.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_test_stats(self, t_index: NonnegativeInt, test_loss: PositiveFlt, test_resid: pd.DataFrame):
        """ Collect the testing statistics.
        """
        raise NotImplementedError
