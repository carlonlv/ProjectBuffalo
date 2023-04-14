""" This module contains the algorithm class for update rules to decide when to update the model.
"""

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

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
    def __init__(self) -> None:
        self.update_logs = pd.DataFrame()
        self.train_logs = pd.DataFrame()
        self.train_record = pd.DataFrame()
        self.test_logs = pd.DataFrame()
        self.train_residuals = pd.DataFrame()
        self.test_residuals = pd.DataFrame()
        self.obs = None

    def collect_obs(self, obs: Dataset):
        """ Collect the observations.

        :param obs: The observations.
        """
        if self.obs is None:
            self.obs = obs
        else:
            self.obs = ConcatDataset([self.obs, obs])

    def clear_logs(self):
        """ Clear the logs.
        """
        self.update_logs = pd.DataFrame()
        self.train_logs = pd.DataFrame()
        self.test_logs = pd.DataFrame()
        self.train_record = pd.DataFrame()
        self.train_residuals = pd.DataFrame()
        self.test_residuals = pd.DataFrame()

    def collect_train_stats(self, t_index: NonnegativeInt, train_loss: PositiveFlt, train_resid: pd.DataFrame, train_record: pd.DataFrame):
        """ Collect the training statistics.

        :param t_index: The index of the current time step.
        :param train_loss: The training loss.
        :param train_resid: The training residuals.
        :param train_record: The training record.
        """
        self.train_logs = pd.concat((self.train_logs, pd.DataFrame({'train_loss': train_loss}, index=[t_index])), axis=0)
        self.train_record = pd.concat((self.train_record, train_record.assign(t_index=t_index)), axis=0)
        self.train_residuals = pd.concat((self.train_residuals, train_resid.assign(t_index=t_index)), axis=0)

    def collect_test_stats(self, t_index: NonnegativeInt, test_loss: PositiveFlt, test_resid: pd.DataFrame):
        """ Collect and store the testing statistics.

        :param t_index: The index of the current time step.
        :param test_loss: The testing loss.
        :param test_resid: The testing residuals.
        """
        self.test_logs = pd.concat((self.test_logs, pd.DataFrame({'test_loss': test_loss}, index=[t_index])), axis=0)
        self.test_residuals = pd.concat((self.test_residuals, test_resid.assign(t_index=t_index)), axis=0)

    @abstractmethod
    def get_epochs(self, t_index: NonnegativeInt) -> PositiveInt:
        """ Get the number of epochs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_clip_grad(self) -> PositiveFlt:
        """ Get the clip gradient.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_indices(self, t_index: NonnegativeInt) -> list:
        """ Get the indices of the training data.
        """
        raise NotImplementedError


class IncrementalBatchGradientDescent(OnlineUpdateRule):
    """ Incremental Batch Gradient Descent Update Rule.
    """

    def __init__(self, epochs_per_update: PositiveInt, update_freq: PositiveInt, clip_grad_norm: Optional[PositiveFlt]=None, pretrained=False) -> None:
        """ Initialize the Incremental Batch Gradient Descent Update Rule. Incremental Batch Gradient Descent Update Rule updates the model with a pre-defined frequency and the new dataset is sweeped through the model for a pre-defined number of epochs.
        
        :param epochs_per_update: The number of epochs per update.
        :param update_freq: The frequency of 
        updates.
        :param clip_grad_norm: The clip gradient norm. If None, no clipping will be performed.
        :param pretrained: Whether the model is pretrained. If not pretrained, the model will be trained upon encountering the first batch of data.
        """
        super().__init__()
        self.epochs_per_update = epochs_per_update
        self.clip_grad_norm = clip_grad_norm
        self.update_freq = update_freq
        self.next_update = 0
        self.pretrained = pretrained
        self.name = 'IncrementalBatchGradientDescent'
        self.info = {
            'epochs_per_update': epochs_per_update,
            'update_freq': update_freq,
            'clip_grad_norm': clip_grad_norm,
            'pretrained': pretrained,
            'name': self.name
        }

    def __str__(self) -> str:
        """ Get the string representation of the Incremental Batch Gradient Descent Update Rule.
        """
        return rf'Incremental Batch Gradient Descent Update Rule ( \n epochs_per_update={self.epochs_per_update}, \n update_freq={self.update_freq}, \n clip_grad_norm={self.clip_grad_norm} \n )'

    def get_epochs(self, t_index: NonnegativeInt) -> PositiveInt:
        """ Get the number of epochs. This update rule returns pre-defined number of epochs per update.

        :param t_index: The index of the current time step.
        """
        return self.epochs_per_update

    def get_clip_grad(self) -> PositiveFlt:
        """ Get the clip gradient. This update rule returns pre-defined clip gradient."""
        return self.clip_grad_norm

    def get_train_indices(self, t_index: NonnegativeInt) -> list:
        """ Get the indices of the training data. This update rule returns all the indices.

        :param t_index: The index of the current time step.
        :return: A list of indices to train on.
        """
        if not self.pretrained and self.next_update == 0:
            end_index = t_index+1
            self.next_update = t_index + self.update_freq
            self.update_logs = pd.concat((self.update_logs, pd.DataFrame({'t_index': [t_index], 'start_index': 0, 'end_index': end_index-1})))
            return range(end_index)
        if t_index >= self.next_update:
            start_index = t_index-self.update_freq+1
            end_index = t_index+1
            self.update_logs = pd.concat((self.update_logs, pd.DataFrame({'t_index': [t_index], 'start_index': start_index, 'end_index': end_index-1})))
            self.next_update = t_index + self.update_freq
            return range(start_index, end_index)
        else:
            return range(0)
