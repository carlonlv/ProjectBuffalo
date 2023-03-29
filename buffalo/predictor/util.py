"""
This module contains helper functions to manipulate predictors.
"""

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..utility import PositiveInt, Prob
from .seasonality import ChisquaredtestSeasonalityDetection


def predict_next_timestamps(timestamps: pd.DatetimeIndex,
                            num_indices: PositiveInt=1,
                            unit: Literal['D', 'H', 'T', 'S', 'L', 'U', 'N']='D',
                            max_period: Optional[PositiveInt]=None) -> Optional[PositiveInt]:
    """
    Predict the occurrence of next timestamps. Assuming the timestamps

    :param timestamps: Past time stamps in Datetime indices.
    :param unit: The unit is used for extrapolation. D or days: days, H or hours: hours, T or min or minutes: minutes, S or seconds: seconds, L or ms or milliseconds: milliseconds, U or us or microseconds: microseconds, N or ns or nanoseconds: nanoseconds.
    :param num_indices: The future number of indices to be extrapolated.
    :param max_period: Maximum period allowed.
    :return: The predicted future timestamps.
    """
    tmp = pd.DataFrame({'ts': timestamps.sort_values()})
    tmp['ts'] = tmp['ts'].diff()
    tmp['ts'] = tmp['ts'].apply(lambda x: x.total_seconds())
    if unit == 'D':
        tmp['ts'] /= 86400
    elif unit == 'H':
        tmp['ts'] /= 3600
    elif unit == 'T':
        tmp['ts'] /= 60
    elif unit == 'L':
        tmp['ts'] *= 1000
    elif unit == 'U':
        tmp['ts'] *= 1000000
    elif unit == 'N':
        tmp['ts'] *= 1000000000
    tmp = tmp.dropna()
    seasonality_detector = ChisquaredtestSeasonalityDetection(max_period = max_period)
    freqs, fitted_model = seasonality_detector.fit(tmp)['ts']
    predictors = seasonality_detector.get_harmonic_exog(num_indices, freqs, len(timestamps))
    predicted =  np.cumsum(np.round(fitted_model.predict(predictors)))
    if unit == 'D':
        predicted = pd.Series([pd.Timedelta(days=x) for x in predicted])
    elif unit == 'H':
        predicted = pd.Series([pd.Timedelta(hours=x) for x in predicted])
    elif unit == 'T':
        predicted = pd.Series([pd.Timedelta(minutes=x) for x in predicted])
    elif unit == 'L':
        predicted = pd.Series([pd.Timedelta(milliseconds=x) for x in predicted])
    elif unit == 'U':
        predicted = pd.Series([pd.Timedelta(microseconds=x) for x in predicted])
    elif unit == 'N':
        predicted = pd.Series([pd.Timedelta(nanoseconds=x) for x in predicted])
    else:
        predicted = pd.Series([pd.Timedelta(seconds=x) for x in predicted])
    return timestamps.max() + predicted

def align_dataframe_by_time(target_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align the other dataframe to the first dataframe. by index, where indices are assumed to be of type Timestamp.

    :param target_df: The first dataframe.
    :param other_df: The second dataframe.
    :return: The concatenated dataframe by rows.
    """
    assert isinstance(target_df.index, pd.DatetimeIndex) and isinstance(other_df.index, pd.DatetimeIndex)
    other_df = other_df.reindex(target_df.index).sort_index().ffill()
    return pd.concat([target_df, other_df], axis=1)

class TimeSeries:
    """
    Time Series Data Type that uses dataframes as input, and convert them to pytorch tensors.
    """

    class TimeSeriesData(Dataset):
        """
        Time series Data that is loaded into memory. All operations involving time series data preserves ordering.
        """
        def __init__(self, data: torch.Tensor, seq_len: PositiveInt, target_cols: torch.Tensor):
            """
            Initializer for Time Series Data. The row of data is the time dimension. Assuming time in ascending order(past -> future).

            :param data: The concatenated pytorch tensor. The first column is time series of interest.
            :param seq_len: The length of sequence, the last row contains label.
            :param target_cols: The column index for labels.
            """
            self.data = data
            self.seq_len = seq_len
            self.target_cols = target_cols

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, index):
            start_index = index
            end_index = index + self.seq_len
            return self.data[start_index:end_index,:], self.data[end_index,self.target_cols]

    def __init__(
            self,
            endog: pd.DataFrame,
            exog: Optional[pd.DataFrame],
            seq_len: PositiveInt,
            batch_size: PositiveInt,
            pin_memory: bool,
            pin_memory_device: str) -> None:
        """
        Intialize time series data.

        :param endog: The endogenous variable. The row of data is the time dimension.
        :param exog: The exogenous variable The row of data is the time dimension. The exogenous variable must be enforced such that information is available before the timestamps for endog variables. Exogenous time series with the same timestamps are not assumed to be available for prediction, so only past timestamps are used.
        :param seq_len: The length of past information, can only focus on the endogenous variable.
        :param batch_size: The size of batch for training.
        :param pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
        :param pin_memory_device: The data loader will copy Tensors into device pinned memory before returning them if pin_memory is set to true.
        """
        assert endog.shape[0] == exog.shape[0]

        self.endog = endog.sort_index(ascending=True)
        self.exog = exog.sort_index(ascending=True)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device

        endog_array = torch.Tensor(self.endog)
        exog_array = torch.Tensor(self.exog)
        target_cols = torch.arange(0, endog_array.shape[1])
        self.dataset = self.TimeSeriesData(torch.cat((endog_array, exog_array), dim=0), self.seq_len, target_cols)

    def get_splitted_dataset(self, train_ratio: Prob, test_ratio: Prob, include_valid: bool) -> List[Dataset]:
        """
        Return splitted data set into training set, testing set and validation set.

        :param train_ratio: A positive float from 0 to 1.
        :param test_ratio: A positive float from 0 to 1.
        :param include_valid: Whether to split validation set, the remainder from train_ratio and test_ratio is used.
        :return: Splitted datasets.
        """
        assert train_ratio + test_ratio <= 1

        train_size = int(self.dataset.shape[0] * train_ratio)
        test_size = int(self.dataset.shape[0] * test_ratio)
        splitted_size = [train_size, test_size]

        if include_valid:
            valid_size = self.dataset.shape[0] - train_size - test_size
            splitted_size.append(valid_size)

        return random_split(self.dataset, splitted_size)

    def get_dataset_loader(
            self,
            splitted_dataset: List[Dataset],
            batch_sizes: Union[PositiveInt, List[PositiveInt]]):
        """
        Return dataset loader for training set, testing set and validation set.

        :param splitted_dataset: Returned value from get_splitted_dataset.
        :param batch_sizes: Can be a single value or a list of values. If a list of values is provided, each of the value is mapped to the Dataset from splitted_dataset.
        """
        if isinstance(batch_sizes, list):
            assert len(batch_sizes) == len(splitted_dataset)
        else:
            batch_sizes = [batch_sizes] * len(splitted_dataset)

        return [DataLoader(x,
                           batch_sizes[i],
                           pin_memory=self.pin_memory,
                           pin_memory_device=self.pin_memory_device) for i, x in enumerate(splitted_dataset)]
