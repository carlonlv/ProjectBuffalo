"""
This module contains helper functions to manipulate predictors.
"""

from typing import List, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..utility import PositiveInt, Prob
from .seasonality import ChisquaredtestSeasonalityDetection

ALL_UNITS = ['D', 'H', 'T', 'S', 'L', 'U', 'N']

def predict_next_timestamps(timestamps: pd.DatetimeIndex,
                            num_indices: PositiveInt=1,
                            max_period: Optional[PositiveInt]=None) -> Optional[PositiveInt]:
    """
    Predict the occurrence of next timestamps. Assuming the timestamps

    :param timestamps: Past time stamps in Datetime indices.
    :param num_indices: The future number of indices to be extrapolated.
    :param max_period: Maximum period allowed.
    :return: The predicted future timestamps.
    """
    tmp = pd.DataFrame({'ts': timestamps.sort_values()})
    tmp['ts'] = tmp['ts'].diff()
    tmp.dropna(inplace=True)
    units = tmp['ts'].apply(lambda x: x.resolution_string).unique()
    tmp['ts'] = tmp['ts'].apply(lambda x: x.total_seconds())
    if len(units) > 1:
        unit = ALL_UNITS[max([x for x in range(len(ALL_UNITS)) if ALL_UNITS[x] in units])]
        warn(f'Multiple resolutions in timestamps detected, using smallest time unit {unit}.')
    else:
        unit = units[0]
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
    seasonality_detector = ChisquaredtestSeasonalityDetection(max_period = max_period)
    freqs, fitted_model = seasonality_detector.fit(tmp, verbose=False)['ts']
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

def align_dataframe_by_time(target_df: pd.DataFrame,
                            other_df: pd.DataFrame,
                            max_period: Optional[PositiveInt]=None) -> pd.DataFrame:
    """
    Align the other dataframe to the first dataframe. by index, where indices are assumed to be of type Timestamp.

    :param target_df: The first dataframe.
    :param other_df: The second dataframe. This dataframe will align the target_df in terms of time.
    :param max_period: Maximum period allowed. This parameter is used when determining expiry time of the last observation for other_df.
    :return: The concatenated dataframe by rows.
    """
    assert isinstance(target_df.index, pd.DatetimeIndex) and isinstance(other_df.index, pd.DatetimeIndex)
    expire_time = predict_next_timestamps(other_df.index, max_period=max_period)
    if other_df.index.tz is None:
        other_df.index = other_df.index.tz_localize(target_df.index.tz)
        expire_time.iloc[0] = expire_time.iloc[0].tz_localize(target_df.index.tz)
    else:
        other_df.index = other_df.index.tz_convert(target_df.index.tz)
        expire_time.iloc[0] = expire_time.iloc[0].tz_convert(target_df.index.tz)
    missing_indices = target_df.index.difference(other_df.index)
    if len(missing_indices) > 0:
        other_df = pd.concat([other_df, pd.DataFrame(index=missing_indices)], axis=0, join='outer').sort_index().ffill()
    other_df = other_df.reindex(target_df.index)
    other_df = other_df[other_df.index < expire_time.iloc[0]]
    result = pd.concat([target_df, other_df], axis=1, verify_integrity=True)
    result = result.dropna()
    return result

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

        endog_array = torch.Tensor(self.endog.to_numpy())
        exog_array = torch.Tensor(self.exog.to_numpy())
        target_cols = torch.arange(0, endog_array.shape[1])
        self.dataset = self.TimeSeriesData(torch.cat((endog_array, exog_array), dim=1), self.seq_len, target_cols)

    def get_traintest_splitted_dataloader(self, train_ratio: Prob, batch_size: PositiveInt) -> List[Dataset]:
        """
        Return splitted data set into training set, testing set and validation set.

        :param train_ratio: A positive float from 0 to 1.
        :param test_ratio: A positive float from 0 to 1.
        :param include_valid: Whether to split validation set, the remainder from train_ratio and test_ratio is used.
        :return: Splitted datasets.
        """
        train_size = int(len(self.dataset) * train_ratio)
        test_size = len(self.dataset) - train_size
        splitted_size = [train_size, test_size]

        trainset, testset= random_split(self.dataset, splitted_size)
        return DataLoader(trainset, batch_size, pin_memory=self.pin_memory, pin_memory_device=self.pin_memory_device), DataLoader(testset, batch_size, pin_memory=self.pin_memory, pin_memory_device=self.pin_memory_device)
