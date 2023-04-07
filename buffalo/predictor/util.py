"""
This module contains helper functions to manipulate predictors.
"""

import os
import re
import sqlite3
from typing import Any, Optional
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utility import NonnegativeInt, PositiveInt, concat_list
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


class TimeSeriesData(Dataset):
    """
    Time series Data that is loaded into memory. All operations involving time series data preserves ordering.
    """
    def __init__(self, endog: pd.DataFrame, exog: pd.DataFrame, seq_len: Optional[PositiveInt], name: Optional[str]=None):
        """
        Initializer for Time Series Data. The row of data is the time dimension. Assuming time in ascending order(past -> future).

        Intialize time series data.

        :param endog: The endogenous variable. The row of data is the time dimension.
        :param exog: The exogenous variable The row of data is the time dimension. The exogenous variable must be enforced such that information is available before the timestamps for endog variables. Exogenous time series with the same timestamps are not assumed to be available for prediction, so only past timestamps are used.
        :param seq_len: The length of sequence, the last row contains label. If not provided, all the past information starting from the beginning is used.
        :param name: The convenient name for the dataset.
        """
        assert endog.shape[0] == exog.shape[0]
        self.endog = endog.sort_index(ascending=True)
        self.exog = exog.sort_index(ascending=True)
        self.seq_len = seq_len
        self.target_cols = torch.arange(self.endog.shape[1])
        self.name = name

        endog_array = torch.Tensor(self.endog.to_numpy())
        exog_array = torch.Tensor(self.exog.to_numpy())
        self.dataset = torch.cat((endog_array.unsqueeze(1) if isinstance(self.endog, pd.Series) else endog_array, exog_array), dim=1)
        self.info = {'endog': concat_list(self.endog.columns), 'exog': concat_list(self.exog.columns), 'seq_len': seq_len, 'name': name, 'n_obs': self.dataset.shape[0], 'start_time': self.endog.index[0], 'end_time': self.endog.index[-1], 'create_time': pd.Timestamp.now()}

    def __len__(self):
        if self.seq_len is None:
            return len(self.dataset) - 1
        else:
            return len(self.dataset) - self.seq_len

    def __getitem__(self, index):
        ## index goes from 0 to self.__len__() - 1
        if self.seq_len is not None:
            start_index = index
            end_index = index + self.seq_len
            return self.dataset[start_index:end_index,:], self.dataset[end_index,self.target_cols], end_index
        else:
            return self.dataset[:(index+1),:], self.dataset[index+1,self.target_cols], index + 1


class ModelPerformance:
    """
    This class stores the model performance during training, validation and testing.
    """

    def __init__(self, model: torch.nn.Module, training_record: pd.DataFrame, training_residuals: pd.Series, testing_residuals: pd.Series, dataset: TimeSeriesData, train_indice: Dataset, test_indice: Dataset, test_loss: float, loss_func: str, optimizer: str) -> None:
        """
        Initializer for ModelPerformance object.

        :param training_record: The training record generated by train_model function, each training record stores the averaged loss on batches .
        :param training_residuals: The training residuals.
        :param testing_residuals: The testing residuals.
        :param dataset: The dataset used for training and testing.
        :param train_indice: The start and stop indices of training data.
        :param test_indice: The start and stop indices of testing data.
        :param test_loss: The loss on the testing data.
        :param loss_func: The string representation of loss function used for training.
        :param optimizer: The string representation of optimizer used for training.
        """
        self.model = model
        self.training_record = training_record
        self.training_residuals = training_residuals
        self.testing_residuals = testing_residuals
        self.dataset = dataset
        self.train_indice = train_indice
        self.test_indice = test_indice
        self.test_loss = test_loss
        self.loss_func = loss_func
        self.optimizer = optimizer

    def serialize_to_file(self, sql_path: str, additional_note_dataset: str='', additonal_note_model: str='') -> None:
        """ Write the performance of the model to a csv file and the trained model to file.

        Primary keys for dataset is dataset_id, primary keys for model is model_id, dataset_id, train_start, train_end.
        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param additional_note_dataset: Additional notes to be added to describe the dataset.
        :param additonal_note_model: Additional notes to be added to describe the model.
        """
        newconn = sqlite3.connect(sql_path)

        ## Store Dataset Information
        dataset_info = self.dataset.info
        dataset_info['additonal_notes'] = additional_note_dataset
        if 'dataset_info' in newconn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
            all_datasets = pd.read_sql_table('dataset', newconn)
            next_id = all_datasets['dataset_id'].max() + 1
        else:
            all_datasets = pd.DataFrame()
            next_id = 0

        all_existing_datasets = [x for x in newconn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall() if re.match(r'^dataset_\d+$', x)]
        current_dataset_exists = False
        if len(all_existing_datasets) > 0:
            for dat in all_existing_datasets:
                existing_dataset = pd.read_sql_table(dat, newconn)
                current_dataset = pd.concat([self.endog, self.exog], axis=1)
                if existing_dataset.equals(current_dataset):
                    current_dataset_exists = True
                    break

        if not current_dataset_exists:
            dataset_info['dataset_id'] = next_id
            all_datasets = pd.concat([all_datasets, pd.DataFrame(dataset_info, index=[0])], axis=0, ignore_index=True)
            all_datasets.to_sql('dataset', newconn, if_exists='append')
            current_dataset.to_sql(f'dataset_{next_id}', newconn)

        ## Store Model Information
        model_info = self.model.info
        model_info['additonal_notes'] = additonal_note_model
        model_info['dataset_id'] = dataset_info['id']
        model_info['train_start'] = self.train_indice[0]
        model_info['train_end'] = self.train_indice[1]
        model_info['train_loss_func'] = self.loss_func
        model_info['train_optimizer'] = self.optimizer

        if 'model_info' in newconn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
            all_models = pd.read_sql_table('model', newconn)
            next_id = all_models['model_id'].max() + 1
        else:
            all_models = pd.DataFrame()
            next_id = 0

        model_info['model_id'] = next_id
        all_models = pd.concat([all_models, pd.DataFrame(model_info, index=[0])], axis=0, ignore_index=True)
        all_models.to_sql('model', newconn, if_exists='append')

        torch.save(self.model, sql_path.replace(os.path.basename(sql_path), f'model_{next_id}.pt'))

        ## Store training and testing residuals
        training_residuals = self.training_residuals.reset_index().assign(type='train').copy()
        training_residuals['model_id'] = model_info['model_id']
        training_residuals['dataset_id'] = dataset_info['dataset_id']
        training_residuals['start_idx'] = self.train_indice[0]
        training_residuals['end_idx'] = self.train_indice[1]
        testing_residuals = self.training_residuals.reset_index().assign(type='test').copy()
        testing_residuals['model_id'] = model_info['model_id']
        testing_residuals['dataset_id'] = dataset_info['dataset_id']
        testing_residuals['start_idx'] = self.test_indice[0]
        testing_residuals['end_idx'] = self.test_indice[1]
        residuals = pd.concat((training_residuals, testing_residuals), axis=0, ignore_index=True)
        residuals.to_sql('residuals', newconn, if_exists='append')

        ## Store training record
        training_record = self.training_record.copy()
        training_record['model_id'] = model_info['model_id']
        training_record['dataset_id'] = dataset_info['dataset_id']
        training_record['train_start'] = self.train_indice[0]
        training_record['train_end'] = self.train_indice[1]
        training_record.to_sql('training_record', newconn, if_exists='append')

        ## Store testing record
        testing_record = pd.DataFrame({'model_id': model_info['model_id'], 'dataset_id': dataset_info['dataset_id'], 'test_start': self.test_indice[0], 'test_end': self.test_indice[1], 'test_loss': self.test_loss, 'test_loss_func': self.loss_func}, index=[0])
        testing_record.to_sql('testing_record', newconn, if_exists='append')
        newconn.close()

    def deserialize_from_file(self, sql_path: str, dataset_id: NonnegativeInt, model_id: NonnegativeInt) -> None:
        """ Read the performance of the model from a csv file and the trained model from file.

        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param dataset_id: The id of the dataset.
        :param model_id: The id of the model.
        """
        newconn = sqlite3.connect(sql_path)
        ## Load Dataset
        dataset_info = pd.read_sql_table('dataset_info', newconn).query('dataset_id == @dataset_id').T
        data_info = pd.read_sql_table(f'dataset_{dataset_id}', newconn)
        endog_cols = dataset_info['endog'].split(',')
        exog_cols = dataset_info['exog'].split(',')
        self.dataset = TimeSeriesData(data_info[endog_cols], data_info[exog_cols], data_info['seq_len'] if pd.isna(data_info['seq_len']) else None, data_info['name'])

        ## Load Model
        self.model = torch.load(sql_path.replace(os.path.basename(sql_path), f'model_{model_id}.pt'))

        model_info = pd.read_sql_table('model_info', newconn).query('model_id == @model_id').T
        self.train_indice = (model_info['train_start'], model_info['train_end'])
        self.loss_func = model_info['train_loss_func']
        self.optimizer = model_info['train_optimizer']

        ## Load Training Record
        self.training_record = pd.read_sql_table('training_record', newconn).query('model_id == @model_id and dataset_id=@dataset_id')

        ## Load Testing Record
        test_info = pd.read_sql_table('testing_record', newconn).query('model_id == @model_id and dataset_id=@dataset_id').T
        self.test_indice = (test_info['test_start'], test_info['test_end'])
        self.test_loss = test_info['test_loss']

        ## Load Residuals
        self.training_residuals = pd.read_sql_table('residuals', newconn, where=f'dataset_id=@dataset_id and model_id=@model_id and type="train" and start_idx={self.train_indice[0]} and end_idx={self.train_indice[1]}')

        self.testing_residuals = pd.read_sql_table('residuals', newconn, where=f'dataset_id=@dataset_id and model_id=@model_id and type="test" and start_idx={self.test_indice[0]} and end_idx={self.test_indice[1]}')
        newconn.close()
