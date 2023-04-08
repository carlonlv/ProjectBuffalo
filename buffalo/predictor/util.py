"""
This module contains helper functions to manipulate predictors.
"""

import os
import sqlite3
from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utility import (NonnegativeInt, PositiveFlt, PositiveInt, concat_list,
                       create_parent_directory)
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

    Database schema:
        dataset_info: The information of the dataset stored. Primary Key: dataset_id
        model_info: The information of the model stored. Primary Key: model_id
        training_info: The information of the training stored, generated by each instance of model trained on each dataset of specific slicing. Primary Key: training_id
        training_record: The training record generated by train_model function, each training record stores the averaged loss on batches. Primary Key: training_id, fold, epoch
        testing_info: The information of the testing stored, generated by each instance of model tested on each dataset of specify slicing. Primary Key: testing_id
        residuals: The residuals generated by each instance of model trained or tested on each dataset of specify slicing. Primary Key: type, train_or_test_id, index
    """

    def __init__(self, model: torch.nn.Module, dataset: TimeSeriesData, training_record: pd.DataFrame, training_residuals: pd.Series, testing_residuals: pd.Series, training_info: pd.Series, testing_info: pd.Series) -> None:
        """
        Initializer for ModelPerformance object.

        :param model: The model used for training and testing.
        :param dataset: The dataset used for training and testing.
        :param training_record: The training record generated by train_model function, each training record stores the averaged loss on batches .
        :param training_residuals: The training residuals.
        :param testing_residuals: The testing residuals.
        :param training_info: The information of the training stored, generated by each instance of model trained on each dataset of specific slicing.
        :param testing_info: The information of the testing stored, generated by each instance of model tested on each dataset of specify slicing.
        """
        self.model = model
        self.training_record = training_record
        self.training_residuals = training_residuals
        self.testing_residuals = testing_residuals
        self.dataset = dataset
        self.training_info = training_info
        self.testing_info = testing_info

    def serialize_to_file(self, sql_path: str, additional_note_dataset: str='', additonal_note_model: str='') -> None:
        """ Write the performance of the model to a csv file and the trained model to file.

        Primary keys for dataset is dataset_id, primary keys for model is model_id, dataset_id, train_start, train_end.
        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param additional_note_dataset: Additional notes to be added to describe the dataset.
        :param additonal_note_model: Additional notes to be added to describe the model.
        """
        def search_id_given_pk(conn, table_name, pks, id_col):
            if table_name not in [x[0] for x in newconn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]:
                return 0
            query = f"SELECT MAX({id_col}) FROM {table_name} WHERE"
            for pk_name, pk_value in pks.items():
                if isinstance(pk_value, str) or isinstance(pk_value, pd.Timestamp):
                    query += f" {pk_name} = '{pk_value}' AND"
                else:
                    query += f" {pk_name} = {pk_value} AND"
            if len(pks) > 0:
                query = query[:-4]
            else:
                query = query[:-6]
            result = conn.execute(query).fetchall()
            if result[0][0] is None:
                return 0
            else:
                return result[0][0]

        create_parent_directory(sql_path)
        newconn = sqlite3.connect(sql_path)

        ## Store Dataset Information
        table_name = 'dataset_info'
        id_col = 'dataset_id'
        self.dataset.info['additional_notes'] = additional_note_dataset
        searched_id = search_id_given_pk(newconn, table_name, pd.Series(self.dataset.info).drop('create_time'), id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.dataset.info[id_col] = searched_id
            pd.DataFrame(self.dataset.info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
            pd.concat((self.dataset.endog, self.dataset.exog), axis=1).to_sql(f'dataset_{searched_id}', newconn, index=True, index_label='time')
        else:
            warn(f"dataset_info with the same primary keys already exists with id {searched_id}, will not store dataset information.")

        ## Store Model Information
        table_name = 'model_info'
        id_col = 'model_id'
        self.model.info['additional_notes'] = additonal_note_model
        searched_id = search_id_given_pk(newconn, table_name, self.model.info, id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.model.info[id_col] = searched_id
            pd.DataFrame(self.model.info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
        else:
            warn(f'model_info with the same primary keys already exists with id {searched_id}, will not store model information.')

        table_name = 'training_info'
        id_col = 'training_id'
        self.training_info['dataset_id'] = self.dataset.info['dataset_id']
        self.training_info['model_id'] = self.model.info['model_id']
        searched_id = search_id_given_pk(newconn, table_name, pd.Series(self.training_info).drop(['train_start_time', 'train_stop_time', 'train_elapsed_time']).to_dict(), id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.training_info[id_col] = searched_id
            pd.DataFrame(self.training_info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
            torch.save(self.model, f'{os.path.dirname(sql_path)}/model_{searched_id}.pt')
            self.training_record.assign(training_id = searched_id).to_sql('training_record', newconn, if_exists='append', index=False)
            self.training_residuals.reset_index().assign(train_or_test_id = searched_id, type = 'training').to_sql('residuals', newconn, if_exists='append', index=False)
        else:
            warn(f'training_info ({searched_id}) with the same primary keys already exists, will not store model information.')

        ## Store Testing Information
        table_name = 'testing_info'
        id_col = 'testing_id'
        self.testing_info['training_id'] = self.training_info['training_id']
        searched_id = search_id_given_pk(newconn, table_name, pd.Series(self.testing_info).drop(['test_start_time', 'test_stop_time', 'test_elapsed_time']).to_dict(), id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.testing_info[id_col] = searched_id
            pd.DataFrame(self.testing_info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
            self.testing_residuals.reset_index().assign(train_or_test_id = searched_id, type = 'testing').to_sql('residuals', newconn, if_exists='append', index=False)
        else:
            warn(f'testing_info ({searched_id}) with the same primary keys already exists, will not store model information.')
        newconn.close()

    def deserialize_from_file(self, sql_path: str, testing_id: NonnegativeInt) -> None:
        """ Read the performance of the model from a csv file and the trained model from file.

        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param testing_id: The id of the testing. Other information will be inferred and loaded.
        """
        newconn = sqlite3.connect(sql_path)

        ## Load Testing Information
        self.testing_info = pd.read_sql_table('testing_info', newconn).query(f'testing_id={testing_id}').T
        self.testing_residuals = pd.read_sql_table('residuals', newconn).query(f'train_or_test_id={testing_id} and type="testing"')

        ## Load Training Information
        self.training_info = pd.read_sql_table('training_info', newconn).query(f'training_id={self.testing_info["training_id"]}').T
        self.model = torch.load(f'{os.path.dirname(sql_path)}/model_{self.training_info["training_id"]}.pt')
        self.training_record = pd.read_sql_table('training_record', newconn).query(f'training_id={self.training_info["training_id"]}')
        self.training_residuals = pd.read_sql_table('residuals', newconn).query(f'train_or_test_id={self.training_info["training_id"]} and type="training"')

        ## Load Model Information
        self.model.info = pd.read_sql_table('model_info', newconn).query(f'model_id={self.training_info["model_id"]}').T

        ## Load Dataset Information
        dataset_info = pd.read_sql_table('dataset_info', newconn).query(f'dataset_id={self.training_info["dataset_id"]}').T
        data_info = pd.read_sql_table(f'dataset_{self.training_info["dataset_id"]}', newconn, index_col='time', parse_dates=['time'])
        endog_cols = dataset_info['endog'].split(',')
        exog_cols = dataset_info['exog'].split(',')
        self.dataset = TimeSeriesData(data_info[endog_cols], data_info[exog_cols], data_info['seq_len'] if pd.isna(data_info['seq_len']) else None, data_info['name'])
        self.dataset.info = dataset_info
        newconn.close()
