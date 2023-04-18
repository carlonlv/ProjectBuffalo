"""
This module contains helper functions to manipulate predictors.
"""

import os
import sqlite3
from functools import reduce
from typing import List, Optional
from warnings import warn

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from torch.utils.data import Dataset

from ..algorithm import OnlineUpdateRuleFactory
from ..utility import (NonnegativeInt, PositiveInt, concat_list,
                       create_parent_directory, search_id_given_pk)
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
    other_df = other_df.sort_index()
    target_df = target_df.sort_index()
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
    def __init__(self,
                 endog: pd.DataFrame,
                 exog: Optional[pd.DataFrame],
                 seq_len: Optional[PositiveInt],
                 label_len: PositiveInt=1,
                 n_ahead: int=1,
                 name: Optional[str]=None,
                 split_endog: Optional[List[List[NonnegativeInt]]]=None,
                 target_indices: Optional[List[NonnegativeInt]]=None):
        """
        Initializer for Time Series Data. The row of data is the time dimension. Assuming time in ascending order(past -> future).

        Detailed behaviors of seq_len, label_len, n_ahead: If seq_len and label_len are not None, then the sequence of predictors and labels are offsetted by n_ahead. The overlapping indices are determined by seq_len and label_len. If seq_len is not provided, then all observations from index 0 to the selected index is used. If label_len is not provided, labels returned will have the same length as predictors.

        :param endog: The endogenous variable. The row of data is the time dimension.
        :param exog: The exogenous variable The row of data is the time dimension. The exogenous variable must be enforced such that information is available before the timestamps for endog variables. Exogenous time series with the same timestamps are not assumed to be available for prediction, so only past timestamps are used.
        :param seq_len: The length of sequence, the last row contains label. If not provided, all the past information starting from the beginning is used.
        :param label_len: The length of label. The length of the label to be returned.
        :param n_ahead: The number of observations between the end of the predictor sequence and the end of the label sequence. Default is 1, i.e., the end of sequences are offsetted by 1.
        :param name: The convenient name for the dataset.
        :param split_endog: Used to control the splitting of endog dataset. If not provided, the data will not be splitted. Otherwise, endog will be splitted into multiple tensors of equal length during sampling. This is useful when model is Autoencoder/decoder and endog should be splitted into input sequence and target input sequence. Note that the order of the tuples in this list should correspond to the positional arguments of the model.
        :param target_indices: The indices of the target columns used to compute loss. If not provided, all columns of endog will be used.
        """
        assert target_indices is None or all(np.array(target_indices) < endog.shape[1])
        assert split_endog is None or reduce(lambda x, y: set(x).union(set(y)), split_endog) == set(range(endog.shape[1])) if target_indices is None else set(target_indices), 'split_endog does not cover all indices in target_indices'

        self.endog = endog.sort_index(ascending=True)
        self.seq_len = seq_len
        self.label_len = label_len
        self.n_ahead = n_ahead
        self.target_cols = torch.arange(self.endog.shape[1]) if target_indices is None else torch.tensor(target_indices)
        self.name = name
        self.split_endog = None if split_endog is None else [torch.tensor(x) for x in split_endog]

        endog_array = torch.Tensor(self.endog.to_numpy())
        if exog is not None:
            assert endog.shape[0] == exog.shape[0]
            self.exog = exog.sort_index(ascending=True)

            exog_array = torch.Tensor(self.exog.to_numpy())
            self.dataset = torch.cat((endog_array, exog_array), dim=1)
        else:
            self.exog = None
            self.dataset = endog_array
        self.info = {'endog': concat_list(self.endog.columns),
                     'exog': concat_list(self.exog.columns) if self.exog is not None else '',
                     'label_len': label_len,
                     'seq_len': seq_len,
                     'n_ahead': n_ahead,
                     'name': name,
                     'split_endog': str(split_endog) if split_endog is not None else None,
                     'target_indices': str(target_indices) if target_indices is not None else None,
                     'n_obs': self.dataset.shape[0],
                     'start_time': self.endog.index[0],
                     'end_time': self.endog.index[-1],
                     'create_time': pd.Timestamp.now()}

    def __len__(self):
        seq_res = self.seq_len if self.seq_len is not None else 0
        label_res = self.label_len if self.label_len is not None else 0
        return len(self.dataset) - max(seq_res, label_res) - self.n_ahead

    def __getitem__(self, index):
        ## index goes from 0 to self.__len__() - self.seq_len - self.label_len
        if self.seq_len is not None:
            start_index_predictor = index
            end_index_predictor = index + self.seq_len ## Not included
        else:
            start_index_predictor = 0
            end_index_predictor = index

        if self.label_len is not None:
            end_index_label = end_index_predictor + self.n_ahead
            start_index_label = end_index_label - self.label_len
        else:
            end_index_label = end_index_predictor + self.n_ahead
            start_index_label = end_index_label - (end_index_predictor - start_index_predictor) ## By default match the same length as predictor

        if self.split_endog is None:
            return [self.dataset[start_index_predictor:end_index_predictor,:]], self.dataset[start_index_label:end_index_label,self.target_cols], torch.arange(start_index_label, end_index_label)
        else:
            return [self.dataset[start_index_predictor:end_index_predictor, x] for x in self.split_endog], self.dataset[start_index_label:end_index_label,self.target_cols], torch.arange(start_index_label, end_index_label)

    def serialize_to_file(self, sql_path: str, additional_note: str='', conn: Optional[sqlite3.Connection]=None) -> NonnegativeInt:
        """ Write the performance of the model to a csv file and the trained model to file.

        Primary keys for dataset is dataset_id, primary keys for model is model_id, dataset_id, train_start, train_end.
        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param additional_note: Additional notes to be added to describe the dataset.
        :param conn: The connection to the sqlite file. If not provided, a new connection will be created.
        :return: The dataset_id of the dataset.
        """
        create_parent_directory(sql_path)
        if conn is None:
            newconn = sqlite3.connect(sql_path)
        else:
            newconn = conn

        table_name = 'dataset_info'
        id_col = 'dataset_id'
        self.info['additional_notes'] = additional_note
        searched_id = search_id_given_pk(newconn, table_name, pd.Series(self.info).drop('create_time'), id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.info[id_col] = searched_id
            pd.DataFrame(self.info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
            pd.concat((self.endog, self.exog), axis=1).to_sql(f'dataset_{searched_id}', newconn, index=True, index_label='time')
        else:
            warn(f"dataset_info with the same primary keys already exists with id {searched_id}, will not store dataset information.")
            self.info[id_col] = searched_id

        if conn is None:
            newconn.close()
        return searched_id

    @classmethod
    def deserialize_from_file(cls, sql_path: str, dataset_id: NonnegativeInt, conn: Optional[sqlite3.Connection]=None):
        """ Read the performance of the model from a csv file and the trained model from file.

        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param dataset_id: The dataset_id of the dataset.
        :param conn: The connection to the sqlite file. If not provided, a new connection will be created.
        :return: The loaded TimeSeriesData object.
        """
        if conn is None:
            newconn = sqlite3.connect(sql_path)
        else:
            newconn = conn

        dataset_info = pd.read_sql_query(f'SELECT * FROM dataset_info WHERE dataset_id={dataset_id}', newconn).T[0]
        data_info = pd.read_sql_query(f'SELECT * FROM dataset_{dataset_id}', newconn, index_col='time', parse_dates=['time'])
        endog_cols = dataset_info['endog'].split(',')
        exog_cols = dataset_info['exog'].split(',')
        dataset = TimeSeriesData(
            endog=data_info[endog_cols],
            exog=data_info[exog_cols],
            seq_len=dataset_info['seq_len'] if not pd.isna(dataset_info['seq_len']) else None,
            label_len=dataset_info['label_len'] if not pd.isna(dataset_info['label_len']) else None,
            n_ahead=dataset_info['n_ahead'],
            name=dataset_info['name'],
            split_endog=eval(dataset_info['split_endog']) if not pd.isna(dataset_info['split_endog']) else None,
            target_indices=eval(dataset_info['target_indices']) if not pd.isna(dataset_info['target_cols']) else None)
        dataset.info = dataset_info
        if conn is None:
            newconn.close()
        return dataset


class ModelPerformance:
    """
    This class stores the model performance during training, validation and testing.

    Database schema:
        dataset_info: The information of the dataset stored. Primary Key: dataset_id
        model_info: The information of the model stored. Primary Key: model_id
        training_info: The information of the training stored, generated by each instance of model trained on each dataset of specific slicing. Primary Key: training_id
        training_record: The training record generated by train_model function, each training record stores the averaged loss on batches. Primary Key: training_id, fold, epoch
        testing_info: The information of the testing stored, generated by each instance of model tested on each dataset of specify slicing. Primary Key: testing_id

        dataset_[digit]: The actual dataset stored, digit represents dataset_id.
        residuals_[digit]: The residuals generated by each instance of model tested on each dataset of specify slicing. digits represents dataset_id. Primary Key: id, type
    """

    def __init__(self, model: torch.nn.Module, dataset: TimeSeriesData, training_record: pd.DataFrame, training_residuals: pd.DataFrame, testing_residuals: pd.DataFrame, training_info: pd.Series, testing_info: pd.Series) -> None:
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

    def serialize_to_file(self, sql_path: str, additional_note_dataset: str='', additonal_note_model: str='', conn: Optional[sqlite3.Connection]=None) -> NonnegativeInt:
        """ Write the performance of the model to a csv file and the trained model to file.

        Primary keys for dataset is dataset_id, primary keys for model is model_id, dataset_id, train_start, train_end.
        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param additional_note_dataset: Additional notes to be added to describe the dataset.
        :param additonal_note_model: Additional notes to be added to describe the model.
        :param conn: The connection to the sqlite file. If not provided, a new connection will be created.
        :return: The assigned id of the testing instance.
        """
        if conn is None:
            create_parent_directory(sql_path)
            newconn = sqlite3.connect(sql_path)
        else:
            newconn = conn

        ## Store Dataset Information
        searched_id = self.dataset.serialize_to_file(sql_path, additional_note_dataset, newconn)
        self.dataset.info['dataset_id'] = searched_id

        ## Store Model Information
        table_name = 'model_info'
        id_col = 'model_id'
        self.model.info['additional_notes'] = additonal_note_model
        searched_id = search_id_given_pk(newconn, table_name, self.model.info, id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.model.info[id_col] = searched_id
            if searched_id == 1:
                pd.DataFrame(self.model.info, index=[0]).to_sql(table_name, newconn, index=False)
            else:
                pd.concat((pd.read_sql_query(f'SELECT * FROM {table_name}', newconn), pd.DataFrame(self.model.info, index=[0])), axis=0, ignore_index=True).to_sql(table_name, newconn, index=False, if_exists='replace')
        else:
            warn(f'model_info with the same primary keys already exists with id {searched_id}, will not store model information.')
            self.model.info[id_col] = searched_id

        table_name = 'training_info'
        id_col = 'training_id'
        self.training_info['dataset_id'] = self.dataset.info['dataset_id']
        self.training_info['model_id'] = self.model.info['model_id']
        searched_id = search_id_given_pk(newconn, table_name, pd.Series(self.training_info).drop(['average_train_loss', 'last_train_loss', 'average_validation_loss', 'train_start_time', 'train_stop_time', 'train_elapsed_time']).to_dict(), id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.training_info[id_col] = searched_id
            pd.DataFrame(self.training_info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
            torch.save(self.model, f'{os.path.dirname(sql_path)}/model-{id_col}-{searched_id}.pt')
            self.training_record.assign(training_id = searched_id).to_sql('training_record', newconn, if_exists='append', index=False)
            self.training_residuals.to_sql(f'training_residuals-{searched_id}', newconn, if_exists='replace', index=True, index_label='index')
        else:
            warn(f'training_info ({searched_id}) with the same primary keys already exists, will not store model information.')
            self.training_info[id_col] = searched_id

        ## Store Testing Information
        table_name = 'testing_info'
        id_col = 'testing_id'
        self.testing_info['training_id'] = self.training_info['training_id']
        searched_id = search_id_given_pk(newconn, table_name, pd.Series(self.testing_info).drop(['test_loss', 'test_start_time', 'test_stop_time', 'test_elapsed_time']).to_dict(), id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.testing_info[id_col] = searched_id
            pd.DataFrame(self.testing_info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
            self.testing_residuals.to_sql(f'testing_residuals-{searched_id}', newconn, if_exists='replace', index=True, index_label='index')
        else:
            warn(f'testing_info ({searched_id}) with the same primary keys already exists, will not store model information.')
            self.testing_info[id_col] = searched_id

        if conn is None:
            newconn.close()
        return searched_id

    @classmethod
    def deserialize_from_file(cls, sql_path: str, testing_id: NonnegativeInt, conn: Optional[sqlite3.Connection]=None):
        """ Read the performance of the model from a csv file and the trained model from file.

        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param testing_id: The id of the testing. Other information will be inferred and loaded.
        :return: The loaded ModelPerformance object.
        """
        if conn is None:
            create_parent_directory(sql_path)
            newconn = sqlite3.connect(sql_path)
        else:
            newconn = conn

        ## Load Testing Information
        testing_info = pd.read_sql_query(f'SELECT * FROM testing_info WHERE testing_id={testing_id}', newconn).T[0]

        ## Load Training Information
        training_info = pd.read_sql_query(f'SELECT * FROM training_info WHERE training_id={testing_info["training_id"]}', newconn).T[0]
        model = torch.load(f'{os.path.dirname(sql_path)}/model-training_id-{training_info["training_id"]}.pt')
        training_record = pd.read_sql_query(f'SELECT * FROM training_record WHERE training_id={training_info["training_id"]}', newconn).drop(columns=['training_id'])

        ## Load Residuals
        testing_residuals = pd.read_sql_query(f'SELECT * FROM "testing_residuals-{testing_info["testing_id"]}"', newconn, index_col='index')
        training_residuals = pd.read_sql_query(f'SELECT * FROM "training_residuals-{training_info["training_id"]}"', newconn, index_col='index')

        ## Load Model Information
        model.info = pd.read_sql_query(f'SELECT * FROM model_info WHERE model_id={training_info["model_id"]}', newconn).T[0]

        ## Load Dataset Information
        dataset_id = training_info['dataset_id']
        dataset = TimeSeriesData.deserialize_from_file(sql_path, dataset_id, newconn)

        if conn is None:
            newconn.close()

        return cls(model, dataset, training_record, training_residuals, testing_residuals, training_info, testing_info)

    def plot_training_records(self):
        """ Plot the training loss and validation loss over epochs, used to check the convergence speed.
        """
        training_records = self.training_record.copy()
        training_records['fold'] = training_records['fold'].astype(int).astype(str)
        plt.subplot(2, 1, 1) # Create subplot for training loss plot
        plt1 = sns.lineplot(x='epoch', y='training_loss', hue='fold', data=training_records)
        plt1.set_title('Training Loss over Time')
        plt1.set_xlabel('Epoch')
        plt1.set_ylabel('Training Loss')
        plt.subplot(2, 1, 2) # Create subplot for validation loss plot
        plt2 = sns.lineplot(x='epoch', y='validation_loss', hue='fold', data=training_records)
        plt2.set_title('Validation Loss over Time')
        plt2.set_xlabel('Epoch')
        plt2.set_ylabel('Validation Loss')

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def plot_residuals(self, figsize=(12, 8)):
        """ Plot original time series and residual time series, used to check the performance of the model. Vertical line in the residual plot indicates the start of the testing period.

        :param figsize: The size of the figure.
        """

        endog_long = self.dataset.endog.reset_index(names='time').melt(id_vars='time', var_name='series', value_name='price')
        plt.subplots(figsize=figsize)
        plt.subplot(3, 1, 1)
        plt1 = sns.lineplot(data=endog_long, x='time', y='price')
        plt1.set_title('Original Time Series', fontsize=12)
        plt1.set_xlabel('Time', fontsize=10)
        plt1.set_ylabel('Price', fontsize=10)

        plt.subplot(3, 1, 2)
        train_predicted = {}
        test_predicted = {}
        for col in self.training_residuals.columns:
            colname, n_ahead = col.split(':')
            train_predicted[col] = self.dataset.endog.iloc[(self.training_residuals.index + int(n_ahead)-1)][colname].to_numpy() - self.training_residuals[col].to_numpy()
            test_predicted[col] = self.dataset.endog.iloc[(self.testing_residuals.index + int(n_ahead)-1)][colname].to_numpy() - self.testing_residuals[col].to_numpy()
        train_predicted = pd.DataFrame(train_predicted, index = self.training_residuals.index)
        test_predicted = pd.DataFrame(test_predicted, index = self.testing_residuals.index)
        predicted_long = pd.concat((train_predicted, test_predicted), axis=0).reset_index(names='time').melt(id_vars='time', var_name='series', value_name='price')
        predicted_long['time'] = endog_long['time'].iloc[predicted_long['time']].values
        plt2 = sns.lineplot(data=predicted_long, x='time', y='price')
        plt2.axvline(x=endog_long['time'].iloc[self.testing_residuals.index.min()], color='black')
        plt2.set_title('Predicted Time Series', fontsize=12)
        plt2.set_xlabel('Time', fontsize=10)
        plt2.set_ylabel('Price', fontsize=10)

        plt.subplot(3, 1, 3)
        residual_long = pd.concat((self.training_residuals, self.testing_residuals), axis=0).reset_index(names='time').melt(id_vars='time', var_name='series', value_name='price')
        residual_long['time'] = endog_long['time'].iloc[residual_long['time']].values
        plt3 = sns.lineplot(data=residual_long, x='time', y='price')
        plt3.axvline(x=endog_long['time'].iloc[self.testing_residuals.index.min()], color='black')
        plt3.set_title('Residual Time Series', fontsize=12)
        plt3.set_xlabel('Time', fontsize=10)
        plt3.set_ylabel('Price', fontsize=10)

        plt.subplots_adjust(hspace=0.5)
        plt.show()


class ModelPerformanceOnline:
    """
    This class stores the model performance during training and testing in an online fashion.

    Database schema:
        dataset_info: The information of the dataset stored. Primary Key: dataset_id
        model_info: The information of the model stored. Primary Key: model_id
        training_record: The training record generated by train_model function, each training record stores the averaged loss on batches. Primary Key: sim_id, fold, epoch
        info: The information of the training and testing stored, generated by each instance of model tested on each dataset of specify slicing. Primary Key: sim_id

        dataset_[digit]: The actual dataset stored, digit represents dataset_id.
        residuals_[digit]: The residuals generated by each instance of model tested on each dataset of specify slicing. digits represents dataset_id. Primary Key: id, type
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataset: TimeSeriesData,
                 update_rule,
                 info: pd.Series) -> None:
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
        self.dataset = dataset
        self.info = info
        self.update_rule = update_rule

    def serialize_to_file(self, sql_path: str, additional_note_dataset: str='', additonal_note_model: str='', conn: Optional[sqlite3.Connection]=None) -> NonnegativeInt:
        """ Write the performance of the model to a csv file and the trained model to file.

        Primary keys for dataset is dataset_id, primary keys for model is model_id, dataset_id, train_start, train_end.
        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param additional_note_dataset: Additional notes to be added to describe the dataset.
        :param additonal_note_model: Additional notes to be added to describe the model.
        :param conn: The connection to the sqlite file. If None, a new connection will be created.
        :return: The assigned id for the simulation.
        """
        if conn is None:
            create_parent_directory(sql_path)
            newconn = sqlite3.connect(sql_path)
        else:
            newconn = conn

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
            self.dataset.info[id_col] = searched_id

        ## Store Model Information
        table_name = 'model_info'
        id_col = 'model_id'
        self.model.info['additional_notes'] = additonal_note_model
        searched_id = search_id_given_pk(newconn, table_name, self.model.info, id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.model.info[id_col] = searched_id
            if searched_id == 1:
                pd.DataFrame(self.model.info, index=[0]).to_sql(table_name, newconn, index=False)
            else:
                pd.concat((pd.read_sql_query(f'SELECT * FROM {table_name}', newconn), pd.DataFrame(self.model.info, index=[0])), axis=0, ignore_index=True).to_sql(table_name, newconn, index=False, if_exists='replace')
        else:
            warn(f'model_info with the same primary keys already exists with id {searched_id}, will not store model information.')
            self.model.info[id_col] = searched_id

        ## Store Update Information
        table_name = 'updaterule_info'
        id_col = 'updaterule_id'
        searched_id = search_id_given_pk(newconn, table_name, self.update_rule.info, id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.update_rule.info[id_col] = searched_id
            if searched_id == 1:
                pd.DataFrame(self.update_rule.info, index=[0]).to_sql(table_name, newconn, index=False)
            else:
                pd.concat((pd.read_sql_query(f'SELECT * FROM {table_name}', newconn), pd.DataFrame(self.update_rule.info, index=[0])), axis=0, ignore_index=True).to_sql(table_name, newconn, index=False, if_exists='replace')
        else:
            warn(f'model_info with the same primary keys already exists with id {searched_id}, will not store model information.')
            self.update_rule.info[id_col] = searched_id

        table_name = 'online_sim_info'
        id_col = 'sim_id'
        self.info['dataset_id'] = self.dataset.info['dataset_id']
        self.info['model_id'] = self.model.info['model_id']
        self.info['updaterule_id'] = self.update_rule.info['updaterule_id']
        searched_id = search_id_given_pk(newconn, table_name, pd.Series(self.info).drop(['start_time', 'stop_time', 'elapsed_time']).to_dict(), id_col)
        if searched_id == 0:
            searched_id = search_id_given_pk(newconn, table_name, {}, id_col) + 1
            self.info[id_col] = searched_id
            pd.DataFrame(self.info, index=[0]).to_sql(table_name, newconn, if_exists='append', index=False)
            torch.save(self.model, f'{os.path.dirname(sql_path)}/onlinemodel-sim_id-{searched_id}.pt')
            self.update_rule.update_logs.to_sql(f'online_update_logs-sim_id-{searched_id}', newconn, index=False, if_exists='replace')
            self.update_rule.train_logs.to_sql(f'online_train_logs-sim_id-{searched_id}', newconn, index=True, index_label='time', if_exists='replace')
            self.update_rule.test_logs.to_sql(f'online_test_logs-sim_id-{searched_id}', newconn, index=True, index_label='time', if_exists='replace')
            self.update_rule.train_record.to_sql(f'online_train_record-sim_id-{searched_id}', newconn, index=True, index_label='time', if_exists='replace')
            self.update_rule.train_residuals.to_sql(f'online_train_residuals-sim_id-{searched_id}', newconn, index=True, index_label='time', if_exists='replace')
            self.update_rule.test_residuals.to_sql(f'online_test_residuals-sim_id-{searched_id}', newconn, index=True, index_label='time', if_exists='replace')
        else:
            warn(f'info ({searched_id}) with the same primary keys already exists, will not store model information.')
            self.info[id_col] = searched_id

        if not newconn.closed:
            newconn.close()
        return searched_id

    @classmethod
    def deserialize_from_file(cls, sql_path: str, sim_id: NonnegativeInt, conn: Optional[sqlite3.Connection]=None):
        """ Read the performance of the model from a csv file and the trained model from file.

        :param sql_path: The path to the sqlite file. The same folder will be used to store the model as well.
        :param testing_id: The id of the testing. Other information will be inferred and loaded.
        :param conn: The connection to the sqlite file. If None, a new connection will be created.
        :return: The loaded ModelPerformance object.
        """
        if conn is None:
            create_parent_directory(sql_path)
            newconn = sqlite3.connect(sql_path)
        else:
            newconn = conn

        sim_info = pd.read_sql_query(f'SELECT * FROM online_sim_info WHERE sim_id={sim_id}', newconn).T[0]

        update_rule_id = sim_info['updaterule_id']
        dataset_id = sim_info['dataset_id']
        model_id = sim_info['model_id']

        update_logs = pd.read_sql_query(f'SELECT * FROM "online_update_logs-sim_id-{sim_id}"', newconn)
        train_logs = pd.read_sql_query(f'SELECT * FROM "online_train_logs-sim_id-{sim_id}"', newconn, index_col='time', parse_dates=['time'])
        test_logs = pd.read_sql_query(f'SELECT * FROM "online_test_logs-sim_id-{sim_id}"', newconn, index_col='time', parse_dates=['time'])
        train_record = pd.read_sql_query(f'SELECT * FROM "online_train_record-sim_id-{sim_id}"', newconn, index_col='time', parse_dates=['time'])
        train_residuals = pd.read_sql_query(f'SELECT * FROM "online_train_residuals-sim_id-{sim_id}"', newconn, index_col='time', parse_dates=['time'])
        test_residuals = pd.read_sql_query(f'SELECT * FROM "online_test_residuals-sim_id-{sim_id}"', newconn, index_col='time', parse_dates=['time'])

        model = torch.load(f'{os.path.dirname(sql_path)}/model-sim_id-{sim_id}.pt')

        update_info = pd.read_sql_query(f'SELECT * FROM updaterule_info WHERE updaterule_id={update_rule_id}', newconn).T[0]
        update_rule = OnlineUpdateRuleFactory.create_update_rule(update_info['name'], **update_info.to_dict())

        ## Load Model Information
        model.info = pd.read_sql_query(f'SELECT * FROM model_info WHERE model_id={model_id}', newconn).T[0]

        ## Load Dataset Information
        dataset_info = pd.read_sql_query(f'SELECT * FROM dataset_info WHERE dataset_id={dataset_id}', newconn).T[0]
        data_info = pd.read_sql_query(f'SELECT * FROM "dataset_{dataset_id}"', newconn, index_col='time', parse_dates=['time'])
        endog_cols = dataset_info['endog'].split(',')
        exog_cols = dataset_info['exog'].split(',')
        dataset = TimeSeriesData(data_info[endog_cols], data_info[exog_cols], dataset_info['seq_len'] if not pd.isna(dataset_info['seq_len']) else None, dataset_info['name'])
        dataset.info = dataset_info

        if not newconn.closed:
            newconn.close()

        result = cls(model, dataset, update_rule, sim_info)
        result.update_rule.update_logs = update_logs
        result.update_rule.train_logs = train_logs
        result.update_rule.test_logs = test_logs
        result.update_rule.train_record = train_record
        result.update_rule.train_residuals = train_residuals
        result.update_rule.test_residuals = test_residuals

        return result

    def plot_training_records(self):
        """ Plot the training loss and validation loss over epochs, used to check the convergence speed.
        """
        def helper(time_step):
            training_records = self.update_rule.train_record.query(f't_index == {time_step}').copy()
            training_records['fold'] = training_records['fold'].astype(int).astype(str)
            plt1 = sns.lineplot(x='epoch', y='training_loss', hue='fold', data=training_records)
            plt1.set_title('Training Loss over Time')
            plt1.set_xlabel('Epoch')
            plt1.set_ylabel('Training Loss')
            plt.show()

        # create a widget for selecting the time step
        time_step_slider = widgets.SelectionSlider(options=self.update_rule.train_record['t_index'].unique(), description='Time step:')

        # link the widget to the plot function
        widgets.interact(helper, time_step=time_step_slider)

        # display the widget
        display(time_step_slider)

    def plot_logs(self):
        """ Plot the training loss and validation loss over time, used to check the convergence speed.
        """
        train_logs = pd.concat([self.update_rule.train_logs.reset_index(drop=True), self.update_rule.update_logs.reset_index(drop=True)], axis=1).rename(columns={'train_loss': 'loss'})

        test_logs = self.update_rule.test_logs.rename(columns={'test_loss': 'loss'}).reset_index(names='t_index')

        ## Assume non overlapping
        plt.subplot(2, 1, 1)
        plt1 = sns.lineplot(x='t_index', y='loss', data=train_logs)
        plt1.set_title('Train Loss over Time')
        plt1.set_xlabel('Time Index')
        plt1.set_ylabel('Loss')
        plt.subplot(2, 1, 2)
        plt2 = sns.lineplot(x='t_index', y='loss', data=test_logs)
        plt2.set_title('Test Loss over Time')
        plt2.set_xlabel('Time Index')
        plt2.set_ylabel('Loss')
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def plot_residuals(self, fig_size=(12, 8)):
        """ Plot original time series and residual time series, used to check the performance of the model. Vertical line in the residual plot indicates the start of the testing period.

        :param fig_size: tuple, the size of the figure.
        """
        def helper(time_step):
            curr_update_time = time_step
            next_update_time = self.update_rule.update_logs.query(f't_index > {time_step}')['t_index'].min()
            if pd.isna(next_update_time):
                next_update_time = self.update_rule.test_residuals['t_index'].max()
            endog_long = self.dataset.endog.reset_index(names='time').melt(id_vars='time', var_name='series', value_name='price')

            _, axes = plt.subplots(3, 1, figsize=fig_size)
            sns.lineplot(ax = axes[0], data=endog_long, x='time', y='price')
            axes[0].set_title('Original Time Series', fontsize=12)
            axes[0].set_xlabel('Time', fontsize=10)
            axes[0].set_ylabel('Price', fontsize=10)

            update_logs = self.update_rule.update_logs.query(f't_index == {time_step}').iloc[0]
            test_residuals = self.update_rule.test_residuals.query(f't_index > {curr_update_time} and t_index <= {next_update_time}').copy()
            train_residuals = self.update_rule.train_residuals.query(f't_index == {curr_update_time}').copy()
            train_residuals.index = range(int(update_logs['start_index']), int(update_logs['end_index'])+1)
            test_residuals.index = range(int(update_logs['end_index'])+1, int(update_logs['end_index'])+len(test_residuals.index)+1)
            residual_long = pd.concat((train_residuals.reset_index(names='time').assign(type='training'), test_residuals.reset_index(names='time').assign(type='testing')), axis=0).melt(id_vars=['t_index', 'type', 'time'], var_name='series', value_name='price')

            predicted_long = residual_long.copy()
            predicted_long['price'] = endog_long.iloc[residual_long['time']]['price'].values - residual_long['price'].values

            residual_long['time'] = endog_long.iloc[residual_long['time']]['time'].values
            predicted_long['time'] = endog_long.iloc[predicted_long['time']]['time'].values

            sns.lineplot(ax = axes[1], data=predicted_long, x='time', y='price', color='green')
            sns.scatterplot(ax = axes[1], data=predicted_long, x='time', y='price', color='green')
            axes[1].axvline(x=endog_long['time'].iloc[test_residuals['t_index'].min()], color='black', linestyle='--')
            axes[1].set_title('Predicted Time Series', fontsize=12)
            axes[1].set_xlabel('Time', fontsize=10)
            axes[1].set_ylabel('Price', fontsize=10)

            sns.lineplot(ax = axes[2], data=residual_long, x='time', y='price', color='red')
            sns.scatterplot(ax = axes[2], data=residual_long, x='time', y='price', color='red')
            axes[2].axvline(x=endog_long['time'].iloc[test_residuals['t_index'].min()], color='black', linestyle='--')
            axes[2].set_title('Residual Time Series', fontsize=12)
            axes[2].set_xlabel('Time', fontsize=10)
            axes[2].set_ylabel('Price', fontsize=10)
            plt.subplots_adjust(hspace=0.5)
            plt.show()

        # create a widget for selecting the time step
        time_step_slider = widgets.SelectionSlider(options=self.update_rule.train_residuals['t_index'].unique(), description='Time step:')

        # link the widget to the plot function
        widgets.interact(helper, time_step=time_step_slider)

        # display the widget
        display(time_step_slider)
