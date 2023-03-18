"""
This module stores all the implementation related to data ingestion streaming(Sync/Async)/RESTful.
"""

import sqlite3
import time
import warnings
from typing import List, Optional, Union, overload

import pandas as pd

from ..utility import (PositiveFlt, PositiveInt, concat_list,
                       create_and_dodge_new_name, create_parent_directory)
from . import advantage, enum
from .configuration import Configuration


class DataIngestion:
    """
    This is the endpoint for accessing data ingestion class.
    """

    def __init__(self, end_points: Optional[Union[enum.API, List[enum.API]]]) -> None:
        """
        Initializer for DataIngestion class

        :param end_points: The endpoints to be used, functions a
        """
        self.data = {}
        if end_points is None:
            end_points = Configuration.api_keys.keys()
        self.end_points = {}
        if isinstance(end_points, enum.API):
            end_points = [end_points]
        for end_point in end_points:
            if end_point == enum.API.ADVANTAGE:
                self.end_points[end_point] = advantage.AdvantageStockGrepper()
            else:
                raise NotImplementedError(f'{end_point} not implemented.')

    def ingestion_from_web(
            self,
            data_type: enum.DataType,
            ingestion_type: enum.IngestionType,
            end_points: Optional[Union[enum.API, List[enum.API]]]=None,
            data_name: Optional[str]=None,
            **ingestion_args) -> None:
        """
        Download data from web. Cache the loaded data into memory.

        :param data_type: The data type to be ingested.
        :param ingestion_type: The type of ingestion.
        :param end_points: The optional endpoints to be used. Enpoints not initialized are not used.
        :param ingestion_args: The additional arguments passed into ingestion method.
        """
        if end_points is None:
            end_points = self.end_points.keys()
        else:
            remove_end_points = end_points[end_points not in self.end_points]
            if len(remove_end_points) > 0:
                warnings.warn(f'{concat_list(remove_end_points)} excluded since they are not included in initialization.')
            end_points = end_points[end_points not in remove_end_points]

        target_end_points = {k: v for k, v in self.end_points.items() if (data_type, ingestion_type) in v.ingestion_methods}

        if len(target_end_points) == 0:
            raise ValueError(f'{data_type}, {ingestion_type} not supported by endpoints included by any of the endpoints.')

        target_end_point = list(target_end_points.keys())[0]
        if len(target_end_points) > 0:
            warnings.warn(f'Multiple endpoints support detected, ({concat_list(target_end_points.keys())}), {target_end_point} used.')
        target_end_point = target_end_points[target_end_point]

        if data_name is None:
            data_name = create_and_dodge_new_name(self.data.keys(), 'NewFile', '')

        self.data[data_name] = target_end_point.ingestion_methods[(data_type, ingestion_type)](**ingestion_args)

    @overload
    def ingestion_from_web(self, args_df: pd.DataFrame, wait_time: PositiveFlt=60, max_retries: PositiveInt=5):
        """
        Download data from web. Cache the loaded data into memory.

        :param args_df: The dataframe containing ingestion arguments, each row represents one ingestion call.
        :param wait_time: Wait time to retry ingestion from web. Only used when requests are too frequent. Unit in seconds.
        :param max_retries: The number of retries allowed for each argument before expection is thrown.
        """
        for i in args_df.index:
            retries = 0
            try:
                self.ingestion_from_web(**args_df.loc[i].to_dict())
            except ConnectionRefusedError:
                if retries > max_retries:
                    raise
                retries += 1
                time.sleep(wait_time)

    def load_data(
        self,
        file_path: str,
        table_names: Optional[List[str]]=None,
        query_template: str='SELECT * FROM {table}'):
        """
        Load data from file.

        :param file_path: The file path of the write file.
        :param table_names: The names of tables to be read to file.
        :param query_template: The template to be used for reading. Format {table} is used for table name.
        """
        conn = sqlite3.connect(file_path)
        if table_names is None:
            read_data = pd.read_sql_query('SELECT name FROM sqlite_master WHERE type="table"', conn)['name']
        else:
            read_data = pd.Series(table_names)
        for tbl in read_data:
            self.data[tbl] = pd.read_sql(query_template.format(table=tbl), conn)
        conn.close()

    def store_data(
        self,
        file_path: str,
        data_names: Optional[List[str]]=None):
        """
        Write data to file. This function will replace tables if they already exists.

        :param file_path: The file path of the write file.
        :param data_names: The names of data to be written to file.
        """
        if data_names is None:
            write_data = self.data
        else:
            write_data = {k: self.data[k] for k in data_names if k in self.data}
        create_parent_directory(file_path)
        conn = sqlite3.connect(file_path)
        for key, val in write_data.items():
            val.to_sql(key, conn, if_exists='replace', index=False)
        conn.close()
