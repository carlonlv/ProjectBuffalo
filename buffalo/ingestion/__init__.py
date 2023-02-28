"""
This module stores all the implementation related to data ingestion streaming(Sync/Async)/RESTful.
"""

import sqlite3
from typing import List, Optional, Union

import pandas as pd
from ..utility import concat_list, create_parent_directory

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
        self.data = pd.DataFrame()
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

    def ingestion_from_file(self) -> None:
        """
        Load data from file. Cache the loaded data into memory.
        """
        raise NotImplementedError()

    def ingestion_from_web(self, tickers:Optional[Union[str, List[str]]], from_time: pd.Timestamp, to_time: pd.Timestamp, interval: pd.Timedelta, adjusted: bool) -> None:
        """
        Downlaod data from web. Cache the loaded data into memory.
        """
        raise NotImplementedError()

    def stream_from_web(self) -> pd.DataFrame:
        """
        Stream data from web, load to memory and write to file, and returns data stream.
        """
        raise NotImplementedError()

    def stream_from_file(self) -> pd.DataFrame:
        """
        Stream data from file, load to memory.
        """
        raise NotImplementedError()

    def store_data(
        self,
        file_path: str,
        table_name: Optional[str]=None,
        file_type: enum.Storage=enum.Storage.SQLITE
        ):
        """
        Write data to file.

        If file_type is Storage.SQLITE or Storage.EXCEL and the file path already exists, this function will add a table to the existing file path, rather than replacing it.
        :param data: The data to be written.
        :param file_path: The file path of the write file.
        :param file_type: The file types to store the data to, one of types from ingestion.enum.Storage.
        """
        if file_type == enum.Storage.SQLITE:
            create_parent_directory(file_path)
            conn = sqlite3.connect(file_path)
            self.data.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
        elif file_type == enum.Storage.CSV:
            create_parent_directory(file_path)
            self.data.to_csv(file_path, index=False)
        elif file_type == enum.Storage.PICKLE:
            create_parent_directory(file_path)
            self.data.to_pickle(file_path)
        else:
            raise TypeError("Acceptable storage types are: " + concat_list(enum.Storage.__members__.keys()))
