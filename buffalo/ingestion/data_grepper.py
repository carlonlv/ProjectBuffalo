"""
This module contain the parent class for all the ingestion classes. Avoid importing submodules in this file to prevent circular imports.
"""
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import pandas as pd
from utility import concat_list, create_parent_directory

from . import enum


class DataGrepper(ABC):
    """
    Parent class for StockGrepper, ForexGrepper, OptionsGrepper, etc.
    """

    def __init__(self) -> None:
        self.query_methods = {}
        self.data = pd.DataFrame()

    @abstractmethod
    def ingestion_from_file(self):
        """
        Load data from file.
        """

    @abstractmethod
    def ingestion_from_web(self, tickers:Optional[Union[str, List[str]]], from_time: pd.Timestamp, to_time: pd.Timestamp, interval: pd.Timedelta, adjusted: bool):
        """
        Downlaod data from web.
        """

    @abstractmethod
    def stream_from_web(self):
        """
        Stream data from web, load to memory and write to file.
        """

    @abstractmethod
    def stream_from_file(self):
        """
        Stream data from file, load to memory.
        """

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
        