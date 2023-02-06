"""
This module contain the parent class for all the ingestion classes.
"""
import sqlite3
from typing import Optional

import pandas as pd
from utility import concat_list, create_parent_directory

from . import enum


class DataGrepper:
    """
    Parent class for StockGrepper, ForexGrepper, OptionsGrepper, etc.
    """

    def __init__(self) -> None:
        self.query_methods = {}

    def store_data(
        self,
        data: pd.DataFrame,
        file_path: str,
        table_name: Optional[str]=None,
        file_type: enum.Storage=enum.Storage.SQLITE
        ):
        """
        Save downloaded data.

        If file_type is Storage.SQLITE or Storage.EXCEL and the file path already exists, this function will add a table to the existing file path, rather than replacing it.
        :param data: The data to be written.
        :param file_path: The file path of the write file.
        :param file_type: The file types to store the data to.
        """
        if file_type == enum.Storage.SQLITE:
            create_parent_directory(file_path)
            conn = sqlite3.connect(file_path)
            data.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
        elif file_type == enum.Storage.CSV:
            create_parent_directory(file_path)
            data.to_csv(file_path, index=False)
        elif file_type == enum.Storage.PICKLE:
            create_parent_directory(file_path)
            data.to_pickle(file_path)
        else:
            raise TypeError("Acceptable storage types are: " + concat_list(enum.Storage.__members__.keys()))
        