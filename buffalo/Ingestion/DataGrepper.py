import sqlite3
from typing import Dict, Optional

import pandas as pd

from ..Metrics import MetricTimeSeries
from ..Utility import *
from .Configuration import *
from .DataGrepper import *
from .Enum import *


class DataGrepper:
    """
    Parent class for StockGrepper, ForexGrepper, OptionsGrepper, etc.
    """

    def __init__(self) -> None:
        pass

    def _populate_query_methods(self, queries: Dict[MetricTimeSeries, str]={}):
        """
        Helper function for storing methods for Grepper objects to retrieve time series metrics from api. 

        :param queries: A dictionary with keys indicating the recongized time series metrics types, and the value is the method name to retrieve the data.
        """
        pd.DataFrame()
        self.query_methods = {}
        for query in queries:
            self.query_methods[query] = eval("self.client.{func}".format(func=queries[query]))

    def _store_data(
        self, 
        data: pd.DataFrame,
        file_path: str,
        table_name: Optional[str]=None,
        file_type: Storage=Storage.SQLITE
        ):
        """
        Helper function for saving downloaded data.

        If file_type is Storage.SQLITE or Storage.EXCEL and the file path already exists, this function will add a table to the existing file path, rather than replacing it. 
        :param data: The data to be written.
        :param file_path: The file path of the write file.
        :param file_type: The file types to store the data to. 
        """
        if file_type == Storage.SQLITE:
            create_parent_directory(file_path)
            conn = sqlite3.connect(file_path)
            data.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
        elif file_type == Storage.CSV:
            create_parent_directory(file_path)
            data.to_csv(file_path, index=False)
        elif file_type == Storage.EXCEL:
            create_parent_directory(file_path)
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
                data.to_excel(writer, table_name, index=False)
        elif file_type == Storage.PICKLE:
            create_parent_directory(file_path)
            data.to_pickle(file_path)
        else:
            raise TypeError("Acceptable storage types are: " + print_list(Storage._member_names_))