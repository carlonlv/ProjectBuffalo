import json
from datetime import datetime
from typing import Optional, Union

import polygon.rest

from ..DataGrepper import *

QUERIES = {
    MetricTimeSeries.SMA: "get_sma",
    MetricTimeSeries.EMA: "get_ema",
    MetricTimeSeries.RSI: "get_rsi"
}


class PolygonStockGrepper(DataGrepper):
    """
    The PolygonStockGrepper is subclass of StockGrepper, it uses polygon api to grep stock data.
    """

    def __init__(self, **init_args) -> None:
        """
        Initializer for PolygonStockGrepper object.

        :param **init_args: Additional arguments passed into api polygon.rest.RESTClient. Arguments are presented below.
        :param connect_timeout: The connection timeout in seconds. Defaults to 10. basically the number of seconds to
                                wait for a connection to be established. Raises a ``ConnectTimeout`` if unable to
                                connect within specified time limit.
        :param read_timeout: The read timeout in seconds. Defaults to 10. basically the number of seconds to wait for
                            date to be received. Raises a ``ReadTimeout`` if unable to connect within the specified
                            time limit.
        :param num_pools: Max number of connections in the pool for HTTPs requests. Defaults to 10.
        :param retries: Number of retries allowed for connections to fail. Defaults to 3.
        :param verbose: Whether to print debug messages. Default to False,
        """
        self.client = do_call(polygon.rest.RESTClient, api_key=Configuration.api_key, **init_args)
        super()._populate_query_methods(QUERIES)


    def quotes_download(self, stores: bool=True, returns: bool=True, file_name: str='stock_quote_ingestion_{datetime}.{file_type}', table_name: Optional[str]=None, file_type: Storage=Storage.SQLITE, **download_args) -> Union[pd.DataFrame, None]:
        """ Download Quotes using Polygon API
        
        :param stores: Whether to store the data instead of simply returning it.
        :param returns: Whether to return the data.
        :param **download_args: Additional arguments passed into api list_quotes. Arguments are presented below.
        :param ticker: The ticker symbol to get quotes for.
        :param timestamp: Query by timestamp. Either a date with the format YYYY-MM-DD or a nanosecond timestamp.
        :param timestamp_lt: Timestamp less than
        :param timestamp_lte: Timestamp less than or equal to
        :param timestamp_gt: Timestamp greater than
        :param timestamp_gte: Timestamp greater than or equal to
        :param limit: Limit the number of results returned per-page, default is 10 and max is 50000.
        :param sort: Sort field used for ordering.
        :param order: Order results based on the sort field.
        :param params: Any additional query params.
        :param raw: Return HTTPResponse object instead of results object.
        :return: None if returns is set to False, otherwise, return the data.
        """
        download_args["raw"] = False
        data = pd.concat([pd.json_normalize(json.loads(x)) for x in do_call(self.client.list_quotes, **download_args)])
        if stores:
            file_path = os.path.join(Configuration.storage_folder, file_name.format(datetime=datetime.now().strftime(r'%y%m%d%H%M%S'), file_type=file_type))
            super()._store_data(data, file_path, table_name, file_type)

        if returns:
            return data
        else:
            return None


    def trades_download(self, stores: bool=True, returns: bool=True, file_name: str='stock_quote_ingestion_{datetime}.{file_type}', table_name: Optional[str]=None, file_type: Storage=Storage.SQLITE, **download_args) -> Union[pd.DataFrame, None]:
        """ Download Quotes using Polygon API
        
        :param stores: Whether to store the data instead of simply returning it.
        :param returns: Whether to return the data.
        :param **download_args: Additional arguments passed into api list_quotes. Arguments are presented in below.
        :param ticker: The ticker symbol to get quotes for.
        :param timestamp: Query by timestamp. Either a date with the format YYYY-MM-DD or a nanosecond timestamp.
        :param timestamp_lt: Timestamp less than
        :param timestamp_lte: Timestamp less than or equal to
        :param timestamp_gt: Timestamp greater than
        :param timestamp_gte: Timestamp greater than or equal to
        :param limit: Limit the number of results returned per-page, default is 10 and max is 50000.
        :param sort: Sort field used for ordering.
        :param order: Order results based on the sort field.
        :param params: Any additional query params.
        :param raw: Return HTTPResponse object instead of results object.
        :return: None if returns is set to False, otherwise, return the data.
        """
        download_args["raw"] = False
        data = pd.concat([pd.json_normalize(json.loads(x)) for x in do_call(self.client.list_trades, **download_args)])
                
        if stores:
            file_path = os.path.join(Configuration.storage_folder, file_name.format(datetime=datetime.now().strftime(r'%y%m%d%H%M%S'), file_type=file_type))
            super()._store_data(data, file_path, table_name, file_type)

        if returns:
            return data
        else:
            return None

    def snapshot_download(self, stores: bool=True, returns: bool=True, file_name: str='stock_quote_ingestion_{datetime}.{file_type}', table_name: Optional[str]=None, file_type: Storage=Storage.SQLITE, **download_args) -> Union[pd.DataFrame, None]:
        """ Download Snapshot of a ticker of using Polygon API
        
        :param stores: Whether to store the data instead of simply returning it.
        :param returns: Whether to return the data.
        :param **download_args: Additional arguments passed into api list_quotes. Arguments are presented in below.
        :param tickers: A comma separated list of tickers to get snapshots for.
        :param include_otc: Include OTC securities in the response. Default is false (don't include OTC securities).
        :return: List of Snapshots
        """
        download_args["raw"] = False
        download_args["market_type"] = polygon.rest.snapshot.SnapshotMarketType.STOCKS.value
        data = pd.concat([pd.Series(unfold_object(x)) for x in do_call(self.client.get_snapshot_all, **download_args)])
        if stores:
            file_path = os.path.join(Configuration.storage_folder, file_name.format(datetime=datetime.now().strftime(r'%y%m%d%H%M%S'), file_type=file_type))
            super()._store_data(data, file_path, table_name, file_type)

        if returns:
            return data
        else:
            return None
