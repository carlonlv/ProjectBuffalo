from datetime import date

import polygon
from StockGrepper import *

from Metrics.MetricTypes import *
from Utility.helper import *

QUERIES = {
    MetricTimeSeries.SMA: "get_sma",
    MetricTimeSeries.EMA: "get_ema",
    MetricTimeSeries.RSI: "get_rsi"
}

class PolygonStockGrepper(StockGrepper):
    """
    The PolygonStockGrepper is subclass of StockGrepper, it uses polygon api to grep stock data.
    
    Attributes:
        make (str): The make of the car.
        model (str): The model of the car.
        year (int): The year the car was manufactured.
    """

    def __init__(self, **init_args) -> None:
        self.client = doCall(polygon.StocksClient, api_key=Configuration.api_key, **init_args)
        self._populate_query_methods()
    
    def _populate_query_methods(self):
        self.query_methods = {}
        for query in QUERIES:
            self.query_methods[query] = eval("self.client.{func}".format(func=QUERIES[query]))

    def quote_download(self, file_name, **download_args):
        """ Download Quote using Polygon API
        
        :param file_name
        :return
        """
        download_args["raw"] = False
        result = self.client.list_quotes(**download_args)

    def snapshot_download(self, file_name, **download_args):
        download_args["market_type"] = polygon.rest.snapshot.SnapshotMarketType.STOCKS
        download_args["raw"] = False
        self.client.list_snapshot_options_chain(**download_args)

    def trade_download(self, file_name, **download_args):
        ## Non-overridable args
        download_args["raw"] = False

        ## Overridable args
        self.client.list_trades(**download_args)
