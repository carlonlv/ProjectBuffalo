import polygon.rest

from StockGrepper import *
from Utility.helper import *
from datetime import date

class PolygonStockGrepper(StockGrepper):
    quote_download_default_args = {
        "ticker": "AAPL",
        "timestamp_lt": date.today(),
        "limit": 50000,
        "sort": polygon.rest.quotes.Sort.ASC,
        "order": "timestamp"
    }


    def __init__(self) -> None:
        self.client = doCall(polygon.rest.RESTClient, api_key=Configuration.api_key, **Configuration.additional_configs)

    def quote_download(self, file_name, **download_args):
        """ Download Quote using Polygon API
        
        :param file_name
        :return
        """
        download_args["raw"] = False
        result = self.client.list_quotes(**download_args)

    def quote_last_download(self, file_name, **download_args):
        self.client.get_last_quote(**download_args)

    def snapshot_download(self, file_name, **download_args):
        download_args["market_type"] = polygon.rest.snapshot.SnapshotMarketType.STOCKS
        download_args["raw"] = False
        self.client.list_snapshot_options_chain(**download_args)

    def trade_download(self, file_name, **download_args):
        ## Non-overridable args
        download_args["raw"] = False

        ## Overridable args
        self.client.list_trades(**download_args)
