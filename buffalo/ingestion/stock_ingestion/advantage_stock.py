"""
This module provide api access to Alpha-advantage api.
"""

import warnings
from typing import List, Optional, Union

from datetime import datetime
import pandas as pd

from .. import Configuration, data_grepper, enum

URL_BASE = r'https://www.alphavantage.co/query?'

class AdvantageStockGrepper(data_grepper.DataGrepper):
    """
    This class is used to retrieve stock data from Alpha Advantage endpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        self.api_key = Configuration.api_keys[enum.API.ADVANTAGE]

    def construct_url(self, function_name: str, ticker: str, interval: str, slice_str: Optional[str]):
        """
        Helper function to construct url for Alpha vantage Endpoint.

        :param function_name:
        :param ticker:
        :param interval:
        :param slice_str:
        """
        url = URL_BASE + f'function={function_name}' + f'&symbol={ticker}' + f'&interval={interval}' + '&datatype=csv&outputsize=full'
        url += f'&apikey={self.api_key}'
        if slice_str is not None:
            url += f'slice={slice_str}'
        return url

    def _interpret_time_args(self, from_time: pd.Timestamp, to_time: pd.Timestamp, interval=pd.Timedelta):
        """
        Helper function used to translate input interval to acceptable strings.

        :param interval: A timedelta representing interval.
        :param return: A function derived from construct_url.
        """
        today = datetime.now().date()
        min_from_time = pd.Timestamp(datetime.combine(today, datetime.strptime("05:00", "%H:%M").time()))
        max_to_time = pd.Timedelta(datetime.now())

        if from_time < min_from_time:
            warnings.warn(f'from_time is before the minimum from_time allowed from api, truncated using {min_from_time}.')
            from_time = min_from_time
        if to_time > max_to_time:
            warnings.warn(f'to_time is after the maximum to_time allowed from api, truncated using {max_to_time}.')
            to_time = max_to_time

        warn_message = 'Interval {interval} is not found, rounding to {to_interval}.'

        interday_interval_translation = {
            pd.Timedelta(minutes=1): '1min',
            pd.Timedelta(minutes=5): '5min',
            pd.Timedelta(minutes=15): '15min',
            pd.Timedelta(minutes=30): '30min',
            pd.Timedelta(minutes=60): '60min'
        }

        acceptable_intervals = list(interday_interval_translation.keys()) + [pd.Timedelta(days=1), pd.Timedelta(weeks=1), pd.Timedelta(days=30)]

        to_interval = interval
        if interval not in acceptable_intervals and interval not in acceptable_intervals:
            diff_interval = [interval - x for x in list(interday_interval_translation.keys()) + acceptable_intervals]
            to_interval = diff_interval[diff_interval.index(min(diff_interval))]
            warnings.warn(warn_message.format(interval = interval, to_interval = to_interval))

        if to_interval < pd.Timedelta(days=1):
            ## Interday
            return ''
        elif to_interval == pd.Timedelta(days=1):
            ## Daily
            print()
        elif to_interval == pd.Timedelta(weeks=1):
            ## Months
            print()
        else:
            to_interval = min()
        return

    def ingestion_from_web(
        self,
        tickers:Optional[Union[str, List[str]]],
        from_time: pd.Timestamp=pd.Timestamp.now('UTC')-pd.Timedelta(days=365),
        to_time: pd.Timestamp=pd.Timestamp.now('UTC'),
        interval: pd.Timedelta=pd.Timedelta(days=1),
        adjusted: bool=True):
        """
        Download data from Alpha vantage Endpoint.

        :param tickers: A list or string representing tickers, such as IBM, MSFT, etc.
        :param from_time: A timestamp representing the start time.
        :param to_time: A timestamp representing the to time.
        :param interval: A timedelta representing interval. Currently, alpha advantage api supports 1min, 5min, 15min, 30min, 60min for interday data, and 1d, 1week, 30d(1month).
        :param adjusted: Whether the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
        """
        if tickers is None:
            tickers = Configuration.watched_tickers
        for ticker in tickers:
            self.construct_url(function_name = 'TIME_SERIES_INTRADAY_EXTENDED', ticker = ticker, interval = interval, slice_str = None)
        return
