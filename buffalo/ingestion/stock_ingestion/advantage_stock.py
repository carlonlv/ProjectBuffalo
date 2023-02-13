"""
This module provide api access to Alpha-advantage api.
"""

import warnings
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
import utility

from .. import Configuration, data_grepper, enum

URL_BASE = r'https://www.alphavantage.co/query?'
INTERDAY_INTERVAL_TRANSLATION = {
    pd.Timedelta(minutes=1): '1min',
    pd.Timedelta(minutes=5): '5min',
    pd.Timedelta(minutes=15): '15min',
    pd.Timedelta(minutes=30): '30min',
    pd.Timedelta(minutes=60): '60min'
}

class AdvantageStockGrepper(data_grepper.DataGrepper):
    """
    This class is used to retrieve stock data from Alpha Advantage endpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        self.api_key = Configuration.api_keys[enum.API.ADVANTAGE]

    @staticmethod
    def _construct_url(api_key:str, function_name: str, ticker: str, interval: str, slice_str: Optional[str], adjusted: Optional[str]):
        """
        Helper function to construct url for Alpha vantage Endpoint.

        :param function_name:
        :param ticker:
        :param interval:
        :param slice_str:
        """
        url = URL_BASE + f'function={function_name}' + f'&symbol={ticker}' + f'&interval={interval}' + f'&apikey={api_key}' + '&datatype=csv'

        if function_name in ['TIME_SERIES_INTRADAY_EXTENDED', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED']:
            url += '&outputsize=full'

        if function_name == 'TIME_SERIES_INTRADAY_EXTENDED':
            if slice_str is not None:
                url += f'slice={slice_str}'
            if adjusted is not None:
                url += f'adjusted={adjusted}'

        return url

    @staticmethod
    def _generated_args_df(tickers: Union[str, List[str]], from_time: pd.Timestamp, to_time: pd.Timestamp, interval: pd.Timedelta, adjusted: bool, api_key: str) -> pd.DataFrame:
        """
        Helper function used to translate input interval to acceptable strings.

        :param tickers:
        :param from_time: A timestamp representing the start time.
        :param to_time: A timestamp representing the to time.
        :param interval: A timedelta representing interval.
        :param adjusted: Whether the raw (as-traded) and split/dividend-adjusted data should be downloaded.
        :return: A function derived from construct_url.
        """
        time_slices = utility.expand_grid(ticker = tickers, year = range(1, 3), month = range(1, 13))
        time_slices['slice_str'] = 'year'+ time_slices['year'].astype(str) + 'month' + time_slices['month'].astype(str)
        time_slices['from_time'] = time_slices.apply(lambda x: pd.Timestamp(datetime.now()) - pd.Timedelta(days=360*(x['year']-1)) - pd.Timedelta(days=30*x['month']), axis=1)
        time_slices['to_time'] = time_slices['from_time'] + pd.Timedelta(days=30)

        from_time = utility.find_nearest_in_list(from_time, time_slices['from_time'], round_down=True)
        to_time = utility.find_nearest_in_list(to_time, time_slices['to_time'], round_up=True)

        time_slices = time_slices.query('from_time >= @from_time & to_time <= @to_time')

        acceptable_intervals = list(INTERDAY_INTERVAL_TRANSLATION.keys()) + [pd.Timedelta(days=1), pd.Timedelta(weeks=1), pd.Timedelta(days=30)]

        interval = utility.find_nearest_in_list(interval, acceptable_intervals)

        time_slices['interval'] = interval
        time_slices['adjusted'] = adjusted
        time_slices['api_key'] = api_key

        if interval < pd.Timedelta(days=1):
            time_slices['function_name'] = 'TIME_SERIES_INTRADAY_EXTENDED'
        elif interval == pd.Timedelta(days=1):
            time_slices['function_name'] = 'TIME_SERIES_DAILY'
        elif interval == pd.Timedelta(weeks=1):
            time_slices['function_name'] = 'TIME_SERIES_WEEKLY'
        else:
            time_slices['function_name'] = 'TIME_SERIES_MONTHLY'

        if adjusted and interval > pd.Timedelta(days=1):
            time_slices['function_name'] += '_ADJUSTED'

        time_slices['url'] = time_slices.apply(lambda x: utility.do_call(AdvantageStockGrepper._construct_url, **x.to_dict()))

        return time_slices

    def ingestion_from_web(
        self,
        tickers:Optional[Union[str, List[str]]],
        from_time: pd.Timestamp=pd.Timestamp.now('UTC')-pd.Timedelta(days=365),
        to_time: pd.Timestamp=pd.Timestamp.now('UTC'),
        interval: pd.Timedelta=pd.Timedelta(days=1),
        adjusted: bool=True):
        """
        Download data from Alpha vantage Endpoint.

        :param tickers: A list or string representing tickers, such as IBM, MSFT, etc. If not supplied, watched_tickers from Configuration will be used.
        :param from_time: A timestamp representing the start time.
        :param to_time: A timestamp representing the to time.
        :param interval: A timedelta representing interval. Currently, alpha advantage api supports 1min, 5min, 15min, 30min, 60min for interday data, and 1d, 1week, 30d(1month).
        :param adjusted: Whether the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
        """
        if tickers is None:
            tickers = Configuration.watched_tickers

        args = AdvantageStockGrepper._generated_args_df(tickers, from_time, to_time, interval, adjusted, self.api_key)

        result = []
        for i in args.index:
            url = args.loc[i,'url']
            temp = pd.read_csv(url)
            if len(temp.index) == 0:
                warnings.warn(f'Reading from {url} results in 0 rows.')
            else:
                temp.loc['url'] = url
            result.append(temp)

        return pd.concat(result).reset_index(drop=True)
