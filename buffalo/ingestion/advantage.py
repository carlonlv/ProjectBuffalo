"""
This module provide api access to Alpha-advantage api.
"""

import itertools
import warnings
from datetime import datetime
from typing import List, Literal, Optional, Union, get_args

import pandas as pd

from buffalo.utility import do_call, expand_grid, find_nearest_in_list

from . import configuration, enum

URL_BASE = r'https://www.alphavantage.co/query?'

Intervals = Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']

EconIndicators = Literal['REAL_GDP', 'REAL_GDP_PER_CAPITA', 'TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 'CPI', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL']
EconIntervals = Literal['annual', 'quarterly', 'daily', 'weekly', 'monthly', 'semiannual']
EconMaturities = Literal['3month', '2year', '5year', '7year', '10year']

Indicators = Literal['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'VWAP', 'T3', 'MACD', 'MACDEXT', 'STOCH', 'STOCHF', 'RSI', 'STOCHRSI', 'WILLR', 'ADX', 'ADXR', 'APO', 'PPO', 'MOM', 'BOP', 'CCI', 'CMO', 'ROC', 'ROCR', 'AROON', 'AROONOSC', 'MFI', 'TRIX', 'ULTOSC', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'BBANDS', 'MIDPOINT', 'MIDPRICE', 'SAR', 'TRANGE', 'ATR', 'NATR', 'AD', 'ADOSC', 'OBV', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR']

INGESTION_METHODS = {
    (enum.DataType.STOCK, enum.IngestionType.STREAM): [
        {
            'function': Literal['TIME_SERIES_INTRADAY_EXTENDED'],
            'symbol': str,
            'interval': Literal['1min', '5min', '15min', '30min', '60min'],
            'slice': Literal['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6', 'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12', 'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6', 'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12'],
            'adjusted': Optional[bool],
            'apikey': str
        },
        {
            'function': Literal['TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED'],
            'symbol': str,
            'outputsize': 'full',
            'datatype': 'csv',
            'apikey': str
        },
        {
            'function': Literal['TIME_SERIES_WEEKLY', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY'],
            'symbol': str,
            'datatype': 'csv',
            'apikey': str
        }
    ],
    (enum.DataType.ECON, enum.IngestionType.STREAM): [
        {
            'function': 'REAL_GDP',
            'interval': Optional[Literal['quarterly', 'annual']],
            'datatype': 'csv',
            'apikey': str
        },
        {
            'function': ['REAL_GDP_PER_CAPITA', 'FEDERAL_FUNDS_RATE', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL'],
            'datatype': 'csv',
            'apikey': str
        },
        {
            'function': 'TREASURY_YIELD',
            'interval': Optional[Literal['daily', 'weekly', 'monthly']],
            'maturity': Optional[Literal['3month', '2year', '5year', '7year', '10year', '30year']],
            'datatype': 'csv',
            'apikey': str
        },
        {
            'function': 'CPI',
            'interval': Optional[Literal['monthly', 'semiannual']],
            'datatype': 'csv',
            'apikey': str
        }
    ]
}


class AdvantageStockGrepper:
    """
    This class is used to retrieve stock data from Alpha Advantage endpoint.
    """

    def __init__(self) -> None:
        self.api_key = configuration.Configuration.api_keys[enum.API.ADVANTAGE]
        self.ingestion_methods = {
            (enum.DataType.STOCK, enum.IngestionType.REST): self.stock_download,
            (enum.DataType.ECON, enum.IngestionType.REST): self.econ_download
        }

    @staticmethod
    def _construct_url(api_key:str, function_name: str, ticker: Optional[str], interval: Optional[str], slice_str: Optional[str], adjusted: Optional[str], maturity: Optional[str], time_periord: Optional[str], series_type: Optional[str]):
        """
        Helper function to construct url for Alpha vantage Endpoint.

        :param api_key: A string representing access key.
        :param function_name: A string representing function name.
        :param ticker: A string representing tickers.
        :param interval: A string representing the interval between measurements.
        :param slice_str: A string representation acceptable by the api, e.g., yearmonthly1.
        :param adjusted: A string represeting whether the raw (as-traded) and split/dividend-adjusted data should be downloaded.
        :param time_period: A string representing number of data points used to calculate each value.
        :param series_type: A string representing the desired price type in the time series.
        :return:
        """
        url = URL_BASE + f'function={function_name}' + f'&apikey={api_key}'

        if function_name in ['TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED', 'TIME_SERIES_WEEKLY', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY', 'TIME_SERIES_MONTHLY_ADJUSTED']:
            url += '&datatype=csv'

        if function_name in ['TIME_SERIES_INTRADAY_EXTENDED', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED']:
            url += '&outputsize=full'

        if function_name in ['TIME_SERIES_INTRADAY_EXTENDED', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED', 'TIME_SERIES_WEEKLY', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY', 'TIME_SERIES_MONTHLY_ADJUSTED']:
            url += f'&symbol={ticker}'

        if function_name == 'TIME_SERIES_INTRADAY_EXTENDED':
            if slice_str is not None:
                url += f'slice={slice_str}'
            if adjusted is not None:
                url += f'adjusted={adjusted}'

        if function_name in get_args(Indicators):
            url += '&datatype=csv'

        if function_name in get_args(EconIndicators):
            url += '&datatype=csv'

        if interval is not None:
            url += f'&interval={interval}'

        if maturity is not None:
            url += f'&maturity={maturity}'

        if time_periord is not None:
            url += f'&time_period={time_periord}'

        if series_type is not None:
            url += f'&series_type={series_type}'
        
        return url

    def stock_download(
        self,
        tickers: Optional[Union[str, List[str]]],
        interval: Intervals,
        from_time: Optional[pd.Timestamp]=None,
        to_time: Optional[pd.Timestamp]=None,
        adjusted: bool=True) -> pd.DataFrame:
        """
        Download stock data from Alpha vantage Endpoint.

        :param tickers: A list or string representing tickers, such as IBM, MSFT, etc. If not supplied, watched_tickers from Configuration will be used.
        :param interval: A timedelta representing interval. Currently, alpha advantage api supports 1min, 5min, 15min, 30min, 60min for interday data, and daily, weekly, monthly.
        :param from_time: A timestamp representing the start time. If None is supplied, then no filtering is done.
        :param to_time: A timestamp representing the to time. If None is supplied, then no filtering is done.
        :param adjusted: Whether the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
        :return: Downloaded dataframe.
        """
        if tickers is None:
            tickers = configuration.Configuration.watched_tickers

        args = expand_grid(ticker = tickers, year = range(1, 3), month = range(1, 13))
        args['slice_str'] = 'year'+ args['year'].astype(str) + 'month' + args['month'].astype(str)
        args['from_time'] = args.apply(lambda x: pd.Timestamp(datetime.now()) - pd.Timedelta(days=360*(x['year']-1)) - pd.Timedelta(days=30*x['month']), axis=1)
        args['to_time'] = args['from_time'] + pd.Timedelta(days=30)

        if from_time is not None:
            from_time = find_nearest_in_list(from_time, args['from_time'], round_down=True)
        else:
            from_time = args['from_time'].min()
        if to_time is not None:
            to_time = find_nearest_in_list(to_time, args['to_time'], round_up=True)
        else:
            to_time = args['to_time'].max()

        args = args.query(f'from_time >= {from_time} & to_time <= {to_time}')

        args['interval'] = interval
        args['adjusted'] = adjusted
        args['api_key'] = self.api_key

        if interval in ['1min', '5min', '15min', '30min', '60min']:
            args['function_name'] = 'TIME_SERIES_INTRADAY_EXTENDED'
        elif interval == 'daily':
            args['function_name'] = 'TIME_SERIES_DAILY'
        elif interval == 'weekly':
            args['function_name'] = 'TIME_SERIES_WEEKLY'
        else:
            args['function_name'] = 'TIME_SERIES_MONTHLY'

        if adjusted and interval in ['daily', 'weekly', 'monthly']:
            args['function_name'] += '_ADJUSTED'

        args['url'] = args.apply(lambda x: do_call(AdvantageStockGrepper._construct_url, **x.to_dict()))

        result = []
        for i in args.index:
            url = args.loc[i,'url']
            temp = pd.read_csv(url)
            if len(temp.index) == 0:
                warnings.warn(f'Reading from {url} results in 0 rows.')
            else:
                temp.loc['url'] = url
            result.append(temp)

        result = pd.concat(result).reset_index(drop=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        return result

    def econ_download(
        self,
        indicators: Union[List[EconIndicators], EconIndicators],
        interval: Optional[EconIntervals],
        maturity: Optional[EconMaturities],
        from_time: Optional[pd.Timestamp]=None,
        to_time: Optional[pd.Timestamp]=None) -> pd.DataFrame:
        """
        Download US economics data from Alpha vantage Endpoint.

        Details on the choice of indicators: 
        1. REAL_GDP: Real GDP of the United States
        2. REAL_GDP_PER_CAPITA: Real GDP per Capita data of the United States
        3. TREASURY_YIELD: US treasury yield of a given maturity timeline
        4. FEDERAL_FUNDS_RATE: federal funds rate (interest rate) of the United States
        5. CPI: consumer price index (CPI) of the United States. CPI is widely regarded as the barometer of inflation levels in the broader economy.
        6. INFLATION: annual inflation rates (consumer prices) of the United States.
        7. RETAIL_SALES: monthly Advance Retail Sales: Retail Trade data of the United States.
        8. DURABLES: monthly manufacturers' new orders of durable goods in the United States.
        9. UNEMPLOYMENT: monthly unemployment data of the United States. The unemployment rate represents the number of unemployed as a percentage of the labor force. 
        10. NONFARM_PAYROLL: monthly US All Employees: Total Nonfarm (commonly known as Total Nonfarm Payroll), a measure of the number of U.S. workers in the economy that excludes proprietors, private household employees, unpaid volunteers, farm employees, and the unincorporated self-employed.

        :param indicators: One of the economics indicators from U.S. Bureau of Economic Analysis, Real Gross Domestic Product, retrieved from FRED, Federal Reserve Bank of St. Louis.
        :param interval: A timedelta representing interval. Currently, alpha advantage api supports 1min, 5min, 15min, 30min, 60min for interday data, and daily, weekly, monthly.
        :param maturity: maturity timeline, only used when indicator is TREASURY_YIELD.
        :param from_time: A timestamp representing the start time.
        :param to_time: A timestamp representing the to time.
        :return: Downloaded dataframe.
        """
        if indicators in ['REAL_GDP']:
            assert interval in ['quarterly', 'annual']
        elif indicators in ['TREASURY_YIELD']:
            assert interval in ['daily', 'weekly', 'monthly']
            assert maturity in ['3month', '2year', '5year', '7year', '10year', '30year']
        elif indicators in ['FEDERAL_FUNDS_RATE']:
            assert interval in ['daily', 'weekly', 'monthly']
        elif indicators in ['CPI']:
            assert interval in ['monthly', ' semiannual']
        else:
            interval = None
            maturity = None

        args = expand_grid(function_name = indicators, interval = interval, maturity = maturity)
        args['api_key'] = self.api_key

        args['url'] = args.apply(lambda x: do_call(AdvantageStockGrepper._construct_url, **x.to_dict()))

        result = []
        for i in args.index:
            url = args.loc[i,'url']
            temp = pd.read_csv(url)
            if len(temp.index) == 0:
                warnings.warn(f'Reading from {url} results in 0 rows.')
            else:
                temp.loc['url'] = url
            result.append(temp)

        result = pd.concat(result).reset_index(drop=True)
        result['date'] = pd.to_datetime(result['date'])
        result = result.query(f'date >= {from_time} & date <= {to_time}')
        return result
    
    def indicator_download(
        self,
        indicators: Union[List[Indicators], Indicators],
        tickers: Optional[Union[str, List[str]]],
        interval: Intervals,
        time_period: int,
        series_type: Literal['close', 'open', 'high', 'low'],
        from_time: Optional[pd.Timestamp]=None,
        to_time: Optional[pd.Timestamp]=None,
        **additional_args) -> pd.DataFrame:
        """
        Download Technical indicators from Alpha vantage Endpoint.

        See wiki page for the representation of technical indicators.

        :param indicators: One of the economics indicators from U.S. Bureau of Economic Analysis, Real Gross Domestic Product, retrieved from FRED, Federal Reserve Bank of St. Louis.
        :param tickers: A list or string representing tickers, such as IBM, MSFT, etc. If not supplied, watched_tickers from Configuration will be used.
        :param interval: Time interval between two consecutive data points in the time series.
        :param time_period: Number of data points used to calculate each value. Positive integers are accepted.
        :param from_time: A timestamp representing the start time.
        :param to_time: A timestamp representing the to time.
        :return: Downloaded dataframe.
        """
        if indicators in ['VWAP', 'STOCH', 'BOP', 'ULTOSC', 'SAR', 'TRANGE', 'AD', 'ADOSC', 'OBV']:
            time_period = None
            series_type = None
        elif indicators in ['MAMA', 'MACD', 'MACDEXT', 'STOCHF', 'APO', 'PPO', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR']:
            time_period = None
        elif indicators in ['WILLR', 'ADX', 'ADX', 'CCI', 'AROON', 'AROONOSC', 'MFI', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'MIDPRICE', 'ATR', 'NATR']:
            series_type = None

        if tickers is None:
            tickers = configuration.Configuration.watched_tickers

        additional_args['function_name'] = indicators
        additional_args['tickers'] = tickers
        additional_args['interval'] = interval
        additional_args['time_period'] = time_period
        additional_args['series_type'] = series_type

        args = expand_grid(**additional_args)
        args['api_key'] = self.api_key

        args['url'] = args.apply(lambda x: do_call(AdvantageStockGrepper._construct_url, **x.to_dict()))

        result = []
        for i in args.index:
            url = args.loc[i,'url']
            temp = pd.read_csv(url)
            if len(temp.index) == 0:
                warnings.warn(f'Reading from {url} results in 0 rows.')
            else:
                temp.loc['url'] = url
            result.append(temp)

        result = pd.concat(result).reset_index(drop=True)
        result['date'] = pd.to_datetime(result['date'])
        result = result.query(f'date >= {from_time} & date <= {to_time}')
        return result
