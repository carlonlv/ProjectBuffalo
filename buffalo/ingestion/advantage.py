"""
This module provide api access to Alpha-advantage api.
"""

import json
import warnings
from typing import List, Literal, Optional, Union, get_args

import pandas as pd
import requests

from . import configuration, enum

# Intervals = Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']

# EconIndicators = Literal['REAL_GDP', 'REAL_GDP_PER_CAPITA', 'TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 'CPI', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL']
# EconIntervals = Literal['annual', 'quarterly', 'daily', 'weekly', 'monthly', 'semiannual']
# EconMaturities = Literal['3month', '2year', '5year', '7year', '10year']

# Indicators = Literal['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'VWAP', 'T3', 'MACD', 'MACDEXT', 'STOCH', 'STOCHF', 'RSI', 'STOCHRSI', 'WILLR', 'ADX', 'ADXR', 'APO', 'PPO', 'MOM', 'BOP', 'CCI', 'CMO', 'ROC', 'ROCR', 'AROON', 'AROONOSC', 'MFI', 'TRIX', 'ULTOSC', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'BBANDS', 'MIDPOINT', 'MIDPRICE', 'SAR', 'TRANGE', 'ATR', 'NATR', 'AD', 'ADOSC', 'OBV', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR']

# INGESTION_METHODS = {
#     (enum.DataType.STOCK, enum.IngestionType.STREAM): [
#         {
#             'function': Literal['TIME_SERIES_INTRADAY_EXTENDED'],
#             'symbol': str,
#             'interval': Literal['1min', '5min', '15min', '30min', '60min'],
#             'slice': Literal['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6', 'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12', 'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6', 'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12'],
#             'adjusted': Optional[bool],
#             'apikey': str
#         },
#         {
#             'function': Literal['TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED'],
#             'symbol': str,
#             'outputsize': 'full',
#             'datatype': 'csv',
#             'apikey': str
#         },
#         {
#             'function': Literal['TIME_SERIES_WEEKLY', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY'],
#             'symbol': str,
#             'datatype': 'csv',
#             'apikey': str
#         }
#     ],
#     (enum.DataType.ECON, enum.IngestionType.STREAM): [
#         {
#             'function': 'REAL_GDP',
#             'interval': Optional[Literal['quarterly', 'annual']],
#             'datatype': 'csv',
#             'apikey': str
#         },
#         {
#             'function': ['REAL_GDP_PER_CAPITA', 'FEDERAL_FUNDS_RATE', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL'],
#             'datatype': 'csv',
#             'apikey': str
#         },
#         {
#             'function': 'TREASURY_YIELD',
#             'interval': Optional[Literal['daily', 'weekly', 'monthly']],
#             'maturity': Optional[Literal['3month', '2year', '5year', '7year', '10year', '30year']],
#             'datatype': 'csv',
#             'apikey': str
#         },
#         {
#             'function': 'CPI',
#             'interval': Optional[Literal['monthly', 'semiannual']],
#             'datatype': 'csv',
#             'apikey': str
#         }
#     ]
# }


class AdvantageStockGrepper:
    """
    This class is used to retrieve stock data from Alpha Advantage endpoint.
    """
    url_base = r'https://www.alphavantage.co/query?'

    def __init__(self) -> None:
        self.api_key = configuration.Configuration.api_keys[enum.API.ADVANTAGE]
    
    def _construct_url(self, **kwargs) -> str:
        """ Construct url from key word arguments.

        :param kwargs: The keywords and arguments for url construction.
        :return: The url address for query.
        """
        kwargs = {k:v for k, v in kwargs.items() if v is not None}
        lst = []
        for key, value in kwargs.items():
            lst.append(f'&{key}={value}')
        return self.url_base + ''.join(lst)

    def interday_stock_download(
            self,
            symbol: str,
            interval: Literal['1min', '5min', '15min', '30min', '60min'],
            year_slice: Optional[Literal['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6', 'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12', 'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6', 'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12']]='year1month1',
            adjusted: Optional[bool]=True
        ) -> pd.DataFrame:
        """
        This method returns historical intraday time series for the trailing 2 years, covering over 2 million data points per ticker. The intraday data is derived from the Securities Information Processor (SIP) market-aggregated data. You can query both raw (as-traded) and split/dividend-adjusted intraday data from this endpoint. Common use cases for This method include data visualization, trading simulation/backtesting, and machine learning and deep learning applications with a longer horizon.

        :param symbol: The name of the equity of your choice. For example: symbol=IBM.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min.
        :param year_slice: Two years of minute-level intraday data contains over 2 million data points, which can take up to Gigabytes of memory. To ensure optimal API response speed, the trailing 2 years of intraday data is evenly divided into 24 "slices" - year1month1, year1month2, year1month3, ..., year1month11, year1month12, year2month1, year2month2, year2month3, ..., year2month11, year2month12. Each slice is a 30-day window, with year1month1 being the most recent and year2month12 being the farthest from today. By default, slice=year1month1. 
        :param adjusted: By default, adjusted=true and the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = 'TIME_SERIES_INTRADAY_EXTENDED',
            symbol = symbol,
            interval = interval,
            slice = year_slice,
            adjusted = adjusted,
            apikey = self.api_key)
        
        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result
    
    def extraday_stock_download(
            self,
            symbol: str,
            interval: Literal['daily', 'weekly', 'monthly'],
            adjusted: Optional[bool]=True
        ) -> pd.DataFrame:
        """
        This method returns raw (as-traded) daily/weekly/monthly time series (date, open, high, low, close, volume) of the global equity specified, covering 20+ years of historical data. If you are also interested in split/dividend-adjusted historical data, please use the Daily Adjusted API, which covers adjusted close values and historical split and dividend events.

        :param symbol: The name of the equity of your choice. For example: symbol=IBM.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: daily, weekly, monthly.
        :param adjusted: By default, adjusted=true and the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
        :return: Downloaded data frame.
        """
        function = 'TIME_SERIES_' + interval.upper()

        if adjusted:
            function += '_ADJUTSED'

        url = self._construct_url(
            function = function,
            symbol = symbol,
            apikey = self.api_key,
            outputsize = 'full',
            datatype = 'csv')

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result
    
    def market_news_sentiment_download(
            self,
            tickers: Optional[str],
            topics: Optional[Literal['blockchain', 'earnings', 'ipo', 'mergers_and_acquisitions', 'financial_markets', 'economy_fiscal', 'economy_monetary', 'economy_macro', 'energy_transportation', 'finance', 'life_sciences', 'manufacturing', 'real_estate', 'retail_wholesale', 'technology']],
            time_from: Optional[str]=None,
            time_to: Optional[str]=None,
            sort: Optional[Literal['LATEST', 'EARLIEST', 'RELEVANCE']]='LATEST',
            limit: Optional[bool]=50,
        ) -> pd.Timestamp:
        """
        This method returns live and historical market news & sentiment data derived from over 50 major financial news outlets around the world, covering stocks, cryptocurrencies, forex, and a wide range of topics such as fiscal policy, mergers & acquisitions, IPOs, etc. This method, combined with our core stock API, fundamental data, and technical indicator APIs, can provide you with a 360-degree view of the financial market and the broader economy.

        :param tickers: The stock/crypto/forex symbols of your choice. For example: tickers=IBM will filter for articles that mention the IBM ticker; tickers=COIN,CRYPTO:BTC,FOREX:USD will filter for articles that simultaneously mention Coinbase (COIN), Bitcoin (CRYPTO:BTC), and US Dollar (FOREX:USD) in their content.
        :param topics: The news topics of your choice. For example: topics=technology will filter for articles that write about the technology sector; topics=technology,ipo will filter for articles that simultaneously cover technology and IPO in their content. Below is the full list of supported topics:
            1. Blockchain: blockchain
            2. Earnings: earnings 
            3. IPO: ipo
            4. Mergers & Acquisitions: mergers_and_acquisitions
            5. Financial Markets: financial_markets
            6. Economy - Fiscal Policy (e.g., tax reform, government spending): economy_fiscal
            7. Economy - Monetary Policy (e.g., interest rates, inflation): economy_monetary
            8. Economy - Macro/Overall: economy_macro
            9. Energy & Transportation: energy_transportation
            10. Finance: finance
            11. Life Sciences: life_sciences
            12. Manufacturing: manufacturing
            13. Real Estate & Construction: real_estate
            14. Retail & Wholesale: retail_wholesale
            15. Technology: technology
        :param time_from: The time range of the news articles you are targeting, in YYYYMMDDTHHMM format. For example: time_from=20220410T0130. If time_from is specified but time_to is missing, the API will return articles published between the time_from value and the current time.
        :param time_to: The time range of the news articles you are targeting, in YYYYMMDDTHHMM format. For example: time_from=20220410T0130. If time_from is specified but time_to is missing, the API will return articles published between the time_from value and the current time.
        :param sort: By default, sort=LATEST and the API will return the latest articles first. You can also set sort=EARLIEST or sort=RELEVANCE based on your use case.
        :param limit:
        """
        function = 'NEWS_SENTIMENT'

        url = self._construct_url(
            function = function,
            tickers = tickers,
            topics = topics,
            time_from = time_from,
            time_to = time_to,
            sort = sort,
            limit = limit,
            apikey = self.api_key)

        response = requests.get(url, timeout=10)
        data = json.loads(response.text)
        return pd.json_normalize(data)
    
    def company_info_download(
        self,
        function: Literal['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS'],
        symbol: str) -> pd.Timestamp:
        """
        This method returns the company information, financial ratios, and other key metrics for the equity specified. Data is generally refreshed on the same day a company reports its latest earnings and financials.
        
        :param function: The 
        """
        
        function = 'NEWS_SENTIMENT'

        url = self._construct_url(
            function = function,
            symbol = symbol,
            apikey = self.api_key)

        response = requests.get(url, timeout=10)
        data = json.loads(response.text)
        return pd.json_normalize(data)
        
        return

    # def stock_download(
    #     self,
    #     symbol: Optional[Union[str, List[str]]],
    #     interval: Intervals,
    #     from_time: Optional[pd.Timestamp]=None,
    #     to_time: Optional[pd.Timestamp]=None,
    #     adjusted: bool=True) -> pd.DataFrame:
    #     """
    #     Download stock data from Alpha vantage Endpoint.

    #     :param symbol: A list or string representing symbol, such as IBM, MSFT, etc. If not supplied, watched_symbol from Configuration will be used.
    #     :param interval: A timedelta representing interval. Currently, alpha advantage api supports 1min, 5min, 15min, 30min, 60min for interday data, and daily, weekly, monthly.
    #     :param from_time: A timestamp representing the start time. If None is supplied, then no filtering is done.
    #     :param to_time: A timestamp representing the to time. If None is supplied, then no filtering is done.
    #     :param adjusted: Whether the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
    #     :return: Downloaded dataframe.
    #     """
    #     if symbol is None:
    #         symbol = configuration.Configuration.watched_symbol

    #     args = expand_grid(ticker = symbol, year = range(1, 3), month = range(1, 13))
    #     args['slice_str'] = 'year'+ args['year'].astype(str) + 'month' + args['month'].astype(str)
    #     args['from_time'] = args.apply(lambda x: pd.Timestamp(datetime.now()) - pd.Timedelta(days=360*(x['year']-1)) - pd.Timedelta(days=30*x['month']), axis=1)
    #     args['to_time'] = args['from_time'] + pd.Timedelta(days=30)

    #     if from_time is not None:
    #         from_time = find_nearest_in_list(from_time, args['from_time'], round_down=True)
    #     else:
    #         from_time = args['from_time'].min()
    #     if to_time is not None:
    #         to_time = find_nearest_in_list(to_time, args['to_time'], round_up=True)
    #     else:
    #         to_time = args['to_time'].max()

    #     args = args.query(f'from_time >= {from_time} & to_time <= {to_time}')

    #     args['interval'] = interval
    #     args['adjusted'] = adjusted
    #     args['api_key'] = self.api_key

    #     if interval in ['1min', '5min', '15min', '30min', '60min']:
    #         args['function_name'] = 'TIME_SERIES_INTRADAY_EXTENDED'
    #     elif interval == 'daily':
    #         args['function_name'] = 'TIME_SERIES_DAILY'
    #     elif interval == 'weekly':
    #         args['function_name'] = 'TIME_SERIES_WEEKLY'
    #     else:
    #         args['function_name'] = 'TIME_SERIES_MONTHLY'

    #     if adjusted and interval in ['daily', 'weekly', 'monthly']:
    #         args['function_name'] += '_ADJUSTED'

    #     args['url'] = args.apply(lambda x: do_call(AdvantageStockGrepper._construct_url, **x.to_dict()))

    #     result = []
    #     for i in args.index:
    #         url = args.loc[i,'url']
    #         temp = pd.read_csv(url)
    #         if len(temp.index) == 0:
    #             warnings.warn(f'Reading from {url} results in 0 rows.')
    #         else:
    #             temp.loc['url'] = url
    #         result.append(temp)

    #     result = pd.concat(result).reset_index(drop=True)
    #     result['timestamp'] = pd.to_datetime(result['timestamp'])
    #     return result

    # def econ_download(
    #     self,
    #     indicators: Union[List[EconIndicators], EconIndicators],
    #     interval: Optional[EconIntervals],
    #     maturity: Optional[EconMaturities],
    #     from_time: Optional[pd.Timestamp]=None,
    #     to_time: Optional[pd.Timestamp]=None) -> pd.DataFrame:
    #     """
    #     Download US economics data from Alpha vantage Endpoint.

    #     Details on the choice of indicators: 
    #     1. REAL_GDP: Real GDP of the United States
    #     2. REAL_GDP_PER_CAPITA: Real GDP per Capita data of the United States
    #     3. TREASURY_YIELD: US treasury yield of a given maturity timeline
    #     4. FEDERAL_FUNDS_RATE: federal funds rate (interest rate) of the United States
    #     5. CPI: consumer price index (CPI) of the United States. CPI is widely regarded as the barometer of inflation levels in the broader economy.
    #     6. INFLATION: annual inflation rates (consumer prices) of the United States.
    #     7. RETAIL_SALES: monthly Advance Retail Sales: Retail Trade data of the United States.
    #     8. DURABLES: monthly manufacturers' new orders of durable goods in the United States.
    #     9. UNEMPLOYMENT: monthly unemployment data of the United States. The unemployment rate represents the number of unemployed as a percentage of the labor force. 
    #     10. NONFARM_PAYROLL: monthly US All Employees: Total Nonfarm (commonly known as Total Nonfarm Payroll), a measure of the number of U.S. workers in the economy that excludes proprietors, private household employees, unpaid volunteers, farm employees, and the unincorporated self-employed.

    #     :param indicators: One of the economics indicators from U.S. Bureau of Economic Analysis, Real Gross Domestic Product, retrieved from FRED, Federal Reserve Bank of St. Louis.
    #     :param interval: A timedelta representing interval. Currently, alpha advantage api supports 1min, 5min, 15min, 30min, 60min for interday data, and daily, weekly, monthly.
    #     :param maturity: maturity timeline, only used when indicator is TREASURY_YIELD.
    #     :param from_time: A timestamp representing the start time.
    #     :param to_time: A timestamp representing the to time.
    #     :return: Downloaded dataframe.
    #     """
    #     if indicators in ['REAL_GDP']:
    #         assert interval in ['quarterly', 'annual']
    #     elif indicators in ['TREASURY_YIELD']:
    #         assert interval in ['daily', 'weekly', 'monthly']
    #         assert maturity in ['3month', '2year', '5year', '7year', '10year', '30year']
    #     elif indicators in ['FEDERAL_FUNDS_RATE']:
    #         assert interval in ['daily', 'weekly', 'monthly']
    #     elif indicators in ['CPI']:
    #         assert interval in ['monthly', ' semiannual']
    #     else:
    #         interval = None
    #         maturity = None

    #     args = expand_grid(function_name = indicators, interval = interval, maturity = maturity)
    #     args['api_key'] = self.api_key

    #     args['url'] = args.apply(lambda x: do_call(AdvantageStockGrepper._construct_url, **x.to_dict()))

    #     result = []
    #     for i in args.index:
    #         url = args.loc[i,'url']
    #         temp = pd.read_csv(url)
    #         if len(temp.index) == 0:
    #             warnings.warn(f'Reading from {url} results in 0 rows.')
    #         else:
    #             temp.loc['url'] = url
    #         result.append(temp)

    #     result = pd.concat(result).reset_index(drop=True)
    #     result['date'] = pd.to_datetime(result['date'])
    #     result = result.query(f'date >= {from_time} & date <= {to_time}')
    #     return result
    
    # def indicator_download(
    #     self,
    #     indicators: Union[List[Indicators], Indicators],
    #     symbol: Optional[Union[str, List[str]]],
    #     interval: Intervals,
    #     time_period: int,
    #     series_type: Literal['close', 'open', 'high', 'low'],
    #     from_time: Optional[pd.Timestamp]=None,
    #     to_time: Optional[pd.Timestamp]=None,
    #     **additional_args) -> pd.DataFrame:
    #     """
    #     Download Technical indicators from Alpha vantage Endpoint.

    #     See wiki page for the representation of technical indicators.

    #     :param indicators: One of the economics indicators from U.S. Bureau of Economic Analysis, Real Gross Domestic Product, retrieved from FRED, Federal Reserve Bank of St. Louis.
    #     :param symbol: A list or string representing symbol, such as IBM, MSFT, etc. If not supplied, watched_symbol from Configuration will be used.
    #     :param interval: Time interval between two consecutive data points in the time series.
    #     :param time_period: Number of data points used to calculate each value. Positive integers are accepted.
    #     :param from_time: A timestamp representing the start time.
    #     :param to_time: A timestamp representing the to time.
    #     :return: Downloaded dataframe.
    #     """
    #     if indicators in ['VWAP', 'STOCH', 'BOP', 'ULTOSC', 'SAR', 'TRANGE', 'AD', 'ADOSC', 'OBV']:
    #         time_period = None
    #         series_type = None
    #     elif indicators in ['MAMA', 'MACD', 'MACDEXT', 'STOCHF', 'APO', 'PPO', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR']:
    #         time_period = None
    #     elif indicators in ['WILLR', 'ADX', 'ADX', 'CCI', 'AROON', 'AROONOSC', 'MFI', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'MIDPRICE', 'ATR', 'NATR']:
    #         series_type = None

    #     if symbol is None:
    #         symbol = configuration.Configuration.watched_symbol

    #     additional_args['function_name'] = indicators
    #     additional_args['symbol'] = symbol
    #     additional_args['interval'] = interval
    #     additional_args['time_period'] = time_period
    #     additional_args['series_type'] = series_type

    #     args = expand_grid(**additional_args)
    #     args['api_key'] = self.api_key

    #     args['url'] = args.apply(lambda x: do_call(AdvantageStockGrepper._construct_url, **x.to_dict()))

    #     result = []
    #     for i in args.index:
    #         url = args.loc[i,'url']
    #         temp = pd.read_csv(url)
    #         if len(temp.index) == 0:
    #             warnings.warn(f'Reading from {url} results in 0 rows.')
    #         else:
    #             temp.loc['url'] = url
    #         result.append(temp)

    #     result = pd.concat(result).reset_index(drop=True)
    #     result['date'] = pd.to_datetime(result['date'])
    #     result = result.query(f'date >= {from_time} & date <= {to_time}')
    #     return result
