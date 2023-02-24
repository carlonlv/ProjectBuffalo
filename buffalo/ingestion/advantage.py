"""
This module provide api access to Alpha-advantage api.
"""

import json
import warnings
from typing import List, Literal, Optional, Union, get_args, NewType

import pandas as pd
import requests

from . import configuration, enum

PositiveInt = NewType('PositiveInt', int)
PositiveFloat = NewType('PositiveFloat', float)

class PositiveInteger(int):
    def __new__(cls, value):
        if value < 0:
            raise ValueError("PositiveInteger cannot be negative")
        return super().__new__(cls, value)

class PositiveFloat(float):
    def __new__(cls, value):
        if value < 0:
            raise ValueError("PositiveFloat cannot be negative")
        return super().__new__(cls, value)

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
            tickers: Optional[str]=None,
            topics: Optional[Literal['blockchain', 'earnings', 'ipo', 'mergers_and_acquisitions', 'financial_markets', 'economy_fiscal', 'economy_monetary', 'economy_macro', 'energy_transportation', 'finance', 'life_sciences', 'manufacturing', 'real_estate', 'retail_wholesale', 'technology']]=None,
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
        :param limit: The number of maximum matching results to be returned. If you are looking for an even higher output limit, please contact support@alphavantage.co to have your limit boosted.
        :return: Downloaded data frame.
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
        if len(data) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return pd.json_normalize(data)
    
    def company_info_download(
        self,
        function: Literal['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS'],
        symbol: str) -> pd.Timestamp:
        """
        This method returns the company information, financial ratios, and other key metrics for the equity specified. Data is generally refreshed on the same day a company reports its latest earnings and financials.
        
        :param function: The function to retrieve different company related info.
            1. OVERVIEW: The company information, financial ratios, and other key metrics for the equity specified. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            2. INCOME_STATEMENT: The annual and quarterly income statements for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            3. BALANCE_SHEET: The annual and quarterly balance sheets for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            4. CASH_FLOW: The annual and quarterly cash flow for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            5. EARNINGS: The annual and quarterly earnings (EPS) for the company of interest. Quarterly data also includes analyst estimates and surprise metrics.
        :param symbol: The symbol of the token of your choice. For example: symbol=IBM.
        :return: Downloaded data frame.
        """
        function = 'NEWS_SENTIMENT'

        url = self._construct_url(
            function = function,
            symbol = symbol,
            apikey = self.api_key)

        response = requests.get(url, timeout=10)
        data = json.loads(response.text)
        if len(data) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return pd.json_normalize(data)

    def listing_info_download(
        self,
        date: Optional[str],
        state: Optional[Literal['active', 'delisted']]='active') -> pd.Timestamp:
        """
        This method returns a list of active or delisted US stocks and ETFs, either as of the latest trading day or at a specific time in history. The endpoint is positioned to facilitate equity research on asset lifecycle and survivorship.

        :param date: If no date is set, the API endpoint will return a list of active or delisted symbols as of the latest trading day. If a date is set, the API endpoint will "travel back" in time and return a list of active or delisted symbols on that particular date in history. Any YYYY-MM-DD date later than 2010-01-01 is supported. For example, date=2013-08-03
        :param state: By default, state=active and the API will return a list of actively traded stocks and ETFs. Set state=delisted to query a list of delisted assets.
        :return: Downloaded data frame.
        """
        function = 'LISTING_STATUS'

        url = self._construct_url(
            function = function,
            date = date,
            state = state,
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result
    
    def expected_earning_download(
        self,
        symbol: Optional[str],
        horizon: Optional[Literal['3month', '6month', '12month']]) -> pd.Timestamp:
        """
        This method returns a list of company earnings expected in the next 3, 6, or 12 months.
        
        :param symbol: By default, no symbol will be set for this API. When no symbol is set, the API endpoint will return the full list of company earnings scheduled. If a symbol is set, the API endpoint will return the expected earnings for that specific symbol. For example, symbol=IBM.
        :param horizon: By default, horizon=3month and the API will return a list of expected company earnings in the next 3 months. You may set horizon=6month or horizon=12month to query the earnings scheduled for the next 6 months or 12 months, respectively.
        :return: Downloaded data frame.
        """
        function = 'EARNINGS_CALENDAR'

        url = self._construct_url(
            function = function,
            symbol = symbol,
            horizon = horizon,
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def ipo_calendar_download(self) -> pd.Timestamp:
        """
        This method returns a list of IPOs expected in the next 3 months.

        :return: Downloaded data frame.
        """
        function = 'IPO_CALENDAR'

        url = self._construct_url(function = function, apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def moving_average_indicator_download(
        self, 
        function: Literal['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt,
        series_type: Literal['close', 'open', 'high', 'low'],
        fastlimit: PositiveFloat=0.01,
        slowlimit: PositiveFloat=0.01) -> pd.Timestamp:
        """
        This method returns moving average indicators of prices.

        :param function: The function to retrieve different moving average indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. SMA: Simple moving average (SMA) values.
            2. EMA: Exponential moving average (EMA) values.
            3. WMA: Weighted moving average (WMA) values.
            4. DEMA: Double exponential moving average (DEMA) values.
            5. TEMA: Triple exponential moving average (TEMA) values.
            6. TRIMA: Triangular moving average (TRIMA) values.
            7. KAMA: Kaufman adaptive moving average (KAMA) values.
            8. MAMA: MESA adaptive moving average (MAMA) values.
            9. T3: Triple exponential moving average (T3) values.
        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200).
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low.
        :param fastlimit: Positive floats are accepted. By default, fastlimit=0.01.
        :param slowlimit: Positive floats are accepted. By default, slowlimit=0.01.
        :return: Downloaded data frame.
        """
        if function != 'MAMA':
            fastlimit = None
            slowlimit = None

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            fastlimit = fastlimit,
            slowlimit = slowlimit,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def volume_weighted_average_price_indicator_download(
        self, 
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min']) -> pd.Timestamp:
        """
        This method returns the volume weighted average price (VWAP) for intraday time series. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)

        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min.
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = 'VWAP',
            symbol = symbol,
            interval = interval,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def moving_average_convergence_indicator_download(
        self,
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        series_type: Literal['close', 'open', 'high', 'low'],
        fastperiod: Optional[PositiveInt]=12,
        slowperiod: Optional[PositiveInt]=26,
        signalperiod: Optional[PositiveInt]=9,
        fastmatype: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]]=0,
        slowmatype: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]]=0,
        signalmatype: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]]=0) -> pd.Timestamp:
        """
        This method returns the moving average convergence / divergence indicators with controllable moving average type. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)

        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low.
        :param fastperiod: Positive integers are accepted. By default, fastperiod=12.
        :param slowperiod: Positive integers are accepted. By default, fastperiod=26.
        :param signalperiod: Positive integers are accepted. By default, signalperiod=9.
        :param fastmatype: Moving average type for the faster moving average. By default, fastmatype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).
        :param slowmatype: Moving average type for the slower moving average. By default, slowmatype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).
        :param signalmatype: Moving average type for the signal moving average. By default, signalmatype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = 'MACDEXT',
            symbol = symbol,
            interval = interval,
            series_type = series_type,
            fastperiod = fastperiod,
            slowperiod = slowperiod,
            signalperiod = signalperiod,
            fastmatype = fastmatype,
            slowmatype = slowmatype,
            signalmatype = signalmatype,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result
    
    def rsi_indicator_download(
        self,
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt,
        series_type: Literal['close', 'open', 'high', 'low']) -> pd.Timestamp:
        """
        This method returns the relative strength index(RSI) values. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)

        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200).
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low.
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = 'RSI',
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result


    def stochastic_oscillator_indicator_download(
        self,
        function: Literal['STOCH', 'STOCHF', 'STOCHRSI'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt,
        series_type: Literal['close', 'open', 'high', 'low'],
        fastkperiod: Optional[PositiveInt],
        fastdperiod: Optional[PositiveInt],
        fastdmatype: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]]=0) -> pd.Timestamp:
        """
        This method returns the stochastic oscilator indicator values. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)

        :param function: The function to retrieve different stochastic oscillator indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. STOCH: stochastic oscillator (STOCH) values.
            2. STOCHF: stochastic fast (STOCHF) values.
            3. STOCHRSI: stochastic relative strength index (STOCHRSI) values.
        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200).
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low.
        :param fastkperiod: The time period of the fastk moving average. Positive integers are accepted. By default, fastkperiod=5.
        :param fastdperiod: The time period of the fastd moving average. Positive integers are accepted. By default, fastdperiod=3.
        :param fastdmatype: Moving average type for the fastd moving average. By default, fastdmatype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            fastkperiod = fastkperiod,
            fastdperiod = fastdperiod,
            fastdmatype = fastdmatype,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result
