"""
This module provide api access to Alpha-advantage api.
"""

import json
import warnings
from typing import Literal, Optional, NewType

import pandas as pd
import requests

from . import configuration, enum

PositiveInt = NewType('PositiveInt', int)
PositiveFlt = NewType('PositiveFloat', float)

class PositiveInteger(int):
    """ Custom data type of positive integer to enforce type checking.
    """
    def __new__(cls, value):
        if value < 0:
            raise ValueError("PositiveInteger cannot be negative")
        return super().__new__(cls, value)

class PositiveFloat(float):
    """ Custom data type of positive float to enforce type checking.
    """
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
            lst.append(f'{key}={value}')
        return self.url_base + '&'.join(lst)

    def stock_download(
            self,
            symbol: str,
            interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
            year_slice: Optional[Literal['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6', 'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12', 'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6', 'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12']]='year1month1',
            adjusted: Optional[bool]=True) -> pd.DataFrame:
        """
        This method returns raw (as-traded) intraday/daily/weekly/monthly time series (date, open, high, low, close, volume) of the global equity specified, covering 20+ years of historical data. If you are also interested in split/dividend-adjusted historical data, please use the Daily Adjusted API, which covers adjusted close values and historical split and dividend events.

        :param symbol: The name of the equity of your choice. For example: symbol=IBM.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param year_slice: Two years of minute-level intraday data contains over 2 million data points, which can take up to Gigabytes of memory. To ensure optimal API response speed, the trailing 2 years of intraday data is evenly divided into 24 "slices" - year1month1, year1month2, year1month3, ..., year1month11, year1month12, year2month1, year2month2, year2month3, ..., year2month11, year2month12. Each slice is a 30-day window, with year1month1 being the most recent and year2month12 being the farthest from today. By default, slice=year1month1. 
        :param adjusted: By default, adjusted=true and the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
        :return: Downloaded data frame.
        """
        if interval in ['daily', 'weekly', 'monthly']:
            function = 'TIME_SERIES_' + interval.upper()
        else:
            function = 'TIME_SERIES_INTRADAY_EXTENDED'

        if adjusted and interval in ['daily', 'weekly', 'monthly']:
            function += '_ADJUSTED'
            adjusted = None
            interval = None

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            slice = year_slice,
            outputsize = 'full',
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def forex_download(
            self,
            from_symbol: str,
            to_symbol: str,
            interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']) -> pd.DataFrame:
        """
        This API returns intraday/daily/weekly/montly time series (timestamp, open, high, low, close) of the FX currency pair specified, updated realtime.

        :param from_symbol: A three-letter symbol from the forex currency list. For example: from_symbol=EUR
        :param to_symbol: A three-letter symbol from the forex currency list. For example: to_symbol=USD
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: realtime, 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly. 
        :return: Downloaded data frame.
        """
        if interval in ['daily', 'weekly', 'monthly']:
            function = 'FX_' + interval.upper()

            if interval in ['weekly', 'monthly']:
                outputsize = None
            else:
                outputsize = 'full'
            interval = None
        else:
            function = 'FX_INTRADAY'

        url = self._construct_url(
            function = function,
            from_symbol = from_symbol,
            to_symbol = to_symbol,
            outputsize = outputsize,
            interval = interval,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def currency_exchange_download(
            self,
            from_currency: str,
            to_currecy: str) -> pd.DataFrame:
        """ This method returns the realtime exchange rate for a pair of digital currency (e.g., Bitcoin) and physical currency (e.g., USD).

        :param from_currency: The currency you would like to get the exchange rate for. It can either be a physical currency or digital/crypto currency. For example: from_currency=USD or from_currency=BTC.
        :param to_currency: The destination currency for the exchange rate. It can either be a physical currency or digital/crypto currency. For example: to_currency=USD or to_currency=BTC.
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = 'CURRENCY_EXCHANGE_RATE',
            from_currency = from_currency,
            to_currecy = to_currecy,
            apikey = self.api_key)

        response = requests.get(url, timeout=10)
        data = json.loads(response.text)
        if len(data) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return pd.json_normalize(data)

    def crypto_exchange_download(
            self,
            digital_symbol: str,
            physical_symbol: str,
            interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']) -> pd.DataFrame:
        """
        This method returns intraday/daily/weekly/monthly time series (timestamp, open, high, low, close, volume) of the cryptocurrency specified, updated realtime.

        :param digital_symbol: The digital/crypto currency of your choice. It can be any of the currencies in the digital currency list. For example: digital_symbol=ETH.
        :param physical_symbol: The exchange market of your choice. It can be any of the market in the market list. For example: physical_symbol=USD.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :return: Downloaded data frame.
        """
        if interval in ['daily', 'weekly', 'monthly']:
            function = 'DIGITAL_CURRENCY' + interval.upper()
            interval = None
            outputsize = None
        else:
            function = 'CRYPTO_INTRADAY'
            outputsize = 'full'

        url = self._construct_url(
            function = function,
            symbol = digital_symbol,
            market = physical_symbol,
            outputsize = outputsize,
            interval = interval,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def currency_list_download(self, currency: Literal['physical', 'digital']='physical'):
        """
        This method returns the realtime exchange rate for a pair of digital currency (e.g., Bitcoin) and physical currency (e.g., USD).

        :param currency: Either physical currency or digital/crypto currency.
        """
        if currency == 'physical':
            url = 'https://www.alphavantage.co/physical_currency_list/'
        else:
            url = 'https://www.alphavantage.co/digital_currency_list/'

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

        result = pd.json_normalize(data, record_path=['feed'], meta = ['items', 'sentiment_score_definition', 'relevance_score_definition'])
        result = result.reset_index(drop=False)

        topic_lst = []
        ticker_lst = []
        for index in result.index:
            temp = pd.json_normalize(result.loc[index,'topics']).assign(index = index).rename(columns={'relevance_score': 'topic_relevance_score'})
            topic_lst.append(temp)
            temp = pd.json_normalize(result.loc[index,'ticker_sentiment']).assign(index = index).rename(columns={'relevance_score': 'ticker_relevance_score'})
            ticker_lst.append(temp)
        topic_lst = pd.concat(topic_lst)
        ticker_lst = pd.concat(ticker_lst)

        result = result.drop(columns=['topics', 'ticker_sentiment'])
        
        return result.merge(topic_lst), result.merge(ticker_lst)

    def company_info_download(
        self,
        function: Literal['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS', 'EARNINGS_CALENDAR'],
        symbol: str,
        horizon: Optional[Literal['3month', '6month', '12month']]='3month') -> pd.DataFrame:
        """
        This method returns the company information, financial ratios, and other key metrics for the equity specified. Data is generally refreshed on the same day a company reports its latest earnings and financials.
        
        :param function: The function to retrieve different company related info.
            1. OVERVIEW: The company information, financial ratios, and other key metrics for the equity specified. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            2. INCOME_STATEMENT: The annual and quarterly income statements for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            3. BALANCE_SHEET: The annual and quarterly balance sheets for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            4. CASH_FLOW: The annual and quarterly cash flow for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
            5. EARNINGS: The annual and quarterly earnings (EPS) for the company of interest. Quarterly data also includes analyst estimates and surprise metrics.
            6. EARNINGS_CALENDAR: A list of company earnings expected in the next 3, 6, or 12 months.
        :param symbol: The symbol of the token of your choice. For example: symbol=IBM.
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = function,
            symbol = symbol,
            horizon = horizon,
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
        url = self._construct_url(
            function = 'LISTING_STATUS',
            date = date,
            state = state,
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

    def trend_indicator_download(
        self,
        function: Literal['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3', 'ADX', 'ADXR', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'SAR', 'TRANGE', 'BBANDS', 'MIDPOINT', 'MIDPRICE'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt,
        series_type: Literal['close', 'open', 'high', 'low'],
        fastlimit: PositiveFlt=0.01,
        slowlimit: PositiveFlt=0.01,
        acceleration: PositiveFlt=0.01,
        maximum: PositiveFlt=0.2,
        nbdevup: PositiveInt=2,
        nbdevdn: PositiveInt=2,
        matype: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]=0) -> pd.Timestamp:
        """
        This method returns trend indicators of prices.

        :param function: The function to retrieve different trend indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. SMA: Simple moving average (SMA) values.
            2. EMA: Exponential moving average (EMA) values.
            3. WMA: Weighted moving average (WMA) values.
            4. DEMA: Double exponential moving average (DEMA) values.
            5. TEMA: Triple exponential moving average (TEMA) values.
            6. TRIMA: Triangular moving average (TRIMA) values.
            7. KAMA: Kaufman adaptive moving average (KAMA) values.
            8. MAMA: MESA adaptive moving average (MAMA) values.
            9. T3: Triple exponential moving average (T3) values.
            10. ADX: Average directional movement index (ADX) values. 
            11. ADXR: Average directional movement index rating (ADXR) values.
            12. DX: Directional movement index (DX) values.
            13. MINUS_DI: Minus directional indicator (MINUS_DI) values.
            14. PLUS_DI: Plus directional indicator (PLUS_DI) values.
            15. MINUS_DM: Minus directional movement (MINUS_DM) values.
            16. PLUS_DM: Plus directional movement (PLUS_DM) values.
            17. SAR: Parabolic SAR (SAR) values.
            18. TRANGE: True range (TRANGE) values.
            19. BBANDS: Bollinger bands (BBANDS) values.
            20. MIDPOINT: Midpoint (MIDPOINT) values.
            21. MIDPRICE: Midpoint price (MIDPRICE) values.
        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200). Not used when function is MAMA.
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low. Not used when function is ADX, ADXR, DX, MINUS_DI, PLUS_DI, MINUS_DM, PLUS_DM, ATR, NATR, MIDPRICE.
        :param fastlimit: Positive floats are accepted. By default, fastlimit=0.01. Only used when function is MAMA.
        :param slowlimit: Positive floats are accepted. By default, slowlimit=0.01. Only used when function is MAMA.
        :param acceleration: The acceleration factor. Positive floats are accepted. By default, acceleration=0.01. Only used when function is SAR.
        :param maximum: The acceleration factor maximum value. Positive floats are accepted. By default, maximum=0.20. Only used when function is SAR.
        :param nbdevup: The standard deviation multiplier of the upper band. Positive integers are accepted. By default, nbdevup=2. Only used when function is BBAND.
        :param nbdevdn: The standard deviation multiplier of the lower band. Positive integers are accepted. By default, nbdevdn=2. Only used when function is BBAND.
        :param matype: Moving average type of the time series. By default, matype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). Only used when function is BBAND.
        :return: Downloaded data frame.
        """
        if function not in ['MAMA']:
            fastlimit = None
            slowlimit = None

        if function not in ['SAR']:
            acceleration = None
            maximum = None

        if function not in ['BBAND']:
            nbdevup = None
            nbdevdn = None
            matype = None

        if function in ['MAMA']:
            time_period = None

        if function in ['ADX', 'ADXR', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'ATR', 'NATR', 'MIDPRICE']:
            series_type = None

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            fastlimit = fastlimit,
            slowlimit = slowlimit,
            acceleration = acceleration,
            maximum = maximum,
            nbdevup = nbdevup,
            nbdevdn = nbdevdn,
            matype = matype,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def momentum_indicator_download(
        self,
        function: Literal['RSI', 'WILLR', 'MOM', 'ROC', 'ROCR', 'AROON', 'AROONOSC', 'TRIX'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt,
        series_type: Literal['close', 'open', 'high', 'low']) -> pd.Timestamp:
        """
        This method returns momentum indicators of prices.

        :param function: The function to retrieve different momentum indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. RSI: Relative strength index (RSI) values.
            2. WILLR: Williams' %R (WILLR) values.
            3. MOM: Momentum (MOM) values.
            4. ROC: Rate of change (ROC) values. 
            5. ROCR: Rate of change ratio (ROCR) values.
            6. AROON: Aroon (AROON) values.
            7. AROONOSC: Aroon oscillator (AROONOSC) values.
            8. TRIX: 1-day rate of change of a triple smooth exponential moving average (TRIX) values.
        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200).
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low. Not used when function is WILLR, AROON, AROONOSC.
        :return: Downloaded data frame.
        """
        if function in ['WILLR', 'AROON', 'AROONOSC']:
            series_type = None

        url = self._construct_url(
            function = function,
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

    def oscillator_indicator_download(
        self,
        function: Literal['APO', 'PPO', 'BOP', 'CCI', 'MFI', 'STOCH', 'STOCHF'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt,
        series_type: Literal['close', 'open', 'high', 'low'],
        fastperiod: Optional[PositiveInt]=12,
        slowperiod: Optional[PositiveInt]=26,
        matype: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]]=0,
        fastkperiod: Optional[PositiveInt]=5,
        slowkperiod: Optional[PositiveInt]=3,
        slowdperiod: Optional[PositiveInt]=3,
        slowkmatype: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]]=0,
        slowdmatype: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]]=0) -> pd.Timestamp:
        """
        This method returns oscillator indicators of prices.

        :param function: The function to retrieve different oscillator indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. APO: Absolute price oscillator (APO) values.
            2. PPO: Percentage price oscillator (PPO) values.
            3. CCI: Commodity channel index (CCI) values. 
            4. MFI: Money flow index (MFI) values.
            5. STOCH: Stochastic oscillator (STOCH) values.
            6. STOCHF: Stochastic fast (STOCHF) values. 
        :param symbol: The name of the token of your choice.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200).
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low. Not used when function is CCI, MFI.
        :param fastperiod: Positive integers are accepted. By default, fastperiod=12. Only used when function is APO and PPO.
        :param slowperiod: Positive integers are accepted. By default, slowperiod=26. Only used when function is APO and PPO.
        :param matype: Moving average type. By default, matype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). Only used when function is APO and PPO.
        :param fastkperiod: The time period of the fastk moving average. Positive integers are accepted. By default, fastkperiod=5. Only used when function is STOCH and STOCHF.
        :param slowkperiod: The time period of the slowk moving average. Positive integers are accepted. By default, slowkperiod=3. Only used when function is STOCH and STOCHF.
        :param slowdperiod: The time period of the slowd moving average. Positive integers are accepted. By default, slowdperiod=3. Only used when function is STOCH and STOCHF.
        :param slowkmatype: Moving average type for the slowk moving average. By default, slowkmatype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving
        Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). Only used when function is STOCH and STOCHF.
        :param slowdmatype: Moving average type for the slowd moving average. By default, slowkmatype=0. Integers 0 - 8 are accepted with the following mappings. 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). Only used when function is STOCH and STOCHF.
        :return: Downloaded data frame.
        """
        if function not in ['APO', 'PPO']:
            fastperiod = None
            slowperiod = None
            matype = None

        if function not in ['STOCH', 'STOCHF']:
            fastkperiod = None
            slowkperiod = None
            slowdperiod = None
            slowkmatype = None
            slowdmatype = None

        if function in ['CCI', 'MFI']:
            series_type = None

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            fastperiod = fastperiod,
            slowperiod = slowperiod,
            matype = matype,
            fastkperiod = fastkperiod,
            slowkperiod = slowkperiod,
            slowdperiod = slowdperiod,
            slowkmatype = slowkmatype,
            slowdmatype = slowdmatype,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def volume_indicator_download(
        self,
        function: Literal['VWAP', 'OBV', 'AD', 'ADOSC'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        fastperiod: Optional[PositiveInt]=3,
        slowperiod: Optional[PositiveInt]=10) -> pd.Timestamp:
        """
        This method returns volume indicators of prices.

        :param function: The function to retrieve different volume indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. VWAP: Volume Weighted Average Price (VWAP) for intraday time series.
            2. OBV: On balance volume (OBV) values.
            3. AD: Chaikin A/D line (AD) values.
            4. ADOSC: Chaikin A/D oscillator (ADOSC) values.
        :param symbol: The name of the token of your choice. For example: symbol=IBM.
        :param interval: Time interval between two consecutive data points in the time series. In keeping with mainstream investment literatures on VWAP, the following intraday intervals are supported: 1min, 5min, 15min, 30min, 60min. For other indicators, the following values are supported: 1min,
            5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param fastperiod: The time period of the fast EMA. Positive integers are accepted. By default, fastperiod=3. Only used when function is ADOSC.
        :param slowperiod: The time period of the slow EMA. Positive integers are accepted. By default, slowperiod=10. Only used when function is ADOSC.
        :return: Downloaded data frame.
        """
        if function not in ['ADOSC']:
            fastperiod = None
            slowperiod = None

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            fastperiod = fastperiod,
            slowperiod = slowperiod,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def volatility_indicator_download(
        self,
        function: Literal['ATR', 'NATR'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt) -> pd.Timestamp:
        """
        This method returns volatility indicators of prices.

        :param function: The function to retrieve different volatility indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. ATR: Average true range (ATR) values. 
            2. NATR: Normalized Average True Range (NATR) values.
        :param symbol: The name of the token of your choice. For example: symbol=IBM.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200).
        :return: Downloaded data frame.
        """
        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result

    def cycle_indicator_download(
        self,
        function: Literal['CMO', 'ULTOSC', 'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR'],
        symbol: str,
        interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
        time_period: PositiveInt,
        series_type: Literal['close', 'open', 'high', 'low'],
        timeperiod1: Optional[PositiveInt]=7,
        timeperiod2: Optional[PositiveInt]=14,
        timeperiod3: Optional[PositiveInt]=28) -> pd.Timestamp:
        """
        This method returns cycle indicators of prices.

        :param function: The function to retrieve different cycle indicators. [Formula](https://github.com/carlonlv/ProjectBuffalo/wiki/Metrics)
            1. CMO: Chande momentum oscillator (CMO) values.
            2. ULTOSC: Ultimate oscillator (ULTOSC) values.
            3. HT_TRENDLINE: Hilbert transform, instantaneous trendline (HT_TRENDLINE) values.
            4. HT_SINE: Hilbert transform, sine wave (HT_SINE) values.
            5. HT_TRENDMODE: Hilbert transform, trend vs cycle mode (HT_TRENDMODE) values.
            6. HT_DCPERIOD: Hilbert transform, dominant cycle period (HT_DCPERIOD) values.
            7. HT_DCPHASE: Hilbert transform, dominant cycle phase (HT_DCPHASE) values.
            8. HT_PHASOR: Hilbert transform, phasor components (HT_PHASOR) values.
        :param symbol: The name of the token of your choice. For example: symbol=IBM.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200). Only used when function is CMO.
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low. Only used when function is CMO.
        :param timeperiod1: The first time period for the indicator. Positive integers are accepted. By default, timeperiod1=7. Only used when function is ULTOSC.
        :param timeperiod2: The second time period for the indicator. Positive integers are accepted. By default, timeperiod2=14. Only used when function is ULTOSC.
        :param timeperiod3: The third time period for the indicator. Positive integers are accepted. By default, timeperiod3=28. Only used when function is ULTOSC.
        :return: Downloaded data frame.
        """
        if series_type not in ['ULTOSC']:
            timeperiod1 = None
            timeperiod2 = None
            timeperiod3 = None

        if series_type not in ['CMO']:
            series_type = None
            time_period = None

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            timeperiod1 = timeperiod1,
            timeperiod2 = timeperiod2,
            timeperiod3 = timeperiod3,
            datatype = 'csv',
            apikey = self.api_key)

        result = pd.read_csv(url)
        if len(result.index) == 0:
            warnings.warn(f'Reading from {url} results in 0 rows.')

        return result
