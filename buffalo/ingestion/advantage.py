"""
This module provide api access to Alpha-advantage api.
"""

import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from pytz import timezone

from ..utility import (PositiveFlt, PositiveInt, concat_list,
                       split_string_to_words)
from . import configuration, enum


class AdvantageStockGrepper:
    """
    This class is used to retrieve stock data from Alpha Advantage endpoint.
    """
    url_base = r'https://www.alphavantage.co/query?'

    def __init__(self) -> None:
        self._api_key = configuration.Configuration.api_keys[enum.API.ADVANTAGE]
        self._ingestion_methods = {
            (enum.DataType.STOCK, enum.IngestionType.REST): self.stock_download,
            (enum.DataType.CRYPTO, enum.IngestionType.REST): self.crypto_exchange_download,
            (enum.DataType.FOREX, enum.IngestionType.REST): self.forex_download,
            (enum.DataType.COMPANY, enum.IngestionType.REST): self.company_info_download,
            (enum.DataType.ECON, enum.IngestionType.REST): self.econ_download,
            (enum.DataType.TREND_INDICATOR, enum.IngestionType.REST): self.trend_indicator_download,
            (enum.DataType.CYCLE_INDICATOR, enum.IngestionType.REST): self.cycle_indicator_download,
            (enum.DataType.MOMENTUM_INDICATOR, enum.IngestionType.REST): self.momentum_indicator_download,
            (enum.DataType.OSCILLATOR_INDICATOR, enum.IngestionType.REST): self.oscillator_indicator_download,
            (enum.DataType.VOLATILITY_INDICATOR, enum.IngestionType.REST): self.volatility_indicator_download,
            (enum.DataType.VOLUME_INDICATOR, enum.IngestionType.REST): self.volume_indicator_download,
            (enum.DataType.STOCK_LISTING, enum.IngestionType.REST): self.listing_info_download,
            (enum.DataType.MARKET_NEWS, enum.IngestionType.REST): self.market_news_sentiment_download,
            (enum.DataType.IPO_CALENDAR, enum.IngestionType.REST): self.ipo_calendar_download
        }

    @property
    def ingestion_methods(self) -> Dict[Tuple[enum.DataType, enum.IngestionType], Callable]:
        """
        Called by higher level classes, used to access different ingestion methods.
        :return: Returns a dictionary with keys being a tuple of DataType and IngestionType, and the values being a method.
        """
        return self._ingestion_methods

    @property
    def api_key(self) -> str:
        """
        Getter for api_key property.
        :return: Returns a string of api key.
        """
        return self._api_key

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

    def _check_schema(self, ingested_result: Union[pd.DataFrame, dict], url: str, schema: Optional[Union[List[str], Dict[str, np.dtype]]]):
        """
        Check the retuned dataframe on matching the schema and catch and raise error if detected.

        :param ingested_result: The downloaded dataframe or dictionary.
        :param url: The orginal url where data comes from.
        :param schema: The expected list of strings for the schema.
        """
        if schema is not None:
            ## Schema enforcing
            if isinstance(ingested_result, pd.DataFrame):
                actual_schema = ingested_result.columns
            elif isinstance(ingested_result, dict):
                actual_schema = ingested_result.keys()
            else:
                actual_schema = None

            if isinstance(schema, list):
                expected_schema = schema
            else:
                expected_schema = schema.keys()

            if actual_schema is not None and set(actual_schema) != set(expected_schema):
                if not set(actual_schema).issubset(set(schema)):
                    raise KeyError(f'Ingestion is not a subset of expected schema from {url}. (Expected: {concat_list(schema)} Actual: {concat_list(actual_schema)})')
                else:
                    warnings.warn(f'Ingestion deviates from expected schema from {url}. Missing keys: {concat_list(set(schema) - set(actual_schema))}.')

            ## Type enforcing
            if isinstance(schema, dict) and isinstance(ingested_result, pd.DataFrame):
                for key, value in ingested_result.dtypes.items():
                    if value != schema[key]:
                        #warnings.warn(f'Ingestion results from {url} has wrong type for {key} (Expected: {schema[key]} Actual {value}).')
                        ingested_result[key] = ingested_result[key].astype({key: schema[key]})

    def _check_args(self, ingested_result: Union[pd.DataFrame, dict], url: str):
        """
        Check the retuned dataframe and catch and raise error if detected.

        :param ingested_result: The downloaded dataframe or dictionary.
        :param url: The orginal url where data comes from.
        """
        if isinstance(ingested_result, pd.DataFrame):
            if len(ingested_result.index) == 0:
                warnings.warn(f'Ingestion results from {url} in 0 rows.')
                return

            if isinstance(ingested_result.iloc[0,0], str) and any([x in ingested_result.iloc[0,0]for x in ['Error Message', 'Note']]):
                msg = ingested_result.iloc[0,0]
            else:
                msg = None
        elif isinstance(ingested_result, dict) and any([x in ingested_result for x in ['Error Message', 'Note']]):
            if len(ingested_result) == 0:
                warnings.warn(f'Ingestion results from {url} in 0 rows.')
                return

            if 'Error Message' in ingested_result:
                msg = ingested_result['Error Message']
            elif 'Note' in ingested_result:
                msg = ingested_result['Note']
            elif 'Information' in ingested_result:
                msg = ingested_result['Information']
        else:
            msg = None
        if msg is not None:
            if 'Invalid API call' in msg or 'Invalid inputs' in msg:
                raise ValueError(f'Invalid arguments passed from {url}.')
            elif 'unlock all premium endpoints' in msg:
                raise PermissionError(f'Premium api key needed from {url}.')
            elif 'higher API call frequency' in msg:
                raise ConnectionRefusedError(f'API call frequency reached from {url}.')
            elif 'This API function' in msg:
                raise ValueError(f'Invalid function passed from {url}')

    def stock_download(
            self,
            symbol: str,
            interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
            year_slice: Optional[Literal['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6', 'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12', 'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6', 'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12']]='year1month1',
            adjusted: Optional[bool]=True) -> pd.DataFrame:
        """
        This method returns raw (as-traded) intraday/daily/weekly/monthly time series (date, open, high, low, close, volume) of the global equity specified, covering 20+ years of historical data. If you are also interested in split/dividend-adjusted historical data, please use the Daily Adjusted API, which covers adjusted close values and historical split and dividend events. Timestamps returned are in US/Eastern Time zone.

        :param symbol: The name of the equity of your choice. For example: symbol=IBM.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param year_slice: Two years of minute-level intraday data contains over 2 million data points, which can take up to Gigabytes of memory. To ensure optimal API response speed, the trailing 2 years of intraday data is evenly divided into 24 "slices" - year1month1, year1month2, year1month3, ..., year1month11, year1month12, year2month1, year2month2, year2month3, ..., year2month11, year2month12. Each slice is a 30-day window, with year1month1 being the most recent and year2month12 being the farthest from today. By default, slice=year1month1.
        :param adjusted: By default, adjusted=true and the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
        :return: Downloaded data frame.
        """
        if interval in ['daily', 'weekly', 'monthly']:
            function = 'TIME_SERIES_' + interval.upper()
            interval = None
            if adjusted:
                function += '_ADJUSTED'
                adjusted = None
                if function == 'TIME_SERIES_DAILY_ADJUSTED':
                    schema = ['timestamp', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
                    to_schema = {'open': np.dtype('float32'), 'high': np.dtype('float32'), 'low': np.dtype('float32'), 'close': np.dtype('float32'), 'adjusted_close': np.dtype('float32'), 'volume': np.dtype('int64'), 'dividend_amount': np.dtype('float32'), 'split_coefficient': np.dtype('float32')}
                else:
                    schema = ['timestamp', 'open', 'high', 'low', 'close', 'adjusted close', 'volume', 'dividend amount']
                    to_schema = {'open': np.dtype('float32'), 'high': np.dtype('float32'), 'low': np.dtype('float32'), 'close': np.dtype('float32'), 'adjusted_close': np.dtype('float32'), 'volume': np.dtype('int64'), 'dividend_amount': np.dtype('float32')}
            else:
                schema = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                to_schema = {'open': np.dtype('float32'), 'high': np.dtype('float32'), 'low': np.dtype('float32'), 'close': np.dtype('float32'), 'volume': np.dtype('int64')}
        else:
            function = 'TIME_SERIES_INTRADAY_EXTENDED'
            schema = ['time', 'open', 'high', 'low', 'close', 'volume']
            to_schema = {'open': np.dtype('float32'), 'high': np.dtype('float32'), 'low': np.dtype('float32'), 'close': np.dtype('float32'), 'volume': np.dtype('int64')}

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            slice = year_slice,
            outputsize = 'full',
            datatype = 'csv',
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        ## Postprocessing
        if 'time' in result.columns:
            result.index = pd.to_datetime(result['time'])
            result = result.drop(columns='time')
        if 'timestamp' in result.columns:
            result.index = pd.to_datetime(result['timestamp'])
            result = result.drop(columns='timestamp')
        result.index.name = None
        result.index = result.index.tz_localize('EST')
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        self._check_schema(result, url, to_schema)
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
            apikey = self._api_key)

        schema = ['Realtime Currency Exchange Rate.1. From_Currency Code',
                  'Realtime Currency Exchange Rate.2. From_Currency Name',
                  'Realtime Currency Exchange Rate.3. To_Currency Code',
                  'Realtime Currency Exchange Rate.4. To_Currency Name',
                  'Realtime Currency Exchange Rate.5. Exchange Rate',
                  'Realtime Currency Exchange Rate.6. Last Refreshed',
                  'Realtime Currency Exchange Rate.7. Time Zone',
                  'Realtime Currency Exchange Rate.8. Bid Price',
                  'Realtime Currency Exchange Rate.9. Ask Price']
        to_schema = {'from_currency_code': np.dtype('str'), 'from_currency_name': np.dtype('str'), 'to_currency_code': np.dtype('str'), 'to_currency_name': np.dtype('str'), 'exchange_rate': np.dtype('float32'), 'bid_price': np.dtype('float32'), 'ask_price': np.dtype('float32')}

        response = requests.get(url, timeout=10)
        data = response.json()
        self._check_args(data, url)
        self._check_schema(data, url, schema)
        if len(data) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result = pd.json_normalize(data)
        result.index = pd.to_datetime(result['Realtime Currency Exchange Rate.6. Last Refreshed'])
        result.index = result.index.tz_localize(timezone(result['Realtime Currency Exchange Rate.7. Time Zone'].iloc[0]))
        result = result.drop(columns=['Realtime Currency Exchange Rate.6. Last Refreshed', 'Realtime Currency Exchange Rate.7. Time Zone'])
        result.columns = result.columns.str.replace(r'Realtime Currency Exchange Rate\.\d+\.\s', '', regex=True)
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        result.columns = result.columns.str.lower()
        self._check_schema(result, url, to_schema)
        return result

    def forex_download(
            self,
            from_symbol: str,
            to_symbol: str,
            interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']) -> pd.DataFrame:
        """
        This API returns intraday/daily/weekly/monthly time series (timestamp, open, high, low, close) of the FX currency pair specified, updated realtime. Timestamps returned are in UTC.

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

        schema = ['timestamp', 'open', 'high', 'low', 'close']
        to_schema = {'open': np.dtype('float32'), 'high': np.dtype('float32'), 'low': np.dtype('float32'), 'close': np.dtype('float32')}

        url = self._construct_url(
            function = function,
            from_symbol = from_symbol,
            to_symbol = to_symbol,
            outputsize = outputsize,
            interval = interval,
            datatype = 'csv',
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.index = pd.to_datetime(result['timestamp'], utc=True)
        result = result.drop(columns='timestamp')
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        self._check_schema(result, url, to_schema)
        return result

    def crypto_exchange_download(
            self,
            digital_symbol: str,
            physical_symbol: str,
            interval: Literal['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']) -> pd.DataFrame:
        """
        This method returns intraday/daily/weekly/monthly time series (timestamp, open, high, low, close, volume) of the cryptocurrency specified, updated realtime. Timestamps returned are in UTC.

        :param digital_symbol: The digital/crypto currency of your choice. It can be any of the currencies in the digital currency list. For example: digital_symbol=ETH.
        :param physical_symbol: The exchange market of your choice. It can be any of the market in the market list. For example: physical_symbol=USD.
        :param interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :return: Downloaded data frame.
        """
        if interval in ['daily', 'weekly', 'monthly']:
            function = 'DIGITAL_CURRENCY' + interval.upper()
            interval = None
            outputsize = None
            schema = ['timestamp',
                      f'open ({physical_symbol})', f'high ({physical_symbol})', f'low ({physical_symbol})', f'close ({physical_symbol})',
                      'open (USD)', 'high (USD)', 'low (USD)', 'close (USD)',
                      'volume',
                      'market cap (USD)']
            to_schema = {'open': np.dtype('float32'), 'high': np.dtype('float32'), 'low': np.dtype('float32'), 'close': np.dtype('float32'), 'open_usd': np.dtype('float32'), 'high_usd': np.dtype('float32'), 'low_usd': np.dtype('float32'), 'close_usd': np.dtype('float32'), 'volume': np.dtype('float32'), 'market_cap_usd': np.dtype('float32')}
        else:
            function = 'CRYPTO_INTRADAY'
            outputsize = 'full'
            schema = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            to_schema = {'open': np.dtype('float32'), 'high': np.dtype('float32'), 'low': np.dtype('float32'), 'close': np.dtype('float32'), 'volume': np.dtype('int64')}

        url = self._construct_url(
            function = function,
            symbol = digital_symbol,
            market = physical_symbol,
            outputsize = outputsize,
            interval = interval,
            datatype = 'csv',
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.index = pd.to_datetime(result['timestamp'], utc=True)
        result = result.drop(columns='timestamp')
        result.columns = result.columns.str.replace(f' ({physical_symbol})', '', regex=False)
        result.columns = result.columns.str.replace(' (USD)', '_usd', regex=False)
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        self._check_schema(result, url, to_schema)
        return result

    def commodity_price_download(
            self,
            commodity: Literal['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES'],
            interval:Optional[Literal['daily', 'weekly', 'monthly', 'quarterly', 'annual']]='monthly') -> pd.Timestamp:
        """
        This method provides historical data for major commodities such as crude oil, natural gas, copper, wheat, etc., spanning across various temporal horizons (daily, weekly, monthly, quarterly, etc.)

        :param commodity: The commodity to be downloaded.
            1. WTI: West Texas Intermediate (WTI) crude oil prices.
            2. BRENT: Brent (Europe) crude oil prices.
            3. NATURAL_GAS: Henry Hub natural gas spot prices.
            4. COPPER: global price of copper.
            5. ALUMINUM: global price of aluminum.
            6. WHEAT: global price of wheat.
            7. CORN: global price of corn.
            8. COTTON: global price of cotton.
            9. SUGAR: global price of sugar.
            10. COFFEE: global price of coffee.
            11. ALL_COMMODITIES: global price index of all commodities.
        :param interval: By default, interval=monthly. For commodity equals WTI, BRENT, NATURAL_GAS, strings daily, weekly, monthly are accepted. For other commodities, monthly, quarterly, annual are accepted.
        :return: Downloaded data frame.
        """
        if commodity in ['WTI', 'BRENT', 'NATURAL_GAS']:
            acceptable_intervals = [None, 'daily', 'weekly', 'monthly']
            assert interval in acceptable_intervals, f'interval needs to be one of {concat_list(acceptable_intervals)}.'
        else:
            acceptable_intervals = [None, 'monthly', 'quarterly', 'annual']
            assert interval in acceptable_intervals, f'interval needs to be one of {concat_list(acceptable_intervals)}.'

        schema = ['name', 'interval', 'unit', 'date', 'value']
        to_schema = ['name', 'interval', 'unit', 'value']
        to_schema = {'name': np.dtype('str'), 'interval': np.dtype('str'), 'unit': np.dtype('str'), 'value': np.dtype('float32')}

        url = self._construct_url(
            function = commodity,
            interval = interval,
            datatype = 'json',
            apikey = self._api_key)

        response = requests.get(url, timeout=10)
        data = response.json()
        self._check_args(data, url)
        self._check_schema(data, url, schema)
        if len(data) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result = pd.json_normalize(data, record_path=['data'], meta=['name', 'interval', 'unit'])
        result.index = pd.to_datetime(result['date'])
        result = result.drop(columns='date')
        self._check_schema(result, url, to_schema)
        return result

    def econ_download(
            self,
            function: Literal['REAL_GDP', 'REAL_GDP_PER_CAPITA', 'TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 'CPI', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL'],
            interval: Optional[Literal['daily', 'weekly', 'monthly', 'quarterly', 'semiannual', 'annual']]='monthly',
            maturity: Optional[Literal['3month', '2year', '5year', '7year', '10year', '30year']]='30year') -> pd.Timestamp:
        """
        This method provides key US economic indicators frequently used for investment strategy formulation and application development.

        :param function: The economic indicators to be returned.
            1. REAL_GDP: annual and quarterly Real GDP of the United States.
            2. REAL_GDP_PER_CAPITA: quarterly Real GDP per Capita data of the United States.
            3. TREASURY_YIELD: daily, weekly, and monthly US treasury yield of a given maturity timeline.
            4. FEDERAL_FUNDS_RATE: daily, weekly, and monthly federal funds rate (interest rate) of the United States.
            5. CPI: monthly and semiannual consumer price index (CPI) of the United States. CPI is widely regarded as the barometer of inflation levels in the broader economy.
            6. INFLATION: annual inflation rates (consumer prices) of the United States.
            7. RETAIL_SALES: monthly Advance Retail Sales: Retail Trade data of the United States.
            8. DURABLES: monthly manufacturers' new orders of durable goods in the United States.
            9. UNEMPLOYMENT: monthly unemployment data of the United States. The unemployment rate represents the number of unemployed as a percentage of the labor force. Labor force data are restricted to people 16 years of age and older, who currently reside in 1 of the 50 states or the District of Columbia, who do not reside in institutions (e.g., penal and mental facilities, homes for the aged), and who are not on active duty in the Armed Forces (source).
            10. NONFARM_PAYROLL: monthly US All Employees: Total Nonfarm (commonly known as Total Nonfarm Payroll), a measure of the number of U.S. workers in the economy that excludes proprietors, private household employees, unpaid volunteers, farm employees, and the unincorporated self-employed.
        :param interval: One of 'daily', 'weekly', 'monthly', 'quarterly', 'semiannual', 'annual'. The acceptable value depends on the function provided.
        :param maturity: By default, maturity=10year. Strings 3month, 2year, 5year, 7year, 10year, and 30year are accepted. Only used when function is TREASURY_YIELD.
        :return: Downloaded data frame.
        """
        if function == 'REAL_GDP':
            acceptable_intervals = [None, 'quarterly', 'annual']
            assert interval in acceptable_intervals, f'interval needs to be one of {concat_list(acceptable_intervals)}.'
        elif function in ['TREASURY_YIELD', 'FEDERAL_FUNDS_RATE']:
            acceptable_intervals = [None, 'daily', 'weekly', 'monthly']
            assert interval in acceptable_intervals, f'interval needs to be one of {concat_list(acceptable_intervals)}.'
        elif function in ['CPI']:
            acceptable_intervals = [None, 'monthly', 'semiannual']
            assert interval in acceptable_intervals, f'interval needs to be one of {concat_list(acceptable_intervals)}.'
        else:
            interval = None

        if function != 'TREASURY_YIELD':
            maturity = None

        schema = ['name', 'interval', 'unit', 'data']
        to_schema = {'name': np.dtype('str'), 'interval': np.dtype('str'), 'unit': np.dtype('str'), 'value': np.dtype('float32')}

        url = self._construct_url(
            function = function,
            maturity = maturity,
            interval = interval,
            datatype = 'json',
            apikey = self._api_key)

        response = requests.get(url, timeout=10)
        data = response.json()
        self._check_args(data, url)
        self._check_schema(data, url, schema)
        if len(data) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result = pd.json_normalize(data, record_path=['data'], meta=['name', 'interval', 'unit'])
        result.index = pd.to_datetime(result['date'])
        result = result.drop(columns='date')
        result = result.replace('.', np.nan, regex=False)
        self._check_schema(result, url, to_schema)
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

        schema = ['currency code', 'currency name']
        result = pd.read_csv(url)
        if len(result) == 0:
            return pd.DataFrame(columns=schema.keys(), dtype=schema.values())
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        result.index = pd.Index([pd.Timestamp.now()])
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
        schema = ['items', 'sentiment_score_definition', 'relevance_score_definition', 'feed']
        to_schema = {'items': np.dtype('str'),
                     'sentiment_score_definition': np.dtype('str'),
                     'relevance_score_definition': np.dtype('str'),
                     'title': np.dtype('str'),
                     'url': np.dtype('str'),
                     'authors': np.dtype('str'),
                     'summary': np.dtype('str'),
                     'banner_image': np.dtype('str'),
                     'source': np.dtype('str'),
                     'category_within_source': np.dtype('str'),
                     'source_domain': np.dtype('str'),
                     'topic': np.dtype('str'),
                     'topic_relevance_score': np.dtype('float32'),
                     'overall_sentiment_score': np.dtype('float32'),
                     'overall_sentiment_label': np.dtype('str'),
                     'ticker': np.dtype('str'),
                     'ticker_relevance_score': np.dtype('float32'),
                     'ticker_sentiment_score': np.dtype('float32'),
                     'ticker_sentiment_label': np.dtype('str')}

        url = self._construct_url(
            function = function,
            tickers = tickers,
            topics = topics,
            time_from = time_from,
            time_to = time_to,
            sort = sort,
            limit = limit,
            apikey = self._api_key)

        response = requests.get(url, timeout=10)
        data = response.json()
        self._check_args(data, url)
        self._check_schema(data, url, schema)
        if len(data) == 0:
            return pd.DataFrame(columns=to_schema.keys())
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
        result.index = pd.to_datetime(result['time_published'], utc=True)
        result.set_index('time_published', inplace=True)
        result = result.merge(topic_lst).merge(ticker_lst)
        result = result.drop(columns=['index'])
        self._check_schema(result, url, to_schema)
        return result

    def company_info_download(
        self,
        function: Literal['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS', 'EARNINGS_CALENDAR'],
        symbol: Optional[str],
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
        :param symbol: The symbol of the token of your choice. For example: symbol=IBM. symbol is optional when function is EARNINGS_CALENDAR.
        :param horizon: By default, horizon=3month and the API will return a list of expected company earnings in the next 3 months. You may set horizon=6month or horizon=12month to query the earnings scheduled for the next 6 months or 12 months, respectively. Only used when function is EARNINGS_CALENDAR.
        :return: Downloaded data frame.
        """
        if function == 'EARNINGS_CALENDAR':
            acceptable_horizon = [None, '3month', '6month', '12month']
            assert horizon in acceptable_horizon, f'horizon needs to be one of {concat_list(acceptable_horizon)}.'

            schema = ['symbol', 'name', 'reportDate', 'fiscalDateEnding', 'estimate', 'currency']
            to_schema = {'symbol': np.dtype('str'), 'name': np.dtype('str'), 'fiscal_date_ending': np.dtype('datetime64[ns]'), 'estimate': np.dtype('float32'), 'currency': np.dtype('str')}

            url = self._construct_url(
                function = function,
                symbol = symbol,
                horizon = horizon,
                apikey = self._api_key)

            data = pd.read_csv(url)
        else:
            horizon = None
            if function == 'INCOME_STATEMENT':
                schema = ['symbol', 'annualReports', 'quarterlyReports']
                to_schema = {'reported_currency': np.dtype('str'),
                             'gross_profit': np.dtype('float32'),
                             'total_revenue': np.dtype('float32'),
                             'cost_of_revenue': np.dtype('float32'),
                             'costof_goods_and_services_sold': np.dtype('float32'),
                             'operating_income': np.dtype('float32'),
                             'selling_general_and_administrative': np.dtype('float32'),
                             'research_and_development': np.dtype('float32'),
                             'operating_expenses': np.dtype('float32'),
                             'investment_income_net': np.dtype('float32'),
                             'net_interest_income': np.dtype('float32'),
                             'interest_income': np.dtype('float32'),
                             'interest_expense': np.dtype('float32'),
                             'noninterest_income': np.dtype('float32'),
                             'other_nonoperating_income': np.dtype('float32'),
                             'depreciation': np.dtype('float32'),
                             'depreciation_and_amortization': np.dtype('float32'),
                             'income_before_tax': np.dtype('float32'),
                             'income_tax_expense': np.dtype('float32'),
                             'interest_and_debt_expense': np.dtype('float32'),
                             'net_income_from_continuing_operations': np.dtype('float32'),
                             'comprehensive_income_net_of_tax': np.dtype('float32'),
                             'ebit': np.dtype('float32'),
                             'ebitda': np.dtype('float32'),
                             'net_income': np.dtype('float32'),
                             'symbol': np.dtype('str'),
                             'freq': np.dtype('str')}
            elif function == 'BALANCE_SHEET':
                schema = ['symbol', 'annualReports', 'quarterlyReports']
                to_schema = {'reported_currency': np.dtype('str'),
                             'total_assets': np.dtype('float32'),
                             'total_current_assets': np.dtype('float32'),
                             'cash_and_cash_equivalents_at_carrying_value': np.dtype('float32'),
                             'cash_and_short_term_investments': np.dtype('float32'),
                             'inventory': np.dtype('float32'),
                             'current_net_receivables': np.dtype('float32'),
                             'total_noncurrent_assets': np.dtype('float32'),
                             'property_plant_equipment': np.dtype('float32'),
                             'accumulated_depreciation_amortization_ppe': np.dtype('float32'),
                             'intangible_assets': np.dtype('float32'),
                             'intangible_assets_excluding_goodwill': np.dtype('float32'),
                             'goodwill': np.dtype('float32'),
                             'investments': np.dtype('float32'),
                             'long_term_investments': np.dtype('float32'),
                             'short_term_investments': np.dtype('float32'),
                             'other_current_assets': np.dtype('float32'),
                             'other_noncurrent_assets': np.dtype('float32'),
                             'total_liabilities': np.dtype('float32'),
                             'total_current_liabilities': np.dtype('float32'),
                             'current_accounts_payable': np.dtype('float32'),
                             'deferred_revenue': np.dtype('float32'),
                             'current_debt': np.dtype('float32'),
                             'short_term_debt': np.dtype('float32'),
                             'total_noncurrent_liabilities': np.dtype('float32'),
                             'capital_lease_obligations': np.dtype('float32'),
                             'long_term_debt': np.dtype('float32'),
                             'current_long_term_debt': np.dtype('float32'),
                             'long_term_debt_noncurrent': np.dtype('float32'),
                             'short_long_term_debt_total': np.dtype('float32'),
                             'other_current_liabilities': np.dtype('float32'),
                             'other_noncurrent_liabilities': np.dtype('float32'),
                             'total_shareholder_equity': np.dtype('float32'),
                             'treasury_stock': np.dtype('float32'),
                             'retained_earnings': np.dtype('float32'),
                             'common_stock': np.dtype('float32'),
                             'common_stock_shares_outstanding': np.dtype('float32'),
                             'symbol': np.dtype('str'),
                             'freq': np.dtype('str')}
            elif function == 'CASH_FLOW':
                schema = ['symbol', 'annualReports', 'quarterlyReports']
                to_schema = {'reported_currency': np.dtype('str'),
                             'operating_cashflow': np.dtype('float32'),
                             'payments_for_operating_activities': np.dtype('float32'),
                             'proceeds_from_operating_activities': np.dtype('float32'),
                             'change_in_operating_liabilities': np.dtype('float32'),
                             'change_in_operating_assets': np.dtype('float32'),
                             'depreciation_depletion_and_amortization': np.dtype('float32'),
                             'capital_expenditures': np.dtype('float32'),
                             'change_in_receivables': np.dtype('float32'),
                             'change_in_inventory': np.dtype('float32'),
                             'profit_loss': np.dtype('float32'),
                             'cashflow_from_investment': np.dtype('float32'),
                             'cashflow_from_financing': np.dtype('float32'),
                             'proceeds_from_repayments_of_short_term_debt': np.dtype('float32'),
                             'payments_for_repurchase_of_common_stock': np.dtype('float32'),
                             'payments_for_repurchase_of_equity': np.dtype('float32'),
                             'payments_for_repurchase_of_preferred_stock': np.dtype('float32'),
                             'dividend_payout': np.dtype('float32'),
                             'dividend_payout_common_stock': np.dtype('float32'),
                             'dividend_payout_preferred_stock': np.dtype('float32'),
                             'proceeds_from_issuance_of_common_stock': np.dtype('float32'),
                             'proceeds_from_issuance_of_long_term_debt_and_capital_securities_net': np.dtype('float32'),
                             'proceeds_from_issuance_of_preferred_stock': np.dtype('float32'),
                             'proceeds_from_repurchase_of_equity': np.dtype('float32'),
                             'proceeds_from_sale_of_treasury_stock': np.dtype('float32'),
                             'change_in_cash_and_cash_equivalents': np.dtype('float32'),
                             'change_in_exchange_rate': np.dtype('float32'),
                             'net_income': np.dtype('float32'),
                             'symbol': np.dtype('str'),
                             'freq': np.dtype('str')}
            elif function == 'EARNINGS':
                schema = ['symbol', 'annualEarnings', 'quarterlyEarnings']
                to_schema = {'reported_eps': np.dtype('float32'),
                             'symbol': np.dtype('str'),
                             'freq': np.dtype('str'),
                             'reported_date': np.dtype('datetime64[ns]'),
                             'estimated_eps': np.dtype('float32'),
                             'surprise': np.dtype('float32'),
                             'surprise_percentage': np.dtype('float32')}
            else:
                schema = ['Symbol', 'AssetType', 'Name', 'Description', 'CIK', 'Exchange', 'Currency', 'Country', 'Sector', 'Industry', 'Address',
                          'FiscalYearEnd', 'LatestQuarter', 'MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio', 'BookValue', 'DividendPerShare', 'DividendYield',
                          'EPS', 'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM', 'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM', 'GrossProfitTTM', 'DilutedEPSTTM',
                          'QuarterlyEarningsGrowthYOY', 'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE', 'ForwardPE',
                          'PriceToSalesRatioTTM', 'PriceToBookRatio', 'EVToRevenue', 'EVToEBITDA', 'Beta', '52WeekHigh', '52WeekLow',
                          '50DayMovingAverage', '200DayMovingAverage', 'SharesOutstanding', 'DividendDate', 'ExDividendDate']
                to_schema = {'symbol': np.dtype('str'),
                             'asset_type': np.dtype('str'),
                             'name': np.dtype('str'),
                             'description': np.dtype('str'),
                             'cik': np.dtype('str'),
                             'exchange': np.dtype('str'),
                             'currency': np.dtype('str'),
                             'country': np.dtype('str'),
                             'sector': np.dtype('str'),
                             'industry': np.dtype('str'),
                             'address': np.dtype('str'),
                             'fiscal_year_end': np.dtype('str'),
                             'market_capitalization': np.dtype('float32'),
                             'ebitda': np.dtype('float32'),
                             'pe_ratio': np.dtype('float32'),
                             'peg_ratio': np.dtype('float32'),
                             'book_value': np.dtype('float32'),
                             'dividend_per_share': np.dtype('float32'),
                             'dividend_yield': np.dtype('float32'),
                             'eps': np.dtype('float32'),
                             'revenue_per_share_ttm': np.dtype('float32'),
                             'profit_margin': np.dtype('float32'),
                             'operating_margin_ttm': np.dtype('float32'),
                             'return_on_assets_ttm': np.dtype('float32'),
                             'return_on_equity_ttm': np.dtype('float32'),
                             'revenue_ttm': np.dtype('float32'),
                             'gross_profit_ttm': np.dtype('float32'),
                             'diluted_eps_ttm': np.dtype('float32'),
                             'quarterly_earnings_growth_yoy': np.dtype('float32'),
                             'quarterly_revenue_growth_yoy': np.dtype('float32'),
                             'analyst_target_price': np.dtype('float32'),
                             'trailing_pe': np.dtype('float32'),
                             'forward_pe': np.dtype('float32'),
                             'price_to_sales_ratio_ttm': np.dtype('float32'),
                             'price_to_book_ratio': np.dtype('float32'),
                             'ev_to_revenue': np.dtype('float32'),
                             'ev_to_ebitda': np.dtype('float32'),
                             'beta': np.dtype('float32'),
                             '52_week_high': np.dtype('float32'),
                             '52_week_low': np.dtype('float32'),
                             '50_day_moving_average': np.dtype('float32'),
                             '200_day_moving_average': np.dtype('float32'),
                             'shares_outstanding': np.dtype('float32'),
                             'dividend_date': np.dtype('datetime64[ns]'),
                             'ex_dividend_date': np.dtype('datetime64[ns]')}

            url = self._construct_url(
                function = function,
                symbol = symbol,
                apikey = self._api_key)

            response = requests.get(url, timeout=10)
            data = response.json()

        self._check_args(data, url)
        self._check_schema(data, url, schema)
        if len(data) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        if function in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']:
            annual_data = pd.json_normalize(data, record_path='annualReports', meta='symbol').assign(freq = 'annual')
            quarterly_data = pd.json_normalize(data, record_path='quarterlyReports', meta='symbol').assign(freq = 'quarterly')
            result = pd.concat([annual_data, quarterly_data], axis=0).reset_index(drop=True)
        elif function in ['OVERVIEW']:
            result = pd.json_normalize(data)
        elif function in ['EARNINGS']:
            annual_data = pd.json_normalize(data, record_path='annualEarnings', meta='symbol').assign(freq = 'annual')
            quarterly_data = pd.json_normalize(data, record_path='quarterlyEarnings', meta='symbol').assign(freq = 'quarterly')
            result = pd.concat([annual_data, quarterly_data], axis=0).reset_index(drop=True)
        else: ## EARNINGS_CALENDAR
            result = data
            result['fiscalDateEnding'] = pd.to_datetime(result['fiscalDateEnding'])

        if function == 'OVERVIEW':
            result['LatestQuarter'] = pd.to_datetime(result['LatestQuarter'])
            result.set_index('LatestQuarter', inplace=True)
            result['DividendDate'] = pd.to_datetime(result['DividendDate'])
            result['ExDividendDate'] = pd.to_datetime(result['ExDividendDate'])
        elif function in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'EARNINGS', 'CASH_FLOW']:
            result['fiscalDateEnding'] = pd.to_datetime(result['fiscalDateEnding'])
            result.set_index('fiscalDateEnding', inplace=True)
            if function == 'EARNINGS':
                result['reportedDate'] = pd.to_datetime(result['reportedDate'])
        else:
            result['reportDate'] = pd.to_datetime(result['reportDate'])
            result.set_index('reportDate', inplace=True)
        result.columns = ['_'.join(split_string_to_words(x)).lower() for x in result.columns]
        result = result.rename(columns={'diluted_epsttm': 'diluted_eps_ttm'}) ## special case that helper function cannot identify
        result.columns = result.columns.str.replace('non_', 'non', regex=False)
        result = result.replace(r'None', np.nan, regex=True)
        self._check_schema(result, url, to_schema)
        return result

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
            apikey = self._api_key)
        schema = ['symbol', 'name', 'exchange', 'assetType', 'ipoDate', 'delistingDate', 'status']
        to_schema = {'symbol': np.dtype('str'),
                     'name': np.dtype('str'),
                     'exchange': np.dtype('str'),
                     'asset_type': np.dtype('str'),
                     'ipo_date': np.dtype('datetime64[ns]'),
                     'delisting_date': np.dtype('datetime64[ns]'),
                     'status': np.dtype('str')}

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = ['_'.join(split_string_to_words(x)).lower() for x in result.columns]
        result.columns = result.columns.str.lower()
        if date is None:
            result.index = pd.Index([pd.Timestamp.now().date()] * len(result.index))
        else:
            result.index = pd.Index([pd.Timestamp(date)] * len(result.index))
        result['ipo_date'] = pd.to_datetime(result['ipo_date'])
        result['delisting_date'] = pd.to_datetime(result['delisting_date'])
        self._check_schema(result, url, to_schema)
        return result

    def ipo_calendar_download(self) -> pd.Timestamp:
        """
        This method returns a list of IPOs expected in the next 3 months.

        :return: Downloaded data frame.
        """
        function = 'IPO_CALENDAR'

        url = self._construct_url(function = function, apikey = self._api_key)
        schema = ['symbol', 'name', 'ipoDate', 'priceRangeLow', 'priceRangeHigh', 'currency', 'exchange']
        to_schema = {'symbol': np.dtype('str'), 'name': np.dtype('str'), 'ipo_date': np.dtype('datetime64[ns]'), 'price_range_low': np.dtype('float64'), 'price_range_high': np.dtype('float64'), 'currency': np.dtype('str'), 'exchange': np.dtype('str')}

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = ['_'.join(split_string_to_words(x)).lower() for x in result.columns]
        result.columns = result.columns.str.lower()
        result['ipo_date'] = pd.to_datetime(result['ipo_date'])
        self._check_schema(result, url, to_schema)
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

        if function in ['ADX', 'ADXR', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'ATR', 'NATR', 'MIDPRICE']:
            series_type = None

        if function in ['MAMA']:
            time_period = None
            schema = ['time', 'FAMA', 'MAMA']
            to_schema = {'fama': np.dtype('float32'), 'mama': np.dtype('float32')}
        else:
            schema = ['time', function]
            to_schema = {function.lower(): np.dtype('float32')}

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
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = result.columns.str.lower()
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        result.index = pd.to_datetime(result['time'])
        result = result.drop(columns='time')
        self._check_schema(result, url, to_schema)
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

        schema = ['time', function]

        to_schema = {function.lower(): np.dtype('float32')}

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            series_type = series_type,
            datatype = 'csv',
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = result.columns.str.lower()
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        result.index = pd.to_datetime(result['time'])
        result = result.drop(columns='time')
        self._check_schema(result, url, to_schema)
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

        schema = ['time', function]
        to_schema = {function.lower(): np.dtype('float32')}

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
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = result.columns.str.lower()
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        result.index = pd.to_datetime(result['time'])
        result = result.drop(columns='time')
        self._check_schema(result, url, to_schema)
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
        :param interval: Time interval between two consecutive data points in the time series. In keeping with mainstream investment literatures on VWAP, the following intraday intervals are supported: 1min, 5min, 15min, 30min, 60min. For other indicators, the following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly.
        :param fastperiod: The time period of the fast EMA. Positive integers are accepted. By default, fastperiod=3. Only used when function is ADOSC.
        :param slowperiod: The time period of the slow EMA. Positive integers are accepted. By default, slowperiod=10. Only used when function is ADOSC.
        :return: Downloaded data frame.
        """
        if function not in ['ADOSC']:
            fastperiod = None
            slowperiod = None

        schema = ['time', function]
        to_schema = {function.lower(): np.dtype('float32')}

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            fastperiod = fastperiod,
            slowperiod = slowperiod,
            datatype = 'csv',
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = result.columns.str.lower()
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        result.index = pd.to_datetime(result['time'])
        result = result.drop(columns='time')
        self._check_schema(result, url, to_schema)
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
        schema = ['time', function]
        to_schema = {function.lower(): np.dtype('float32')}

        url = self._construct_url(
            function = function,
            symbol = symbol,
            interval = interval,
            time_period = time_period,
            datatype = 'csv',
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = result.columns.str.lower()
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        result.index = pd.to_datetime(result['time'])
        result = result.drop(columns='time')
        self._check_schema(result, url, to_schema)
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
        :param series_type: The desired price type in the time series. Four types are supported: close, open, high, low. Used when function is CMO, HT_TRENDLINE, HT_SINE, HT_TRENDMODE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR.
        :param timeperiod1: The first time period for the indicator. Positive integers are accepted. By default, timeperiod1=7. Only used when function is ULTOSC.
        :param timeperiod2: The second time period for the indicator. Positive integers are accepted. By default, timeperiod2=14. Only used when function is ULTOSC.
        :param timeperiod3: The third time period for the indicator. Positive integers are accepted. By default, timeperiod3=28. Only used when function is ULTOSC.
        :return: Downloaded data frame.
        """
        if function not in ['ULTOSC']:
            timeperiod1 = None
            timeperiod2 = None
            timeperiod3 = None
        else:
            series_type = None
            time_period = None

        if function == 'HT_SINE':
            schema = ['time', 'LEAD SINE', 'SINE']
            to_schema = {'lead_sine': np.dtype('float32'), 'sine': np.dtype('float32')}
        else:
            schema = ['time', function]
            to_schema = {function.lower(): np.dtype('float32')}

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
            apikey = self._api_key)

        result = pd.read_csv(url)
        self._check_args(result, url)
        self._check_schema(result, url, schema)
        if len(result.index) == 0:
            return pd.DataFrame(columns=to_schema.keys())
        result.columns = result.columns.str.lower()
        result.columns = result.columns.str.replace(r'\s', '_', regex=True)
        result.index = pd.to_datetime(result['time'])
        result = result.drop(columns='time')
        self._check_schema(result, url, to_schema)
        return result
