"""
This module contains all the enumerator types that are useful in data ingestion.
"""

import enum


class API(enum.Enum):
    """ Supported API enums for data ingestion.
    """
    ADVANTAGE = enum.auto()


class DataType(enum.Enum):
    """ Suppored DataType for data ingestion.

    Details:
        1. STOCK: stock time series
        2, FOREX: foreign exchange prices
        3. CRYPTO: crypto exchange prices
        4. COMMODITY: global commodity prices
        5. ECON: economic indicator
        6. COMPANY: company information
        7. TREND_INDICATOR: trend indicators
        8. CYCLE_INDICATOR: cycle indicators
        9. VOLATILITY_INDICATOR: volatility indicators
        10. MOMENTUM_INDICATOR: momentum indicators
        11. OSCILLATOR_INDICATOR: oscillator indicators
        12. VOLUME_INDICATOR: volume indicators
        13. STOCK_LISTING: stock listing
        14. MARKET_NEWS: market news
        15. IPO_CALENDAR: IPO calendar
    """
    STOCK = enum.auto()
    FOREX = enum.auto()
    CRYPTO = enum.auto()
    COMMODITY = enum.auto()
    ECON = enum.auto()
    COMPANY = enum.auto()
    TREND_INDICATOR = enum.auto()
    CYCLE_INDICATOR = enum.auto()
    VOLATILITY_INDICATOR = enum.auto()
    MOMENTUM_INDICATOR = enum.auto()
    OSCILLATOR_INDICATOR = enum.auto()
    VOLUME_INDICATOR = enum.auto()
    STOCK_LISTING = enum.auto()
    MARKET_NEWS = enum.auto()
    IPO_CALENDAR = enum.auto()

class IngestionType(enum.Enum):
    """ Suppored data ingestion type.
    """
    STREAM = enum.auto()
    REST = enum.auto()
