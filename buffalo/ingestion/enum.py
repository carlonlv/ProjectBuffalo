"""
This module contains all the enumerator types that are useful in data ingestion.
"""

import enum


class Storage(enum.Enum):
    """ File types enum for storing data.
    """
    SQLITE = "sqlite"
    CSV = "csv"
    PICKLE = "pkl"


class Frquency(enum.Enum):
    """ Frequency of Data
    """
    ONE_MIN = "1m"
    FIVE_MIN = "5m"


class API(enum.Enum):
    """ Supported API enums for data ingestion.
    """
    ADVANTAGE = enum.auto()


class DataType(enum.Enum):
    """ Suppored DataType for data ingestion.
    """
    STOCK = enum.auto()
    FOREX = enum.auto()
    CRYPTO = enum.auto()
    OPTION = enum.auto()
    ECON = enum.auto()
    INDICATOR = enum.auto()
    COMPARY = enum.auto()

class IngestionType(enum.Enum):
    """ Suppored data ingestion type.
    """
    STREAM = enum.auto()
    REST = enum.auto()
