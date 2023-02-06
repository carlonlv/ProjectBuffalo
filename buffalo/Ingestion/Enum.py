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


class API(enum.Enum):
    """ Supported API enums for data ingestion.
    """
    POLYGON = enum.auto()
