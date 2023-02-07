"""
Enumerator class used in metrics are defined here.
"""

import enum


class MetricTimeSeries(enum.Enum):
    """
    Suppored time series metrics of time series data.
    """
    SMA = enum.auto()
    EMA = enum.auto()
    RSI = enum.auto()
