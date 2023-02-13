"""
Configuration options for data ingestions.
"""

import inspect
import warnings

from utility import concat_dict

from . import enum


class Configuration:
    """
    This is the class that user configures the settings to download data.
    """
    storage_folder=r'../__cached_data'
    api_keys={enum.API.ADVANTAGE: r'2rrNROO0beX90lPH7ixQOp0mT_9SwF0d'}
    watched_tickers=['IBM', 'MSFT']

    @staticmethod
    def update_setting(**setting_dict):
        """
        Todo
        """
        for item in setting_dict.items():
            if item[0] in inspect.getmembers(Configuration):
                Configuration.__setattr__(item[0], item[1])
            else:
                warnings.warn(f'Setting {item[0]} not found in configuration')

    def __str__(self) -> str:
        return concat_dict(self.__dict__)
