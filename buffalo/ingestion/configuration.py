"""
Configuration options for data ingestions.
"""

import inspect
import warnings

from . import enum


class Configuration:
    """
    This is the class that user configures the settings to download data.
    """
    api_keys={enum.API.ADVANTAGE: r'2rrNROO0beX90lPH7ixQOp0mT_9SwF0d'}

    @staticmethod
    def update_api_keys(api: enum.API, api_key: str):
        """
        Update or add api keys.

        :param api: The name of api.
        :param api_key: The api key to be stored.
        """
        if api in Configuration.api_keys:
            warnings.warn(f'Updating {api} key to {api_key}')
        Configuration.api_keys[api]= api_key

    @staticmethod
    def update_setting(**setting_dict):
        """
        Update the Configurations.
        """
        for item in setting_dict.items():
            if item[0] in inspect.getmembers(Configuration):
                Configuration.__setattr__(item[0], item[1])
            else:
                warnings.warn(f'Setting {item[0]} not found in configuration')

    @staticmethod
    def print():
        """
        Print the current setting.
        """
        print({
            "api_keys": Configuration.api_keys
        })
