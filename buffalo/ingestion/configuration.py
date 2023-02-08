"""
This is the moduel containing configurations for data ingestion.
"""

import inspect
import os
import warnings
from typing import Optional

import yaml

from . import enum


class Configuration:
    """
    This is the class that user configures the settings to download data.
    """

    api_keys={enum.API.ADVANTAGE: r'2rrNROO0beX90lPH7ixQOp0mT_9SwF0d'}
    storage_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cached_data")
    configuration_fp=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".config.yaml")

    watched_tickers = ["IBM", "APPL", "MSFT"]

    @staticmethod
    def update_setting(**setting_dict):
        """
        Update configuration.

        :param **setting_dict: keywords and values indicating which option to be set.
        :param api_keys: Dictionary
        """
        for item in setting_dict.items():
            if item[0] in inspect.getmembers(Configuration):
                Configuration.__setattr__(item[0], item[1])
            else:
                warnings.warn(f'Setting {item[0]} not found in configuration')

    @staticmethod
    def load_configuration(config_fp: Optional[str]=None):
        """
        Load the configurations from YAML file.

        :param config_fp: Alternative filepath other than what's stored in Configuration.
        """
        if config_fp is not None:
            Configuration.update_setting(configuration_fp=config_fp)
        from_path = Configuration.configuration_fp
        with open(from_path, "w", encoding = 'utf-8') as file:
            options = yaml.load(file, Loader=yaml.FullLoader)
        Configuration.update_setting(**options)

    @staticmethod
    def write_configuration(config_fp: Optional[str]=None):
        """
        Write the current configurations to YAML file.

        :param config_fp: Alternative filepath other than what's stored in Configuration.
        """
        if config_fp is not None:
            Configuration.update_setting(configuration_fp=config_fp)
        to_path = Configuration.configuration_fp
        with open(to_path, "w", encoding = 'utf-8') as file:
            yaml.dump(Configuration.__dict__, file)
