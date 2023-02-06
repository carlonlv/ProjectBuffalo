"""
asd
"""

import inspect
import warnings

class Configuration:
    """
    This is the class that user configures the settings to download data.
    """

    api_key=r'2rrNROO0beX90lPH7ixQOp0mT_9SwF0d'
    storage_folder=r'../__cached_data'

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
