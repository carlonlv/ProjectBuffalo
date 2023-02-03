import inspect
import warnings

from Utility.errors import *

class Configuration:

    api_choices=['alphavantage']

    api_url_mapping = {
        'alphavantage': r"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey="
    }

    api_choice=api_choices[0]

    api_key=r'TBUGEDGGVAK59VD2'
    
    storage_folder="__cacheddata__"

    def update_setting(self, **setting_dict):
        for item in setting_dict:
            if item in inspect.getmembers(Configuration):
                self.__setattr__(item, setting_dict[item])
            else:
                warnings.warn('Setting {sett} not found in configuration'.format(sett = item))