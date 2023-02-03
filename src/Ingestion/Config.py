import inspect
import warnings

from Utility.errors import *
from APITypes import *

class Configuration:

    api_choice=API(1)

    api_key=r'2rrNROO0beX90lPH7ixQOp0mT_9SwF0d'
    
    storage_folder="__cacheddata__"

    @staticmethod
    def update_setting(self, **setting_dict):
        for item in setting_dict:
            if item in inspect.getmembers(Configuration):
                self.__setattr__(item, setting_dict[item])
            else:
                warnings.warn('Setting {sett} not found in configuration'.format(sett = item))