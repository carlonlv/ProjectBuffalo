"""
This module 
"""

from . import enum
import jinja2
import typing
import re

class IngestionMethod:
    """
    This class stores the information of ingestion methods and generates ingestion methods.
    """

    def __init__(
            self,
            api: enum.API,
            template: jinja2.Template) -> None:
        """
        TBD
        """
        self.api = api
        self.template = template
        self.methods_args = {}
        self.rendered_methods = {}

    def register_method(self, data_type: enum.DataType, ingestion_type: enum.IngestionType, name: str, args: typing.Dict[str, typing.Type]):
        """
        TBD
        """
        if (data_type, ingestion_type) not in self.methods_args:
            self.methods_args[(data_type, ingestion_type)] = {}
        
        self.methods_args[(data_type, ingestion_type)][name] = args

    def function_generator(self, data_type: enum.DataType, ingestion_type: enum.IngestionType, name: str):
        """
        """
        args = self.methods_args[(data_type, ingestion_type)][name]
        for k, v in args.items():
            str_v = re.sub(str(v), )
        self.template.render(var_dict=self.methods_args[(data_type, ingestion_type)][name])
