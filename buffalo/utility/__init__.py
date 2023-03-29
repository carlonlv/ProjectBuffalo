"""
Common helper functions that can be useful to all modules.
"""

import inspect
import os
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, NewType, Optional, Union

import numpy as np
import pandas as pd

NonnegativeInt = NewType('NonnegativeInteger', int)
NonnegativeFlt = NewType('NonnegativeFloat', float)

PositiveInt = NewType('PositiveInteger', int)
PositiveFlt = NewType('PositiveFloat', float)

Prob = NewType('Probability', float)

class Probability(float):
    """ Custom data type for probability to enforce type checking/
    """
    def __new__(cls, value):
        if value < 0 or value > 1:
            raise ValueError("Probability cannot be negative or greater than 1.")
        return super().__new__(cls, value)

class NonnegativeInteger(int):
    """ Custom data type of nonnegative integer to enforce type checking.
    """
    def __new__(cls, value):
        if value < 0:
            raise ValueError("NonnegativeInteger cannot be negative.")
        return super().__new__(cls, value)

class NonnegativeFloat(float):
    """ Custom data type of nonnegative float to enforce type checking.
    """
    def __new__(cls, value):
        if value < 0:
            raise ValueError("NonnegativeFloat cannot be negative.")
        return super().__new__(cls, value)

class PositiveInteger(int):
    """ Custom data type of positive integer to enforce type checking.
    """
    def __new__(cls, value):
        if value <= 0:
            raise ValueError("PositiveInteger cannot be negative or zero.")
        return super().__new__(cls, value)

class PositiveFloat(float):
    """ Custom data type of positive float to enforce type checking.
    """
    def __new__(cls, value):
        if value <= 0:
            raise ValueError("PositiveFloat cannot be negative or zero.")
        return super().__new__(cls, value)


def do_call(func: Callable, **kwargs):
    """
    Call function, ignore nonexited arguments

    :param func: Function to be executed.
    :param **kwargs: Additional keyword arguments to be passed into func.
    :return: Returned results from func.
    """
    sig = inspect.signature(func)
    filtered_dict = {filter_item[0] : filter_item[1] for filter_item in kwargs.items() if filter_item[0] in sig.parameters.keys()}
    return func(**filtered_dict)

def do_call_for_each_group(data: pd.DataFrame, func: Callable, grouping: Optional[Union[List, str]]=None, **kwargs):
    """
    Call function for each group.

    :param data: The input dataframe.
    :param func: Function to be executed.
    :param grouping: The grouping used before applying the function.
    :param **kwargs: Additional keyword arguments to be passed into func.
    """
    if len(grouping) == 0:
        return func(data, **kwargs)
    else:
        return data.groupby(grouping).apply(partial(do_call, func, **kwargs)).reset_index()

def concat_list(lst: List[Any], sep=","):
    """
    Concat all elements in a given list.

    :param lst: A list of objects that can be converted to strings.
    :param sep: The symbol used to separate strings.
    :return: Concatenated string.
    """
    return sep.join(map(str, lst))

def concat_dict(dct: Dict[Any,Any], kv_sep: str=":", sep: str=","):
    """
    Concat all items in a given dictionary.

    :param dct: A dictionary of keys and values can be converted to strings.
    :param kv_sep: The symbol used to separate keys and values.
    :param sep: The symbol used to separate items.
    :return: Concatenated string.
    """
    items = []
    for key, value in dct.items():
        items.append(f'{key}{kv_sep}{value}')
    return sep.join(items)

def create_parent_directory(filepath: str) -> None:
    """
    Create parent directories following the filepath.
    :param filepath: Filepath.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def expand_grid(**kwargs) -> pd.DataFrame:
    """
    Create an cross join product from given arguments.
    :param **kwargs: Arguments to create columns.
    :return: A data frame.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    columns = kwargs.keys()
    xii = kwargs.values()
    return pd.DataFrame({
        coln: arr.flatten() for coln, arr in zip(columns, np.meshgrid(*xii))
    })

def find_nearest_in_list(item: Any, lst: List[Any], round_up: bool=False, round_down: bool=False):
    """
    Find the closet approximation of item from a list. If round_up and
    round_down are both false, minimum absolute values are used.
    :param item: An element to be approximated.
    :param lst: A list of elements to be returned.
    :param round_up: Whether to only round up.
    :param round_down: Whether to only round down. only_greater takes precedence.
    """
    if round_up:
        n_lst = [x for x in lst if x >= item]
        if len(n_lst) == 0:
            warnings.warn(f'{item} is smaller (out of bound) than acceptable {concat_list(n_lst)}')
            return min(lst)
    elif round_down:
        n_lst = [x for x in lst if x <= item]
        if len(n_lst) == 0:
            warnings.warn(f'{item} is greater (out of bound) than acceptable {concat_list(n_lst)}')
            return max(lst)
    return min(lst, key=lambda y: abs(y - item))

def create_and_dodge_new_name(lst: List[str], prefix: str, suffix: str) -> str:
    """
    Find the used name {prefix}{digits}{suffix} in the list. Find the smallest number that is not used, and returns the name.
    Helpful when creating default naming. E.g. prefix = 'NewFile', suffix = '.txt'.

    :param lst: A list of strings to be searched.
    :param prefix: The prefix value.
    :param suffix: The suffix value.
    :return: A dodged new name.
    """
    patt = f'{prefix}\d*{suffix}'
    lst = [x for x in lst if re.match(x, patt)]
    lst = [int(x.replace(prefix, '').replace(suffix, '')) if x != prefix + suffix else 0 for x in lst]
    if len(lst) == 0:
        return prefix + suffix
    else:
        next_num = max(lst) + 1
        return prefix + str(next_num) + suffix

def split_string_to_words(concat_words: str) -> List[str]:
    """
    Split concatenated string of words into the original list of words.

    :param concat_words: For example, a string like VectorFormOfSS
    :return: Splitted list of words, ['Vector', 'Form', 'Of', 'SS'].
    """
    concat_words = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', concat_words)
    words = concat_words.split('_')
    words = [w if w.isupper() else re.sub(r'(?<!^)(?=[A-Z])', '_', w) for w in words]
    return words
