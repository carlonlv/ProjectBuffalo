"""
Common helper functions that can be useful to all modules.
"""

import inspect
import os
import warnings

import numpy as np
import pandas as pd


def do_call(func, *args, **kwargs):
    """
    Call function, ignore nonexited arguments

    :param func: Function to be executed.
    :param *args: Positional arguments to be passed into func.
    :param **kwargs: Additional keyword arguments to be passed into func.
    :return: Returned results from func.
    """
    sig = inspect.signature(func)
    filtered_dict = {filter_item[0] : filter_item[1] for filter_item in kwargs.items() if filter_item[0] in sig.parameters.keys()}
    return func(*args, **filtered_dict)

def concat_list(lst, sep=","):
    """
    Concat all elements in a given list.

    :param lst: A list of objects that can be converted to strings.
    :param sep: The symbol used to separate strings.
    :return: Concatenated string.
    """
    return sep.join(map(str, lst))

def concat_dict(dct, kv_sep=":", sep=","):
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

def create_parent_directory(filepath):
    """
    Create parent directories following the filepath.
    :param filepath: Filepath.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def expand_grid(**kwargs):
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

def find_nearest_in_list(item, lst, round_up=False, round_down=False):
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
