"""
Common helper functions that can be useful to all modules.
"""

import inspect
import os


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
    :return: Concatenated string using the separator.
    """
    return sep.join(map(str, lst))

def create_parent_directory(filepath):
    """
    Create parent directories following the filepath.
    :param filepath: Filepath.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
