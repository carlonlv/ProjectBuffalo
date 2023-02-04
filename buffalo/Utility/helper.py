import inspect
import os

def do_call(func, *args, **kwargs):
    """
    Call function, ignore nonexited arguments

    :param func: Function to be executed.
    :args 
    """
    sig = inspect.signature(func)
    filtered_dict = {filter_key : kwargs[filter_key] for filter_key in kwargs.keys() if filter_key in sig.parameters.keys()}
    return func(*args, **filtered_dict)

def print_list(lst, sep=","):
    return sep.join(map(str, lst))

def create_parent_directory(fp):
    directory = os.path.dirname(fp)
    if not os.path.exists(directory):
        os.makedirs(directory)

def unfold_object(obj):
    if "__dict__" not in dir(obj):
        return obj
    else:
        atts = getattr(obj, "__dict__")
        return {x:unfold_object(atts[x]) for x in atts}
