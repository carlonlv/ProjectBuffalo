import inspect

def doCall(func, *args, **kwargs):
    sig = inspect.signature(func)
    filtered_dict = {filter_key : kwargs[filter_key] for filter_key in kwargs.keys() if filter_key in sig.parameters.keys()}
    return func(*args, **filtered_dict)