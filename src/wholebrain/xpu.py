""" GPU-agnostic helper fuctions go here.
"""

import inspect, functools, logging
import numpy as np

try:
    import cupy as cp
    import cupy as xp
    using_cupy = True
    from rmm.allocators.cupy import rmm_cupy_allocator
    cp.cuda.set_allocator(rmm_cupy_allocator)
    from cupyx.scipy import ndimage
except ImportError as e:
    logging.warning(e)
    using_cupy = False
    import numpy as xp
    from scipy import ndimage


def iscupy(x):
    """ Check if input is cupy ndarray

    Args:
        x (ndarray): input array
    """
    return x.__class__.__module__  in ['cupy' , 'cupy._core.core']


def get_array_module(*args):
    """ Returns the array module for arguments. If any argument is cupy, return cupy

    Args:
        *args (any): inputs
    """
    for arg in args:
        if arg.__class__.__module__ == 'cupy':
            return cp
    return np


def to_numpy(x):
    """ Recursively convert all cupy arrays in input to numpy arrays

    Args:
        x (any): input
    """
    if x.__class__.__module__ == 'cupy':
        out = x.get()
        if np.isscalar(out):
            out = np.asscalar(out)
        return out
    elif type(x) is list:
        return [to_numpy(i) for i in x]
    elif type(x) is tuple:
        return tuple(to_numpy(list(x)))
    elif type(x) is dict:
        return {k: to_numpy(v) for k, v in x.items()}
    else:
        return x


def cupy_wrapper(args_to_convert=None, dtype=None):
    """ Returns decorator to convert numpy inputs to cupy arrays.
    
    Args: 
        args_to_convert (list of str): list of arguments (by signature name) to convert to cupy arrays. Default: all
        dtype (dtype): dtype to convert to. Default: None (keep input dtype)
    
    Returns:
        (callable): returns the original functino with additional callable .gpu attribute. It has the same signature 
        as the original function, but internally operatures on cupy arrays. If all selected args are cupy arrays, 
        outputs will be cupy. If any of the selected input args are numpy, the output will be converted to numpy.
    """

    def decorator(func):
        if not using_cupy:
            return func

        args_in_signature = inspect.getfullargspec(func).args

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            wasnumpy = False
            for i in range(len(args)):
                if (args_to_convert is None) or (args_in_signature[i] in args_to_convert):
                    wasnumpy = wasnumpy or (args[i].__class__.__module__ == 'numpy')
                    args[i] = cp.array(args[i], copy=False, dtype=dtype)
            for k, v in kwargs.items():
                if (args_to_convert is None) or (k in args_to_convert):
                    wasnumpy = wasnumpy or (v.__class__.__module__ == 'numpy')
                    kwargs[k] = cp.array(v, copy=False, dtype=dtype)
            out = func(*args, **kwargs)
            if wasnumpy:
                out = to_numpy(out)
            return out

        func.gpu = wrapper
        return func

    return decorator