import numpy as np

from . import xpu
from .xpu import cupy_wrapper, iscupy


@cupy_wrapper(dtype='float32')
def cov_matrix(arr1, arr2=None, center=True):
    """ Calculate covariance matrix. Time along the first axis

    Args:
        arr1 (array_like): input
        arr2 (array_like): optional input
        center (bool): whether to center the data. If not, return the Gram matrix.

    Returns:
        (array_like): covariance matrix
    """
    if center:
        arr1 = arr1 - arr1.mean(0, keepdims=True)
    if arr2 is None:
        arr2 = arr1
    elif center:
        arr2 = arr2 - arr2.mean(0, keepdims=True)
    out = arr1.T @ arr2
    return out


@cupy_wrapper(['a', 'b'], dtype='float32')
def pearsonr(a, b, ax=0, keepdims=True):
    """ Compute the Pearson's correlation between two signals

    Args:
        a (array_like): ND input a
        b (array_like): ND input b
        ax (int): axis along which to to calculate Pearson's r

    Return:
        r (array_like): ND output
    """
    ac = a - a.mean(axis=ax, keepdims=keepdims, dtype='float32')
    bc = b - b.mean(axis=ax, keepdims=keepdims, dtype='float32')
    a_std = np.sqrt((ac**2).mean(axis=ax, keepdims=keepdims, dtype='float32'))
    b_std = np.sqrt((bc**2).mean(axis=ax, keepdims=keepdims, dtype='float32'))
    r = (ac * bc).mean(axis=ax, keepdims=keepdims) / (a_std * b_std)
    return r




@cupy_wrapper(['a', 'b'], dtype='float32')
def rss(x,ax=0): return np.sqrt(np.sum((x-np.mean(x, axis=ax, keepdims=True))**2, axis=ax,keepdims=True))

@cupy_wrapper(['a', 'b'], dtype='float32')
def corrx(a,b): return ((a-np.mean(a, 0, keepdims=True)) / rss(a, 0)).T @ ((b-np.mean(b, 0, keepdims=True)) / rss(b, 0))


