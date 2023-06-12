import numpy as np
from tqdm.auto import tqdm

from . import xpu, regression
from .xpu import cupy_wrapper, iscupy, get_array_module


@cupy_wrapper(['arr'])
def getdff(arr, medianfilt_size=7, minfilt_size=61, gaussfilt_sigma=61., onlydf=False, epsilon=1.):
    """ Calculate deltaF / F0. This function calculates the dF/F traces after estimating a baseline. The baseline is estimated by a
        minimum filter, after de-noising the data via median filter. It is then smoothed by a gaussian filter.

    Args:
        arr (array_like): input array, time along first dimension
        medianfilt_size (int): size of median cleanup filter
        minfilt_size (int): size of minimum filter to get baseline
        gaussfilt_sigma (float): sigma to smooth baseline
        onlydf (bool): if True, outputs only dF, if False (default) outputs dF/F0
        epsilon (float): clipping value for small F0 (we don't want to divide by F0=0)

    Returns:
        (array_like): dF/F0 trace (or just dF - see onlydf)
    """
    if iscupy(arr):
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage
    tempF0 = ndimage.median_filter(arr.reshape(arr.shape[0], -1), (medianfilt_size, 1))
    tempF0 = ndimage.minimum_filter(tempF0, (minfilt_size, 1))
    tempF0 = ndimage.gaussian_filter(tempF0.astype('float32'), (gaussfilt_sigma, 0.0), truncate=3)
    #some combinations of truncation and sigma lead to a floating point round down
    tempF0 = tempF0.reshape(*arr.shape)
    temp_out = arr - tempF0  #this is dF
    if not onlydf:
        tempF0 = np.maximum(tempF0, epsilon)
        temp_out /= tempF0  #this is dF/F0
    return temp_out


@cupy_wrapper(['data', 'nan_mask'])
def stimulus_triggred_dff(data, stimtimes, F0_range=np.r_[-5:0], F1_range=np.r_[1:4], nan_mask=None):
    """ Calculate stimulus-striggered deltaF / F0

    Args:
        data (array_like): 2D input data, time along the first dimension
        stimtimes (array_like): 1D array of stimulus onset times (one per stimulus)
        F0_range (array_like): 1D array of F0 selection indices, relative to stimulus onset time
        F1_range (array_like): 1D array of F1 selection indices, relative to stimulus onset time
        nan_mask (array_like): optional 1D mask array, ones or nans

    Returns:
        dFF (array_like): stimulus-striggered deltaF / F0
    """
    F0_selector = stimtimes[:, None] + F0_range
    F1_selector = stimtimes[:, None] + F1_range
    d0 = data[F0_selector].astype('float32')
    d1 = data[F1_selector].astype('float32')
    if nan_mask is not None:
        d0 *= nan_mask[F0_selector][:, :, None, None, None]
        d1 *= nan_mask[F1_selector][:, :, None, None, None]
    F0 = np.nanmean(d0, axis=1, dtype='float32') + np.float32(1)
    F1 = np.nanmean(d1, axis=1, dtype='float32') + np.float32(1)
    dF = F1 - F0
    #confidence = dF / np.sqrt(np.nanstd(d0, axis=1,dtype='float32')**2 + np.nanstd(d1, axis=1,dtype='float32')**2)
    dFF = dF / F0
    return dFF  #, confidence


@cupy_wrapper(['traces'], dtype='float32')
def simple_sparse_deconvolve(traces, alpha=0.001, tau=5, fit_poly=-1, force_positive=False, n_iter=500):
    """ Sparse deconvolution using LASSO regression and an exponential kernel. 
    This function will run on CPU or GPU, but is substantially faster on GPU (use `.gpu()`).

    Args:
        traces (array_like): 2D input data, time along the first axis
        alpha (float): LASSO regularization parameter. If 0, use regular least-squares regression.
        tau (float): exponential kernel decay time constant
        fit_poly (int): if >= 0, fit a polynomial of this order to the traces (keep this <= 3). Set to -1 to not fit a polynomial.
        force_positive (bool): if True, force the weights to be positive
        n_iter (int): number of iterations for LASSO regression

    Returns:
        weights (array_like): 2D array of weights. The last fit_poly columns are the polynomial fit. 
        polyfit (array_like): polynomial fit
        loss (float): loss values (if using GPU)
        T (array_like): mixing matrix (if using GPU)
    """
    if iscupy(traces):
        import cupy as xp
        from cupyx.scipy.linalg import toeplitz
    else:
        import numpy as xp
        from scipy.linalg import toeplitz
    ntp = traces.shape[0]
    kernel = xp.exp(-xp.arange(ntp) / tau).astype('float32')
    T = toeplitz(kernel, xp.zeros(ntp, dtype='float32'))
    n_unconstrained = fit_poly + 1
    for exp in range(n_unconstrained):
        T = xp.hstack([T, xp.linspace(-1, 1, ntp, dtype='float32')[:, None]**exp])
    if alpha == 0:
        weights, loss = xp.linalg.lstsq(T, traces, rcond=None)[:2]
    else:
        weights, loss = regression.lasso_fista(T, traces, alpha=alpha, force_positive=force_positive, n_iter=n_iter, n_unconstrained=n_unconstrained)
    traces = weights[:-n_unconstrained]
    polyfit = weights[-n_unconstrained:]
    return traces, polyfit, loss, T



@cupy_wrapper(['arr'])
def getdff_seg(arr, trace_mask=None, medianfilt_size=7, minfilt_size=15, gaussfilt_sigma=5., truncate=3, min_len=30,
            mode='mirror',only_df=False, epsilon=1. ):
    """
    Calculate deltaF / F0. This function calculates the dF/F traces after estimating a baseline. The baseline is estimated by a
        minimum filter, after de-noising the data via median filter. It is then smoothed by a gaussian filter. If trace_mask is given, the baseline is estimated separately 
        for each segment of the trace where trace_mask is True.

    Args:
        arr (array_like): input array, time along first dimension
        trace_mask (array_like, optional): Boolean mask 1D mask for timepoints, where movement occurs. Defaults to None.
        medianfilt_size (int): size of median cleanup filter. Defaults to 7.
        minfilt_size (int): size of minimum filter to get baseline. Defaults to 15.
        gaussfilt_sigma (float): sigma to smooth baseline. Defaults to 5..
        truncate (int, optional): truncate filter kernels at this many standard deviations. Defaults to 3.
        min_len (int, optional): If trace mask is given and dff is computed over segments, this is the minimum segment length. Defaults to 30.
        mode (str, optional): How to compute baseline at segment edges. Defaults to 'mirror'.
        onlydf (bool): if True, outputs only dF, if False (default) outputs dF/F0. Defaults to False.
        epsilon (float, optional): Minimum F0 value, values below a clipped. Defaults to 1..

    Returns:
        tuple containing
        (array_like): dF/F0 trace (or just dF - see onlydf)
        (array_like): estimated baseline
    """
    if iscupy(arr):
        import cupy as xp
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage
        import numpy  as xp

    if trace_mask is None: 
        temp_f0 = ndimage.median_filter(arr.reshape(arr.shape[0], -1), (medianfilt_size, 1))
        temp_f0 = ndimage.minimum_filter(temp_f0, (minfilt_size, 1))
        temp_f0 = ndimage.gaussian_filter(temp_f0.astype('float32'), (gaussfilt_sigma, 0.0), truncate=truncate)
        #some combinations of truncation and sigma lead to a floating point round down
        temp_f0 = temp_f0.reshape(*arr.shape)
        temp_out = arr - temp_f0  #this is dF
        if not only_df:
            temp_f0 = np.maximum(temp_f0, epsilon)
            temp_out /= temp_f0  #this is dF/F0
        return temp_out, temp_f0


    start = np.squeeze(np.argwhere(np.diff(trace_mask.astype('int'), prepend=False) == 1))
    stop = np.squeeze(np.argwhere(np.diff(trace_mask.astype('int'), prepend=False) == -1))
    if (len(start) - len(stop)) == 1:
        stop = np.append(stop, trace_mask.shape[0])

    seg_len = stop - start
    seg_val = seg_len >= min_len
    stop = stop[seg_val]
    start = start[seg_val]
    n_pad = int(max([medianfilt_size, minfilt_size, truncate * gaussfilt_sigma]))

    if n_pad > min_len:
        print(
            "The padding necessary for mirrored continuation is bigger than the minimum segment length. This can lead to underestimation of baseline.")

    #arr = arr.T

    dff = xp.nan*xp.zeros_like(arr)
    bsl = xp.nan*xp.zeros_like(arr)

   
    for ii, (s0, s1) in enumerate(zip(start, stop)):
        seg = arr[s0:s1, ]
        temp_f0 = ndimage.median_filter(seg, (medianfilt_size, 1))
        temp_f0 = ndimage.minimum_filter(temp_f0, (minfilt_size, 1))
        temp_f0 = ndimage.gaussian_filter(temp_f0.astype('float32'), (gaussfilt_sigma, 0.0), truncate=truncate)

        temp_out = seg - temp_f0  #this is dF
        dff[s0:s1] = temp_out
        bsl[s0:s1] = temp_f0

    if not only_df:
        bsl = xp.maximum(bsl, epsilon)
        dff /= bsl  #this is dF/F0
    return dff,bsl,