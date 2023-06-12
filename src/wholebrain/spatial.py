import numpy as np

from . import xpu
from .xpu import cupy_wrapper, iscupy


@cupy_wrapper(['X', 'Y'], dtype='float32')
def pdist(X, Y=None, metric='euclidean'):
    if iscupy(X):
        import pylibraft.distance.pairwise_distance
        import cupy as cp
        if Y is None:
            Y = X
        D = pylibraft.distance.pairwise_distance(X, Y, metric=metric)
        D = cp.array(D, copy=False)
    else:
        import sklearn.metrics
        D = sklearn.metrics.pairwise_distances(X, Y, metric=metric)
    return D
    
    
import numba
@numba.njit(parallel=True)
def pdist_numba(A,B, dtype='float32'):
    pd = np.zeros((A.shape[0], B.shape[0]), dtype=dtype)
    for i in numba.prange(A.shape[0]):
        for j in range(B.shape[0]):
            pd[i,j] = np.sqrt(((A[i]-B[j])**2).sum())
    return pd

#TODO Not agnostic to gpu
def voxelate(coords, traces, binning_factor, average=True):
    """ Voxelate time traces by averaging them in bins of size binning_factor.

    Args:
        coords (array_like): 2D array of coordinates (n_cells, 3)
        traces (array_like): 2D array of time traces (n_timebins, n_cells)
        binning_factor (int): size of the binning factor
        average (bool): if True, average the time traces in each bin. If False, sum them.

    Returns:
        V (array_like): 2D array of voxelated time traces (n_cells, n_voxels)
        cdu (array_like): 2D array of voxel coordinates (n_voxels, 3)
        M (array_like): 2D array of voxel membership (n_cells, n_voxels)
        cells_per_voxel (array_like): 1D array of number of cells per voxel (n_voxels)
    """
    import cupy as cp
    import cupyx
    import scipy.sparse
    cd = (coords / binning_factor).astype('int')
    cdu, ui = np.unique(cd, axis=0, return_inverse=True)
    M = scipy.sparse.coo_matrix((np.ones(ui.size), (np.arange(coords.shape[0]), ui)), shape=(coords.shape[0], cdu.shape[0]), dtype=traces.dtype)
    M = cupyx.scipy.sparse.csr_matrix(M)
    cells_per_voxel = np.bincount(ui, minlength=M.shape[1])
    V = (cp.array(traces) @ M)
    if average:
        V = V / cp.array(cells_per_voxel[None, :].astype(traces.dtype))  #M = M /cells_per_voxel[None,:]
    return V.get(), cdu, M.get(), cells_per_voxel


@cupy_wrapper(['D0', 'D'], dtype='float32')
def distortion_stress(D0, D):
    """ Calculate the distortion stress between two distance matrices. 
    The distortion stress is a measure of how well the distances in D0 are preserved in D. See e.g. https://web.stanford.edu/~boyd/papers/pdf/min_dist_emb.pdf page 16.

    Args:
        D0 (array_like): 2D array of reference distances (n_cells, n_cells)
        D (array_like): 2D array of distances (n_cells, n_cells)

    Returns:
        stress (float): distortion stress
    """
    k = np.sum(D0 * D) / np.sum(D**2)
    numerator = np.sum((D0 - k * D)**2)
    denominator = np.sum(D0**2)
    stress = np.sqrt(numerator / denominator)
    return stress