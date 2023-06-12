import numpy as np
import scipy.spatial.distance, scipy.cluster.hierarchy


from tqdm.auto import tqdm

import scipy.sparse
from . import xpu, stats, spatial
from .xpu import cupy_wrapper, iscupy, to_numpy


#TODO: remove this function and use spatial.pdist below
def pdist(X, Y, metric='euclidean'):
    if iscupy(X):
        import pylibraft.distance.pairwise_distance
        import cupy as cp
        D = pylibraft.distance.pairwise_distance(X, Y, metric=metric)
        D = cp.array(D, copy=False)
    else:
        import scipy.spatial.distance
        D = scipy.spatial.distance.pdist(X, metric=metric)
    return D


#TODO: use spatial.pdist and remove pdist in this file
@cupy_wrapper(['X'], dtype='float32')
def embed1D_ward(X):
    """ Embed 1D data using Ward clustering. This is a heuristic for finding the best order of the data (rastermap). 
    Executes the distance calculation on the GPU (when calling .gpu). The rest of the function is executed on the CPU.

    Args:
        X (array_like): 2D input

    Returns:
        cix (array_like): 1D index for sorting the data
    """
    D = to_numpy(pdist(X.T, X.T))
    dist_sf = scipy.spatial.distance.squareform(D)
    ward_links = scipy.cluster.hierarchy.ward(dist_sf)
    clix = np.argsort(scipy.cluster.hierarchy.fcluster(ward_links, ward_links.shape[0], criterion='maxclust'))
    return clix


@cupy_wrapper(['X'], dtype='float32')
def embed1D(X, method='ward', **kwargs):
    """ Sort data using one of the following clustering or embedding methods: 'ward', 'umap', 'rastermap'. 
    This is a heuristic for visualizing time traces. 

    Args:
        X (array_like): 2D input
        method (str): 'ward', 'umap', 'rastermap'. Default: 'ward'
        **kwargs: additional arguments for the method

    Returns:
        cix (array_like): 1D index for sorting the data
    """
    if method == 'ward':
        D = to_numpy(spatial.pdist(X.T))
        dist_sf = scipy.spatial.distance.squareform(D)
        ward_links = scipy.cluster.hierarchy.ward(dist_sf, **kwargs)
        clix = np.argsort(scipy.cluster.hierarchy.fcluster(ward_links, ward_links.shape[0], criterion='maxclust'))

    elif method == 'umap':
        if iscupy(X):
            from cuml import UMAP
        else:
            from umap import UMAP
        reducer = UMAP(n_components=1, **kwargs)
        embedded = reducer.fit_transform(X.T)
        clix = np.argsort(embedded.ravel())

    elif method == 'hdbscan':
        if iscupy(X):
            from cuml.cluster import HDBSCAN
        else:
            from hdbscan import HDBSCAN
        clusterer = HDBSCAN(**kwargs)
        tree = clusterer.single_linkage_tree_.to_numpy()
        clix = np.argsort(scipy.cluster.hierarchy.fcluster(tree, tree.shape[0], criterion='maxclust'))

    elif method == 'rastermap':
        import rastermap
        model = rastermap.Rastermap(n_components=1, **kwargs)
        model.fit(to_numpy(X.T))
        clix = model.isort

    return clix