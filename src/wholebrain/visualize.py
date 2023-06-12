import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.colors as mpcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

from . import xpu, stats, cluster
from .xpu import cupy_wrapper, iscupy, to_numpy, using_cupy


@cupy_wrapper(['X'], dtype='float32')
def rastermap(X, method='knnward', embed_options=dict(), **kwargs):
    """ Embed 1D data for visualization (rastermap). Uses either k-means followed by Ward clustering (method='knnward') or UMAP (method='umap').
        
    Args:
        X (array_like): 2D input
        method (str): method for embedding. Options: 'knnward', 'umap'
        embed_options (dict): options passed as keyword arguments to the embedding function. For example, n_neighbors for UMAP and n_clusters for k-means.
        **kwargs: keyword arguments for plt.imshow

    Returns:
        cix (array_like): 1D index for sorting the data
        fig (matplotlib.figure.Figure): figure handle
    """
    if iscupy(X):
        from cuml import UMAP, KMeans
    else:
        from sklearn.manifold import UMAP
        from sklearn.cluster import KMeans

    if method == 'knnward':
        options = dict(n_clusters=100)
        options.update(embed_options)
        kmeans = KMeans(**options).fit(X.T)
        ix = cluster.embed1D_ward(kmeans.cluster_centers_.T)
        cix = np.argsort(ix.argsort()[to_numpy(kmeans.labels_)])
    elif method == 'umap':
        cix = UMAP(n_components=1, **embed_options).fit_transform(X.T).flatten().argsort()
    fig = plt.figure()
    plt.imshow(to_numpy(X.T[cix]), aspect='auto', **kwargs)
    return cix, fig


def plot_waterfall(X, sep=1, **kwargs):
    """ Waterfall plot for 1D data. Plots each row of X as a line, with a separation between each line.
        
    Args:
        X (array_like): 2D input
        sep (float): separation between each line
        **kwargs: keyword arguments for plt.plot
    """
    plt.plot(X + np.arange(X.shape[1])[None,:] * sep, **kwargs);



def scatter_brain(coords, cmap='gray', c=None, vmin=0, vmax=1, s=1, alpha=None, quantile=None,off=[-1200,3000],plot_midline=False,rasterized=True,figsize=(12 ,6)):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    args = [np.argsort(coords[:, i]) for i in range(3)]
    x = np.hstack([coords[args[0], 2], coords[args[1], 2], -coords[args[2], 1] + off[1]])
    y = np.hstack([coords[args[0], 1] + off[0], coords[args[1], 0], coords[args[2], 0]])

    if c is None: c=np.ones(len(x))
    c = np.hstack([c[args[0]], c[args[1]], c[args[2]]])
    c = np.clip((c - vmin) / (vmax - vmin), 0, 1)

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(aspect=1), figsize=figsize)

    ax.set_facecolor('black')
    collection = mc.CircleCollection(s * np.ones(len(x)), offsets=np.stack((x, y)).T, transOffset=ax.transData,
                                    facecolor=cmap(c), edgecolor=None,rasterized=rasterized,alpha=alpha)
    ax.add_collection(collection)
    ax.margins(0.03)
    plt.show()
    if plot_midline: ax.axhline(0, c='r', ls='--')

    # cb = plt.colorbar(sc, ax=ax)
    # cb.set_alpha(1)
    # cb.draw_all()
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    scalebar = AnchoredSizeBar(ax.transData, 400, '400 µm', 'lower right', pad=0, color='white', frameon=False,
                            size_vertical=30, sep=5)
    ax.add_artist(scalebar)
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.2)
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=mpcolors.Normalize(vmin, vmax), orientation='vertical')
    plt.gcf().add_axes(ax_cb)

    plt.tight_layout()
    return fig,ax


def scatterplot_matrix(data, names, xlims, **kwargs):
    """ Scatterplot matrix for 2D data. 
    Args:
        data (array_like): 2D input
        names (list): list of names for each variable
        xlims (list): list of tuples for x-limits for each variable
        **kwargs: keyword arguments for plt.scatter
    """
    numvars = len(data)
    fig, axes = plt.subplots(numvars, numvars, figsize=(12, 12))
    #fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for x in range(numvars):
        for y in range(numvars):
            # axes[x,y].xaxis.set_visible(False)
            # axes[x,y].yaxis.set_visible(False)
            if x == numvars - 1:
                axes[x, y].xaxis.set_visible(True)
                axes[x, y].set(xlabel=names[y])
            if y == 0:
                axes[x, y].yaxis.set_visible(True)
                axes[x, y].set(ylabel=names[x])
            if x == y:
                axes[y, x].hist(data[x], np.linspace(xlims[x][0], xlims[x][1], 100))
                axes[y, x].set(xlim=xlims[x])
            else:
                axes[y, x].scatter(data[x], data[y], **kwargs)
                axes[y, x].set(xlim=xlims[x], ylim=xlims[y])
    plt.tight_layout()