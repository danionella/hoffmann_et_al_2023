import numpy as np

from tqdm.auto import tqdm
import cupy as cp
from . import cluster,stats,regression,spatial,traces
from scipy.spatial.distance import squareform
from scipy.ndimage import maximum_filter
import xarray as xr


def create_crossvalidation_mask(arr,stripe_period=300,test_fraction=0.2,val_fraction=0.2):
    """
    Creates striped data partitions for input arr

    Args:
        arr (array): _description_
        stripe_period (int, optional): Striping period. Defaults to 300.
        test_fraction (float, optional): Fraction of each stripe that is used as test data. Defaults to 0.2.
        val_fraction (float, optional): Fraction of each strup that is used as trainig data. Defaults to 0.2.

    Returns:
        boolean masks for training,test and validation data

   
    """
    cv_ind = (
    np.arange(arr.shape[0]) % stripe_period
    ) / stripe_period
    cv_test = (
    cv_ind > 1 - (test_fraction + val_fraction)
    ) * (cv_ind <= (1 - val_fraction))
    cv_val = cv_ind > (1 - val_fraction)
    cv_train = ~(cv_test + cv_val)
    assert np.sum(cv_test * cv_val) == 0
    test = np.random.randn(arr.shape[0])
    assert (
    np.sum(test[cv_test + cv_train][cv_train[cv_test + cv_train]] - test[cv_train])
    == 0
    )
    return cv_train,cv_val,cv_test

def apply_to_bins(x, bins, weights, fcn=np.mean):
    out = []
    xs = np.argsort(x)
    x = x[xs]
    weights = weights[xs]
    binix = np.searchsorted(x, bins)
    for i in tqdm(range(len(bins)-1)):
        vals = weights[binix[i]:binix[i+1]]
        out.append(fcn(vals[~np.isnan(vals)]))
    return np.array(out)





def get_target_predictor_split(cix,n_targets,seed=None):
    if seed is not None:
        np.random.seed(seed)
    target_cells = np.random.choice(cix, size=n_targets, replace=False)
    predictor_cells = np.setdiff1d(cix, target_cells)
    return target_cells,predictor_cells



def neighbourhood_correlation(coords, cc, ringlims=[25, 50],**kwargs):
    """
    Given coordinates and correlation values, calcualte the average correlation with
    the neighbourhood including only cell between neighbourlims.

    Also return values for contralateral side assuming that the brain is straight and that the midline
    is located at (cooords.max(0)-cooords.min(0))/2.


    -------

    Args:
        coords (_type_): _description_
        cc (_type_): _description_
        ringlims (list, optional): _description_. Defaults to [25, 50].

    Returns:
        _type_: _description_
    """
    coords_flip = coords.copy()
    coords_flip[:, 0] = -1 * coords_flip[:, 0]
    pd = (spatial.pdist_numba(coords, coords))
    pd_contra = (spatial.pdist_numba(coords, coords_flip))
    mask_ipsi = (pd < ringlims[1]) & (pd > ringlims[0])
    mask_contra = ((pd_contra < ringlims[1]) & (pd_contra > ringlims[0]))
    axis = 1
    ipsi_cc = np.sum((cc * mask_ipsi), axis=axis) / np.sum(mask_ipsi, axis=axis)
    contra_cc = np.sum((cc * mask_contra), axis=axis) / np.sum(mask_contra, axis=axis)
    return ipsi_cc, contra_cc



def corr_fc(cc, pd,dff_traces_m, min_dist=400, radius_list=[5, 10, 15, 25, 50, 100, 150, 200], save_full_at=[5, 50, 100, 200],
            dec_factor=20,**kwargs):
    """
    Given pairwise distances, correlations and traces this function calculates the secondary correlation between
    the functional connectivity of a cell and the voxel centered at that given cell with varying size. 

    Mixes GPU and CPU calculation in a suboptimal way to lower memory usage on GPU. Re-write if not constrained by this. 

    Args:
        cc (_type_): correlation matrix of cells
        pd (_type_): pairwise distance matrix of cells
        dff_traces_m (_type_): temporal traces of cells 
        min_dist (int, optional): Minimal distance for cells to br considered.  Defaults to 400.
        radius_list (list, optional): _description_. Defaults to [5, 10, 15, 25, 50, 100, 150, 200].
        save_full_at (list, optional): _description_. Defaults to [5, 50, 100, 200].
        dec_factor (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """

    y_list = []
    r_all=[]
    mask = (pd > min_dist).astype('single')
    x,y=np.nonzero(mask == 0)
    mask[x,y]=np.nan
    del x,y
    radius_list = np.sort(list(set(radius_list) | set(save_full_at)))
    x = cc.flatten()[::dec_factor] * mask.flatten()[::dec_factor]
    mask=cp.array(mask,'single')

    for i, s in enumerate(tqdm(radius_list,leave=False)):
        kernel = (pd <= s).astype('float32')
        kernel = kernel / kernel.sum(0, keepdims=True)
        dff_traces_m_la = dff_traces_m @ cp.array(kernel)
        del kernel
        cc_corr_dff_la = stats.corrx(dff_traces_m_la, dff_traces_m_la)
        cc_corr_dff_la *= mask
        y = cc_corr_dff_la.flatten()[::dec_factor]
        if s in save_full_at:
            y_list.append(y.get())
        ind=~np.isnan(x)
        r = stats.pearsonr(cp.array(x[ind]), y[ind]).get()
        r_all.append(r)
        del cc_corr_dff_la,y,dff_traces_m_la
    r_all = np.array(r_all)
    return x, r_all, np.array(y_list)


def pca_run(dff_traces,cv_mask,dims_list,n_repeats=20,n_targets=4000):
    """
     Using different predictor/target splits compute the cross-validated pca_ref

    Parameters
    ----------
    dff_traces
    cv_mask
    dims_list
    n_repeats
    n_targets

    Returns
    -------

    """
    cix = np.arange(dff_traces.shape[1])
    r2_bcvpca = np.zeros((n_repeats, len(dims_list)))
    for j in tqdm(range(n_repeats),leave=False):
        target_cells,predictor_cells=get_target_predictor_split(cix,n_targets)
        r2_bcvpca[j, :] = \
            regression.bcv_pca_dim(cp.array(dff_traces[:, predictor_cells]), cp.array(dff_traces[:, target_cells]), cv_mask,
                            dims_list)[1]
    return xr.DataArray(data=r2_bcvpca,dims=["n_repeats","dimensions"],coords=[np.arange(n_repeats),dims_list])


def voxelate_regression(s_bins,dff_traces_m,coords,cv_mask ,n_repeats=20,alpha=100,n_targets=4000):
    """
    Computes ridge regression for a subset of target cells from voxelised data via ridge regression.
    Regression is cross-validated via a cvmask. Targetcells are not part of predictors.

    Parameters
    ----------
    s_bins
    dff_traces_m
    coords
    cv_mask
    n_repeats
    alpha
    n_targets

    Returns
    -------
    nnz - numbers for predictors
    r2_voxelate - R2 values for targets reached by regression

    """

    r2_voxelate = xr.DataArray(np.zeros((n_repeats, len(s_bins))),dims=["n_repeats","voxelsize"],
                               coords=[np.arange(n_repeats),s_bins])
    nnz = r2_voxelate.copy()
    cix = np.arange(dff_traces_m.shape[1])


    for j in tqdm(range(n_repeats),leave=False):
        target_cells = np.random.choice(cix, size=n_targets, replace=False)
        predictor_cells = np.setdiff1d(cix, target_cells)
        std_0 = dff_traces_m[:, predictor_cells].std(0).mean()
        offset = np.random.rand()
        for k, s in enumerate(tqdm(s_bins, leave=False)):
            X = spatial.voxelate(coords[predictor_cells] + offset * s, dff_traces_m[:, predictor_cells], s)[0]
            # TODO We don't need to do this anymore?
            #Scaling data for constant alpha
            std_1=X.std(0).mean()
            X=X*std_0/std_1
            nnz[j, k] = X.shape[1]
            r2_voxelate[j, k] = regression.ridgeCV(X, dff_traces_m[:, target_cells], cv_mask, alpha=alpha)[1]

    return nnz,r2_voxelate

def ridge_random(dff_traces,  npreds_list,  cv_mask, n_repeats=20, alpha=100, n_targets=4000):
    """
    Computes the ridge regression of n_targets for random predictors chosen from the rest of the cells for different numbers of predictors
    specified in npreds_list. Regression is cross-validated via a cvmask.

    Args:
        dff_traces (array-like): Temporal traces of cells
        npreds_list (list): List of number of predictors to use
        cv_mask (array-like): cross-validation mask 
        n_repeats (int, optional): number of repeats for every number of predictors,  Defaults to 20.
        alpha (int, optional): alpha regularization value for ridge regression.  Defaults to 100.
        n_targets (int, optional): Number of regression targets to use. Defaults to 4000.

    Returns:
        array-like: R2-score for every repeat and every number of predictors
    """
    cix = np.arange(dff_traces.shape[1])
    y = dff_traces
    r2_rand_pred = np.zeros((n_repeats, len(npreds_list))) * np.nan
    target_cells = np.sort(np.random.choice(cix, size=n_targets, replace=False))
    cix=np.setdiff1d(cix,target_cells)
    for j in tqdm(range(n_repeats),leave=False):
        for k, n_preds in enumerate(tqdm(npreds_list, leave=False)):
            if n_preds>len(cix):
                r2_rand_pred[j, k]=None
                continue
            predictor_cells = np.sort(np.random.choice(cix, size=n_preds, replace=False))
            X_tmp = dff_traces[:, predictor_cells]
            r2_rand_pred[j, k]= regression.ridgeCV(X_tmp, y[:, target_cells], cv_mask, alpha=alpha)[1]

   
    return  xr.DataArray(r2_rand_pred, dims=["n_repeats", "n_predictors"], coords=[np.arange(n_repeats), npreds_list])


def voxelate_alpha_scan(s_bins, n_pred_list, coords, dff_traces_m, cv_test, cv_train, alphas,n_repeats=20, n_targets=4000):
    """
    This function computs a ridge regression for a subset of target cells from voxelised data via ridge regression for many different alpha values.
    It returns the R2-score on validation data for every alpha value and every voxel size. Regression is cross-validated via test,training and validation split

    Args:
        s_bins (list): list of voxel sizes to use
        n_pred_list (list): list contains number of predictors to use
        coords (array-like): 3D coordinates of cells
        dff_traces_m (array-like): temporal traces of cells
        cv_test (array-like): cross-validation mask for testing data
        cv_train (array-like): cross-validation mask for training data
        alphas (list): list of alpha values to uses
        n_repeats (int, optional): Number of repeats for each condition.  Defaults to 20.
        n_targets (int, optional): Number of prediction targets to use. Defaults to 4000.

    Returns:
        xarray: Containing R2-scores for all conditions 


    """
    cix = np.arange(dff_traces_m.shape[1])

    R2s = xr.DataArray(np.zeros((2, len(alphas), len(n_pred_list), n_repeats, len(s_bins))) * np.nan,
                    dims=["shuffled", "alphas", "num_predictors", "repeats", "voxel_size"]
                    , coords=[[False, True], alphas, n_pred_list, range(n_repeats), s_bins])


    ind_mask = cv_test + cv_train

    for i_pred, n_pred in enumerate(tqdm(n_pred_list, leave=False)):
        for k, voxel_size in enumerate(tqdm(s_bins, leave=False)):
            for i in tqdm(range(n_repeats), leave=False):
                target_cells = np.random.choice(cix, size=n_targets, replace=False)
                predictor_cells = np.setdiff1d(cix, target_cells)
                X = dff_traces_m[:, predictor_cells]
                y = dff_traces_m[:, target_cells]

                V = spatial.voxelate(coords[predictor_cells] + voxel_size * np.random.rand(), X, voxel_size)[0]
                V_sh = \
                    spatial.voxelate(coords[predictor_cells] + voxel_size * np.random.rand(),
                                    np.random.permutation(X.T).T,
                                    voxel_size)[0]
                if min(V.shape[1], V_sh.shape[1]) < n_pred: continue
                ix = np.random.choice(np.arange(V.shape[1]), size=(n_pred), replace=False)
                Xv = V[:, ix]

                ix = np.random.choice(np.arange(V_sh.shape[1]), size=(n_pred), replace=False)
                Xv_sh = V_sh[:, ix]

                R2s[0, :, i_pred, i, k] = regression.alphascan(cp.array(Xv[ind_mask],'single'), cp.array(y[ind_mask],'single'), cv_train[ind_mask], alphas)
                R2s[1, :, i_pred, i, k] = regression.alphascan(cp.array(Xv_sh[ind_mask],'single'), cp.array(y[ind_mask],'single'), cv_train[ind_mask], alphas)


    return R2s





def voxelate_all(s_bins, n_pred_list, coords, dff_traces_m, cv_mask, alphas_max,n_repeats=20, n_targets=4000):
    """
    This function computs a ridge regression for a subset of target cells from voxelised data with a different numbers of predictors via ridge regression by using the alpha values determined from the output of voxelate_alpha_scan. It returns the R2-score on validation data (alphascan operated on test data) for every voxel size and predictor number. This is done in a whole-brain fashion, i.e. for every repeat every cell is part of the targets once.

    Args:
        s_bins (list): list of voxel sizes to use
        n_pred_list (list): list contains number of predictors to use
        coords (array-like): 3D coordinates of cells
        dff_traces_m (array-like): temporal traces of cells
        cv_mask (array-like): cross-validation mask 
        alphas_max(list): alpha values to use for each condition , determined from the output of voxelate_alpha_scan
        n_repeats (int, optional): Number of repeats of whole brain predictions to use, i.e. for every repeat every cell is part of the target once.  Defaults to 20.
        n_targets (int, optional): Number of prediction targets to use. Defaults to 4000.

    Returns:
        xarray: R2score for every condition

  
    """

    R2s_val = xr.DataArray(np.zeros((2, len(n_pred_list), n_repeats, coords.shape[0], len(s_bins))) * np.nan,
                           dims=["shuffled", "num_predictors", "repeats", "cells", "voxel_size"]
                           , coords=[[False, True], n_pred_list, range(n_repeats), range(coords.shape[0]), s_bins])

    batch_id = xr.DataArray(np.zeros((2, len(n_pred_list), n_repeats, coords.shape[0], len(s_bins))) * np.nan,
                            dims=["shuffled", "num_predictors", "repeats", "cells", "voxel_size"]
                            , coords=[[False, True], n_pred_list, range(n_repeats), range(coords.shape[0]), s_bins])

    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    ibatch = 0

    cix = np.arange(dff_traces_m.shape[1])
    for i_pred, n_pred in enumerate(tqdm(n_pred_list, leave=False)):
        for k, voxel_size in enumerate(tqdm(s_bins, leave=False)):
            for i_rep in tqdm(range(n_repeats), leave=False):
                cix = np.random.permutation(cix)
                for target_cells in tqdm(divide_chunks(cix, n_targets), leave=False):

                    predictor_cells = np.setdiff1d(cix, target_cells)
                    X = dff_traces_m[:, predictor_cells]
                    y = dff_traces_m[:, target_cells]

                    V = spatial.voxelate(coords[predictor_cells] + voxel_size * np.random.rand(), X, voxel_size)[0]
                    V_sh = \
                        spatial.voxelate(coords[predictor_cells] + voxel_size * np.random.rand(),
                                        np.random.permutation(X.T).T,
                                        voxel_size)[0]
                    if min(V.shape[1], V_sh.shape[1]) < n_pred: continue



                    ix = np.random.choice(np.arange(V.shape[1]), size=(n_pred), replace=False)
                    Xv = V[:, ix]

                    alpha = float(alphas_max.isel(shuffled=False)[i_pred, k].data)
                    R2s_val[0, i_pred, i_rep, target_cells, k] = \
                    regression.ridgeCV(cp.array(Xv,'single'), cp.array(y,'single'), cv_mask, alpha=alpha, variance_weighted=False)[1].get()
                    batch_id[0, i_pred, i_rep, target_cells, k] = ibatch
                    ibatch += 1

                    alpha = float(alphas_max.isel(shuffled=True)[i_pred, k].data)
                    ix = np.random.choice(np.arange(V_sh.shape[1]), size=(n_pred), replace=False)
                    Xv = V_sh[:, ix]
                    R2s_val[1, i_pred, i_rep, target_cells, k] = \
                    regression.ridgeCV(cp.array(Xv,'single'), cp.array(y,'single'), cv_mask, alpha=alpha, variance_weighted=False)[1].get()
                    batch_id[1, i_pred, i_rep, target_cells, k] = ibatch
                    ibatch += 1
    return R2s_val, batch_id


                 


def rand_projections(dff_traces_m, n_pred_list, cv_train, cv_test, n_repeats=5, alphas=np.geomspace(0.1, 1e9, 100),n_targets=4000,**kwargs):
    """
    This function computes a ridge regression for a subset of target cells with a different numbers of random predictors, for different alphas.

    Args:
        dff_traces_m (array-like): temporal traces of cells
        n_pred_list (list): list of different numbers of predictors to use
        cv_train (array-like): crossvalidation mask for training data
        cv_test (_type_): crossvalidation mask for test data
        n_repeats (int, optional): Number of repeats. Defaults to 5.
        alphas (list): list of alpha values to use. Defaults to np.geomspace(0.1, 1e9, 100).
        n_targets (int, optional): Number of prediction targets to use. Defaults to 4000.

    Returns:
        xarray: R2score for every condition
    """
    cix = np.arange(dff_traces_m.shape[1])
    ind_mask = cv_test + cv_train

    R2s_rnd = xr.DataArray(
        np.zeros((len(n_pred_list), n_repeats)) * np.nan,
        dims=["num_predictors", "repeats"],
        coords=[n_pred_list, range(n_repeats)],
    )

    for i_pred, n_pred in enumerate(tqdm(n_pred_list, leave=False)):
        for i_rep in tqdm(range(n_repeats), leave=False):
            target_cells = np.random.choice(cix, size=n_targets, replace=False)
            predictor_cells = np.setdiff1d(cix, target_cells)
            X = dff_traces_m[:, predictor_cells]
            y = dff_traces_m[:, target_cells]
            P = np.random.randn(n_pred, len(predictor_cells))

            scan_rnd = regression.alphascan(
                (X @ P.T)[ind_mask], y[ind_mask], cv_train[ind_mask], alphas
            )

            cv_mask = cv_train + cv_test
            R2s_rnd[i_pred, i_rep] = regression.ridgeCV(
                (X @ P.T),
                y,
                cv_mask,
                alpha=alphas[np.argmax(scan_rnd)],
                variance_weighted=True,
            )[1]

    return R2s_rnd



  

def distortion(s_bins, n_pred_list, coords, dff_traces_m, n_repeats=20, n_targets=4000):
    """


Args:
    s_bins (list): list of voxel sizes to use
    n_pred_list (list): list contains number of predictors to use
    coords (array-like): 3D coordinates of cells
    dff_traces_m (array-like): temporal traces of cells
    n_repeats (int, optional): Number of repeats of whole brain predictions to use, i.e. for every repeat every cell is part of the target once.  Defaults to 20.
    n_targets (int, optional): Number of prediction targets to use. Defaults to 4000.

Returns:
    xarray: matrix of similarity score for every condition


"""

    sim_val = xr.DataArray(np.zeros((2, len(n_pred_list), n_repeats, len(s_bins),3)) * np.nan,
                            dims=["shuffled", "num_predictors", "repeats", "voxel_size","measure"]
                            , coords=[[False, True], n_pred_list, range(n_repeats), s_bins,["distortion_euclidian","similarity_corr","similarity_cov"]])



    cix = np.arange(dff_traces_m.shape[1])
    for i_pred, n_pred in enumerate(tqdm(n_pred_list, leave=False)):
        for k, voxel_size in enumerate(tqdm(s_bins, leave=False)):
            for i_rep in tqdm(range(n_repeats), leave=False):
                cix = np.random.permutation(cix)
                target_cells=cix[:n_targets]  
                predictor_cells = np.setdiff1d(cix, target_cells)
                X = dff_traces_m[:, predictor_cells]
                y = dff_traces_m[:, target_cells]
                
                pdist0 = spatial.pdist.gpu(y)
                corrx0 = stats.corrx.gpu(y.T,y.T)
                covx0 = stats.cov_matrix.gpu(y.T,y.T)



                V = spatial.voxelate(coords[predictor_cells] + voxel_size * np.random.rand(), X, voxel_size)[0]
                V_sh = \
                    spatial.voxelate(coords[predictor_cells] + voxel_size * np.random.rand(),
                                    np.random.permutation(X.T).T,
                                    voxel_size)[0]
                if min(V.shape[1], V_sh.shape[1]) < n_pred: continue

                indx=np.random.choice(V.shape[1],size=n_pred)
                indx_sh=np.random.choice(V_sh.shape[1],size=n_pred)
                sim_val[0, i_pred, i_rep,k, 0]  = spatial.distortion_stress(pdist0, spatial.pdist.gpu(V[:,indx]))
                sim_val[0, i_pred, i_rep, k, 1]  = stats.pearsonr.gpu(corrx0.ravel(),stats.corrx.gpu(V[:,indx].T,V[:,indx].T).ravel())[0]
                sim_val[0, i_pred, i_rep, k,  2] = stats.pearsonr.gpu(covx0.ravel(),stats.cov_matrix.gpu(V[:,indx].T,V[:,indx].T).ravel())[0]
                
                sim_val[1, i_pred, i_rep, k , 0] = spatial.distortion_stress(pdist0, spatial.pdist.gpu(V_sh[:,indx_sh]))
                sim_val[1, i_pred, i_rep, k, 1] = stats.pearsonr.gpu(corrx0.ravel(),stats.corrx.gpu(V_sh[:,indx_sh].T,V_sh[:,indx_sh].T).ravel())[0]
                sim_val[1, i_pred, i_rep, k, 2] = stats.pearsonr.gpu(covx0.ravel(),stats.cov_matrix.gpu(V_sh[:,indx_sh].T,V_sh[:,indx_sh].T).ravel())[0]
                        
    return  sim_val