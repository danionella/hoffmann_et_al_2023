import numpy as np
from tqdm.auto import tqdm

from . import xpu, stats
from .xpu import cupy_wrapper, iscupy, get_array_module

cov = lambda x: x @ x.T
covm = lambda x, ax=0: cov(x - x.mean(ax, keepdims=True))
twod = lambda x: x.reshape(x.shape[0], -1)


@cupy_wrapper(['regressors', 'data'], dtype='float32')
def regress(regressors, data, non_negative=False):
    if non_negative:
        regrw = nnls(regressors, twod(data), rcond=None)[0]
    else:
        regrw = np.linalg.lstsq(regressors, twod(data), rcond=None)[0]
    regr_r = stats.pearsonr((regressors @ regrw).reshape(data.shape), data)
    return regrw.reshape(-1, *data.shape[1:]), regr_r


@cupy_wrapper(['A', 'b'], dtype='float32')
def nnls(A, b, max_iter=20, eps=1e-6):
    """ GPU-compatible non-negative least squares regression, solving A@x = b with non-negativity constraint.
    Uses accelerated projected gradient descent (APGD) with restart and Paul Tseng's beta, as described in:
    https://angms.science/doc/NMF/nnls_pgd.pdf

    Args:
        A (array_like): mixing matrix (e.g. regressors)
        b (array_like): target
        max_iter (int): maximum number of iterations
        eps (float): convergence criterion

    Returns:
        x (array_like): solution for x
        err (array_like): 1D array of MSE as a function of iteration
    """
    AtA = A.T @ A
    xp = xpu.get_array_module(A, b)
    L = xp.linalg.norm(AtA, 2)
    T = xp.eye(AtA.shape[0]) - AtA / L
    Atb = A.T @ b / L

    err = xp.ones(max_iter, dtype='float32') * xp.nan
    x = xp.maximum(xp.linalg.lstsq(A, b)[0], 0)
    err[0] = ((A @ x - b)**2).mean()
    y = x

    for k in range(1, max_iter):
        lastx = x
        x = xp.maximum((T @ y) + Atb, 0)
        beta = k / (k + 3)
        y = x + (x - lastx) * beta
        err[k] = ((A @ x - b)**2).mean()
        if err[k] > err[k - 1]:
            x = xp.maximum((T @ lastx) + Atb, 0)
            y = x
            err[k] = ((A @ x - b)**2).mean()
        if (err[k - 1] - err[k]) / err[k] < eps:
            break

    return x, err


@cupy_wrapper(['X', 'y'], dtype='float32')
def ridge(X, y, alpha=1, use_svd=None):
    """ Ridge regression using cupy

    Args:
        X (array_like): regressors
        y (array_like): target
        alpha (float): regularization parameter
        use_svd (bool): whether to use SVD to solve the problem. If True, will use SVD to solve the problem. If False, will use the normal equation. If None, will use SVD if X.shape[1] > 5000

    Returns:
        weights (array_like): weights for ridge regression
    """
    use_svd = (X.shape[1] > 5000) if use_svd is None else use_svd
    xp = xpu.get_array_module(X, y)
    if use_svd:
        u, s, vh = xp.linalg.svd(X, full_matrices=False)
        pinv_ridge = vh.T @ (xp.diag(s / (s**2 + alpha)) @ u.T)
        weights = pinv_ridge @ y
    else:
        XtXa = (X.T @ X) + xp.eye(X.shape[1]) * alpha
        XtY = X.T @ y
        weights = xp.linalg.solve(XtXa, XtY)
    return weights


@cupy_wrapper(['X', 'y', 'cv_mask'], dtype='float32')
def ridgeCV(X, y, cv_mask, alpha=1, use_svd=None,variance_weighted=True):
    """ Cross-validated ridge regression

    Args:
        X (array_like): regressors
        y (array_like): target
        cv_mask (array_like): boolean mask for cross-validation
        alpha (float): regularization parameter
        use_svd (bool): whether to use SVD to solve the problem. If True, will use SVD to solve the problem. If False, will use the normal equation. If None, will use SVD if X.shape[1] > 5000
        variance_weighted (bool): whether to weight the R2 score by the variance of the true values

    """
    weights = ridge(X[cv_mask], y[cv_mask], alpha=alpha, use_svd=use_svd)
    y_pred = X[~cv_mask] @ weights
    r2 = r2_score(y[~cv_mask], y_pred,variance_weighted=variance_weighted)
    return weights, r2


def r2_score(y_true, y_pred, variance_weighted=True):
    """ Calculate R2 score

    Args:
        y_true (array_like): true values
        y_pred (array_like): predicted values
        variance_weighted (bool): whether to weight the R2 score by the variance of the true values

    Returns:
        r2 (float): R2 score
    """
    ss_tot = ((y_true - y_true.mean(0))**2).sum(0)
    ss_res = ((y_true - y_pred)**2).sum(0)
    r2 = 1 - (ss_res / ss_tot)
    if variance_weighted:
        r2 = np.average(r2, weights=ss_tot)
    return r2


@cupy_wrapper(['X', 'y', 'cv_mask'], dtype='float32')
def bcv_pca_dim(X, y, cv_mask, dims_list, svd=None):
    """ Cross-validated PCA dimensionality reduction

    Args:
        X (array_like): regressors
        y (array_like): target
        cv_mask (array_like): boolean mask for cross-validation
        dims_list (array_like): list of dimensions to test
        svd (tuple): tuple of (u, s, vh) from SVD. If None, will compute SVD

    Returns:
        dim_max (int): dimensionality that maximizes R2 score
        scores (array_like): 1D array of R2 scores as a function of dimensionality
    """

    assert np.all(dims_list[:-1] <= dims_list[1:]), "Dimlist needs to be ascending"

    if svd is None:
        u, s, vh = np.linalg.svd(X[cv_mask], full_matrices=False)
    else:
        u, s, vh = svd
    dims_list = np.r_[0, dims_list]
    XVS_inv = X[~cv_mask] @ (vh.T / s)
    UtY = u.T @ y[cv_mask]
    y_pred = 0
    scores = []
    for i, ndim in enumerate(dims_list[:-1]):
        y_pred = y_pred + XVS_inv[:, dims_list[i]:dims_list[i + 1]] @ UtY[dims_list[i]:dims_list[i + 1]]
        r2 = r2_score(y[~cv_mask], y_pred)
        if iscupy(r2):
            r2 = r2.get()
        scores.append(r2)
    dim_max = dims_list[np.argmax(scores) + 1]
    return dim_max, np.array(scores)


@cupy_wrapper(['x', 'y'], dtype='float32')
def cca(x, y):
    """ Canonical correlation analysis using SVD

    Args:
        x (array_like): regressors
        y (array_like): target

    Returns:
        a (array_like): weights for x
        b (array_like): weights for y
        u (array_like): canonical variates for x
        v (array_like): canonical variates for y
    """
    usvh_x = np.linalg.svd(x, full_matrices=False)
    usvh_y = np.linalg.svd(y, full_matrices=False)
    usvh = np.linalg.svd(usvh_x[0].T @ usvh_y[0], full_matrices=False)
    a = usvh_x[2].T @ 1 / usvh_x[1] @ usvh[0]
    b = usvh_y[2].T @ 1 / usvh_y[1] @ usvh[2].T
    u = x @ a
    v = y @ b
    return a, b, u, v


@cupy_wrapper(['x', 'y'], dtype='float32')
def rrr(x, y, rank=None, alpha=0):
    """ Reduced rank regression. If rank is None, returns a function that can be used to compute the reduced rank regression matrix for a given rank. See doi.org/10.1016/j.neuron.2019.01.026

    Args:
        x (array_like): regressors
        y (array_like): target
        rank (int): rank of reduced rank regression matrix
        alpha (float): ridge regularization parameter

    Returns:
        B_rrr (array_like or callable): reduced rank regression matrix or function that computes reduced rank regression matrix for a given rank
    """
    #B_ols = cp.linalg.pinv(x) @ y
    B_ols = ridge(x, y, alpha=alpha)
    u, s, vh = np.linalg.svd(x @ B_ols, full_matrices=False)
    RRR_f = lambda rank: (B_ols @ vh[:rank].T) @ vh[:rank]
    if rank is None:
        return RRR_f
    else:
        return RRR_f(rank)


@cupy_wrapper(['x', 'y', 'cv_mask'], dtype='float32')
def rrr_cv(x, y, cv_mask, ndims, alpha=0):
    """ Cross-validated reduced rank regression. 
    
    Args:
        x (array_like): regressors
        y (array_like): target
        cv_mask (array_like): boolean mask for cross-validation
        ndims (int): number of dimensions to test
        alpha (float): ridge regularization parameter

    Returns:
        scores (array_like): 1D array of R2 scores as a function of dimensionality
    """

    RRR_f = rrr(x[cv_mask], y[cv_mask], rank=None, alpha=alpha)
    scores = []
    for d in range(ndims):
        y_pred = x[~cv_mask] @ RRR_f(d)
        r2 = r2_score(y[~cv_mask], y_pred)
        if iscupy(r2):
            r2 = r2.get()
        scores.append(r2)
    return np.array(scores)


@cupy_wrapper(['x', 'y'], dtype='float32')
def pls_dejong(x, y, dims):
    """ Partial least squares regression using the De Jong algorithm

    Args:
        x (array_like): regressors
        y (array_like): target
        dims (int): number of dimensions

    Returns:
        W (array_like): weights for x
        P (array_like): weights for y
        C (array_like): canonical variates for x
        T (array_like): canonical variates for y
    """
    # https://doi.org/10.1002/cem.1180080208
    # https://personal.utdallas.edu/~herve/abdi-PLSC_and_PLSR2012.pdf
    # https://personal.utdallas.edu/~herve/abdi-kwmaPLS4NeuroImage2010.pdf
    xp = xpu.get_array_module(x, y)
    XtX = x.T @ x
    XtY = x.T @ y
    W = xp.zeros((x.shape[1], dims), dtype=x.dtype)
    P = xp.zeros((x.shape[1], dims), dtype=x.dtype)
    C = xp.zeros((y.shape[1], dims), dtype=x.dtype)

    for a in tqdm(range(dims), leave=True):
        w = XtY @ xp.linalg.eigh(XtY.T @ XtY)[1][:1, :].T
        w = w / xp.sqrt(w.T @ XtX @ w)
        p = XtX @ w
        c = XtY.T @ w
        XtX -= p @ p.T
        XtY -= p @ c.T
        W[:, a] = w[:, 0]
        P[:, a] = p[:, 0]
        C[:, a] = c[:, 0]

    R = W @ xp.linalg.pinv(P.T @ W)
    B_pls = R @ C.T
    return B_pls


@cupy_wrapper(['x', 'y', 'cv_mask'], dtype='float32')
def pls_dejong_cv(x, y, cv_mask, dims):
    """ Cross-validated partial least squares regression using the De Jong algorithm

    Args:
        x (array_like): regressors
        y (array_like): target
        cv_mask (array_like): boolean mask for cross-validation
        dims (int): number of dimensions

    Returns:
        W (array_like): weights for x
        P (array_like): weights for y
        C (array_like): canonical variates for x
        T (array_like): canonical variates for y
    """
    # https://doi.org/10.1002/cem.1180080208
    xp = xpu.get_array_module(x, y)
    XtX = x[cv_mask].T @ x[cv_mask]
    XtY = x[cv_mask].T @ y[cv_mask]
    W = xp.zeros((XtX.shape[0], dims), dtype=x.dtype)
    P = xp.zeros((XtY.shape[0], dims), dtype=x.dtype)
    C = xp.zeros((XtY.shape[1], dims), dtype=x.dtype)
    r2s = []

    for a in range(dims):  # tqdm(range(dims), leave=False):
        _, q = xp.linalg.eigh(XtY.T @ XtY)
        w = XtY @ q[:1, :].T
        w = w / xp.sqrt(w.T @ XtX @ w)
        p = XtX @ w
        c = XtY.T @ w
        XtX -= p @ p.T
        XtY -= p @ c.T
        W[:, a] = w[:, 0]
        P[:, a] = p[:, 0]
        C[:, a] = c[:, 0]
        R = W[:, :a + 1] @ xp.linalg.pinv(P[:, :a + 1].T @ W[:, :a + 1])
        B_pls = R @ C[:, :a + 1].T
        y_pred = x[~cv_mask] @ B_pls
        r2 = r2_score(y[~cv_mask], y_pred)
        r2s.append(r2)

    r2s = xp.array(r2s)
    if iscupy(r2s):
        r2s = r2s.get()
    return r2s


@cupy_wrapper(['x', 'y', 'cv_mask'], dtype='float32')
def svca(x, y, cv_mask):
    """ Shared variance component analysis as described in Stringer 2020 10.1126/science.aav7893

    Args:
        x (array_like): regressors
        y (array_like): target
        cv_mask (array_like): boolean mask for cross-validation

    Returns:
        S (array_like): 
        S_tot (array_like): 
    """
    XtY_train = x[cv_mask].T @ y[cv_mask]
    u, s, vh = np.linalg.svd(XtY_train, full_matrices=False)
    XtY_test = x[~cv_mask].T @ y[~cv_mask]
    XtX_test = x[~cv_mask].T @ x[~cv_mask]
    YtY_test = y[~cv_mask].T @ y[~cv_mask]
    n_time_points = cv_mask.sum()
    S = np.diag(u.T @ XtY_test @ vh.T) / n_time_points
    S_tot = (np.diag(u.T @ XtX_test @ u) + np.diag(vh @ YtY_test @ vh.T)) / 2 / n_time_points
    return S, S_tot


@cupy_wrapper(['X', 'y', 'cv_mask'], dtype='float32')
def alphadimscan(X, y, cv_mask, max_dim=None, svd=None):
    """ Cross-validated ridge regression to estimate dimensionality using alpha.

    Args:
        X (array_like): regressors
        y (array_like): target
        cv_mask (array_like): boolean mask for cross-validation
        max_dim (int): maximum number of dimensions to scan. If None, all dimensions are scanned.
        svd (tuple): precomputed SVD, if available
    """
    svd = svd or np.linalg.svd(X[cv_mask], full_matrices=False)
    singular_values = svd[1]
    max_dim = max_dim or len(singular_values)
    alphas = singular_values[:max_dim]**2
    r2s = alphascan(X, y, cv_mask, alphas)
    return r2s


@cupy_wrapper(['X', 'y', 'cv_mask'], dtype='float32')
def alphascan(X, y, cv_mask, alphas):
    """ Cross-validated ridge regression to determine optimal alpha

    Args:
        X (array_like): regressors
        y (array_like): target
        cv_mask (array_like): boolean mask for cross-validation
        alphas (array_like): alphas to scan

    Returns:
        r2s (array_like): r2 scores for each alpha
    """
    r2s = []
    for a in alphas:
        B_ridge = ridge(X[cv_mask], y[cv_mask], alpha=a)
        y_pred = X[~cv_mask] @ B_ridge
        r2 = r2_score(y[~cv_mask], y_pred)
        if iscupy(r2):
            r2 = r2.get()
        r2s.append(r2)
    return np.array(r2s)


@cupy_wrapper(['A', 'b'], dtype='float32')
def lasso_fista(A, b, alpha, n_iter=500, force_positive=False, n_unconstrained=0):
    """ GPU-accelerated LASSO regression using FISTA, with restarts.
    No early stopping, so n_iter should be evaluated by looking at the loss.

    Args:
        A (array_like): regressors
        b (array_like): target
        alpha (float): regularization parameter
        n_iter (int): number of iterations
        force_positive (bool): constrain coefficients to be positive. Default is False.
        n_unconstrained (int): number of regressors (at the end) to not regularize or constrain. Default is 0.

    Returns:
        x (array_like): regression coefficients
        loss (array_like): loss at each iteration

    See also:
        https://doi.org/10.1137/080716542
        https://doi.org/10.1007/s10878-019-00453-7
        https://pdfs.semanticscholar.org/c924/20f001e023c693db762758f9590571256e35.pdf
        https://gist.github.com/agramfort/ac52a57dc6551138e89b
        https://github.com/rfeinman/pytorch-lasso/blob/master/lasso/conv2d/ista.py
    """

    lam = alpha * A.shape[0]  # converting the regularisation parameter of sklearn.linear_model.Lasso (sse/(2*nsamples)) to the one used in FISTA (sse/2).
    xp = get_array_module(A)
    n_constr = A.shape[1] - n_unconstrained
    if iscupy(A):
        if force_positive:
            soft_thresh = xp.ElementwiseKernel('float32 x, float32 thr', 'float32 z', 'z = fmaxf(x - thr, 0.0)')
        else:
            soft_thresh = xp.ElementwiseKernel('float32 x, float32 thr', 'float32 z', 'z = copysignf(fmaxf(fabsf(x) - thr, 0.0), x)')
        sumabs = xp.ReductionKernel('float32 x, float32 factor', 'float32 out', 'fabsf(x)', 'a + b', 'out = a * factor', '0')
        sse = xp.ReductionKernel('float32 Ax, float32 B, float32 factor', 'float32 out', '(Ax-B)*(Ax-B)', 'a + b', 'out = a * factor', '0')
        loss_func  = lambda x: sse(A @ x, b, 0.5, axis=0) + sumabs(x, lam, axis=0)
    else: # numpy
        if force_positive:
            soft_thresh = lambda x, thr: np.maximum(x - thr, 0)
        else:
            soft_thresh = lambda x, thr: np.sign(x) * np.maximum(np.abs(x) - thr, 0)
        loss_func = lambda x: 0.5 * ((A @ x - b)**2).sum(axis=0) + lam * np.abs(x[:n_constr]).sum(axis=0)

    x = xp.zeros((A.shape[1], b.shape[1]), dtype='float32')
    L = xp.linalg.norm(A, ord=2)**2
    AtBl = A.T @ b / L
    AtAl = A.T @ A / L
    z = x.copy()
    loss = [loss_func(x)]

    for k in tqdm(range(n_iter), leave=False):
        xold = x.copy()
        z -= AtAl @ z   #z = z + A.T @ (b - A@z) / L
        z += AtBl
        x = soft_thresh(z, lam / L)
        x[n_constr:] = z[n_constr:]
        loss.append(loss_func(x))
        restart = (loss[-1] > loss[-2])
        w = np.float32(k / (k + 3)) * ~restart
        z = x + w * (x - xold)

    loss = xp.array(loss)
    return x, loss



