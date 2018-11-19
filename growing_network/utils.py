import numpy as np
from scipy import linalg


def pca(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    mean_centered_data = data - data.mean(axis=0)
    r = np.cov(mean_centered_data, rowvar=False, bias=True)
    evals, evecs = linalg.eigh(r)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :dims_rescaled_data]
    return np.dot(evecs.T, data.T).T, evals, evecs


def squared_distances_sklearn_based(x, y, x_norm_squared=None, y_norm_squared=None):
    if x.ndim == 1:
        x = np.matrix(x, copy=False)

    if x_norm_squared is not None:
        xx = x_norm_squared
    else:
        xx = np.einsum('ij,ij->i', x, x)[:, np.newaxis]

    if y_norm_squared is not None:
        yy = np.atleast_2d(y_norm_squared)
    else:
        yy = np.einsum('ij,ij->i', y, y)[np.newaxis, :]

    return (xx + yy - (2 * np.dot(x, y.T))).A1


def argmin_and_second_argmin(a):
    """
    :param a: array-like
    :return: first argmin, second argmin, min
            or if n > 2 index array, None, min
    """
    if a is None or a.size == 0:
        return np.nan, np.nan, np.nan
    if a.ndim > 1:
        a = a.reshape(-1)
    if a.shape[0] == 1:
        return 0, np.nan, abs(a[0])
    elif a.shape[0] == 2:
        # it can be that both numbers are equal, in that case this doesn't work!
        if (a[0] == a[1]):
            ind = [0, 1]
        else:
            ind = [np.argmin(a), np.argmax(a)]
    else:
        ind = np.argpartition(a, 2)[:2]
    return ind[0], ind[1], a[ind[0]]
