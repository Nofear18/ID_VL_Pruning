import numpy as np
from scipy.stats import chi2, zscore
from scipy.spatial.distance import squareform
from scipy.optimize import newton
from scipy.spatial.distance import pdist


def dens_cdf(x, dim, n):
    return 1 - (1 - chi2.cdf(x, dim)) ** n


def estimate_dens(dim, n):
    acc = 1e-5
    # find the import region
    spc = np.linspace(0, dim * 2, 21)
    Fx_1 = dens_cdf(spc, dim, n)
    if Fx_1[0] > acc:
        raise ValueError("No enought accuracy to estimate the density, left bound low")
    for i in range(1, len(Fx_1)):
        if Fx_1[i] > acc:
            break
    else:
        raise ValueError("No enought accuracy to estimate the density, left bound high")
    left = spc[i - 1]
    if Fx_1[-1] < 1 - acc:
        raise ValueError("No enought accuracy to estimate the density, right bound high")
    for i in range(len(Fx_1) - 1, 0, -1):
        if Fx_1[i] < 1 - acc:
            break
    else:
        raise ValueError("No enought accuracy to estimate the density, right bound low")
    right = spc[i + 1]
    spc = np.linspace(left, right, 101)
    Fx_1 = 1 - (1 - chi2.cdf(spc, dim)) ** n
    fx_1 = Fx_1[1:] - Fx_1[:-1]
    dens = 2 * np.sum(spc[1:] * fx_1)
    return dens


def estimate_dim(dens, n):
    return newton(lambda x: estimate_dens(x, n) - dens, dens)


def id_analysis(X):
    '''
        Perform id-analysis of a system of points from
        the matrics

        Args:
        X : 2-D Matrix X (n,m) where n is the number of points
            and m is the number of dimensions.
    '''
    X = zscore(X, axis=1)
    dist = squareform(pdist(X))

    # TODO: just minimum distance
    Y = np.sort(dist, axis=1, kind="quicksort")
    k1 = Y[:, 1]

    dens = (k1 ** 2).mean()
    n = X.shape[0]
    return estimate_dim(dens, n), dens, n

