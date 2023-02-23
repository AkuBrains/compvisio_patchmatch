import numpy as np
from PatchMatch import NNS
from numba import njit, prange, set_num_threads, config
from numba_progress import ProgressBar

set_num_threads(3)
config.NUMBA_NUM_THREADS = 3


def estimate_mask(mask, f):
    res = np.zeros(mask.shape)
    m = np.max(mask)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if mask[i, j] != 0:
<<<<<<< HEAD
                res[f[i,j][0], f[i,j][1]] = m
    return res
=======
                res[f[i, j, 0], f[i, j, 1]] = m
    return res


def estimate_mask_inv(mask, f):
    res = np.zeros(mask.shape)
    m = np.max(mask)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            x, y = f[i, j, 0], f[i, j, 1]
            if mask[x, y] == m:
                res[i, j] = m
    return res


@njit(nogil=True, parallel=True)
def monte_carlo_core(img, ref, n_iter, bar, temp, p_size=7, pm_iter=10):
    for k in prange(n_iter):
        temp[k, :, :, :] = NNS(img, ref, p_size, pm_iter)
        bar.update(1)
    return temp


def monte_carlo(img, ref, n_iter, p_size=7, pm_iter=10):
    k, l, _ = img.shape
    temp = np.empty((n_iter, k, l, 2), dtype=np.int32)
    with ProgressBar(total=n_iter) as progress:
        temp = monte_carlo_core(img, ref, n_iter, progress, temp, p_size=p_size, pm_iter=pm_iter)
    return temp


def thresholding(res, threshold=3):
    temp = np.zeros((240, 427))
    for k in range(240):
        for l in range(427):
            if np.sum(res[k, l, :]) >= threshold * 255:
                temp[k, l] = 255
    return temp
>>>>>>> 70c5a60 (Parallized version of PM)
