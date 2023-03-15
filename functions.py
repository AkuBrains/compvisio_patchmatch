from numba import njit, prange, set_num_threads, config
from numba_progress import ProgressBar
import numpy as np
from PatchMatch import NNF


set_num_threads(3)
config.NUMBA_NUM_THREADS = 3


def estimate_mask(mask, f):
    res = np.zeros(mask.shape)
    m = np.max(mask)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            x, y = f[i, j, 0], f[i, j, 1]
            if mask[x, y] == m:
                res[i, j] = m
    return res


@njit(nogil=True, parallel=True)
def monte_carlo_core(img, ref, n_iter, temp, p_size=7, pm_iter=10, bar=None):
    for k in prange(n_iter):
        temp[k, :, :, :] = NNF(img, ref, p_size, pm_iter)
        if bar is not None:
            bar.update(1)
    return temp


def monte_carlo(img, ref, n_iter=5, p_size=9, pm_iter=10, bar=False):
    k, l, _ = img.shape
    temp = np.empty((n_iter, k, l, 2), dtype=np.int32)
    if bar:
        with ProgressBar(total=n_iter) as progress:
            return progress, monte_carlo_core(img, ref, n_iter, progress, temp, p_size=p_size, pm_iter=pm_iter)
    return monte_carlo_core(img, ref, n_iter, temp, p_size=p_size, pm_iter=pm_iter)


def dice_assessment(groundtruth, estimated, label=255):
    A = groundtruth == label
    B = estimated == label
    TP = len(np.nonzero(A*B)[0])
    FN = len(np.nonzero(A*(~B))[0])
    FP = len(np.nonzero((~A)*B)[0])
    DICE = 0
    if (FP+2*TP+FN) != 0:
        DICE = float(2)*TP/(FP+2*TP+FN)
    return DICE*100