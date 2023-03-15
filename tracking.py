import numpy as np

from scipy.signal import medfilt2d
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
            x, y = f[i, j, 0], f[i, j, 1]
            if mask[x, y] == m:
                res[i, j] = m
    return res


@njit(nogil=True, parallel=True)
def monte_carlo_core(img, ref, n_iter, temp, p_size=7, pm_iter=10, bar=None):
    for k in prange(n_iter):
        temp[k, :, :, :] = NNS(img, ref, p_size, pm_iter)
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

def smooth(mask, kernel=5):
    return medfilt2d(mask, kernel_size=kernel)


def thresholding(res, threshold):
    m, n, _ = res.shape
    temp = np.zeros((m, n))
    M=np.max(res)
    for k in range(m):
        for l in range(n):
            if np.sum(res[k, l, :]) >= threshold * M:
                temp[k, l] = M
    return temp


class PatchMatchTracking:
    def __init__(self, p_size=9, pm_iter=10, n_iter=5, monte_carlo=True, smooth=False, threshold=4, sm_kernel=5, nb_core=3):
        self.monte_carlo = monte_carlo
        self.smooth = smooth
        self.p_size = p_size
        self.pm_iter = pm_iter
        self.n_iter = n_iter
        self.threshold = threshold
        self.nb_core = nb_core
        self._temp_mc_masks = []
        self._temp_esti_masks = []
        self._temp_esti_masks_smooth = []
        self.sm_kernel = sm_kernel
        self._init_multithreading()

    def _init_multithreading(self):
        set_num_threads(self.nb_core)
        config.NUMBA_NUM_THREADS = self.nb_core

    def _track_core(self, imgs, mask, bar):
        self._temp_mc_masks = []
        self._temp_esti_masks = [mask]
        self._temp_esti_masks_smooth = [mask]
        ref = imgs[0]
        m, n, _ = ref.shape
        for img in imgs[1:]:
            if self.monte_carlo:
                nnf = monte_carlo(img, ref, n_iter=self.n_iter, p_size=self.p_size, pm_iter=self.pm_iter, bar=False)
                res = np.zeros((m, n, self.n_iter))
                for k in range(self.n_iter):
                    res[:, :, k] = estimate_mask(mask, nnf[k, :, :, :])
                self._temp_mc_masks.append(res)
                esti_mask = thresholding(res, self.threshold)
            else:
                nnf = NNS(img, ref, self.p_size, self.pm_iter)
                esti_mask = estimate_mask(mask, nnf)
            self._temp_esti_masks.append(esti_mask)
            if self.smooth:
                esti_mask = smooth(esti_mask, self.sm_kernel)
            self._temp_esti_masks_smooth.append(esti_mask)
            bar.update(1)
        return self._temp_esti_masks_smooth

    def track(self, imgs, mask):
        with ProgressBar(total=len(imgs)-1) as progress:
            return self._track_core(imgs, mask, progress)
    def track_with_step(self, imgs, mask, step = 1):
        with ProgressBar(total=len(imgs)-1) as progress:
            return self._track_core_with_step(imgs, mask, progress, step)
    def get_monte_carlo_res(self):
        return self._temp_mc_masks

    def get_estimated_masks(self):
        return self._temp_esti_masks

    def thresholding(self, threshold):
        self._temp_esti_masks = [self._temp_esti_masks[0]]
        for temp in self._temp_mc_masks:
            self._temp_esti_masks.append(thresholding(temp, threshold))
        return self._temp_esti_masks

    def smoothing(self, masks=None, kernel_size=5):
        if masks is None:
            masks = self._temp_esti_masks
        self._temp_esti_masks_smooth = [masks[0]]
        for mask in masks[1:]:
            self._temp_esti_masks_smooth.append(smooth(mask, kernel=kernel_size))
        return self._temp_esti_masks_smooth
    
    def _track_core_with_step(self, imgs, mask, bar,step):
        self._temp_mc_masks = []
        self._temp_esti_masks = [mask]
        self._temp_esti_masks_smooth = [mask]
        ref = imgs[0]
        m, n, _ = ref.shape
        for index,img in enumerate(imgs[1:]):
            print(index)
            if index % step == 0 :
                ref = imgs[index]
                mask = self._temp_esti_masks_smooth[-1]
            if self.monte_carlo:
                nnf = monte_carlo(img, ref, n_iter=self.n_iter, p_size=self.p_size, pm_iter=self.pm_iter, bar=False)
                res = np.zeros((m, n, self.n_iter))
                for k in range(self.n_iter):
                    res[:, :, k] = estimate_mask(mask, nnf[k, :, :, :])
                self._temp_mc_masks.append(res)
                esti_mask = thresholding(res, self.threshold)
            else:
                nnf = NNS(img, ref, self.p_size, self.pm_iter)
                esti_mask = estimate_mask(mask, nnf)
            self._temp_esti_masks.append(esti_mask)
            if self.smooth:
                esti_mask = smooth(esti_mask, self.sm_kernel)
            self._temp_esti_masks_smooth.append(esti_mask)
            bar.update(1)
        return self._temp_esti_masks_smooth



