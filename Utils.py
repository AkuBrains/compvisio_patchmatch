import numpy as np

from PatchMatch import NNS
from numba import njit, prange, set_num_threads, config
from numba_progress import ProgressBar
from PIL import Image

set_num_threads(3)
config.NUMBA_NUM_THREADS = 3


def estimate_mask(mask, f):
    res = np.zeros(mask.shape)
    m = np.max(mask)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
<<<<<<< HEAD
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
=======
>>>>>>> 17554a9 (Create new class to track object using PM)
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



def thresholding(res, threshold):
    m, n, _ = res.shape
    temp = np.zeros((m, n))
    M=np.max(res)
    for k in range(m):
        for l in range(n):
            if np.sum(res[k, l, :]) >= threshold * M:
                temp[k, l] = M
    return temp
<<<<<<< HEAD
>>>>>>> 70c5a60 (Parallized version of PM)
=======

def make_gif(imgs, name):
    frames = [Image.fromarray(img) for img in imgs]
    frame_one = frames[0]
    frame_one.save(f"{name}.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=200, loop=0)


def mask_on_image(img, mask):
    m = np.max(mask)
    temp = np.copy(img)
    temp1 = m * np.ones(img.shape)
    temp[mask==m] = np.array([m, m, m])
    temp1[mask==m,:] = img[mask==m,:]
    return temp, temp1.astype(np.uint8)

class PatchMatchTracking:
    def __init__(self, p_size=9, pm_iter=10, n_iter=5, monte_carlo=True, smooth=True, threshold=4, nb_core=3):
        self.monte_carlo = monte_carlo
        self.smooth = smooth
        self.p_size = p_size
        self.pm_iter = pm_iter
        self.n_iter = n_iter
        self.threshold = threshold
        self.nb_core = nb_core

    def _init_multithreading(self):
        set_num_threads(self.nb_core)
        config.NUMBA_NUM_THREADS = self.nb_core

    def _track_core(self, imgs, mask, bar):
        masks = []
        ref = imgs[0]
        m, n, _ = ref.shape
        for img in imgs[1:]:
            if self.monte_carlo:
                nnf = monte_carlo(img, ref, n_iter=self.n_iter, p_size=self.p_size, pm_iter=self.pm_iter, bar=False)
                res = np.zeros((m, n, self.n_iter))
                for k in range(self.n_iter):
                    res[:, :, k] = estimate_mask(mask, nnf[k, :, :, :])

                esti_mask = thresholding(res, self.threshold)
            else:
                nnf = NNS(img, ref, self.p_size, self.pm_iter)
                esti_mask = estimate_mask(mask, nnf)
            masks.append(esti_mask)
            bar.update(1)
        return [mask]+masks

    def track(self, imgs, mask):
        with ProgressBar(total=len(imgs)-1) as progress:
            return self._track_core(imgs, mask, progress)


>>>>>>> 17554a9 (Create new class to track object using PM)
