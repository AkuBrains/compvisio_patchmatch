import numpy as np

def estimate_mask(mask, f):
    res = np.zeros(mask.shape)
    m = np.max(mask)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if mask[i, j] != 0:
                res[f[i,j][0], f[i,j][1]] = m
    return res