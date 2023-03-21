"""
This module contains the functions related the the mask propagation and the
evaluation of the estimated masks.

Authors: Franck UX, Nampoina RAVELOMANANA and Selman SEZGIN
"""


from numba import njit, prange, set_num_threads, config
from numba_progress import ProgressBar
import numpy as np
from PatchMatch import NNF
from skimage.measure import regionprops

set_num_threads(3)
config.NUMBA_NUM_THREADS = 3


def estimate_mask(mask, f):
<<<<<<< HEAD
    """Propagate the given mask according to the NNF.
    
    Parameters
    ----------
    mask : array-like
        Binary mask.
    f : array-like
        Nearest-Neighbor Field.

    Returns
    -------
    res : array-like
        Binary mask, result of the propagation.
=======
    """
    Predict current mask using reference mask, and optical flow from current image to reference image
    Arguments:
        mask (ndarray): binary reference mask image.
        f (ndarray)   : array containing the optical flow
    Returns:
        res (ndarray) : Current predicted mask
>>>>>>> f791f49 (get metrics rhino)
    """
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
    """
    Compute several optical flow from the current image to reference image
    Arguments:
        img (ndarray)   : current image
        ref (ndarray)   : reference image
        n_iter (int)    : number of iterations
    Returns
        temp (ndarray)  : several optical flows
    """
    for k in prange(n_iter):
        #Compute optical flow at each iteration
        temp[k, :, :, :] = NNF(img, ref, p_size, pm_iter)
        if bar is not None:
            bar.update(1)
    return temp


def monte_carlo(img, ref, n_iter=5, p_size=9, pm_iter=10, bar=False):
    """
    Same function as monte_carlo_core but with a visual progression bar
    """
    k, l, _ = img.shape
    temp = np.empty((n_iter, k, l, 2), dtype=np.int32)
    if bar:
        with ProgressBar(total=n_iter) as progress:
            return progress, monte_carlo_core(img, ref, n_iter, progress, temp, p_size=p_size, pm_iter=pm_iter)
    return monte_carlo_core(img, ref, n_iter, temp, p_size=p_size, pm_iter=pm_iter)


def dice_assessment(groundtruth, estimated, label=255):
    """
    Computes dice score =  2|X.Y| / (|X| + |Y|)
    Arguments:
        groundtruth (ndarray): binary grountruth mask image.
        estimated(ndarray)   : binary predicted mask image.
    Returns:
        DICE (float): dice score between 0 and 100
    """
    A = groundtruth == label
    B = estimated == label
    TP = len(np.nonzero(A*B)[0])
    FN = len(np.nonzero(A*(~B))[0])
    FP = len(np.nonzero((~A)*B)[0])
    DICE = 0
    if (FP+2*TP+FN) != 0:
        DICE = float(2)*TP/(FP+2*TP+FN)
    return DICE*100

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F*100.

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width   : Width of desired bmap  <= seg.shape[1]
        height  : Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.
    """

    seg = seg.astype(bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
        'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+np.floor((y-1)+height / h)
                    i = 1+np.floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap

def centroid_assessment(groundtruth,estimated):
    """
    Computes distances between centroids of the groundtruth and estimated binary mask
    Arguments:
        groundtruth (ndarray)   : binary groundtruth mask
        estimated (ndarray)     : binary estimated mask
    Returns:
        (float)                 : distances between centroids
    """
    a = regionprops(groundtruth)
    b = regionprops(estimated)
    return np.linalg.norm(np.array(a[0].centroid)-np.array(b[0].centroid))