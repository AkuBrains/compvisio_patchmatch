import numpy as np
from functions import monte_carlo, estimate_mask


def get_masks_direct(img_ref, mask_ref, imgs, n_iter_mc, p_size, n_iter_pm, thr=100):
    """Compute the estimation of the masks with direct integration.

    Direct integration means that the masks are propagated always according to
    the initial mask.
    
    Parameters
    ----------
    img_ref : array-like
        Reference image.
    mask_ref : array-like
        Reference binary mask.
    imgs : list(array-like)
        The images for which we want to estimate the mask.
    n_iter_mc : int
        Monte-Carlo number of iterations.
    p_size : int
        Patch size.
    n_iter_pm : int
        PatchMatch number of iterations.
    thr : float, optional
        Threshold applied to the mean mask compute by the Monte-Carlo
        experiments. Default 100.

    Returns
    -------
    estimated_masks : array-like
        The estimations of the masks.
    """
    n, m = img_ref.shape[0], img_ref.shape[1] # Dimension of the images/masks
    n_images = len(imgs) # Number of masks we have to compute
    estimated_masks = np.zeros((n_images, n, m))
    for i in range(n_images):
        print(f'Mask estimation image {i+2}')
        # Compute n_iter_mc approximations of the NNF
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        # For each NNF, compute the binary mask
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j])
            estimated_masks[i,...] += mask_i
        # Take the mean of the masks
        estimated_masks[i] /= n_iter_mc
        # Apply threshold
        estimated_masks[i][estimated_masks[i] < thr] = 0
        
    return estimated_masks.astype(np.int32)


def get_masks_sequential(img_ref, mask_ref, imgs, n_iter_mc, p_size, n_iter_pm, thr=100):
    """Compute the estimation of the masks with sequential integration.

    Sequential integration means that the masks are propagated always according
    to the previous computed mask.
    
    Parameters
    ----------
    img_ref : array-like
        Reference image.
    mask_ref : array-like
        Reference binary mask.
    imgs : list(array-like)
        The images for which we want to estimate the mask.
    n_iter_mc : int
        Monte-Carlo number of iterations.
    p_size : int
        Patch size.
    n_iter_pm : int
        PatchMatch number of iterations.
    thr : float, optional
        Threshold applied to the mean mask compute by the Monte-Carlo
        experiments. Default 100.

    Returns
    -------
    estimated_masks : array-like
        The estimations of the masks.
    """
    # Dimension of the images/masks
    n, m = img_ref.shape[0], img_ref.shape[1]
    # Number of masks we have to compute
    n_images = len(imgs)
    # estimated_masks[i] is the approximated mask if image imgs[i]
    estimated_masks = np.zeros((n_images, n, m))
    for i in range(n_images):
        print(f'Mask estimation image {i+2}')
        # Compute n_iter_mc approximations of the NNF
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        # For each NNF, compute the binary mask
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j,...])
            estimated_masks[i,...] += mask_i
        # Take the mean of the masks
        estimated_masks[i] /= n_iter_mc
        # Apply threshold
        estimated_masks[i][estimated_masks[i] < thr] = 0
        # Update the reference image and mask
        img_ref = np.copy(imgs[i])
        mask_ref = np.copy(estimated_masks[i])

    return estimated_masks.astype(np.int32)


def get_masks_hybrid(img_ref, mask_ref, imgs, n_iter_mc, p_size, n_iter_pm, step=5, thr=100):
    """Compute the estimation of the masks with hybrid integration.

    Hybrid integration means that the reference image and mask are updated
    only every a given number of steps. It can be seen as a compromise between
    direct and sequential integration.
    
    Parameters
    ----------
    img_ref : array-like
        Reference image.
    mask_ref : array-like
        Reference binary mask.
    imgs : list(array-like)
        The images for which we want to estimate the mask.
    n_iter_mc : int
        Monte-Carlo number of iterations.
    p_size : int
        Patch size.
    n_iter_pm : int
        PatchMatch number of iterations.
    step : int, optional
        Number of steps between two updates of the reference image and mask.
        Default 5.
    thr : float, optional
        Threshold applied to the mean mask compute by the Monte-Carlo
        experiments. Default 100.

    Returns
    -------
    estimated_masks : array-like
        The estimations of the masks.
    """
    # Dimension of the images/masks
    n, m = img_ref.shape[0], img_ref.shape[1]
    # Number of masks we have to compute
    n_images = len(imgs)
    # estimated_masks[i] is the approximated mask if image imgs[i]
    estimated_masks = np.zeros((n_images, n, m))
    for i in range(n_images):
        print(f'Mask estimation image {i+2}')
        # Compute n_iter_mc approximations of the NNF
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        # For each NNF, compute the binary mask
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j,...])
            estimated_masks[i,...] += mask_i
        # Take the mean of the masks
        estimated_masks[i] /= n_iter_mc
        # Apply threshold
        estimated_masks[i][estimated_masks[i] < thr] = 0
        # Update the reference image and mask every given number of steps
        if i%step==0 and i!=0:
            img_ref = np.copy(imgs[i])
            mask_ref = np.copy(estimated_masks[i])

    return estimated_masks.astype(np.int32)