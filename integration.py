"""
This module implements the three integration methods we proposed to perform
object tracking. These methods are
    - direct integration where the reference is always the first image,
    - sequential integration where the reference image is updated at each step,
    and is set to the last processed image,
    - hybrid integration, which is a mix between direct and sequential
    integration, where the reference image is updated at every given number
    of steps.

These methods are applied and compared quantitatively in the three provided
notebooks.

Authors: Franck UX, Nampoina RAVELOMANANA and Selman SEZGIN
"""


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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        print("Mask estimation for ",i)
=======
        print(f'Mask estimation image {i+2}')
<<<<<<< HEAD
>>>>>>> 782d870 (correct merge problems)
=======
        # Compute n_iter_mc approximations of the NNF
>>>>>>> e65f2c0 (bear example)
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
=======
        print(f"Mask estimation for {name+'-%0*d.bmp'%(3, i+2)}")
=======
        print(f'Mask estimation image {i+2}')
>>>>>>> 186755d (Update integration)
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j])
            estimated_masks[i] += mask_i
        estimated_masks[i] /= n_iter_mc
        estimated_masks[i][estimated_masks[i] < thr] = 0
    return estimated_masks
>>>>>>> e91941d (Hyprid integration added)


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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        print("Mask estimation for ",i)
=======
        print(f"Mask estimation for {name+'-%0*d.bmp'%(3, i+2)}")
>>>>>>> e91941d (Hyprid integration added)
=======
        print(f'Mask estimation image {i+2}')
>>>>>>> 186755d (Update integration)
=======
        print(f'Mask estimation image {i+2}')
<<<<<<< HEAD
>>>>>>> 782d870 (correct merge problems)
=======
        # Compute n_iter_mc approximations of the NNF
>>>>>>> e65f2c0 (bear example)
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
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> e65f2c0 (bear example)
    return estimated_masks.astype(np.int32)
=======
    return estimated_masks
>>>>>>> e91941d (Hyprid integration added)


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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        print("Mask estimation for ",i)
=======
        print(f"Mask estimation for {name+'-%0*d.bmp'%(3, i+2)}")
>>>>>>> e91941d (Hyprid integration added)
=======
        print(f'Mask estimation image {i+2}')
>>>>>>> 186755d (Update integration)
=======
        print(f'Mask estimation image {i+2}')
<<<<<<< HEAD
>>>>>>> 782d870 (correct merge problems)
=======
        # Compute n_iter_mc approximations of the NNF
>>>>>>> e65f2c0 (bear example)
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        # For each NNF, compute the binary mask
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j,...])
            estimated_masks[i,...] += mask_i
        # Take the mean of the masks
        estimated_masks[i] /= n_iter_mc
        # Apply threshold
        estimated_masks[i][estimated_masks[i] < thr] = 0
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> e91941d (Hyprid integration added)
=======
        # Update the reference image and mask every given number of steps
>>>>>>> e65f2c0 (bear example)
        if i%step==0 and i!=0:
            img_ref = np.copy(imgs[i])
            mask_ref = np.copy(estimated_masks[i])

<<<<<<< HEAD
    return estimated_masks.astype(np.int32)
=======
    return estimated_masks
>>>>>>> e91941d (Hyprid integration added)
