import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from functions import monte_carlo, estimate_mask, dice_assessment

def get_masks_direct(img_ref, mask_ref, imgs, n_iter_mc, p_size, n_iter_pm, thr=100):
    n, m = img_ref.shape[0], img_ref.shape[1] # Dimension of the images/masks
    n_images = len(imgs)
    estimated_masks = np.zeros((n_images, n, m))
    for i in range(n_images):
<<<<<<< HEAD
        print("Mask estimation for ",i)
=======
        print(f'Mask estimation image {i+2}')
>>>>>>> a9dc941d7e26ef332782b89203f4a0d81bc55733
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j])
            estimated_masks[i,...] += mask_i
        estimated_masks[i] /= n_iter_mc
        estimated_masks[i][estimated_masks[i] < thr] = 0
    return estimated_masks.astype(np.int32)

def get_masks_sequential(img_ref, mask_ref, imgs, n_iter_mc, p_size, n_iter_pm, thr=100):
    n, m = img_ref.shape[0], img_ref.shape[1] # Dimension of the images/masks
    n_images = len(imgs)
    estimated_masks = np.zeros((n_images, n, m))
    for i in range(n_images):
<<<<<<< HEAD
        print("Mask estimation for ",i)
=======
        print(f'Mask estimation image {i+2}')
>>>>>>> a9dc941d7e26ef332782b89203f4a0d81bc55733
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j,...])
            estimated_masks[i,...] += mask_i
        estimated_masks[i] /= n_iter_mc
        estimated_masks[i][estimated_masks[i] < thr] = 0
        img_ref = np.copy(imgs[i])
        mask_ref = np.copy(estimated_masks[i])
    return estimated_masks.astype(np.int32)

def get_masks_hybrid(img_ref, mask_ref, imgs, n_iter_mc, p_size, n_iter_pm, step = 5, thr=100):
    n, m = img_ref.shape[0], img_ref.shape[1] # Dimension of the images/masks
    n_images = len(imgs)
    estimated_masks = np.zeros((n_images, n, m))
    for i in range(n_images):
<<<<<<< HEAD
        print("Mask estimation for ",i)
=======
        print(f'Mask estimation image {i+2}')
>>>>>>> a9dc941d7e26ef332782b89203f4a0d81bc55733
        f_monte_carlo = monte_carlo(imgs[i], img_ref, n_iter=n_iter_mc, p_size=p_size, pm_iter=n_iter_pm)
        for j in range(n_iter_mc):
            mask_i = estimate_mask(mask_ref, f_monte_carlo[j,...])
            estimated_masks[i,...] += mask_i
        estimated_masks[i] /= n_iter_mc
        estimated_masks[i][estimated_masks[i] < thr] = 0
        if i%step==0 and i!=0:
            img_ref = np.copy(imgs[i])
            mask_ref = np.copy(estimated_masks[i])

    return estimated_masks.astype(np.int32)