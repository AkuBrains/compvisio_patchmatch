import numpy as np

from PIL import Image

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


def mask_on_image(img, mask):
    m = np.max(mask)
    temp = np.copy(img)
    temp1 = m * np.ones(img.shape)
    temp[mask == m] = np.array([m, m, m])
    temp1[mask == m, :] = img[mask == m, :]
    return temp, temp1.astype(np.uint8)


def make_gif(imgs, name):
    frames = [Image.fromarray(img) for img in imgs]
    frame_one = frames[0]
    frame_one.save(f"{name}.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=200, loop=0)

