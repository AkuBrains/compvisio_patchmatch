import numpy as np
from PIL import Image
from numba import njit


@njit(nogil=True)
def nan_to_num(temp):
    t = np.zeros(temp.shape, dtype=np.int32)
    m, n, o = temp.shape
    for k in range(m):
        for i in range(n):
            for j in range(o):
                if temp[k, i, j] == np.nan:
                    t[k, i, j] = 0
                else:
                    t[k, i, j] = temp[k, i, j]
    return t


@njit(nogil=True)
def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    patch_a = A_padding[a[0]:a[0] + p_size, a[1]:a[1] + p_size, :]
    patch_b = B[b[0] - p:b[0] + p + 1, b[1] - p:b[1] + p + 1, :]
    temp = patch_b - patch_a
    num = np.sum(1 - (np.isnan(temp).astype(np.int32)))
    t = nan_to_num(temp)
    dist = np.sum(np.square(t)) / num
    return dist


@njit(nogil=True)
def reconstruction(f, A, B):
    A_h, A_w = A.shape
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_w):
            temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]
    Image.fromarray(temp).show()


@njit(nogil=True)
def initialization(A, B, p_size):
    A_h, A_w, _ = A.shape
    B_h, B_w, _ = B.shape
    p = p_size // 2
    random_B_r = np.random.randint(p, B_h - p, (A_h, A_w))
    random_B_c = np.random.randint(p, B_w - p, (A_h, A_w))
    A_padding = np.ones((A_h + p * 2, A_w + p * 2, 3)) * np.nan
    A_padding[p:A_h + p, p:A_w + p, :] = A
    f = np.zeros((A_h, A_w, 2), dtype=np.int32)
    dist = np.zeros((A_h, A_w))
    for i in range(A_h):
        for j in range(A_w):
            a = np.array([i, j])
            b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
            f[i, j, :] = b
            dist[i, j] = cal_distance(a, b, A_padding, B, p_size)
    return f, dist, A_padding


@njit(nogil=True)
def propagation(f, a, dist, A_padding, B, p_size, is_odd):
    A_h, A_w, _ = A_padding.shape
    A_h = A_h - p_size + 1
    A_w = A_w - p_size + 1
    x = a[0]
    y = a[1]
    if is_odd:
        d_left = dist[max(x - 1, 0), y]
        d_up = dist[x, max(y - 1, 0)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        if idx == 1:
            f[x, y, :] = f[max(x - 1, 0), y, :]
            dist[x, y] = cal_distance(a, f[x, y, :], A_padding, B, p_size)
        if idx == 2:
            f[x, y, :] = f[x, max(y - 1, 0), :]
            dist[x, y] = cal_distance(a, f[x, y, :], A_padding, B, p_size)
    else:
        d_right = dist[min(x + 1, A_h - 1), y]
        d_down = dist[x, min(y + 1, A_w - 1)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y, :] = f[min(x + 1, A_h - 1), y, :]
            dist[x, y] = cal_distance(a, f[x, y, :], A_padding, B, p_size)
        if idx == 2:
            f[x, y, :] = f[x, min(y + 1, A_w - 1), :]
            dist[x, y] = cal_distance(a, f[x, y, :], A_padding, B, p_size)


@njit(nogil=True)
def random_search(f, a, dist, A_padding, B, p_size, alpha=0.5):
    x = a[0]
    y = a[1]
    B_h, B_w, _ = B.shape
    p = p_size // 2
    i = 4
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y, 0]
    b_y = f[x, y, 1]
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h - p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        search_min_c = max(b_y - search_w, p)
        search_max_c = min(b_y + search_w, B_w - p)
        random_b_y = np.random.randint(search_min_c, search_max_c)
        search_h = B_h * alpha ** i
        search_w = B_w * alpha ** i
        b = np.array([random_b_x, random_b_y], dtype=np.int32)
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y]:
            dist[x, y] = d
            f[x, y, :] = b
        i += 1


@njit(nogil=True)
def NNS(img, ref, p_size, itr):
    A_h, A_w, _ = img.shape
    f, dist, img_padding = initialization(img, ref, p_size)
    for itr in range(1, itr + 1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_w - 1, -1, -1):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, False)
                    random_search(f, a, dist, img_padding, ref, p_size)
        else:
            for i in range(A_h):
                for j in range(A_w):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, True)
                    random_search(f, a, dist, img_padding, ref, p_size)
        # print(f"Simulation: {sim} - iteration: {itr}")
    return f
