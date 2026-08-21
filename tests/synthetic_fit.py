"""造 fit_check 用的合成場景。座標系＝task2 契約（Y 上、X 右、Z 前，已校正）。"""

import numpy as np

BASE_H = 0.60          # 目前堆疊表面高度
STEP = 0.005


def make_plane(base_h=BASE_H, x=(-0.55, 0.55), z=(0.35, 1.55), step=STEP,
               obstacle=None, as_image=False):
    """
    造一片水平平面。obstacle=((x0,x1),(z0,z1),height) 會在該範圍墊高。

    as_image=True 時額外回傳一張同像素數的 color_img，讓 base_vis 那條路徑也走得到。
    """
    xs = np.arange(x[0], x[1], step)
    zs = np.arange(z[0], z[1], step)
    ZZ, XX = np.meshgrid(zs, xs, indexing="ij")
    YY = np.full_like(XX, base_h)

    if obstacle:
        (ox0, ox1), (oz0, oz1), oh = obstacle
        m = (XX > ox0) & (XX < ox1) & (ZZ > oz0) & (ZZ < oz1)
        YY[m] = base_h + oh

    pc = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1).astype(np.float32)
    if not as_image:
        return pc

    color = np.zeros((ZZ.shape[0], ZZ.shape[1], 3), dtype=np.uint8)
    return pc, color


def empty_ref_plane():
    """平面存在，但完全避開左站 L554001 的 ref 框（x -0.3~-0.1, z 0.7~0.75）。"""
    return make_plane(x=(0.05, 0.55), z=(0.35, 1.55))
