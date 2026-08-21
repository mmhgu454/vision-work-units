"""造 pallet_stack 用的合成場景。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pallet_stack as ps

STATION = "g4_1"          # pd=1.05, pw=1.2, ph=0.0, pz=2.1
CARDBOARD = "L554001"     # (l, w, h) = (0.57, 0.50, 0.50)


def uncalibrate(world, calibrate):
    """calibrate 的反函數，用來從世界座標造出「相機原始」點雲。"""
    tilt, phy_height, flip = calibrate.params
    q = world.astype(np.float64).copy()
    q[:, 0] *= -1
    q[:, 1] -= phy_height
    if flip:
        q = (np.linalg.inv(ps.rotation_matrix(0, 0, 180)) @ q.T).T
    return (np.linalg.inv(ps.rotation_matrix(tilt, 0, 0)) @ q.T).T


def make_scene(quad_heights=(0.0, 0.5, 1.0, 1.5), h=120, w=160,
               station=STATION, calibrate=None, catalog=None):
    """
    造一個四象限各自不同高度的場景。

    quad_heights 依序是 (近左, 近右, 遠左, 遠右) 的世界座標高度 (m)。
    回傳 (color_img, pc_np)，pc_np 是**未校正**的 (H, W, 3)。
    """
    calibrate = ps.make_calibrator() if calibrate is None else calibrate
    catalog = ps.DEFAULT_BLOCK_CATALOG if catalog is None else catalog

    pd, pw, ph = catalog[station].block_size
    px, pz = catalog[station].block_bias
    origin_x = px - pw / 2

    xs = np.linspace(origin_x, origin_x + pw, w)
    zs = np.linspace(pz, pz + pd, h)
    ZZ, XX = np.meshgrid(zs, xs, indexing="ij")

    YY = np.full_like(XX, quad_heights[0])
    YY[:h // 2, w // 2:] = quad_heights[1]
    YY[h // 2:, :w // 2] = quad_heights[2]
    YY[h // 2:, w // 2:] = quad_heights[3]

    world = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    pc_np = uncalibrate(world, calibrate).reshape(h, w, 3).astype(np.float32)
    return np.zeros((h, w, 3), dtype=np.uint8), pc_np


def no_detections(_img):
    return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)


def fake_detect(boxes_labels):
    """boxes_labels: [(box, label), ...]，label 0=pallet 1=top 2=face。"""
    boxes = np.asarray([b for b, _ in boxes_labels], dtype=float).reshape(-1, 4)
    labels = np.asarray([l for _, l in boxes_labels], dtype=int)

    def _detect(_img):
        return boxes, labels, np.ones(len(labels))
    return _detect
