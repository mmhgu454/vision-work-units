"""造合成場景用的工具。不依賴硬體，也不依賴模型。"""

from typing import Tuple
import numpy as np

TOP_ROWS = (40, 100)
TOP_COLS = (100, 220)
FACE_ROWS = (100, 160)
FACE_COLS = (100, 220)


def make_box_scene(L=0.60, W=0.35, H=0.23, img_h=200, img_w=320,
                   y_top=-1.0, z0=1.0, roll_deg=0.0):
    """
    在 box_measure 的契約座標系裡造一個已知尺寸的箱子。

    回傳 (color_img, pc_np, top_box, face_box)。
    頂面法向量為 (0,-1,0)；roll_deg 不為 0 時繞 Z 軸轉，用來觸發契約檢查。
    """
    pc = np.zeros((img_h, img_w, 3), dtype=np.float32)
    tr, tc = slice(*TOP_ROWS), slice(*TOP_COLS)
    fr, fc = slice(*FACE_ROWS), slice(*FACE_COLS)

    nx = tc.stop - tc.start
    xs = np.linspace(-W / 2, W / 2, nx)

    # 頂面：x 跨 W、z 跨 L、y 固定
    zs = np.linspace(z0, z0 + L, tr.stop - tr.start)
    ZZ, XX = np.meshgrid(zs, xs, indexing="ij")
    pc[tr, tc, 0] = XX
    pc[tr, tc, 1] = y_top
    pc[tr, tc, 2] = ZZ

    # 正面：x 跨 W、y 跨 H（離頂面平面 0..H）、z 固定在前緣
    ys = np.linspace(y_top, y_top + H, fr.stop - fr.start)
    YY, XX2 = np.meshgrid(ys, xs, indexing="ij")
    pc[fr, fc, 0] = XX2
    pc[fr, fc, 1] = YY
    pc[fr, fc, 2] = z0

    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        flat = pc.reshape(-1, 3)
        mask = flat[:, 2] > 0.01          # 只轉有效點，背景維持 0
        flat[mask] = flat[mask] @ Rz.T
        pc = flat.reshape(img_h, img_w, 3)

    color_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    top_box = [TOP_COLS[0], TOP_ROWS[0], TOP_COLS[1], TOP_ROWS[1]]
    face_box = [FACE_COLS[0], FACE_ROWS[0], FACE_COLS[1], FACE_ROWS[1]]
    return color_img, pc, top_box, face_box


def fake_detect(top_box, face_box, extra=()):
    """回傳一個假的 detect callable，框固定、方向與傳入影像相同。"""
    boxes = [top_box, face_box, *(b for b, _ in extra)]
    labels = [1, 2, *(l for _, l in extra)]

    def _detect(color_img):
        return (np.asarray(boxes, dtype=float),
                np.asarray(labels, dtype=int),
                np.ones(len(labels), dtype=float))
    return _detect
