#!/usr/bin/env python
"""
estimate_pitch.py — 即時顯示相機的 pitch 與離地高度（用地面的深度）

開一個預覽視窗，用滑鼠框一塊**確定是地面**的區域，畫面上就會即時顯示：
    pitch  光軸相對水平面的角度，負 = 往下看。可直接當 task1 的 --tilt
    高度   鏡頭到地面平面的距離（公尺）。可直接當 task1 的 --phy-height（h 切換顯示）
一邊調相機一邊看數字，不用重開。

操作
----
滑鼠拖曳   框選地面（隨時可以重框）
SPACE      把目前數值（最近幾幀的中位數）印到終端機，附 task1 參數
h          顯示 / 隱藏高度
ESC / q    離開

算法參考 tpmrt 的 task1_clearance/ground.py：SVD 擬合平面，反覆剔除離群點再重擬。
角度直接在相機原始座標裡算（光軸與地面法向的夾角），所以相機怎麼轉著架都不影響。

用法
----
python estimate_pitch.py
python estimate_pitch.py --roi 400,500,900,700   # 開場就用這塊當地面（x0,y0,x1,y1）

解析度跟其他 task 一樣，吃 realsense_source.py 的設定。
"""

from __future__ import annotations

import argparse
import statistics
from collections import deque
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOL_M = 0.03          # 離群門檻：離平面超過 3 cm 的點不算地面（容得下地面起伏，排得掉箱子）
ITERS = 4             # 「擬合 → 剔除離群 → 重擬」的輪數
MAX_POINTS = 20000    # 點太多就隨機抽樣，讓每幀都算得完
MIN_POINTS = 500      # 少於這個數就不擬合
RESIDUAL_WARN_M = 0.015
SMOOTH_FRAMES = 15    # 顯示的「中位數」取最近幾幀


def fit_plane(pts: np.ndarray, seed: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
    """
    擬合平面。回傳 (法向, 平面上一點, 殘差標準差 m, 內點數)；點太少回傳 None。
    法向一律指向相機那一側（從地面往上指）。
    """
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.1)]
    if len(pts) < MIN_POINTS:
        return None
    if len(pts) > MAX_POINTS:
        pts = pts[np.random.default_rng(seed).choice(len(pts), MAX_POINTS, replace=False)]

    cen = pts.mean(0)
    n = np.linalg.svd(pts - cen, full_matrices=False)[2][2]
    for _ in range(ITERS):
        keep = np.abs((pts - cen) @ n) < TOL_M
        if keep.sum() < MIN_POINTS:
            break
        cen = pts[keep].mean(0)
        n = np.linalg.svd(pts[keep] - cen, full_matrices=False)[2][2]

    if (0 - cen) @ n < 0:          # 相機在原點；讓法向朝相機那側
        n = -n
    keep = np.abs((pts - cen) @ n) < TOL_M
    resid = float(np.abs((pts[keep] - cen) @ n).std()) if keep.any() else float("nan")
    return n, cen, resid, int(keep.sum())


def pose_from_plane(n: np.ndarray, cen: np.ndarray) -> dict:
    """
    相機原始座標（RealSense：x 右、y 下、z 前）下的地面 → 姿態。
    某個方向 v 相對水平面的仰角 = asin(v · n_up)，負 = 往下。
    """
    elev = lambda v: float(np.degrees(np.arcsin(np.clip(v @ n, -1.0, 1.0))))
    return {
        "pitch": elev(np.array([0.0, 0.0, 1.0])),         # 光軸
        "height": float(abs(cen @ n)),                     # 原點到平面的距離
        "img_right": elev(np.array([1.0, 0.0, 0.0])),      # 畫面右方
        "img_up": elev(np.array([0.0, -1.0, 0.0])),        # 畫面上方
    }


class RoiPicker:
    """滑鼠拖曳框選。拖曳中的框存在 dragging，放開後才換掉 roi。"""

    def __init__(self, roi=None):
        self.roi, self.start, self.dragging = roi, None, None

    def __call__(self, event, x, y, flags, param):
        import cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start, self.dragging = (x, y), None
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            self.dragging = (*self.start, x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x0, y0 = self.start
            box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            if box[2] - box[0] > 10 and box[3] - box[1] > 10:
                self.roi = box
            self.start = self.dragging = None


def draw_panel(view, lines) -> None:
    """左上角半透明底，逐行寫字。lines = [(文字, BGR 顏色)]。"""
    import cv2
    h = 18 + 40 * len(lines)
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (760, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, view, 0.45, 0, dst=view)
    for i, (text, color) in enumerate(lines):
        cv2.putText(view, text, (12, 42 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description="即時顯示相機 pitch 與離地高度")
    ap.add_argument("--roi", default=None,
                    type=lambda s: tuple(int(v) for v in s.split(",")),
                    help="開場就用的地面區域 x0,y0,x1,y1（像素）；之後仍可用滑鼠重框")
    args = ap.parse_args()

    from realsense_source import RealSenseSource

    win = "estimate_pitch — drag: ground ROI  SPACE: print  h: height  ESC: quit"
    picker = RoiPicker(args.roi)
    history = deque(maxlen=SMOOTH_FRAMES)
    show_height, last_roi = True, None
    WHITE, GREEN, YELLOW, RED = (255, 255, 255), (0, 255, 0), (0, 255, 255), (0, 0, 255)

    print("用滑鼠在畫面上框一塊地面。SPACE 印出數值，h 切換高度，ESC 離開。")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, picker)
    with RealSenseSource() as cam:
        frame_i = 0
        while True:
            color_img, pc = cam.get_frame()
            view = color_img.copy()
            lines, pose = [], None

            if picker.roi != last_roi:          # 換了區域，舊的歷史不能混進來
                history.clear()
                last_roi = picker.roi

            if picker.roi is None:
                lines.append(("Drag a box on the GROUND", YELLOW))
            else:
                x0, y0, x1, y1 = picker.roi
                # get_frame 套過 R_fix（box_measure 的座標修正），轉回相機原始座標
                raw = pc[y0:y1, x0:x1].reshape(-1, 3) @ cam.R_fix
                fit = fit_plane(raw, seed=frame_i)
                cv2.rectangle(view, (x0, y0), (x1, y1), GREEN, 2)
                if fit is None:
                    lines.append(("not enough depth in box", RED))
                else:
                    n, cen, resid, inliers = fit
                    pose = pose_from_plane(n, cen)
                    pose["resid"] = resid
                    history.append(pose)
                    med = {k: statistics.median(h[k] for h in history) for k in pose}
                    bad = resid > RESIDUAL_WARN_M
                    lines.append((f"pitch  {pose['pitch']:+6.2f} deg   (median {med['pitch']:+6.2f})",
                                  RED if bad else GREEN))
                    if show_height:
                        lines.append((f"height {pose['height']:.3f} m    (median {med['height']:.3f})",
                                      RED if bad else WHITE))
                    lines.append((f"resid  {resid * 1000:4.1f} mm" + ("  NOT FLAT? re-draw" if bad else ""),
                                  RED if bad else WHITE))

            if picker.dragging:
                x0, y0, x1, y1 = picker.dragging
                cv2.rectangle(view, (x0, y0), (x1, y1), YELLOW, 1)
            draw_panel(view, lines)
            cv2.imshow(win, view)
            frame_i += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("h"):
                show_height = not show_height
            if key == ord(" ") and history:
                med = {k: statistics.median(h[k] for h in history) for k in history[0]}
                spread = max(h["pitch"] for h in history) - min(h["pitch"] for h in history)
                print("-" * 64)
                print(f"  pitch   {med['pitch']:+.2f}°   （最近 {len(history)} 幀中位數，全距 {spread:.2f}°）")
                print(f"  高度    {med['height']:.3f} m   殘差 {med['resid'] * 1000:.1f} mm")
                print(f"  畫面右方相對水平 {med['img_right']:+.1f}°、畫面上方相對水平 {med['img_up']:+.1f}°"
                      "（相機轉 90° 架時其中一個應接近 0°，否則有側傾）")
                print(f"  task1 試用：--tilt {med['pitch']:.1f} --phy-height {med['height']:.2f}"
                      "   （目前預設 tilt -35.0、高度 2.25）")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
