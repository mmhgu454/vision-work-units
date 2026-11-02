#!/usr/bin/env python
"""
estimate_pitch.py — 用地面的深度估相機的 pitch 與離地高度

在彩色畫面上框一塊**確定是地面**的區域，對那塊的點雲擬合平面，算出：
    pitch  光軸相對水平面的角度，負 = 往下看。可直接當 task1 的 --tilt
    高度   鏡頭到地面平面的距離（公尺）。可直接當 task1 的 --phy-height

算法參考 tpmrt 的 task1_clearance/ground.py：SVD 擬合平面，反覆剔除離群點再重擬。
角度直接在相機原始座標裡算（光軸與地面法向的夾角），所以相機怎麼轉著架都不影響。

用法
----
python estimate_pitch.py                      # 開視窗框選地面，連抓 10 幀取中位數
python estimate_pitch.py -n 30                # 抓 30 幀
python estimate_pitch.py --roi 400,500,900,700  # 不開框選視窗，直接給 x0,y0,x1,y1

解析度跟其他 task 一樣，吃 realsense_source.py 的設定。
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOL_M = 0.03          # 離群門檻：離平面超過 3 cm 的點不算地面（容得下地面起伏，排得掉箱子）
ITERS = 4             # 「擬合 → 剔除離群 → 重擬」的輪數
MAX_POINTS = 60000    # 點太多就隨機抽樣，控制耗時
MIN_POINTS = 500      # 少於這個數就不擬合
RESIDUAL_WARN_M = 0.015


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


def select_roi(img: np.ndarray) -> Tuple[int, int, int, int]:
    import cv2
    win = "框選一塊地面，ENTER 確認（c 取消）"
    x, y, w, h = cv2.selectROI(win, img, showCrosshair=False)
    cv2.destroyWindow(win)
    if w == 0 or h == 0:
        raise SystemExit("沒有框選，結束。")
    return x, y, x + w, y + h


def main() -> int:
    ap = argparse.ArgumentParser(description="用地面深度估相機 pitch 與離地高度")
    ap.add_argument("-n", type=int, default=10, help="抓幾幀取中位數（預設 10）")
    ap.add_argument("--roi", default=None,
                    type=lambda s: tuple(int(v) for v in s.split(",")),
                    help="地面區域 x0,y0,x1,y1（像素）；不給就開視窗框選")
    args = ap.parse_args()

    from realsense_source import RealSenseSource

    with RealSenseSource() as cam:
        color_img, _ = cam.get_frame()
        x0, y0, x1, y1 = args.roi or select_roi(color_img)
        print(f"地面區域 x {x0}~{x1}, y {y0}~{y1}，抓 {args.n} 幀...")

        results = []
        for i in range(args.n):
            _, pc = cam.get_frame()
            # get_frame 套過 R_fix（box_measure 的座標修正），轉回相機原始座標
            raw = pc[y0:y1, x0:x1].reshape(-1, 3) @ cam.R_fix
            fit = fit_plane(raw, seed=i)
            if fit is None:
                print(f"  [{i + 1}] 有效點太少，略過")
                continue
            n, cen, resid, inliers = fit
            pose = pose_from_plane(n, cen)
            pose.update(resid=resid, inliers=inliers)
            results.append(pose)
            print(f"  [{i + 1}] pitch {pose['pitch']:+6.2f}°  高度 {pose['height']:.3f} m  "
                  f"殘差 {resid * 1000:4.1f} mm  內點 {inliers}")

    if not results:
        print("❌ 擬合不出地面——框選區域要是一大片平坦地面，且在深度量得到的範圍內。")
        return 1

    med = {k: statistics.median(r[k] for r in results) for k in results[0]}
    spread = max(r["pitch"] for r in results) - min(r["pitch"] for r in results)
    print("\n" + "-" * 64)
    print(f"  pitch   {med['pitch']:+.2f}°   （負 = 往下看；{len(results)} 幀中位數，全距 {spread:.2f}°）")
    print(f"  高度    {med['height']:.3f} m")
    print(f"  殘差    {med['resid'] * 1000:.1f} mm")
    print(f"  畫面右方相對水平 {med['img_right']:+.1f}°、畫面上方相對水平 {med['img_up']:+.1f}°")
    print("  （相機轉 90° 架時，其中一個應該接近 0°；不接近 0° 代表相機還有側傾）")
    print("-" * 64)
    print(f"  task1 試用：--tilt {med['pitch']:.1f} --phy-height {med['height']:.2f}")
    print("  （task1 現在的預設是 tilt -35.0、高度 2.25，見 pallet_stack.py）")
    if med["resid"] > RESIDUAL_WARN_M:
        print(f"  ⚠ 殘差超過 {RESIDUAL_WARN_M * 1000:.0f} mm，框到的可能不是平坦地面，請重框")
    if spread > 1.0:
        print("  ⚠ pitch 全距超過 1°，結果不穩，建議重跑")
    print("  ⚠ 記得用捲尺量一次鏡頭離地高度，差超過 2 cm 就是擬合到別的平面了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
