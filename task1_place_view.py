#!/usr/bin/env python
"""
task1_place_view.py — 即時看 task1 的擺放判斷

每幀跑 pallet_stack.build_block_list（與 fgo 原本的 task1 演算法逐格相同），把判斷過程畫出來：
    左：相機畫面
        白框  工作區的 2x2 四格（畫在基準高度上）
        黃框  每格實際取樣的角落區域——程式只看這裡，其他地方的箱子不會被算到
        文字  每格的層數與取樣中位數；NO DATA = 相機看不到，程式會當成 0 層
        綠色立體框  下一箱的位置（選中那格，從目前層高往上一個箱高）
    右：俯視高度圖（下方靠相機），看不到的地方打斜線，選中的格子綠框

只支援 pallet 模式（2x2 四角取樣）。

操作
----
s          把目前畫面存到 fixtures/png/
ESC / q    離開

用法
----
python task1_place_view.py --station g8_1 --cardboard L554001 --tilt -38 --phy-height 2.25

tilt / phy-height 可用 estimate_pitch.py 量。解析度吃 realsense_source.py 的設定。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pallet_stack as psm

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "fixtures" / "png"
PROJ_ERR_MAX_PX = 3.0

WHITE, GREEN, YELLOW, RED = (255, 255, 255), (0, 255, 0), (0, 255, 255), (0, 0, 255)


# ------------------------------------------------------------------ 座標

def world_to_cam(w: np.ndarray, tilt: float, phy_height: float, flip: bool,
                 rotation_deg=(0.0, 0.0, -90.0)) -> np.ndarray:
    """
    世界座標 (N,3) -> 相機原始座標 (N,3)。
    是 realsense_source 的 task1 旋轉 + pallet_stack.make_calibrator 的反運算
    （calibrate：tilt -> flip -> y += h -> x *= -1）。
    """
    p = np.atleast_2d(w).astype(float).copy()
    p[:, 0] *= -1
    p[:, 1] -= phy_height
    if flip:
        p = (psm.rotation_matrix(0, 0, 180).T @ p.T).T
    p = (psm.rotation_matrix(tilt, 0, 0).T @ p.T).T
    return (psm.rotation_matrix(*rotation_deg).T @ p.T).T


def quad_cells(r: int, c: int, n: int):
    """2x2 的第 (r, c) 格在 hmap 上的 (列0, 列1, 欄0, 欄1)。列 0 是遠端。"""
    half = n // 2
    return r * half, (r + 1) * half, c * half, (c + 1) * half


def sample_cells(n: int) -> dict:
    """每格實際取樣的角落區域，與 pallet_stack.build_block_list 相同。"""
    gap, corner = psm.CORNER_GAP_CELLS, max(1, n // 4)
    lo, hi = (gap, corner), (n - corner, n - gap)
    return {(0, 0): (*lo, *lo), (0, 1): (*lo, *hi), (1, 0): (*hi, *lo), (1, 1): (*hi, *hi)}


class Geometry:
    """工作區在世界座標的位置。"""

    def __init__(self, station: str, cardboard: str, n: int = psm.HMAP_BASE_SIZE):
        info = psm.DEFAULT_BLOCK_CATALOG[station]
        self.depth, self.width, self.base = info.block_size
        bias_x, self.z0 = info.block_bias
        self.x0 = bias_x - self.width / 2
        self.n = n
        self.box_h = psm.DEFAULT_BOX_DIMENSIONS[cardboard][2]

    def cells_to_world(self, r0, r1, c0, c1):
        """hmap 列/欄範圍 -> 世界 (x0, x1, z0, z1)。"""
        cw, cd = self.width / self.n, self.depth / self.n
        return (self.x0 + c0 * cw, self.x0 + c1 * cw,
                self.z0 + (self.n - r1) * cd, self.z0 + (self.n - r0) * cd)


def rect(x0, x1, z0, z1, y):
    return np.array([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)])


# ------------------------------------------------------------------ 畫圖

def _text(img, s, org, color, scale=0.7, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_camera_view(color_img, project, geo: Geometry, block, place, medians):
    """project(world (N,3)) -> list of (u, v) 或 None（在鏡頭後面）。"""
    view = color_img.copy()

    def poly(world, color, thick):
        uv = project(world)
        if all(p is not None for p in uv):
            cv2.polylines(view, [np.array(uv, np.int32)], True, color, thick, cv2.LINE_AA)
        return uv

    samples = sample_cells(geo.n)
    for (r, c), cells in samples.items():
        x0, x1, z0, z1 = geo.cells_to_world(*quad_cells(r, c, geo.n))
        poly(rect(x0, x1, z0, z1, geo.base), WHITE, 2)
        poly(rect(*geo.cells_to_world(*cells), geo.base), YELLOW, 2)
        (ctr,) = project(np.array([[(x0 + x1) / 2, geo.base, (z0 + z1) / 2]]))
        if ctr is not None:
            m = medians[(r, c)]
            seen = np.isfinite(m)
            _text(view, f"({r + 1},{c + 1}) L{block[r, c]}  " + (f"{m:+.2f}m" if seen else "NO DATA"),
                  (ctr[0] - 90, ctr[1]), WHITE if seen else RED)

    if place is not None:
        r, c, layer = place[0] - 1, place[1] - 1, place[2]
        x0, x1, z0, z1 = geo.cells_to_world(*quad_cells(r, c, geo.n))
        y0, y1 = geo.base + layer * geo.box_h, geo.base + (layer + 1) * geo.box_h
        bot, top = project(rect(x0, x1, z0, z1, y0)), project(rect(x0, x1, z0, z1, y1))
        if all(p is not None for p in bot + top):
            for face in (bot, top):
                cv2.polylines(view, [np.array(face, np.int32)], True, GREEN, 3, cv2.LINE_AA)
            for a, b in zip(bot, top):
                cv2.line(view, a, b, GREEN, 3, cv2.LINE_AA)
        label = f"NEXT BOX: row {place[0]} col {place[1]} layer {layer + 1}"
    else:
        label = "ALL FULL"
    cv2.rectangle(view, (0, 0), (view.shape[1], 46), (0, 0, 0), -1)
    _text(view, label, (10, 32), GREEN, 0.9)
    return view


def draw_top_view(hmap, geo: Geometry, block, place, medians, size: int):
    valid = np.isfinite(hmap)
    norm = np.clip(np.where(valid, hmap - geo.base, 0) / (psm.MAX_LAYERS * geo.box_h), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = (40, 40, 40)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    unseen = cv2.resize((~valid).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    hatch = np.zeros((size, size), np.uint8)
    for k in range(-size, size, 16):
        cv2.line(hatch, (k, 0), (k + size, size), 1, 2)
    img[(unseen & hatch).astype(bool)] = (90, 90, 200)

    s = size / geo.n
    cv2.line(img, (size // 2, 0), (size // 2, size), WHITE, 2)
    cv2.line(img, (0, size // 2), (size, size // 2), WHITE, 2)
    for (r, c), (r0, r1, c0, c1) in sample_cells(geo.n).items():
        cv2.rectangle(img, (int(c0 * s), int(r0 * s)), (int(c1 * s), int(r1 * s)), YELLOW, 2)
        cx, cy = int((c + 0.5) * size / 2), int((r + 0.5) * size / 2)
        m = medians[(r, c)]
        _text(img, f"L{block[r, c]}", (cx - 25, cy), WHITE, 1.4, 3)
        _text(img, f"{m:+.2f} m" if np.isfinite(m) else "NO DATA", (cx - 55, cy + 36),
              WHITE if np.isfinite(m) else RED)
    if place is not None:
        r, c = place[0] - 1, place[1] - 1
        cv2.rectangle(img, (c * size // 2 + 6, r * size // 2 + 6),
                      ((c + 1) * size // 2 - 6, (r + 1) * size // 2 - 6), GREEN, 6)
    _text(img, "FAR", (size // 2 - 25, 26), WHITE)
    _text(img, "NEAR (camera side)", (size // 2 - 120, size - 12), WHITE)
    _text(img, f"0~{psm.MAX_LAYERS * geo.box_h:.1f} m", (size - 130, 26), WHITE, 0.6)
    return img


def corner_medians(hmap, n):
    return {k: float(np.median(hmap[r0:r1, c0:c1])) for k, (r0, r1, c0, c1) in sample_cells(n).items()}


# ------------------------------------------------------------------ 主程式

def main() -> int:
    ap = argparse.ArgumentParser(description="即時看 task1 的擺放判斷")
    ap.add_argument("--station", default="g8_1", choices=sorted(psm.DEFAULT_BLOCK_CATALOG))
    ap.add_argument("--cardboard", default="L554001", choices=sorted(psm.DEFAULT_BOX_DIMENSIONS))
    ap.add_argument("--tilt", type=float, default=psm.DEFAULT_TILT, help="相機 pitch（度，負 = 往下）")
    ap.add_argument("--phy-height", dest="phy_height", type=float, default=psm.DEFAULT_PHY_HEIGHT,
                    help="鏡頭離地高度（公尺）")
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    import pyrealsense2 as rs
    from realsense_source import RealSenseSource, TASK1_ROTATION_DEG

    flip = not args.no_flip
    calibrate = psm.make_calibrator(tilt=args.tilt, phy_height=args.phy_height, flip=flip)
    geo = Geometry(args.station, args.cardboard)
    win = "task1_place_view - s: save | ESC: quit"     # Qt 後端：視窗名稱只用 ASCII

    print(f"task1 擺放判斷即時預覽：{args.station} / {args.cardboard}  "
          f"tilt {args.tilt}  高度 {args.phy_height}  flip {flip}")
    print("  s：存圖到 fixtures/png/    ESC：離開")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    with RealSenseSource() as cam:
        intr = (cam.pipeline.get_active_profile().get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())

        def project(world):
            out = []
            for p in world_to_cam(world, args.tilt, args.phy_height, flip, TASK1_ROTATION_DEG):
                if p[2] <= 0.05:
                    out.append(None)
                else:
                    u, v = rs.rs2_project_point_to_pixel(intr, [float(p[0]), float(p[1]), float(p[2])])
                    out.append((int(round(u)), int(round(v))))
            return out

        proj_err = None
        while True:
            color_img, pc_np = cam.get_frame_task1()
            (_, block), place, hmap = psm.build_block_list(pc_np, args.cardboard, args.station, calibrate)
            medians = corner_medians(hmap, geo.n)

            if proj_err is None:      # 第一幀：影像上的點 -> 世界 -> 投影回影像，要落回原處
                h, w = pc_np.shape[:2]
                errs = []
                for v in range(60, h, 120):
                    for u in range(60, w, 160):
                        p = pc_np[v, u]
                        if p[2] > 0.1:
                            (uv,) = project(calibrate(p[None]))
                            errs.append(np.hypot(uv[0] - u, uv[1] - v))
                proj_err = max(errs) if errs else float("nan")
                print(f"  投影自我檢查：{len(errs)} 點，最大誤差 {proj_err:.1f} px")

            left = draw_camera_view(color_img, project, geo, block, place, medians)
            if not proj_err <= PROJ_ERR_MAX_PX:
                _text(left, f"PROJECTION CHECK FAILED ({proj_err:.1f} px) - boxes not trustworthy",
                      (10, left.shape[0] - 20), RED)
            view = np.hstack([left, draw_top_view(hmap, geo, block, place, medians, left.shape[0])])
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                path = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_task1_place_{args.station}.png"
                cv2.imwrite(str(path), view)
                print(f"  💾 {path}   block={block.tolist()}  place={place}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
