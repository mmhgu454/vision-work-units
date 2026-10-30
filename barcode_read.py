"""
barcode_read.py — task4：拍 barcode、讀出數值（sample）

輸入一張 BGR 影像，回傳畫面裡所有讀得到的條碼。取幀是外面的事，
所以 RealSense、USB webcam、存好的照片都能餵進來。

解碼器只用 pyzbar（Code128 / Code39 / EAN / UPC / QR 等），沒裝就在 import 時報錯。
不退回 OpenCV 的 cv2.barcode：實拍 Code128 時它會把密條碼硬解成隨機的 UPC_A 數字，
讀錯比讀不到更糟。

前處理（依序，讀到就停）：
    1. 原圖直接解
    2. 整張放大 2 倍 + Otsu 二值化
    3. 找條碼候選區（水平梯度強的長條）→ 依傾角轉正 → 放大 3 倍 → Otsu / 銳化 + Otsu

實拍限制（D435 1080p，15 字元 Code128 實測）：
    - 每條模組寬 < 2 px（條碼寬 < 400 px）讀不到——解析度不夠。
    - 距離 20~25 cm 讀得最穩；更近會失焦（D435 彩色是定焦鏡頭），軟體救不回來。

單獨使用
--------
python barcode_read.py photo.jpg               # 讀圖檔，印出結果
python barcode_read.py photo.jpg --show        # 另外開視窗看框在哪
python barcode_read.py --webcam 0              # USB webcam 即時預覽：SPACE 讀、ESC 離開
（RealSense 走 run_live.py --task 4，畫面中央會顯示距離）
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
except ImportError as e:   # 沒裝 pyzbar，或裝了但找不到系統的 libzbar0
    raise ImportError(
        f"barcode_read 需要 pyzbar：{e}\n"
        "  uv pip install pyzbar          # Python 套件\n"
        "  sudo apt install libzbar0      # 系統函式庫") from e

#: 實測讀得最穩的距離（公尺）。D435 1080p、15 字元 Code128：更遠像素不夠，更近失焦。
GOOD_RANGE_M = (0.20, 0.25)


class Barcode(NamedTuple):
    text: str              # 條碼內容
    kind: str              # 種類，例如 CODE128 / EAN13 / QRCODE
    corners: np.ndarray    # (N, 2) int32，原圖座標，畫框用


# --------------------------------------------------------------------- 解碼器

def _decode(img: np.ndarray, scale: float = 1.0) -> List[Barcode]:
    out = []
    for r in pyzbar.decode(img):
        pts = np.array([(p.x, p.y) for p in r.polygon], dtype=np.float32) / scale
        out.append(Barcode(r.data.decode("utf-8", errors="replace"), r.type,
                           pts.astype(np.int32)))
    return out


def _otsu(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _sharpen(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 2.0, blur, -1.0, 0)


def _candidates(gray: np.ndarray, top: int = 5) -> list:
    """
    找 1D 條碼候選區，回傳 cv2.minAreaRect 列表（大的在前）。
    條碼的特徵是「垂直條紋」：水平梯度強、垂直梯度弱，閉運算把條紋連成一塊。
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.blur(cv2.convertScaleAbs(cv2.subtract(np.abs(gx), np.abs(gy))), (9, 9))
    mask = _otsu(grad)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7)))
    mask = cv2.dilate(cv2.erode(mask, None, iterations=3), None, iterations=3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.minAreaRect(c) for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:top]]


def _deskew(gray: np.ndarray, rect, k: float):
    """把候選區轉正、往外留邊、放大 k 倍。候選區不像條碼（太小或不夠扁長）回傳 None。"""
    (cx, cy), (w, h), ang = rect
    if w < h:
        w, h, ang = h, w, ang - 90
    if w < 60 or w < 2 * h:
        return None
    W, H = int(w * 1.3), int(h * 2.2)
    M = cv2.getRotationMatrix2D((cx, cy), ang, k)
    M[0, 2] += W * k / 2 - cx
    M[1, 2] += H * k / 2 - cy
    return cv2.warpAffine(gray, M, (int(W * k), int(H * k)), flags=cv2.INTER_CUBIC, borderValue=255)


def _decode_candidate(gray: np.ndarray, rect) -> List[Barcode]:
    """轉正後解碼。框直接用候選區的旋轉矩形（解碼器在二值化圖上常只回傳一條線）。"""
    roi = _deskew(gray, rect, k=3)
    if roi is None:
        return []
    for prep in (_otsu, lambda r: _otsu(_sharpen(r))):
        codes = _decode(prep(roi))
        if codes:
            box = cv2.boxPoints(rect).astype(np.int32)
            return [c._replace(corners=box) for c in codes]
    return []


def read_barcodes(color_img: np.ndarray) -> List[Barcode]:
    """讀出影像裡所有條碼。讀不到回傳空 list。同一內容只留一筆。"""
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY) if color_img.ndim == 3 else color_img
    codes = _decode(gray)
    if not codes:
        big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        codes = _decode(_otsu(big), scale=2.0)
    if not codes:
        for rect in _candidates(gray):
            codes = _decode_candidate(gray, rect)
            if codes:
                break
    seen, uniq = set(), []
    for c in codes:
        if (c.text, c.kind) not in seen:
            seen.add((c.text, c.kind))
            uniq.append(c)
    return uniq


# --------------------------------------------------------------------- 距離

def center_distance(pc: np.ndarray, half: int = 20) -> Optional[float]:
    """
    畫面中心 (2*half)x(2*half) 區塊的深度中位數，單位公尺；沒有有效深度回傳 None。
    pc 是與彩色圖對齊的 (H, W, 3) 點雲，第 3 軸是深度。
    """
    h, w = pc.shape[:2]
    z = pc[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half, 2]
    z = z[z > 0.01]
    return float(np.median(z)) if z.size else None


def draw_distance(view: np.ndarray, dist_m: Optional[float]) -> None:
    """在畫面中央畫十字準心與距離。在 GOOD_RANGE_M 內綠色，否則紅色並提示方向。"""
    h, w = view.shape[:2]
    lo, hi = GOOD_RANGE_M
    if dist_m is None:
        color, text = (0, 0, 255), "no depth"
    elif dist_m < lo:
        color, text = (0, 0, 255), f"{dist_m * 100:.1f} cm  too close"
    elif dist_m > hi:
        color, text = (0, 0, 255), f"{dist_m * 100:.1f} cm  too far"
    else:
        color, text = (0, 255, 0), f"{dist_m * 100:.1f} cm  OK"
    cv2.drawMarker(view, (w // 2, h // 2), color, cv2.MARKER_CROSS, 40, 2)
    cv2.putText(view, text, (w // 2 + 30, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


# --------------------------------------------------------------------- 畫圖 / 輸出

def draw(color_img: np.ndarray, codes: List[Barcode]) -> np.ndarray:
    view = color_img.copy()
    for c in codes:
        cv2.polylines(view, [c.corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        x, y = c.corners.min(axis=0)
        cv2.putText(view, f"{c.kind}: {c.text}", (int(x), max(20, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return view


def describe(codes: List[Barcode]) -> None:
    print("-" * 64)
    if not codes:
        print("  ❌ 沒讀到條碼")
    for i, c in enumerate(codes, 1):
        print(f"  [{i}] {c.kind:<8} {c.text}")
    print("-" * 64)


def save_frame(img: np.ndarray, save_dir: Path, tag: str) -> Path:
    """存原始畫面（未畫框），之後可以用 barcode_read.py <檔案> 離線重讀。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{tag}.png"
    cv2.imwrite(str(path), img)
    print(f"  💾 已存 {path}")
    return path


def preview(grab, save_dir: Optional[Path] = None,
            win: str = "task4 barcode — SPACE:辨識  s:存圖  ESC:離開") -> int:
    """
    即時預覽。grab() 每次回傳 (BGR 影像, 中心距離公尺 或 None)；
    距離是 None 時（例如 webcam）不畫準心。平常只顯示畫面；按 SPACE 才解碼當下那幀，結果印到終端機並疊在畫面上，
    直到下一次 SPACE。有給 save_dir 時，SPACE 會順便存原始畫面（_ok / _fail），
    s 只存不辨識。
    """
    lo, hi = GOOD_RANGE_M
    print(f"\ntask4 即時預覽（條碼對準畫面中央十字，距離 {lo * 100:.0f}~{hi * 100:.0f} cm 時綠色）")
    print("  SPACE：辨識" + ("並存圖   s：只存圖" if save_dir else "") + "   ESC / q：離開\n")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    codes, status = [], "SPACE: read barcode"
    try:
        while True:
            img, dist_m = grab()
            view = draw(img, codes)
            if dist_m is not None or getattr(grab, "has_depth", False):
                draw_distance(view, dist_m)
            cv2.putText(view, status, (10, view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord(" "):
                codes = read_barcodes(img)
                describe(codes)
                status = " | ".join(c.text for c in codes) if codes else "NOT FOUND"
                if save_dir:
                    save_frame(img, save_dir, "ok" if codes else "fail")
            if key == ord("s") and save_dir:
                save_frame(img, save_dir, "raw")
    finally:
        cv2.destroyAllWindows()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="task4：讀條碼（圖檔或 webcam）")
    ap.add_argument("images", nargs="*", help="要讀的圖檔")
    ap.add_argument("--show", action="store_true", help="讀圖檔時開視窗顯示框")
    ap.add_argument("--webcam", type=int, metavar="INDEX", help="改用 USB webcam 即時預覽")
    args = ap.parse_args(argv)

    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            print(f"❌ 開不了 webcam {args.webcam}")
            return 2

        def grab():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("webcam 取不到影像")
            return frame, None
        try:
            return preview(grab, Path("fixtures/barcode"))
        finally:
            cap.release()

    if not args.images:
        ap.print_help()
        return 2

    rc = 0
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 讀不到圖檔 {path}")
            rc = 2
            continue
        print(f"\n▶ {path}")
        codes = read_barcodes(img)
        describe(codes)
        rc = max(rc, 0 if codes else 1)
        if args.show:
            cv2.imshow(path, draw(img, codes))
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    return rc


if __name__ == "__main__":
    sys.exit(main())
