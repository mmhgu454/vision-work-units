"""
barcode_read.py — task4：拍 barcode、讀出數值（sample）

輸入一張 BGR 影像，回傳畫面裡所有讀得到的條碼。取幀是外面的事，
所以 RealSense、USB webcam、存好的照片都能餵進來。

解碼器（依序嘗試，先成功的為準）：
    1. pyzbar      — 支援 Code128 / Code39 / EAN / UPC / QR 等，較耐用。需要系統的 libzbar0。
    2. cv2.barcode — OpenCV 內建，只認 EAN / UPC 這類 1D 碼。pyzbar 不在時的退路。

讀不到時會再試一次「放大 2 倍 + 灰階 + Otsu 二值化」的版本，條碼拍得小或光線不均時有用。

單獨使用
--------
python barcode_read.py photo.jpg               # 讀圖檔，印出結果
python barcode_read.py photo.jpg --show        # 另外開視窗看框在哪
python barcode_read.py --webcam 0              # USB webcam 即時預覽：SPACE 讀、ESC 離開
（RealSense 走 run_live.py --task 4）
"""

from __future__ import annotations

import argparse
import sys
from typing import List, NamedTuple, Optional

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
except Exception:          # 沒裝 pyzbar 或找不到 libzbar0
    pyzbar = None


class Barcode(NamedTuple):
    text: str              # 條碼內容
    kind: str              # 種類，例如 CODE128 / EAN13 / QRCODE
    corners: np.ndarray    # (N, 2) int32，原圖座標，畫框用
    backend: str           # 哪個解碼器讀到的：pyzbar / opencv


def backend_name() -> str:
    return "pyzbar" if pyzbar is not None else "opencv"


# --------------------------------------------------------------------- 解碼器

def _decode_pyzbar(img: np.ndarray, scale: float) -> List[Barcode]:
    out = []
    for r in pyzbar.decode(img):
        pts = np.array([(p.x, p.y) for p in r.polygon], dtype=np.float32) / scale
        out.append(Barcode(r.data.decode("utf-8", errors="replace"), r.type,
                           pts.astype(np.int32), "pyzbar"))
    return out


_cv_detector = None


def _decode_opencv(img: np.ndarray, scale: float) -> List[Barcode]:
    global _cv_detector
    if _cv_detector is None:
        _cv_detector = cv2.barcode_BarcodeDetector()
    ok, texts, kinds, points = _cv_detector.detectAndDecodeWithType(img)
    if not ok or points is None:
        return []
    out = []
    for text, kind, pts in zip(texts, kinds, points):
        if text:                                   # 有偵測到框但沒解出來的會是空字串
            out.append(Barcode(text, str(kind), (pts / scale).astype(np.int32), "opencv"))
    return out


def _decode(img: np.ndarray, scale: float = 1.0) -> List[Barcode]:
    codes = _decode_pyzbar(img, scale) if pyzbar is not None else []
    return codes or _decode_opencv(img, scale)


def _enhance(color_img: np.ndarray) -> np.ndarray:
    """放大 2 倍 + 灰階 + Otsu 二值化。"""
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY) if color_img.ndim == 3 else color_img
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def read_barcodes(color_img: np.ndarray) -> List[Barcode]:
    """讀出影像裡所有條碼。讀不到回傳空 list。同一內容只留一筆。"""
    codes = _decode(color_img) or _decode(_enhance(color_img), scale=2.0)
    seen, uniq = set(), []
    for c in codes:
        if (c.text, c.kind) not in seen:
            seen.add((c.text, c.kind))
            uniq.append(c)
    return uniq


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
        print(f"  [{i}] {c.kind:<8} {c.text}    ({c.backend})")
    print("-" * 64)


def preview(grab, win: str = "task4 barcode — SPACE:讀取  ESC:離開") -> int:
    """
    即時預覽。grab() 每次回傳一張 BGR 影像。
    平常只顯示畫面；按 SPACE 才解碼當下那幀，結果印到終端機並疊在畫面上，
    直到下一次 SPACE。
    """
    print(f"\ntask4 即時預覽（解碼器：{backend_name()}）")
    print("  SPACE：辨識   ESC / q：離開\n")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    codes, status = [], "SPACE: read barcode"
    try:
        while True:
            img = grab()
            view = draw(img, codes)
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
            return frame
        try:
            return preview(grab)
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
