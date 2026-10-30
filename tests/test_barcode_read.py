"""barcode_read 的單元測試。不需要相機。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import barcode_read as br
from synthetic_barcode import make_ean13_image, ean13_check_digit

DIGITS = "471234567890"
EXPECTED = DIGITS + ean13_check_digit(DIGITS)


def test_讀得出EAN13():
    codes = br.read_barcodes(make_ean13_image(DIGITS))
    assert [c.text for c in codes] == [EXPECTED]
    assert "EAN" in codes[0].kind.upper()


def test_框落在條碼上():
    img = make_ean13_image(DIGITS)
    (code,) = br.read_barcodes(img)
    cx, cy = code.corners.mean(axis=0)
    H, W = img.shape[:2]
    assert abs(cx - W / 2) < W * 0.15 and abs(cy - H / 2) < H * 0.15


def test_空白畫面回傳空list():
    assert br.read_barcodes(np.full((480, 640, 3), 255, np.uint8)) == []


def test_draw不改原圖():
    img = make_ean13_image(DIGITS)
    before = img.copy()
    view = br.draw(img, br.read_barcodes(img))
    assert np.array_equal(img, before) and not np.array_equal(view, before)


def test_斜放的條碼靠轉正讀得到():
    img = make_ean13_image(DIGITS, module_px=3)
    M = cv2.getRotationMatrix2D((320, 240), 12, 1.0)
    tilted = cv2.warpAffine(img, M, (640, 480), borderValue=(255, 255, 255))
    assert [c.text for c in br.read_barcodes(tilted)] == [EXPECTED]


def test_中心距離取中位數並忽略無效深度():
    pc = np.zeros((100, 100, 3), np.float32)
    pc[40:60, 40:60, 2] = 0.23
    pc[50, 50, 2] = 5.0                 # 一個離群值不影響中位數
    assert br.center_distance(pc, half=10) == pytest.approx(0.23)
    assert br.center_distance(np.zeros((100, 100, 3), np.float32)) is None
