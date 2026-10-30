"""barcode_read 的單元測試。不需要相機。pyzbar 沒裝時只測 OpenCV 那條路。"""

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


@pytest.fixture(params=["pyzbar", "opencv"])
def backend(request, monkeypatch):
    """兩個解碼器各跑一次；pyzbar 不在就跳過那一輪。"""
    if request.param == "pyzbar" and br.pyzbar is None:
        pytest.skip("沒裝 pyzbar / libzbar0")
    if request.param == "opencv":
        monkeypatch.setattr(br, "pyzbar", None)
    return request.param


def test_讀得出EAN13(backend):
    codes = br.read_barcodes(make_ean13_image(DIGITS))
    assert [c.text for c in codes] == [EXPECTED]
    assert codes[0].backend == backend
    assert "EAN" in codes[0].kind.upper()


def test_框落在條碼上(backend):
    img = make_ean13_image(DIGITS)
    (code,) = br.read_barcodes(img)
    cx, cy = code.corners.mean(axis=0)
    H, W = img.shape[:2]
    assert abs(cx - W / 2) < W * 0.15 and abs(cy - H / 2) < H * 0.15


def test_空白畫面回傳空list(backend):
    assert br.read_barcodes(np.full((480, 640, 3), 255, np.uint8)) == []


def test_有pyzbar時不退回OpenCV(monkeypatch):
    """實拍 Code128 時 OpenCV 會把密條碼硬解成隨機 UPC_A，所以有 pyzbar 就不能用它。"""
    if br.pyzbar is None:
        pytest.skip("沒裝 pyzbar / libzbar0")
    def boom(*a, **k):
        raise AssertionError("不該呼叫 OpenCV 解碼器")
    monkeypatch.setattr(br, "_decode_opencv", boom)
    assert br.read_barcodes(np.full((480, 640, 3), 255, np.uint8)) == []


def test_draw不改原圖():
    img = make_ean13_image(DIGITS)
    before = img.copy()
    view = br.draw(img, br.read_barcodes(img))
    assert np.array_equal(img, before) and not np.array_equal(view, before)
