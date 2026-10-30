"""
task4 的合成場景：用純 numpy 畫一張 EAN-13 條碼圖，不需要相機、不需要條碼產生器套件。

EAN-13 規格：左護線 101、左 6 碼（依首碼決定 L/G 編碼）、中護線 01010、
右 6 碼（R 編碼）、右護線 101，每條模組寬一樣。
"""

import cv2
import numpy as np

_L = ["0001101", "0011001", "0010011", "0111101", "0100011",
      "0110001", "0101111", "0111011", "0110111", "0001011"]
_G = ["".join("1" if b == "0" else "0" for b in code)[::-1] for code in _L]
_R = ["".join("1" if b == "0" else "0" for b in code) for code in _L]
_PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
           "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def ean13_check_digit(first12: str) -> str:
    s = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(first12))
    return str((10 - s % 10) % 10)


def ean13_bits(first12: str) -> str:
    code = first12 + ean13_check_digit(first12)
    parity = _PARITY[int(code[0])]
    left = "".join((_L if p == "L" else _G)[int(d)] for p, d in zip(parity, code[1:7]))
    right = "".join(_R[int(d)] for d in code[7:])
    return "101" + left + "01010" + right + "101"


def make_ean13_image(first12: str = "471234567890", module_px: int = 4,
                     bar_h: int = 200, quiet: int = 15, canvas=(480, 640),
                     noise: float = 8.0, seed: int = 0) -> np.ndarray:
    """
    回傳 (H, W, 3) uint8 BGR 白底黑條的 EAN-13 圖，條碼置中。

    預設加一點模糊與雜訊，模擬相機拍到的樣子。
    """
    bits = ean13_bits(first12)
    row = np.array([0 if b == "1" else 255 for b in bits], dtype=np.uint8)
    row = np.pad(np.repeat(row, module_px), quiet * module_px, constant_values=255)
    strip = np.tile(row, (bar_h, 1))

    H, W = canvas
    img = np.full((H, W), 255, np.uint8)
    y0, x0 = (H - strip.shape[0]) // 2, (W - strip.shape[1]) // 2
    img[y0:y0 + strip.shape[0], x0:x0 + strip.shape[1]] = strip
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return np.dstack([img] * 3)
