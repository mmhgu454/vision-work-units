"""
接線層的測試：取幀、偵測、畫圖，以及它們與量測單元之間的整合。

為什麼獨立成一檔
----------------
三個量測單元（box_measure / fit_check / pallet_stack）的測試檔**只認識自己那個單元**，
這樣「單元 + 它的測試 + 它的合成場景」三個檔案就能一起搬到新專案，
`pytest` 一跑就知道在新環境還對不對。

底下這些測試會同時碰到單元與接線層（`rfdetr_detector` / `realsense_source` / `viz`），
所以放這裡。搬單元的時候**不需要帶走這一檔**——新專案的取幀與偵測是自己的實作，
這些測試對它沒有意義。

相依：pyrealsense2、cv2（`rfdetr_detector` 的 rfdetr 是延後 import 的，這裡不會載到）。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import box_measure as bm
import fit_check as fc
import viz
from synthetic import make_box_scene, fake_detect
from synthetic_fit import make_plane, empty_ref_plane, BASE_H

LEFT = (1, 1)


# ------------------------------------------------- 偵測器 x 量測單元

def test_偵測器與單元的旋轉互為反函數():
    """
    偵測器把框轉回輸入影像的方向，單元再把框轉成正立視角去挑 top/face。
    兩邊必須是精確的反向操作，否則框與點雲會靜默錯位。
    """
    from rfdetr_detector import rot90_boxes

    orig = [[100.0, 200.0, 300.0, 450.0], [10.0, 20.0, 30.0, 40.0]]
    H, W = 800, 1280
    for k in range(4):
        up_h, up_w = (H, W) if k % 2 == 0 else (W, H)
        back = rot90_boxes(orig, up_h, up_w, -k)          # 偵測器：轉回輸入方向
        again = [bm._rot90_box(b, H, W, k) for b in back]  # 單元：轉成正立視角
        assert np.allclose(again, orig), f"k={k} 對不上"


def test_視角旋轉不影響量測結果():
    """k=2 搭配旋轉過的輸入，應與 k=0 搭配原始輸入得到相同答案。"""
    from rfdetr_detector import rot90_boxes

    color, pc, tb, fb = make_box_scene(0.60, 0.35, 0.23)
    base = bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)

    img_h, img_w = pc.shape[:2]
    pc_r = np.rot90(pc, k=2, axes=(0, 1)).copy()
    color_r = np.rot90(color, k=2, axes=(0, 1)).copy()
    tb_r, fb_r = rot90_boxes([tb, fb], img_h, img_w, 2)

    rot = bm.measure_box(color_r, pc_r, fake_detect(list(tb_r), list(fb_r)), view_rot_k=2)
    assert rot.lwh is not None, rot.stage
    assert np.allclose(rot.lwh, base.lwh, rtol=0.02), f"{rot.lwh} vs {base.lwh}"


# ------------------------------------------------- 取幀元件

def test_點雲座標系旋轉是取幀元件的參數而非寫死():
    """座標修正由取幀端負責且可調，量測單元不認識它。"""
    pytest.importorskip("pyrealsense2")
    import realsense_source as rsrc

    assert rsrc.DEFAULT_PC_ROTATION_DEG == (0.0, 0.0, 180.0)   # task3 現場值
    assert rsrc.TASK1_ROTATION_DEG == (0.0, 0.0, -90.0)        # task1 現場值
    R0 = rsrc.get_rotation_matrix(0, 0, 0)
    R180 = rsrc.get_rotation_matrix(0, 0, 180)
    p = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose((R0 @ p.T).T, [[1.0, 2.0, 3.0]])
    assert np.allclose((R180 @ p.T).T, [[-1.0, -2.0, 3.0]])


def test_三個task的取幀方法都能覆寫座標修正():
    """換相機架法時不用改檔案，傳參數即可。"""
    pytest.importorskip("pyrealsense2")
    import inspect
    import realsense_source as rsrc

    sigs = {
        "get_frame_task1": "rotation_deg",
        "get_frame_task2": "pose",
        "__init__": "pc_rotation_deg",
    }
    for method, param in sigs.items():
        params = inspect.signature(getattr(rsrc.RealSenseSource, method)).parameters
        assert param in params, f"RealSenseSource.{method}() 沒有 {param} 參數"


# ------------------------------------------------- 畫圖 x 量測單元

def test_對準疊圖回報正確的基準高度與點數():
    pc, color = make_plane(as_image=True)
    _, ref, bound = fc.resolve_rois(LEFT, "L554001")
    img, base_h, n_ref = viz.draw_task2_aim(color, pc, ref, bound)
    assert base_h == pytest.approx(BASE_H, abs=1e-3)
    assert n_ref >= fc.MIN_REF_POINTS
    assert img.shape == color.shape


def test_對準疊圖在基準框沒點時回報不足():
    """這是 task2 最常見的失敗，預覽必須當場看得出來。"""
    pc = empty_ref_plane()
    color = np.zeros((1, pc.shape[0], 3), dtype=np.uint8)
    _, ref, bound = fc.resolve_rois(LEFT, "L554001")
    img, base_h, n_ref = viz.draw_task2_aim(color, pc, ref, bound)
    assert n_ref == 0
    assert base_h is None


def test_對準疊圖夠便宜可以每幀跑():
    """完整 check_fit 在真實解析度要 0.5 秒以上，疊圖必須遠低於一幀的時間。"""
    pc, color = make_plane(as_image=True)
    _, ref, bound = fc.resolve_rois(LEFT, "L554001")
    viz.draw_task2_aim(color, pc, ref, bound)          # 暖機
    t0 = time.time()
    for _ in range(10):
        viz.draw_task2_aim(color, pc, ref, bound)
    per_frame = (time.time() - t0) / 10
    assert per_frame < 0.05, f"每幀 {per_frame * 1000:.0f} ms，太慢了"
