"""fit_check 的單元測試。不需要相機、不需要模型、不需要 pyrealsense2。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fit_check as fc
from fit_check import ContractError
from synthetic_fit import make_plane, empty_ref_plane, BASE_H

LEFT = (1, 1)      # position_2d[1] == 1 -> 左站
RIGHT = (1, 2)


# ----------------------------------------------------------------- 可攜性

def test_沒有拉進硬體或專案相依():
    """可攜性驗收：import fit_check 不得帶進硬體或專案模組（cv2 是它合法的相依）。"""
    import subprocess
    probe = (
        "import sys; sys.path.insert(0, %r); import fit_check; "
        "forbidden = {'pyrealsense2','rfdetr','torch','config',"
        "'log_manager','camera_manager','open3d','joblib'}; "
        "bad = forbidden & {m.split('.')[0] for m in sys.modules}; "
        "print(','.join(sorted(bad)))" % str(Path(fc.__file__).parent)
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert not out.stdout.strip(), f"不該被載入的模組: {out.stdout.strip()}"


# ----------------------------------------------------------------- 主要路徑

def test_乾淨大平面判定可擺放():
    r = fc.check_fit(None, make_plane(), LEFT, "L554001")
    assert r.fit_ok, r.stage
    assert r.stage.endswith("可擺放")
    assert r.base_h == pytest.approx(BASE_H, abs=1e-3)
    assert len(r.vis) == 4, "四種擺放嘗試都該留下視覺化圖"
    assert r.mask is not None


def test_平面太小判定不可擺放():
    """平面只有 36cm x 27cm，放不下 50cm x 50cm 的 L554001。"""
    r = fc.check_fit(None, make_plane(x=(-0.18, 0.18), z=(0.68, 0.95)), LEFT, "L554001")
    assert not r.fit_ok
    assert "超出區域" in r.stage


def test_層數換算():
    """base_h=0.60、棧板 0.15、箱高 0.50 -> (0.60-0.15)/0.50 = 0.9 -> 第 1 層 -> 回報第 2 層。"""
    r = fc.check_fit(None, make_plane(), LEFT, "L554001")
    assert r.new_height == 1
    assert r.position_3d == [1, 1, 2]


def test_地面模式不扣棧板高度():
    on_pallet = fc.check_fit(None, make_plane(), LEFT, "L554001", block_type="pallet")
    on_ground = fc.check_fit(None, make_plane(), LEFT, "L554001", block_type="ground")
    # ground: 0.60/0.50 = 1.2 -> 1；pallet: (0.60-0.15)/0.50 = 0.9 -> 1。層數同，但參數確實有吃進去
    assert on_pallet.base_h == on_ground.base_h
    r_high = fc.check_fit(None, make_plane(base_h=1.30), LEFT, "L554001", block_type="ground")
    assert r_high.new_height == 3, "1.30/0.50 = 2.6 -> 3"


def test_堆到上限時回報已滿():
    r = fc.check_fit(None, make_plane(base_h=1.60), LEFT, "L554001", block_type="ground")
    assert not r.fit_ok
    assert "已達上限" in r.stage
    assert r.new_height == fc.MAX_STACK_LAYERS


def test_左右站都能跑():
    for pos, cardboard in ((LEFT, "L554001"), (RIGHT, "L554001")):
        r = fc.check_fit(None, make_plane(), pos, cardboard)
        assert r.base_h == pytest.approx(BASE_H, abs=1e-3), (pos, r.stage)


def test_給了影像就會產生標註圖():
    pc, color = make_plane(as_image=True)
    r = fc.check_fit(color, pc, LEFT, "L554001")
    assert r.base_vis is not None
    assert r.base_vis.shape == color.shape


# ----------------------------------------------------------------- 失敗路徑

def test_基準區域沒點時說明停在哪():
    r = fc.check_fit(None, empty_ref_plane(), LEFT, "L554001")
    assert not r.fit_ok
    assert "基準區域點太少" in r.stage


def test_完全沒有有效點():
    pc = np.full((100, 3), np.nan, dtype=np.float32)
    r = fc.check_fit(None, pc, LEFT, "L554001")
    assert not r.fit_ok
    assert "沒有有效點" in r.stage


# ----------------------------------------------------------------- 契約檢查

@pytest.mark.parametrize("bad", [
    "hwc",        # 誤傳 box_measure 的 (H, W, 3) 格式
    "two_cols",   # 只有兩欄
    "transposed", # 轉置成 (3, N)
])
def test_點雲形狀不對時報錯(bad):
    pc = make_plane()
    arr = {"hwc": pc[: 100 * 100].reshape(100, 100, 3),
           "two_cols": pc[:, :2],
           "transposed": pc.T}[bad]
    with pytest.raises(ContractError, match=r"\(N, 3\)"):
        fc.check_fit(None, arr, LEFT, "L554001")


def test_影像與點雲數量不符時報錯():
    pc = make_plane()
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ContractError, match="一一對應"):
        fc.check_fit(color, pc, LEFT, "L554001")


def test_箱型的範圍是None時提前擋下():
    """來源專案 config 就把這些箱型留成 None，直接呼叫會在讀 .x_min 時炸掉。"""
    with pytest.raises(ContractError, match="是 None"):
        fc.check_fit(None, make_plane(), LEFT, "L514157")


def test_不認識的箱型報錯():
    with pytest.raises(ContractError, match="尺寸表裡沒有箱型"):
        fc.check_fit(None, make_plane(), LEFT, "NOT_A_BOX")


# ----------------------------------------------------------------- 迴歸守門

def test_狀態字串在單邊為None時不會崩潰():
    """原始碼 task2.py:668-673 在這個情況會 TypeError；副本已修。"""
    ok = {"fit_ok": True}
    assert fc._fit_status(None, None, 0.1) == "失敗"
    assert fc._fit_status(ok, None, 0.1) == "可擺放"
    assert fc._fit_status(None, ok, 0.1) == "可擺放"
    assert "超出區域" in fc._fit_status({"fit_ok": False}, None, 0.1)


def test_單元不碰檔案系統(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    fc.check_fit(None, make_plane(), LEFT, "L554001")
    assert set(tmp_path.rglob("*")) == before, "check_fit 產生了檔案"


def test_所有log訊息的格式參數都對得上():
    """
    logging 會吞掉格式錯誤只印到 stderr，所以測試得自己檢查。
    這條會走過整條主要路徑，把每一筆 log record 實際格式化一次。
    """
    import logging

    failures = []

    class Checking(logging.Handler):
        def emit(self, record):
            try:
                record.getMessage()
            except Exception as e:
                failures.append(f"{record.pathname}:{record.lineno} -> {e}")

    lg = logging.getLogger("fit_check")
    handler = Checking()
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        fc.check_fit(None, make_plane(), LEFT, "L554001")                      # 可擺放路徑
        fc.check_fit(None, make_plane(x=(-0.18, 0.18), z=(0.68, 0.95)), LEFT, "L554001")  # 超出路徑
        fc.check_fit(None, empty_ref_plane(), LEFT, "L554001")                 # 早退路徑
    finally:
        lg.removeHandler(handler)

    assert not failures, "log 格式參數對不上:\n" + "\n".join(failures)


# ------------------------------------------------- task2 即時對準疊圖（viz）


def test_ROI查表可以獨立呼叫():
    """預覽靠這個拿 ref/bound，不必跑完整分析。"""
    side, ref, bound = fc.resolve_rois(LEFT, "L554001")
    assert side == "left"
    assert ref.x_min == -0.30 and ref.z_max == 0.75
    assert fc.resolve_rois(RIGHT, "L554001")[0] == "right"
    with pytest.raises(ContractError, match="是 None"):
        fc.resolve_rois(LEFT, "L514157")
