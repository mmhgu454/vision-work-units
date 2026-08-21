"""pallet_stack 的單元測試。不需要相機、不需要模型、不需要 pyrealsense2。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pallet_stack as ps
from pallet_stack import ContractError
from synthetic_stack import (STATION, CARDBOARD, make_scene, no_detections,
                             fake_detect, uncalibrate)


# ----------------------------------------------------------------- 可攜性

def test_沒有拉進硬體或專案相依():
    import subprocess
    probe = (
        "import sys; sys.path.insert(0, %r); import pallet_stack; "
        "forbidden = {'pyrealsense2','rfdetr','torch','cv2','config',"
        "'log_manager','camera_manager','open3d','joblib'}; "
        "bad = forbidden & {m.split('.')[0] for m in sys.modules}; "
        "print(','.join(sorted(bad)))" % str(Path(ps.__file__).parent)
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert not out.stdout.strip(), f"不該被載入的模組: {out.stdout.strip()}"


def test_joblib只有評分那條路徑才需要():
    """沒有評分模型時不該 import joblib——複製到沒裝 joblib 的專案也要能跑堆疊判定。"""
    import subprocess
    probe = (
        "import sys; sys.path.insert(0, %r); import numpy as np, pallet_stack as ps; "
        "print('joblib' in sys.modules)" % str(Path(ps.__file__).parent)
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ----------------------------------------------------------------- 座標校正

def test_校正與反校正互為反函數():
    cal = ps.make_calibrator()
    world = np.array([[0.1, 0.5, 2.5], [-0.4, 1.2, 3.0]])
    assert np.allclose(cal(uncalibrate(world, cal)), world, atol=1e-6)


def test_校正參數可以調():
    a = ps.make_calibrator()
    b = ps.make_calibrator(phy_height=a.params[1] + 0.3)
    pts = np.array([[0.0, 0.0, 1.0]])
    assert b(pts)[0, 1] - a(pts)[0, 1] == pytest.approx(0.3, abs=1e-9)


# ----------------------------------------------------------------- 堆疊判定

def test_四角層數換算正確():
    """世界高度 0 / 0.5 / 1.0 / 1.5，箱高 0.5 -> 層數 0 / 1 / 2 / 3。"""
    cal = ps.make_calibrator()
    color, pc = make_scene((0.0, 0.5, 1.0, 1.5), calibrate=cal)
    r = ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD, calibrate=cal)
    assert sorted(r.block.ravel().tolist()) == [0, 1, 2, 3]


def test_挑層數最少的位置():
    cal = ps.make_calibrator()
    color, pc = make_scene((1.5, 1.5, 0.0, 1.5), calibrate=cal)
    r = ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD, calibrate=cal)
    assert r.place_position is not None
    assert r.place_position[2] == 0, "應該挑到唯一那個空位"
    assert r.ok


def test_全部堆滿時回傳None而不是崩潰():
    """
    原始碼 task1.py:215 / :479 的 return 寫在 else: 裡面，
    這個情況會隱含回傳 None，讓 vision_engine 解包時 TypeError。副本已修。
    """
    cal = ps.make_calibrator()
    color, pc = make_scene((1.5, 1.5, 1.5, 1.5), calibrate=cal)
    for block_type in ("pallet", "ground"):
        r = ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD,
                              block_type=block_type, calibrate=cal)
        assert r.place_position is None, block_type
        assert r.full, block_type
        assert not r.ok
        assert "堆滿" in r.stage
        assert r.hmap is not None, "放滿也要回傳高度圖給呼叫端畫"
        assert any("堆到上限" in w for w in r.warnings)


def test_pick_best在放滿時不會漏掉return():
    """直接守 _pick_best，這是原始碼漏 return 的那一段。"""
    assert ps._pick_best(np.full((2, 2), ps.MAX_LAYERS))[0] is None
    assert ps._pick_best(np.zeros((2, 2), dtype=int))[0] == (1, 1, 0)


def test_地面模式的區域數依箱子短邊切分():
    """
    g4_1 深度 1.05m，箱子短邊 0.50m + 0.01 留隙 -> 切得出 2 排。

    用短邊不是長邊：原始碼寫的是 `box.depth`，而舊命名的 depth 就是短邊
    （舊 width=長邊、depth=短邊；改名後 length=長邊、width=短邊）。
    """
    cal = ps.make_calibrator()
    color, pc = make_scene(calibrate=cal)
    r = ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD,
                          block_type="ground", calibrate=cal)
    assert r.block.shape == (2, 2)


def test_短邊長邊沒有互換():
    """
    守住欄位對應：舊 depth -> 新 width（短邊）、舊 width -> 新 length（長邊）。
    互換的話 row_count 和特徵向量都會錯，但不會報錯——只會靜默算錯。
    """
    long_side, short_side, _ = ps.DEFAULT_BOX_DIMENSIONS[CARDBOARD]
    assert long_side > short_side, "尺寸表的順序應為 (length, width, height)"

    # extract_normalized_features 的分群門檻是 box_l * 0.5。
    # 挑一個落在「短邊門檻」與「長邊門檻」之間的深度差，兩者才分得出來：
    #   短邊 0.50 -> 門檻 0.250   長邊 0.57 -> 門檻 0.285
    dz = (short_side + long_side) / 2 * 0.5          # 0.2675
    assert short_side * 0.5 < dz < long_side * 0.5, "測試場景失效，門檻沒有夾住 dz"

    faces = [[0.0, 0.5, 2.20], [0.0, 0.9, 2.20 + dz]]
    feats_short = ps.extract_normalized_features(faces, [], 2, short_side, short_side, 0.5)
    feats_long = ps.extract_normalized_features(faces, [], 2, long_side, short_side, 0.5)

    # 用短邊：dz 超過門檻 -> 分成兩排，每排各一點 -> 排內高低差 0
    # 用長邊：dz 未達門檻 -> 同一排 -> 排內高低差 0.4
    assert feats_short[1] == pytest.approx(0.0)
    assert feats_long[1] == pytest.approx(0.4 / 0.5)
    assert not np.allclose(feats_short, feats_long), "box_l 取長邊或短邊會算出不同特徵"


def test_棧板模式固定四格():
    cal = ps.make_calibrator()
    color, pc = make_scene(calibrate=cal)
    r = ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD, calibrate=cal)
    assert r.block.shape == (2, 2)


# ----------------------------------------------------------------- 偵測整合

def test_偵測框轉成3D質心並分類():
    cal = ps.make_calibrator()
    color, pc = make_scene((0.0, 0.0, 0.0, 0.0), calibrate=cal)
    h, w = pc.shape[:2]
    mid = [(w // 2 - 10, h // 2 - 10, w // 2 + 10, h // 2 + 10)]
    detect = fake_detect([(mid[0], 1), (mid[0], 2), (mid[0], 0)])
    r = ps.analyze_pallet(color, pc, detect, STATION, CARDBOARD, calibrate=cal)
    assert len(r.top_entries) == 1 and len(r.face_entries) == 1
    assert len(r.detected_pallets) == 1
    assert r.top_entries[0][3] == CARDBOARD
    assert r.top_entries[0][1] == pytest.approx(0.0, abs=0.02), "質心高度應接近世界座標 0"


def test_畫面邊緣的非棧板框會被濾掉():
    """x_bound=0.2：左右各 20% 內的 top/face 不算，但 pallet 不受限制。"""
    cal = ps.make_calibrator()
    color, pc = make_scene(calibrate=cal)
    h, w = pc.shape[:2]
    edge = (2, h // 2 - 5, 12, h // 2 + 5)          # 中心 x 約 7px，遠小於 0.2*w
    r = ps.analyze_pallet(color, pc, fake_detect([(edge, 1), (edge, 0)]),
                          STATION, CARDBOARD, calibrate=cal)
    assert len(r.top_entries) == 0, "邊緣的 top 應該被濾掉"
    assert len(r.detected_pallets) == 1, "pallet 不受邊界過濾限制"


def test_偵測到人會壓低優先度並留下警告():
    cal = ps.make_calibrator()
    color, pc = make_scene(calibrate=cal)
    people = lambda img: (2, np.zeros((2, 4)), np.ones(2))
    r = ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD,
                          calibrate=cal, detect_people=people)
    assert r.people_count == 2
    assert r.prior_score == pytest.approx(0.5)
    assert any("偵測到 2 人" in w for w in r.warnings)


# ----------------------------------------------------------------- 評分特徵

def test_特徵向量是10維():
    faces = [[0.0, 0.5, 2.2], [0.1, 0.5, 2.8]]
    tops = [[0.0, 1.0, 2.2]]
    f = ps.extract_normalized_features(faces, tops, 2, 0.57, 0.5, 0.5)
    assert f.shape == (10,)
    assert f[-1] == 2 and f[-2] == len(tops) and f[-3] == len(faces)


def test_沒有評分模型時回傳預設分數():
    assert ps.get_prior_pallet_score([], [], CARDBOARD, STATION) == 1.0
    assert ps.get_prior_pallet_score([], [], CARDBOARD, STATION,
                                     model_path="/nonexistent.joblib") == 1.0


# ----------------------------------------------------------------- 契約檢查

@pytest.mark.parametrize("bad,match", [
    ("flat", r"\(H, W, 3\)"),
    ("none", r"\(H, W, 3\)"),
])
def test_點雲形狀不對時報錯(bad, match):
    color, pc = make_scene()
    arr = pc.reshape(-1, 3) if bad == "flat" else None
    with pytest.raises(ContractError, match=match):
        ps.analyze_pallet(color, arr, no_detections, STATION, CARDBOARD)


def test_影像與點雲解析度不一致時報錯():
    color, pc = make_scene()
    with pytest.raises(ContractError, match="解析度不一致"):
        ps.analyze_pallet(color[:10], pc, no_detections, STATION, CARDBOARD)


def test_不認識的工作站或箱型報錯():
    color, pc = make_scene()
    with pytest.raises(ContractError, match="沒有工作站"):
        ps.analyze_pallet(color, pc, no_detections, "NOT_A_STATION", CARDBOARD)
    with pytest.raises(ContractError, match="沒有箱型"):
        ps.analyze_pallet(color, pc, no_detections, STATION, "NOT_A_BOX")


def test_block_type只能是兩種():
    color, pc = make_scene()
    with pytest.raises(ContractError, match="block_type"):
        ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD, block_type="tape")


def test_單元不碰檔案系統(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    color, pc = make_scene()
    ps.analyze_pallet(color, pc, no_detections, STATION, CARDBOARD)
    assert set(tmp_path.rglob("*")) == before


def test_真實SVR模型能算出限幅內的分數():
    """整合測試：模型檔不在就跳過。"""
    model_path = Path(__file__).resolve().parents[1] / "model" / "pallet_svr_model.joblib"
    if not model_path.exists():
        pytest.skip("找不到 SVR 模型檔")
    model = ps.load_score_model(model_path)
    assert model is not None
    faces = [(0.1, 0.5, 2.3, CARDBOARD), (0.2, 0.5, 2.9, CARDBOARD)]
    tops = [(0.1, 1.0, 2.3, CARDBOARD)]
    score = ps.get_prior_pallet_score(faces, tops, CARDBOARD, STATION, model=model)
    assert 1.0 <= score <= 5.0


def test_評分模型只載入一次就能重複用():
    """原始碼每次呼叫都 joblib.load 一次，等於每幀讀磁碟。"""
    model_path = Path(__file__).resolve().parents[1] / "model" / "pallet_svr_model.joblib"
    if not model_path.exists():
        pytest.skip("找不到 SVR 模型檔")
    model = ps.load_score_model(model_path)
    faces = [(0.1, 0.5, 2.3, CARDBOARD)]
    a = ps.get_prior_pallet_score(faces, [], CARDBOARD, STATION, model=model)
    b = ps.get_prior_pallet_score(faces, [], CARDBOARD, STATION, model=model)
    assert a == b
