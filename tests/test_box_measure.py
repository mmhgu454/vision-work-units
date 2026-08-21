"""box_measure 的單元測試。不需要相機、不需要模型、不需要 pyrealsense2。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import box_measure as bm
from box_measure import ContractError
from synthetic import make_box_scene, fake_detect


# ----------------------------------------------------------------- 可攜性

def test_沒有拉進硬體或專案相依():
    """
    可攜性的驗收標準：import box_measure 不得帶進任何硬體/專案模組。

    必須在乾淨的子行程裡驗——同一個 pytest 行程裡別的測試會 import
    realsense_source，那會把 pyrealsense2 載進 sys.modules，
    直接檢查 sys.modules 會變成依賴測試執行順序。
    """
    import subprocess

    probe = (
        "import sys; sys.path.insert(0, %r); import box_measure; "
        "forbidden = {'pyrealsense2','rfdetr','torch','cv2','config',"
        "'log_manager','camera_manager','open3d','joblib'}; "
        "bad = forbidden & {m.split('.')[0] for m in sys.modules}; "
        "print(','.join(sorted(bad)))" % str(Path(bm.__file__).parent)
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    bad = out.stdout.strip()
    assert not bad, f"不該被載入的模組: {bad}"


# ----------------------------------------------------------------- 純數學

def test_平面擬合能找回已知法向量():
    _, pc, top_box, _ = make_box_scene()
    pts = pc[45:95, 105:215].reshape(-1, 3)
    pts = pts[pts[:, 2] > 0.01]
    normal, d, inliers = bm.fit_plane_ransac_svd(pts)
    assert normal is not None
    assert abs(abs(float(normal[1])) - 1.0) < 1e-3, f"法向量應接近 ±Y，得到 {normal}"
    assert abs(float(normal[0])) < 1e-3
    assert len(inliers) > 0.9 * len(pts)


def test_正交基底確實正交且為單位向量():
    for n in [np.array([0.0, -1.0, 0.0]), np.array([0.0, -0.9, 0.3]),
              np.array([0.05, -0.8, 0.4])]:
        u, v = bm.get_orthonormal_basis(n)
        n_hat = n / np.linalg.norm(n)
        assert abs(np.linalg.norm(u) - 1) < 1e-9
        assert abs(np.linalg.norm(v) - 1) < 1e-9
        assert abs(np.dot(u, v)) < 1e-9
        assert abs(np.dot(u, n_hat)) < 1e-9
        assert abs(np.dot(v, n_hat)) < 1e-9


def test_箱型比對_完全吻合時回傳該型號():
    dims = {"A": (0.60, 0.35, 0.23), "B": (0.40, 0.38, 0.40)}
    assert bm.find_closest_box_type(0.60, 0.35, 0.23, dimensions=dims)[0] == "A"
    assert bm.find_closest_box_type(0.40, 0.38, 0.40, dimensions=dims)[0] == "B"


def test_箱型比對_誤差超過門檻時回傳Err():
    dims = {"A": (0.60, 0.35, 0.23)}
    box_type, loss = bm.find_closest_box_type(0.30, 0.30, 0.30, dimensions=dims)
    assert box_type == "Err"
    assert loss >= bm.DEFAULT_MATCH_LOSS_THRESHOLD


def test_框旋轉來回還原():
    box = [100.0, 200.0, 300.0, 450.0]
    H, W = 800, 1280
    for k in range(4):
        up_h, up_w = (H, W) if k % 2 == 0 else (W, H)
        rotated = bm._rot90_box(box, H, W, k)
        assert np.allclose(bm._rot90_box(rotated, up_h, up_w, -k), box)


# ------------------------------------------------------- measure_box 端到端

@pytest.mark.parametrize("L,W,H", [(0.60, 0.35, 0.23), (0.68, 0.41, 0.40), (0.40, 0.38, 0.40)])
def test_量測出的長寬高比例符合真實比例(L, W, H):
    """
    絕對值會因為 5% 內縮與 2–98 百分位裁剪而系統性偏小。
    L 與 W 的縮水係數相同（都經過內縮＋裁剪），所以 L/W 比例守恆；
    H 只經過裁剪、沒有內縮，係數不同，因此 L/H 比例「不」守恆
    ——這個差異由 test_量測的系統性縮水率穩定 負責記錄。
    """
    color, pc, tb, fb = make_box_scene(L, W, H)
    r = bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)
    assert r.lwh is not None, r.stage
    l_m, w_m, h_m = r.lwh
    assert l_m / w_m == pytest.approx(L / W, rel=0.02)


@pytest.mark.parametrize("L,W,H", [(0.60, 0.35, 0.23), (0.68, 0.41, 0.40)])
def test_量測的系統性縮水率穩定(L, W, H):
    """記錄住已知的系統性偏差，將來若演算法被改動這個測試會亮。"""
    color, pc, tb, fb = make_box_scene(L, W, H)
    r = bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)
    l_m, w_m, h_m = r.lwh
    assert l_m / L == pytest.approx(0.864, abs=0.01)
    assert w_m / W == pytest.approx(0.866, abs=0.01)
    assert h_m / H == pytest.approx(0.966, abs=0.01)


def test_比對成功時回傳箱型與中間結果():
    color, pc, tb, fb = make_box_scene(0.60, 0.35, 0.23)
    probe = bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)
    dims = {"TESTBOX": probe.lwh}                    # 用實測值當尺寸表
    r = bm.measure_box(color, pc, fake_detect(tb, fb), dimensions=dims, view_rot_k=0)
    assert r.ok and r.box_type == "TESTBOX"
    assert r.top_box is not None and r.face_box is not None
    assert r.boxes is not None and len(r.boxes) == 2
    assert r.top_plane_normal is not None
    assert "完成" in r.stage


# ----------------------------------------------------------------- 契約檢查

def test_相機有roll時明確報錯而不是靜默給錯答案():
    color, pc, tb, fb = make_box_scene(roll_deg=40.0)
    with pytest.raises(ContractError, match="X 分量"):
        bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)


def test_roll在容許範圍內不報錯():
    color, pc, tb, fb = make_box_scene(roll_deg=5.0)
    r = bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)
    assert r.lwh is not None, r.stage


def test_影像與點雲解析度不一致時報錯():
    color, pc, tb, fb = make_box_scene()
    with pytest.raises(ContractError, match="解析度不一致"):
        bm.measure_box(color[:100], pc, fake_detect(tb, fb), view_rot_k=0)


def test_點雲形狀不對時報錯():
    color, pc, tb, fb = make_box_scene()
    with pytest.raises(ContractError, match=r"\(H, W, 3\)"):
        bm.measure_box(color, pc[..., 0], fake_detect(tb, fb), view_rot_k=0)


# ----------------------------------------------------------------- 失敗路徑

def test_沒有偵測框時回傳Err並說明停在哪():
    color, pc, tb, fb = make_box_scene()
    empty = lambda img: (np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))
    r = bm.measure_box(color, pc, empty, view_rot_k=0)
    assert not r.ok and r.box_type == "Err"
    assert "沒有任何偵測框" in r.stage


def test_只有top沒有face時回傳Err並說明停在哪():
    color, pc, tb, fb = make_box_scene()
    only_top = lambda img: (np.asarray([tb], dtype=float),
                            np.asarray([1], dtype=int), np.ones(1))
    r = bm.measure_box(color, pc, only_top, view_rot_k=0)
    assert not r.ok
    assert "找不到成對" in r.stage


def test_單元不碰檔案系統(tmp_path, monkeypatch):
    """Q14：單元不得寫任何檔案。"""
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    color, pc, tb, fb = make_box_scene()
    bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)
    assert set(tmp_path.rglob("*")) == before, "measure_box 產生了檔案"


def test_偵測器與單元的view_rot_k不一致時報錯():
    """兩邊的 k 對不齊會讓框與點雲靜默錯位，必須當場攔下來。"""
    color, pc, tb, fb = make_box_scene()
    detect = fake_detect(tb, fb)
    detect.view_rot_k = 2                       # 偵測器宣稱自己轉了 2 次
    with pytest.raises(ContractError, match="不一致"):
        bm.measure_box(color, pc, detect, view_rot_k=0)


def test_偵測器沒公開view_rot_k時不影響運作():
    """檢查是選擇性的：不公開這個屬性的偵測器照樣能用。"""
    color, pc, tb, fb = make_box_scene()
    detect = fake_detect(tb, fb)
    assert not hasattr(detect, "view_rot_k")
    assert bm.measure_box(color, pc, detect, view_rot_k=0).lwh is not None


def test_框貼到畫面邊緣時給出警告但不中斷():
    """
    真實資料驗出來的問題：face 框碰到畫面下緣時箱子被裁切，
    高度會偏小但數字看起來仍合理。必須講出來。
    """
    color, pc, tb, fb = make_box_scene(img_h=160)      # 讓 face 框剛好貼到下緣
    fb = [fb[0], fb[1], fb[2], 159]
    r = bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)
    assert r.warnings, "face 框貼到下緣卻沒有警告"
    assert any("下緣" in w and "face" in w for w in r.warnings), r.warnings
    assert r.lwh is not None, "警告不該中斷流程"
    assert not r.trustworthy


def test_框沒碰到邊緣時沒有警告():
    color, pc, tb, fb = make_box_scene()
    r = bm.measure_box(color, pc, fake_detect(tb, fb), view_rot_k=0)
    assert r.warnings == []
    assert r.trustworthy is (r.box_type != "Err")
