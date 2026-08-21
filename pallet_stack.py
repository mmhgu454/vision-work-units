"""
pallet_stack.py — 棧板堆疊位置判定（可攜單元）

把這一個檔案複製到別的專案就能用。相依只有 numpy（SVR 評分那條路徑另外需要 joblib）。

來源
----
fork 自 fine_good_unit 的:
  - src/stage4/task1.py              （第 724 行以前的計算部分）
  - src/vision_engine.py             （process_stage4_task1 的流程部分）

沒有複製過來的東西:
  - 第 724 行以後的 save_* 渲染函數（491 行）——它們是 open3d/cv2 的來源，
    而且本單元回傳中間結果讓呼叫端自己畫，不需要它們
  - get_prediction / get_peoples_detect ——偵測改為注入
  - world_to_grid、build_block_list_multi、build_tape_block_multi（共 275 行死碼，無人呼叫）

⚠ 原始碼的三個 bug（原檔未動，副本已修，全部有測試守著）
--------------------------------------------------------
1. vision_engine.py:194/196 曾用 `region_type=` 呼叫，但函數參數叫 `station`
   -> TypeError。**已於 2026-08-19 在 src/ 修正**（改名沒改乾淨的殘留，
   改名前的快照見 stage4/task1存圖轉正還沒測過.py）。副本一律用 `station`。
2. task1.py:359 與 :695 讀 `box.depth`，但 config 的 BoxSize 只有 length/width/height
   -> AttributeError。這是改 task3 時要順便改、但漏掉的殘留。
   欄位對應（**已向作者確認**）：
       舊 width（長邊） -> 新 length
       舊 depth（短邊） -> 新 width
   所以 `box.depth` 對應 `box.width`，而**同一行的 `box.width` 也要改成 `box.length`**——
   否則長短邊會互換。兩處都已對照原始值驗證過。
3. task1.py:215 與 :479 的 `return` 寫在 else: 裡面，棧板放滿（layer >= 3）時
   隱含回傳 None，呼叫端解包會 TypeError。副本改成回傳 (result, None, hmap)，
   與同一函數上方 `if not candidates:` 的慣例一致。

也就是說：**這個檔案不是「行為等價的複製」，因為原始碼沒有可用的行為可以等價。**
計算邏輯本身一行未改，改的只有上面三處會讓程式跑不完的錯誤。

呼叫契約
--------
pc_np : (H, W, 3) float ndarray
    只做過 Z 軸 -90 度旋轉的相機座標系，**尚未套用傾角/高度校正**。
    校正必須在演算法內部、`z > 0` 過濾之後才做（校正會改到 z），順序不能換，
    所以本單元收一個 calibrate callable 而不是收已校正的點雲。

calibrate : Callable[[ndarray(N,3)], ndarray(N,3)]
    把過濾後的點轉到世界座標系（Y 上、地面為 0）。
    用 make_calibrator() 產生，預設值對齊來源專案的 TASK1_CAM_POSE。

detect : Callable[[ndarray], (boxes, labels, scores)]
    框必須與傳入的 color_img 同方向。label：0=pallet, 1=top, 2=face。

本單元不碰相機、不碰檔案系統、不寫檔。

用法
----
    from pallet_stack import analyze_pallet, make_calibrator, load_score_model

    calibrate = make_calibrator(tilt=-35.0, phy_height=2.25, flip=True)
    score_model = load_score_model("model/pallet_svr_model.joblib")   # 載一次重複用

    result = analyze_pallet(
        color_img, pc_np, detect,
        station="g4_1",             # g4_1 / g4_2 / g8_1 / p_1
        cardboard_type="L554001",
        block_type="pallet",        # "pallet" 或 "ground"
        calibrate=calibrate,
        detect_people=detect_people,  # 選用；有人時優先度會被壓低
        score_model=score_model,      # 不給就跳過評分，回傳 1.0
    )

    if result.ok:
        print(result.place_position)  # (row, col, layer)
    elif result.full:
        print("所有位置都堆滿了")     # 這是正常終態，不是錯誤
    result.block                      # 各位置的層數矩陣
    result.hmap                       # 高度圖，拿去渲染

堆疊判定本身不需要 AI 模型——只有優先度評分需要偵測框。
只想知道「下一箱放哪」的話，直接呼叫 build_block_list / build_tape_block 即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 可覆寫設定

#: 箱型 -> (length, width, height)，單位公尺。與 box_measure / fit_check 的表一致。
DEFAULT_BOX_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "L514157N": (0.35, 0.35, 0.19),
    "L514157":  (0.35, 0.35, 0.23),
    "L514158":  (0.60, 0.35, 0.23),
    "L514158N": (0.60, 0.35, 0.19),
    "L514142":  (0.68, 0.41, 0.40),
    "L514141":  (0.40, 0.38, 0.40),
    "L554002":  (0.60, 0.56, 0.40),
    "L554001":  (0.57, 0.50, 0.50),
}


@dataclass(frozen=True)
class BlockInfo:
    """一個工作站的區域尺寸。對齊 src/config.py 的 BlockInfo。"""
    horizon_bias: float
    camera_distance: float
    width: float
    length: float
    height: float

    @property
    def block_size(self) -> Tuple[float, float, float]:
        """(深度, 寬度, 基準高度)，單位公尺。注意第一個是 length 欄位。"""
        return self.length, self.width, self.height

    @property
    def block_bias(self) -> Tuple[float, float]:
        """(水平偏移, 相機到區域前緣的距離)，單位公尺。"""
        return self.horizon_bias, self.camera_distance


#: 對齊 src/config.py 的 BLOCK_CATALOG
DEFAULT_BLOCK_CATALOG: Dict[str, BlockInfo] = {
    "g4_1": BlockInfo(horizon_bias=0.0, camera_distance=2.10, width=1.2, length=1.05, height=0.00),
    "g4_2": BlockInfo(horizon_bias=0.0, camera_distance=1.10, width=1.2, length=1.05, height=0.00),
    "g8_1": BlockInfo(horizon_bias=0.0, camera_distance=1.10, width=1.2, length=2.10, height=0.00),
    "p_1":  BlockInfo(horizon_bias=0.0, camera_distance=1.25, width=1.0, length=1.00, height=0.15),
}

HMAP_BASE_SIZE = 100        # 對齊 HMAP_CFG.base_size
X_BOUND = 0.2               # 對齊 PLACE_CFG.x_bound，畫面左右各濾掉這個比例
MAX_LAYERS = 3              # 每個位置最多堆幾層
CORNER_GAP_CELLS = 5        # build_block_list 四角取樣時留的空隙（防多棧板誤判）
TAPE_BOX_PADDING = 0.01     # build_tape_block 算排數時的箱間留隙 (m)

#: 對齊 src/config.py 的 TASK1_CAM_POSE
DEFAULT_TILT = -35.0
DEFAULT_PHY_HEIGHT = 2.25
DEFAULT_FLIP = True


class ContractError(ValueError):
    """輸入資料或設定違反本單元的契約。"""


# ---------------------------------------------------------------- 座標校正

def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """歐拉角 (XYZ) 轉旋轉矩陣。內聯自 camera_manager，避免相依。"""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def make_calibrator(tilt: float = DEFAULT_TILT,
                    phy_height: float = DEFAULT_PHY_HEIGHT,
                    flip: bool = DEFAULT_FLIP) -> Callable[[np.ndarray], np.ndarray]:
    """
    產生一個座標校正函數（傾角、翻轉、高度平移）。

    預設值對齊來源專案的 TASK1_CAM_POSE。相機架法不同就改這三個參數——
    調法與 task2 相同：先用「水平面的殘餘斜率」定 tilt，再用高度定 phy_height。
    """
    R_tilt = rotation_matrix(tilt, 0, 0)
    R_flip = rotation_matrix(0, 0, 180) if flip else None

    def calibrate(points: np.ndarray) -> np.ndarray:
        pts = (R_tilt @ points.T).T
        if R_flip is not None:
            pts = (R_flip @ pts.T).T
        pts = pts.copy()
        pts[:, 1] += phy_height
        pts[:, 0] *= -1
        return pts

    calibrate.params = (tilt, phy_height, flip)      # 方便除錯時看得到
    return calibrate


#: 相機原始點雲要先轉到本單元契約座標系的旋轉（對齊 task1.py 的 R_fix）
PC_ROTATION_DEG = (0.0, 0.0, -90.0)


# ---------------------------------------------------------------- 回傳型別

@dataclass
class StackResult:
    """堆疊判定結果 + 中間產物。呼叫端可用這些自行畫高度圖或做斷言。"""

    place_position: Optional[Tuple[int, int, int]]     # (row, col, layer)，None 代表放滿
    block: Optional[np.ndarray] = None                 # 各位置的層數矩陣
    station: str = ""
    stage: str = ""
    hmap: Optional[np.ndarray] = None                  # 高度圖，呼叫端拿去渲染
    prior_score: Optional[float] = None
    top_entries: List[tuple] = field(default_factory=list)
    face_entries: List[tuple] = field(default_factory=list)
    detected_pallets: List[tuple] = field(default_factory=list)
    people_count: int = 0
    boxes: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """找到了可放的位置。放滿時是 False，但那是正常終態，見 full。"""
        return self.place_position is not None

    @property
    def full(self) -> bool:
        """所有位置都堆到上限。這是正常終態，不是錯誤。"""
        return self.block is not None and bool((self.block >= MAX_LAYERS).all())


# ============================================================================
# 以下為 src/stage4/task1.py 第 724 行以前的計算部分，逐字複製。
# 改動只有檔頭列出的三個 bug，以及：
#   - rs_pc / rs_depth_frame 換成傳入的 pc_np（Q2 的邊界）
#   - camera_manager.apply_calibration 換成傳入的 calibrate callable
#   - config 的常數換成可覆寫的參數
#   - joblib.load 改為可預先載入並重複使用（原本每次呼叫都讀一次磁碟）
# 計算邏輯本身一行未改。
# ============================================================================

def get_centroid(pc_np, y1: int, y2: int, x1: int, x2: int,
                 calibrate: Callable) -> Optional[Tuple[float, float, float]]:
    """計算指定 ROI 區域的質心 (世界座標)"""
    roi = pc_np[y1:y2, x1:x2].reshape(-1, 3)
    roi = roi[roi[:, 2] > 0]
    if len(roi) == 0:
        return None
    roi = calibrate(roi)
    centroid = np.median(roi, axis=0)
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))


def get_object_3d_entries(pc_np, boxes, labels, cardboard_type: str,
                          calibrate: Callable, x_bound: float = X_BOUND):
    """把偵測框轉成 3D 質心，依類別分成 top / face / pallet 三組。"""
    h, w = pc_np.shape[:2]

    face_entries = []
    top_entries = []
    detected_pallets = []

    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)

        # 邊界過濾（非棧板類別）
        if cls != 0:
            cx_px = (x1 + x2) / 2
            if cx_px < x_bound * w or cx_px > (1 - x_bound) * w:
                continue

        pt = get_centroid(pc_np, y1, y2, x1, x2, calibrate)
        if pt is None:
            continue
        x3d, y3d, z3d = pt

        if cls == 2:      # Face
            face_entries.append((x3d, y3d, z3d, cardboard_type))
        elif cls == 1:    # Top
            top_entries.append((x3d, y3d, z3d, cardboard_type))
        elif cls == 0:    # Pallet
            detected_pallets.append((x3d, y3d, z3d))

    return top_entries, face_entries, detected_pallets


def _prepare_ground_points(pc_np, calibrate):
    """攤平 -> 濾掉 z<=0 -> 校正。順序不能換：校正會改到 z。"""
    flat = pc_np.reshape(-1, 3)
    flat = flat[flat[:, 2] > 0]
    if len(flat) > 0:
        flat = calibrate(flat)
    return flat


def _fill_hmap(flat, origin_x, pz, pw, pd, base_size):
    """把點雲投影成高度圖：每個網格取最大的 y。"""
    hmap = np.full((base_size, base_size), -np.inf, dtype=float)
    if len(flat) == 0:
        return hmap

    rel_x = flat[:, 0] - origin_x
    rel_z = flat[:, 2] - pz
    valid = (rel_x >= 0) & (rel_x < pw) & (rel_z >= 0) & (rel_z < pd)

    cell_w = pw / base_size
    cell_d = pd / base_size
    if np.any(valid):
        cols = np.floor(rel_x[valid] / cell_w).astype(int)
        rows_from_front = np.floor(rel_z[valid] / cell_d).astype(int)
        cols = np.clip(cols, 0, base_size - 1)
        rows_from_front = np.clip(rows_from_front, 0, base_size - 1)
        rows = base_size - 1 - rows_from_front
        np.maximum.at(hmap, (rows, cols), flat[valid, 1])
    return hmap


def _make_height_to_layer(box_height: float):
    def height_to_layer(max_y: float, base_y: float) -> int:
        if not np.isfinite(max_y):
            return 0
        return max(0, min(MAX_LAYERS, int(np.rint((max_y - base_y) / box_height))))
    return height_to_layer


def _pick_best(block):
    """挑層數最少、其次 row 最小的位置。回傳 (place_position, best)。"""
    candidates = [{"row": r, "col": c, "layer": int(block[r, c])}
                  for r in range(block.shape[0]) for c in range(block.shape[1])]
    if not candidates:
        return None, None
    best = min(candidates, key=lambda x: (x["layer"], x["row"]))
    if best["layer"] >= MAX_LAYERS:
        # ⚠ 原始碼在這裡沒有 return（return 寫在 else: 裡面），會隱含回傳 None
        #   讓呼叫端解包時 TypeError。這裡補上，與上方 `if not candidates` 一致。
        return None, best
    return (int(best["row"]) + 1, int(best["col"]) + 1, int(best["layer"])), best


def build_block_list(pc_np, cardboard_type: str, station: str, calibrate: Callable, *,
                     dimensions=None, block_catalog=None, base_size: int = HMAP_BASE_SIZE):
    """
    從整張點雲建立棧板四角區域的箱高層數矩陣。
    回傳 (block_state, place_position, hmap)，block shape = (2, 2)。
    """
    dimensions = DEFAULT_BOX_DIMENSIONS if dimensions is None else dimensions
    block_catalog = DEFAULT_BLOCK_CATALOG if block_catalog is None else block_catalog

    flat = _prepare_ground_points(pc_np, calibrate)
    bh = dimensions[cardboard_type][2]
    height_to_layer = _make_height_to_layer(bh)

    pd, pw, ph = block_catalog[station].block_size
    px, pz = block_catalog[station].block_bias
    origin_x = px - pw / 2

    hmap = _fill_hmap(flat, origin_x, pz, pw, pd, base_size)

    # 多棧板怕誤判，留空隙 5cm
    gap = CORNER_GAP_CELLS
    corner = max(1, base_size // 4)
    corner_regions = (
        ((gap, corner), (gap, corner)),
        ((gap, corner), (base_size - corner, base_size - gap)),
        ((base_size - corner, base_size - gap), (gap, corner)),
        ((base_size - corner, base_size - gap), (base_size - corner, base_size - gap)),
    )

    block = np.zeros((2, 2), dtype=int)
    for idx, ((r0, r1), (c0, c1)) in enumerate(corner_regions):
        max_y = float(np.median(hmap[r0:r1, c0:c1]))
        block[idx // 2, idx % 2] = height_to_layer(max_y, ph)

    place_position, _ = _pick_best(block)
    return (station, block), place_position, hmap


def build_tape_block(pc_np, cardboard_type: str, station: str, calibrate: Callable, *,
                     dimensions=None, block_catalog=None, base_size: int = HMAP_BASE_SIZE):
    """
    地面暫存區版本：依箱子長度切成 row_count x 2 個區域，各取中央小塊的中位高度。
    回傳 (block_state, place_position, hmap)。
    """
    dimensions = DEFAULT_BOX_DIMENSIONS if dimensions is None else dimensions
    block_catalog = DEFAULT_BLOCK_CATALOG if block_catalog is None else block_catalog

    flat = _prepare_ground_points(pc_np, calibrate)

    # 原始碼是 `bh, bd = box.height, box.depth`。舊命名 width=長邊、depth=短邊，
    # 改名後 width=短邊、length=長邊，所以舊的 depth 對應現在的 width（短邊）。
    # bd 是「沿棧板深度方向的箱子邊」，用短邊。
    _long, _short, bh = dimensions[cardboard_type]
    bd = _short
    height_to_layer = _make_height_to_layer(bh)

    pd, pw, ph = block_catalog[station].block_size
    px, pz = block_catalog[station].block_bias
    origin_x = px - pw / 2

    row_count = max(1, int(np.floor(pd / (bd + TAPE_BOX_PADDING))))
    col_count = 2

    hmap = _fill_hmap(flat, origin_x, pz, pw, pd, base_size)

    row_edges = np.linspace(0, base_size, row_count + 1, dtype=int)
    col_edges = np.linspace(0, base_size, col_count + 1, dtype=int)

    block = np.zeros((row_count, col_count), dtype=int)
    for r in range(row_count):
        for c in range(col_count):
            region_r0, region_r1 = row_edges[r], row_edges[r + 1]
            region_c0, region_c1 = col_edges[c], col_edges[c + 1]

            # 寬、高各取 1/4，因此面積為小區域的 1/16
            sample_h = max(1, (region_r1 - region_r0) // 4)
            sample_w = max(1, (region_c1 - region_c0) // 4)

            center_r = (region_r0 + region_r1) // 2
            center_c = (region_c0 + region_c1) // 2

            r0 = center_r - sample_h
            r1 = r0 + sample_h
            c0 = center_c - sample_w
            c1 = c0 + sample_w
            if r0 >= r1 or c0 >= c1:
                continue

            max_y = float(np.median(hmap[r0:r1, c0:c1]))
            block[r, c] = height_to_layer(max_y, ph)

    place_position, _ = _pick_best(block)
    return (station, block), place_position, hmap


def extract_normalized_features(faces, tops, N, box_l, box_w, box_h) -> np.ndarray:
    """
    轉換為 10 維特徵：
    [face_overall_h, face_flat_h, top_overall_h, top_flat_h,
     top_z_val_1..3, face_qty, top_qty, N]
    """
    features = []

    def get_ranges(pts):
        if not pts:
            return 0.0, 0.0

        # 1. 穩健的分群法 (1D Clustering)
        sorted_pts = sorted(pts, key=lambda p: p[2])

        depth_groups = []
        current_group = [sorted_pts[0][1]]
        current_depth = sorted_pts[0][2]

        for pt in sorted_pts[1:]:
            # 深度差距小於紙箱深度一半，視為同一排
            if abs(pt[2] - current_depth) < (box_l * 0.5):
                current_group.append(pt[1])
            else:
                depth_groups.append(current_group)
                current_group = [pt[1]]
                current_depth = pt[2]
        depth_groups.append(current_group)

        max_h_diff = max((max(g) - min(g) for g in depth_groups), default=0.0)
        row_flatness_norm = max_h_diff / box_h

        # 2. Y Range (整體高低差)
        y_vals = [pt[1] for pt in pts]
        overall_h_norm = (max(y_vals) - min(y_vals)) / box_h
        return overall_h_norm, row_flatness_norm

    features.extend(get_ranges(faces))     # 1 & 2: Face Ranges
    features.extend(get_ranges(tops))      # 3 & 4: Top Ranges

    # 5, 6, 7: Top 前三高
    if not tops:
        top_y_vals = [0.0, 0.0, 0.0]
    else:
        sorted_tops = sorted(tops, key=lambda pt: pt[2])
        top_y_vals = [pt[1] / box_h for pt in sorted_tops[:3]]
        while len(top_y_vals) < 3:
            top_y_vals.append(0.0)
    features.extend(top_y_vals)

    features.append(len(faces))            # 8: Qty
    features.append(len(tops))
    features.append(N)                     # 9: N (排數)
    return np.array(features)


def load_score_model(model_path):
    """
    載入 SVR 評分模型。原始碼在每次呼叫 get_prior_pallet_score 時都 joblib.load 一次，
    等於每幀讀一次磁碟；這裡拆出來讓呼叫端載一次重複用。
    joblib 沒安裝、或找不到檔案，都回傳 None（評分會退回預設 1.0）。
    這是刻意的：本單元的核心功能只需要 numpy，評分是選配，
    不該因為少裝一個套件就讓整個堆疊判定跑不起來。
    """
    try:
        import joblib
    except ImportError:
        logger.warning("沒有安裝 joblib，無法載入評分模型，評分將回傳預設值 1.0")
        return None
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        logger.warning("找不到模型檔案 %s，評分將回傳預設值 1.0", model_path)
        return None


def get_prior_pallet_score(face_entries, top_entries, box_type, station, *,
                           model=None, model_path=None,
                           dimensions=None, block_catalog=None) -> float:
    """接收真實視覺特徵，轉換為 10 維特徵，透過 SVR 預測分數（限幅 1.0~5.0）。"""
    dimensions = DEFAULT_BOX_DIMENSIONS if dimensions is None else dimensions
    block_catalog = DEFAULT_BLOCK_CATALOG if block_catalog is None else block_catalog

    # 原始碼是 `box_l, box_w, box_h = box.depth, box.width, box.height`，
    # 在舊命名下等於 (短邊, 長邊, 高)。改名後 width=短邊、length=長邊，
    # 所以 box_l 取 width、box_w 取 length —— 注意 box_l 拿到的是短邊，這是原本的語意。
    _long, _short, box_h = dimensions[box_type]
    box_l, box_w = _short, _long

    pd, pw, ph = block_catalog[station].block_size
    row_count = max(1, int(np.floor(pd / box_l))) if ph == 0 else 2

    # 座標解構清理（剔除第 4 個元素 box_type）
    faces = [[x, y, z] for x, y, z, _ in face_entries]
    tops = [[x, y, z] for x, y, z, _ in top_entries]

    if model is None:
        if model_path is None:
            logger.warning("沒有提供評分模型，回傳預設值 1.0")
            return 1.0
        model = load_score_model(model_path)
        if model is None:
            return 1.0

    features = extract_normalized_features(faces, tops, row_count, box_l, box_w, box_h)
    X_scaled = model["scaler"].transform(features.reshape(1, -1))
    predicted_score = model["model"].predict(X_scaled)[0]
    return float(np.clip(predicted_score, 1.0, 5.0))


# ---------------------------------------------------------------- 頂層流程

def analyze_pallet(color_img, pc_np, detect, station: str, cardboard_type: str, *,
                   block_type: str = "pallet",
                   calibrate: Optional[Callable] = None,
                   detect_people: Optional[Callable] = None,
                   score_model=None, score_model_path=None,
                   dimensions=None, block_catalog=None,
                   base_size: int = HMAP_BASE_SIZE,
                   x_bound: float = X_BOUND) -> StackResult:
    """
    跑完一次 task1：偵測 -> 3D 質心 -> 優先度評分 -> 堆疊位置判定。

    等同來源專案的 vision_engine.process_stage4_task1，但不碰相機、不碰檔案系統，
    也不做 API 格式轉換（那是呼叫端的事）。
    """
    dimensions = DEFAULT_BOX_DIMENSIONS if dimensions is None else dimensions
    block_catalog = DEFAULT_BLOCK_CATALOG if block_catalog is None else block_catalog
    calibrate = make_calibrator() if calibrate is None else calibrate

    if pc_np is None or pc_np.ndim != 3 or pc_np.shape[2] < 3:
        raise ContractError(
            f"pc_np 必須是 (H, W, 3)，收到 {None if pc_np is None else pc_np.shape}")
    if color_img is not None and color_img.shape[:2] != pc_np.shape[:2]:
        raise ContractError(
            f"color_img {color_img.shape[:2]} 與 pc_np {pc_np.shape[:2]} 解析度不一致；"
            " 偵測框是 color_img 的像素座標，卻要拿去索引 pc_np。")
    if cardboard_type not in dimensions:
        raise ContractError(f"尺寸表裡沒有箱型 {cardboard_type!r}；可用的有 {sorted(dimensions)}")
    if station not in block_catalog:
        raise ContractError(f"區域表裡沒有工作站 {station!r}；可用的有 {sorted(block_catalog)}")
    if block_type not in ("pallet", "ground"):
        raise ContractError(f"block_type 只能是 'pallet' 或 'ground'，收到 {block_type!r}")

    result = StackResult(place_position=None, station=station, stage="1_偵測")

    boxes, labels, scores = detect(color_img)
    result.boxes, result.labels, result.scores = boxes, labels, scores

    if detect_people is not None:
        people_count, _person_boxes, _person_scores = detect_people(color_img)
        result.people_count = int(people_count)

    result.stage = "2_偵測框轉3D質心"
    top_entries, face_entries, detected_pallets = get_object_3d_entries(
        pc_np, boxes, labels, cardboard_type, calibrate, x_bound=x_bound)
    result.top_entries = top_entries
    result.face_entries = face_entries
    result.detected_pallets = detected_pallets

    result.stage = "3_優先度評分"
    prior_score = get_prior_pallet_score(
        face_entries, top_entries, cardboard_type, station,
        model=score_model, model_path=score_model_path,
        dimensions=dimensions, block_catalog=block_catalog)
    if result.people_count > 0:
        prior_score = 1 / result.people_count
        result.warnings.append(
            f"畫面中偵測到 {result.people_count} 人，優先度降為 {prior_score:.3f}")
    result.prior_score = prior_score

    result.stage = "4_堆疊位置判定"
    builder = build_block_list if block_type == "pallet" else build_tape_block
    (_, block), place_position, hmap = builder(
        pc_np, cardboard_type, station, calibrate,
        dimensions=dimensions, block_catalog=block_catalog, base_size=base_size)

    result.block = block
    result.hmap = hmap
    result.place_position = place_position
    result.stage = (f"5_完成→{place_position}" if place_position is not None
                    else "5_完成→所有位置都已堆滿")
    if place_position is None:
        result.warnings.append("所有位置都已堆到上限，沒有可放置的位置")
    return result
