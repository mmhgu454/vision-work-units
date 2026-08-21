"""
fit_check.py — 棧板擺放位置檢查（可攜單元）

把這一個檔案複製到別的專案就能用。相依只有 numpy 與 cv2。

來源
----
fork 自 fine_good_unit 的:
  - src/stage4/task2.py              （全部演算法）
  - src/vision_engine.py             （process_stage4_task2 的參數挑選流程）

複製過來時的改動（都不影響計算結果）:
  - from config import ... / from log_manager import ... 移除，
    改為內建可覆寫的預設值與 stdlib logging
  - 兩處 cv2.imwrite 的副作用移除；產生的視覺化影像改為隨回傳值帶出，
    由呼叫端決定要不要存（見 FitResult.vis / FitResult.base_vis）
  - 修掉一個會崩潰的 log 字串：原始碼在 trim_result_1 為 None 而
    trim_result_2 不是 None 時會 TypeError（見 _fit_status），
    那段只用來組 log 訊息，不影響 fit_ok 的計算
  - 重試迴圈移出本單元：取幀是注入的，要重試請由呼叫端多取幾幀再呼叫一次

呼叫契約
--------
pc_calib : (N, 3) float ndarray —— 注意是**扁平**的，不是 (H, W, 3)
    座標系必須與來源專案 task2 收到的一致：
      左站 (position_2d[1] == 1): 繞 Z 軸 +90 度後，再套 tilt=-72.0 / 高度 2.2 / flip 的校正
      右站 (其他):                繞 Z 軸 -90 度後，同上
    在來源專案裡這是由 realsense_source.py 產生的。
    若同時給了 color_img，N 必須等於 color_img 的 H*W（本單元會 reshape 回去畫圖）。

color_img : 可為 None。給了才會產生 base_vis 標註圖。

本單元不碰相機、不碰檔案系統、不寫檔。

用法
----
    from fit_check import check_fit

    result = check_fit(
        color_img,            # 可為 None，給了才會產生 base_vis 標註圖
        pc_calib,             # (N, 3) 扁平、已校正
        position_2d=(1, 1),   # (row, col)；col == 1 是左站，其他是右站
        cardboard_type="L554001",
        block_type="pallet",  # "pallet" 扣 0.15m 棧板高；"ground" 不扣
    )

    if result.fit_ok:
        print(result.position_3d)     # [row, col, 第幾層]
    else:
        print(result.stage)           # 停在哪一步，例如「基準區域點太少」

    result.base_h                     # 量到的目前堆疊表面高度
    result.vis                        # {label: 擺放檢查圖}，要存自己存
    result.base_vis                   # 原圖上標出 ref/bound 範圍

原本的 4 次重試迴圈不在這裡——每次重試要重新取幀，而取幀是注入的，
所以請由呼叫端多取幾幀再呼叫一次。

換工作站或換相機架法時，references / bounds 要重新量，兩者都是可覆寫的參數。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 可覆寫設定

@dataclass(frozen=True)
class RoiBox:
    """一個 3D 長方體範圍。y 預設 -2~+2（來源專案的慣例）。"""
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    y_min: float = -2.0
    y_max: float = 2.0


#: 箱型 -> (length, width, height)，單位公尺。與 box_measure.py 的表一致。
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

#: 對焦用的小基準框，用來量出目前的堆疊高度 base_h。
DEFAULT_REFERENCES: Dict[str, Dict[str, Optional[RoiBox]]] = {
    "left": {
        "L554001": RoiBox(x_min=-0.30, x_max=-0.10, z_min=0.70, z_max=0.75),
        "L514141": RoiBox(x_min=-0.45, x_max=-0.15, z_min=0.82, z_max=0.90),
        "L514157": None,
        "L514157N": None,
    },
    "right": {
        "L554001": RoiBox(x_min=0.06, x_max=0.36, z_min=0.70, z_max=0.89),
        "L514141": RoiBox(x_min=0.22, x_max=0.37, z_min=0.81, z_max=0.94),
        "L514157": None,
        "L514157N": None,
    },
}

#: 搜尋擺放位置的外框範圍。
DEFAULT_BOUNDS: Dict[str, Dict[str, Optional[RoiBox]]] = {
    "left": {
        "L554001": RoiBox(x_min=-0.50, x_max=0.50, z_min=0.40, z_max=1.50),
        "L514141": RoiBox(x_min=-0.60, x_max=0.37, z_min=0.64, z_max=1.68),
        "L514157": None,
        "L514157N": None,
    },
    "right": {
        "L554001": RoiBox(x_min=-0.19, x_max=0.62, z_min=0.36, z_max=1.37),
        "L514141": RoiBox(x_min=-0.26, x_max=0.65, z_min=0.59, z_max=1.61),
        "L514157": None,
        "L514157N": None,
    },
}

DEFAULT_PALLET_HEIGHT = 0.15      # 對齊 src/config.py 的 PALLET_CFG.height
MAX_STACK_LAYERS = 3              # new_height 上限，達到就視為滿
MIN_REF_POINTS = 50               # 基準區域至少要有幾個點


class ContractError(ValueError):
    """輸入資料或設定違反本單元的契約。"""


# ---------------------------------------------------------------- 回傳型別

@dataclass
class FitResult:
    """檢查結果 + 中間產物。呼叫端可用這些自行存圖或做斷言。"""

    fit_ok: bool
    position_3d: List[int]
    stage: str
    base_h: Optional[float] = None            # 量到的目前堆疊表面高度 (m)
    new_height: Optional[int] = None          # 換算成第幾層
    mask: Optional[np.ndarray] = None         # 可用平面的二值圖
    vis: Dict[str, np.ndarray] = field(default_factory=dict)   # label -> 擺放檢查圖
    base_vis: Optional[np.ndarray] = None     # 原圖上標註 ref/bound 範圍
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """trustworthy 的別名的相反面：只看可不可擺放，不管有沒有警告。"""
        return self.fit_ok

    @property
    def trustworthy(self) -> bool:
        """可擺放，而且沒有任何可疑之處。"""
        return self.fit_ok and not self.warnings


# ---------------------------------------------------------------- ROI 查表

def resolve_rois(position_2d, cardboard_type, references=None, bounds=None):
    """
    依照左右站與箱型挑出 (side, ref, bound)。

    抽成公開函數是為了讓對準預覽能在不跑完整 check_fit 的情況下畫出這兩個框——
    完整分析在 1280x800 要 0.5 秒以上，每幀跑不動。
    """
    references = DEFAULT_REFERENCES if references is None else references
    bounds = DEFAULT_BOUNDS if bounds is None else bounds

    side = "left" if position_2d[1] == 1 else "right"
    ref = references.get(side, {}).get(cardboard_type, "MISSING")
    bound = bounds.get(side, {}).get(cardboard_type, "MISSING")

    for name, val in (("references", ref), ("bounds", bound)):
        if val == "MISSING":
            raise ContractError(f"{name} 沒有 {side} 站的 {cardboard_type!r} 這一項")
        if val is None:
            raise ContractError(
                f"{name}[{side}][{cardboard_type!r}] 是 None —— 這個箱型在這一站的範圍還沒量。"
                " 來源專案的 config 就是留 None，直接呼叫會在讀取 .x_min 時炸掉；"
                " 這裡提前擋下來。"
            )
    return side, ref, bound


def roi_mask(pc_calib, roi: RoiBox) -> np.ndarray:
    """一個 ROI 的布林遮罩。形狀跟著 pc_calib 走（扁平或影像形狀都可以）。"""
    x, y, z = pc_calib[..., 0], pc_calib[..., 1], pc_calib[..., 2]
    return ((x > roi.x_min) & (x < roi.x_max)
            & (y > roi.y_min) & (y < roi.y_max)
            & (z > roi.z_min) & (z < roi.z_max))


# ---------------------------------------------------------------- 頂層流程

def check_fit(
    color_img: Optional[np.ndarray],
    pc_calib: np.ndarray,
    position_2d: Sequence[int],
    cardboard_type: str,
    *,
    block_type: str = "pallet",
    pallet_height: Optional[float] = None,
    dimensions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    references: Optional[Dict[str, Dict[str, Optional[RoiBox]]]] = None,
    bounds: Optional[Dict[str, Dict[str, Optional[RoiBox]]]] = None,
    grid_res: float = 0.01,
    height_thresh: float = 0.05,
    area_thresh: float = 0.1,
    normal_offset_m: float = 0.0,
) -> FitResult:
    """
    跑一次擺放位置檢查。

    等同來源專案 vision_engine.process_stage4_task2 的單次嘗試——
    原本的 4 次重試迴圈留給呼叫端：每次重試都要重新取幀，而取幀是注入的。
    """
    dimensions = DEFAULT_BOX_DIMENSIONS if dimensions is None else dimensions
    references = DEFAULT_REFERENCES if references is None else references
    bounds = DEFAULT_BOUNDS if bounds is None else bounds

    fail = FitResult(fit_ok=False, position_3d=[*position_2d, -1], stage="0_參數檢查")

    # --- 契約檢查（Q15：寧可當場報錯，也不要靜默給錯答案）---
    if pc_calib is None or pc_calib.ndim != 2 or pc_calib.shape[1] != 3:
        raise ContractError(
            f"pc_calib 必須是扁平的 (N, 3)，收到 "
            f"{None if pc_calib is None else pc_calib.shape}。"
            " 注意這裡和 box_measure 不同：task2 吃的是攤平的點雲，"
            " 而且必須剛好三欄（收緊到 == 3 是為了擋下轉置或多帶欄位的輸入）。"
        )

    if color_img is not None:
        h, w = color_img.shape[:2]
        if h * w != pc_calib.shape[0]:
            raise ContractError(
                f"color_img {h}x{w} = {h * w} 個像素，但 pc_calib 有 {pc_calib.shape[0]} 個點，"
                " 兩者必須一一對應（本單元會把點雲 reshape 回影像形狀來畫標註圖）。"
            )

    if cardboard_type not in dimensions:
        raise ContractError(f"尺寸表裡沒有箱型 {cardboard_type!r}；可用的有 {sorted(dimensions)}")

    side, ref, bound = resolve_rois(position_2d, cardboard_type, references, bounds)

    if pallet_height is None:
        pallet_height = DEFAULT_PALLET_HEIGHT if block_type == "pallet" else 0.0

    _, box_w, box_h = dimensions[cardboard_type]

    return find_front_plane_size(
        color_img,
        pc_calib,
        ref=ref,
        bound=bound,
        position=list(position_2d),
        side=side,
        grid_res=grid_res,
        height_thresh=height_thresh,
        fit_width=box_w,
        fit_height=box_h,
        area_thresh=area_thresh,
        cardboard_type=cardboard_type,
        normal_offset_m=normal_offset_m,
        pallet_height=pallet_height,
        dimensions=dimensions,
        _fail=fail,
    )


# ============================================================================
# 以下為 src/stage4/task2.py 的演算法，逐字複製。
# 唯一的改動見檔頭說明：移除 config/log_manager import、移除 cv2.imwrite 副作用、
# 修掉 _fit_status 的 None 崩潰、把 CARDBOARD_DIMENSIONS 改為傳入。
# 計算邏輯本身一行未改。
# ============================================================================

def draw_rotated_rect_on_mask(mask, p_left_bottom, normal, tangent,
                              fit_width, fit_height, grid_res, area_thresh,
                              normal_offset_m=0.0):
    """
    在可用平面的遮罩上，從左下角沿著邊界方向放一個箱子大小的矩形，算它超出多少。

    mask          可用平面的二值圖（高度圖網格，不是像素）
    p_left_bottom 矩形的起點，由 fit_multi_segment_left_edge 找出的左下角
    normal        沿著它放寬邊；tangent 沿著它放長邊
    fit_width/height  箱子的兩個邊，單位公尺
    grid_res      每格幾公尺，用來把公尺換算成格數
    area_thresh   超出比例超過這個值就判定放不下

    回傳 (視覺化圖, 矩形四角, 是否超標)。注意第三個是「超標」不是「可放」。
    """
    vis = cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    width_cell = fit_width / grid_res
    height_cell = fit_height / grid_res
    offset_cell = normal_offset_m / grid_res

    # 原始左下角
    p_origin = p_left_bottom.astype(np.float32)

    # 沿著法向量方向偏移 x cm 後，才開始放置
    p0 = p_origin + normal * offset_cell
    p1 = p0 + normal * width_cell
    p2 = p1 - tangent * height_cell
    p3 = p0 - tangent * height_cell

    poly = np.round(np.array([p0, p1, p2, p3])).astype(np.int32)

    box_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(box_mask, [poly], 255)

    region_mask = mask > 0
    box_inside = box_mask > 0

    outside_mask = box_inside & (~region_mask)
    inside_image_count = np.sum(box_inside)
    total_cell = (fit_width / grid_res) * (fit_height / grid_res)
    # 超出 mask 的
    outside_region_count = np.sum(outside_mask)
    # 超出圖片外的
    outside_image_count = max(0, total_cell - inside_image_count)
    outside_count = outside_region_count + outside_image_count

    ratio = outside_count / total_cell
    res = ratio > area_thresh

    if res:
        logger.info("矩形超出區域超過 %.1f%%", area_thresh * 100)
    else:
        logger.info("可擺放")
    logger.info("矩形超出比例: %s", ratio)

    vis[outside_mask] = (0, 0, 255)
    cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
    return vis, poly, res


def fit_multi_segment_left_edge(valid_edges, centroid, side="left"):
    """
    多段左邊界合併：
    1. 找最左線的那條邊界當方向基準
    2. 合併方向相近且貼近同一條左邊界線的邊
    3. 避免右側或其他內部線段混進來
    4. 用端點 fitLine 得到穩定 tangent
    """
    if len(valid_edges) == 0:
        return None

    # 用中點 x 最小的邊當「最左線」基準 或 最 x最大的當最右線
    if side == "left":
        base_edge = min(valid_edges, key=lambda e: e["mid"][0])
    elif side == "right":
        base_edge = max(valid_edges, key=lambda e: e["mid"][0])
    ba = base_edge["a"]
    bb = base_edge["b"]

    base_vec = bb - ba
    base_vec = base_vec / (np.linalg.norm(base_vec) + 1e-8)

    merged_pts = np.array([ba, bb], dtype=np.float32)

    if len(merged_pts) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(merged_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    tangent = np.array([vx, vy], dtype=np.float32)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

    # tangent 固定往下
    if tangent[1] < 0:
        tangent = -tangent

    center = np.array([x0, y0], dtype=np.float32)

    rel = merged_pts - center
    t = rel @ tangent

    # 改取存在的點
    idx_top = np.argmin(t)
    idx_bottom = np.argmax(t)
    left_top = merged_pts[idx_top].astype(np.float32)
    left_bottom = merged_pts[idx_bottom].astype(np.float32)

    n1 = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    n2 = -n1

    edge_mid = (left_top + left_bottom) / 2.0
    to_inside = centroid - edge_mid

    normal = n1 if np.dot(n1, to_inside) > np.dot(n2, to_inside) else n2

    return left_top, left_bottom, tangent, normal, merged_pts


def check_fit_from_rows(mask_u8, keep_rows, fit_width, fit_height, grid_res,
                        area_thresh, label, side, normal_offset_m=0.0):
    """
    對指定的列範圍做一次擺放檢查：取最大連通區 -> 找輪廓 -> 擬合邊界 -> 放矩形。

    keep_rows  布林陣列，False 的列會被排除。呼叫端用它試「整片」與「削掉底部」兩種。
    side       "left" / "right"，決定從哪一側找邊界
    label      只是給 log 與回傳值用的標記

    失敗回傳 None（找不到區域、輪廓、有效邊…），成功回傳
    {"label", "vis", "vis_large", "fit_ok"}。vis 是視覺化圖，本函數不存檔。
    """
    row_index_map = np.flatnonzero(keep_rows)
    mask_forfind = np.ascontiguousarray(mask_u8[keep_rows, :])

    logger.info("%s removed rows: %s", label, mask_u8.shape[0] - mask_forfind.shape[0])

    # 加上平滑
    kernel = np.ones((3, 3), np.uint8)
    mask_forfind = cv2.morphologyEx(mask_forfind, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_forfind = cv2.morphologyEx(mask_forfind, cv2.MORPH_OPEN, kernel, iterations=1)

    # 重新取最大連通區
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_forfind > 0).astype(np.uint8), connectivity=8)

    if num_labels <= 1:
        logger.info("找不到區域")
        return None

    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    mask_forfind = np.zeros_like(mask_forfind, dtype=np.uint8)
    mask_forfind[labels == best_label] = 255

    contours, _ = cv2.findContours(mask_forfind, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        logger.info("[平面] %s 找不到輪廓", label)
        return None

    cnt = max(contours, key=cv2.contourArea)

    cnt_original = cnt.copy()
    cnt_original[:, 0, 1] = row_index_map[cnt[:, 0, 1]]

    # 不要補成凸包
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts_poly = approx[:, 0, :].astype(np.float32)
    pts_poly[:, 1] = row_index_map[pts_poly[:, 1].astype(np.int32)]

    if len(pts_poly) < 3:
        logger.info("[平面] %s 輪廓點太少", label)
        return None

    M = cv2.moments(cnt_original)

    if M["m00"] == 0:
        logger.info("[平面] %s 輪廓面積異常", label)
        return None

    centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)

    edges = []
    for i in range(len(pts_poly)):
        a = pts_poly[i]
        b = pts_poly[(i + 1) % len(pts_poly)]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        length = np.linalg.norm(b - a)
        edges.append({"a": a, "b": b, "mid": (a + b) / 2.0, "length": length, "angle": angle})

    valid_edges = [e for e in edges if e["length"] > 5]

    if len(valid_edges) == 0:
        logger.info("[平面] %s 找不到有效邊", label)
        return None

    fit_result = fit_multi_segment_left_edge(valid_edges, centroid, side)

    if fit_result is None:
        logger.info("[平面] %s 多段左邊界 fit 失敗", label)
        return None

    left_top, left_bottom, tangent, normal, merged_pts = fit_result

    vis, poly, res = draw_rotated_rect_on_mask(
        mask_u8, p_left_bottom=left_bottom, normal=normal, tangent=tangent,
        fit_width=fit_width, fit_height=fit_height, grid_res=grid_res,
        area_thresh=area_thresh, normal_offset_m=normal_offset_m)

    # 原始碼在這裡把 vis 放大 4 倍後 cv2.imwrite 出去；
    # 這裡改成一併回傳，存不存由呼叫端決定（本單元不碰檔案系統）。
    cv2.drawContours(vis, [cnt_original], -1, (255, 0, 0), 1)
    scale_factor = 4
    h_vis, w_vis = vis.shape[:2]
    vis_large = cv2.resize(vis, (w_vis * scale_factor, h_vis * scale_factor),
                           interpolation=cv2.INTER_NEAREST)

    return {"label": label, "vis": vis, "vis_large": vis_large, "fit_ok": not res}


def _fit_status(result_1, result_2, area_thresh):
    """
    組 log 用的狀態字串。

    ⚠ 這裡與原始碼不同：原始碼寫成
        "失敗" if r1 is None and r2 is None else ("可擺放" if r1["fit_ok"] or r2["fit_ok"] ...)
    只要其中一個是 None、另一個不是，就會 TypeError。這段只用來組訊息，
    不影響 fit_ok 的計算，所以在副本裡補上 None 防護。
    """
    oks = [r["fit_ok"] for r in (result_1, result_2) if r is not None]
    if not oks:
        return "失敗"
    return "可擺放" if any(oks) else f"矩形超出區域超過 {area_thresh * 100:.1f}%"


def find_front_plane_size(color_img, pc_calib, ref, bound, position, side,
                          grid_res=0.01, height_thresh=0.05, fit_width=0.5,
                          fit_height=0.45, area_thresh=0.1, cardboard_type="L514157N",
                          normal_offset_m=0.0, pallet_height=0, dimensions=None,
                          _fail: Optional[FitResult] = None) -> FitResult:
    """
    主要流程。一般請呼叫 check_fit()，它會幫你挑好 ref / bound / 尺寸。

    步驟：
      1. 用 ref 這個小框量出目前堆疊表面的高度 base_h（取 y 的中位數）
      2. 把 bound 範圍內的點投影成 x-z 高度圖，每格取最高的 y
      3. 與 base_h 相差超過 height_thresh 的格子視為障礙物，其餘是可用平面
      4. 取最大連通區，用兩種列範圍 x 兩種箱子方向共四次擺放檢查
      5. 由 base_h 換算目前第幾層

    任何一步走不下去就回傳 fit_ok=False 的 FitResult，並在 stage 說明停在哪，
    不會拋例外（契約違反才拋）。
    """
    dimensions = DEFAULT_BOX_DIMENSIONS if dimensions is None else dimensions

    def stop(stage: str) -> FitResult:
        r = FitResult(fit_ok=False, position_3d=[*position, -1], stage=stage)
        logger.info("%s", stage)
        return r

    valid = (np.isfinite(pc_calib[:, 0]) & np.isfinite(pc_calib[:, 1])
             & np.isfinite(pc_calib[:, 2]))
    pts = pc_calib[valid]

    if len(pts) == 0:
        return stop("1_取點→沒有有效點")

    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]

    # ref: 一開始對焦的那個小框的判斷、找到目前高度
    ref_mask = ((xs > ref.x_min) & (xs < ref.x_max)
                & (ys > ref.y_min) & (ys < ref.y_max)
                & (zs > ref.z_min) & (zs < ref.z_max))
    ref_y = ys[ref_mask]

    if len(ref_y) < MIN_REF_POINTS:
        return stop(f"2_量基準高度→基準區域點太少（{len(ref_y)} < {MIN_REF_POINTS}）")

    base_h = float(np.median(ref_y))
    logger.info("ref_y count: %s, base_h=%.4f", len(ref_y), base_h)

    # --- 標註圖：原始碼在這裡 cv2.imwrite，改成回傳 ---
    base_vis = _draw_base_vis(color_img, pc_calib, ref, bound, base_h)

    # roi區選取
    map_mask = ((xs > bound.x_min) & (xs < bound.x_max)
                & (zs > bound.z_min) & (zs < bound.z_max))
    xs = xs[map_mask]
    ys = ys[map_mask]
    zs = zs[map_mask]

    if len(xs) == 0:
        return stop("3_建高度圖→搜尋區域沒有點")

    cols = ((xs - xs.min()) / grid_res).astype(int)
    rows = ((zs.max() - zs + 0) / grid_res).astype(int)

    n_cols = int((xs.max() - xs.min()) / grid_res) + 1
    n_rows = int((zs.max() - zs.min()) / grid_res) + 1

    hmap = np.full((n_rows, n_cols), np.nan)
    for r, c, y in zip(rows, cols, ys):
        if 0 <= r < n_rows and 0 <= c < n_cols:
            if np.isnan(hmap[r, c]):
                hmap[r, c] = y
            else:
                hmap[r, c] = max(hmap[r, c], y)

    diff = np.abs(hmap - base_h)

    # 可優化但不重要
    valid_mask = ~np.isnan(hmap)
    obstacle_mask = (diff > height_thresh) & valid_mask
    usable_plane = valid_mask & (~obstacle_mask)
    usable_u8 = usable_plane.astype(np.uint8)

    if not usable_u8.any():
        return stop("3_建高度圖→沒有可用平面")

    # 只會找白色區塊（我們要的那面）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        usable_u8, connectivity=8)

    if num_labels <= 1:
        return stop("3_建高度圖→沒有連續可用區域")

    best_label = None
    best_area = 0
    # 找最大那塊（一定最大嗎？）
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_label = label_id

    # labels: 整張黑白圖->重新只留最大那塊白色
    largest_region = labels == best_label
    mask_u8 = largest_region.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)   # 補洞2次
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)    # 削邊

    H_full, W_full = mask_u8.shape
    # 只看最下面50% 的左邊15% 80%是空
    left_cols = max(1, int(np.ceil(W_full * 0.15)))
    max_scan_rows = int(np.floor(H_full * 0.5))
    keep_rows = np.ones(H_full, dtype=bool)
    candidate_remove_rows = []

    if side == "left":
        for row in range(H_full - 1, H_full - max_scan_rows - 1, -1):
            if np.mean(mask_u8[row, :left_cols] == 255) >= 0.8:
                candidate_remove_rows.append(row)
    elif side == "right":
        for row in range(H_full - 1, H_full - max_scan_rows - 1, -1):
            if np.mean(mask_u8[row, W_full - left_cols:] == 255) >= 0.8:
                candidate_remove_rows.append(row)

    keep_rows[candidate_remove_rows] = False
    keep_rows_trimmed = keep_rows
    keep_rows_full = np.ones(H_full, dtype=bool)

    # 箱子兩個擺放方向都嘗試
    def run(keep, w, h, label):
        return check_fit_from_rows(mask_u8, keep, w, h, grid_res, area_thresh,
                                   label, side, normal_offset_m)

    trim_result_1 = run(keep_rows_trimmed, fit_height, fit_width, "trim_rows")
    trim_result_2 = run(keep_rows_trimmed, fit_width, fit_height, "trim_rows")
    full_result_1 = run(keep_rows_full, fit_height, fit_width, "full_rows")
    full_result_2 = run(keep_rows_full, fit_width, fit_height, "full_rows")

    valid_results = [r for r in (trim_result_1, trim_result_2, full_result_1, full_result_2)
                     if r is not None]

    # 兩個都是None
    if len(valid_results) == 0:
        return stop("4_擺放檢查→四種嘗試都失敗")

    fit_ok = any(r["fit_ok"] for r in valid_results)
    new_height = int(round((base_h - pallet_height) / dimensions[cardboard_type][2]))
    new_height = max(0, min(MAX_STACK_LAYERS, new_height))

    vis = {}
    for i, r in enumerate((trim_result_1, trim_result_2, full_result_1, full_result_2)):
        if r is not None:
            vis[f"{r['label']}_{i % 2 + 1}"] = r["vis_large"]

    if new_height >= MAX_STACK_LAYERS:
        r = stop(f"5_換算層數→已達上限 {MAX_STACK_LAYERS} 層，視為滿")
        r.base_h, r.new_height, r.mask, r.vis, r.base_vis = base_h, new_height, mask_u8, vis, base_vis
        return r

    logger.info("trim_rows Fit Status: %s", _fit_status(trim_result_1, trim_result_2, area_thresh))
    logger.info("full_rows Fit Status: %s", _fit_status(full_result_1, full_result_2, area_thresh))
    logger.info("目前將擺放高度: %s", new_height + 1)

    return FitResult(
        fit_ok=fit_ok,
        position_3d=[*position, new_height + 1],
        stage=("5_完成→可擺放" if fit_ok
               else f"5_完成→矩形超出區域超過 {area_thresh * 100:.1f}%"),
        base_h=base_h,
        new_height=new_height,
        mask=mask_u8,
        vis=vis,
        base_vis=base_vis,
    )


def _draw_base_vis(color_img, pc_calib, ref, bound, base_h):
    """把 ref / bound 的 3D 範圍投影回原圖畫出來。原始碼會存檔，這裡只回傳影像。"""
    if color_img is None:
        return None
    try:
        H_img, W_img = color_img.shape[:2]
        pc_structured = pc_calib.reshape(H_img, W_img, 3)
        xs_img, ys_img, zs_img = (pc_structured[:, :, i] for i in range(3))

        ref_mask_2d = ((xs_img > ref.x_min) & (xs_img < ref.x_max)
                       & (ys_img > ref.y_min) & (ys_img < ref.y_max)
                       & (zs_img > ref.z_min) & (zs_img < ref.z_max))
        if not np.any(ref_mask_2d):
            logger.info("在原圖像素中找不到符合 base_h 3D 範圍的點雲")
            return None

        ys_idx, xs_idx = np.where(ref_mask_2d)
        xmin, xmax = int(np.min(xs_idx)), int(np.max(xs_idx))
        ymin, ymax = int(np.min(ys_idx)), int(np.max(ys_idx))

        base_vis = color_img.copy()
        overlay = base_vis.copy()
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 255), -1)
        base_vis = cv2.addWeighted(overlay, 0.25, base_vis, 0.75, 0)
        cv2.rectangle(base_vis, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(base_vis, f"Base_H Sampling Box (Y Median: {base_h:.3f}m)",
                    (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        bound_mask_2d = ((xs_img > bound.x_min) & (xs_img < bound.x_max)
                         & (ys_img > bound.y_min) & (ys_img < bound.y_max)
                         & (zs_img > bound.z_min) & (zs_img < bound.z_max))
        if np.any(bound_mask_2d):
            b_ys, b_xs = np.where(bound_mask_2d)
            cv2.rectangle(base_vis, (int(b_xs.min()), int(b_ys.min())),
                          (int(b_xs.max()), int(b_ys.max())), (255, 0, 255), 2)
            cv2.putText(base_vis, "boundary", (int(b_xs.min()), max(int(b_ys.min()) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        return base_vis
    except Exception as roi_err:
        logger.error("標註 base_h 區域失敗: %s", roi_err)
        return None
