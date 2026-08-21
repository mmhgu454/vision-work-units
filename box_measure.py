"""
box_measure.py — 紙箱尺寸量測單元（可攜）

把這一個檔案複製到別的專案就能用。相依只有 numpy。

來源
----
fork 自 fine_good_unit 的:
  - src/stage4/task3.py            （全部數學，一行未改）
  - src/vision_engine.py           （process_stage4_task3 的流程部分）

複製過來時移除的東西（皆不影響回傳值）:
  - 4 個裸 print()
  - cv2.minAreaRect 那條鏈：它的結果 ool/oow 只餵給 print，不參與計算
  - from config import ... / from log_manager import ...（改為內建預設值與 stdlib logging）
  - 存 debug 圖的副作用（改為回傳中間結果，由呼叫端自行處理）

呼叫契約
--------
pc_np : (H, W, 3) float ndarray
    座標系必須與來源專案 task3 收到的一致：
        Z 軸旋轉 180 度後的相機座標系（X、Y 皆翻轉，Z 為深度、往前為正），
        未做傾角校正、未做高度偏移。
    在來源專案裡這是由 realsense_source.py 產生的。

detect : Callable[[ndarray], (boxes, labels, scores)]
    回傳的框必須與傳入的 color_img 同方向（同一個像素座標系）。
    label 語意：1 = 箱子頂面(top)，2 = 箱子正面(face)。

view_rot_k : int
    把「傳入影像的方向」轉成「正立視角」所需的 np.rot90 次數。
    正立視角 = 箱子的 face 在 top 的下方。來源專案的相機是倒裝的，故為 2。
    相機正裝的專案請設 0。

用法
----
    import numpy as np
    from box_measure import measure_box, DEFAULT_BOX_DIMENSIONS

    # detect 回傳 (boxes, labels, scores)。label: 1=箱子頂面, 2=箱子正面。
    # 框的座標必須跟傳進來的 img 同方向。
    def detect(img):
        ...

    result = measure_box(color_img, pc_np, detect, view_rot_k=0)

    if result.trustworthy:            # 成功且沒有可疑之處
        print(result.box_type)        # 例如 "L554001"
        print(result.lwh)             # (長, 寬, 高)，單位公尺
    else:
        print(result.stage)           # 停在哪一步
        for w in result.warnings:     # 例如「框貼到畫面下緣」
            print(w)

要換尺寸表就傳 dimensions={"MYBOX": (0.5, 0.4, 0.3), ...}。
契約違反（點雲形狀不對、相機有 roll、影像與點雲解析度不符）會拋 ContractError。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 可覆寫設定

#: 箱型 -> (length, width, height)，單位公尺。可用 measure_box(dimensions=...) 覆寫。
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

#: 箱型比對的總平方誤差上限（0.0036 = 各邊誤差合計約六公分）
DEFAULT_MATCH_LOSS_THRESHOLD = 0.0036

#: 偵測框距離畫面邊緣幾個像素內就視為「被畫面切掉」。
#: face 框被下緣切掉時，箱子下緣不在畫面裡，量到的高度會偏小但看起來仍合理。
DEFAULT_EDGE_MARGIN_PX = 3

#: 座標系契約檢查：頂面法向量的 X 分量絕對值上限。
#: get_orthonormal_basis 把基準向量寫死成 camera_right=(1,0,0)，
#: 這只有在相機沒有 roll（法向量 X 分量接近 0）時才成立。
#: 0.30 約對應 17 度的 roll，是保守值，請依實測收緊。
DEFAULT_NO_ROLL_TOLERANCE = 0.30


class ContractError(ValueError):
    """輸入資料違反本單元的座標系契約。"""


# ---------------------------------------------------------------- 回傳型別

@dataclass
class BoxMeasurement:
    """量測結果 + 中間產物。呼叫端可用這些自行畫 debug 圖或做斷言。"""

    box_type: str                                    # 箱型名稱，或 "Err"
    stage: str                                       # 走到哪一步（失敗時看這個）
    lwh: Optional[Tuple[float, float, float]] = None  # 量到的長寬高（公尺）
    match_loss: Optional[float] = None                # 比對到最近箱型的平方誤差
    top_box: Optional[Sequence[float]] = None         # 已轉回輸入影像方向
    face_box: Optional[Sequence[float]] = None        # 已轉回輸入影像方向
    boxes: Optional[np.ndarray] = None                # detect 的原始輸出
    labels: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    top_plane_normal: Optional[np.ndarray] = None     # 頂面平面法向量
    warnings: List[str] = field(default_factory=list)  # 結果可疑的理由（不中斷流程）

    @property
    def ok(self) -> bool:
        """比對出了箱型（不是 "Err"）。不保證沒有警告，那個看 trustworthy。"""
        return self.box_type != "Err"

    @property
    def trustworthy(self) -> bool:
        """量測成功且沒有任何可疑之處。"""
        return self.ok and not self.warnings


# ---------------------------------------------------------------- 頂層流程

def measure_box(
    color_img: np.ndarray,
    pc_np: np.ndarray,
    detect: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    dimensions: Dict[str, Tuple[float, float, float]] = None,
    view_rot_k: int = 2,
    match_loss_threshold: float = DEFAULT_MATCH_LOSS_THRESHOLD,
    no_roll_tolerance: float = DEFAULT_NO_ROLL_TOLERANCE,
    edge_margin_px: int = DEFAULT_EDGE_MARGIN_PX,
) -> BoxMeasurement:
    """跑完一次紙箱型號判斷。不碰相機、不碰檔案系統、不寫檔。"""

    if dimensions is None:
        dimensions = DEFAULT_BOX_DIMENSIONS

    if pc_np is None or pc_np.ndim != 3 or pc_np.shape[2] < 3:
        raise ContractError(f"pc_np 必須是 (H, W, 3)，收到 {None if pc_np is None else pc_np.shape}")

    img_h, img_w = pc_np.shape[:2]

    if color_img is not None and color_img.shape[:2] != (img_h, img_w):
        raise ContractError(
            f"color_img {color_img.shape[:2]} 與 pc_np {(img_h, img_w)} 解析度不一致；"
            " 偵測框是用 color_img 的像素座標，卻要拿去索引 pc_np，兩者必須對齊"
            "（來源專案是靠 rs.align 對齊的）。"
        )

    # 偵測器若有公開自己的 view_rot_k，就檢查兩邊是否一致。
    # 兩者必須相同，否則框與點雲會靜默錯位、量出看似合理但錯誤的尺寸。
    # 用 getattr 而非 isinstance，所以不會對任何特定偵測器實作產生相依。
    detect_k = getattr(detect, "view_rot_k", None)
    if detect_k is not None and detect_k % 4 != view_rot_k % 4:
        raise ContractError(
            f"偵測器的 view_rot_k={detect_k} 與 measure_box 的 view_rot_k={view_rot_k} 不一致。"
            " 兩者必須相同，否則偵測框與點雲會錯位。"
        )

    # 1. AI 偵測（框與 color_img 同方向）
    boxes, labels, scores = detect(color_img)
    result = BoxMeasurement(box_type="Err", stage="1_偵測完成",
                            boxes=boxes, labels=labels, scores=scores)

    if boxes is None or len(boxes) == 0:
        result.stage = "1_偵測完成→沒有任何偵測框"
        return result

    # 2. 挑 top / face。
    #    select_top_and_face 的判準（face 在 top 下方、ROI 中心偏左）是對「正立視角」
    #    而言的，所以先把框轉成正立視角再挑，挑完再轉回輸入影像的方向，
    #    因為 pc_np 是用輸入影像的像素座標索引的。
    # k 為奇數時，正立視角的畫面長寬會對調，轉回去時必須用對調後的尺寸
    up_h, up_w = (img_h, img_w) if view_rot_k % 2 == 0 else (img_w, img_h)

    upright_boxes = [_rot90_box(b, img_h, img_w, view_rot_k) for b in boxes]
    top_box_up, face_box_up = select_top_and_face(upright_boxes, labels, up_h, up_w)

    if top_box_up is None or face_box_up is None:
        result.stage = "2_挑選top/face→找不到成對的 top 與 face"
        return result

    top_box = _rot90_box(top_box_up, up_h, up_w, -view_rot_k)
    face_box = _rot90_box(face_box_up, up_h, up_w, -view_rot_k)
    result.top_box, result.face_box = top_box, face_box
    result.stage = "2_挑選top/face完成"

    # 被畫面邊緣切掉的框會讓量測值偏小，但數字本身看起來仍然合理。
    # 不中斷流程（有時仍堪用），但一定要講出來。
    for name, box in (("top", top_box), ("face", face_box)):
        clipped = _clipped_edges(box, img_h, img_w, edge_margin_px)
        if clipped:
            msg = (f"{name}_box 貼到畫面{'、'.join(clipped)}，物體被裁切，"
                   f"{'高度' if name == 'face' else '長寬'}會偏小")
            result.warnings.append(msg)
            logger.warning("%s", msg)

    # 3. 契約檢查：頂面法向量的 X 分量必須接近 0（相機無 roll）
    normal = _fit_top_plane_normal(pc_np, top_box)
    if normal is not None:
        result.top_plane_normal = normal
        if abs(float(normal[0])) > no_roll_tolerance:
            raise ContractError(
                f"頂面法向量 X 分量 {float(normal[0]):.3f} 超過容許值 {no_roll_tolerance}。"
                " get_orthonormal_basis 把基準向量寫死為 (1,0,0)，只在相機無 roll 時成立；"
                " 目前的點雲不滿足此契約，量出的長寬高會是錯的。"
                " 請檢查 pc_np 的座標系，或在確認無誤後放寬 no_roll_tolerance。"
            )

    # 4. 尺寸計算
    lwh = calculate_box_lwh(pc_np, top_box, face_box)
    if lwh is None or lwh[0] is None:
        result.stage = "3_計算長寬高→點雲不足或平面擬合失敗"
        return result

    box_l, box_w, box_h = lwh
    result.lwh = (float(box_l), float(box_w), float(box_h))
    result.stage = "3_計算長寬高完成"

    # 5. 比對箱型
    box_type, loss = find_closest_box_type(
        box_l, box_w, box_h, dimensions=dimensions, match_loss_threshold=match_loss_threshold
    )
    result.box_type = box_type
    result.match_loss = loss
    result.stage = (f"4_比對完成→{box_type}" if box_type != "Err"
                    else f"4_比對完成→誤差 {loss:.5f} 超過門檻 {match_loss_threshold}")
    return result


def _clipped_edges(box, img_h: int, img_w: int, margin: int) -> List[str]:
    """回傳這個框貼到了哪幾個畫面邊緣。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    edges = []
    if x1 <= margin:
        edges.append("左緣")
    if y1 <= margin:
        edges.append("上緣")
    if x2 >= img_w - 1 - margin:
        edges.append("右緣")
    if y2 >= img_h - 1 - margin:
        edges.append("下緣")
    return edges


def _rot90_box(box, img_h: int, img_w: int, k: int):
    """把一個 xyxy 框旋轉 k 次 90 度（正值為逆時針，對應 np.rot90）。"""
    if box is None:
        return None
    k = k % 4
    x1, y1, x2, y2 = (float(v) for v in box)
    for _ in range(k):
        # 逆時針 90 度：(x, y) -> (y, W-1-x)，來源與 vision_engine 的還原公式一致
        x1, y1, x2, y2 = y1, img_w - 1 - x2, y2, img_w - 1 - x1
        img_h, img_w = img_w, img_h
    return [x1, y1, x2, y2]


def _fit_top_plane_normal(pc_np, top_box):
    """只為了契約檢查而重跑一次頂面擬合；失敗就回 None，不影響主流程。"""
    try:
        pc_xyz = pc_np[..., :3]
        img_h, img_w, _ = pc_xyz.shape
        x1_t, y1_t, x2_t, y2_t = map(int, top_box)
        x1_t, x2_t = max(0, min(x1_t, img_w)), max(0, min(x2_t, img_w))
        y1_t, y2_t = max(0, min(y1_t, img_h)), max(0, min(y2_t, img_h))
        if x2_t <= x1_t or y2_t <= y1_t:
            return None
        top_roi = pc_xyz[y1_t:y2_t, x1_t:x2_t]
        h_t, w_t, _ = top_roi.shape
        dy_t, dx_t = int(h_t * 0.05), int(w_t * 0.05)
        if h_t - 2 * dy_t <= 0 or w_t - 2 * dx_t <= 0:
            return None
        top_pts = top_roi[dy_t:h_t - dy_t, dx_t:w_t - dx_t].reshape(-1, 3)
        valid_top = top_pts[(top_pts[:, 2] > 0.01) & (~np.isnan(top_pts).any(axis=1))]
        if len(valid_top) < 15:
            return None
        normal, _, inliers = fit_plane_ransac_svd(valid_top)
        if normal is None or len(inliers) < 3:
            return None
        return normal
    except Exception:
        return None


# ============================================================================
# 以下為 src/stage4/task3.py 的數學，逐字複製。
# 唯一的改動：
#   - find_closest_box_type 改為接受 dimensions/threshold 參數，並一併回傳誤差值
#   - 移除 4 個裸 print()
#   - 移除 cv2.minAreaRect 那條鏈（其結果只餵給 print）
#   - 移除約 50 行註解掉的舊實驗程式碼
# 計算邏輯本身一行未改。
# ============================================================================

def find_closest_box_type(
    l, w, h,
    dimensions: Dict[str, Tuple[float, float, float]] = None,
    match_loss_threshold: float = DEFAULT_MATCH_LOSS_THRESHOLD,
) -> Tuple[str, float]:
    """
    根據偵測到的尺寸，尋找最匹配的箱型名稱。回傳 (箱型, 平方誤差)。

    註（沿用原始碼作者的說明）：上位會給紙箱 type，實務上不需要比對；
    保留此函數是為了讓本單元可以獨立驗證量測結果。
    """
    if dimensions is None:
        dimensions = DEFAULT_BOX_DIMENSIONS

    min_loss = 9999
    best_type = "NONE"
    for box_type, (bl, bw, bh) in dimensions.items():
        loss = (bl - l) ** 2 + (bw - w) ** 2 + (bh - h) ** 2
        if loss <= min_loss:
            min_loss = loss
            best_type = box_type

    if min_loss >= match_loss_threshold:   # 總誤差不超過六公分
        return "Err", min_loss
    return best_type, min_loss


def fit_plane_ransac_svd(pts, distance_threshold=0.015, max_iterations=100):
    """
    用 RANSAC 找出頂面點雲的平面方程式。
    回傳: unit_normal (A,B,C), d_parameter (D), 以及內點索引
    """
    n_pts = len(pts)
    if n_pts < 3:
        return None, None, None

    best_inliers = []

    for _ in range(max_iterations):
        idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts[idx]

        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)

        distances = np.abs(np.dot(pts, normal) + d)
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 10:
        return None, None, None

    # 使用 SVD (最小平方法) 重新校準平面，拿到極度精準的法向量
    inlier_pts = pts[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    pts_centered = inlier_pts - centroid
    cov_matrix = np.dot(pts_centered.T, pts_centered)

    # 對 3x3 的矩陣做 SVD，瞬間完成且不佔記憶體
    _, _, vh = np.linalg.svd(cov_matrix)

    refined_normal = vh[2, :]  # SVD 的最後一列即為最小變異方向 (法向量)
    refined_normal = refined_normal / np.linalg.norm(refined_normal)
    refined_d = -np.dot(refined_normal, centroid)

    # 重新計算最終符合的內點
    final_distances = np.abs(np.dot(pts, refined_normal) + refined_d)
    final_inliers = np.where(final_distances < 0.6 * distance_threshold)[0]

    return refined_normal, refined_d, final_inliers


def get_orthonormal_basis(normal):
    """
    根據平面法向量，建立一個正交的 2D 平面座標系基底 (u, v)

    ⚠ 座標系契約：n[0] 必須 near 0（相機沒有 roll，x 分量為 0），
      基準向量才可以寫死。measure_box() 會在呼叫前檢查這件事，
      見 DEFAULT_NO_ROLL_TOLERANCE。
    """
    n = normal / np.linalg.norm(normal)
    camera_right = np.array([1.0, 0.0, 0.0])
    camera_down = np.array([0.0, 1.0, 0.0])

    # 正交投影，使 u 垂直於 n
    u = camera_right - np.dot(camera_right, n) * n
    if np.linalg.norm(u) < 1e-6:
        # 極端情況：法向量剛好平行水平軸，退回用垂直軸
        u = camera_down - np.dot(camera_down, n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def calculate_box_lwh(pc_np, top_box, face_box):
    """
    完全不依賴外部基準的 LWH 計算器 (自適應高度版，輸出單位：公尺 m)
    """
    if pc_np is None or len(pc_np.shape) < 3 or pc_np.shape[2] < 3:
        return None

    pc_xyz = pc_np[..., :3]
    img_h, img_w, _ = pc_xyz.shape

    # ====================================================
    # 1. 頂面 (Top) 計算：長 L 與 寬 W
    # ====================================================
    x1_t, y1_t, x2_t, y2_t = map(int, top_box)
    x1_t, x2_t = max(0, min(x1_t, img_w)), max(0, min(x2_t, img_w))
    y1_t, y2_t = max(0, min(y1_t, img_h)), max(0, min(y2_t, img_h))

    if x2_t <= x1_t or y2_t <= y1_t:
        return None, None, None

    top_roi = pc_xyz[y1_t:y2_t, x1_t:x2_t]
    h_t, w_t, _ = top_roi.shape
    dy_t, dx_t = int(h_t * 0.05), int(w_t * 0.05)

    if h_t - 2 * dy_t <= 0 or w_t - 2 * dx_t <= 0:
        return None, None, None

    top_pts = top_roi[dy_t: h_t - dy_t, dx_t: w_t - dx_t].reshape(-1, 3)
    valid_top = top_pts[(top_pts[:, 2] > 0.01) & (~np.isnan(top_pts).any(axis=1))]
    if len(valid_top) < 15:
        return None, None, None

    normal, d_param, inliers = fit_plane_ransac_svd(valid_top)
    if normal is None or len(inliers) < 3:
        return None, None, None

    inlier_pts = valid_top[inliers]
    u, v = get_orthonormal_basis(normal)
    centroid = np.mean(inlier_pts, axis=0)
    proj_2d = np.dot(inlier_pts - centroid, np.column_stack((u, v)))

    if len(proj_2d) < 3:
        return None, None, None

    # 用協方差矩陣特徵分解(PCA)找主軸方向，取代 minAreaRect(凸包法)。
    # 這跟算 normal 的方法(協方差矩陣 SVD)是同一套邏輯：
    # 用「所有 inlier 點」加權擬合方向，而不是被邊界極值點單獨拍板決定，
    # 因此對邊緣雜訊、摺痕凸起、flying pixel 有抵抗力，跟凸包法完全不同等級。
    centroid_2d = np.mean(proj_2d, axis=0)
    centered = proj_2d - centroid_2d
    cov_2d = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_2d)  # 由小到大排序

    dir0 = eigvecs[:, 1]  # 變異數較大的主軸方向
    dir1 = eigvecs[:, 0]  # 與 dir0 正交的次軸方向

    coord0 = centered @ dir0
    coord1 = centered @ dir1

    # 這一步才真正對應到 face 的「percentile 裁剪 distances_m」邏輯：
    # 差別在於，這裡的軸是用協方差法(跟 normal 同一套方法)求出來的，
    # 不是被邊界離群點污染過的軸，裁剪才有意義、才抓得到真正的離群值
    c0_low, c0_high = np.percentile(coord0, [2, 98])
    c1_low, c1_high = np.percentile(coord1, [2, 98])
    len0 = c0_high - c0_low
    len1 = c1_high - c1_low

    # 用 dir0 跟水平軸(u軸)的夾角，判斷 len0 對應垂直邊(長)還是水平邊(寬)
    angle0 = abs(np.degrees(np.arctan2(dir0[1], dir0[0])))
    angle0 = angle0 if angle0 <= 90 else 180 - angle0

    if angle0 > 45:
        l_m, w_m = len0, len1
    else:
        l_m, w_m = len1, len0

    # ====================================================
    # 2. 正面 (Face) 計算：高度 H
    # ====================================================
    x1_f, y1_f, x2_f, y2_f = map(int, face_box)
    x1_f, x2_f = max(0, min(x1_f, img_w)), max(0, min(x2_f, img_w))
    y1_f, y2_f = max(0, min(y1_f, img_h)), max(0, min(y2_f, img_h))

    if x2_f <= x1_f or y2_f <= y1_f:
        return None, None, None

    # 左右內縮，避免切到背景牆面
    w_f = x2_f - x1_f
    dx_f = int(w_f * 0.08)
    x1_f_inner = max(0, min(x1_f + dx_f, img_w))
    x2_f_inner = max(0, min(x2_f - dx_f, img_w))

    face_roi = pc_xyz[y1_f:y2_f, x1_f_inner:x2_f_inner]
    face_pts = face_roi.reshape(-1, 3)
    valid_face = face_pts[(face_pts[:, 2] > 0.01) & (~np.isnan(face_pts).any(axis=1))]
    if len(valid_face) < 30:
        return None, None, None

    # 對正面做第二次 RANSAC+SVD，把「真正屬於箱子正面」的點抓出來
    # （排除背景牆、地板延伸、flying pixels）
    face_normal, face_d, face_inliers_idx = fit_plane_ransac_svd(
        valid_face, distance_threshold=0.01, max_iterations=150
    )

    if face_normal is not None and len(face_inliers_idx) >= 20:
        # 用法向量夾角驗證：正面法向量應該接近垂直於頂面法向量
        # (兩向量內積接近 0 代表兩個平面互相垂直，符合物理直覺)
        cos_angle = abs(np.dot(face_normal, normal))
        if cos_angle < 0.5:
            clean_face_pts = valid_face[face_inliers_idx]
        else:
            # 擬合到的平面跟頂面太平行，代表可能抓到別的水平面（地板？）
            # 退回用全部點，避免直接回傳 None 中斷流程
            clean_face_pts = valid_face
    else:
        clean_face_pts = valid_face

    # 計算「乾淨」正面點到頂面平面的垂直距離
    distances_m = np.abs(np.dot(clean_face_pts, normal) + d_param)

    # 用 2~98 百分位「差值」取代單純 percentile(95)
    # 這樣同時對抗上緣（應該=0附近）跟下緣（應該=H附近）兩端的雜訊
    low_m = np.percentile(distances_m, 2)
    high_m = np.percentile(distances_m, 98)

    h_m = high_m - low_m

    return l_m, w_m, h_m


def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def select_top_and_face(filtered_boxes, filtered_labels, img_h, img_w):
    """
    挑選最靠近畫面中心的 top_box，
    再從「位於該 top_box 下方」的 face 候選框中，
    挑選與 top_box 空間上最接近的 face_box。
    箱子擺正、相機視角固定時，face 理論上一定在 top 的下方，
    用這個限制可以避免多箱子情境下配對錯誤。

    ⚠ 傳入的框必須是「正立視角」的，否則「face 在 top 下方」不成立。
      measure_box() 會先用 view_rot_k 轉好再呼叫本函數。
    """

    # w: 左小右大 h: 上小下大
    roi_center = (img_w * 0.4, img_h * 0.5)

    top_candidates = [(b, _box_center(b)) for b, l in zip(filtered_boxes, filtered_labels) if l == 1]
    face_candidates = [(b, _box_center(b)) for b, l in zip(filtered_boxes, filtered_labels) if l == 2]

    if not top_candidates or not face_candidates:
        return None, None

    # 1. 挑最靠近畫面中心的 top_box
    top_box, top_center = min(top_candidates, key=lambda item: _dist2(item[1], roi_center))
    top_y2 = top_box[3]  # top_box 的下緣 y 座標

    # 2. 只保留「中心點 y 座標在 top_box 下緣以下」的 face 候選
    #    （允許一點誤差容忍，避免因框稍微重疊而被誤濾掉）
    tolerance_px = 5
    valid_face_candidates = [
        (b, c) for b, c in face_candidates if c[1] >= top_y2 - tolerance_px
    ]

    # 如果套用限制後沒有任何候選框（理論上不該發生，保底防呆）
    if not valid_face_candidates:
        valid_face_candidates = face_candidates

    # 3. 從符合「在 top 下方」的候選中，挑離 top_box 中心最近的
    face_box, _ = min(valid_face_candidates, key=lambda item: _dist2(item[1], top_center))

    # TODO: 要加一個沒偵測到箱子和偵測失敗的差異（empty vs error）

    return top_box, face_box
