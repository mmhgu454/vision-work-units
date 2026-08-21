"""
viz.py — 畫圖工具

刻意跟 box_measure.py 分開：量測單元不碰 cv2、不碰檔案系統（Q14），
所有視覺化都是「呼叫端拿中間結果自己畫」。這個檔案就是那個呼叫端的一部分。
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import cv2

LABEL_NAMES = {0: "pallet", 1: "top", 2: "face"}

# OpenCV 的 putText 畫不出中文（會變問號），所以圖上用英文代碼，
# 完整的中文階段字串留在 console 與回傳值裡。
STAGE_EN = {
    "1_": "detect", "2_": "select top/face", "3_": "measure LWH", "4_": "match type",
}


def stage_ascii(stage: str) -> str:
    """把中文的 stage 轉成畫得出來的英文標籤。"""
    if stage.isascii():
        return stage
    tag = STAGE_EN.get(stage[:2], "stage " + stage[:1])
    return tag + ("  [FAILED]" if "→" in stage and "完成" not in stage.split("→")[-1] else "")
LABEL_COLORS = {0: (160, 160, 160), 1: (0, 220, 0), 2: (255, 160, 0)}
SELECTED_TOP = (0, 255, 0)
SELECTED_FACE = (255, 200, 0)


def _put(img, text, org, color=(255, 255, 255), scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _rect(img, box, color, thick=2):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)


def draw_detections(color_img, result, *, true_label: str = "") -> np.ndarray:
    """把偵測框、選中的 top/face、量測結果畫在影像上。回傳新影像，不改動輸入。"""
    img = color_img.copy()

    # 所有偵測框（細線）
    if result.boxes is not None and len(result.boxes):
        labels = result.labels if result.labels is not None else [None] * len(result.boxes)
        scores = result.scores if result.scores is not None else [None] * len(result.boxes)
        for box, lab, sc in zip(result.boxes, labels, scores):
            lab_i = int(lab) if lab is not None else -1
            _rect(img, box, LABEL_COLORS.get(lab_i, (200, 200, 200)), 1)
            tag = LABEL_NAMES.get(lab_i, str(lab_i))
            if sc is not None:
                tag += f" {float(sc):.2f}"
            _put(img, tag, (int(box[0]), max(14, int(box[1]) - 6)),
                 LABEL_COLORS.get(lab_i, (200, 200, 200)), 0.5, 1)

    # 被選中的 top / face（粗線）
    if result.top_box is not None:
        _rect(img, result.top_box, SELECTED_TOP, 3)
        _put(img, "TOP", (int(result.top_box[0]) + 4, int(result.top_box[3]) - 8), SELECTED_TOP)
    if result.face_box is not None:
        _rect(img, result.face_box, SELECTED_FACE, 3)
        _put(img, "FACE", (int(result.face_box[0]) + 4, int(result.face_box[3]) - 8), SELECTED_FACE)

    # 左上角資訊面板
    ok_color = (0, 255, 0) if result.ok else (0, 80, 255)
    y = 28
    _put(img, f"type: {result.box_type}", (12, y), ok_color, 0.8); y += 30
    if result.lwh:
        l, w, h = result.lwh
        _put(img, f"L={l:.3f}  W={w:.3f}  H={h:.3f}", (12, y), (255, 255, 255), 0.7); y += 28
    if true_label:
        match = result.box_type == true_label
        _put(img, f"label: {true_label}  {'MATCH' if match else 'MISMATCH'}",
             (12, y), (0, 255, 0) if match else (0, 80, 255), 0.7); y += 28
    if result.match_loss is not None:
        _put(img, f"loss: {result.match_loss:.5f}", (12, y), (200, 200, 200), 0.6); y += 24
    if result.top_plane_normal is not None:
        n = result.top_plane_normal
        _put(img, f"normal: ({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})", (12, y), (200, 200, 200), 0.6); y += 24
    _put(img, f"stage: {stage_ascii(result.stage)}", (12, y), (200, 200, 200), 0.55, 1); y += 26
    if result.warnings:
        _put(img, f"! CLIPPED BY FRAME ({len(result.warnings)})", (12, y), (0, 200, 255), 0.7)
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 200, 255), 6)
    return img


def depth_validity_map(pc_np, color_img=None) -> np.ndarray:
    """有效深度點的分布圖。有效=綠、無效(破洞)=紅，用來判斷取幀品質。"""
    valid = pc_np[..., 2] > 0.01
    vis = np.zeros((*valid.shape, 3), dtype=np.uint8)
    vis[valid] = (0, 160, 0)
    vis[~valid] = (0, 0, 160)
    if color_img is not None:
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        vis = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.6, vis, 0.4, 0)
    _put(vis, f"valid depth: {valid.mean():.1%}", (12, 28), (255, 255, 255), 0.8)
    return vis


def fit_to_screen(img, max_w=1280, max_h=760) -> np.ndarray:
    """縮到螢幕放得下的尺寸，只縮不放。"""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return img if s >= 1.0 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- task2 對準

AIM_REF_COLOR = (0, 255, 255)      # 黃：基準框
AIM_BOUND_COLOR = (255, 0, 255)    # 紫：搜尋範圍


def _roi_bbox_2d(mask_2d):
    """把一個 3D ROI 的像素遮罩壓成 2D 外接框。沒有任何像素命中就回 None。"""
    if not mask_2d.any():
        return None
    ys, xs = np.where(mask_2d)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_task2_aim(color_img, pc_calib, ref, bound, *, min_ref_points=50, extra=()):
    """
    task2 的即時對準疊圖。**刻意做得便宜**：只投影兩個 ROI 並算基準高度，
    不跑完整的 check_fit（那在 1280x800 要 0.5 秒以上，每幀跑不動）。

    回傳 (疊圖, base_h 或 None, 基準框內的點數)。
    """
    import fit_check as fcm

    h, w = color_img.shape[:2]
    pc_img = pc_calib.reshape(h, w, 3)
    img = color_img.copy()

    ref_mask = fcm.roi_mask(pc_img, ref)
    n_ref = int(ref_mask.sum())
    base_h = float(np.median(pc_img[ref_mask][:, 1])) if n_ref else None

    box = _roi_bbox_2d(fcm.roi_mask(pc_img, bound))
    if box:
        cv2.rectangle(img, box[:2], box[2:], AIM_BOUND_COLOR, 2)
        _put(img, "bound", (box[0], max(box[1] - 8, 18)), AIM_BOUND_COLOR, 0.5, 1)

    box = _roi_bbox_2d(ref_mask)
    if box:
        overlay = img.copy()
        cv2.rectangle(overlay, box[:2], box[2:], AIM_REF_COLOR, -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.rectangle(img, box[:2], box[2:], AIM_REF_COLOR, 2)

    # 字級跟著畫面寬度走，這樣小圖上文字不會蓋掉 ROI 框
    k = max(0.35, min(1.0, w / 1280.0))
    enough = n_ref >= min_ref_points
    y = int(28 * k)
    _put(img, f"ref points: {n_ref}" + ("" if enough else f"  < {min_ref_points} 不足"),
         (12, y), (0, 255, 0) if enough else (0, 80, 255), 0.75 * k, max(1, int(2 * k)))
    y += int(30 * k)
    _put(img, f"base_h: {base_h:.4f} m" if base_h is not None else "base_h: --",
         (12, y), (255, 255, 255), 0.75 * k, max(1, int(2 * k)))
    y += int(30 * k)
    valid = float((pc_img[..., 2] != 0).mean())
    _put(img, f"valid depth: {valid:.1%}", (12, y), (200, 200, 200), 0.6 * k, 1)
    y += int(26 * k)
    for line in extra:
        _put(img, line, (12, y), (200, 200, 200), 0.6 * k, 1)
        y += int(24 * k)

    if not enough:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 80, 255), 6)
    return img, base_h, n_ref


# ---------------------------------------------------------------- task1 高度圖

def draw_hmap(hmap, block=None, place_position=None, *, size=420, layers=3):
    """
    把高度圖畫成彩色圖。-inf（沒有點）畫成深灰。

    block 給了就在對應的區域上疊層數字；place_position 那格加綠框。
    """
    valid = np.isfinite(hmap)
    vis = np.zeros((*hmap.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = float(hmap[valid].min()), float(hmap[valid].max())
        norm = np.zeros_like(hmap, dtype=np.float32)
        if hi - lo > 1e-9:
            norm[valid] = (hmap[valid] - lo) / (hi - lo)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[valid] = colored[valid]
    vis[~valid] = (40, 40, 40)

    vis = cv2.resize(vis, (size, size), interpolation=cv2.INTER_NEAREST)

    if block is not None:
        rows, cols = block.shape
        ch, cw = size / rows, size / cols
        for r in range(rows):
            for c in range(cols):
                x0, y0 = int(c * cw), int(r * ch)
                x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
                full = int(block[r, c]) >= layers
                cv2.rectangle(vis, (x0, y0), (x1, y1), (90, 90, 90), 1)
                _put(vis, str(int(block[r, c])), (x0 + int(cw / 2) - 8, y0 + int(ch / 2) + 8),
                     (0, 80, 255) if full else (255, 255, 255), 0.9)
        if place_position is not None:
            r, c = place_position[0] - 1, place_position[1] - 1
            cv2.rectangle(vis, (int(c * cw) + 2, int(r * ch) + 2),
                          (int((c + 1) * cw) - 2, int((r + 1) * ch) - 2), (0, 255, 0), 3)

    _put(vis, "height map", (10, 24), (255, 255, 255), 0.6, 1)
    return vis


def draw_task1_aim(color_img, hmap, block, place_position, *, extra=()):
    """task1 的即時對準疊圖：左邊原圖、右邊高度圖。"""
    h, w = color_img.shape[:2]
    panel = draw_hmap(hmap, block, place_position, size=h)

    img = color_img.copy()
    k = max(0.35, min(1.0, w / 1280.0))
    y = int(28 * k)
    if place_position is None:
        _put(img, "ALL FULL", (12, y), (0, 80, 255), 0.9 * k, max(1, int(2 * k)))
    else:
        _put(img, f"place: row {place_position[0]}  col {place_position[1]} "
                  f" layer {place_position[2]}", (12, y), (0, 255, 0), 0.75 * k,
             max(1, int(2 * k)))
    y += int(30 * k)
    valid = float(np.isfinite(hmap).mean())
    _put(img, f"hmap coverage: {valid:.1%}", (12, y), (200, 200, 200), 0.6 * k, 1)
    y += int(26 * k)
    for line in extra:
        _put(img, line, (12, y), (200, 200, 200), 0.6 * k, 1)
        y += int(24 * k)

    return np.hstack([img, panel])
