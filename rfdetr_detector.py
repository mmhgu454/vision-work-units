"""
rfdetr_detector.py — RF-DETR 偵測元件（本專案的實作）

職責（Q13c 的另一半）：像素空間的旋轉關在這裡面。
    輸入影像 --rot90(k)--> 模型看得懂的正立影像 --預測--> 框 --轉回去--> 與輸入同方向的框

「轉過去」和「轉回來」是一對必須同時修改的操作，所以刻意關在同一個檔案裡，
物理上不可能改了一邊忘記另一邊。

對外契約（box_measure 只認這個）：
    detect(color_img) -> (boxes, labels, scores)，框與 color_img 同方向。

換模型時只要換掉這個檔案，box_measure 一行都不用動。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "model" / "checkpoint_0702.pth"
DEFAULT_CONF_THRES = 0.5        # 對齊 src/config.py 的 MODEL_CFG.conf_thres
TARGET_CLASSES = (0, 1, 2)      # 0: pallet, 1: top, 2: face


def rot90_boxes(boxes, img_h: int, img_w: int, k: int) -> np.ndarray:
    """把一批 xyxy 框旋轉 k 次 90 度（正值逆時針，對應 np.rot90）。"""
    k = k % 4
    out = []
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box)
        h, w = img_h, img_w
        for _ in range(k):
            x1, y1, x2, y2 = y1, w - 1 - x2, y2, w - 1 - x1
            h, w = w, h
        out.append([x1, y1, x2, y2])
    return np.asarray(out, dtype=float).reshape(-1, 4)


class RFDETRDetector:
    """把 RF-DETR 包成 box_measure 要的 detect(color_img) callable。"""

    def __init__(self, weights=DEFAULT_WEIGHTS, num_classes: int = 3,
                 conf_thres: float = DEFAULT_CONF_THRES, view_rot_k: int = 2,
                 target_classes=TARGET_CLASSES, model_cls: str = "RFDETRMedium"):
        """
        view_rot_k : 把輸入影像轉成模型看得懂的正立方向所需的 rot90 次數。
                     必須與 measure_box(view_rot_k=...) 傳的值相同。
        """
        from rfdetr import RFDETRMedium, RFDETRNano  # 延後 import，載入很慢
        cls = {"RFDETRMedium": RFDETRMedium, "RFDETRNano": RFDETRNano}[model_cls]

        self.view_rot_k = view_rot_k
        self.conf_thres = conf_thres
        self.target_classes = tuple(target_classes)
        logger.info("載入模型 %s ...", weights)
        self.model = cls(pretrain_weights=str(weights), num_classes=num_classes)

    def __call__(self, color_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_h, img_w = color_img.shape[:2]

        # 1. 轉成模型看得懂的正立方向
        upright = np.rot90(color_img, k=self.view_rot_k).copy()

        # 2. 推論（等同 src/stage4/task1.py 的 get_prediction）
        predictions = self.model.predict(upright, confidence=self.conf_thres)
        boxes = predictions.xyxy
        scores = predictions.confidence
        labels = predictions.class_id
        mask = np.isin(labels, self.target_classes)
        boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

        # 3. 把框轉回輸入影像的方向。
        #    upright 的尺寸在 k 為奇數時與輸入對調，所以要用 upright 自己的尺寸。
        up_h, up_w = upright.shape[:2]
        boxes = rot90_boxes(boxes, up_h, up_w, -self.view_rot_k)

        return boxes, labels, scores


DEFAULT_PERSON_WEIGHTS = Path(__file__).resolve().parent / "model" / "person-nano.pth"
PERSON_PREDICT_CONF = 0.8      # 對齊 task1.get_peoples_detect
PERSON_SCORE_MIN = 0.6
PERSON_CLASS_ID = 1


class RFDETRPeopleDetector:
    """
    人員偵測，包成 pallet_stack 要的 detect_people(color_img) callable。

    回傳 (人數, boxes, scores)。框與傳入的 color_img 同方向，
    旋轉的來回一樣關在這個類別裡，與 RFDETRDetector 同一套契約。
    """

    def __init__(self, weights=DEFAULT_PERSON_WEIGHTS, view_rot_k: int = 1,
                 predict_conf: float = PERSON_PREDICT_CONF,
                 score_min: float = PERSON_SCORE_MIN,
                 class_id: int = PERSON_CLASS_ID):
        from rfdetr import RFDETRNano
        self.view_rot_k = view_rot_k
        self.predict_conf = predict_conf
        self.score_min = score_min
        self.class_id = class_id
        logger.info("載入人員模型 %s ...", weights)
        self.model = RFDETRNano(pretrain_weights=str(weights))

    def __call__(self, color_img: np.ndarray):
        upright = np.rot90(color_img, k=self.view_rot_k).copy()
        predictions = self.model.predict(upright, confidence=self.predict_conf)
        boxes, scores, labels = (predictions.xyxy, predictions.confidence,
                                 predictions.class_id)
        mask = (scores >= self.score_min) & (labels == self.class_id)
        boxes, scores = boxes[mask], scores[mask]

        up_h, up_w = upright.shape[:2]
        boxes = rot90_boxes(boxes, up_h, up_w, -self.view_rot_k)
        return len(boxes), boxes, scores
