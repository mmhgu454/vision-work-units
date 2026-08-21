"""
realsense_source.py — RealSense 取幀元件（本專案的實作）

職責（Q13c 的一半）：把相機吐出來的東西轉成 box_measure 要的座標系。
出去的是純 numpy，box_measure 因此完全不需要認識 pyrealsense2。

座標系轉換沿用 src/vision_engine.py 的 process_stage4_task3：
    pc.calculate -> get_vertices -> stack -> 繞 Z 軸轉 180 度 -> reshape(h, w, 3)
未做傾角校正、未做高度偏移，與來源專案一致。

換相機時只要換掉這個檔案，box_measure 一行都不用動。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple
import logging

import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 1280      # 對齊 src/config.py 的 CAM_HW_CFG
DEFAULT_HEIGHT = 800
DEFAULT_FPS = 15
WARMUP_FRAMES = 30        # 對齊 camera_manager._camera_worker 的暖機幀數


def get_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """歐拉角 (XYZ) 轉旋轉矩陣。與 src/camera_manager.py 的實作相同，內聯以避免相依。"""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


#: box_measure 契約要求的座標系修正：繞 Z 軸 180 度。
#: 這是「相機怎麼架」的事實，不是演算法的一部分，所以是 RealSenseSource 的參數。
DEFAULT_PC_ROTATION_DEG = (0.0, 0.0, 180.0)
R_FIX = get_rotation_matrix(*DEFAULT_PC_ROTATION_DEG)


class RealSenseSource:
    """開一條 RealSense 串流，吐出 (color_img, pc_np)。可當 context manager 用。"""

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS,
                 warmup_frames=WARMUP_FRAMES, pc_rotation_deg=DEFAULT_PC_ROTATION_DEG):
        """
        pc_rotation_deg : 套用在點雲上的歐拉角 (rx, ry, rz)，單位度。
                          預設 (0, 0, 180) 沿用現場設定。相機架法不同就改這個，
                          box_measure 一行都不用動。
        """
        self.width, self.height, self.fps = width, height, fps
        self.warmup_frames = warmup_frames
        self.pc_rotation_deg = tuple(pc_rotation_deg)
        self.R_fix = get_rotation_matrix(*self.pc_rotation_deg)
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        self._started = False

    def start(self):
        if self._started:
            return self
        self.rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.rs_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipeline.start(self.rs_config)
        self._started = True
        logger.info("RealSense 已啟動，暖機 %d 幀...", self.warmup_frames)
        for _ in range(self.warmup_frames):
            try:
                self.pipeline.wait_for_frames()
            except Exception:
                pass
        logger.info("暖機完成。")
        return self

    def get_frame(self, timeout_ms: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """回傳 (color_img BGR (H,W,3) uint8, pc_np (H,W,3) float32)。"""
        if not self._started:
            raise RuntimeError("尚未 start()")

        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("取不到對齊後的 color/depth frame")

        color_img = np.asanyarray(color_frame.get_data()).copy()
        pc_np = self.to_pointcloud(depth_frame)
        return color_img, pc_np

    def to_pointcloud(self, depth_frame) -> np.ndarray:
        """深度幀 -> box_measure 契約座標系下的 (H, W, 3) 點雲。"""
        pts = self.pc.calculate(depth_frame)
        vtx = np.asanyarray(pts.get_vertices())
        h, w = depth_frame.get_height(), depth_frame.get_width()
        pc_np = np.stack([vtx["f0"], vtx["f1"], vtx["f2"]], axis=-1)
        pc_np = (self.R_fix @ pc_np.T).T
        return pc_np.reshape(h, w, 3)

    def get_frame_task2(self, position_2d, timeout_ms: int = 5000, *,
                        pose: "CameraPose" = None, rotation_deg=None):
        """
        回傳 Task2Frame(color_img, pc_calib, pc_raw, rotation_deg, pose)。

        pose / rotation_deg 不給就用 task2_pose_for(position_2d) 的現場預設值。
        pc_raw 是**還沒套用任何校正**的點雲——存進 fixture 後就能離線重算不同 pose，
        不必為了微調傾角或高度重新拍一次。
        """
        if not self._started:
            raise RuntimeError("尚未 start()")
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("取不到對齊後的 color/depth frame")
        color_img = np.asanyarray(color_frame.get_data()).copy()

        default_rot, default_pose = task2_pose_for(position_2d)
        rotation_deg = default_rot if rotation_deg is None else tuple(rotation_deg)
        pose = default_pose if pose is None else pose

        pts = self.pc.calculate(depth_frame)
        vtx = np.asanyarray(pts.get_vertices())
        pc_raw = np.stack([vtx["f0"], vtx["f1"], vtx["f2"]], axis=-1)
        pc_calib = calibrate_flat(pc_raw, rotation_deg, pose)
        return Task2Frame(color_img, pc_calib, pc_raw, rotation_deg, pose)

    def get_frame_task1(self, timeout_ms: int = 5000, *, rotation_deg=None):
        """
        回傳 (color_img, pc_np)，pc_np 是 (H, W, 3)、只做過 Z 軸 -90 度旋轉，
        **尚未套用傾角/高度校正**——task1 的校正要在演算法內部、z>0 過濾之後才做，
        所以由 pallet_stack 收一個 calibrate callable 自己處理。

        這也表示 task1 的 fixture 存這個 pc_np 就夠了：換一個 calibrator 就能離線調姿態。
        """
        if not self._started:
            raise RuntimeError("尚未 start()")
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("取不到對齊後的 color/depth frame")
        color_img = np.asanyarray(color_frame.get_data()).copy()

        pts = self.pc.calculate(depth_frame)
        vtx = np.asanyarray(pts.get_vertices())
        h, w = depth_frame.get_height(), depth_frame.get_width()
        pc_np = np.stack([vtx["f0"], vtx["f1"], vtx["f2"]], axis=-1)
        rotation_deg = TASK1_ROTATION_DEG if rotation_deg is None else tuple(rotation_deg)
        pc_np = (get_rotation_matrix(*rotation_deg) @ pc_np.T).T
        return color_img, pc_np.reshape(h, w, 3)

    def stop(self):
        if self._started:
            self.pipeline.stop()
            self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


# ---------------------------------------------------------------------------
# task2 用的座標系：與 task3 不同，需要傾角校正、高度偏移與左右站的 ±90 度旋轉。
# 一樣是「相機怎麼架」的事實，所以住在取幀元件裡，fit_check.py 不認識這些。
# ---------------------------------------------------------------------------

class Task2Frame(NamedTuple):
    """一幀 task2 資料。pc_raw 保留未校正版本，供離線調 pose 用。"""
    color_img: np.ndarray
    pc_calib: np.ndarray
    pc_raw: np.ndarray
    rotation_deg: tuple
    pose: "CameraPose"


@dataclass(frozen=True)
class CameraPose:
    """相機物理架設姿態。對齊 src/config.py 的 CameraPoseConfig。"""
    tilt: float
    phy_height: float
    flip: bool = True


#: 對齊 src/config.py 的 TASK2_CAM_POSE_LEFT / _RIGHT（目前兩站數值相同）
TASK2_POSE_LEFT = CameraPose(tilt=-72.0, phy_height=2.2, flip=True)
TASK2_POSE_RIGHT = CameraPose(tilt=-72.0, phy_height=2.2, flip=True)

#: task1 的座標修正，對齊 task1.py 的 R_fix
TASK1_ROTATION_DEG = (0.0, 0.0, -90.0)

#: 左站繞 Z 軸 +90 度、右站 -90 度，對齊 vision_engine.process_stage4_task2
TASK2_ROTATION_DEG_LEFT = (0.0, 0.0, 90.0)
TASK2_ROTATION_DEG_RIGHT = (0.0, 0.0, -90.0)


def apply_calibration(pose: CameraPose, points: np.ndarray) -> np.ndarray:
    """
    應用相機校正：旋轉、翻轉與平移。
    換座標系，把 Y下X右Z前 換成 Y上X右Z前（只有改到 Y）。
    與 src/camera_manager.py 的實作相同，內聯以避免相依。
    """
    R_tilt = get_rotation_matrix(pose.tilt, 0, 0)
    pts = (R_tilt @ points.T).T

    if pose.flip:
        R_flip = get_rotation_matrix(0, 0, 180)
        pts = (R_flip @ pts.T).T

    pts = pts.copy()
    pts[:, 1] += pose.phy_height
    pts[:, 0] *= -1
    return pts


def task2_pose_for(position_2d) -> tuple:
    """依照 position_2d 挑出左/右站的 (rotation_deg, pose)。"""
    if position_2d[1] == 1:
        return TASK2_ROTATION_DEG_LEFT, TASK2_POSE_LEFT
    return TASK2_ROTATION_DEG_RIGHT, TASK2_POSE_RIGHT


def calibrate_flat(pc_raw: np.ndarray, rotation_deg, pose: CameraPose) -> np.ndarray:
    """
    把**未校正**的扁平點雲轉成 fit_check 契約座標系下的 (N, 3)。

    這是純函數、不碰硬體，所以可以拿存好的 pc_raw 離線重算不同 pose——
    微調傾角/高度/鏡像時就是呼叫這個，不需要重拍。

    注意輸出是扁平的，不是 (H, W, 3)：fit_check 吃扁平的，box_measure 吃 (H, W, 3)。
    """
    pts = (get_rotation_matrix(*rotation_deg) @ pc_raw.T).T
    return apply_calibration(pose, pts)


def to_pointcloud_flat(pc, depth_frame, rotation_deg, pose: CameraPose) -> np.ndarray:
    """深度幀 -> fit_check 契約座標系下的扁平 (N, 3) 點雲。"""
    pts = pc.calculate(depth_frame)
    vtx = np.asanyarray(pts.get_vertices())
    pc_raw = np.stack([vtx["f0"], vtx["f1"], vtx["f2"]], axis=-1)
    return calibrate_flat(pc_raw, rotation_deg, pose)
