"""
相機姿態校正的測試。

守的是「調 pose」這條路徑：pose 必須真的可以傳入、而且各參數的效果要可預測，
否則微調時分不清是參數沒吃到還是調錯方向。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realsense_source as rs_src
from realsense_source import CameraPose, calibrate_flat, get_rotation_matrix, task2_pose_for


def flat_surface(n=40, y=0.0, z0=0.5, z1=1.5):
    """一片水平面（世界座標系：Y 上、Z 前）。"""
    xs = np.linspace(-0.5, 0.5, n)
    zs = np.linspace(z0, z1, n)
    ZZ, XX = np.meshgrid(zs, xs, indexing="ij")
    return np.stack([XX.ravel(), np.full(XX.size, y), ZZ.ravel()], axis=-1).astype(np.float64)


def uncalibrate(pc_world, rotation_deg, pose):
    """calibrate_flat 的反函數，用來從理想世界座標造出「相機原始」點雲。"""
    q = pc_world.astype(np.float64).copy()
    q[:, 0] *= -1
    q[:, 1] -= pose.phy_height
    if pose.flip:
        q = (np.linalg.inv(get_rotation_matrix(0, 0, 180)) @ q.T).T
    q = (np.linalg.inv(get_rotation_matrix(pose.tilt, 0, 0)) @ q.T).T
    return (np.linalg.inv(get_rotation_matrix(*rotation_deg)) @ q.T).T


# ----------------------------------------------------------------- 對齊原始設定

def test_pose_預設值與原始config一致():
    """src/config.py:132-133 的 TASK2_CAM_POSE_LEFT / _RIGHT。"""
    for pose in (rs_src.TASK2_POSE_LEFT, rs_src.TASK2_POSE_RIGHT):
        assert pose.tilt == -72.0
        assert pose.phy_height == 2.2
        assert pose.flip is True


def test_左右站的旋轉角與原始一致():
    """vision_engine.process_stage4_task2：左站 +90 度、右站 -90 度。"""
    assert task2_pose_for((1, 1))[0] == (0.0, 0.0, 90.0)
    assert task2_pose_for((1, 2))[0] == (0.0, 0.0, -90.0)


# ----------------------------------------------------------------- 校正可逆

def test_校正與反校正互為反函數():
    rot, pose = task2_pose_for((1, 1))
    world = flat_surface(y=0.6)
    raw = uncalibrate(world, rot, pose)
    assert np.allclose(calibrate_flat(raw, rot, pose), world, atol=1e-6)


# ----------------------------------------------------------------- 各參數的效果

def test_物理高度是純粹的平移():
    """phy_height 改 X 公尺，量到的高度就整體平移 X 公尺，不影響斜率。"""
    rot, pose = task2_pose_for((1, 1))
    raw = uncalibrate(flat_surface(y=0.6), rot, pose)

    base = calibrate_flat(raw, rot, pose)
    moved = calibrate_flat(raw, rot, CameraPose(pose.tilt, pose.phy_height + 0.25, pose.flip))

    assert np.allclose(moved[:, 1] - base[:, 1], 0.25, atol=1e-6)
    assert np.allclose(moved[:, 0], base[:, 0], atol=1e-6)
    assert np.allclose(moved[:, 2], base[:, 2], atol=1e-6)


@pytest.mark.parametrize("delta", [-4.0, -2.0, 2.0, 4.0])
def test_傾角錯誤會讓水平面變斜(delta):
    """傾角正確時水平面的 dy/dz 為 0；有誤差就會出現系統性斜率。"""
    rot, pose = task2_pose_for((1, 1))
    raw = uncalibrate(flat_surface(y=0.6), rot, pose)

    correct = calibrate_flat(raw, rot, pose)
    wrong = calibrate_flat(raw, rot, CameraPose(pose.tilt + delta, pose.phy_height, pose.flip))

    slope_ok = np.polyfit(correct[:, 2], correct[:, 1], 1)[0]
    slope_bad = np.polyfit(wrong[:, 2], wrong[:, 1], 1)[0]

    assert abs(slope_ok) < 1e-6, "正確傾角下水平面不該有斜率"
    assert abs(slope_bad) > 0.01, f"傾角差 {delta} 度卻幾乎沒有斜率，這個旋鈕可能沒吃到"
    assert np.sign(slope_bad) == np.sign(delta), "斜率方向應該跟著傾角誤差的方向"


def test_斜率不受物理高度影響():
    """這是兩步驟校正的前提：先用斜率定 tilt，再用 base_h 定 height。"""
    rot, pose = task2_pose_for((1, 1))
    raw = uncalibrate(flat_surface(y=0.6), rot, pose)

    slopes = []
    for h in (2.0, 2.2, 2.4):
        pc = calibrate_flat(raw, rot, CameraPose(pose.tilt + 3.0, h, pose.flip))
        slopes.append(np.polyfit(pc[:, 2], pc[:, 1], 1)[0])
    assert np.allclose(slopes, slopes[0], atol=1e-9)


def test_flip關掉會改變結果():
    rot, pose = task2_pose_for((1, 1))
    raw = uncalibrate(flat_surface(y=0.6), rot, pose)
    with_flip = calibrate_flat(raw, rot, pose)
    without = calibrate_flat(raw, rot, CameraPose(pose.tilt, pose.phy_height, False))
    assert not np.allclose(with_flip, without), "flip 這個旋鈕沒有吃到"
