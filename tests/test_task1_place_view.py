"""task1_place_view 的幾何測試。不需要相機。"""

import numpy as np
import pytest

import pallet_stack as psm
import task1_place_view as tpv


@pytest.mark.parametrize("tilt,height,flip", [(-35.0, 2.25, True), (-38.0, 2.1, True), (-60.0, 1.5, False)])
def test_world_to_cam是校正的反運算(tilt, height, flip):
    rng = np.random.default_rng(0)
    cam = rng.uniform([-1, -1, 0.5], [1, 1, 4], (200, 3))
    R = psm.rotation_matrix(0, 0, -90)              # realsense_source.TASK1_ROTATION_DEG
    world = psm.make_calibrator(tilt=tilt, phy_height=height, flip=flip)((R @ cam.T).T)
    back = tpv.world_to_cam(world, tilt, height, flip, (0, 0, -90))
    assert np.allclose(back, cam, atol=1e-9)


def test_取樣區與pallet_stack相同():
    """pallet_stack 只看四個角落；畫出來的黃框必須是同一塊。"""
    n = psm.HMAP_BASE_SIZE
    hmap = np.zeros((n, n))
    marks = {}
    for k, (r0, r1, c0, c1) in tpv.sample_cells(n).items():
        hmap[r0:r1, c0:c1] = 1 + 2 * k[0] + k[1]          # 每個角落不同高度
        marks[k] = hmap[r0, c0]
    meds = tpv.corner_medians(hmap, n)
    assert meds == marks


def test_工作區範圍對應世界座標():
    geo = tpv.Geometry("g8_1", "L554001")
    x0, x1, z0, z1 = geo.cells_to_world(0, geo.n, 0, geo.n)
    info = psm.DEFAULT_BLOCK_CATALOG["g8_1"]
    assert (x0, x1) == pytest.approx((-info.width / 2, info.width / 2))
    assert (z0, z1) == pytest.approx((info.camera_distance, info.camera_distance + info.length))
    # 列 0 是遠端
    _, _, fz0, _ = geo.cells_to_world(*tpv.quad_cells(0, 0, geo.n))
    _, _, nz0, _ = geo.cells_to_world(*tpv.quad_cells(1, 0, geo.n))
    assert fz0 > nz0
