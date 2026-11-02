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


@pytest.mark.parametrize("station,block_type,n_cells", [("g8_1", "ground", 8), ("g4_1", "ground", 4),
                                                         ("p_1", "pallet", 4)])
def test_畫出的層數與演算法相同(station, block_type, n_cells):
    """用每格不同高度的 hmap：畫圖用的中位數換算層數，要等於 pallet_stack 算出的 block。"""
    lay = tpv.Layout(station, "L554001", block_type)
    assert len(lay.regions) == n_cells
    hmap = np.full((lay.n, lay.n), -np.inf)
    for (r, c), (region, _) in lay.regions.items():
        r0, r1, c0, c1 = region
        hmap[r0:r1, c0:c1] = lay.base + ((r + 2 * c) % 4) * lay.box_h
    to_layer = psm._make_height_to_layer(lay.box_h)
    for (r, c), m in lay.medians(hmap).items():
        assert to_layer(m, lay.base) == min(psm.MAX_LAYERS, (r + 2 * c) % 4)


def test_工作區範圍對應世界座標():
    lay = tpv.Layout("g8_1", "L554001", "ground")
    x0, x1, z0, z1 = lay.cells_to_world(0, lay.n, 0, lay.n)
    info = psm.DEFAULT_BLOCK_CATALOG["g8_1"]
    assert (x0, x1) == pytest.approx((-info.width / 2, info.width / 2))
    assert (z0, z1) == pytest.approx((info.camera_distance, info.camera_distance + info.length))
    # 第 1 排是遠端
    far = lay.cells_to_world(*lay.regions[(0, 0)][0])
    near = lay.cells_to_world(*lay.regions[(3, 0)][0])
    assert far[2] > near[2]
