#!/usr/bin/env python
"""
run_live.py — 手動測試驅動腳本

把 realsense_source（取幀）+ rfdetr_detector（偵測）+ box_measure（量測）接起來，
跑一次完整的 task3，並可存 fixture、匯出圖片、或開即時預覽視窗。

用法
----
# 即時預覽對準相機：SPACE 拍一張並量測，s 存檔，ESC 離開
python run_live.py --preview --label L554001

# 只測取幀，不載模型（快）
python run_live.py --no-model --save --label L554001

# 直接抓 N 張
python run_live.py -n 5 --save --label L554001

# 把已存的 fixture 匯出成 PNG（可以指定單一檔案或整個資料夾）
python run_live.py --replay fixtures/ --export
python run_live.py --replay fixtures/20260819-151112_L554001.npz --export out/

# 用 fixture 重跑，完全不需要相機與模型
python run_live.py --replay fixtures/20260819-151112_L554001.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import box_measure as bm

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures"


# --------------------------------------------------------------------- 輸出

def describe(result: bm.BoxMeasurement, elapsed: float = None, true_label: str = "") -> None:
    print("-" * 64)
    print(f"  階段      : {result.stage}")
    print(f"  箱型      : {result.box_type}" + ("  ✅" if result.ok else "  ❌"))
    if result.lwh:
        l, w, h = result.lwh
        print(f"  量測 LWH  : L={l:.4f}  W={w:.4f}  H={h:.4f}   (公尺)")
    if result.match_loss is not None:
        print(f"  比對誤差  : {result.match_loss:.6f}  (門檻 {bm.DEFAULT_MATCH_LOSS_THRESHOLD})")
    if result.top_plane_normal is not None:
        n = result.top_plane_normal
        print(f"  頂面法向量: ({n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f})"
              f"   X 分量 {abs(float(n[0])):.4f} / 容許 {bm.DEFAULT_NO_ROLL_TOLERANCE}")
    if result.boxes is not None:
        print(f"  偵測框數  : {len(result.boxes)}   labels={[int(x) for x in result.labels]}")
    for w in result.warnings:
        print(f"  ⚠ 警告    : {w}")
    print(f"  top_box   : {result.top_box}")
    print(f"  face_box  : {result.face_box}")
    if elapsed is not None:
        print(f"  耗時      : {elapsed:.3f}s")
    if true_label:
        print(f"  對照標註 {true_label}: "
              + ("✅ 一致" if result.box_type == true_label
                 else f"❌ 不一致（量到 {result.box_type}）"))
    print("-" * 64)


def save_fixture(color_img, pc_np, result, label: str) -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = FIXTURE_DIR / f"{stamp}_{label or 'unlabeled'}.npz"
    np.savez_compressed(
        path,
        color_img=color_img, pc_np=pc_np,
        boxes=result.boxes if result.boxes is not None else np.zeros((0, 4)),
        labels=result.labels if result.labels is not None else np.zeros(0, dtype=int),
        scores=result.scores if result.scores is not None else np.zeros(0),
        true_box_type=label or "",
        measured_lwh=np.asarray(result.lwh if result.lwh else (np.nan,) * 3),
        measured_box_type=result.box_type,
    )
    print(f"  💾 已存 {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return path


def save_fixture_at(path: Path, color_img, pc_np, result, label: str) -> None:
    """把重新偵測的結果寫回既有的 fixture（檔名不變）。"""
    np.savez_compressed(
        path,
        color_img=color_img, pc_np=pc_np,
        boxes=result.boxes if result.boxes is not None else np.zeros((0, 4)),
        labels=result.labels if result.labels is not None else np.zeros(0, dtype=int),
        scores=result.scores if result.scores is not None else np.zeros(0),
        true_box_type=label or "",
        measured_lwh=np.asarray(result.lwh if result.lwh else (np.nan,) * 3),
        measured_box_type=result.box_type,
    )
    print(f"  ♻  已更新 {path.name}")


def export_pngs(color_img, pc_np, result, out_dir: Path, stem: str, true_label: str = "") -> None:
    """把一幀匯出成三張 PNG：原圖、標註圖、深度有效性圖。"""
    import cv2
    import viz

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_1_raw.png"), color_img)
    cv2.imwrite(str(out_dir / f"{stem}_2_detect.png"),
                viz.draw_detections(color_img, result, true_label=true_label))
    cv2.imwrite(str(out_dir / f"{stem}_3_depth.png"),
                viz.depth_validity_map(pc_np, color_img))
    print(f"  🖼  匯出 {out_dir}/{stem}_[1_raw|2_detect|3_depth].png")


def describe_fit(result, elapsed=None):
    print("-" * 64)
    print(f"  階段      : {result.stage}")
    print(f"  可擺放    : {result.fit_ok}" + ("  ✅" if result.fit_ok else "  ❌"))
    print(f"  position_3d: {result.position_3d}")
    if result.base_h is not None:
        print(f"  基準高度  : {result.base_h:.4f} m   -> 第 {result.new_height} 層")
    if result.mask is not None:
        print(f"  可用平面  : {result.mask.shape[1]}x{result.mask.shape[0]} cells")
    for w in result.warnings:
        print(f"  ⚠ 警告    : {w}")
    if elapsed is not None:
        print(f"  耗時      : {elapsed:.3f}s")
    print("-" * 64)


def save_fixture_task2(frame, result, position_2d, cardboard_type, block_type):
    """存 task2 fixture。存的是**未校正**的 pc_raw，這樣之後才能離線重算不同 pose。"""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = FIXTURE_DIR / f"{stamp}_task2_{cardboard_type}.npz"
    np.savez_compressed(
        path, task=2, color_img=frame.color_img, pc_raw=frame.pc_raw,
        rotation_deg=np.asarray(frame.rotation_deg),
        pose_tilt=float(frame.pose.tilt), pose_phy_height=float(frame.pose.phy_height),
        pose_flip=bool(frame.pose.flip),
        position_2d=np.asarray(position_2d), cardboard_type=cardboard_type,
        block_type=block_type,
        fit_ok=bool(result.fit_ok), position_3d=np.asarray(result.position_3d),
        base_h=float(result.base_h) if result.base_h is not None else np.nan,
        stage=result.stage,
    )
    print(f"  💾 已存 {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return path


def export_task2_pngs(color_img, result, out_dir: Path, stem: str) -> None:
    """把 fit_check 回傳的視覺化圖寫出來（單元本身不碰檔案系統，存檔是這裡的事）。"""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if color_img is not None:
        cv2.imwrite(str(out_dir / f"{stem}_1_raw.png"), color_img); n += 1
    if result.base_vis is not None:
        cv2.imwrite(str(out_dir / f"{stem}_2_base.png"), result.base_vis); n += 1
    for label, img in result.vis.items():
        cv2.imwrite(str(out_dir / f"{stem}_3_fit_{label}.png"), img); n += 1
    print(f"  🖼  匯出 {n} 張到 {out_dir}/")


def run_task2(args, cam) -> int:
    """task2：擺放位置檢查。不需要模型。重試迴圈在這裡（取幀是注入的）。"""
    import fit_check as fcm

    position_2d = tuple(int(v) for v in args.position.split(","))
    pose = _pose_override(args)
    rc = 1
    result = frame = None
    for attempt in range(args.retries):
        t0 = time.time()
        frame = cam.get_frame_task2(position_2d, pose=pose)
        print(f"\n[{attempt + 1}/{args.retries}] color={frame.color_img.shape} "
              f"pc={frame.pc_calib.shape}  pose=tilt {frame.pose.tilt} / "
              f"height {frame.pose.phy_height} / flip {frame.pose.flip}")
        try:
            result = fcm.check_fit(frame.color_img, frame.pc_calib, position_2d, args.cardboard,
                                   block_type=args.block_type)
        except fcm.ContractError as e:
            print(f"  ❌ 契約檢查失敗: {e}")
            return 2
        describe_fit(result, time.time() - t0)
        if result.fit_ok:
            rc = 0
            break

    if result is not None and args.save:
        path = save_fixture_task2(frame, result, position_2d, args.cardboard, args.block_type)
        if args.export is not None:
            out_dir = Path(args.export) if args.export != "-" else FIXTURE_DIR / "png"
            export_task2_pngs(frame.color_img, result, out_dir, path.stem)
    return rc


def run_preview_task2(args, cam) -> int:
    """
    task2 的即時預覽。SPACE 跑完整檢查，s 存檔，ESC/q 離開。

    每幀只畫 ROI 對準疊圖（約 4ms）；完整的 check_fit 在真實解析度要 0.5 秒以上，
    所以只有按 SPACE 才跑。
    """
    import cv2
    import fit_check as fcm
    import viz

    position_2d = tuple(int(v) for v in args.position.split(","))
    pose = _pose_override(args)
    try:
        side, ref, bound = fcm.resolve_rois(position_2d, args.cardboard)
    except fcm.ContractError as e:
        print(f"❌ {e}")
        return 2

    win = "run_live task2 — SPACE:檢查  s:存檔  ESC:離開"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    saved = 0
    print(f"\ntask2 即時預覽（{side} 站, {args.cardboard}）")
    print("  黃框 = 基準框（必須落在要量高度的那個平面上）")
    print("  紫框 = 搜尋範圍")
    print("  SPACE = 跑完整檢查    s = 存成 fixture    ESC/q = 離開\n")

    try:
        while True:
            frame = cam.get_frame_task2(position_2d, pose=pose)
            view, base_h, n_ref = viz.draw_task2_aim(
                frame.color_img, frame.pc_calib, ref, bound,
                min_ref_points=fcm.MIN_REF_POINTS,
                extra=(f"tilt {frame.pose.tilt} / height {frame.pose.phy_height} "
                       f"/ flip {frame.pose.flip}", f"saved {saved}"))
            cv2.imshow(win, viz.fit_to_screen(view))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key != 32:
                continue

            t0 = time.time()
            try:
                result = fcm.check_fit(frame.color_img, frame.pc_calib, position_2d,
                                       args.cardboard, block_type=args.block_type)
            except fcm.ContractError as e:
                print(f"❌ 契約檢查失敗: {e}")
                continue
            describe_fit(result, time.time() - t0)

            shot = result.base_vis if result.base_vis is not None else view
            shot = shot.copy()
            viz._put(shot, ("FIT OK" if result.fit_ok else "NO FIT"), (12, shot.shape[0] - 54),
                     (0, 255, 0) if result.fit_ok else (0, 80, 255), 0.9)
            viz._put(shot, "s = 存檔    其他鍵 = 丟棄", (12, shot.shape[0] - 20), (0, 255, 255), 0.8)
            cv2.imshow(win, viz.fit_to_screen(shot))

            k2 = cv2.waitKey(0) & 0xFF
            if k2 == ord('s'):
                path = save_fixture_task2(frame, result, position_2d,
                                          args.cardboard, args.block_type)
                saved += 1
                if args.export is not None:
                    out_dir = Path(args.export) if args.export != "-" else FIXTURE_DIR / "png"
                    export_task2_pngs(frame.color_img, result, out_dir, path.stem)
            else:
                print("  已丟棄。")
            if k2 in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()

    print(f"\n預覽結束，共存了 {saved} 張。")
    return 0


# --------------------------------------------------------------------- task1

def describe_stack(result, elapsed=None):
    print("-" * 64)
    print(f"  階段      : {result.stage}")
    if result.place_position is None:
        print("  擺放位置  : 無（所有位置都已堆滿）  ❌")
    else:
        r, c, l = result.place_position
        print(f"  擺放位置  : row {r}  col {c}  layer {l}  ✅")
    if result.block is not None:
        print(f"  層數矩陣  : {result.block.tolist()}")
    if result.prior_score is not None:
        print(f"  優先度    : {result.prior_score:.4f}")
    print(f"  偵測      : top {len(result.top_entries)} / face {len(result.face_entries)} "
          f"/ pallet {len(result.detected_pallets)} / 人 {result.people_count}")
    for w in result.warnings:
        print(f"  ⚠ 警告    : {w}")
    if elapsed is not None:
        print(f"  耗時      : {elapsed:.3f}s")
    print("-" * 64)


def _task1_calibrator(args):
    import pallet_stack as psm
    return psm.make_calibrator(
        tilt=psm.DEFAULT_TILT if args.tilt is None else args.tilt,
        phy_height=psm.DEFAULT_PHY_HEIGHT if args.phy_height is None else args.phy_height,
        flip=not args.no_flip)


def save_fixture_task1(color_img, pc_np, result, args):
    """存 task1 fixture。pc_np 是未校正的，換 calibrator 就能離線調姿態。"""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = FIXTURE_DIR / f"{stamp}_task1_{args.station}_{args.cardboard}.npz"
    np.savez_compressed(
        path, task=1, color_img=color_img, pc_np=pc_np,
        station=args.station, cardboard_type=args.cardboard, block_type=args.block_type,
        boxes=result.boxes if result.boxes is not None else np.zeros((0, 4)),
        labels=result.labels if result.labels is not None else np.zeros(0, dtype=int),
        scores=result.scores if result.scores is not None else np.zeros(0),
        block=result.block if result.block is not None else np.zeros((0, 0), dtype=int),
        place_position=np.asarray(result.place_position if result.place_position else (-1, -1, -1)),
        prior_score=float(result.prior_score) if result.prior_score is not None else np.nan,
        people_count=int(result.people_count), stage=result.stage,
    )
    print(f"  💾 已存 {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return path


def export_task1_pngs(color_img, result, out_dir: Path, stem: str) -> None:
    import cv2
    import viz
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_1_raw.png"), color_img)
    if result.hmap is not None:
        cv2.imwrite(str(out_dir / f"{stem}_2_hmap.png"),
                    viz.draw_hmap(result.hmap, result.block, result.place_position))
        cv2.imwrite(str(out_dir / f"{stem}_3_combined.png"),
                    viz.draw_task1_aim(color_img, result.hmap, result.block,
                                       result.place_position))
    print(f"  🖼  匯出 {out_dir}/{stem}_[1_raw|2_hmap|3_combined].png")


def _build_task1_detectors(args):
    """載入 task1 需要的兩顆模型。--no-model 時回傳 (無偵測, None)。"""
    if args.no_model:
        print("⚠ --no-model：跳過偵測與評分，只做堆疊位置判定（那部分本來就不需要模型）。")
        return (lambda _img: (np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))), None
    from rfdetr_detector import RFDETRDetector, RFDETRPeopleDetector
    print("載入模型中（兩顆，第一次會比較久）...")
    return (RFDETRDetector(view_rot_k=args.rot_k, conf_thres=args.conf),
            RFDETRPeopleDetector(view_rot_k=args.rot_k))


def _run_task1_once(color_img, pc_np, detect, detect_people, args, calibrate):
    import pallet_stack as psm
    return psm.analyze_pallet(
        color_img, pc_np, detect, args.station, args.cardboard,
        block_type=args.block_type, calibrate=calibrate, detect_people=detect_people,
        score_model_path=args.score_model or None)


def run_task1(args, cam) -> int:
    import pallet_stack as psm

    detect, detect_people = _build_task1_detectors(args)
    calibrate = _task1_calibrator(args)
    rc = 1
    for i in range(args.n):
        t0 = time.time()
        color_img, pc_np = cam.get_frame_task1()
        print(f"\n[{i + 1}/{args.n}] color={color_img.shape} pc={pc_np.shape}  "
              f"pose=tilt {calibrate.params[0]} / height {calibrate.params[1]} "
              f"/ flip {calibrate.params[2]}")
        try:
            result = _run_task1_once(color_img, pc_np, detect, detect_people, args, calibrate)
        except psm.ContractError as e:
            print(f"  ❌ 契約檢查失敗: {e}")
            return 2
        describe_stack(result, time.time() - t0)
        if result.ok:
            rc = 0
        if args.save:
            path = save_fixture_task1(color_img, pc_np, result, args)
            if args.export is not None:
                out_dir = Path(args.export) if args.export != "-" else FIXTURE_DIR / "png"
                export_task1_pngs(color_img, result, out_dir, path.stem)
    return rc


def run_preview_task1(args, cam) -> int:
    """
    task1 即時預覽。堆疊判定不需要模型，所以每幀就能算出層數矩陣與擺放位置；
    SPACE 才跑完整流程（含兩顆模型的偵測與 SVR 評分）。
    """
    import cv2
    import pallet_stack as psm
    import viz

    calibrate = _task1_calibrator(args)
    detect = detect_people = None
    saved = 0

    win = "run_live task1 — SPACE:完整分析  s:存檔  ESC:離開"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"\ntask1 即時預覽（{args.station}, {args.cardboard}, {args.block_type}）")
    print("  右側為高度圖，格內數字是層數，綠框是目前會擺放的位置")
    print("  每幀只算堆疊判定（不用模型）；SPACE 才跑偵測與評分\n")

    try:
        while True:
            color_img, pc_np = cam.get_frame_task1()
            builder = (psm.build_block_list if args.block_type == "pallet"
                       else psm.build_tape_block)
            (_, block), place, hmap = builder(pc_np, args.cardboard, args.station, calibrate)
            view = viz.draw_task1_aim(
                color_img, hmap, block, place,
                extra=(f"tilt {calibrate.params[0]} / height {calibrate.params[1]}",
                       f"saved {saved}"))
            cv2.imshow(win, viz.fit_to_screen(view))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key != 32:
                continue

            if detect is None:
                detect, detect_people = _build_task1_detectors(args)
            t0 = time.time()
            try:
                result = _run_task1_once(color_img, pc_np, detect, detect_people,
                                         args, calibrate)
            except psm.ContractError as e:
                print(f"❌ 契約檢查失敗: {e}")
                continue
            describe_stack(result, time.time() - t0)

            shot = viz.draw_task1_aim(color_img, result.hmap, result.block,
                                      result.place_position,
                                      extra=(f"score {result.prior_score:.3f}",))
            viz._put(shot, "s = 存檔    其他鍵 = 丟棄", (12, shot.shape[0] - 20),
                     (0, 255, 255), 0.8)
            cv2.imshow(win, viz.fit_to_screen(shot))

            k2 = cv2.waitKey(0) & 0xFF
            if k2 == ord('s'):
                path = save_fixture_task1(color_img, pc_np, result, args)
                saved += 1
                if args.export is not None:
                    out_dir = Path(args.export) if args.export != "-" else FIXTURE_DIR / "png"
                    export_task1_pngs(color_img, result, out_dir, path.stem)
            else:
                print("  已丟棄。")
            if k2 in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()
    print(f"\n預覽結束，共存了 {saved} 張。")
    return 0


def _replay_task1(d, path: Path, args, out_dir: Path) -> int:
    """用存好的 task1 fixture 重跑。pc_np 未校正，所以可以離線換 calibrator。"""
    import pallet_stack as psm

    class A:  # 讓 _run_task1_once 用 fixture 裡的設定
        pass
    a = A()
    a.station = str(d["station"])
    a.cardboard = str(d["cardboard_type"])
    a.block_type = str(d["block_type"])
    a.score_model = args.score_model

    boxes, labels, scores = d["boxes"], d["labels"], d["scores"]
    calibrate = _task1_calibrator(args)
    print(f"  {a.station} / {a.cardboard} / {a.block_type}   "
          f"pose=tilt {calibrate.params[0]} / height {calibrate.params[1]}")
    try:
        result = psm.analyze_pallet(
            d["color_img"], d["pc_np"], lambda _img: (boxes, labels, scores),
            a.station, a.cardboard, block_type=a.block_type, calibrate=calibrate,
            score_model_path=args.score_model or None)
    except psm.ContractError as e:
        print(f"  ❌ 契約檢查失敗: {e}")
        return 2
    describe_stack(result)
    if "place_position" in d:
        was = tuple(int(v) for v in d["place_position"])
        now = result.place_position or (-1, -1, -1)
        print(f"  對照存檔當下 {was}: " + ("✅ 一致" if was == tuple(now) else "❌ 不一致"))
    if args.export is not None:
        export_task1_pngs(d["color_img"], result, out_dir, path.stem)
    return 0 if result.ok else 1


def _pose_override(args):
    """從 CLI 參數組出 CameraPose；三個都沒給就回 None（用現場預設值）。"""
    from realsense_source import CameraPose, task2_pose_for
    if args.tilt is None and args.phy_height is None and not args.no_flip:
        return None
    position_2d = tuple(int(v) for v in args.position.split(","))
    _, base = task2_pose_for(position_2d)
    return CameraPose(
        tilt=base.tilt if args.tilt is None else args.tilt,
        phy_height=base.phy_height if args.phy_height is None else args.phy_height,
        flip=False if args.no_flip else base.flip,
    )


# --------------------------------------------------------------------- 重跑

def _load_fixture(path: Path):
    d = np.load(path, allow_pickle=False)
    return (d["color_img"], d["pc_np"], d["boxes"], d["labels"], d["scores"],
            str(d["true_box_type"]) if "true_box_type" in d else "")


def _plane_slope(pc_calib, base_h, band=0.06):
    """
    量「應該是水平的那個面」在深度方向的殘餘斜率 dy/dz。

    傾角校正正確時，一個物理上水平的表面在校正後的 y 應該不隨 z 變化，
    所以 dy/dz ≈ 0。傾角有誤差就會出現系統性的斜率。
    """
    if base_h is None:
        return None
    m = np.isfinite(pc_calib).all(axis=1) & (np.abs(pc_calib[:, 1] - base_h) < band)
    if m.sum() < 200:
        return None
    z = pc_calib[m, 2]
    y = pc_calib[m, 1]
    if np.ptp(z) < 0.05:
        return None
    return float(np.polyfit(z, y, 1)[0])


def _sweep_pose(d, position_2d, cardboard, block_type, rotation_deg, pose, args) -> int:
    """
    掃一輪 tilt / phy_height，印出各組合量到的 base_h 與是否可擺放。

    調 pose 的做法：拿一個「已知實際高度」的場景拍一張 fixture，
    然後掃到 base_h 等於你量到的真實高度為止。
    """
    import fit_check as fcm
    from realsense_source import CameraPose, calibrate_flat

    # 掃描會跑幾十次，把 fit_check 的 INFO 壓掉，只留表格
    fit_logger = logging.getLogger("fit_check")
    prev_level = fit_logger.level
    fit_logger.setLevel(logging.WARNING)

    tilts = np.arange(pose.tilt - 6, pose.tilt + 6.01, 2.0)
    heights = np.arange(pose.phy_height - 0.15, pose.phy_height + 0.151, 0.05)

    print(f"  掃描 tilt {tilts[0]:.1f}~{tilts[-1]:.1f} 度 x "
          f"height {heights[0]:.2f}~{heights[-1]:.2f} m（flip={pose.flip}）")
    print("  dy/dz = 平面在深度方向的殘餘斜率。傾角正確時應該接近 0；")
    print("  height 只會整體平移 base_h，不影響斜率——所以先用 dy/dz 定 tilt，再用 base_h 定 height。")
    print(f"  {'tilt':>7} {'height':>7} {'base_h':>9} {'dy/dz':>8} {'層':>3}  結果")
    for t in tilts:
        for hgt in heights:
            p = CameraPose(float(t), float(hgt), pose.flip)
            pc = calibrate_flat(d["pc_raw"], rotation_deg, p)
            try:
                r = fcm.check_fit(None, pc, position_2d, cardboard, block_type=block_type)
            except fcm.ContractError as e:
                print(f"  {t:7.1f} {hgt:7.2f} {'—':>9} {'—':>3}  契約錯誤: {e}")
                continue
            bh = f"{r.base_h:.4f}" if r.base_h is not None else "—"
            lv = str(r.new_height) if r.new_height is not None else "—"
            slope = _plane_slope(pc, r.base_h)
            sl = f"{slope:+.4f}" if slope is not None else "—"
            mark = "✅ 可擺放" if r.fit_ok else r.stage.split("→")[-1]
            print(f"  {t:7.1f} {hgt:7.2f} {bh:>9} {sl:>8} {lv:>3}  {mark}")
    fit_logger.setLevel(prev_level)
    print("\n  調法：找一個你知道實際高度的表面，掃到 base_h 等於那個高度的那一組就是對的。")
    return 0


def _replay_task2(d, path: Path, args, out_dir: Path) -> int:
    """用存好的 task2 fixture 重跑 check_fit，不需要相機。"""
    import fit_check as fcm

    from realsense_source import CameraPose, calibrate_flat

    position_2d = tuple(int(v) for v in d["position_2d"])
    cardboard = str(d["cardboard_type"])
    block_type = str(d["block_type"]) if "block_type" in d else "pallet"

    if "pc_raw" not in d:
        print("  ⚠ 這是舊格式的 fixture（只存了校正後的點雲），無法離線調 pose。"
              " 重拍一次就會存成新格式。")
        return 2

    rotation_deg = tuple(float(v) for v in d["rotation_deg"])
    pose = CameraPose(tilt=float(d["pose_tilt"]),
                      phy_height=float(d["pose_phy_height"]),
                      flip=bool(d["pose_flip"]))
    if args.tilt is not None:
        pose = CameraPose(args.tilt, pose.phy_height, pose.flip)
    if args.phy_height is not None:
        pose = CameraPose(pose.tilt, args.phy_height, pose.flip)
    if args.no_flip:
        pose = CameraPose(pose.tilt, pose.phy_height, False)

    if args.sweep:
        return _sweep_pose(d, position_2d, cardboard, block_type, rotation_deg, pose, args)

    print(f"  pose: tilt {pose.tilt} / height {pose.phy_height} / flip {pose.flip}")
    pc_calib = calibrate_flat(d["pc_raw"], rotation_deg, pose)
    try:
        result = fcm.check_fit(d["color_img"], pc_calib, position_2d, cardboard,
                               block_type=block_type)
    except fcm.ContractError as e:
        print(f"  ❌ 契約檢查失敗: {e}")
        return 2
    describe_fit(result)
    if "fit_ok" in d:
        was = bool(d["fit_ok"])
        print(f"  對照存檔當下的結果 fit_ok={was}: "
              + ("✅ 一致" if was == result.fit_ok else "❌ 不一致"))
    if args.export is not None:
        export_task2_pngs(d["color_img"], result, out_dir, path.stem)
    return 0 if result.fit_ok else 1


def run_replay(args) -> int:
    target = Path(args.replay)
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    files = sorted(target.glob("*.npz")) if target.is_dir() else [target]
    if not files:
        print(f"❌ {target} 底下找不到 .npz")
        return 2

    out_dir = Path(args.export) if args.export and args.export != "-" else (
        (target if target.is_dir() else target.parent) / "png")

    detector = None
    if args.redetect:
        from rfdetr_detector import RFDETRDetector
        print("載入模型以重新偵測（--redetect）...")
        detector = RFDETRDetector(view_rot_k=args.rot_k, conf_thres=args.conf)

    rc = 0
    for path in files:
        d = np.load(path, allow_pickle=False)
        if "task" in d and int(d["task"]) == 1:
            print(f"\n▶ {path.name}   (task1)")
            rc = max(rc, _replay_task1(d, path, args, out_dir))
            continue
        if "task" in d and int(d["task"]) == 2:
            print(f"\n▶ {path.name}   (task2)")
            rc = max(rc, _replay_task2(d, path, args, out_dir))
            continue

        color_img, pc_np, boxes, labels, scores, true_type = _load_fixture(path)
        print(f"\n▶ {path.name}   (標註: {true_type or '無'})")

        detect = detector if detector is not None else (lambda _img: (boxes, labels, scores))
        result = bm.measure_box(color_img, pc_np, detect, view_rot_k=args.rot_k)
        if args.redetect and args.rewrite:
            save_fixture_at(path, color_img, pc_np, result, true_type)
        describe(result, true_label=true_type)
        if args.export is not None:
            export_pngs(color_img, pc_np, result, out_dir, path.stem, true_type)
        if not result.ok:
            rc = 1
    return rc


# --------------------------------------------------------------------- 即時

def run_preview(args, cam, detect) -> int:
    """即時預覽視窗。SPACE 拍一張並量測，s 存檔，ESC/q 離開。"""
    import cv2
    import viz

    win = "run_live — SPACE:拍攝  s:存檔  d:深度圖  ESC:離開"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    show_depth = False
    saved = 0
    print("\n即時預覽已開啟。畫面已轉成正立方向（view_rot_k=%d）。" % args.rot_k)
    print("  SPACE = 拍攝並量測    s = 存成 fixture    d = 切換深度圖    ESC/q = 離開\n")

    try:
        while True:
            color_img, pc_np = cam.get_frame()

            # 轉成人眼看得順的正立方向再顯示（不影響送進單元的資料）
            if show_depth:
                view = viz.depth_validity_map(pc_np, color_img)
            else:
                view = color_img.copy()
                valid = float((pc_np[..., 2] > 0.01).mean())
                viz._put(view, f"valid depth: {valid:.1%}   已存 {saved} 張", (12, 28))
            view = np.rot90(view, k=args.rot_k).copy()
            cv2.imshow(win, viz.fit_to_screen(view))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('d'):
                show_depth = not show_depth
                continue
            if key != 32:                      # 只有 SPACE 觸發拍攝
                continue

            # --- 拍攝並量測 ---
            if detect is None:
                print("⚠ --no-model：只存原始資料，不做量測。")
                result = bm.BoxMeasurement(box_type="", stage="--no-model")
            else:
                t0 = time.time()
                try:
                    result = bm.measure_box(color_img, pc_np, detect, view_rot_k=args.rot_k)
                except bm.ContractError as e:
                    print(f"❌ 契約檢查失敗: {e}")
                    continue
                describe(result, time.time() - t0, args.label)

            shot = viz.draw_detections(color_img, result, true_label=args.label)
            shot = np.rot90(shot, k=args.rot_k).copy()
            viz._put(shot, "s = 存檔    其他鍵 = 丟棄", (12, shot.shape[0] - 20), (0, 255, 255), 0.8)
            cv2.imshow(win, viz.fit_to_screen(shot))

            k2 = cv2.waitKey(0) & 0xFF
            if k2 == ord('s'):
                path = save_fixture(color_img, pc_np, result, args.label)
                saved += 1
                if args.export is not None:
                    out_dir = Path(args.export) if args.export != "-" else FIXTURE_DIR / "png"
                    export_pngs(color_img, pc_np, result, out_dir, path.stem, args.label)
            else:
                print("  已丟棄。")
            if k2 in (27, ord('q')):
                break
    finally:
        cv2.destroyAllWindows()

    print(f"\n預覽結束，共存了 {saved} 張。")
    return 0


def run_live(args) -> int:
    from realsense_source import RealSenseSource

    detect = None
    if args.task == 2:
        args.no_model = True          # task2 純點雲，用不到模型
    if not args.no_model:
        from rfdetr_detector import RFDETRDetector
        print("載入模型中（第一次會比較久）...")
        detect = RFDETRDetector(view_rot_k=args.rot_k, conf_thres=args.conf)
    else:
        print("⚠ --no-model：只測取幀，不做偵測與量測。")

    with RealSenseSource() as cam:
        if args.task == 1:
            return run_preview_task1(args, cam) if args.preview else run_task1(args, cam)
        if args.task == 2:
            return run_preview_task2(args, cam) if args.preview else run_task2(args, cam)
        if args.preview:
            return run_preview(args, cam, detect)

        rc = 0
        for i in range(args.n):
            t0 = time.time()
            color_img, pc_np = cam.get_frame()
            valid = float((pc_np[..., 2] > 0.01).mean())
            print(f"\n[{i + 1}/{args.n}] color={color_img.shape} pc={pc_np.shape} "
                  f"有效深度 {valid:.1%}")

            if detect is None:
                result = bm.BoxMeasurement(box_type="", stage="--no-model")
            else:
                try:
                    result = bm.measure_box(color_img, pc_np, detect, view_rot_k=args.rot_k)
                except bm.ContractError as e:
                    print(f"  ❌ 契約檢查失敗: {e}")
                    rc = 2
                    continue
                describe(result, time.time() - t0, args.label)
                if not result.ok:
                    rc = 1

            if args.save:
                path = save_fixture(color_img, pc_np, result, args.label)
                if args.export is not None:
                    out_dir = Path(args.export) if args.export != "-" else FIXTURE_DIR / "png"
                    export_pngs(color_img, pc_np, result, out_dir, path.stem, args.label)
        return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=1, help="抓幾幀（預設 1）")
    ap.add_argument("--preview", action="store_true",
                    help="開即時預覽視窗，SPACE 拍攝、s 存檔、d 深度圖、ESC 離開")
    ap.add_argument("--save", action="store_true", help="把這幀存成 .npz fixture")
    ap.add_argument("--export", nargs="?", const="-", metavar="DIR",
                    help="匯出 PNG（原圖/標註圖/深度圖）。不給路徑就存到 fixtures/png/")
    ap.add_argument("--label", default="", help="人工標註的正確箱型，例如 L554001")
    ap.add_argument("--rot-k", dest="rot_k", type=int, default=2,
                    help="view_rot_k，把輸入影像轉正立所需的 rot90 次數（預設 2）")
    ap.add_argument("--conf", type=float, default=0.5, help="偵測信心門檻（預設 0.5）")
    ap.add_argument("--no-model", action="store_true", help="不載模型，只測取幀")
    ap.add_argument("--replay", metavar="PATH", help="用 .npz 或整個資料夾重跑，不需硬體")
    ap.add_argument("--redetect", action="store_true",
                    help="重跑時重新執行模型偵測，而不是用 fixture 裡存的框"
                         "（用來補完 --no-model 拍的 fixture，或測試不同 --conf）")
    ap.add_argument("--rewrite", action="store_true",
                    help="搭配 --redetect：把新的偵測結果寫回原 fixture")
    ap.add_argument("--task", type=int, choices=(1, 2, 3), default=3,
                    help="要跑哪個 task：3=紙箱尺寸量測（預設），2=擺放位置檢查，1=棧板堆疊判定")
    ap.add_argument("--station", default="g4_1",
                    help="task1 的工作站：g4_1 / g4_2 / g8_1 / p_1")
    ap.add_argument("--score-model", dest="score_model", default="",
                    help="task1 的 SVR 評分模型路徑，不給就跳過評分（回傳 1.0）")
    ap.add_argument("--position", default="1,1",
                    help="task2 的 position_2d，格式 row,col；col=1 為左站，其他為右站")
    ap.add_argument("--cardboard", default="L554001", help="task2 的箱型")
    ap.add_argument("--block-type", dest="block_type", default="pallet",
                    choices=("pallet", "ground"), help="task2：棧板上還是地面")
    ap.add_argument("--retries", type=int, default=4,
                    help="task2 重試次數（每次重新取幀），對齊原本 vision_engine 的 4 次")
    ap.add_argument("--tilt", type=float, default=None,
                    help="覆寫相機俯仰角（度）。不給就用 realsense_source 的現場值 -72.0")
    ap.add_argument("--phy-height", dest="phy_height", type=float, default=None,
                    help="覆寫相機物理高度（公尺）。不給就用現場值 2.2")
    ap.add_argument("--no-flip", action="store_true",
                    help="關掉左右鏡像（預設 flip=True，與現場一致）")
    ap.add_argument("--sweep", action="store_true",
                    help="搭配 --replay：掃一輪 tilt/height 組合，印出各自量到的 base_h，用來校正相機姿態")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    return run_replay(args) if args.replay else run_live(args)


if __name__ == "__main__":
    sys.exit(main())
