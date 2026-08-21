"""
守住「單元能被單獨搬走」這件事。

移植一個能力時要帶走的就是 MANIFEST 裡的三個檔案。這一檔用 AST 檢查
那三個檔案有沒有偷偷相依到專案裡的別的東西——一旦有，移植過去就會 ImportError。

這是規則本身的測試，不是功能測試。改動這個 repo 時如果這裡紅了，
代表可攜性破了，不是測試寫錯了。
"""

import ast
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LIB = HERE.parent

#: 單元 -> (測試檔, 合成場景檔)。移植時這三個一起帶走。
MANIFEST = {
    "box_measure":  ("test_box_measure.py",  "synthetic.py"),
    "fit_check":    ("test_fit_check.py",    "synthetic_fit.py"),
    "pallet_stack": ("test_pallet_stack.py", "synthetic_stack.py"),
}

#: 量測單元允許的第三方相依。除此之外只能用標準函式庫。
UNIT_ALLOWED = {"numpy", "cv2", "joblib"}

#: 測試檔額外允許的東西。
TEST_EXTRA = {"pytest"}

#: 專案內部模組——單元與其測試都不該碰到這些。
PROJECT_MODULES = {
    "realsense_source", "rfdetr_detector", "viz", "run_live",
    "config", "log_manager", "camera_manager", "vision_engine",
}


def _imported_names(path: Path) -> set:
    """收集一個檔案 import 的所有頂層模組名（含函數內的延後 import）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _is_stdlib(name: str) -> bool:
    return name in sys.stdlib_module_names


@pytest.mark.parametrize("unit", sorted(MANIFEST))
def test_移植清單裡的檔案都存在(unit):
    test_file, synth_file = MANIFEST[unit]
    assert (LIB / f"{unit}.py").exists(), f"找不到單元 {unit}.py"
    assert (HERE / test_file).exists(), f"找不到測試 {test_file}"
    assert (HERE / synth_file).exists(), f"找不到合成場景 {synth_file}"


@pytest.mark.parametrize("unit", sorted(MANIFEST))
def test_單元只依賴numpy與標準函式庫(unit):
    outside = {n for n in _imported_names(LIB / f"{unit}.py")
               if not _is_stdlib(n) and n not in UNIT_ALLOWED}
    assert not outside, f"{unit}.py 相依到了 {sorted(outside)}"


@pytest.mark.parametrize("unit", sorted(MANIFEST))
def test_單元不碰專案內部模組(unit):
    hit = _imported_names(LIB / f"{unit}.py") & PROJECT_MODULES
    assert not hit, f"{unit}.py import 了專案內部模組 {sorted(hit)}"


@pytest.mark.parametrize("unit", sorted(MANIFEST))
def test_測試檔只認識自己的單元(unit):
    """
    單元的測試檔不得 import 接線層。跨層的整合測試放 test_adapters.py，
    那一檔移植時不用帶走。
    """
    test_file, synth_file = MANIFEST[unit]
    allowed = UNIT_ALLOWED | TEST_EXTRA | {unit, synth_file[:-3]}

    for path in (HERE / test_file, HERE / synth_file):
        outside = {n for n in _imported_names(path)
                   if not _is_stdlib(n) and n not in allowed}
        assert not outside, (
            f"{path.name} 相依到了 {sorted(outside)}——"
            f"搬走 {unit} 時會 ImportError。跨層測試請放 test_adapters.py。")


def test_三個單元互不相依():
    """單元之間不能互相 import，否則搬一個要搬三個。"""
    units = set(MANIFEST)
    for unit in units:
        hit = _imported_names(LIB / f"{unit}.py") & (units - {unit})
        assert not hit, f"{unit}.py import 了其他單元 {sorted(hit)}"
