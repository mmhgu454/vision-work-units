"""讓 tests/ 底下的測試找得到 repo 根目錄的模組，不需要安裝、不需要改 import 結構。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
