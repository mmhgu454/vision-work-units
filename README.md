# vision-work-units

一組**可移植的視覺量測單元**，附一套 RealSense + RF-DETR 的參考接線。

每個單元是一個獨立的 `.py` 檔，只依賴 numpy（或 numpy + cv2）。
取幀與物件偵測是**注入**進去的，所以換相機、換模型都不用改單元。
移植的方式是把單元連同它的測試複製到新專案——見〈把單元搬到別的專案〉。

| 單元 | 做什麼 | 相依 |
|---|---|---|
| `box_measure.py` | 從偵測框 + 點雲量出紙箱長寬高，比對箱型 | numpy |
| `fit_check.py` | 判斷指定位置放不放得下箱子 | numpy + cv2 |
| `pallet_stack.py` | 判斷棧板各位置堆了幾層、下一箱該放哪 | numpy |

> 來源：從一個貨物堆疊的視覺專案拆出來。原始碼的行號引用（`src/…`、`vision_engine.py:…`）
> 指的是那個專案，不在這個 repo 裡。


---

## 30 秒上手

```bash
PY=~/code/fgo/.fgovenv/bin/python

# 跑測試（不需要相機、不需要模型，1.4 秒）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests -q

# 接上相機，開即時預覽
$PY run_live.py --task 3 --preview --rot-k 0 --label L554001   # 紙箱尺寸
$PY run_live.py --task 2 --preview --position 1,1              # 擺放檢查
$PY run_live.py --task 1 --preview --station g4_1              # 堆疊判定

# 拿存好的資料離線重跑，不需要相機
$PY run_live.py --replay fixtures/ --rot-k 0
```

> ⚠ `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 不能省。這個 venv 的 python 是系統 python3 的
> symlink，`/opt/ros/humble` 的 `launch_pytest` 外掛會被自動載入，而它本身壞的（缺 `lark`），
> 會讓 pytest 在收集測試前就崩潰。

---

## 檔案地圖

### ★ 三個可攜單元 — 複製單一檔案到別的專案就能用

| 檔案 | 對應 | 相依 | 做什麼 |
|---|---|---|---|
| `box_measure.py` | task3 | **numpy** | 從偵測框 + 點雲量出紙箱長寬高，比對箱型 |
| `fit_check.py` | task2 | **numpy + cv2** | 判斷指定位置放不放得下箱子 |
| `pallet_stack.py` | task1 | **numpy**（+joblib 評分才要） | 判斷棧板各位置堆了幾層、下一箱該放哪 |

這三個檔案**互不 import**，也不 import 專案內任何模組。各自有一個測試在乾淨子行程裡驗這件事：

```python
def test_沒有拉進硬體或專案相依():   # 出現 pyrealsense2 / config / log_manager 就紅燈
```

### 本專案的接線 — 換相機/換模型時改這些

| 檔案 | 職責 |
|---|---|
| `realsense_source.py` | 取幀：開 pipeline、算點雲、套座標系修正 |
| `rfdetr_detector.py` | 偵測：載模型、影像轉正、框轉回來 |
| `viz.py` | 畫圖：偵測框標註、高度圖、深度有效性圖 |
| `run_live.py` | 手動測試驅動：即時預覽、錄 fixture、離線重跑、匯出 PNG |

---

## 三個任務怎麼跑

### task3 — 紙箱尺寸量測

```bash
$PY run_live.py --task 3 --preview --rot-k 0 --label L554001
```

預覽視窗：`SPACE` 拍攝並量測 → `s` 存檔 / 其他鍵丟棄 → `d` 切換深度圖 → `ESC` 離開。

⚠ **face 框不能碰到畫面邊緣。** 一碰到，箱子下緣被裁掉，高度會量成偏小值但數字看起來仍合理。
`measure_box` 會把這件事放進 `result.warnings`，標註圖會畫黃框。用 `result.trustworthy`
判斷「成功且無警告」。

### task2 — 擺放位置檢查

```bash
$PY run_live.py --task 2 --preview --position 1,1 --cardboard L554001
```

`--position` 第二個數字決定左右站（`1` = 左，其他 = 右）。

預覽畫面：**黃框 = 基準框**（必須落在要量高度的平面上）、**紫框 = 搜尋範圍**，
左上顯示基準框內點數與即時 `base_h`。點數不足 50 時整個畫面加紅框。

每幀只畫這層疊圖（約 2 ms）；完整的 `check_fit` 在 1280×800 要 **0.5 秒以上**，
所以只有按 `SPACE` 才跑。

### task1 — 棧板堆疊判定

```bash
$PY run_live.py --task 1 --preview --station g4_1 --cardboard L554001
```

工作站：`g4_1` / `g4_2` / `g8_1` / `p_1`。`--block-type pallet|ground` 切棧板或地面。

右側是高度圖，格內數字是層數，綠框是目前會擺放的位置。**堆疊判定不需要 AI 模型**
（只有優先度評分需要偵測框），所以每幀就能算（0.058 s / 17 fps），`SPACE` 才載模型跑完整流程。
想完全不碰模型就加 `--no-model`。

要用 SVR 優先度評分：`--score-model model/pallet_svr_model.joblib`。

---

## Fixture：錄一次，離線重跑無數次

```bash
# 錄（預覽中按 s，或直接 --save）
$PY run_live.py --task 1 --station g4_1 --save --export

# 重跑：吃單一檔案或整個資料夾，三種 task 的 fixture 都認得
$PY run_live.py --replay fixtures/

# 匯出 PNG（原圖 / 標註圖 / 高度圖）
$PY run_live.py --replay fixtures/ --export
```

**task3 專屬**：`--redetect` 會對存好的畫面重跑模型（補完 `--no-model` 拍的 fixture，
或試不同 `--conf`），加 `--rewrite` 把新結果寫回原檔。

---

## 相機姿態怎麼調

三個 task 的座標系**不一樣**，各自的旋鈕也不同：

| task | 點雲格式 | 座標處理 | 在哪調 |
|---|---|---|---|
| task3 | `(H, W, 3)` | Z 軸 180° | `RealSenseSource(pc_rotation_deg=)` |
| task2 | 扁平 `(N, 3)` | ±90° + 傾角/高度/鏡像校正 | `CameraPose(tilt, phy_height, flip)` |
| task1 | `(H, W, 3)` | Z 軸 -90°，**校正在演算法內部做** | `pallet_stack.make_calibrator(tilt, phy_height, flip)` |

現場預設值（抄自 `src/config.py`）：task1 `tilt=-35.0, height=2.25`；task2 `tilt=-72.0, height=2.2`。

### 兩步驟校正法（順序不能反）

```bash
# 1. 對著一個你知道實際高度的水平面拍一張
$PY run_live.py --task 2 --position 1,1 --save

# 2. 離線掃描，不需要相機
$PY run_live.py --replay fixtures/<檔名>.npz --sweep
```

掃描表有兩欄關鍵數字：

- **`dy/dz`** — 水平面在深度方向的殘餘斜率。**傾角正確時為 0**，而且完全不受高度影響。
  **先用這欄定 `tilt`。**
- **`base_h`** — 量到的表面高度。`phy_height` 對它是 1:1 平移。tilt 定了之後再用這欄
  把 `base_h` 調到你實際量到的高度。

單次試用 `--tilt -70 --phy-height 2.15 --no-flip`，確定了再寫回 `realsense_source.py` 的常數。

⚠ **改了 pose 就要重新量 `fit_check` 的 `references` / `bounds`。** 那些 ROI 是世界座標下的
x/z 範圍，姿態一改就全部偏掉。它們是 `check_fit()` 的參數，可以覆寫不必改檔案。

⚠ task1 / task2 的 fixture 存的是**未校正**的點雲，所以調 pose 不用重拍。

---

## 把單元搬到別的專案

### 帶走三個檔案

| 要移植的能力 | 單元 | 它的測試 | 它的合成場景 |
|---|---|---|---|
| task3 紙箱尺寸量測 | `box_measure.py` | `tests/test_box_measure.py` | `tests/synthetic.py` |
| task2 擺放位置檢查 | `fit_check.py` | `tests/test_fit_check.py` | `tests/synthetic_fit.py` |
| task1 棧板堆疊判定 | `pallet_stack.py` | `tests/test_pallet_stack.py` | `tests/synthetic_stack.py` |

**測試要一起帶。** 那不只是「檢查有沒有壞」——它把座標系契約、三個 bug 的迴歸守門、
以及演算法在合成場景下的量測特性都編碼在裡面。到了新專案 `pytest` 一跑就知道還對不對，
而且不需要相機、不需要模型、1.5 秒跑完。

三個檔案放同一層即可，不需要 `conftest.py`：

```bash
mkdir newproject/vision && cd newproject/vision
cp .../box_measure.py .
cp .../tests/test_box_measure.py .
cp .../tests/synthetic.py .
pytest -q            # 23 passed
```

**不需要帶走的**：`tests/test_adapters.py`（那是這個專案的取幀/偵測/畫圖的整合測試）、
`tests/test_portability.py`、`viz.py`、`run_live.py`。

> `tests/test_portability.py` 用 AST 守著這條規則：單元與它的測試若相依到專案內部模組
> （`viz` / `realsense_source` / `rfdetr_detector` / `config` …）就紅燈。
> 跨層的整合測試請放 `test_adapters.py`。

### 呼叫方式

複製整個檔案，或只挑需要的函數（見上一節的相依地圖），然後餵它三樣東西：

```python
from box_measure import measure_box

result = measure_box(
    color_img,        # (H, W, 3) BGR，原始方向不要預先旋轉
    pc_np,            # (H, W, 3) 點雲，座標系見檔頭 docstring
    detect,           # callable: detect(img) -> (boxes, labels, scores)，框與 img 同方向
    view_rot_k=0,     # 把輸入影像轉正立所需的 rot90 次數；相機正裝就是 0
)
result.box_type    # 箱型或 "Err"
result.lwh         # 量到的長寬高
result.warnings    # 可疑之處（例如框被畫面裁切）
result.trustworthy # 成功且無警告
```

三個單元的共同約定：

- **不碰相機、不碰檔案系統、不寫檔。** 中間結果全部隨回傳值帶出，要存圖是呼叫端的事。
- **設定內建但可覆寫**：箱型尺寸表、ROI 範圍、工作站區域全部有預設值，也全部是參數。
- **契約違反就明確報錯**（`ContractError`），不會靜默給你錯的數字。例如點雲格式錯、
  影像與點雲解析度不符、相機 roll 超出容許值、箱型的 ROI 還沒量。

各檔案開頭的 docstring 寫了完整的座標系契約，搬過去之前先讀那一段。

---

## 相依與安裝

**每個單元的相依不一樣，差距很大。** 只要量紙箱尺寸的話不需要裝 torch。

| 要用什麼 | `pip install` | 體積 |
|---|---|---|
| `box_measure.py`（task3 量測） | `numpy` | 41 MB |
| `fit_check.py`（task2 擺放檢查） | `numpy opencv-python` | 120 MB |
| `pallet_stack.py`（task1 堆疊判定） | `numpy` | 41 MB |
| `pallet_stack.py` 的 **SVR 優先度評分** | 追加 `joblib scikit-learn` | +48 MB |
| `realsense_source.py`（RealSense 取幀） | `pyrealsense2` | 32 MB |
| `rfdetr_detector.py`（RF-DETR 偵測） | `rfdetr`（會拉進 torch） | **1.7 GB** |
| `viz.py`（畫圖） | `opencv-python` | 79 MB |
| 跑測試 | 追加 `pytest` | 小 |

三個量測單元**都不需要 torch，也不需要 pyrealsense2**。偵測與取幀是注入進去的，
所以新專案可以用自己的相機、自己的模型，只要滿足契約即可。

```bash
# 最小：只要 task3 的量測
pip install numpy

# 完整的現場環境
pip install numpy opencv-python pyrealsense2 rfdetr joblib scikit-learn pytest
```

模型權重（`model/checkpoint_0702.pth`、`person-nano.pth`、`pallet_svr_model.joblib`）
不在版控裡，要另外取得。

---

## 複製片段時要一起帶走什麼

實際的移植方式通常不是複製整個檔案，而是把需要的函數挑出來貼進新專案的 class 或模組。
下表是每個函數的**內部相依閉包**——複製它時要一併帶走的同檔函數。

「獨立」代表那個函數只依賴 numpy／cv2，貼過去就能跑。

### `box_measure.py`（task3）

| 函數 | 要一起帶走 |
|---|---|
| `fit_plane_ransac_svd` — RANSAC + SVD 擬合平面 | **獨立** |
| `get_orthonormal_basis` — 由法向量建正交基底 | **獨立** |
| `find_closest_box_type` — 尺寸比對箱型 | **獨立** |
| `_box_center` / `_dist2` / `_rot90_box` / `_clipped_edges` | **獨立** |
| `calculate_box_lwh` — 量長寬高 | `fit_plane_ransac_svd`, `get_orthonormal_basis` |
| `select_top_and_face` — 挑配對的 top/face | `_box_center`, `_dist2` |
| `measure_box` — 完整流程 | 上面全部（10 個） |

### `fit_check.py`（task2）

| 函數 | 要一起帶走 |
|---|---|
| `draw_rotated_rect_on_mask` — 在遮罩上放矩形並算超出比例 | **獨立** |
| `fit_multi_segment_left_edge` — 多段邊界擬合 | **獨立** |
| `roi_mask` / `resolve_rois` / `_fit_status` / `_draw_base_vis` | **獨立** |
| `check_fit_from_rows` — 單次擺放檢查 | `draw_rotated_rect_on_mask`, `fit_multi_segment_left_edge` |
| `find_front_plane_size` — 建高度圖 + 四種嘗試 | 上面 5 個 |
| `check_fit` — 完整流程 | 上面全部（7 個） |

### `pallet_stack.py`（task1）

| 函數 | 要一起帶走 |
|---|---|
| `rotation_matrix` — 歐拉角轉旋轉矩陣 | **獨立** |
| `get_centroid` — ROI 質心 | **獨立** |
| `_fill_hmap` — 點雲投影成高度圖 | **獨立** |
| `_pick_best` — 挑層數最少的位置 | **獨立** |
| `extract_normalized_features` — 10 維特徵 | **獨立** |
| `_prepare_ground_points` / `_make_height_to_layer` / `load_score_model` | **獨立** |
| `make_calibrator` — 產生座標校正函數 | `rotation_matrix` |
| `get_object_3d_entries` — 偵測框轉 3D | `get_centroid` |
| `get_prior_pallet_score` — SVR 評分 | `extract_normalized_features`, `load_score_model` |
| `build_block_list` / `build_tape_block` — 堆疊判定 | 4 個底層函數 |
| `analyze_pallet` — 完整流程 | 上面全部（7 個） |

> 30 個函數裡有 17 個是完全獨立的。挑演算法片段時優先看那些。

**複製時記得一起看的東西**：每個檔案開頭的 docstring 寫了完整的座標系契約，
還有模組層級的常數（尺寸表、ROI 範圍、姿態預設值）。那些常數全部都是函數的預設參數，
不需要跟著複製，但要知道新環境該傳什麼進去。

---

## 常見問題

**Q: 量出來的尺寸普遍偏小、常常拿到 `Err`？**
合成資料實測顯示演算法有系統性低估（L/W 約 86%、H 約 97%，來自 5% 內縮與 2–98 百分位裁剪）。
但**真實資料上沒有出現**——真實點雲會延伸到偵測框之外。如果真的偏小，先看 `result.warnings`
有沒有「框貼到畫面邊緣」。`L/W` 比例守恆，可以用來分辨「整體縮水」還是「真的量錯」。

**Q: task2 一直說「基準區域點太少」？**
黃色基準框沒落在要量高度的平面上。用 `--task 2 --preview` 對準，點數不足 50 時畫面會加紅框。

**Q: `--rot-k` 設錯會怎樣？**
模型吃到顛倒的圖 → 偵測失敗或配對不到，`stage` 會顯示「找不到成對的 top 與 face」。
注意畫面只有**一個**箱子時，k 設錯也可能剛好過關（候選只有一個，配對邏輯的 fallback 會救回來），
多箱子時才會現形。偵測器與單元的 `view_rot_k` 不一致會直接拋 `ContractError`。

**Q: 測試怎麼跑都崩潰？**
少加 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。見本文最上方。
