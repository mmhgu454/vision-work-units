# 操作手冊

這份只講**怎麼開、怎麼按**。演算法、移植到別的專案，看 [README.md](README.md)。

每一節都是：先給一行可以直接複製的指令，再說畫面上要按什麼、看什麼。

---

## 0. 每次開工先做

**1. 進到 repo 根目錄**（下面所有指令都從這裡下）

```bash
cd vision-work-units
```

**2. 設定要用哪個 Python**（每開一個新的終端機視窗做一次）

```bash
PY="env -u PYTHONPATH ../fgo/.fgovenv/bin/python"
```

之後指令裡的 `$PY` 就是指它。
- `../fgo/.fgovenv` 是 gt-1 這台機器上已經裝好全部套件的環境。換機器就換成你自己的環境路徑。
- `env -u PYTHONPATH` 是把 shell 自動帶進來的 ROS 路徑拿掉，避免拿到錯的套件版本。

**3. 確認相機有接上**

```bash
$PY -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) + ' USB' + d.get_info(rs.camera_info.usb_type_descriptor) for d in rs.context().query_devices()])"
```

要看到 `['Intel RealSense D435 USB3.2']` 之類的字。出現 `[]` 就是沒接好；`USB2` 開頭代表插錯孔，要換 USB 3 的孔。

> 同一時間只能有**一支**程式開相機。開新的之前，先把上一個預覽視窗按 ESC 關掉。

---

## 功能一覽

| 我要… | 指令 | 在第幾節 |
|---|---|---|
| 量相機的俯仰角、離地高度 | `$PY estimate_pitch.py` | 1 |
| 看 task1 判斷下一箱放哪（畫在畫面上） | `$PY task1_place_view.py --station g8_1 …` | 2 |
| 跑 task1 完整流程（含 AI 模型） | `$PY run_live.py --task 1 --preview …` | 3 |
| task2 擺放位置檢查 | `$PY run_live.py --task 2 --preview …` | 4 |
| task3 量紙箱尺寸 | `$PY run_live.py --task 3 --preview …` | 5 |
| task4 讀條碼 | `$PY run_live.py --task 4 --preview` | 6 |
| 拿存好的資料重跑（不用相機） | `$PY run_live.py --replay fixtures/` | 7 |
| 跑自動測試（不用相機） | `$PY -m pytest tests -q` | 8 |

---

## 1. 量相機姿態：`estimate_pitch.py`

task1 要知道相機往下看幾度、離地多高。這支直接用地面量出來。

```bash
$PY estimate_pitch.py
```

**操作**

| 按鍵 / 動作 | 作用 |
|---|---|
| 滑鼠拖曳 | 框一塊**確定是平坦地面**的區域（隨時可以重框） |
| SPACE | 把目前數值印到終端機，最後一行可以直接複製給 task1 用 |
| h | 顯示 / 隱藏高度 |
| ESC | 離開 |

**看什麼**
- `pitch`：往下看的角度，負數 = 往下。給 task1 的 `--tilt`。
- `height`：鏡頭離地高度。給 task1 的 `--phy-height`。
- `resid` 變紅、出現 `NOT FLAT? re-draw`：框到的不是平地（框到箱子或牆了），重框。
- 按 SPACE 會印出「畫面右方相對水平」。task1 預設相機是**轉 90° 架、畫面右邊朝上**，這個值要約等於 **90 + pitch**（例如 pitch -38 → 約 +52°）。如果是負的或接近 0，代表相機轉向不對，task1 的地面會算歪。

相機轉 90° 或不轉，量到的 pitch 都一樣，不用擔心。

---

## 2. 看 task1 的擺放判斷：`task1_place_view.py`

即時把「程式判斷下一箱放哪」畫在相機畫面上，擺箱子時邊看邊對。

```bash
$PY task1_place_view.py --station g8_1 --cardboard L554001 --tilt -38 --phy-height 2.25
```

`--tilt`、`--phy-height` 用第 1 節量到的值。

**畫面**

| 看到 | 意思 |
|---|---|
| 白框 | 每個候選位置的範圍（一格約一個箱子大） |
| 黃框 | 程式**實際取樣**的小塊。只有這裡的高度會被算進去 |
| `(排,欄) L層數 高度` | 每格目前幾層。`NO DATA` = 相機看不到，會當成 0 層 |
| 綠色立體框 | 下一箱要放的位置 |
| 右邊俯視圖 | 由上往下看的高度圖，下方靠相機；斜線 = 看不到的地方 |

**按鍵**：`s` 存圖到 `fixtures/png/`，`ESC` 離開。

**格子怎麼切**（依工作站自動決定）

| 工作站 | 模式 | 格數 |
|---|---|---|
| `g4_1`、`g4_2` | 地面（ground），依箱子大小切 | 2 排 × 2 欄 = 4 格 |
| `g8_1` | 地面（ground），依箱子大小切 | 4 排 × 2 欄 = 8 格 |
| `p_1` | 棧板（pallet），只看四個角落 | 2 × 2 = 4 格 |

（以 L554001 為例；換箱型排數會變。）要強制指定模式就加 `--block-type ground` 或 `--block-type pallet`。

**擺相機的位置**：程式假設工作區從「**相機正下方的地面點**往前 `camera_distance`」開始，左右置中。g8_1 是往前 1.10 m 開始、深 2.10 m、寬 1.2 m。格子和實際箱子錯開的話，先檢查這個距離。各站的數字在 `pallet_stack.py` 的 `DEFAULT_BLOCK_CATALOG`。

---

## 3. task1 完整流程：`run_live.py --task 1`

跟第 2 節判斷一樣，另外按 SPACE 會載入 AI 模型，做偵測與評分。

```bash
$PY run_live.py --task 1 --preview --station g8_1 --cardboard L554001 --tilt -38 --phy-height 2.25
```

| 按鍵 | 作用 |
|---|---|
| SPACE | 跑完整流程（第一次要載模型，比較久）。結果出來後：`s` 存檔、其他鍵丟棄 |
| ESC | 離開 |

- 要 SVR 評分就加 `--score-model model/pallet_svr_model.joblib`。
- 模型檔放在 `model/`（不在版控裡，要另外拿）。
- 只想看判斷、不想載模型：加 `--no-model`，或直接用第 2 節的工具。

---

## 4. task2 擺放位置檢查：`run_live.py --task 2`

```bash
$PY run_live.py --task 2 --preview --position 1,1 --cardboard L554001
```

- `--position 列,欄`：第二個數字是 `1` = 左站，其他 = 右站。
- `--block-type ground`：在地面上檢查（不給是棧板）。
- 不需要 AI 模型。

| 按鍵 | 作用 |
|---|---|
| SPACE | 跑完整檢查（約 0.5 秒）。結果出來後：`s` 存檔、其他鍵丟棄 |
| ESC | 離開 |

畫面上的黃框（基準框）要落在要量高度的平面上；左上角點數不足 50 時整個畫面會加紅框。

---

## 5. task3 量紙箱尺寸：`run_live.py --task 3`

```bash
$PY run_live.py --task 3 --preview --rot-k 0 --label L554001
```

- `--label`：這個箱子實際的箱型，用來比對對錯。
- 每次量測連抓 5 幀取中位數（`--frames` 可改）。
- 需要 AI 模型（`model/`）。

| 按鍵 | 作用 |
|---|---|
| SPACE | 拍攝並量測。結果出來後：`s` 存檔、其他鍵丟棄 |
| d | 切換深度圖 |
| ESC | 離開 |

⚠ 箱子的框不能碰到畫面邊緣，碰到的話高度會偏小、但數字看起來仍正常。標註圖會畫黃框警告。

---

## 6. task4 讀條碼：`run_live.py --task 4`

**先改解析度**（只有 D435 有 1080p，細條碼需要它）：打開 `realsense_source.py`，改這兩行

```python
DEFAULT_COLOR_WIDTH = 1920
DEFAULT_COLOR_HEIGHT = 1080
```

> 測完記得改回 `1280` / `720`，否則 task1~3 也會跟著用 1080p。

```bash
$PY run_live.py --task 4 --preview
```

| 按鍵 | 作用 |
|---|---|
| SPACE | 辨識，並把原始畫面存到 `fixtures/barcode/`（檔名結尾 `_ok` / `_fail`） |
| s | 只存圖、不辨識 |
| ESC | 離開 |

**怎麼拍才讀得到**
- 條碼對準畫面中央的**十字**，旁邊會顯示距離。**變綠色（20~25 cm）再按 SPACE。**
- 太遠：條碼太小，像素不夠。太近：失焦（D435 彩色鏡頭是定焦的）。
- 條碼歪斜（在平面上旋轉）沒關係；斜著看（透視）還沒驗證過。
- 讀到的數值一定是對的（實測零誤判）；讀不到就重拍。
- D455 讀不到這種細的 Code128，請用 D435。

**其他用法**

```bash
$PY barcode_read.py 照片.jpg --show                    # 讀照片，不需要相機
$PY barcode_read.py fixtures/barcode/*_fail.png        # 把讀不到的圖重讀一次
$PY barcode_read.py --webcam 0                         # 改用 USB webcam
$PY run_live.py --task 4 -n 3                          # 不開視窗，連讀 3 次
```

需要 `pyzbar` 套件和系統的 `libzbar0`，缺的話程式一啟動就會報錯並告訴你要裝什麼。

---

## 7. 存資料、離線重跑

在預覽裡按 `s` 存下來的資料，之後不接相機也能重跑。

| 資料 | 存在哪 | 怎麼重跑 |
|---|---|---|
| task1~3 的現場資料（`.npz`） | `fixtures/` | `$PY run_live.py --replay fixtures/` |
| 匯出的圖片 | `fixtures/png/` | — |
| task4 條碼照片 | `fixtures/barcode/` | `$PY barcode_read.py fixtures/barcode/*.png` |

```bash
$PY run_live.py --replay fixtures/                    # 整個資料夾重跑，三種 task 都認得
$PY run_live.py --replay fixtures/某個檔.npz --export  # 順便匯出 PNG
```

- task1 的資料存的是**還沒校正**的點雲，重跑時可以換 `--tilt` / `--phy-height` 看結果怎麼變。
- task3 加 `--redetect` 會重跑模型；再加 `--rewrite` 會把新結果寫回原檔。
- 這些資料夾不進版控，**拍到難得的資料要自己另外備份**。

---

## 8. 跑自動測試

改完程式碼後跑一次，不需要相機、不需要模型，約 1 秒。

```bash
$PY -m pytest tests -q
```

全部 `passed`（少數 `skipped` 正常）就沒問題。

---

## 常見狀況

| 看到 | 原因 | 怎麼辦 |
|---|---|---|
| `Couldn't resolve requests` | 相機不支援要求的解析度組合，或相機被別的程式占用 | 關掉其他開著相機的視窗；檢查 `realsense_source.py` 的解析度（D435/D455 深度最高 1280×720，1080p 彩色只有 D435 有） |
| `No device connected` | 沒偵測到相機 | 重插 USB（要 USB 3 孔），再跑第 0 節的檢查 |
| `Frame didn't arrive within 5000` | 相機有開但收不到畫面，多半是被占用或 USB 不穩 | 關掉其他程式、重插 USB |
| 條碼一直讀不到 | 距離不對、失焦、反光，或解析度不是 1080p | 照第 6 節：先改 1080p，等距離變綠再按 |
| task1 格子跟實際箱子錯開 | 相機到工作區的距離、或相機姿態跟設定不符 | 用第 1 節重量姿態；量相機正下方到膠帶前緣的距離，跟 `camera_distance` 比對 |
| task1 地面不是 0、層數怪怪的 | `--tilt` / `--phy-height` 不準 | 用第 1 節重量，並確認「畫面右方相對水平」≈ 90 + pitch |

---

## 設定值在哪改

| 要改什麼 | 檔案 | 名稱 |
|---|---|---|
| 相機解析度 | `realsense_source.py` | `DEFAULT_WIDTH/HEIGHT`（深度）、`DEFAULT_COLOR_WIDTH/HEIGHT`（彩色） |
| task1 相機姿態預設值 | `pallet_stack.py` | `DEFAULT_TILT`、`DEFAULT_PHY_HEIGHT`（臨時試用就用 `--tilt` / `--phy-height`） |
| task1 各工作站的位置與大小 | `pallet_stack.py` | `DEFAULT_BLOCK_CATALOG` |
| 箱型尺寸 | `pallet_stack.py` | `DEFAULT_BOX_DIMENSIONS` |
| task2 相機姿態 | `realsense_source.py` | `TASK2_POSE_LEFT`、`TASK2_POSE_RIGHT` |
| task4 綠色距離範圍 | `barcode_read.py` | `GOOD_RANGE_M` |
