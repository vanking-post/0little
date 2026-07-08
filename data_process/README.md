# 交通流数据处理流水线 — Data Pipeline

处理高速公路分流区 5 个 Location 的原始轨迹数据（XLSX/CSV），
经过拆分、清洗、补全、平滑、特征重构、变道/跟驰样本提取等步骤，
生成可用于风险建模分析的标准数据集。

---

## 环境依赖

```bash
pip install pandas numpy scipy matplotlib openpyxl
```

---

## 流水线总览

```
原始数据 (XLSX/CSV)
    │
    ▼
Step 1 ──── split_by_direction()    ──── 按 Direction(1/2) 拆分为上下行
    │
    ▼
Step 2 ──── clean_by_direction()    ──── 异常帧清洗、插值修复、列对齐
    │
    ▼
Step 3 ──── complete_by_direction() ──── 基于 Frame 连续性补全缺失帧
    │
    ▼
Step 4 ──── smooth_by_direction()   ──── Savgol 滤波平滑 X/Y/速度/加速度
    │
    ▼
Step 5 ──── compute_features_with_mttc() ─── 特征重构：距离/TTC/mTTC/THW/RSD/Jerk
    │
    ▼
Step 6 ──── extract_lane_change/following_samples() ─── 变道/跟驰样本提取
    │
    ▼
输出: traffic_left_change.csv / traffic_right_change.csv / traffic_following_change.csv
```

---

## 脚本说明

### 1. `process_all_locations.py` — 主调度器

批量处理 location1~5 的完整流水线。根据 Location 自动选择处理路径：

| 流程段 | location1~4 | location5 |
|--------|-------------|-----------|
| 数据读取 | `read/location{1-4}/*.xlsx` → CSV | `read/location5/5_trajectory.xlsx` (双 sheet 合并) |
| Step 1~5 | 共用 `step01.py` ~ `step05_sample.py` | 共用同一套 step 函数 |
| Step 6 | 内置 lane change / following 提取 | 加载 `location5/step06change.py` (独立模块) |

**关键配置**（脚本顶部）：

```python
BASE_READ = r"E:\0little\read"         # 原始数据目录
BASE_OUT  = r"E:\0little"              # 输出目录（各 location 文件夹下）
LANE_COEFFS_FILE = r"E:\0little\lane_coeffs.xlsx"  # 车道线系数
REACTION_TIME = 2.0                    # 制动反应时间 (s)
```

**主要函数：**

| 函数 | 功能 |
|------|------|
| `convert_xlsx_to_csv(xlsx_path, csv_path)` | XLSX → CSV 转换，修复浮点精度 |
| `process_single_file(csv_path, out_dir)` | 单文件完整流水线 (step1~6 + 安全编码 + 保存) |
| `process_location5()` | location5 专用处理（数据格式特殊 + 独立 step06） |
| `visualize_location(out_dir)` | 可视化：随机抽取车辆叠加车道线绘制轨迹 |
| `load_lane_coeffs()` | 从 `lane_coeffs.xlsx` 读取车道线系数 |
| `main()` | 主入口，遍历 5 个 location 并调用上述函数 |

**用法：**

```bash
cd E:/0little/data_process
python process_all_locations.py
```

---

### 2. `step01.py` — 按方向拆分

**函数：** `split_by_direction(file_path)`

将原始轨迹 CSV 按 `Direction` 列拆分为上行/下行两个 DataFrame。

| 项目 | 说明 |
|------|------|
| 输入 | XLSX 或 CSV 文件路径（需含 `Direction` 列） |
| 输出 | `(df_dir1, df_dir2)` — Direction=1 和 Direction=2 的 DataFrame |
| 处理 | 修复 Frame 浮点精度；清理列名中的特殊符号 |

---

### 3. `step02clean.py` — 数据清洗

**函数：** `clean_by_direction(df_dir1, df_dir2, ...)`

对上下行数据分别执行以下清洗：

- **单调性判断**：检查 X/Y 随时间是否单调变化
- **异常帧移除**：剔除跳跃过大、方向错误的帧
- **线性插值**：复原小间隙缺失值
- **列对齐**：确保两个方向列结构一致

**辅助函数：**
- `check_monotonicity(values)` — 判断数据是否近似单调
- `get_monotonicity_direction(values)` — 判断单调方向（增/减/持平）

---

### 4. `step03complete.py` — 数据补全

**函数：** `complete_by_direction(df_clean_1, df_clean_2, df_raw_1, df_raw_2, ...)`

基于 Frame 连续性检测断点，优先从原始数据恢复缺失帧，否则线性插值填补。

**策略：**
1. 按 `[ID, Frame]` 排序检查连续性
2. 对帧号不连续的位置，尝试从 `df_raw` 恢复
3. 若原始数据也缺失 → 线性插值
4. 每辆车首尾 1 帧内的缺失用最近邻填充

---

### 5. `step04smooth.py` — Savgol 平滑

**函数：** `smooth_by_direction(df_1, df_2, ...)`

使用 Savitzky-Golay 滤波器对轨迹数据进行平滑：

| 参数组 | 窗口/阶数 | 适用列 |
|--------|----------|--------|
| `xy` | (9, 3) | `X`, `Y` |
| `vel` | (5, 3) | `long_Vel`, `lat_Vel` |
| `acc` | (9, 3) | `long_Acc`, `lat_Acc` |
| `other` | (9, 3) | 标线距离等 |

---

### 6. `step05_sample.py` — 特征重构

**函数：** `compute_features_with_mttc(df_1_smooth, df_2_smooth, ...)`

对平滑后数据计算全部微观交通指标：

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| `Following_dist` | 欧几里得距离 − 车辆半长 | 保险杠到保险杠的跟车距离 |
| `B_Dist` | 同 | 后车距离 |
| `LB/RB/LF/RF_Dist` | 同 | 左后/右后/左前/右前邻车距离 |
| `Time_Headway` | `Following_dist / long_Vel` | 车头时距 (THW) |
| `TTC` | `Following_dist / Δv` | 碰撞时间 |
| `mTTC` | `TTC × exp(−Time_Headway / k)` | 修正碰撞时间 |
| `Lateral_Jerk` | `diff(lat_Acc) / diff(Time)` | 横向加加速度 |
| `RSD` | 停车距离模型 | 危险停车距离 |
| `F_ERSD` | 同 | 前车紧急停车距离 |
| `B_ERSD` | 同 | 后车紧急停车距离 |
| `PET` | 冲突点时空差 | 后侵入时间 |
| `OL_PET` | 原车道后车 PET | 原车道后侵入时间 |

参数：`reaction_time=2.0`（制动反应时间）

---

### 7. `step06change.py` — 变道/跟驰样本提取

**函数：**

| 函数 | 功能 |
|------|------|
| `extract_lane_change_samples(df1, df2, ...)` | 提取左变道和右变道样本 |
| `extract_following_samples(df1, df2, ...)` | 提取跟驰样本（无变道行为） |

**变道样本提取逻辑：**
1. 检测 `LaneID` 变化帧 → 确定变道起始帧
2. 取变道帧前后各 offset 帧作为样本窗口（100+50+100=250 帧，即前 100 帧 + 变道 50 帧 + 后 100 帧）
3. 过滤掉冲突时间过大的样本
4. 计算 `PET`（后侵入时间）用于风险评价

**参数：**
- `offset=5` — 读取邻车 ID 时的偏移帧数
- `conflict_tolerance=1.5` — 冲突点 X 坐标匹配误差 (m)
- `pre_frames=100, sample_frames=50` — 样本窗口配置

---

### 8. `step_visualizeXY.py` — 轨迹可视化

**常量：**

| 变量 | 含义 |
|------|------|
| `lane_coeffs_dir11` | 方向 1 内四条车道的多项式系数 |
| `lane_coeffs_dir12` | 方向 1 外四条车道 |
| `lane_coeffs_dir21` | 方向 2 内四条车道 |
| `lane_coeffs_dir22` | 方向 2 外四条车道 |

多项式形式：`y = a5·x⁵ + a4·x⁴ + a3·x³ + a2·x² + a1·x + a0`

**函数：** `visualize_lane_change_samples(full_data, sample_data, save_dir=None)`

为跟驰、左变道、右变道三个类别分别生成轨迹对比图（各 10 辆随机抽样）。

---

### 9. `model.py` — 安全编码 + 单 location 处理

**函数：** `encode_safety_categories(df)`

为 TTC、PET、mTTC、THW 等安全指标添加分类编码列：

| 列名 | 取值 |
|------|------|
| `TTC_cat` | `no_leader` / `dangerous` / `cautious` / `safe` |
| `PET_cat` | `no_follower` / `dangerous` / `cautious` / `safe` |
| `mTTC_cat` | 同 TTC_cat（无邻车=0时=no_leader） |
| `Time_Headway_cat` | `no_leader` / `dangerous`(<1.5s) / `cautious`(<3s) / `safe` |
| `has_front_vehicle` | bool，是否有前车 |
| `has_rear_vehicle` | bool，是否有后车 |

**主流程**（`__main__` 中直接调用）：
1. 加载 `1-1_trajectory.csv` 和 `1-2_trajectory.csv`
2. 运行 step1~6 完整流水线
3. 输出结果到 `E:/0little/location1/`

---

### 10. `merge_location1.py` — 子文件合并

将 location1 各 Source（1-1, 1-2）的变道数据合并为统一文件：

```
traffic_1-1_left_change.csv  ─┐
traffic_1-2_left_change.csv  ─┤→ traffic_left_change.csv
traffic_1-1_right_change.csv ─┤→ traffic_right_change.csv
traffic_1-2_right_change.csv ─┘
```

---

## 输入数据格式

### 原始数据 (`E:/0little/read/location{1-4}/`)

XLSX 文件，含以下关键列：

| 列 | 描述 |
|----|------|
| `ID` | 车辆唯一标识 |
| `Frame` | 帧号（时间步） |
| `X`, `Y` | 车辆位置 (m) |
| `Direction` | 行驶方向 (1=上行, 2=下行) |
| `LaneID` | 车道编号 |
| `Velocity` | 速度 |
| `Length`, `Width` | 车辆长宽 (m) |

### 最终输出 (`E:/0little/location{1-5}/`)

| 文件 | 内容 |
|------|------|
| `traffic_left_change.csv` | 左变道样本（每车 ~250 帧，含 PET） |
| `traffic_right_change.csv` | 右变道样本 |
| `traffic_following_change.csv` | 跟驰样本（无变道行为） |
| `trajectory_full.csv` | 全量轨迹（含所有 ID） |
| `trajectory_full_smoothed.csv` | 平滑后全量轨迹 |

---

## 运行方式

完整处理所有 5 个 Location：

```bash
cd E:/0little/data_process
python process_all_locations.py
```

单独合并 location1 的子文件：

```bash
cd E:/0little/data_process
python merge_location1.py
```

单 location 快速测试（使用 model.py 中的主流程）：

```bash
cd E:/0little/data_process
python model.py
```

> **注意：** 所有脚本使用绝对路径 (`E:\0little\...`)，无需修改工作目录。
> `process_all_locations.py` 已自动将自身目录加入 `sys.path`，可被任意位置调用。
