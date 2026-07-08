# Data Statistics — 数据分析可视化脚本说明

对处理完成的标准数据集进行统计分析、分布可视化与特征探索。
所有脚本位于 `data_statistics/`，输出至同目录下的 `*_output/` 子文件夹。

---

## 1. `distributions_analysis.py` — 安全指标分布分析

### 功能
对 5 个 location 的全部变道车辆数据，计算并绘制各安全指标（TTC / mTTC / PET / OL_PET / Time_Headway）的分布直方图、箱线图、饼图、KDE 对比图、山脊密度图、风险热力图等。

### 输入
- `E:/0little/location{1-5}/traffic_left_change.csv`
- `E:/0little/location{1-5}/traffic_right_change.csv`
- `E:/0little/location{1-5}/traffic_following_change.csv`
- `E:/0little/location{1-5}/lane_coeffs.csv`（车道线系数，热力图使用）

### 关键参数

**指标配置 (METRICS)：**

| 指标 | 颜色 | 范围 | bins |
|------|------|------|------|
| TTC | `#c0392b` | 0~20s | 200 |
| mTTC | `#d35400` | 0~20s | 200 |
| PET | `#e74c3c` | 0~10s | 200 |
| OL_PET | `#8e44ad` | 0~12.5s | 200 |
| Time_Headway | `#2980b9` | 0~8s | 100 |

**风险等级阈值：**
- 变道：中风险≥0.40，高风险≥0.60
- 跟驰：中风险≥0.20，高风险≥0.35

### 输出

| 文件名 | 内容 | 来源函数 |
|--------|------|---------|
| `dist_TTC.png` | TTC 直方图 + 分布拟合 | `plot_distributions()` |
| `dist_mTTC.png` | mTTC 直方图 + 分布拟合 | 同上 |
| `dist_PET.png` | PET 直方图 + 分布拟合 | 同上 |
| `dist_OL_PET.png` | OL_PET 直方图 + 分布拟合 | 同上 |
| `dist_Time_Headway.png` | THW 直方图 + 分布拟合 | 同上 |
| `dist_kde_comparison.png` | 五指标 KDE 密度对比 | 同上 |
| `dist_boxplot.png` | 五指标箱线图对比 | `plot_boxplot()` |
| `dist_ol_pet_pie.png` | OL_PET_cat 饼图（按帧） | `plot_cat_distribution()` |
| `dist_ol_pet_bar.png` | OL_PET_cat 柱状图（按车） | 同上 |
| `dist_ETTC_hist.png` | ETTC 直方图 + 拟合 | `plot_ettc_distributions()` |
| `dist_F_ETTC_hist.png` | F_ETTC 直方图 + 拟合 | 同上 |
| `dist_B_ETTC_hist.png` | B_ETTC 直方图 + 拟合 | 同上 |
| `dist_ettc_kde.png` | ETTC/F_ETTC/B_ETTC KDE 对比 | 同上 |
| `dist_RSD_hist.png` | RSD 直方图 + 拟合 | `plot_rsd_distributions()` |
| `dist_F_ERSD_hist.png` | F_ERSD 直方图 + 拟合 | 同上 |
| `dist_B_ERSD_hist.png` | B_ERSD 直方图 + 拟合 | 同上 |
| `dist_rsd_kde.png` | RSD/F_ERSD/B_ERSD KDE 对比 | 同上 |
| `risk_by_location.png` | 山脊密度图（5 场景×跟驰/变道） | `plot_risk_distribution_by_location()` |
| `overall_risk_left_pie.png` | 左变道风险等级饼图 | `plot_overall_risk_distribution()` |
| `overall_risk_left_bar.png` | 左变道风险等级柱状图 | 同上 |
| `overall_risk_right_pie.png` | 右变道风险等级饼图 | 同上 |
| `overall_risk_right_bar.png` | 右变道风险等级柱状图 | 同上 |
| `risk_heatmap_location1.png` ~ `location5.png` | 各场景风险密度热力图 | `plot_risk_heatmap()` |

### 输出文件夹
`distributions_analysis_output/`，共约 27 张图。

---

## 2. `feature_correlation.py` — 特征相关性分析

### 功能
计算安全指标间的 Pearson 相关系数并可视化。同时分析各特征与风险标签的关联强度。

### 输入
- `E:/0little/location{1-5}/traffic_left_change.csv`
- `E:/0little/location{1-5}/traffic_right_change.csv`

### 关键参数
- 基准速度：location5=80km/h，其余=100km/h
- `plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']`

### 输出

| 文件名 | 内容 |
|--------|------|
| `feature_correlation.png` | 安全指标间相关系数热力图 |
| `feature_risk_correlation.png` | 安全指标与风险标签的相关性条形图 |

### 输出文件夹
`feature_correlation_output/`

---

## 3. `plot_trajectories.py` — 车辆轨迹平面图

### 功能
基于平滑全轨迹数据（`trajectory_full_smoothed.csv`）绘制 5 个 location 的车辆轨迹 XY 平面图，叠加车道线，区分上下行方向。

### 输入
- `E:/0little/location{1-5}/trajectory_full_smoothed.csv`（平滑后全轨迹）
- `E:/0little/location{1-5}/lane_coeffs.csv`（loc5 的车道线系数）
- `E:/0little/lane_coeffs.xlsx`（loc1-4 的车道线系数）

### 关键参数
- 车道线多项式：`y = a5·x⁵ + a4·x⁴ + a3·x³ + a2·x² + a1·x + a0`
- 轨迹采样：loc1-4 每方向 75 条，loc5 每方向 150 条
- 车道线颜色：`#2c2c2c` 实线，线宽 2.0

### 输出

| 文件名 | 内容 |
|--------|------|
| `trajectory_location1.png` | location1 全轨迹 + 车道线 |
| `trajectory_location2.png` | location2 全轨迹 + 车道线 |
| `trajectory_location3.png` | location3 全轨迹 + 车道线 |
| `trajectory_location4.png` | location4 全轨迹 + 车道线 |
| `trajectory_location5.png` | location5 全轨迹 + 车道线 |

- 上行轨迹：Reds 色阶（浅红→深红）
- 下行轨迹：Blues 色阶（浅蓝→深蓝）
- 图幅：20×10 inch，150 dpi

### 输出文件夹
`plot_trajectories_output/`

---

## 4. `plot_velocity_distribution.py` — 速度分布分析

### 功能
绘制变道车辆和跟驰车辆的速度分布直方图，包含正态拟合 / GMM 双峰拟合，标注 V85 和参考速度。

### 输入
- `E:/0little/location{1-5}/traffic_left_change.csv`（变道左）
- `E:/0little/location{1-5}/traffic_right_change.csv`（变道右）
- `E:/0little/location{1-5}/traffic_following_change.csv`（跟驰）

### 关键参数

**各场景 X 轴范围（跟驰）：**
| 场景 | 范围 |
|------|------|
| location2 | 40~150 km/h |
| location3 | 40~150 km/h |
| location4 | 40~150 km/h |
| location5 | 15~120 km/h |

**安全指标配置 (METRICS_FOLLOW)：**

| 指标 | 标签 | X 轴范围 |
|------|------|---------|
| mTTC | mTTC (s) | 0~50 |
| B_mTTC | B_mTTC (s) | 0~50 |

### 输出

| 文件名 | 内容 |
|--------|------|
| `velocity_distribution.png` | 10 个子图（5 loc × 左右变道）速度分布 |
| `mTTC_distribution.png` | mTTC 分布直方图 |
| `F_ETTC_distribution.png` | F_ETTC 分布直方图 |
| `PET_distribution.png` | PET 分布直方图 |
| `OL_PET_distribution.png` | OL_PET 分布直方图 |
| `velocity_following_distribution_location1.png` | loc1 跟驰速度分布 |
| `velocity_following_distribution_location2.png` | loc2 跟驰速度分布（xlim 40-150） |
| `velocity_following_distribution_location3.png` | loc3 跟驰速度分布（xlim 40-150） |
| `velocity_following_distribution_location4.png` | loc4 跟驰速度分布（xlim 40-150） |
| `velocity_following_distribution_location5.png` | loc5 跟驰速度分布（xlim 15-120） |
| `mTTC_following_distribution.png` | 跟驰车辆 mTTC 分布（5 loc 子图） |
| `B_mTTC_following_distribution.png` | 跟驰车辆 B_mTTC 分布（5 loc 子图） |

### 输出文件夹
`plot_velocity_distribution_output/`

---

## 5. `risk_scoring_exp_analysis.py` — 风险评分分布分析

### 功能
对 exp 版风险评分（变道 + 跟驰）进行分布对比，输出分布直方图 + 箱线图（组合图）、CDF + 风险等级比例（组合图）。

### 输入
- `E:/0little/location{1-5}/traffic_left_change.csv`
- `E:/0little/location{1-5}/traffic_right_change.csv`
- `E:/0little/location{1-5}/traffic_following_change.csv`

### 关键参数

**风险等级阈值：**
| 场景 | mid | high |
|------|-----|------|
| `lane_change` | 0.40 | 0.60 |
| `following` | 0.20 | 0.35 |

**场景颜色：**
- 变道：`#e74c3c`
- 跟驰：`#3498db`

### 输出

| 文件名 | 内容 |
|--------|------|
| `risk_score_distribution_box.png` | 左：风险分分布直方图（含 score=0 小窗）+ 右：箱线图+统计标注 |
| `risk_score_cdf_levels.png` | 左：CDF 累积分布曲线 + 右：风险等级比例堆叠柱状图 |

### 输出文件夹
`risk_scoring_exp_analysis_output/`

---

## 6. `visualize_per_vehicle_all.py` — 逐车辆多维可视化

### 功能
对每辆车生成 3×2 布局的多面板可视化图，展示轨迹、速度曲线、安全指标时序、路面风险定位等。

### 输入
- `E:/0little/location{1-5}/traffic_left_change.csv`
- `E:/0little/location{1-5}/traffic_right_change.csv`
- `E:/0little/location{1-5}/lane_coeffs.csv`

### 可视化布局（3×2 = 6 面板）

| 位置 | 内容 |
|------|------|
| (0,0) | 变道轨迹 XY + 车道线 |
| (0,1) | Velocity / long_Vel 时序 + 变道标识 |
| (1,0) | mTTC / THW 时序 + 安全阈值线 |
| (1,1) | PET / OL_PET 时序（标注冲突点） |
| (2,0) | 路面风险视角（底盘视角热力图） |
| (2,1) | 关键指标汇总表 + 标签 |

### 输出

| 文件命名 | 说明 |
|----------|------|
| `{location}_{side}_{ID}.png` | 每辆车一张独立图 |

例如 `location1_left_123.png`，位置在 location1 左变道的 ID 123 号车。
共约 1,500+ 张图（每辆车一张）。

### 输出文件夹
`visualize_per_vehicle_all_output/`

---

## 7. `gen_roadmap.py` — 研究路线图

### 功能
生成论文研究框架的示意图，涵盖数据采集 → 指标设计 → 特征工程 → 模型训练 → 可视化分析 → SHAP 解释共 6 个阶段。

### 输入
无外部数据依赖（纯 matplotlib 绘制）。

### 输出

| 文件名 | 内容 |
|--------|------|
| `research_roadmap.png` | 研究技术路线图（17×22 inch，250 dpi） |

### 输出文件夹
`gen_roadmap_output/`

---

## 完整输出文件清单

```
data_statistics/
├── distributions_analysis_output/
│   ├── dist_TTC.png             ├── dist_mTTC.png
│   ├── dist_PET.png             ├── dist_OL_PET.png
│   ├── dist_Time_Headway.png    ├── dist_kde_comparison.png
│   ├── dist_boxplot.png         ├── dist_ol_pet_pie.png
│   ├── dist_ol_pet_bar.png      ├── dist_ETTC_hist.png
│   ├── dist_F_ETTC_hist.png     ├── dist_B_ETTC_hist.png
│   ├── dist_ettc_kde.png        ├── dist_RSD_hist.png
│   ├── dist_F_ERSD_hist.png     ├── dist_B_ERSD_hist.png
│   ├── dist_rsd_kde.png         ├── risk_by_location.png
│   ├── overall_risk_left_pie.png  ├── overall_risk_left_bar.png
│   ├── overall_risk_right_pie.png ├── overall_risk_right_bar.png
│   ├── risk_heatmap_location1.png ─── risk_heatmap_location5.png
│
├── feature_correlation_output/
│   ├── feature_correlation.png
│   └── feature_risk_correlation.png
│
├── plot_trajectories_output/
│   ├── trajectory_location1.png ─── trajectory_location5.png
│
├── plot_velocity_distribution_output/
│   ├── velocity_distribution.png
│   ├── mTTC_distribution.png    ├── F_ETTC_distribution.png
│   ├── PET_distribution.png     ├── OL_PET_distribution.png
│   ├── velocity_following_distribution_location1.png ─── loc5.png
│   ├── mTTC_following_distribution.png
│   └── B_mTTC_following_distribution.png
│
├── risk_scoring_exp_analysis_output/
│   ├── risk_score_distribution_box.png
│   └── risk_score_cdf_levels.png
│
├── visualize_per_vehicle_all_output/
│   └── {location}_{side}_{ID}.png  (约 1500+ 张)
│
└── gen_roadmap_output/
    └── research_roadmap.png
```
