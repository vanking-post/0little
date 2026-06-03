# Source Inventory

> 素材盘点 — 论文构建的材料基础
> 2026-06-02

## 1. 数据源

### CQSkyEyeX 轨迹数据集
| 来源 | 描述 |
|---|---|
| 官网 | http://www.cqskyeyex.com/CQSkyEyeX |
| 数据集描述 | `E:\0little\CQSkyEyeX轨迹数据集描述.docx` |
| 场景记录 | `E:\0little\CQSkyEyeX_recordlog.xlsx` |
| 车辆索引 | `E:\0little\CQSkyEyeX_index.xlsx` |
| 提取方法 | YOLOX+DeepSORT，论文见 `E:\0little\paper\基于改进YOLOX的无人机航拍图像密集小目标车辆检测.pdf` |

### 5 个场景

| Location | 场景编号 | 道路 | 类型 | 车道 | 限速(km/h) | 长度(m) | 拍摄次数 |
|---|---|---|---|---|---|---|---|
| location1 | 1-1, 1-2 | 重庆绕城高速 | 路基段 | 3 | 120 | ~420 | 2 |
| location2 | 2-1~2-4 | 重庆绕城高速 | 路基段 | 3 | 120 | ~420 | 4 |
| location3 | 3-1~3-7 | 渝蓉高速 | 桥梁段 | 3 | 120 | ~420 | 7 |
| location4 | 4-1~4-9 | 渝蓉高速 | 桥梁段 | 3 | 120 | ~420 | 9 |
| location5 | 5 | 兰海高速/内环 | 分合流区 | 3 | 100 | ~350 | 1 |

## 2. 轨迹处理管线

| 步骤 | 文件 | 说明 |
|---|---|---|
| 原始轨迹 | `location{1-5}/traffic_{left,right}_change.csv` | 每帧一行 |
| 安全评分 | `traffic_full/safety_scoring.py` | mTTC/THW/PET/ETTC → 三级风险标签 |
| 特征工程 | `risk_modeling/risk_modeling_utils.py:load_and_engineer()` | 时序→聚合特征(52维) |
| 模型训练 | `risk_modeling/risk_modeling_all.py` | 4模型, 2评估策略 |

## 3. 分析输出

| 文件 | 内容 |
|---|---|
| `analysis/09_all_model_comparison.png` | 跨 Location 模型对比柱状图 |
| `analysis/10_all_confusion_matrices.png` | 混淆矩阵 |
| `analysis/shap_all_XGBoost_importance_bar.png` | XGBoost SHAP bar |
| `analysis/shap_all_XGBoost_beeswarm.png` | XGBoost SHAP beeswarm |
| `analysis/shap_all_RandomForest_importance_bar.png` | RF SHAP bar |
| `analysis/shap_all_RandomForest_beeswarm.png` | RF SHAP beeswarm |
| `analysis/risk_heatmap.png` | 5场景高风险热力图 |
| `analysis/01_risk_distribution.png` | 风险分布 |
| `analysis/03_location_comparison.png` | 场景对比 |
| `analysis/risk_by_location.png` | 各场景高中低风险对比 |

## 4. 参考文献

8篇: 见 `source_map.md`
