# Material Inventory

> 构建素材清单 — paper-spine-build 阶段生成
> 2026-06-02

## 数据素材

| 素材 | 路径 | 状态 | 用途 |
|---|---|---|---|
| 5 location 变道轨迹 | `E:/0little/location{1-5}/traffic_{left,right}_change.csv` | ✅ 已处理 | 模型输入 |
| CQSkyEyeX 数据集描述 | `CQSkyEyeX轨迹数据集描述.docx` | ✅ 可用 | 数据节描述 |
| CQSkyEyeX 场景记录 | `CQSkyEyeX_recordlog.xlsx` | ✅ 可用 | 场景信息表 |
| CQSkyEyeX 车辆索引 | `CQSkyEyeX_index.xlsx` | ✅ 可用 | 车辆统计 |

## 代码素材

| 模块 | 路径 | 功能 |
|---|---|---|
| 安全评分 | `traffic_full/safety_scoring.py` | 多指标风险标签生成 |
| 特征工程 | `risk_modeling/risk_modeling_utils.py` | 52维聚合特征 |
| 建模脚本 | `risk_modeling/risk_modeling_all.py` | 4模型训练评估 |

## 分析结果

| 输出 | 路径 |
|---|---|
| 跨 Location 模型对比 | `analysis/09_all_model_comparison.png` |
| 混淆矩阵 | `analysis/10_all_confusion_matrices.png` |
| XGBoost SHAP bar | `analysis/shap_all_XGBoost_importance_bar.png` |
| XGBoost SHAP beeswarm | `analysis/shap_all_XGBoost_beeswarm.png` |
| RF SHAP bar | `analysis/shap_all_RandomForest_importance_bar.png` |
| RF SHAP beeswarm | `analysis/shap_all_RandomForest_beeswarm.png` |
| 高风险热力图 | `analysis/risk_heatmap.png` |
| 各场景风险分布 | `analysis/risk_by_location.png` |
