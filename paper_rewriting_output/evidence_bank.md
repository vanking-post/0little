# Evidence Bank

> 证据清单 — 每一条 claim 的证据支撑
> 2026-06-02

| ID | Evidence | Type | Source | Supports Claim |
|---|---|---|---|---|
| E01 | 1,574 辆变道样本, 52 维特征 | Data | `risk_modeling_all.py` 运行输出 | 数据集规模 |
| E02 | 高风险 323, 中风险 608, 低风险 643 | Stat | 标签分布 | 标签平衡性描述 |
| E03 | XGBoost Acc=0.761±0.070, F1=0.759±0.069 | Model | Cross-location CV | XGBoost 泛化最优 |
| E04 | RF Acc=0.745±0.069, F1=0.742±0.067 | Model | Cross-location CV | RF 稳定第二 |
| E05 | MLP Acc=0.591±0.083, F1=0.586±0.086 | Model | Cross-location CV | 神经网络泛化差 |
| E06 | LSTM Acc=0.612±0.062, F1=0.598±0.063 | Model | Cross-location CV | LSTM 优于 MLP不足XGBoost |
| E07 | Loc5(分合流区) XGBoost F1=0.631 (最差) | Fold | Cross-location CV Fold5 | 路段几何影响泛化 |
| E08 | Loc1(绕城基段) XGBoost F1=0.821 (最佳) | Fold | Cross-location CV Fold1 | 基段可迁移性最好 |
| E09 | XGBoost 随机80/20: F1=0.773; 跨location: F1=0.759, 差距1.4% | Comparison | 两种策略对比 | XGBoost 泛化稳定 |
| E10 | MLP 随机: F1=0.635; 跨location: F1=0.586, 差距4.9% | Comparison | 两种策略对比 | 神经网络过拟合路段特征 |
| E11 | SHAP Top-1: Time_Headway_min | Explainability | SHAP XGBoost+RF | 车头时距是普适因子 |
| E12 | SHAP Top-2: B_Dist_mean | Explainability | SHAP XGBoost | 后方距离是重要风险因子 |
| E13 | SHAP Top-3: Following_dist_min | Explainability | SHAP XGBoost+RF | 跟车距离在两类模型中稳定 |
| E14 | 热力图显示高风险集中在特定空间位置 | Visualization | risk_heatmap.png | 风险空间异质性 |
