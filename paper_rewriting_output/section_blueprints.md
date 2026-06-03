# Section Blueprints

> 面向: 基于多路段自然驾驶轨迹数据的变道风险 ML 预测与跨路段泛化研究
> 动机: A1（跨路段泛化）+ A3（SHAP 可解释性）
> 目标期刊: SCI 三区/四区英文（如 J. Transportation Safety & Security, IET ITS, IEEE Access, PLOS ONE）

---

## 整体结构

```
Title + Abstract + Keywords
1. Introduction
2. Related Work
   2.1 Surrogate Safety Measures and Lane Change Risk
   2.2 Machine Learning for Lane Change Risk Prediction
   2.3 Cross-Location and Cross-Dataset Generalization
3. Methodology
   3.1 Data Description (CQSkyEyeX)
   3.2 Risk Labeling Framework
   3.3 Feature Engineering
   3.4 Machine Learning Models
   3.5 Evaluation Strategy (Cross-Location + SHAP)
4. Results
   4.1 Data Overview and Risk Distribution
   4.2 Cross-Location Model Performance
   4.3 Random Split Model Comparison
   4.4 SHAP Feature Importance Analysis
5. Discussion
   5.1 Generalization Across Locations
   5.2 Universal vs. Location-Specific Risk Factors
   5.3 Practical Implications
   5.4 Limitations
6. Conclusion
```

---

## Section Detail

### Title (待定建议)

> **"Cross-Location Lane Change Risk Prediction Using Machine Learning: A Multi-Site Naturalistic Trajectory Study with SHAP Interpretation"**

备用: "Transferability Assessment of Machine Learning-Based Lane Change Risk Models Across Expressway Segments"

### Abstract (150-250 word)

四句结构:
1. Lane change risk is critical for traffic safety, but most ML-based risk models are developed and validated on single datasets/locations.
2. This study evaluates four ML models (XGBoost, RF, MLP, LSTM) for lane change risk prediction across five expressway segments from the CQSkyEyeX dataset.
3. [Key results: best model, cross-location performance, SHAP findings]
4. Findings highlight the importance of location-aware model selection and identify universal risk factors for proactive safety management.

### Introduction (4-6段)

- **段1 (Hook)**: 全球事故数据 → 换道风险重要性 → SSM 发展
- **段2 (文献批评1)**: 传统 SSM (TTC/PET/SDI) 的局限 → ML 方法的兴起
- **段3 (文献批评2)**: 现有 ML 模型多在单一数据集验证 → 跨路段泛化问题未充分研究
- **段4/5 (本文方案)**: 使用 CQSkyEyeX 5 路段数据 → 系统评估跨路段泛化 → SHAP 可解释性
- **段6**: Contributions (numbered) + Paper organization

### Related Work (3子节)

**2.1 SSM and Lane Change Risk**
- 传统 SSM (TTC, PET, SDI)
- 综合指标 (LCRI, CPI, TSRE) [C04, C10]
- 风险场方法 [C01, C08]
- → 引出：SSM 已被广泛研究，但 ML 路线是趋势

**2.2 ML for Lane Change Risk**
- 传统 ML (RF, XGBoost, SVM) [C05, C07, C18]
- 深度学习 (LSTM, CNN, Attention) [C15, C17, C63]
- SHAP 可解释性 [C11, C12, C36, C64]
- → 引出：ML 效果好，但跨场景泛化验证不足

**2.3 Cross-Location and Cross-Dataset Generalization**
- 跨数据集研究 [C38, C40, C49, C72]
- Domain adaptation 尝试 [C52]
- → 引出：这是当前 gap，本研究贡献所在

### Methodology (6-10段)

**3.1 CQSkyEyeX Dataset**
- 5 locations description (桥基、桥梁、分流区)
- Collection method (UAV + YOLOX + DeepSORT)
- Accuracy (position < 10cm, velocity < 1.5 km/h, 30 Hz)
- 30 Hz → downsampling
- Vehicle classes and trajectory parameters

**3.2 Risk Labeling Framework** (safety_scoring.py)
- Multi-SSM weighted scoring (mTTC, THW core; PET, ETTC, OL_PET auxiliary)
- Speed correction factor (V85/V0)^B
- Three risk levels: high (0), mid (1), low (2)
- Threshold determination

**3.3 Feature Engineering** (load_and_engineer)
- Time-series trajectory data → per-vehicle aggregation (mean, std, min, max)
- Features: Velocity, long/lat Vel, long/lat Acc, Following dist, B/LB/RB/LF/RF distances, TTC, Time Headway

**3.4 Models**
- XGBoost: 500 trees, max_depth=5, lr=0.05, early stopping
- RF: 300 trees, max_depth=10, class_weight=balanced
- MLP: (64,32) hidden, ReLU, adaptive lr
- LSTM: (64→32) layers, Dropout 0.25, Adam 0.0005

**3.5 Evaluation**
- 5-fold cross-location validation (leave-one-location-out)
- Random 80/20 split for baseline
- Metrics: Accuracy, Weighted F1, Macro F1
- SHAP analysis for XGBoost and RF

### Results (4-6段)

**4.1 Data Overview**
- Table: per-location sample counts, risk distribution
- Risk distribution bar charts across locations [01_risk_distribution.png]

**4.2 Cross-Location Performance** (核心结果)
- Table: per-fold accuracy, F1, macro F1 for each model
- Bar chart: cross-location F1 comparison [09_all_model_comparison.png]
- Key finding: which model generalizes best, which location is hardest

**4.3 Random Split Baseline**
- Table: random 80/20 performance vs cross-location
- Confusion matrices [10_all_confusion_matrices.png]
- Key finding: gap between random split and cross-location reveals transferability challenge

**4.4 SHAP Analysis**
- XGBoost beeswarm [shap_all_XGBoost_beeswarm.png] + bar [shap_all_XGBoost_importance_bar.png]
- RF beeswarm + bar
- Identify top 5-10 universal features
- Location-specific feature ranking variations

### Discussion (3-4段)

**5.1 Generalization Across Locations**
- Compare results with literature
- Why certain models transfer better
- Location characteristics that affect transferability

**5.2 Universal vs. Location-Specific Factors**
- From SHAP: which features are consistently important
- Which features vary by location
- Implications for model design

**5.3 Practical Implications**
- For ADAS/lane change warning systems
- Need for location-aware calibration

**5.4 Limitations**
- Only 5 locations from same region
- Simplified risk labeling (threshold-based)
- No causal analysis

### Conclusion (2-3段)

- Summary of key findings
- Contributions restatement
- Future work (more locations, domain adaptation, online learning)

---

## Figures & Tables Plan

| # | Type | Content | Source File |
|---|---|---|---|
| Fig 1 | Framework | Overall methodology flowchart | New |
| Fig 2 | Map | 5 CQSkyEyeX locations with characteristics | New |
| Fig 3 | Bar | Risk distribution by location | 01_risk_distribution.png |
| Fig 4 | Bar | Cross-location model comparison | 09_all_model_comparison.png |
| Fig 5 | Matrix | Confusion matrices | 10_all_confusion_matrices.png |
| Fig 6 | Beeswarm | SHAP XGBoost | shap_all_XGBoost_beeswarm.png |
| Fig 7 | Bar | SHAP feature importance | shap_all_XGBoost_importance_bar.png |
| Table I | Table | Dataset description per location | New |
| Table II | Table | Model hyperparameters | — |
| Table III | Table | Cross-location performance (Acc, F1, Macro F1) | — |
| Table IV | Table | Random split performance | — |
| Table V | Table | Top features from SHAP | — |
