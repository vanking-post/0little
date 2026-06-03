# Claim Register

> 声明注册 — 所有查声明与证据对应
> 2026-06-02

| Claim ID | Claim | Section | Evidence ID | Verification |
|---|---|---|---|---|
| C01 | CQSkyEyeX dataset covers 5 expressway scenes with diverse geometries | Data | E01, source_inventory.md | ✅ |
| C02 | Risk labels are imbalanced: 323 high / 608 mid / 643 low | Data | E02 | ✅ |
| C03 | XGBoost achieves best cross-location generalization (F1=0.759±0.069) | Results 4.2 | E03 | ✅ |
| C04 | RF shows competitive but slightly lower performance (F1=0.742±0.067) | Results 4.2 | E04 | ✅ |
| C05 | MLP generalizes poorly across locations (F1=0.586±0.086) | Results 4.2 | E05 | ✅ |
| C06 | LSTM also underperforms tree-based models cross-location (F1=0.598) | Results 4.2 | E06 | ✅ |
| C07 | Location 5 (merge/diverge) is hardest to generalize to (F1=0.631) | Results 4.2, Discussion | E07 | ✅ |
| C08 | Location 1 (roadbed) is easiest (F1=0.821) | Results 4.2, Discussion | E08 | ✅ |
| C09 | Tree-based models (XGBoost, RF) show smaller random vs cross-location gap | Discussion 5.1 | E09, E10 | ✅ |
| C10 | Time Headway is the most universal risk factor across locations | Results 4.4, Discussion 5.2 | E11 | ✅ |
| C11 | Following distance and rear distance are stable secondary factors | Results 4.4 | E12, E13 | ✅ |
| C12 | High-risk lane changes cluster at specific spatial locations | Results 4.1 | E14 | ✅ |
| C13 | Model performance varies substantially by road geometry type | Discussion 5.1 | E07 vs E08 | ✅ |
