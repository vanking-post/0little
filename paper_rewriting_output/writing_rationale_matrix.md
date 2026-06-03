# Writing Rationale Matrix

> 写作执行计划 — 动因驱动的逐段设计
> 动机: A1 跨路段泛化 + A3 SHAP 可解释性
> 生成日期: 2026-06-02

---

## 整体框架正当性

**整篇论文的框架选择**: 采用标准 IMRaD + 独立 Related Work 结构（6节）。整篇论文以"跨路段泛化评估"为主叙事线：Introduction 建立"ML模型缺乏跨路段验证"的 gap → Related Work 三个子节从 SSM→ML→跨路段逐层聚焦 → Method 明确对比评估策略（5-fold cross-location CV + 随机 80/20） → Results 回答两个研究问题（模型泛化差异 + 特征普适性）→ Discussion 解释泛化成因并分离普适与路段特异性因子 → Conclusion 收束全篇。这个结构服务于确认动机 A1+A3：A1 要求跨路段对比的实验设计嵌在 Method 和 Results 中；A3 要求的可解释性嵌在 Results 4.4 中作为独立的分析单元。从 SOTA 示例学习来看，选择独立 Related Work 节而非融入引言（对比 C02 的做法），更符合 SCI 三区/四区期刊审稿人对系统性文献梳理的期待。用户证据锚点包括 5 个 location 的完整实验输出、跨 location CV 的评估代码和 SHAP 分析结果，全部嵌入 Results 节对应的子标题下。最终验查标准：每个编号贡献在 Results 中必须有对应的定量结果支撑，Discussion 中的泛化论断必须与跨 location 结果的标准差一致。理由：

1. **目标场景（SCI 三区/四区英文期刊）**：从参考文献看，该档期期刊偏好标准 IMRaD（E3: JTSS; C01: IEEE T-ITS 属更高档但框架可借鉴）。对比 E2 (AAP) 无独立 Related Work 的做法，设独立 Related Work 更符合我们所在档位的期望（审稿人期待看到系统的文献梳理）。

2. **动机驱动**：确认动机 A1 的叙事链要求 Introduction 建立"现有 ML 风险模型缺乏跨路段验证"的 gap → Method 明确对比评估策略 → Results 回答泛化问题 → Discussion 讨论成因。标准 IMRaD 最适合这条叙事线。

3. **SOTA 示例学习**：E1 (F-F Diagram) 和 E3 (IRE model) 均采用独立 Related Work + IMRaD 结构，在 Introduction 末尾注明 numbered contributions + paper organization。

4. **用户证据锚点**：用户已有 5 个 location 的完整分析管线、跨 location CV 评估、SHAP 输出，所有结果天然嵌入 Results 节。

5. **最终验查标准**：每个 claim 必须有对应的表格/图表/统计量支撑，Discussion 中的泛化论断必须与跨 location 结果的方差一致。

---

## 逐段矩阵

| Row ID | Manuscript Unit | Current/Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|---|
| FWK-00 | Whole-Work Framework | IMRaD+Related Work, 6-section structure: Intro (ML缺乏跨路段验证gap) → RW (SSM→ML→跨路段聚焦) → Method (cross-location CV + random 80/20) → Results (泛化差异+SHAP) → Discussion (普适vs特异因子) → Conclusion. 服务于A1+A3动机。 | A1+A3共同决定 | E1/E3使用独立RW+IMRaD; C38/C72建立跨路段gap | SCI期检查待独立RW节、编号贡献 | 5-location管线、CV代码、SHAP输出 | 标准IMRaD+RW, 对比E2无独立RW的做法 | 每项贡献需对应Results量化结果; Discussion泛化论断需与跨location标准差一致 |
| M01 | Title | 精炼概括论文核心贡献：跨路段 + ML + SHAP | A1 是主干，A3 是方法亮点 | E1–E4 标题均含方法关键词（F-F Diagram, Risk Map, IRE） | SCI 期刊标题 15-20 词，含方法+对象 | 待拟定最终版本 | 提供 2-3 个备选标题让用户选择 | 标题是否准确反映 A1+A3 内容 |
| M02 | Abstract | 四句：问题 → 方法 → 关键结果 → 意义 | A1 跨路段泛化问题需在第一句点明 | E1, E2, E3 的 Abstract 均包含数字结果以增强可信度 | SCI 期刊 150-250 词 | 跨 location F1 均值±标准差 | 定稿时填入实际性能数字 | Abstract 是否包含量化结果 |
| M03 | Keywords (6个) | Lane change risk, Machine learning, Trajectory data, Cross-location validation, SHAP, Surrogate safety measure | 与 A1 方向严格对齐 | E1: Lane change risk / driving risk field / vehicle trajectory | 6-8 个关键词 | CQSkyEyeX 数据集 | 确定关键词列表 | 是否覆盖全部方法要素 |
| M04 | Intro — 段1 (Hook) | 全球事故统计 → 换道安全重要性 → SSM 概念引出 | 建立研究必要性 | E1/E3 用 WHO/NHTSA 数据开篇；E4 用"Risky LC behavior"直接切入 | 3-5 句，数据来源可靠 | WHO 1.35M 死亡；NHTSA 451K LC事故 | 参考 E1 的 Intro 开篇 | 引用的统计数据是否有可靠出处 |
| M05 | Intro — 段2 (SSM局限) | 传统 TTC/PET/SDI 局限 → 一维/离散/仅纵向 → ML 方法优势 | 引出"为什么需要 ML 方法" | E1 的 Related Work 对 SSM 分类批评；C08 的"行为确定性+空间离散性" | 4-6 句，至少 4 个引用 | C04（TSRE gap识别方法）、C22（TET/TIT）、C26（SSM综述） | 从文献中提取 2-3 个具体局限 | 是否明确指出现有 SSM 的至少 2 个局限 |
| M06 | Intro — 段3 (ML局限) | ML 模型已广泛应用 → 但多集中于单一数据集验证 | 核心 gap：跨路段泛化未充分研究 | C38（跨路段迁移性下降）、C40（跨场景不可迁移）、C72（跨数据集 F1 下降 18-25%） | 4-5 句，聚焦 gap | C49（跨域评价精度下降 15-25%） | 使用文献证据支撑"单一数据集验证不足"的判断 | gap 表述是否具体而非笼统 |
| M07 | Intro — 段4 (本文方案) | 本研究使用 CQSkyEyeX 5 路段 → 系统评估跨路段泛化 | A1 核心：解决 gap 的方案 | E2 的 Intro 末尾用 numbered contributions | SCI 论文标配：本文方案段落 | CQSkyEyeX 数据集描述、5 路段信息 | 写 3-5 句描述研究设计 | 是否清晰说明"做了什么" |
| M08 | Intro — 段5 (贡献) | 3 条编号贡献声明 + 论文结构预告 | 三条贡献对应 A1 三要素 | E2/E3 的 numbered contributions 模式 | 编号贡献是 SCI 常见范式 | 跨 location 评估 + SHAP 可解释性 | 写 3 条编号贡献 + "remainder of paper" 结构预告 | 贡献是否具体且可验证 |
| M09 | Related Work — 2.1 SSM & LC Risk | 系统梳理 SSM 发展史：TTC→PET→SDI→综合指标→风险场 | 建立领域知识基础 | E1 的 Related Work 分 4 子节：A. LC Risk Indicator / B. Risk Label / C. LC Assessment / D. Safety Potential Field | 引用 8-12 篇，覆盖经典+近 3 年 | C20(C0-C22(C01), C10(LCRI), C26(综述), C04(TSRE) | 按时间/方法演进组织 | 是否覆盖关键里程碑论文 |
| M10 | Related Work — 2.2 ML for LC Risk | ML 方法：RF→XGBoost→LSTM→Attention → SHAP 可解释性 | 建立与本文方法的连接 | C05（RF+SSM）、C07（XGBoost+SHAP）、C15（DS-LSTM）、C17（CNN-BiLSTM-Attention） | 引用 6-10 篇 | 用户模型（XGBoost/RF/MLP/LSTM）对标 C05/C07 | 以方法演进+性能对比组织 | 是否引用用户方法的直接对标文献 |
| M11 | Related Work — 2.3 Cross-Location | 跨路段/跨数据集泛化的现有探索 | 核心 gap 的子节 | C38（跨路段迁移性）、C40（跨场景不可迁移）、C52（域自适应）、C49（跨域评价） | 引用 4-6 篇，说明该方向研究有限 | C72（跨数据集 F1 下降 18-25%） | 直接论证：现有研究有限→本研究填补空白 | 是否清晰界定本研究与其他跨域研究的区别 |
| M12 | Method — 3.1 CQSkyEyeX | 数据集描述：采集方式→5 路段特征→精度→参数 | 让读者理解数据基础 | E1/E3 的 Data 节结构一致：数据集名→采集方式→研究区域→样本提取 | 6-8 句，表格优先 | XLSX 记录表描述、DOCX 参数说明、官网信息 | 用 Table I 列出 5 路段对比（路段类型/长度/车辆数/时段） | Table I 是否包含关键信息 |
| M13 | Method — 3.2 Risk Labeling | 多指标加权评分管线：mTTC/THW/PET/ETTC → weighted score → 三级标签 | A1 的方法基础 | C02（MTTC 概率变换）、C06（RSL+REL 动静结合）、E3（概率×严重度框架） | 公式+文字解释，3-6 句 | safety_scoring.py 的 overall_risk 函数 | 写公式+阈值选取依据 | 公式是否准确，阈值是否有依据 |
| M14 | Method — 3.3 Feature Engineering | 时序→聚合特征：每车 mean/std/min/max of 13 指标 | A1 的特征空间定义 | C05（17 维特征+RF-RFE 筛选）、C07（32 维特征） | 1-2 段，表格列出特征 | load_and_engineer 函数的 TS_FEATURES | Table II 或文中说明 | 特征选取是否合理，是否提及异常值处理 |
| M15 | Method — 3.4 Models | XGBoost/RF/MLP/LSTM 配置和超参数 | A1 的模型基础 | C07（XGBoost vs LightGBM vs GBDT vs RF 对比）、C05（RF vs SVM vs LR） | 表格列出超参数，2-3 句解释选择理由 | risk_modeling_utils.py 中各 train 函数 | 为每个模型写 3-5 句描述 | 超参数是否可复现 |
| M16 | Method — 3.5 Evaluation & SHAP | 5-fold cross-location + random 80/20 + SHAP 分析 | A1 的评估策略 + A3 的支持 | E1（RMSE 损失评估）、C03（ANOVA 对比） | 2-3 段 | evaluate_cross_location 和 evaluate_random_split 函数 | 描述两种评估策略的设计意图 | 评估策略能否回答研究问题 1 和 2 |
| M17 | Results — 4.1 Data Overview | 每路段样本量、风险分布统计 | A1 的数据基础 | C02（604 组）、C03（1536 LC）、E1（1122 LC） | 2-3 句 + 图/表 | 各 location 样本量、风险标签分布 | 从 df 统计生成 Table/Fig | 数据统计是否符合论文要求 |
| M18 | Results — 4.2 Cross-Location | 各 fold 性能对比表+柱状图 | A1 核心结果：回答研究问题 1 | C07 表 4（准确率/精确率/召回率/F1）、E1 表 II&III（RMSE 对比） | Table III + Fig 4 | 用户已有 09_all_model_comparison.png | 写出结果段文字框架，由用户后期填入实际数值 | 表格是否涵盖 Acc/F1/MacroF1 |
| M19 | Results — 4.3 Random Split | 随机划分对照 | A1 的基线对照 | C05（训练集 70% 测试集 30%） | 1-2 段 | 用户已有 09_all_model_comparison.png + 10_all_confusion_matrices.png | 对比跨 location 与随机划分的差距 | 差距是否定量呈现 |
| M20 | Results — 4.4 SHAP | XGBoost + RF 的 SHAP beeswarm + bar | A3 核心：回答研究问题 2 | C07（SHAP 分析图）、C36（SHAP-based LC risk）、C64（XGBoost+SHAP） | Fig 6-7 + 文字分析 2-3 段 | shap_all_XGBoost_beeswarm.png + importance_bar.png | 写 SHAP 解读文字：前 5 特征+物理含义 | 特征重要性是否有物理可解释性 |
| M21 | Discussion — 5.1 泛化讨论 | 跨 location 结果解读：哪些模型/路段泛化好 | A1 的结果解读 | C38（模型跨路段下降）、C72（跨数据集下降 18-25%） | 3-5 句 | 与文献对比本研究的跨路段 F1 变化 | 将本研究的泛化结果与文献对比 | 泛化结论是否有充分证据 |
| M22 | Discussion — 5.2 普适vs特异因子 | SHAP 跨路段特征排名对比 | A3 的深入讨论 | C60（普适 vs 路段特异性风险因子）、C57（不同几何条件下的特征重要性变化） | 3-5 句 | 跨 location SHAP 图的对比分析 | 列出稳定性高 vs 变化大的特征 | 区分普适因子和特异因子是否有数据支撑 |
| M23 | Discussion — 5.3 实践意义 | 对 ADAS/换道预警系统的启示 | A1+A3 的综合应用价值 | E1（F-F 图可集成到 ADAS）、E2（风险地图预警） | 3-5 句 | 需结合实际场景 | 写 3-4 条实践建议 | 建议是否具体而非泛泛而谈 |
| M24 | Discussion — 5.4 局限 | 5 个路段同区域、简化打分、无因果 | 诚实展示边界 | E1（局限+未来工作）、E3（局限段落） | SCI 论文必配段落 | 具体局限：样本量、标签噪声、数据区域 | 列举 3 条具体局限 | 局限是否实质性而非客套话 |
| M25 | Conclusion | 3-5 句：总结→关键发现→未来工作 | 全篇收束 | E1/E2/E3 的结论均包含总结+发现+未来 | 1-2 段 | 对应研究问题 1-2 的回答 | 浓缩 Intro 的贡献+Results 的关键发现 | 是否回答了 Introduction 提出的研究问题 |
| M26 | Abstract (终版) | 终稿时填入实际数字 | — | — | — | 跨 location 均值±标准差 | 等 Results 定稿后更新 | 数字是否从 Results 准确提取 |

---

## 前后回响设计

| 前向（Introduction 建立预期） | 后向（Discussion/Conclusion 回应） |
|---|---|
| "This study fills the gap by evaluating ML models across multiple expressway segments..." (段3) | "The cross-location evaluation confirms that model performance varies substantially across segments..." (5.1) |
| "SHAP analysis reveals both universal and location-specific risk factors..." (段4) | "The top 3 SHAP features are consistent across locations, while features 7-10 show significant variation..." (5.2) |
| 三条编号贡献 | 结论中对应三条贡献的验证 |
