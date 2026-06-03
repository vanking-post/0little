# Exemplar Learning Dossier — 写作范例学习

> 从 8 篇参考文献中提取可迁移的写作策略
> 目标场景: SCI 三区/四区英文期刊

---

## 1. 全篇框架学习

### 范例 E1 (IEEE T-ITS) — 最完整的期刊框架

```
框架: IMRaD + 独立 Related Work
Introduction    →  引出 gap、贡献（隐含在叙述中）
Related Work    →  4 个子节，系统梳理
Methodology     →  Framework → Formulation → Algorithm → Clustering
Results & Disc. →  Data → Algorithm Comparison → Clustering → Confidence
Conclusion      →  总结 + 未来方向
```

**可迁移策略**：
- Related Work 独立成节更显深度
- Method 按逻辑链组织（概念 → 数学 → 算法 → 应用），而非按 IMRaD 模板
- Results 和 Discussion 合并，减少重复

### 范例 E2 (AAP) — 含编号贡献的简洁框架

```
框架: 无独立 Related Work（文献融入 Intro）
Introduction    →  背景 → 文献不足 → 3 条编号贡献 → 论文结构
Method          →  Risk Map 构建 → 预警方法
Results & Disc. →  仿真分析 → 对比讨论
Conclusion      →  总结 + 展望
```

**可迁移策略**：
- 编号贡献清晰有力
- 不设独立 Related Work 节可节省篇幅
- 适合贡献集中、方法单一的论文

### 范例 C4 (中国公路学报) — 理念驱动的框架

```
框架: 问题导向型
Introduction    →  两个核心局限（有图示意）→ 解决方案 → 论文结构
Method          →  概念 → 构建 → 求解 → 实现
Experiments     →  多场景验证
Conclusion      →  范式总结 + 实践意义
```

**可迁移策略**：
- 从"局限性"而非"文献罗列"入手，更有冲击力
- 用示意图说明方法学缺陷（如图1(a)(b)(c)）
- "范式转变"类表述提升学术贡献感知

---

## 2. Introduction 写作技巧

### 2.1 开篇句式模式

**"全球统计"模式**（E1, E2, E3）:
```
"Road traffic accidents are the leading cause of death worldwide..."
"Unsafe lane change behaviors have negative impacts on traffic safety..."
```

**"直接问题"模式**（E4, C1）:
```
"Lane change behavior is one of the most common maneuvers..."
"汇入行为对高速公路运行安全有重要影响..."
```

### 2.2 Gap 表述模式

| 文献 | Gap 表述方式 |
|---|---|
| E1 | "the classification boundaries for lane change risk levels are vague" |
| E3 | "two gaps remain to be addressed. First... Second..." |
| E4 | "Existing methods tend to ignore the interactive impacts..." |
| C4 | 两个局限命名：「行为确定性」「空间离散性」并配示意图 |
| C1 | "以往研究仍有不足之处：(1)...(2)..." |

### 2.3 贡献声明模式

- **编号列表**（E2，E3）："The contributions of this paper are threefold: (1)... (2)... (3)..."
- **隐含叙述**（E1，C4）：贡献散布在目的是论述中，不单独编号
- **推荐**：SCI 三区/四区更偏爱编号贡献，清晰直接

### 2.4 论文结构预告

几乎通用模式：
```
"The remainder of this paper is organized as follows. Section 2... 
Section 3... Section 4... Finally, Section 5 concludes the study..."
```

---

## 3. Abstract 写作模式

英文期刊典型的四句结构：

| 句子 | 功能 | E1 原文示例 |
|---|---|---|
| 1 | 问题 | "Unsafe lane change behaviors have negative impacts on traffic safety." |
| 2 | 方案 | "In this paper, we develop a two-dimensional indicator based on field theory..." |
| 3 | 方法/结果 | "Parameter calibration with highD trajectory data... Experimental results demonstrate..." |
| 4 | 意义/展望 | "In the future, the F-F diagram could be integrated into lane change advisory systems..." |

中文期刊倾向于结构化摘要（C2: 【目标】【方法】【数据】【结果】【结论】）。

---

## 4. 方法节写作技巧

### 4.1 框架图优先
8 篇中 7 篇方法节第一个元素是框架流程图（Figure 1 或 2）。建议：

- **第一个图**: 整个方法的框架流程
- **第二个图**: 场景/数据示意
- **后续图**: 逐步展开的模型细节

### 4.2 数学模型写作节奏
```
文字概念引入 → 类比（"Similar to electric field..."）→ 数学定义 → 公式 → 参数说明 → 物理解释
```

### 4.3 参数标定
几乎所有论文都包含参数标定/优化算法节（梯度下降、PSO、GA、Adam 等），这已成为该领域的标准做法。

---

## 5. 结果节呈现技巧

### 5.1 表格设计模式
- 多算法对比表（E1: Table II, III; C1: Table 4; C3: Table 3）
- 混淆矩阵（C1: Table 2）
- 特征重要性排序（C1: 图/表）
- 统计描述（C1: Table 3）

### 5.2 可视化模式
- 散点图（E1: F-F Diagram）
- 箱线图/提琴图
- 时间序列对比
- ROC 曲线（C3: 图 5）
- SHAP 值图（C3）

---

## 6. 讨论/结论节模式

英文论文结论结构：
```
1. 总结论文做了什么（1-2 句）
2. 关键发现（分条）
3. 实践意义/应用前景
4. 局限性 + 未来工作方向（必须！）
```

几乎所有 SCI 论文结论都包含明确的"Limitations and future work"段落。中文论文也有类似结构但更简短。

---

## 7. 可迁移的修辞技巧

| 技巧 | 来源 | 应用场景 |
|---|---|---|
| 物理类比（如电场、场论） | E1, C4 | 引入新概念 |
| 局限性命名（行为确定性、空间离散性） | C4 | 清晰定位 gap |
| 两阶段风险（概率×严重度） | E3 | 方法论框架 |
| 动静结合（静态冲突+动态风险） | C2 | 方法论创新 |
| 范式转变表述 | C4 | 提升贡献感知 |
| "两 gap" 编号 | E3 | Introduction 结构 |
