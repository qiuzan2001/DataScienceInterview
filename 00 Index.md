---
title: 00 Index · 面试准备总入口
aliases: [Index, 总入口, 首页]
tags: [meta, index, moc]
status: 完成
updated: 2026-09-16
---

# 🎯 面试准备总入口

> **面试侧重：ML + 统计，核心是 GLM 与 Tree-Based。**
> 🗣️ 英文面试：术语见 [[06. 中英术语对照表]]；每题笔记末尾有「英文表达」块。
> 优先级以 **[[优先级矩阵]]** 为唯一事实来源；真题见 [[99.1 题库总览]]。

---

## 0. 先看这三件事（10 分钟）

1. **考什么**：[[01. 面试画像与考点分布]] —— 岗位技能栈、面试流程、高频主题排行
2. **怎么准备**：[[02. 三条学习路径]] —— 1 天冲刺 / 1 周标准 / 3 周系统
3. **怎么答**：[[03. 答题模板与追问应对]] —— 30 秒结构 + 追问链拆解 + 错误答法

---

## 1. 优先级图例

| 标记 | 含义 | 准备策略 |
|---|---|---|
| ⭐⭐⭐ | **必背**：面试必问，答不出直接减分 | 能默写公式 + 讲清直觉 |
| ⭐⭐ | **高频**：常见追问 / 区分度题 | 理解 + 能举例 |
| ⭐ | **了解**：问到能聊两句 | 记住结论即可；**1 天冲刺可跳过** |
| ⚠️ | **陷阱**：常见错误答法 | 特别记「不要怎么说」 |

> 判定标准与逐篇清单见 **[[优先级矩阵]]**（唯一事实来源）。

---

## 2. 核心知识地图（按面试权重排序）

### 🔴 第一梯队：必考（占面试 60%+）

| 主题 | 笔记 | 要点 |
|---|---|---|
| **统计基础** ⭐⭐⭐ | [[9. 统计基础总览]] | 分布、CLT、MLE、**p 值**、贝叶斯 |
| ↳ 假设检验与 p 值 | [[9.4 假设检验与 p 值]] | p 值误读是头号陷阱题 |
| ↳ 估计与 MLE | [[9.3 估计：点估计、MLE 与区间估计]] | 一切 GLM 的根基 |
| ↳ 抽样分布与 CLT | [[9.2 抽样分布与中心极限定理]] | 标准误 vs 标准差 |
| **逻辑回归 / GLM** ⭐⭐⭐ | [[1. Logistic Regression & GLMs]] | logit link、MLE、odds ratio、分离问题 |
| ↳ 指数族与 link | [[1.2 Generalized Linear Models (GLMs)]] | `μ=b'(θ)`、`Var=φb''(θ)`、canonical link |
| ↳ 系数解释 | [[1.3 Introducing Logistic Regression]] | `exp(β)`=odds ratio |
| ↳ 假设清单 | [[1.7 Assumptions]] | Top 5 必问 |
| ↳ 不平衡数据 | [[1.5 Dealing with Unbalanced Samples]] | SMOTE、病例对照只改截距 |
| ↳ 分离与惩罚 | [[1.6 Estimation Issues & Separation]] | 完全分离 → **Firth**（非 Ridge） |
| ↳ 正则化 | [[1.6.3 Regularization]] | Ridge / Lasso / ElasticNet |
| **决策树** ⭐⭐⭐ | [[7.2.2 Decision Tree]] | 熵/Gini 严格凹 vs 误分类率分段线性 |
| ↳ CART | [[7.5 CART Algorithm]] | Gini、二叉、surrogate、CCP |
| ↳ 剪枝 | [[7.2.2.4. Pruning Decision Trees]] | 预剪枝 vs 后剪枝、CCP、1-SE |
| **随机森林** ⭐⭐⭐ | [[7. Random Forest]] | bagging + 随机子空间、OOB、`m=√p` |
| ↳ 建树流程 | [[7.2. Building the Forest]] | ΔI 杂质下降、bootstrap 63.2% |
| ↳ 特征重要性 | [[7.4. Quantifying Feature Importance]] | MDI 偏差 vs permutation（要在 OOB 上算） |
| **Boosting** ⭐⭐⭐ | [[8.4. Boosting]] | 前向分步、加性模型 |
| ↳ AdaBoost | [[8.4.2.1 AdaBoost (Adaptive Boosting)]] | `α=½ln((1−ε)/ε)`、权重更新 |
| ↳ GBM | [[8.4.2.2 Gradient Boosting]] | 伪残差 = 负梯度、shrinkage |
| ↳ 公式推导 | [[8.4.2.1.4 AdaBoost Formulas]] | 指数损失 → AdaBoost |
| **Bagging vs Boosting** ⭐⭐⭐ | [[8.6 Bagging VS Boosting]] | 偏差/方差、并行性、调参 |
| **集成总览** ⭐⭐⭐ | [[8. Ensemble Learning]] | 三大家族框架、stacking OOF |
| **评估指标** ⭐⭐⭐ | [[6.3 Performance Metrics]] | AUC/KS/Lift、混淆矩阵 |
| **数据划分与 CV** ⭐⭐⭐ | [[6.1 Data Preparation & Validation]] | 分层、防泄漏（每场必问） |
| **偏差方差** ⭐⭐⭐ | [[6.2 Model Diagnosis_Bias-Variance Tradeoff]] | bias²+variance+irreducible |
| **多重共线性** ⭐⭐⭐ | [[3. Multicollinearity]]、[[3.4.2 Variance Inflation Factor (VIF)]] | VIF；**SE 只放大 √VIF** |
| **缺失值** ⭐⭐⭐ | [[4. Missing Data]] | MCAR/MAR/MNAR、MICE + Rubin 合并 |
| **变量变换** ⭐⭐⭐ | [[2. Transformations]] | 分箱、Box-Cox、样条 |
| **WOE / IV** ⭐⭐⭐ | [[2.3.1 WOE & IV]] | `ln(%Goods/%Bads)`、正 WOE = 低风险 |
| **变量筛选** ⭐⭐⭐ | [[5.3 Multivariate Selection]] | RFE/Stepwise、筛选须在 CV 内 |


### 🟡 第二梯队：高频（占 25%）

| 主题 | 笔记 | 要点 |
|---|---|---|
| OLS 与假设 | [[1.1 Ordinary Least Squares (OLS)]] | 正规方程、A1–A7、BLUE |
| MLE vs 最小二乘 | [[1.4 Model Fitting]] | 梯度尺度（不是「有偏」） |
| PCA | [[3.5.4 PCA]] | 特征值=方差、先中心化 |
| VARCLUS | [[3.5.2 SAS VARCLUS]] | 变量聚类、λ₂ 判据 |
| 降维总览 | [[5. Dimension Reduction]] | filter/wrapper/embedded |
| 单变量筛选 | [[5.2 Univariate Selection]] | χ²/ANOVA/IV/AUC |
| 诊断可视化 | [[6.4 Diagnostic & Visualization Tools]] | ROC、残差图、Lift/Gain |
| 树算法细节 | [[7.2.2.3.1 ID3 Algorithm]] | 信息增益、熵三视角 |
| C4.5 | [[7.2.3.3 C4.5 Algorithm]] | 增益率、缺失值、误差剪枝 |
| 相关≠因果 | [[9.5 相关、协方差与相关不等于因果]] | 混淆/中介/对撞、辛普森悖论 |
| 贝叶斯与基础率 | [[9.6 贝叶斯基础与基础率谬误]] | 16.7% 经典题 |
| 多重比较与选择偏差 | [[9.7 多重比较与选择偏差]] | 与特征筛选强相关 |
| 统计题精练 | [[9.8 统计面试题精练]] | 现场手算 |

### 🟢 第三梯队：了解（占 10–15%）

| 主题 | 笔记 |
|---|---|
| 本章 MOC | [[6. Model Assessment]] |
| 概率分布 | [[9.1 概率分布与随机变量]] |
| ID3 实现与手算 | [[7.2.2.3.1.4 ID3 Complete Code]]、[[7.2.2.3.1.9 ID3 Build Example]] |

---

## 3. 保险 / 风控专属（差异化加分）

> 通用 ML 候选人答不好的地方，正是拉开差距的地方。

**主线：[[10. 保险风控考点总览]]**

| 主题 | 笔记 | 要点 |
|---|---|---|
| **Tweedie GLM** ⭐⭐⭐ | [[10.8 Tweedie GLM 专章]] | 纯保费建模主力；`Var=φμ^p`、`1<p<2` |
| **项目穿讲** ⭐⭐⭐ | [[10.9 项目实战：法语车险纯保费定价]] | 端到端：678,013 份保单、两阶段 vs Tweedie |
| 频率-严重度与纯保费 | [[10.1 频率-严重度与纯保费]] | 纯保费 = 频率 × 严重度 |
| 计数模型与 offset | [[10.2 计数模型与 offset]] | `offset=log(exposure)` |
| 评分卡与 WOE 落地 | [[10.4 评分卡与 WOE 落地]] | `score = A − B·ln(odds)` |
| 模型评估与监控 | [[10.5 模型评估与监控]] | KS / PSI / OOT |
| 严重度模型与 Gamma | [[10.3 严重度模型与 Gamma]] | 为什么不用正态 |
| 采样偏差与拒绝推断 | [[10.6 采样偏差与拒绝推断]] | 差异化加分 |
| SAS 实操速查 | [[10.7 SAS 实操速查]] | PROC 语句 |

---

## 4. 速查与自测

- [[99.1 题库总览]] —— 全部题目（含答案 / 追问链 / 错误答法）
- [[06. 中英术语对照表]] —— 中英术语 + 面试常见说法
- [[04. 公式速查卡]] —— 一页纸背完所有公式
- [[05. 高频追问 TOP 30]] —— 面试官最爱追问的问题（附英文一句话答案）
- [[优先级矩阵]] —— 全库优先级的唯一事实来源
- [[事实核验报告]] —— 全部内容的权威来源核验记录（PSL / Loss Data Analytics）

---

## 5. 命名与编号约定

- `0x` 开头 = 冲刺层（精简、面试导向）
- `1–8` 章 = 系统知识（原有，保留）
- `09` 章 = **统计学基础**
- `10` 章 = 保险风控实务（含 Tweedie 专章与项目穿讲）
- `99` = 题库
- `_meta` = 规范、优先级矩阵、核验、脚本（不参与学习）

---

## 6. 维护

```bash
bash _meta/tools/vault_check.sh     # 体检：断链/空文件/AI残留/编号
```
