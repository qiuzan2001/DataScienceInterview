---
title: 00 Index · 面试准备总入口
aliases: [Index, 总入口, 首页]
tags: [meta, index, moc]
status: 完成
updated: 2026-09-16
---

# 🎯 面试准备总入口

> **面试侧重：ML + 统计，核心是 GLM 与 Tree-Based。**
> 全部内容按「优先级」分层，先看 ⭐⭐⭐，真题见 [[99.1 题库总览]]。

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
| ⭐⭐ | **加分**：区分候选人水平 | 理解 + 能举例 |
| ⭐ | **了解**：问到能聊两句 | 记住结论即可 |
| ⚠️ | **陷阱**：常见错误答法 | 特别记「不要怎么说」 |

---

## 2. 核心知识地图（按面试权重排序）

### 🔴 第一梯队：必考（占面试 60%+）

| 主题 | 笔记 | 要点 |
|---|---|---|
| **逻辑回归 / GLM** ⭐⭐⭐ | [[1. Logistic Regression & GLMs]] | logit link、MLE、odds ratio、分离问题 |
| ↳ 指数族与 link | [[1.2 Generalized Linear Models (GLMs)]] | `μ=b'(θ)`、`Var=φb''(θ)`、canonical link |
| ↳ 系数解释 | [[1.3 Introducing Logistic Regression]] | `exp(β)`=odds ratio |
| ↳ 分离与惩罚 | [[1.6 Estimation Issues & Separation]] | 完全分离 → Firth / 正则化 |
| ↳ 正则化 | [[1.6.3 Regularization]] | Ridge / Lasso / ElasticNet |
| **决策树** ⭐⭐⭐ | [[7.2.2 Decision Tree]] | 分裂准则、剪枝、过拟合 |
| ↳ CART | [[7.5 CART Algorithm]] | Gini、二叉、CCP |
| ↳ 剪枝 | [[7.2.2.4. Pruning Decision Trees]] | 预剪枝 vs 后剪枝、代价复杂度 |
| **随机森林** ⭐⭐⭐ | [[7. Random Forest]] | bagging + 随机子空间、OOB、`m=√p` |
| ↳ 建树流程 | [[7.2. Building the Forest]] | 复杂度、ΔI、方差降 1/B |
| ↳ 特征重要性 | [[7.4. Quantifying Feature Importance]] | MDI 偏差 vs permutation |
| **Boosting** ⭐⭐⭐ | [[8.4. Boosting]] | 前向分步、加性模型 |
| ↳ AdaBoost | [[8.4.2.1 AdaBoost (Adaptive Boosting)]] | `α=½ln((1−ε)/ε)`、权重更新 |
| ↳ GBM | [[8.4.2.2 Gradient Boosting]] | 伪残差 = 负梯度、shrinkage |
| ↳ 公式推导 | [[8.4.2.1.4 AdaBoost Formulas]] | 指数损失 → AdaBoost |
| **Bagging vs Boosting** ⭐⭐⭐ | [[8.6 Bagging VS Boosting]] | 偏差/方差、并行性、调参 |
| **评估指标** ⭐⭐⭐ | [[6.3 Performance Metrics]] | AUC/KS/Lift、混淆矩阵 |


### 🟡 第二梯队：高频（占 25%）


| 主题 | 笔记 | 要点 |
|---|---|---|
| 共线性与 VIF ⭐⭐ | [[3. Multicollinearity]] | VIF 公式、模型影响差异 |
| ↳ VIF 专章 | [[3.4.2 Variance Inflation Factor (VIF)]] | 辅助回归、阈值 |
| 缺失值 ⭐⭐ | [[4. Missing Data]] | MCAR/MAR/MNAR、MICE |
| WOE / IV ⭐⭐ | [[2.3.1 WOE & IV]] | 公式、阈值、单调性 |
| 变量变换 ⭐⭐ | [[2. Transformations]] | 分箱、Box-Cox、样条 |
| 变量筛选 ⭐⭐ | [[5. Dimension Reduction]] | filter/wrapper/embedded |
| ↳ 单变量 | [[5.2 Univariate Selection]] | χ²/ANOVA/IV/AUC |
| ↳ 多变量 | [[5.3 Multivariate Selection]] | VIF/MI/RFE/Stepwise |
| 数据划分与 CV ⭐⭐ | [[6.1 Data Preparation & Validation]] | 分层、泄漏、时间序列 |
| 偏差方差 ⭐⭐ | [[6.2 Model Diagnosis_Bias-Variance Tradeoff]] | 学习曲线、正则化 |
| 树算法细节 ⭐⭐ | [[7.2.2.3.1 ID3 Algorithm]] | 信息增益、熵 |
| ↳ C4.5 | [[7.2.3.3 C4.5 Algorithm]] | 增益率、缺失值 |
| 集成总览 ⭐⭐ | [[8. Ensemble Learning]] | 三家族对比 |
| ↳ Bagging | [[8.3. Bagging (Bootstrap Aggregating)]] | 63.2%、方差公式 |

### 🟢 第三梯队：了解（占 10–15%）

| 主题 | 笔记 |
|---|---|
| OLS 与假设 | [[1.1 Ordinary Least Squares (OLS)]]、[[1.7 Assumptions]] |
| 模型拟合原理 | [[1.4 Model Fitting]] |
| 不平衡样本 | [[1.5 Dealing with Unbalanced Samples]] |
| VARCLUS | [[3.5.2 SAS VARCLUS]] |
| PCA | [[3.5.4 PCA]] |
| 诊断可视化 | [[6.4 Diagnostic & Visualization Tools]] |
| ID3 实现与手算 | [[7.2.2.3.1.4 ID3 Complete Code]]、[[7.2.2.3.1.9 ID3 Build Example]] |

---

## 3. 保险 / 风控专属（差异化加分）

> 通用 ML 候选人答不好的地方，正是拉开差距的地方。

**见 [[10. 保险风控考点总览]]**：频率-严重度、Poisson/Gamma/Tweedie、评分卡刻度、KS/PSI、拒绝推断、SAS 实操。

---

## 4. 速查与自测

- [[99.1 题库总览]] —— 全部题目（含答案 / 追问链 / 错误答法）
- [[04. 公式速查卡]] —— 一页纸背完所有公式
- [[05. 高频追问 TOP 30]] —— 面试官最爱追问的问题
- [[事实核验报告]] —— 全部内容的权威来源核验记录（PSL / Loss Data Analytics）

---

## 5. 命名与编号约定

- `0x` 开头 = 冲刺层（精简、面试导向）
- `1–8` 章 = 系统知识（原有，保留）
- `10` 章 = 保险风控实务
- `99` = 题库
- `_meta` = 规范、模板、核验、脚本（不参与学习）

---

## 6. 维护

```bash
bash _meta/tools/vault_check.sh     # 体检：断链/空文件/AI残留/编号
```
