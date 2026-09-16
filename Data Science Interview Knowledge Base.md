---
title: "Data Science Interview Knowledge Base"
tags: [meta, cheatsheet]
status: 完成
updated: 2026-09-16
---

<!-- priority-banner -->
> [!info] 本页是**一页速查卡**（原总纲精简而成）：每章只留面试前最该背下来的结论，推导、算例、误解都在各章笔记里。
> 👉 准备面试请从 **[[00 Index]]** 进入：那里有考点权重、学习路径与题库。
> 🚀 [[02. 三条学习路径]] ・ 📋 [[04. 公式速查卡]] ・ ✅ [[99.1 题库总览]] ・ 🏢 [[10. 保险风控考点总览]]

# 一页速查（按面试权重排序）

## 1. GLM 与逻辑回归（权重 30%）

- 二值结局不能直接套线性回归：概率被卡在 (0,1) 之间，直线会给出小于 0、大于 1 的预测；而且 Bernoulli 的方差 `p(1−p)` 随均值变化，同方差假设当场作废。
- 逻辑回归建模的是 log-odds：`logit(p) = ln(p/(1−p)) = x'β`。系数只在 log-odds 尺度上可解释，`exp(β)` 是 odds ratio，不是概率的变化量。
- 没有闭式解，用 MLE + IRLS 迭代求；某个变量能完美分开两类时系数发散，普通 MLE 给不出有限解，改用 Firth 惩罚或正则化。
- 真正要验的假设只剩一条：logit 的线性性。残差正态与同方差是对 OLS 的要求，不是对逻辑回归的。

👉 细节：[[1.3 Introducing Logistic Regression]] ・ [[1.4 Model Fitting]] ・ [[1.6 Estimation Issues & Separation]] ・ [[1.7 Assumptions]]

## 2. 变量变换（特征工程 15%）

- 变换修四类毛病：关系非线性、残差异方差、分布偏斜、离群值影响大。
- 三种主力：分箱（无监督用等宽/分位数，有监督让每箱尽量分开目标）、Box-Cox（λ 由数据定，要求全为正数，否则用 Yeo-Johnson）、样条（比多项式更稳，外推不易失控）。
- WOE = `ln(%Good / %Bad)`，与 log-odds 线性对应，所以 logistic 里一个 WOE 变量的系数就是一条单调斜率；IV 判变量强弱，但 IV > 0.5 通常是过拟合嫌疑而不是「信号强」。

👉 细节：[[2. Transformations]] ・ [[2.3.1 WOE & IV]]

## 3. 多重共线性（特征工程 15%）

- 共线让 `(X'X)⁻¹` 不稳定，系数方差被放大；VIF 作用在**方差**上，标准误只放大 `√VIF` 倍（VIF = 10 → SE 约 ×3.16）。
- 树模型基本免疫（不做矩阵求逆），但线上相关结构漂移时预测会变脆。
- 处理顺序：按 VIF 从高到低迭代删变量（每删一个都重算 VIF）、变量聚类（VARCLUS）保业务含义、惩罚回归、PCA。

👉 细节：[[3. Multicollinearity]] ・ [[3.4.2 Variance Inflation Factor (VIF)]]

## 4. 缺失值（特征工程 15%）

- 先判机制再选方法：MCAR（与一切都无关）、MAR（可由其他已观测变量解释）、MNAR（取决于缺失值自身，最难）。
- MAR 下黄金标准是多重插补（MICE）：造 m 份完整数据、分别分析、合并结果，把插补的不确定性算进去。
- 均值插补会毁掉方差、扭曲相关性；完整案例分析只在缺失极少（<5%）且 MCAR 时才安全。

👉 细节：[[4. Missing Data]]

## 5. 降维与变量筛选（特征工程 15%）

- 先分清两件事：筛选＝从原变量里挑（保留语义），投影＝造新变量（PCA/VARCLUS，丢语义）。
- 单变量筛选（方差 / 卡方 / ANOVA F / IV / AUC）快，但会漏掉「单独弱、组合强」的变量，还制造多重比较的假阳性。
- 多元方法：VIF 过滤、互信息、RFE、LASSO；**特征选择必须放在 CV 循环内部**，否则泄漏。

👉 细节：[[5.2 Univariate Selection]] ・ [[5.3 Multivariate Selection]] ・ [[3.5.4 PCA]]

## 6. 模型评估（权重 15%）

- 划分方式决定评估的诚实度：随机划分 vs 时间序 OOT；CV 用来选模型，测试集只碰一次。
- 偏差-方差：训练误差就高 = 欠拟合；训练低、验证高 = 过拟合。平方损失下 `E[test] = bias² + variance + σ²`（0-1 损失没有这种加法分解）。
- 指标要与业务挂钩：不平衡时看 PR/AUC 而不是 accuracy；风控场景看 KS、PSI 与 Lift。

👉 细节：[[6.1 Data Preparation & Validation]] ・ [[6.2 Model Diagnosis_Bias-Variance Tradeoff]] ・ [[6.3 Performance Metrics]] ・ [[6.4 Diagnostic & Visualization Tools]]

## 7. 树与随机森林（权重 20%）

- 分裂准则：分类看 Gini / 熵（ID3 用信息增益，C4.5 用增益率修正对多取值特征的偏好），回归用方差下降。
- 随机森林 = bagging + 每次分裂只看随机特征子集。加树主要降方差且有平台——这是它与 GBM 的关键差别。
- 特征重要性两种：MDI 快但偏向高基数特征；permutation importance 更可信，要给方向就用 SHAP。
- 剪枝：预剪枝（设深度/叶大小）与后剪枝；代价复杂度剪枝（CCP）用 α 平衡树大小与拟合，α 由 CV 选。

👉 细节：[[7.2.2 Decision Tree]] ・ [[7.5 CART Algorithm]] ・ [[7.4. Quantifying Feature Importance]] ・ [[7.2.2.4. Pruning Decision Trees]]

## 8. 集成方法（权重 20%）

- 三大族：bagging 降方差、boosting 降偏差、stacking 组合异质模型（用折外预测当元特征）。
- AdaBoost 用指数损失；梯度提升是函数空间里的梯度下降——每轮拟合损失的负梯度（伪残差），所以任何可微损失都能接进来。
- GBM 防过拟合三件套：浅树 + 小学习率 + 早停；学习率与树数成对调，固定一个再搜另一个。

👉 细节：[[8.3. Bagging (Bootstrap Aggregating)]] ・ [[8.4.2.1 AdaBoost (Adaptive Boosting)]] ・ [[8.4.2.2 Gradient Boosting]] ・ [[8.6 Bagging VS Boosting]]

## 9. 统计基础（权重 10%）

- 分布识别：计数 → Poisson（均值 = 方差是招牌也是软肋，方差偏大就换负二项）；正偏金额 → Gamma 或 Lognormal；一堆零 + 右偏正值 → Tweedie。
- CLT 说的是**均值的抽样分布**趋近正态，不是数据本身正态；标准误 = 标准差 / √n，随 √n 收敛。
- p 值不是「原假设为真的概率」，也不表示效应大小；跑很多次检验要用 Bonferroni 或 BH 控制假阳性。

👉 细节：[[9.1 概率分布与随机变量]] ・ [[9.3 估计：点估计、MLE 与区间估计]] ・ [[9.4 假设检验与 p 值]] ・ [[9.6 贝叶斯基础与基础率谬误]] ・ [[9.7 多重比较与选择偏差]]

## 10. 保险风控实务（权重 10%）

- 纯保费 = 频率 × 严重度；两条路线：两阶段（Poisson 频率 + Gamma 严重度）或单阶段 Tweedie（一次似然同时处理零质量与右偏正值）。
- `offset = log(exposure)` 让模型按「每保单年」工作；评分卡刻度化：`score = A − B·log(odds)`，PDO 定 B、base score / base odds 定 A。
- 监控分两层：KS/AUC 看区分度，PSI 看分数分布漂移；线上下滑先分清是入口人群变了还是模型衰减。

👉 细节：[[10.1 频率-严重度与纯保费]] ・ [[10.2 计数模型与 offset]] ・ [[10.8 Tweedie GLM 专章]] ・ [[10.5 模型评估与监控]] ・ [[10.9 项目实战：法语车险纯保费定价]]

---

## 冲刺层入口

[[00 Index]] ・ [[01. 面试画像与考点分布]] ・ [[02. 三条学习路径]] ・ [[03. 答题模板与追问应对]] ・ [[04. 公式速查卡]] ・ [[05. 高频追问 TOP 30]] ・ [[06. 中英术语对照表]]

> 复习顺序：先在 [[00 Index]] 按权重定时间 → 沿 [[02. 三条学习路径]] 逐章走 → 面试前 30 分钟只看 [[04. 公式速查卡]] 和本页。
