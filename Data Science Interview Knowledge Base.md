---
title: "Data Science Interview Knowledge Base"
tags: [meta, cheatsheet]
status: 完成
updated: 2026-09-16
---

<!-- priority-banner -->
> [!info] 本页是**一页速查卡**（原总纲）。
> 👉 面试准备请从 **[[00 Index]]** 进入：那里有考点权重、三条学习路径、公式速查与题库。
> 🚀 [[02. 三条学习路径]] ・ 📋 [[04. 公式速查卡]] ・ ✅ [[99.1 题库总览]] ・ 🏢 [[10. 保险风控考点总览]]

# [[1. Logistic Regression & GLMs]]

## 📖 标准定义

> **本库规则：定义类内容必须给出教材原文，中文只作解释。** 汇总见 [[定义原文库]]。

**① 教材原文 · PSL §10.1 "Setup"**（<https://liangfgithub.github.io/PSL/w10/w10_1_setup.html>）

> "As we have learned before, in the binary case, the best classifier depends on $\eta(x)=P(Y=1 \mid X=x)$. One type of approaches for classification is to directly model or estimate $\eta(x)$. Since $\eta(x)$ is constrained to between 0 and 1, as it represents a probability. Therefore, it's challenging to model $\eta(x)$ directly with a linear model because linear models are unconstrained. Instead, we model its transformation (or referred to as a **link function**) with a linear model: $g(\eta(x)) = x^t \beta$."
>
> "In logistic regression, we use the so-called **logit link function**, which is equal to $\mathrm{logit}(\eta(x)) = \log \dfrac{\eta(x)}{1 - \eta(x)}$."

**② GLM 框架 · Loss Data Analytics Ch8**（<https://openacttexts.github.io/Loss-Data-Analytics/ChapRiskClass.html>）

> "Poisson regression is a special member of a more general regression model class known as the **generalized linear model (GLM)**. The GLM develops a unified regression framework for datasets when the response variables are continuous, binary or discrete. The classical linear regression model with a normally distributed error is also a member of the GLM."

**③ 标准记法**（GLM 的通用形式，非某本教材独有）

| 成分 | 记法 |
|---|---|
| 随机成分（random component） | $Y_i \sim \mathrm{Bernoulli}(\pi_i)$，相互独立 |
| 系统成分（systematic component） | 线性预测子 $\eta_i = x_i^t \beta$ |
| 联系函数（link function） | $\mathrm{logit}(\pi_i) = \log\dfrac{\pi_i}{1-\pi_i} = \eta_i$ |
| 等价形式 | $\pi_i = E[Y_i \mid x_i] = \dfrac{1}{1+e^{-\eta_i}}$ |
| 估计 | 极大似然估计（MLE），由 IRLS / Fisher scoring 迭代求解 |

**④ 中文解释**

逻辑回归是**二值响应**的广义线性模型（GLM）。要说的三件事：① 响应 $Y$ 服从 **Bernoulli** 分布；② 系统成分是**线性预测子** $x^t\beta$；③ **link 用 logit**，把落在 $(0,1)$ 的概率映到 $(-\infty,+\infty)$ —— 因为线性模型的取值不受限，而概率受限，所以不能直接建模概率。

因此它**直接建模的是 $\eta(x)=P(Y=1\mid X=x)$ 的 log-odds**，概率由反变换得到，**不是直接建模概率本身**。这一点是面试常问的辨析点。

#### Link Function: The Logit
- **是什么（What it is）**：odds 的自然对数。
- **公式（Formula）**：`logit(p) = log(p / (1-p))`
- **目的（Purpose）**：把概率 `p`（0 到 1）变换到连续尺度（-∞ 到 +∞）。

#### Model Equation
$$ \log\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 X_{i1} + \dots + \beta_k X_{ik} $$

- **怎么解释（Interpretation）**：`X_j` 每变动一个单位，结局的 **log-odds** 就变动 `β_j`。把系数取指数 `exp(β_j)`，得到 **odds ratio**。

## ✅ Key Assumptions
- **二值结局（Binary Outcome）**：因变量必须是二值的。
- **Logit 的线性性（Linearity of the Logit）**：预测变量与结局的 log-odds 之间是线性关系。
- **观测独立（Independence of Observations）**：观测之间互不相关。
- **无完全共线性（No Perfect Multicollinearity）**：预测变量之间不是完全相关。
- **样本量足够（Large Sample Size）**：经验法则是每个预测变量至少要有 ≥10–20 个**最稀有那一类**的样本。

## ⚠️ Common Problems & Solutions

### 1. Imbalanced Samples (Rare Events)
- **问题（Problem）**：模型偏向多数类（majority class）。
- **解法（Solutions）**：
  - **重采样（Resampling）**：对少数类过采样（如 SMOTE），或对多数类欠采样。
  - **类别权重（Class Weights）**：把少数类的错分惩罚加重。
  - **指标（Metrics）**：用 **AUROC** 或 **Precision-Recall 曲线**，而不是 accuracy。

### 2. Separation
- **问题（Problem）**：某个预测变量把两个结局类别**完美（或近乎完美）分开**，导致 MLE 失败或标准误巨大。
- **解法（Solutions）**：
  - **惩罚回归（Penalized Regression）**：用 Ridge（L2）或 Lasso（L1）正则化。
  - **Firth 校正（Firth's Correction）**：一种减少偏差（bias-reduction）的方法。
  - **贝叶斯先验（Bayesian Priors）**：用有信息量的先验来正则化系数。

## 📊 Evaluation
主要用 **混淆矩阵（confusion matrix）** 及其派生指标来评估：

- **Accuracy**：整体正确率（类别不平衡时会误导）。
- **Precision**：`TP / (TP + FP)` —— 预测为正的里面，有多少真的是正的？
- **Recall（Sensitivity）**：`TP / (TP + FN)` —— 真正的正例里，找出来了多少？
- **F1-Score**：Precision 与 Recall 的调和平均。
- **AUROC**：模型把一个随机正例排得比一个随机负例更高的能力。
# [[2. Transformations]]
## 🎯 Why Transform Variables?
- **改善线性性（Improve Linearity）**：让线性模型拟合得更好。
- **稳定方差（Stabilize Variance）**：修掉残差图里的漏斗形（异方差 heteroscedasticity）。
- **正态化分布（Normalize Distributions）**：让偏斜数据更对称，推断更可靠。
- **控制离群值（Control Outliers）**：降低极端值的影响。
- **编码类别（Encode Categories）**：把非数值数据转成算法能吃的形式。

## 🔍 When to Transform? (Key Signals)
- **残差图（Residual Plots）**：出现弯曲（非线性）或漏斗形（方差问题）。
- **直方图 / Q-Q 图（Histograms / Q-Q Plots）**：出现明显偏斜或重尾。

---

## 🛠️ Core Transformation Techniques Overview

| 技法（Technique） | 适用于（Applies to） | 看目标吗（Target-Aware?） | 主要目的（Primary Goal） |
| :--- | :--- | :--- | :--- |
| **Dummy / One-Hot 编码** | 类别型 | 否 | 把类别转成数值格式。 |
| **Winsorization（封顶 Capping）** | 连续型 | 否 | 降低离群值影响。 |
| **分箱（Binning，无监督）** | 连续型 | 否 | 简化、处理非线性。 |
| **Box-Cox / 幂变换** | 连续型 | 否 | 正态化偏斜、稳定方差。 |
| **多项式 / 样条（Polynomials / Splines）** | 连续型 | 否 | 建模复杂的非线性关系。 |
| **证据权重（Weight of Evidence, WOE）** | 类别型 | 是（二分类） | 给类别造一个单调、有预测力的数值评分。 |
| **有监督分箱（Supervised Binning）** | 连续型 | 是 | 造出最能区分目标类别的箱。 |
| **GAM 平滑** | 连续型 | 是 | 自动找出并套用一条光滑的非线性变换。 |

---

## ✨ Key Techniques Explained

### 1. Categorical Encoding

#### One-Hot / Dummy Coding
- **目标（Goal）**：把一个类别变成若干二值（0/1）列。
- **Dummy Coding**：造 `k-1` 列，留一个类别当基准（baseline，进截距）。系数可解释。
- **One-Hot Encoding**：造 `k` 列。若在线性模型里全用上，会造成共线性。

#### Weight of Evidence (WOE) & Information Value (IV)
- **目标（Goal）**：按类别与二分类目标的关系，把类别转成一个数值评分。
- **WOE 公式**：`WOE = ln(%Goods / %Bads)`
  - **Good = 非事件 (Y=0)**，**Bad = 事件 (Y=1)**（学分卡主流约定）。
  - **Positive WOE: 该组好人占比更高 ⇒ 低风险（事件 Y=1 的 odds 更低）。**
  - **Negative WOE: 高风险。**
  - 全库统一约定，详见 [[2. Transformations]]、[[2.3.1 WOE & IV]]、[[5.2 Univariate Selection]]。
- **Information Value (IV)**：基于 WOE 衡量变量的预测力。
  - **经验法则（Rule of Thumb）**：`IV < 0.02`（无用）、`0.1 - 0.3`（中）、`> 0.3`（强）。

### 2. Continuous Transformations

#### Binning
- **无监督（Unsupervised）**：只看特征自身的分布来分箱。
  - **等宽（Equal-Width）**：简单，但对离群值敏感。
  - **分位数（Quantile）**：每箱观测数相同；对离群值稳健。
- **有监督（Supervised）**：分箱的目标是最大化目标类别之间的区分度（例如用卡方或熵）。

#### Power Transforms (Box-Cox)
- **目标（Goal）**：找出最优幂次 `λ`（lambda），让数据更接近正态。
- **公式（Formula）**：`(x^λ - 1) / λ`
- **常见 λ**：`λ=0` 是 log，`λ=0.5` 是平方根，`λ=-1` 是倒数。
- **前提（Requirement）**：数据必须为正。

#### Polynomials & Splines (for non-linearity)
- **多项式（Polynomials）**：给模型加特征的幂次（如 `x²`、`x³`）来捕捉曲线。容易过拟合、外推时也容易失控。
- **样条 / GAM（Splines / GAMs）**：用灵活的分段曲线拟合数据。在捕捉复杂关系时比多项式更稳、更强、且不易过拟合。**GAM** 则把这个过程自动化 —— 给每个预测变量都拟合一条光滑函数 `f(x)`。

# [[3. Multicollinearity]]
## **Definition & Core Problem**
共线性（multicollinearity）指的是预测变量之间高度相关，导致**无法分辨单个变量的效应**。数学根子在于：预测变量相关时，矩阵 (X'X)⁻¹ 变得不稳定，系数估计的方差因此爆炸。

**多米诺效应（The Domino Effect）：** 高相关 → 高 VIF → 标准误被放大 → 系数不稳 → p 值很大 → 解释不可靠
（注意：**VIF 作用于方差，标准误只放大 √VIF** —— VIF = 10 时标准误约放大 3.16 倍，不是 10 倍；且 **Tolerance = 1/VIF**。）

## **Model Impact**
| **模型类型（Model Type）** | **影响（Effect）** |
|:---|:---|
| **线性 / Logistic / GLM** | 🚫 **受严重影响** —— 系数不稳、SE 被放大、p 值不可靠 |
| **树模型（Tree-Based Models）** | ✅ **基本免疫** —— 不做矩阵求逆；会从相关变量组里自然挑一个 |

## **Detection Methods**
1. **相关矩阵（Correlation Matrix）**：|r| ≥ 0.70 就有问题
2. **方差膨胀因子（Variance Inflation Factor, VIF）**：VIF = 1/(1-R²)
   - VIF = 1：没有共线性
   - VIF > 5：值得关注
   - VIF > 10：严重问题
   - **Condition index（条件指数）**：来自 X'X 的特征值，**> 10 中度、> 30 严重**；**方差分解比例**能定位是哪些变量共享了坏维度

## **Solutions**

| **方法（Method）** | **做法（Approach）** | **优点（Pros）** | **缺点（Cons）** |
|:---|:---|:---|:---|
| **删变量（Variable Removal）** | 按 VIF 从高到低迭代删除 | 简单、可解释 | 丢信息 |
| **变量聚类（Variable Clustering）** | 把相关变量聚成组，每组选代表 | 保住业务含义 | 更复杂 |
| **惩罚回归（Penalized Regression）** | Ridge/LASSO 对系数加惩罚 | 稳定、自动选择变量 | 多了调参 |
| **PCA 回归（PCA Regression）** | 变换成互不相关的成分 | 数学上优雅 | 丢可解释性 |

## **Key Takeaway**
- **为解释（For Interpretation）**：共线性让 GLM 里的单个系数失去意义
- **为预测（For Prediction）**：树模型天然处理得了；GLM 可能需要干预（注意两个 caveat：线上相关结构漂移时预测变脆、预测区间变宽）
- **快速修法（Quick Fix）**：按 VIF > 10 顺序删变量，**每删一个都要重算 VIF**
# [[4. Missing Data]]

**定义与目的（Definition & Purpose）**：某些变量上存在缺失条目；必须审慎处理，否则会引入偏差。

### **🎲 MCAR (Missing Completely At Random)**

#### **Diagnosis: Pure Bad Luck**

| **它是什么（What it is）**                                                               | **怎么发现（How to Spot It）**                                                                  | **含义（Implication）**                                                                 |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 某个值缺失的概率**与一切都无关** —— 纯粹是随机事件。 | 缺失像随机散布、毫无模式。正式检验：**Little's MCAR 检验**（p > 0.05）。 | 观测到的数据是**无偏**（但更小）的子样本。主要风险是统计功效下降。 |

#### **Fixes for MCAR**

| **方法（Method）**                        | **核心思路（Core Idea）**                                                   | **优点（Pros）**                                       | **缺点（Cons）**                                                                                                     |
| --------------------------------- | --------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **完整案例分析（删除 Complete Case Analysis）** | 删掉任何含缺失值的行。                      | 简单、快，而且**若数据真是 MCAR 则无偏**。 | 浪费数据、功效下降、置信区间变宽。**只建议在缺失 <5% 时用。** |
| **均值 / 众数插补（Mean / Mode Imputation）**        | 用该列的均值或众数填缺失。 | 保住样本量，实现极简单。 | 🚨 **毁掉方差**、**扭曲相关性**，让你对结果过度自信。                  |

---

### **🔗 MAR (Missing At Random)**

#### **Diagnosis: Systematically Explainable**

| **它是什么（What it is）**                                                               | **怎么发现（How to Spot It）**                                                                              | **含义（Implication）**                                                                                                   |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 某个值缺失的概率可以由**其他已观测变量**解释。 | 可视化上能看出模式（例如 `Income` 缺失与 `Age` 相关）。原因**就在你的数据里**。 | **这是好事！** 你可以用数据里的关系，对缺失值做出聪明的、无偏的估计。 |

#### **Fixes for MAR**

| **方法（Method）**                     | **核心思路（Core Idea）**                                                                                           | **优点（Pros）**                                                                                  | **缺点（Cons）**                                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **回归插补（Regression Imputation）**      | 把其他列当特征，用回归模型预测缺失值。                        | 保住变量之间的关系；比均值插补聪明。         | 假设了特定的模型形式（如线性）；若不加随机噪声，会低估方差。 |
| **k-NN 插补（k-NN Imputation）**            | 用最相似的 `k` 条完整行（「邻居」）的平均来填。                      | 无需模型就能捕捉复杂、非线性关系。                       | 大数据上慢；对特征缩放和 `k` 的取值敏感。                     |
| **多重插补（Multiple Imputation, MICE）** | ✨ **黄金标准（Gold Standard）** ✨ <br> 造出 `m` 个看似合理的完整数据集，逐个分析，再合并结果。 | **正确地把不确定性算进去**，给出有效的 p 值与置信区间。 | 实现和理解都最复杂；计算量大。                        |

---

### **❓ MNAR (Missing Not At Random)**

#### **Diagnosis: The Unseen Cause**

| **它是什么（What it is）**                                                               | **怎么发现（How to Spot It）**                                                                         | **含义（Implication）**                                                                                                           |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| 某个值缺失的概率取决于**缺失值本身**。 | 通常要靠**领域知识**。光看数据看不出来（例如收入极高的人会隐瞒）。 | **这是最难的一类。** 标准插补方法会引入**偏差** —— 修复缺口所需的信息，恰恰是缺的那部分。 |

#### **Fixes for MNAR**

| **做法（Approach）**              | **核心思路（Core Idea）**                                                                                                          | **关键考量（Key Consideration）**                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **给缺失机制建模（Model the Missingness）** | 用进阶统计技法（如选择模型 Selection Models），显式地对缺失机制做出假设。 | 需要扎实的统计功底，以及可辩护的缺失原因假设。            |
| **敏感性分析（Sensitivity Analysis）**  | 在不同「如果……会怎样」的情景下插补（例如假设缺失值全都很高，再假设全都很低）。                 | 不给你唯一答案，但能检验结论在各种合理假设下是否稳健。    |
| **去收集更多数据（Collect More Data）**     | 回到源头，用追踪调查等方式把缺失信息补回来。             | 往往不现实，但它是不做无法检验的假设、真正解决问题的唯一办法。 |
# [[5. Dimension Reduction]]

**定义与目的（Definition & Purpose）**：在尽量保住信息的前提下减少变量数量，提升可解释性与表现。
**什么时候用、怎么识别（When & How to Identify）**：当特征数相对观测数偏多、或共线性严重时使用。通过探索性分析、解释方差、过拟合症状来识别。

* **何时 & 为什么（When & why）**：降噪、提升可解释性、对抗过拟合、降低存储 / 计算
* **单变量筛查（Univariate screening）**：IV、χ²、ANOVA、AUROC，逐个给特征排序
* **多变量筛查（Multivariate screening）**：Spearman、Hoeffding's D、mutual information，看变量之间

**降维技法的类型（Types of Dimensionality Reduction Techniques）**

|                                               | **看目标（有监督 Target-aware）**                                                                                                | **不看目标（无监督 Target-unaware）**                                                                                           |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **降维 / 特征选择（Dimension reduction）**   | **Filter**：correlation、ANOVA、IV <br>**Wrapper**：RFE、forward/backward stepwise <br>**Embedded**：LASSO、树的重要性 | **Filter**：方差阈值、mutual info <br>**Wrapper**：配合 CV 的 k-NN <br>**Embedded**：无监督特征重要性 |
| **特征投影（Dimension projection）** | 有监督版的 PCA、有监督 autoencoder                                                                             | PCA、ICA、t-SNE、无监督 autoencoder                                                                                  |

### **1. Univariate Selection (First Pass)**
| 方法（Method）              | 适用场景（Use Case）                  | 最擅长（Best For）                    |
| ------------------- | ------------------------- | --------------------------- |
| **方差过滤（Variance Filter）** | 丢掉近似常量的列       | 极快的清理（Lightning fast cleanup）      |
| **卡方（Chi-Squared）**     | 类别 → 类别 | 非线性关系    |
| **ANOVA F 检验**    | 数值 → 类别     | 线性关系        |
| **AUC 排序**     | 直接的预测力   | 与量纲无关的筛查 |

### **2. Multivariate Selection**
| 方法（Method） | 核心思路（Core Idea） | 最擅长（Best For） |
|--------|-----------|----------|
| **相关 / VIF 过滤** | 删掉高度相关的一对 | 线性冗余 |
| **互信息（Mutual Information）** | 对目标 MI 高、对已选特征 MI 低 | 非线性依赖 |
| **RFE** | 淘汰赛（knockout tournament） | 结合模型的选择 |

### **3. Specialized Methods**
- **VARCLUS：** 把相似特征聚成主题（簇）
- **PCA：** 变换成正交成分（保住 95% 方差）
- **树模型（Tree-Based）：** 用 RF/GBM 的 Gini 或 permutation importance
- **LASSO / Elastic Net：** L1 惩罚把系数收缩到 0
## **Strategic Workflow**
1. **快速清理（Quick Cleanup）：** 删掉零方差列与重复列
2. **第一遍（First Pass）：** 单变量过滤（AUC、卡方）
3. **选方法（Choose Method）：**
   - **特征 < 100：** 带 CV 的 RFE
   - **p ≫ n：** LASSO
   - **需要可解释的分组：** VARCLUS
   - **反正要用 RF/GBM：** 用它们内置的重要性
## **Key Rules**
- **黄金法则（Golden Rule）：** 特征选择必须在 **CV 循环内部**（防泄漏）
- **标准化（Standardize）：** 惩罚类方法一律要做
- **留文档（Document）：** 记录过程与参数

**快速决策（Quick Decision）：** 方差过滤 → AUC 排序 → 大多数问题用 LASSO。
# [[6. Model Assessment]]

**定义与目的（Definition & Purpose）**：评估模型表现与在未见数据上泛化能力的框架。
**什么时候用、怎么识别（When & How to Identify）**：训练之后用来估计真实水平。凡是需要无偏的表现估计与模型比较时就该用它。

* **数据划分（Data splitting）**：train/validation/test、k 折（分层 stratified）、滚动窗口（时间序列）
* **偏差-方差权衡（Bias–variance trade-off）**：欠拟合 / 过拟合、学习曲线、正则化的影响
* **性能指标（Performance Metrics）**：

| 任务（Task）           | 主要指标（Primary Metrics）                            |
| -------------- | ------------------------------------------ |
| 回归（Regression）     | RMSE, MAE, MAPE, R²                        |
| 分类（Classification） | Accuracy, Precision, Recall, F1, AUROC, KS |
| 利润导向（Profit-based）   | Lift charts, Cumulative gains, Net revenue |

---

# [[7. Random Forest]]

**定义（Definition）**
一组**未剪枝**的决策树（unpruned decision trees）构成的集成：每棵树长在 bootstrap 样本上，且每次分裂只用随机的特征子集。它把 bagging 与随机特征选择合在一起，用来降方差、提升泛化。

---

**拟合过程（Fit / Training Procedure）**

1. 对 *B* 棵树中的每一棵：

   * 从训练集里抽一个 bootstrap 样本。
   * 长一棵完整的决策树：每个节点上随机挑一个特征子集，选最优分裂，重复到满足停止准则。
2. 把这一群树聚合起来做预测。

---

**特征重要性（Feature Importance）**

* **MDI（Mean Decrease in Impurity）：** 把每个特征在所有分裂与所有树上的不纯度下降量加总，再归一化。
* **Permutation Importance：** 在 out-of-bag 或验证数据上随机打乱某个特征的取值，看性能（accuracy 或 MSE）掉多少。

---

### Key Parameters

| 参数（Parameter）               | 控制什么（What It Controls）                             | 权衡 / 备注（Trade-Off / Notes）                                                       |
| :---------------------- | :------------------------------------------- | :---------------------------------------------------------------------- |
| **n\_estimators**       | 树的数量                              | ↑ 降方差，但 ↑ 计算与内存                           |
| **max\_depth**          | 每棵树的最大深度                   | ↓ 树更简单 → 偏差 ↑；↑ 树更复杂 → 过拟合风险         |
| **min\_samples\_split** | 一个节点继续分裂所需的最小样本数    | ↑ 值 → 分裂更少 → 偏差 ↑；↓ 值 → 分裂更多 → 方差 ↑     |
| **min\_samples\_leaf**  | 叶子节点所需的最小样本数              | 防止叶子过小；↑ 值 → 预测更平滑但偏差 ↑   |
| **max\_features**       | 每次分裂考虑的特征数                | 分类：√p；回归：p/3；更低 → 树间更不相关，偏差 ↑ |
| **bootstrap**           | 是否用 bootstrap 抽样（True/False）          | False → 不做 bagging → 方差通常 ↑                                   |
| **criterion**           | 分裂质量的度量（Gini/entropy/MSE/MAE） | 影响分裂决策；entropy 更慢，但有时信息量更大  |
| **class\_weight**       | 不平衡分类下的类别权重  | 把少数类权重 ↑，以降低它的错分                 |

---

### Pros & Cons

| 优点（Pros）                                               | 缺点（Cons）                                              |
| :------------------------------------------------- | :------------------------------------------------ |
| 开箱即用的准确率高                       | 比单棵树更难解释             |
| 抗过拟合（bagging + 随机分裂）    | 计算与内存开销大                |
| 能处理非线性与混合特征类型    | MDI 偏向高基数特征       |
| 自带 OOB 误差估计与特征重要性 | 无法外推到训练数据范围之外 |

---

**使用建议（Usage Tips）**

* 用 OOB 误差做快速验证。
* 画「误差 vs 树的数量」曲线来定 `n_estimators`。
* 在 CV 里用随机搜索或贝叶斯搜索调核心参数（`max_depth`、`min_samples_leaf`、`max_features`）。
* 指导特征工程时，优先用 permutation importance 而不是 MDI，避开它的偏差。

Here's your formatted content using `###` for the main title ("Ensemble Learning") and `####` for the subheadings:

---

# [[8. Ensemble Learning]]

---

#### 1. What & Why

* **集成学习（Ensemble Learning）：** 把多个「弱」模型组合成一个更强的预测器。
* **目标（Goal）：** 通过模型多样性来降方差（bagging）、降偏差（boosting），或两者兼得（stacking）。

---

#### 2. Core Concepts

* **偏差-方差权衡（Bias–Variance Tradeoff）：**

  * Bagging 靠平均降方差。
  * Boosting 靠顺序修正错误降偏差。
  * Stacking 通过元学习（meta-learning）两者都能处理。

* **多样性（Diversity）：** 成功的关键 —— 靠重采样（bagging）、重加权（boosting）或异质学习器（stacking）来实现。

---

#### 3. Methods

1. **Bagging（例如 Random Forest）**

   * **怎么做（How）：** 在 bootstrap 样本上训练基学习器，再平均 / 投票。
   * **优点（Pros）：** 可并行，降方差效果强。
   * **缺点（Cons）：** 修不了偏差。

2. **Boosting（例如 AdaBoost、Gradient Boosting）**

   * **怎么做（How）：** 顺序地对残差或错分样本拟合学习器，再用加权和聚合。
   * **优点（Pros）：** 同时降偏差与方差；损失函数灵活。
   * **缺点（Cons）：** 必须串行（并行度低）、有过拟合风险、超参数多。

3. **Stacking**

   * **怎么做（How）：** 第一层模型 → 产生折外预测（out-of-fold predictions）→ 在这些「元特征」上训练一个元学习器。
   * **优点（Pros）：** 组合异质模型；能捕捉复杂模式。
   * **缺点（Cons）：** 复杂、计算重、交叉验证要特别小心。

---

#### 4. Quick Comparison

|              | 降偏差 Bias↓ | 降方差 Variance↓ | 可并行（Parallel）  | 复杂度（Complexity） |
| ------------ | ----- | --------- | --------- | ---------- |
| **Bagging**  | –     | ✔         | ✔         | 低        |
| **Boosting** | ✔     | ✔         | ✖️（串行 seq.） | 中     |
| **Stacking** | ✔/✖️  | ✔         | 部分（Partial）   | 高       |

---

#### 5. Tips

* **从简单开始（Start Simple）：** Random Forest → Gradient Boosting → Stacking。
* **防过拟合（Prevent Overfitting）：** CV、early stopping、树深度 / 正则化。
* **可解释性（Interpretability）：** 用 SHAP 或特征重要性工具。

---

> **收尾**：以上是整库的面试速查主线——先讲清概念与假设，再讲指标与诊断，最后落到风控/保险的业务口径（WOE/IV、评分刻度、PSI）。复习顺序建议从 [[00 Index]] 的分层入口进入，公式只背 [[04. 公式速查卡]]。
