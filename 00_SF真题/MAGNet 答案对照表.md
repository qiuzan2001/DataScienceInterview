---
title: "MAGNet 答案对照表"
aliases: [MAGNet 对照表, 真题对照]
tags: [statefarm, interview, magnet, index]
priority: "⭐⭐⭐"
status: 完成
updated: 2026-09-16
---

# MAGNet 答案对照表

> [[MAGNet 面试真题全解]] 的逐题导航。**先用真题的一句话答案开口，需要细节时再跳到深潜笔记。**
> 覆盖度图例：✅ 库内已充分覆盖 ｜ 🟡 部分覆盖（有缺口）｜ ❌ 库内缺失（已在本次补齐 / 待补）
>
> **计数口径**：MAGNet 共 10 节，其中 §1–§9 是 **55 道题**（本表编号 `1.1`–`9.8`，一道不漏），§10 是 **2 个 STAR 框架**（编号 `10.1`–`10.2`）——所以本表共 **57 行**，对应 MAGNet 的 55 道「Q:」+ 2 个 STAR 框架。
> 「一句话答案的要点」优先取自 MAGNet 每题的引用行（`>` 那句）；⭐ 取自 [[优先级矩阵]] 的该题落点笔记分级，并参考 MAGNet 的 Self-Test Tier。

## 0. 覆盖度总览

| MAGNet 章节 | 题数 | 覆盖度 | 主要落点笔记 | 缺口数 |
|---|---|---|---|---|
| 第 1 节 Logistic Regression & GLMs | 10 | 🟡（8 ✅ / 2 🟡） | 第 1 章全部 + [[10.1 频率-严重度与纯保费]]、[[10.2 计数模型与 offset]]、[[10.3 严重度模型与 Gamma]]、[[10.8 Tweedie GLM 专章]] | 2 |
| 第 2 节 Transformations | 7 | 🟡（5 ✅ / 2 🟡） | [[2. Transformations]]、[[2.3.1 WOE & IV]]、[[10.4 评分卡与 WOE 落地]] | 2 |
| 第 3 节 Missing Data | 2 | 🟡（1 ✅ / 1 🟡） | [[4. Missing Data]]、[[99.7 共线性缺失值与特征工程]] | 1 |
| 第 4 节 Multicollinearity | 5 | 🟡（3 ✅ / 2 🟡） | [[3. Multicollinearity]]、[[3.4.2 Variance Inflation Factor (VIF)]]、[[3.5.2 SAS VARCLUS]]、[[3.5.4 PCA]] | 2 |
| 第 5 节 Feature Selection | 7 | 🟡（5 ✅ / 2 🟡） | [[5. Dimension Reduction]]、[[5.2 Univariate Selection]]、[[5.3 Multivariate Selection]]、[[1.6.3 Regularization]]、[[7.4. Quantifying Feature Importance]] | 2 |
| 第 6 节 Model Assessment | 6 | 🟡（5 ✅ / 1 🟡） | [[6.1 Data Preparation & Validation]]、[[6.2 Model Diagnosis_Bias-Variance Tradeoff]]、[[6.3 Performance Metrics]]、[[6.4 Diagnostic & Visualization Tools]]、[[99.6 评估指标与数据准备]] | 1 |
| 第 7 节 Bias-Variance & Ensembles | 3 | ✅（3 ✅） | [[6.2 Model Diagnosis_Bias-Variance Tradeoff]]、[[8. Ensemble Learning]]、[[8.3. Bagging (Bootstrap Aggregating)]]、[[8.6 Bagging VS Boosting]]、[[10.1 频率-严重度与纯保费]]、[[10.9 项目实战：法语车险纯保费定价]] | 0 |
| 第 8 节 Random Forest | 7 | 🟡（5 ✅ / 2 🟡） | [[7. Random Forest]]、[[7.2. Building the Forest]]、[[7.4. Quantifying Feature Importance]]、[[99.4 随机森林与特征重要性]] | 2 |
| 第 9 节 GBM & XGBoost | 8 | 🟡（5 ✅ / 3 🟡） | [[8.4.2.2 Gradient Boosting]]、[[8.4. Boosting]]、[[8.6 Bagging VS Boosting]]、[[99.5 Boosting 与集成]] | 3 |
| 第 10 节 STAR Answers | 2 | 🟡（1 ✅ / 1 🟡） | [[03. 答题模板与追问应对]] | 1 |
| **合计** | **57 行 = 55 题（§1–§9）+ 2 个 STAR 框架** | **🟡** | — | **16** |

> **没有 ❌ 题**：55 道题在库内都有落点，只是 16 道「有落点但比 MAGNet 的问法浅或更分散」。
> 数字层的 3 处 ❌（RF `n_estimators` 常用区间、AUC 三档、GBM 有序调参配方）见 §11 与 §13。

## 1. Logistic Regression & GLMs

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 1.1 | What is a logit? | logit 是 odds 的自然对数，把 (0,1) 映到 (−∞,+∞)；逆变换是 sigmoid | ⭐⭐⭐ | [[1.3 Introducing Logistic Regression]]（§2 The Logit Link）；[[1.2 Generalized Linear Models (GLMs)]] | ✅ |
| 1.2 | Write the logistic regression equation and interpret a coefficient | `ln(p/(1−p)) = β₀+β₁x₁+…`；xⱼ 每加 1 单位，log-odds 加 βⱼ、odds 乘 `exp(βⱼ)` | ⭐⭐⭐ | [[1.3 Introducing Logistic Regression]]（§4 Interpreting the Coefficients）；[[99.2 GLM 与逻辑回归]] Q2 | ✅ |
| 1.3 | What is a link function and why do we need one? | GLM 三件套（指数族 + 线性预测子 + link）；link 保证预测落在合法范围，同时保住线性结构 | ⭐⭐⭐ | [[1.2 Generalized Linear Models (GLMs)]]（canonical link 表）；[[1.7 Assumptions]]（§2） | ✅ |
| 1.4 | Which distribution would you pick for an insurance target? | 频率 Poisson + offset；过散布 NB / 带 scale 的 Poisson；严重度 Gamma + log；纯保费 Tweedie + log；二值 Binomial + logit | ⭐⭐⭐ | [[10.1 频率-严重度与纯保费]]、[[10.2 计数模型与 offset]]、[[10.3 严重度模型与 Gamma]]、[[10.8 Tweedie GLM 专章]] | ✅ |
| 1.5 | How do you deal with an unbalanced sample? | 默认不重采样，改判阈值；关键是事件的绝对数量，不是类别比例 | ⭐⭐⭐ | [[1.5 Dealing with Unbalanced Samples]]、[[10.6 采样偏差与拒绝推断]]、[[99.6 评估指标与数据准备]] Q2 | ✅ |
| 1.6 | What is separation? How do you detect and fix it? | 某个（或某组合）预测变量完美预测结局，似然随系数跑向无穷仍不断改善，MLE 不存在 | ⭐⭐⭐ | [[1.6 Estimation Issues & Separation]]、[[1.6.3 Regularization]]、[[99.9 GLM 进阶题库]] Q5 | ✅ |
| 1.7 | Explain a confusion matrix | TP / FP（Type I，误报）/ FN（Type II，漏报）/ TN 四格；precision = TP/(TP+FP)、recall = TP/(TP+FN) | ⭐⭐⭐ | [[6.3 Performance Metrics]]（混淆矩阵 + accuracy/precision/recall/AUC）；[[99.6 评估指标与数据准备]] Q2、Q6 | 🟡（F1 / 特异度 / Type I-II 命名分散在 9.4、99.6；MAGNet 的 1000 份保单手算例库内没有） |
| 1.8 | How do a linear model and a GLM differ? | 线性模型：Y 正态、均值线性、最小二乘或 ML；GLM：任意指数族、某个均值函数（link）线性、只用 ML（IRLS） | ⭐⭐ | [[1.1 Ordinary Least Squares (OLS)]]、[[1.2 Generalized Linear Models (GLMs)]]、[[1.4 Model Fitting]] | ✅ |
| 1.9 | What are the GLM assumptions and how do you check each? | 响应服从所选分布、观测独立、link 尺度线性、无严重共线、无强影响点、同方差（仅线性模型）、设定正确 | ⭐⭐⭐ | [[1.7 Assumptions]]、[[1.1 Ordinary Least Squares (OLS)]]（§4 诊断工具表）、[[3.4.2 Variance Inflation Factor (VIF)]]、[[99.9 GLM 进阶题库]] Q3、Q4 | 🟡（假设清单 ✅；GLM 版「逐条诊断」散在 1.1 / 2 / 3.4.2 / 6.4 / 99.9，缺一张集中表） |
| 1.10 | Compare GLM to RF/GBM | GLM 不自动抓非线性与交互，但单调、可外推、系数可报备；树/GBM 精度更高但不能外推、难报备 | ⭐⭐⭐ | [[1. Logistic Regression & GLMs]]（§1.8 GLM vs GBM）、[[10.1 频率-严重度与纯保费]]（为什么偏爱 GLM）、[[10.9 项目实战：法语车险纯保费定价]] | ✅ |

## 2. Transformations

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 2.1 | When do you need a transformation? | 预测侧：link 尺度上非线性、太偏 / 高基数 / 易受异常值影响；响应侧：GLM 里几乎不用，因为挑分布和 link 就够了 | ⭐⭐ | [[2. Transformations]]（§2.2 When & How to Identify the Need） | ✅ |
| 2.2 | How do you identify which transformation you need? | 先让数据露出形状，再挑能复现它的最简参数形式（empirical logit plot 是主力） | ⭐⭐⭐ | [[2. Transformations]]（§2.2 与 empirical logit plot）、[[5.2 Univariate Selection]] | ✅ |
| 2.3 | Explain Weight of Evidence coding | WoE 用「该箱的 log-odds 贡献」替换分箱后的取值，高基数 / 非线性变量塌缩成一个已在 logit 尺度的数值列 | ⭐⭐⭐ | [[2.3.1 WOE & IV]]、[[10.4 评分卡与 WOE 落地]]、[[99.7 共线性缺失值与特征工程]] Q7 | ✅（系数符号口径见 §11.1） |
| 2.4 | Dummy coding — how, and what goes wrong? | k 个水平变 k−1 个指示列 + 一个参照水平，每个系数是对参照的对比；全 k 个加截距就是 dummy variable trap | ⭐⭐ | [[2. Transformations]]（§2.2 Dummy Coding：高基数三后果 + Greenacre + 替代方案表）、[[1.6 Estimation Issues & Separation]] | ✅ |
| 2.5 | Polynomials and Box-Cox | 多项式简单但项间严重共线、外推极差，超过 2 次优先用样条；Box-Cox 由极大似然选 λ，反变换回均值是有偏的 | ⭐⭐ | [[2. Transformations]]（§5 Box-Cox、§6 Polynomial Terms）、[[04. 公式速查卡]] | ✅ |
| 2.6 | What is capping/flooring and why do it? | Winsorizing：把高于高分位、低于低分位的值换成该分位的值，保留记录但限制单个极端值对拟合的影响 | ⭐⭐ | [[2. Transformations]]（§4 Winsorisation）、[[10.3 严重度模型与 Gamma]]（大额点主导） | 🟡（库内用「α 为总缩尾比例、两端各 α/2」的写法，示例是 5% → 2.5 / 97.5；没写 MAGNet 的 1/99 或 5/95） |
| 2.7 | Splines and GAMs — what are they and when do you reach for them? | 样条是结点处光滑拼接的分段多项式；GAM 是每个预测变量有自己的平滑函数、仍然可加的 GLM | ⭐⭐ | [[2. Transformations]]（§7 Splines、§4 GAM Smoothing） | 🟡（有基函数公式与 GAM 概念；自然三次样条、平滑样条的 GCV 选 λ、形状 / 单调约束、GAM 的交互限制未展开） |

## 3. Missing Data

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 3.1 | What are the missing-data mechanisms? | MCAR 与任何数据无关；MAR 只依赖已观测变量；MNAR 依赖没观测到的值本身——只有 MAR 下插补才站得住 | ⭐⭐⭐ | [[4. Missing Data]]、[[99.7 共线性缺失值与特征工程]] Q3 | ✅ |
| 3.2 | What are the imputation methods and when do you use each? | 删行（仅 MCAR 且缺失极少）/ 删列 / 均值中位数 / 众数或「Missing」单独一类 / 缺失指示变量 + 插补值 / 回归与随机回归 / kNN / MICE + Rubin 规则 / 树模型原生处理 | ⭐⭐⭐ | [[4. Missing Data]]、[[99.7 共线性缺失值与特征工程]] Q4、Q5、[[10.7 SAS 实操速查]]（PROC MI + MIANALYZE） | 🟡（缺 hot-deck；缺「缺失 50–70% 考虑删列」的规则；「缺失指示变量 + 插补值」只在 99.7 Q4 一句；缺「上线打分时缺失的处理规则」；插补的泄漏约束散在 99.6 Q3 / Q9） |

## 4. Multicollinearity

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 4.1 | Define it. | 两个及以上预测变量彼此近似线性组合；共线性是两两特例，多重共线性还包含三个以上变量之间的关系 | ⭐⭐ | [[3. Multicollinearity]]（§1） | ✅ |
| 4.2 | What are its effects? Does it hurt predictions? | 估计仍然无偏，但方差爆炸：推断与解释坏掉，而在训练数据范围内预测几乎不受影响 | ⭐⭐⭐ | [[3. Multicollinearity]]（§2）、[[99.7 共线性缺失值与特征工程]] Q2 | ✅ |
| 4.3 | How do you detect it? | 相关矩阵（只抓两两）、VIF = 1/(1−R²ⱼ)（>5 关注、>10 严重）、条件指数（>10 中度、>30 严重）、症状检查 | ⭐⭐⭐ | [[3.4.2 Variance Inflation Factor (VIF)]]（阈值表 + 条件指数 + SAS COLLIN）、[[3. Multicollinearity]]、[[99.7 共线性缺失值与特征工程]] Q1 | 🟡（阈值与条件指数 ✅；「√VIF 是 SE 的放大倍数」在 3.4.2 讲得更准 —— 见 §11；顺序重算 VIF 的流程在 3.4.2 修复表里，未单列） |
| 4.4 | How do you fix it? | 按 VIF 顺序删变量 / 变量聚类取代表 / PCA 回归 / 惩罚回归（ridge 稳、lasso 选、elastic net 兼顾） | ⭐⭐⭐ | [[3.4.2 Variance Inflation Factor (VIF)]]（修复表）、[[3.5.2 SAS VARCLUS]]、[[1.6.3 Regularization]]、[[3.5.4 PCA]] | ✅ |
| 4.5 | Explain PCA. | 找一组按方差排序的正交基，保留前几个成分替代一堆共线变量 | ⭐⭐ | [[3.5.4 PCA]] | 🟡（机制 / 步骤 / 与 VarClus、LDA 的对比 ✅，含 Kaiser 与累计 EVR；载荷符号任意、PCA 回归失去可解释性的代价只在 3.5.4 局限表里一句） |

## 5. Feature Selection

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 5.1 | Why do feature selection at all? | 五条理由，只有一条和精度有关：采集 / 购买成本、计算时间、可解释性、过拟合、参数精度下降 | ⭐⭐ | [[5. Dimension Reduction]]（§1 为什么降维） | 🟡（成本 / 过拟合 / 精度 / 可解释性 ✅；「报送与公平性审查面更小」「数据源故障暴露更少」两条只在 [[01. 面试画像与考点分布]] 一类冲刺层出现） |
| 5.2 | Walk me through your variable reduction workflow on a wide dataset | 业务与合规先筛 → 数据质量 → **先拆数据** → 单变量粗筛 → 无监督去冗余 → 多变量选择（CV 打分）→ 树模型重要性交叉验证 → 业务复核每个符号 → holdout 与分时段 / 分群稳定性 | ⭐⭐⭐ | [[5. Dimension Reduction]]（§7 策略速查 + Golden Rule）、[[5.3 Multivariate Selection]]、[[2.3.1 WOE & IV]] | 🟡（库内有 4 步版速查 + 「选择必须在 CV 内」的黄金法则；缺完整 9 步顺序，尤其「业务 / 合规先筛」与「多时段 / 分群稳定性复核」） |
| 5.3 | Univariate methods — pros, cons, and what you actually use | 快、可扩展到上千变量、与模型无关、易解释；但忽略联合效应、留冗余、看不见条件关系、多重比较假阳性 | ⭐⭐ | [[5.2 Univariate Selection]]、[[2. Transformations]]（Spearman + Hoeffding 配对表）、[[9.7 多重比较与选择偏差]] | ✅ |
| 5.4 | Multivariate methods and subset selection | 前向 / 后向 / 逐步 / 最优子集；逐步法最大的问题是 p 值与 R² 都被乐观偏、选择不稳定 | ⭐⭐⭐ | [[5.3 Multivariate Selection]]、[[9.7 多重比较与选择偏差]]、[[1.6.3 Regularization]] | ✅（best subset 的 2^p 可行性上限与「leaps and bounds」没写；逐步法的六条批评散在 5.3 §4 与 9.7） |
| 5.5 | How does L1 do selection? Why L1 and not L2? | L1 产生精确的 0，L2 只把系数压小、永远不到 0 | ⭐⭐⭐ | [[1.6.3 Regularization]]、[[99.9 GLM 进阶题库]] Q11、[[99.2 GLM 与逻辑回归]] Q10 | ✅ |
| 5.6 | Would you use a GBM's feature importance to pick features for a GLM? | 会——但只当发现工具，不当选择规则；读 PDP / SHAP 学形状，再把形状显式编码进 GLM | ⭐⭐⭐ | [[7.4. Quantifying Feature Importance]]、[[5. Dimension Reduction]]（§6.1 树模型重要性）、[[2. Transformations]]（树模型是发现工具，GLM 是交付物） | ✅ |
| 5.7 | Selection vs. projection, supervised vs. unsupervised | 选择保留原变量、投影造新变量；VarClus / PCA 无监督，LDA 是有监督的投影对应物 | ⭐⭐ | [[5. Dimension Reduction]]、[[3.5.2 SAS VARCLUS]]、[[3.5.4 PCA]]、[[99.7 共线性缺失值与特征工程]] Q6 | ✅ |

## 6. Model Assessment

| #   | MAGNet 问题                                                   | 一句话答案的要点                                                                   | ⭐   | 深潜笔记                                                                                                                           | 覆盖度                                                                                                            |
| --- | ----------------------------------------------------------- | -------------------------------------------------------------------------- | --- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| 6.1 | How do you split your data, and why three sets?             | 训练集拟合参数、验证集做所有选择、测试集只给一次无偏的泛化读数                                            | ⭐⭐⭐ | [[6.1 Data Preparation & Validation]]、[[99.6 评估指标与数据准备]] Q3、[[10.5 模型评估与监控]]（OOT vs 随机 CV）                                     | ✅（比例口径：库内写训练 60–80%、验证 / 测试各 10–20%，见 §11；按实体分组的完整性在 99.6 / 99.9 有一句）                                          |
| 6.2 | What is cross-validation? Which flavor for which data?      | 让验证角色在 k 折间轮换再取平均，每条记录既参与拟合又参与验证                                           | ⭐⭐⭐ | [[99.6 评估指标与数据准备]] Q4、[[6.1 Data Preparation & Validation]]、[[9.7 多重比较与选择偏差]]、[[99.4 随机森林与特征重要性]] Q2                           | 🟡（分层 / 分组 / 时序 / K 取 5 或 10 / 嵌套 CV ✅；缺「哪种数据用哪种」的集中对照表：LOOCV、重复 k 折、滚动起点只在 99.4 Q2 与 9.7 零星出现）                |
| 6.3 | How do you detect overfitting and underfitting?             | 看训练误差与验证 / 测试误差的差距：两头都高是小差距的欠拟合，训练低验证高是大差距的过拟合                             | ⭐⭐⭐ | [[6.2 Model Diagnosis_Bias-Variance Tradeoff]]、[[99.6 评估指标与数据准备]] Q10、[[6.4 Diagnostic & Visualization Tools]]（学习曲线 / 验证曲线）    | ✅                                                                                                              |
| 6.4 | What is data leakage? Give me examples                      | 训练特征里含了真正打分时拿不到的信息——也包括从自己预处理环节漏进来的信息                                      | ⭐⭐⭐ | [[99.6 评估指标与数据准备]] Q3、Q9、[[6.1 Data Preparation & Validation]]（Mistakes to Avoid）、[[2.3.1 WOE & IV]]（§9 坑）、[[4. Missing Data]] | ✅（目标泄漏 / 预处理泄漏 / 时间泄漏 / 组泄漏 / 重复记录 ✅；「AUC 0.97 先怀疑泄漏」这条自查口径在 05. 高频追问与 99.9 Q18）                               |
| 6.5 | Define the metrics and tell me when you'd use each          | 区分度（AUC / Gini / lift）与校准（log-loss / Brier / 实际 vs 预期）是两件事：定价模型两个都要，分诊只要排序 | ⭐⭐⭐ | [[6.3 Performance Metrics]]、[[10.5 模型评估与监控]]（KS / Gini / 校准 / PSI）、[[99.9 GLM 进阶题库]] Q8、Q12                                    | ✅（MAPE 在 6.3、deviance / AIC / BIC 在 99.9 Q8、PR-AUC 在 05. 高频追问 TOP 30 与 04. 公式速查卡；Brier 与校准曲线在 99.9 Q12 / 10.5） |
| 6.6 | Explain a lift chart and a gain chart. How do you read one? | 按预测值降序切十分位，算每箱实际响应率；lift = 箱内响应率 ÷ 整体响应率，gain 看累计捕获                        | ⭐⭐  | [[6.4 Diagnostic & Visualization Tools]]（§3 Lift/Gain Chart）、[[10.5 模型评估与监控]]（§3 十分位 Lift 表）、[[99.6 评估指标与数据准备]] Q7             | ✅（构造步骤、45° 随机基线、按运营产能读图、非单调是红旗、Lorenz / 双 lift 图都已按 MAGNet 补入）                                                 |

## 7. Bias-Variance & Ensembles

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 7.1 | Explain the bias-variance tradeoff. Write the decomposition | `E[(y − f̂)²] = Bias[f̂]² + Var[f̂] + σ²`；σ² 是任何模型都打不破的地板 | ⭐⭐⭐ | [[6.2 Model Diagnosis_Bias-Variance Tradeoff]]、[[04. 公式速查卡]] | ✅ |
| 7.2 | Bagging vs. boosting vs. stacking | bagging 并行、主要降方差；boosting 顺序、主要降偏差；stacking 用折外预测训元模型、靠多样性两头都降 | ⭐⭐⭐ | [[8. Ensemble Learning]]（三大族对比 + §5 Stacking）、[[8.3. Bagging (Bootstrap Aggregating)]]、[[8.4. Boosting]]、[[8.6 Bagging VS Boosting]]、[[99.5 Boosting 与集成]] Q6、Q12 | ✅ |
| 7.3 | You work at an insurance company. GLM or GBM? | 看交付物是决策还是要报备的费率：进费率的用 GLM 的可解释与单调，内部排序与营销用 GBM 的精度 | ⭐⭐⭐ | [[10.1 频率-严重度与纯保费]]、[[10.9 项目实战：法语车险纯保费定价]]（GLM 主 + GBM 挑战者 + 单调约束）、[[1. Logistic Regression & GLMs]]（§1.8）、[[99.9 GLM 进阶题库]] Q9、Q14 | ✅ |

## 8. Random Forest

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 8.1 | Walk me through the algorithm | 对 bootstrap 样本长深树，每次分裂只在随机抽出的 max_features 个变量里找最优切分，最后取平均 / 投票 | ⭐⭐⭐ | [[7. Random Forest]]、[[7.2. Building the Forest]]、[[7.2.2 Decision Tree]] | ✅ |
| 8.2 | What is OOB error? | 每个 bootstrap 约留下 36.8% 的样本；只用没见过某行的树来预测它，就得到一份免费的验证估计 | ⭐⭐⭐ | [[7.2. Building the Forest]]、[[99.4 随机森林与特征重要性]] Q2、[[8.3. Bagging (Bootstrap Aggregating)]] | ✅（库内同时给了 1/e 推导与「不能替代时间外 holdout」的边界） |
| 8.3 | Key hyperparameters — what they do and how they trade off | n_estimators 越多越好直到趋平、max_depth 控偏差方差、min_samples_leaf 最直接的噪声控制、max_features 是 RF 的标志性旋钮、bootstrap 保留以启用 OOB | ⭐⭐⭐ | [[7. Random Forest]]（§5 超参数表）、[[7.2. Building the Forest]]（§5 Practical Considerations）、[[99.4 随机森林与特征重要性]] Q6 | 🟡（库内说「更多树越好、看 OOB 趋平」并给了默认 `n_estimators=100`，但没写 MAGNet 的常用区间 300–1000，见 §11） |
| 8.4 | How do those hyperparameters interact? | max_features × n_estimators 要成对调；max_depth × min_samples_leaf 基本冗余；深树从加树里获益更多；重类权重配小叶节点最易过拟合；bootstrap=False 且 max_features=p 就是 B 棵一样的树 | ⭐⭐ | [[7. Random Forest]]（超参数之间怎么相互影响 —— 五条已按 MAGNet 补入）、[[99.4 随机森林与特征重要性]] Q6 | ✅ |
| 8.5 | What tuning strategies do you use? | 网格搜索（参数少时）、随机搜索（默认第一轮）、贝叶斯优化（每次拟合都贵时）、逐次减半 / Hyperband（搜索空间大且早期信号便宜） | ⭐⭐ | [[7. Random Forest]]（§6 Tuning Strategies）、[[10.9 项目实战：法语车险纯保费定价]]（12 组网格 × 5 折 CV 的实例）、[[9.7 多重比较与选择偏差]] | 🟡（网格 / 随机 / 贝叶斯 ✅ 且强调要嵌在 CV 内；缺 successive halving / Hyperband 与「随机搜索常胜过网格」的论证） |
| 8.6 | How does RF compute feature importance? | MDI（Gini 重要性）、置换重要性、drop-column、SHAP 四条路，各有偏差 | ⭐⭐⭐ | [[7.4. Quantifying Feature Importance]]、[[99.4 随机森林与特征重要性]] Q3 | ✅ |
| 8.7 | Pros and cons | 开箱精度高、自动抓非线性与交互、不用缩放、免费 OOB、内置重要性；代价是黑箱、模型大、不能外推、MDI 偏向高基数、严重不平衡时被多数类淹没 | ⭐⭐ | [[7. Random Forest]]（§7 Pros & Cons）、[[99.4 随机森林与特征重要性]] Q4、Q5、Q7、[[8.6 Bagging VS Boosting]] | ✅ |

## 9. GBM & XGBoost

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 9.1 | Walk me through gradient boosting | 顺序拟合浅树：每棵拟合当前集成还没学到的部分（负梯度 / 伪残差），再乘一个小学习率加进模型 | ⭐⭐⭐ | [[8.4.2.2 Gradient Boosting]]、[[8.4. Boosting]]、[[99.5 Boosting 与集成]] Q4、Q10 | ✅ |
| 9.2 | How is GBM different from random forest? | RF 是深树并行后平均、主要降方差；GBM 是浅树顺序相加、主要降偏差，且树多了会过拟合 | ⭐⭐⭐ | [[8.6 Bagging VS Boosting]]、[[8.4.2.2 Gradient Boosting]]（§6 与 AdaBoost / GB 对照）、[[99.4 随机森林与特征重要性]] Q7 | ✅ |
| 9.3 | What are the important hyperparameters? | 核心是学习率与树数这一对；再加树深（= 交互阶数）、min_child_weight、subsample / colsample、L1 / L2、monotone_constraints | ⭐⭐⭐ | [[8.4.2.2 Gradient Boosting]]（§7 实用超参数表）、[[99.5 Boosting 与集成]] Q5 | 🟡（数值口径：库内 subsample / colsample 写 0.5–1，MAGNet 写 0.5–0.8；其余一致，见 §11） |
| 9.4 | What's your tuning strategy? | 有序配方：先固定 lr≈0.1 用早停定树数 → 调树复杂度 → 调随机性 → 调正则 → 最后降 lr 到 0.01–0.05 再补树 → 用随机 / 贝叶斯搜索代替全网格 | ⭐⭐ | [[8.4.2.2 Gradient Boosting]]（Workflow + 防过拟合 8 步清单）、[[99.5 Boosting 与集成]] Q5、Q14、[[10.9 项目实战：法语车险纯保费定价]] | 🟡（库内只有一句「先网格搜 (η, M) 再调树与采样参数」；MAGNet 的六步有序配方没有成篇） |
| 9.5 | Why is GBM more sensitive to hyperparameters than RF? | RF 的树彼此独立再平均，错误会互相抵消；GBM 每棵树都拟合前面集成的误差，错误会累积 | ⭐⭐⭐ | [[8.6 Bagging VS Boosting]]、[[8.4.2.2 Gradient Boosting]]（§5 调参成本对比）、[[99.5 Boosting 与集成]] Q5、Q14 | ✅ |
| 9.6 | What does XGBoost add over traditional GBM? | 正则化目标、二阶牛顿优化、原生稀疏 / 缺失处理、加权分位草图的近似分裂、深度优先生长后按 gamma 剪枝 + 工程优化 | ⭐⭐⭐ | [[8.4.2.2 Gradient Boosting]]（XGBoost 相对传统 GBM 加了什么）、[[99.5 Boosting 与集成]] Q11、[[8.4. Boosting]]（现代实现表） | ✅（已按 MAGNet 补入三问：XGBoost 增量、单调约束的监管意义、防过拟合 8 步） |
| 9.7 | How does GBM compute variable importance? | gain 是默认要引用的；cover 次要；weight / frequency 偏向高基数连续变量，别引用；置换与 SHAP 作外部验证 | ⭐⭐ | [[8.4. Boosting]]（Feature Importance 一行）、[[7.4. Quantifying Feature Importance]]、[[99.4 随机森林与特征重要性]] Q3 | 🟡（库内只有「树模型重要性 = MDI / 置换 / SHAP」的 RF 版体系与一句 gain；缺 gain / cover / weight 三件套的取舍对照） |
| 9.8 | How do you keep a GBM from overfitting? | 早停第一（影响力最大）；低学习率配更多树、浅树、min_child_weight、子采样、L1 / L2 与 gamma、CV 决策、单调约束 | ⭐⭐⭐ | [[8.4.2.2 Gradient Boosting]]（防过拟合 8 步清单）、[[99.5 Boosting 与集成]] Q5、Q14 | ✅ |

## 10. STAR Answers

| # | MAGNet 问题 | 一句话答案的要点 | ⭐ | 深潜笔记 | 覆盖度 |
|---|---|---|---|---|---|
| 10.1 | 10a 技术版 STAR（一个方法也能用 STAR 讲：S 什么场景 / T 它做什么 / A 你怎么做 / R 输出与怎么读） | 用 STAR 组织「讲一个方法」的回答，保证覆盖面试官打分的每一块；库内已给 LR 与 RF 两个 worked example | ⭐⭐⭐ | [[03. 答题模板与追问应对]]（技术版 STAR）、[[06. 中英术语对照表]] | ✅ |
| 10.2 | 10b 行为面 STAR（模板 + 10 题题库 + 五个扣分点） | S 一到两句（点名利害）、T 一句（你的责任，不是团队的）、A 最长（含取舍，用「我」）、R 二到三句且必须量化 | ⭐⭐⭐ | [[03. 答题模板与追问应对]]（行为面 STAR 模板 + 3 个假设性示例 + 面试当天清单） | 🟡（模板与示例 ✅；MAGNet 的 10 题题库（每题考什么 / 该挑什么故事）与「五个扣分点」库内没有成表） |

## 11. 数字对照（Numbers to Memorize）

MAGNet 的「Numbers to Memorize」表共 **27 项**（原文说「about 25」）。下表逐项核对库内出处与数值口径；`✅ 一致` 表示数值与口径都相符，`🟡` 表示数值相符但口径 / 范围更宽，`❌` 表示库内没有。

| 数字 | MAGNet 取值 | 库内出处 | 核对结论 |
|---|---|---|---|
| VIF — 关注 / 严重 | > 5 / > 10 | [[3.4.2 Variance Inflation Factor (VIF)]]（阈值表）、[[99.7 共线性缺失值与特征工程]] Q1 | ✅ 一致；且库内额外强调这是经验值、随推断或预测场景可调 |
| √VIF | 标准误被放大的倍数 | [[3.4.2 Variance Inflation Factor (VIF)]]（§4 说明）、[[99.9 GLM 进阶题库]] Q16 | ✅ 一致。库内把这点讲得更完整：VIF 膨胀的是**方差**，SE 只放大 √VIF（VIF=10 → SE ≈ 3.16 倍），并明确点名这是常错点 |
| 条件指数 — 中等 / 严重 | > 10 / > 30 | [[3.4.2 Variance Inflation Factor (VIF)]]（CI 段 + SAS PROC REG 的 COLLIN 选项） | ✅ 一致（本次核对时库内已有：CI > 10 中度、CI > 30 严重，含方差分解比例） |
| Kaiser 准则保留 PC | 相关阵上特征值 > 1 | [[3.5.4 PCA]]（§4 选成分 k 的四条判据） | ✅ 一致（累计 EVR ≥ 80–95%、碎石图、Kaiser、交叉验证四条并列） |
| PCA 累计保留方差 | 80–95% | [[3.5.4 PCA]]（§4 与判据说明表） | ✅ 一致 |
| IV 分档 | <0.02 无用・0.02–0.1 弱・0.1–0.3 中・0.3–0.5 强・>0.5 查泄漏 | [[2.3.1 WOE & IV]]（§5 IV 表 + §8 坑表）、[[10.4 评分卡与 WOE 落地]] | 🟡 分档数值完全一致；差别在 >0.5 的解释：库内写「可疑 / 过拟合风险」并归因于小箱 + 小样本把 IV 抬高（赢家诅咒），而「查泄漏」在库内是另一条坑（WoE 在全量数据上用目标计算）。两处并读即等价 |
| EPV（每变量事件数） | 10–20 | [[1.7 Assumptions]]（§3 大样本）、[[1. Logistic Regression & GLMs]]、[[99.2 GLM 与逻辑回归]] Q11 | ✅ 一致（并标注为经验法则、稀有事件时应更保守） |
| WoE 最小箱占比 | ≥ 5% 样本 | [[2.3.1 WOE & IV]]（粗分类步骤 + 坑表）、[[10.4 评分卡与 WOE 落地]]（分箱工作流） | ✅ 一致（同时要求无零计数、有序变量上 WoE 尽量单调） |
| Winsorize 切点 | 1 / 99 或 5 / 95 分位 | [[2. Transformations]]（§4 Winsorisation） | 🟡 口径不同：库内用「α 为总缩尾比例、两端各 α/2」的一般式，示例是 5% → 2.5 / 97.5 分位，没有出现 1/99 或 5/95 这两个具体切点 |
| 划分比例 | 60/20/20 或 70/15/15 | [[6.1 Data Preparation & Validation]]（标准划分表）、[[99.6 评估指标与数据准备]] Q3 | 🟡 口径更宽：库内写训练 60–80%、验证与测试各 10–20%（并补「数据少时用 CV 代替固定验证集」）。兼容但不给固定数字 |
| CV 的 k | 5 或 10 | [[99.6 评估指标与数据准备]] Q4、[[10.9 项目实战：法语车险纯保费定价]]（5 折实例） | ✅ 一致（并解释了「K 越大不一定越好」） |
| bootstrap 未抽中比例 | 1/e ≈ 36.8% | [[7.2. Building the Forest]]（~37%）、[[99.4 随机森林与特征重要性]] Q2（36.8% + 推导）、[[8.3. Bagging (Bootstrap Aggregating)]] | ✅ 一致 |
| AUC — 随机 / 有用 / 强 | 0.5 / > 0.7 / > 0.8 | [[04. 公式速查卡]]（AUC 行）、[[10.5 模型评估与监控]]（Gini ↔ AUC 对照）、[[5.2 Univariate Selection]]（AUC > 0.7 视为 good） | 🟡 部分一致：0.5 = 随机 ✅、> 0.7 有用 ✅；「> 0.8 强」没有直接出现，但可由库内的 Gini 0.6 ≈ AUC 0.80 对照与 KS / Gini 分档推出 |
| Gini 与 AUC | Gini = 2·AUC − 1 | [[10.5 模型评估与监控]]（§2）、[[04. 公式速查卡]]、[[05. 高频追问 TOP 30]] | ✅ 一致（库内还提醒与决策树的 Gini 不纯度同名不同物） |
| RF max_features 默认 | √p（分类）/ p/3（回归） | [[7.2. Building the Forest]]（§5）、[[7. Random Forest]]（默认参数）、[[99.4 随机森林与特征重要性]] Q1 | ✅ 一致（并给出「小 m 增多样性、弱单树」的机制解释） |
| RF n_estimators 常用 | 300–1000（更多不会过拟合） | 库内只有默认值 100（[[7. Random Forest]] 实用工作流）与「B 越大越好、看 OOB 趋平」（[[99.4 随机森林与特征重要性]] Q4、Q6） | ❌ 缺具体常用区间：300–1000 这个数字库内没有；「更多树不会过拟合」✅ 已有 |
| GBM learning_rate | 0.01–0.3 | [[8.4.2.2 Gradient Boosting]]（§7 超参数表）、[[99.5 Boosting 与集成]] Q5 | ✅ 一致 |
| GBM max_depth | 3–8 | [[8.4.2.2 Gradient Boosting]]（浅树深度 3–8；防过拟合清单写 3–6）、[[99.5 Boosting 与集成]] Q5 | ✅ 一致（库内的 3–6 与 MAGNet 自己防过拟合清单里的 3–6 对应） |
| GBM subsample / colsample | 0.5–0.8 | [[8.4.2.2 Gradient Boosting]]（Subsample rows 0.5–1、Colsample 0.5–1）、[[99.5 Boosting 与集成]] Q5 | 🟡 范围更宽：库内两处都写 0.5–1，MAGNet 收窄到 0.5–0.8；面试按 MAGNet 更稳，但要知道库里口径更宽 |
| 学习率 ↔ 树数 | 学习率减半 → 树数约翻倍 | [[99.5 Boosting 与集成]] Q5、[[8.4.2.2 Gradient Boosting]]（η 与 M 成反比，η=0.1 配几百棵、0.01 配几千棵） | ✅ 一致 |
| GBM 树深 ↔ 交互阶数 | depth d → 最多 d 阶交互；depth 1 = 纯加性 | [[8.4.2.2 Gradient Boosting]]（树深 = 交互阶数） | ✅ 一致（已按 MAGNet 补入） |
| Bagging 方差公式 | ρσ² + (1−ρ)σ²/B | [[8.3. Bagging (Bootstrap Aggregating)]]（σ²/B + (B−1)/B·ρσ²，代数等价）、[[8.6 Bagging VS Boosting]]、[[04. 公式速查卡]]、[[99.4 随机森林与特征重要性]] Q5 | ✅ 一致（并接上了「ρσ² 是地板 → 所以 RF 要子采样特征」这条链） |
| 偏差方差分解 | E[(y−f̂)²] = Bias² + Var + σ² | [[6.2 Model Diagnosis_Bias-Variance Tradeoff]]、[[04. 公式速查卡]] | ✅ 一致 |
| AIC / BIC | −2logL + 2k / −2logL + k·ln(n) | [[99.9 GLM 进阶题库]] Q8、[[04. 公式速查卡]] | ✅ 一致（并补了 ΔAIC ≤ 2 等价、> 10 淘汰等经验口径与「不可跨分布 / 跨 link 比较」的前提） |
| Tweedie 纯保费幂 p | p ≈ 1.5 | [[10.1 频率-严重度与纯保费]]、[[10.8 Tweedie GLM 专章]]、[[10.9 项目实战：法语车险纯保费定价]] | 🟡 不是冲突，是「常数 vs 超参数」：库内明确 p 是数据依赖的超参数（1 < p < 2 为复合 Poisson-Gamma），要在 (1,2) 上搜索；10.9 的 678,013 份保单实测 CV 选出 p = 1.9。1.5 是纯保费场景的常识起手值。详见 §11.1 |
| 病例对照截距校正 | β₀ − ln(r₁/r₀) | [[1.5 Dealing with Unbalanced Samples]]（截距校正公式）、[[10.6 采样偏差与拒绝推断]]（ln(1/99) = −4.595 算例）、[[99.2 GLM 与逻辑回归]] Q8、[[99.6 评估指标与数据准备]] Q8 | ✅ 一致（库内等价写法 logit(pop) = logit(sample) − log(π₁/π₀)，并强调 offset 与截距校正不要重复用） |
| 分箱良好的 WoE 变量系数 | ≈ 1.0 | [[2.3.1 WOE & IV]]（§3 与理想系数）、[[10.4 评分卡与 WOE 落地]]（理想系数 −1 与 +1 的适用条件） | ⚠️ 方向问题，见 §11.1 —— 两处都对，只是 WoE 定义方向相反 |

### 11.1 三个必须讲清的口径差异

**① WoE 系数是 +1 还是 −1（重点，必须说清方向，不是矛盾）**

- MAGNet 把 WoE 定义成 **ln(%总事件 / %总非事件)** ——「事件多」的箱 WoE 为正，所以「丢进 logistic 系数应接近 **+1.0**」。
- 库内统一口径是 **WoE = ln(%Good / %Bad)**（Good = 非事件、Bad = 事件），**正 WoE = 低风险**；在这个方向下，事件的 log-odds 与 WoE 反向，所以理想系数是 **−1**（[[2.3.1 WOE & IV]] §3：`ln(P(Y=1)/P(Y=0)) = β₀ − Σ WoE`，并引 `WoE = ln(P(x|Y=0)/P(x|Y=1))` 说明负号来源）。
- [[10.4 评分卡与 WOE 落地]] 把这条讲全了：**预测「Bad」的 log-odds 时理想系数是 −1；预测「Good」的 log-odds 时理想系数是 +1**；实务上拟合出来通常是负系数、绝对值接近 1，符号与业务方向不一致就整箱重做。
- **结论**：MAGNet 的 +1 与库内的 −1 是**同一件事在两种 WoE 定义下的两个写法**，判断标准是「定义方向与系数符号必须配套」。面试里的加分说法就是先声明自己的定义方向（本库统一 ln(%Good/%Bad)），再给符号。
- **一处待修**：[[2.3.1 WOE & IV]] §9 坑表里还残留一句「再标准化会破坏『βⱼ = 1 时证据可直接相加』的解释」，与同篇 §3 的 −1 不一致（保留 +1 是因为预测的是 Good 的 log-odds，但原文没写明），属于措辞残留，建议改成「βⱼ = −1」。

**② Tweedie 的 p：1.5 与 1.9 不冲突**

- MAGNet 说纯保费用 Tweedie、**p ≈ 1.5**；库内 [[10.9 项目实战：法语车险纯保费定价]] 的实测 CV 选出 **p = 1.9**（[[10.8 Tweedie GLM 专章]] 记录了 12 组 (power, alpha) 网格与 `power=1.9, alpha=1.0, CV 34.0823 ± 0.5148`）。
- 正确解释：**p 是数据依赖的超参数，不是固定常数**。1 < p < 2 时 Tweedie 是复合 Poisson-Gamma（p→1 趋 Poisson、p→2 趋 Gamma）；1.5 是「零点质量与连续正尾各占一半」这个直觉最常用的起手值，1.9 表示这批数据的方差几乎随均值平方增长（严重度重尾主导）。
- 还要主动交代两个诚实口径（10.8 已写）：**不同 p 的 deviance 不可直接比较**，所以网格里必须固定一个 p 打分；且 CV 前几名差距小于 1 个标准误，正确说法是「数据支持 p 落在 1.7–1.9 这一段」，不是「p = 1.9 显著更好」。

**③ 其余数值口径差（数值不冲突，范围更宽或换写法）**

- **Winsorize 切点**：MAGNet 给 1/99 或 5/95；库内是 α/2 写法（5% 缩尾 → 2.5 / 97.5 分位），没有列出这两个具体切点。
- **划分比例**：MAGNet 给 60/20/20 或 70/15/15；库内给训练 60–80%、验证 / 测试各 10–20%。
- **GBM subsample / colsample**：MAGNet 0.5–0.8；库内两处写 0.5–1。
- **AUC 三档**：MAGNet 0.5 / > 0.7 / > 0.8；库内只有 0.5 = 随机、0.7–0.8 可接受与 Gini ↔ AUC 对照（Gini 0.6 ≈ AUC 0.80）。
- **IV > 0.5 的原因**：MAGNet 说「查泄漏」；库内说「可疑 / 过拟合（小箱 + 小样本抬高 IV）」，泄漏则单列为「WoE 用全量数据计算」那条坑。
- **RF n_estimators**：MAGNet 给常用区间 300–1000；库内只有默认 100 与「看 OOB 趋平」。

## 12. Cross-Topic Connections 的库内落点

MAGNet 给的 10 条「两个主题其实是同一个想法」，逐条落到库内笔记：

1. **准完全分离（§1）与 WoE 无定义（§2）是同一个问题**（某个类别水平零事件）：[[1.6 Estimation Issues & Separation]]（准完全分离的识别与补救）＋ [[2.3.1 WOE & IV]]（§8 零计数 → 0.5 平滑 / 并箱）。
2. **高基数是 §1 通向 §2 的桥**（分离正是 WoE 与水平合并存在的理由）：[[2. Transformations]]（Dummy Coding 的高基数三后果 + Greenacre）＋ [[2.3.1 WOE & IV]] ＋ [[1.6 Estimation Issues & Separation]]。
3. **多重共线性（§4）是特征选择（§5）存在的理由，惩罚回归同时出现在两边**（ridge 治共线、lasso 做选择、elastic net 两者兼）：[[3. Multicollinearity]] ＋ [[1.6.3 Regularization]] ＋ [[5. Dimension Reduction]]。
4. **PCA 出现两次**（§4 的共线修复 / §5 的特征投影），VarClus 是它的选择对应物：[[3.5.4 PCA]] ＋ [[3.5.2 SAS VARCLUS]] ＋ [[5. Dimension Reduction]]。
5. **Spearman + Hoeffding 既是 §5 的筛选工具，也是 §2 的变换检测工具**（低 Spearman 高 D = 该变换，不是该删）：[[2. Transformations]]（§2.2 识别手段表）＋ [[5.2 Univariate Selection]] ＋ [[9.5 相关、协方差与相关不等于因果]]。
6. **偏差方差（§7）是理论、过 / 欠拟合（§6）是诊断、正则与超参（§8 / §9）是控制**：[[6.2 Model Diagnosis_Bias-Variance Tradeoff]] ＋ [[99.6 评估指标与数据准备]] Q10 ＋ [[1.6.3 Regularization]] ＋ [[8.4.2.2 Gradient Boosting]]。
7. **bagging 的方差公式（§7）解释了 max_features（§8）**（ρσ² 地板是 RF 子采样特征的唯一理由）：[[8.3. Bagging (Bootstrap Aggregating)]] ＋ [[7. Random Forest]] ＋ [[99.4 随机森林与特征重要性]] Q1、Q5。
8. **泄漏（§6）是 §2 / §3 / §5 造出来的**（WoE、插补、capping、特征选择都是监督的参数学习步骤）：[[99.6 评估指标与数据准备]] Q9 ＋ [[6.1 Data Preparation & Validation]] ＋ [[2.3.1 WOE & IV]]（§9 坑）＋ [[4. Missing Data]]。
9. **「缺失本身有信息」（§3）与 WoE 的缺失箱（§2）是同一个答案**（别插补，让 missing 自成一层、有自己的 log-odds）：[[4. Missing Data]] ＋ [[2.3.1 WOE & IV]] ＋ [[10.4 评分卡与 WOE 落地]] ＋ [[99.7 共线性缺失值与特征工程]] Q4。
10. **GLM vs GBM（§1、§7）每次都同一结论**（树做发现、把形状编码进你真正要交付的模型）：[[1. Logistic Regression & GLMs]] §1.8 ＋ [[10.1 频率-严重度与纯保费]] ＋ [[10.9 项目实战：法语车险纯保费定价]] ＋ [[7.4. Quantifying Feature Importance]]。

## 13. 本次发现的库内缺口

严重度：**高** = 面试可能被追到答不出；**中** = 只能答半截、细节要自己补；**低** = 影响小或只是口径不统一。

| 缺口 | 严重度 | 已补到哪篇 | 状态 |
|---|---|---|---|
| 混淆矩阵完整指标族（§1.7：F1 / 特异度 / Type I-II 命名）分散在 6.3、9.4、9.6、99.6；MAGNet 的「1,000 份保单 / 50 例欺诈 / 命中 80 / 真欺诈 30」手算例库内没有 | 中 | 6.3 有基础四格与 accuracy / precision / recall / AUC；99.6 Q2、Q6 有 F1 与权衡；9.4 有 Type I/II；02. 三条学习路径 指向 MAGNet 本身取手算例 | 待补（一张表 + 一个算例） |
| GLM 假设没有「一条假设对一条诊断」的集中表（§1.9）：deviance 残差 Q-Q、残差 LOESS、Cook 距离 / 杠杆、link test、实际 vs 预期 | 中 | 1.1（OLS 版诊断表）、1.7（假设清单）、2. Transformations（LOESS / empirical logit）、3.4.2（VIF / CI）、99.9 Q3、Q4（deviance / 过散布）、10.5（校准） | 待补（拆成 GLM 版集中表） |
| winsorize 的常用切点（§2.6：1/99、5/95）与「上限从训练集学、是全量数据的百分比就是泄漏」这条 TRAP 未写明 | 低 | 2. Transformations §4 有 α/2 通式与 5% 例；capping 的泄漏 TRAP 已按 MAGNet 补入同节 | 部分已补（缺具体切点） |
| 样条与 GAM 深度不足（§2.7）：自然三次样条、平滑样条的 GCV 选 λ、形状 / 单调约束、GAM 不自动抓交互、GAM 在 GLM 与 GBM 之间的定位 | 中 | 2. Transformations §7 有基函数公式、§4 有 GAM 概念与代码骨架 | 待补 |
| 插补菜单缺 3 项（§3.2）：hot-deck、「缺失 50–70% 考虑删列」的规则、「缺失指示变量 + 插补值」作为独立方法；另缺「上线打分时缺失必须有确定规则」 | 中 | 4. Missing Data 有完整案例 / 均值 / 回归 / 随机回归 / kNN / MICE / MNAR 三法；99.7 Q4 有一句「缺失指示 + 缺失当一类」；99.6 Q3、Q9 有插补泄漏 | 待补 |
| 「为什么选变量」的五条里（§5.1），成本、报送 / 审计、公平性审查面、数据源故障暴露这几条库内只有部分 | 低 | 5. Dimension Reduction §1 有 5 条（偏过拟合 / 噪声 / 性能 / 可解释 / 算法可用性） | 待补（补合规与运维两条） |
| 宽表变量筛减的完整 9 步顺序未成篇（§5.2：业务与合规先筛 → 数据质量 → 先拆数据 → 单变量 → 无监督去冗余 → 多变量 → 树模型交叉验证 → 业务复核 → holdout 与分时段 / 分群稳定性） | 中 | 5. Dimension Reduction §7 有 4 步速查 + 「选择必须在 CV 内」的黄金法则；99.7 Q8 有 filter / wrapper / embedded 分类 | 待补 |
| 交叉验证的「哪种数据用哪种」集中对照表缺（§6.2：k 折 / 分层 / LOOCV / 重复 k 折 / GroupKFold / 时序滚动 / 嵌套） | 中 | 99.6 Q4（分层 / 分组 / 时序坑 / K = 5 或 10 / 1-SE）、9.7（嵌套 CV 与选择偏差）、99.4 Q2（LOOCV 一句） | 待补 |
| RF `n_estimators` 的常用区间 300–1000 未写（§8.3） | 低 | 7. Random Forest（默认 100 + 更多树更好的直觉）、99.4 Q4、Q6（B 越大越好、看 OOB 趋平） | 待补（补一个区间即可） |
| 调参策略缺 successive halving / Hyperband（§8.5），也缺「随机搜索为何常胜过网格」的论证 | 低 | 7. Random Forest §6（网格 / 随机 / 贝叶斯三种） | 待补 |
| GBM 有序调参配方未成篇（§9.4：先把 lr 定 0.1 用早停定树数 → 复杂度 → 随机性 → 正则 → 降 lr 再补树 → 随机 / 贝叶斯搜索） | 中 | 8.4.2.2 §7 只有一句「先网格搜 (η, M) 再调树与采样」；10.9 有 12 组网格 × 5 折的真实实例 | 待补 |
| GBM 重要性 gain / cover / weight 三件套的取舍对照缺失（§9.7） | 低 | 8.4. Boosting 一行「gain, split count」；7.4 与 99.4 Q3 是 RF 版（MDI / 置换 / SHAP） | 待补 |
| 行为面 STAR 的 10 题题库与五个扣分点未成表（§10.2：每题考什么 / 该挑什么故事） | 中 | 03. 答题模板与追问应对 有模板 + 3 个假设性示例 + 面试当天清单 | 待补 |
| GBM `subsample` / `colsample` 数值口径库内 0.5–1、MAGNet 0.5–0.8 | 低 | 8.4.2.2 §7、99.5 Q5 | 待统一口径 |
| [[2.3.1 WOE & IV]] §9 坑表残留「βⱼ = 1 时证据可直接相加」，与同篇 §3 的「理想系数 −1」自相矛盾（+1 只在预测 Good 的 log-odds 时成立，原文没写明） | 低 | 2.3.1 §3、10.4（−1 与 +1 的适用条件） | 待修（改一处措辞） |
| 已确认**不是**缺口（本次核对在库内）：条件指数 > 10 / > 30 与 SAS COLLIN、Kaiser 特征值 > 1、RF 超参数交互五条、XGBoost 增量与单调约束、防过拟合 8 步清单、WoE 的 dummy / Greenacre、识别变换的 6 种手段、lift / gain 的批判性读图、截距校正 | — | 3.4.2、3.5.4、7. Random Forest、8.4.2.2、2. Transformations、6.4、1.5、10.6 | 已补（无需再动） |
