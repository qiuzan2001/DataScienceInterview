---
title: "MAGNet 面试真题全解"
aliases: [MAGNet, 面试真题, StateFarm 真题, magnet-ds-modeling-interview-prep]
chapter: "00"
tags: [statefarm, interview, magnet, spine, 真题]
priority: "⭐⭐⭐"
status: 完成
updated: 2026-09-16
source: "IMG_9774–IMG_9808 共 35 张照片（本人拍摄）OCR 恢复；原始导出见 /Users/qiuzan/Downloads/Pic/"
---

# MAGNet 面试真题全解

> [!important] **这是全库的主线（spine）**——真正的面试骨架。
> 结构：**问题 → 先说的那一句（引用行）→ 追问时才展开的要点 → ⚠️ TRAP（会被扣分的错答）→ 🔗 FOLLOW-UP（几乎必来的下一问）**
>
> | 我要… | 去哪 |
> |---|---|
> | 逐题对照深潜 | **[[MAGNet 答案对照表]]** |
> | 只看考点权重 | [[优先级矩阵]] |
> | 一小时速览 | [[04. 公式速查卡]] + [[05. 高频追问 TOP 30]] |
> | 逐题自测 | [[99.1 题库总览]] |
>
> 📌 **本文件是照片原文的忠实恢复**（未对技术观点作修订），仅在末尾「导入说明」标注了两处由 OCR 片段恢复、一处由通用知识补写的内容。
> 与库内其他笔记的技术口径若有出入，以本文件为准并另行核验。

---

<!-- MAGNET-BODY-START -->
# MAGNet / DS Modeling Interview Prep

Read straight through. Every section is the question you'll be asked → the answer to give. The quoted line under each question is the sentence to lead with; the bullets are what you say if they keep pulling. TRAP marks the mistake that gets people dinged. FOLLOW-UP marks the question that almost always comes next.

Companion file: magnet-study-guide.md - the resource/link index for going deeper.

## Contents

| # | Topic | Core questions |
| --- | --- | --- |
| 1 | [Logistic Regression & GLMs](#1-logistic-regression--glms) | logit, link functions, imbalance, separation, confusion matrix |
| 2 | [Transformations](#2-transformations) | when/how to transform, WoE & IV, dummy coding, Box-Cox, capping, splines/GAM |
| 3 | [Missing Data](#3-missing-data) | MCAR/MAR/MNAR, imputation menu, leakage |
| 4 | [Multicollinearity](#4-multicollinearity) | effects, VIF, PCA, VarClus, ridge vs lasso |
| 5 | [Feature Selection](#5-feature-selection) | why reduce, the workflow, Spearman + Hoeffding, stepwise problems, L1 |
| 6 | [Model Assessment](#6-model-assessment) | splits, CV, leakage, every metric, lift/gain |
| 7 | [Bias-Variance & Ensembles](#7-bias-variance--ensembles) | the decomposition, bagging vs boosting vs stacking, GLM vs GBM |
| 8 | [Random Forest](#8-random-forest) | algorithm, OOB, hyperparameters + interactions, importance |
| 9 | [GBM & XGBoost](#9-gbm--xgboost) | algorithm, tuning recipe, XGBoost advantages, importance |
| 10 | [STAR Answers](#10-star-answers) | technical STAR, behavioral bank, worked examples |

Then: [Numbers to memorize](#numbers-to-memorize) · [Cross-topic connections](#cross-topic-connections) · [Self-test](#self-test-no-answers)

## 1. Logistic Regression & GLMs

### Q: What is a logit?

> 📖 **对应讲解**：[[1.3 Introducing Logistic Regression]]

> The logit is the natural log of the odds, and it's the transformation that lets us model a bounded probability with an unbounded linear predictor.

> 💬 **中文精讲**：这题看着像考定义，其实是在考**「为什么需要变换」这个动机**。答案骨架是三步：`p → odds → log-odds`，每一步都把取值范围放大一次，最后落到 `(−∞, +∞)`，正好与线性预测子 `Xβ` 的值域对齐。
> 面试官大概率会追问「凭什么用 logit 而不是别的变换」——那就要说到它是二项分布的 **canonical link（典则联系函数）**，以及给出的是 odds ratio 这种业务能懂的解释。

- If p = P(Y=1), then odds = p/(1-p), and logit(p) = ln(p/(1-p)).

- It maps (0,1) → (-∞, +∞). That's the whole point: Xβ can be any real number, but a probability can't, so we model the unbounded transform of p instead of p itself.

- Inverse (the logistic / sigmoid): p = 1/(1 + e^(-Xβ)) = e^(Xβ)/(1 + e^(Xβ)).

### Q: Write the logistic regression equation and interpret a coefficient.

> 📖 **对应讲解**：[[1.3 Introducing Logistic Regression]]

> `ln(p/(1-p)) = β₀ + β₁x₁ + ... + βₖxₖ`. A one-unit increase in xⱼ adds βⱼ to the log-odds, which means it multiplies the odds by exp(βⱼ), holding everything else fixed.

> 💬 **中文精讲**：这题不是让你推公式，而是考「系数怎么念给业务听」。骨架：先写 `ln(p/(1-p)) = Xβ`，再说 xⱼ 加一个单位 → log-odds 加 βⱼ → odds 乘 exp(βⱼ)，收尾必须补上「其他变量不变」这一句。
> 最容易掉坑的是把 β 说成「概率的改变量」：同一条 S 曲线上，同一个 β 在 p ≈ 0.5 附近能推动很多、在 0.02 附近几乎推不动。被追问「想要概率效应」时，答在给定基础率处报**边际效应（marginal effect）**，或者干脆给两个预测概率。

- exp(βⱼ) is the odds ratio. exp(β) = 1.25 → a one-unit increase raises the odds by 25%.

- For a dummy variable, exp(β) is the odds ratio versus the reference level.

- **TRAP:** saying "βⱼ is the change in probability." It is not. The change in probability depends on where you sit on the S-curve - the same β moves p a lot near 0.5 and almost nothing near 0.02. If they want a probability effect, quote a marginal effect at a stated base rate, or just show two predicted probabilities.

### Q: What is a link function and why do we need one?

> 📖 **对应讲解**：[[1.2 Generalized Linear Models (GLMs)]]

> A GLM has three pieces - a response distribution from the exponential family, a linear predictor η = Xβ, and a link function g that connects them via g(E[Y]) = η. The link is what keeps predictions in the valid range while letting the predictors act linearly.

> 💬 **中文精讲**：考的是你能不能把 GLM 讲成「三件套」而不是一个公式：来自指数族的响应分布、线性预测子 `η = Xβ`、把两者接起来的 link `g(E[Y]) = η`。骨架就是这三件套，再加三条理由：值域（概率落在 [0,1]、计数和严重度必须为正）、保住线性结构、业务上的乘法结构。
> 最能加分的是第三条：log link 让模型在原尺度上变成乘法 `μ = e^β₀ · e^β₁x₁ · …`，这正是「基础费率 × 各因子」的评级计划结构，也是 log link 在定价里占主导的真正原因。追问几乎必到 canonical link 表（Tweedie 用 log）。

Three reasons we need it:

1. Range. Probabilities must land in [0,1]; counts and claim severities must be positive. The link enforces that automatically instead of us hoping Xβ behaves.

2. Linearity where we want it. The relationship is additive on the link scale, so we can keep a linear predictor and all the machinery that goes with it.

3. Business structure. A log link makes the model multiplicative on the original scale: `μ = e^(β₀) · e^(β₁x₁) · e^(β₂x₂) · ...`. That is exactly how an insurance rating plan is built - base rate × factor x factor - which is why log link dominates in pricing.

Canonical links to have memorized:

| Distribution | Canonical link | Used for |
| --- | --- | --- |
| Normal | identity | continuous, symmetric |
| Binomial | logit | binary / proportion |
| Poisson | log | counts, frequency |
| Gamma | inverse (but log in practice) | severity, positive-skewed |
| Inverse Gaussian | 1/μ² | heavier-tailed severity |
| Tweedie | log | pure premium / loss cost |

### Q: Which distribution would you pick for an insurance target?

> 📖 **对应讲解**：[[1.2 Generalized Linear Models (GLMs)]] ・ [[10.1 频率-严重度与纯保费]] ・ [[10.8 Tweedie GLM 专章]]

> 💬 **中文精讲**：纯对号题，考的是「目标形状 → 分布」的映射熟不熟：频率 → Poisson + offset、过散布 → 负二项（negative binomial, NB）、严重度 → Gamma + log、纯保费 → Tweedie + log、二值 → Binomial + logit。逐行念、每行给一句形状理由（计数、方差大于均值、正偏、零点质量加连续正尾、0/1）就够。
> 最该主动送出去的是 offset 那句：频率模型把 `log(exposure)` 当 offset（系数固定为 1）而不是预测子，模型才是「每单位暴露的费率」。被追问 Tweedie 的 p 时，要说是 `1 < p < 2` 上 CV 调的超参数而不是常数（见本题的编辑注）。→ 深潜：`10.8 Tweedie GLM 专章`

| Target | Distribution | Why |
| --- | --- | --- |
| Claim frequency (count per exposure) | Poisson, log link, exposure as offset | counts, variance ≈ mean |
| Frequency, overdispersed | Negative Binomial, or Poisson with a scale parameter | variance > mean |
| Claim severity (avg cost per claim) | Gamma, log link | positive, right-skewed, variance ∝ μ² |
| Pure premium / loss cost in one step | Tweedie, log link, power p ≈ 1.5 | point mass at zero + continuous positive tail |
| Binary (lapse, conversion, fraud flag) | Binomial, logit | 0/1 |

- The offset point is worth volunteering: for frequency you model counts with log (exposure) as an offset (coefficient fixed at 1), not as a predictor, so the model is a rate per exposure.

- Tweedie is the one that impresses. It handles the fact that most policies have zero loss and the rest have a continuous positive amount, so you avoid fitting frequency and severity separately.
  > 📝 **[编辑注 2]** `p ≈ 1.5` 是**常识起手值，不是固定常数** —— `p` 是 `1 < p < 2` 区间上**用交叉验证调的超参数**。实测例：法国车险 678,013 份保单的网格搜索选出 **p = 1.9**（越靠近 2 越像 Gamma）。面试答法是「我把 `p` 当超参数、和正则强度一起做 CV」。深潜见 `10.8 Tweedie GLM 专章`。

### Q: How do you deal with an unbalanced sample?

> 📖 **对应讲解**：[[1.5 Dealing with Unbalanced Samples]]

> My default is to not rebalance the data - I fix the decision threshold instead. What matters isn't the ratio, it's the absolute number of events.

> 💬 **中文精讲**：考的是「你到底动不动数据」这个判断。引用句先定调——默认不重采样、改决策阈值，真正重要的是事件的**绝对数量**而不是类别比例。骨架按顺序六条：什么都不做（EPV 10–20 就够，50 万行里 500 个事件完全能用）→ 改阈值 → 类权重 → 采样后校截距 → SMOTE → Firth 惩罚似然。
> 一句话判断标准：要**校准概率**（定价、期望损失这类要乘金额的）就别动数据，因为重采样会扭曲预测概率；只要**排序**（分诊、推荐名单）就完全不用管比例。追问几乎必到「采样后系数还有效吗」——病例对照抽样下斜率一致、只有截距要减去 `ln(r₁/r₀)`，这是很硬的一个知识点。

Say these in order:

1. Often you do nothing. Logistic regression needs enough events, not a balanced ratio. Rule of thumb: 10-20 events per predictor (EPV). 500 events out of 100,000 rows is a 0.5% rate and perfectly workable.

2. Fix the threshold, not the data. Ranking metrics - AUC, Gini, lift - are unaffected by class balance. Pick the cutoff from the business cost matrix (cost of a false positive vs. a missed event) or from the operating capacity (how many claims can the SIU team actually investigate?).

3. Class weights - upweight the minority class (`class_weight`='balanced'). Equivalent to oversampling but without duplicating rows.

4. Under/oversampling - if you do it, correct the intercept. Logistic coefficients are consistent under case-control sampling except the intercept, so `β₀_corrected = β₀_sampled − ln(r₁/r₀)` where r₁, r₀ are the sampling rates for events and non-events. Slopes are fine as-is. This is a strong thing to know.

5. SMOTE / synthetic minority oversampling - generates synthetic minority points along lines between neighbors. Caveats: can invent implausible records, and it must be applied inside each CV training fold only - SMOTE before splitting is leakage.

6. Firth's penalized likelihood - bias-reduced logistic regression, built for rare events and separation. The right tool when events are in the dozens.

The framing that wins the question:

- If I need calibrated probabilities (pricing, expected-loss, anything multiplied by a dollar amount), I avoid resampling - it distorts predicted probabilities - and use weights plus an intercept correction, or recalibrate afterward (Platt scaling / isotonic regression).

- If I need ranking (triage, referral, marketing lists), balance is irrelevant; I leave the data alone and tune the threshold.

**TRAP:** reporting accuracy on an imbalanced problem. See the worked example below.

### Q: What is separation? How do you detect and fix it?

> 📖 **对应讲解**：[[1.6 Estimation Issues & Separation]]

> Separation means a predictor (or combination) perfectly predicts the outcome, so the maximum likelihood estimate doesn't exist - the likelihood keeps improving as the coefficient runs to infinity.

> 💬 **中文精讲**：考「你撞过这个坑没有」。骨架：先给定义（某个预测变量或它的组合能完美预测结局，似然随系数跑向无穷仍在改善，所以最大似然估计根本不存在），再分清完全分离与准完全分离——后者几乎总是某个高基数类别的某一层全事件或全非事件，邮编、车型、职业码、agent ID 最常见。
> 识别信号要一口气报出：系数大得离谱却标准误巨大、Wald 检验不显著、收敛警告或撞最大迭代、SAS 直接打印「Quasi-complete separation」，再加上人工查每个类别与目标的交叉表找零格。追问会顶到「加 L2 不就好了」——用 CV 调 λ 时常常选出 λ = 0，惩罚只是缓解；真正给出有限、可做推断的估计是 Firth 或贝叶斯先验，而且分离主要破坏推断、不破坏决策边界。→ 深潜：`1.6 Estimation Issues & Separation`

- Complete separation: a linear combination of predictors splits 0s from 1s perfectly.

- Quasi-complete separation: perfect prediction in part of the space - almost always a categorical level (or a cross-tab cell) with all events or all non-events. Extremely common with high-cardinality categoricals: ZIP code, vehicle make/model, occupation code, agent ID.

How you know it's happening:

- Coefficients of absurd magnitude (|β| of 10-20+ on the log-odds scale) with enormous standard errors.

- Wald tests non-significant despite a huge point estimate (because the SE blew up).

- Convergence warnings / hitting max iterations.

- SAS prints it outright: "Quasi-complete separation of data points detected."

- Check the cross-tab of each categorical against the target and look for zero cells.

Fixes, roughly in order of what you'd try:

1. Collapse or bin the offending levels (merge small levels into "Other").

2. WoE-encode the categorical - this is the standard fix and it's why high cardinality shows up under Transformations.

3. Add a penalty - for any fixed λ > 0, L2/ridge always yields a finite solution; L1 also works.
   > 📝 **[编辑注 1]** ⚠️ 但实务上要当心：**用 CV 调 λ 时常常会选出 λ = 0**，因为分离的根因并不是过拟合 —— 这时惩罚形同虚设（PSL §10.3.1 明确点出这一点）。所以惩罚只能算「缓解」；真正给出**有限、可做推断**的估计，是下面的 Firth 惩罚或贝叶斯先验。另外记住分寸：分离主要破坏**参数推断**（系数、标准误、p 值），而**决策边界本身是稳定的** —— 只做排序/预测时影响有限。深潜见 `1.6 Estimation Issues & Separation`。

4. Firth's penalized likelihood - purpose-built, keeps the variable.

5. Bayesian priors on the coefficients (weakly informative, e.g. a Cauchy prior).

6. Drop the variable if it adds nothing after binning.

7. Exact logistic regression for very small samples.

### Q: Explain a confusion matrix.

> 📖 **对应讲解**：[[6.3 Performance Metrics]]

> 💬 **中文精讲**：不是背四格，是要你现场手算并说清「该信哪个指标」。骨架：四格（TP / FP 是 Type I 误报 / FN 是 Type II 漏报 / TN）→ 六条公式 → 那个 1,000 份保单的算例（precision 37.5%、recall 60%、accuracy 93%、特异度 94.7%）。
> 胜负手是收尾那一句：**什么都不标的基线准确率 95%，比模型还高** —— 它一次性解释了稀有事件为什么不能报准确率，这句必须有备而来。追问两个方向：precision 随基础率漂移而 recall 不动（本题的编辑注给了精确式），以及除 AUC / PR-AUC 外的指标全部依赖你选的阈值。

|  | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| Actual 0 | TN | FP — Type I error, false alarm |
| Actual 1 | FN — Type II error, miss | TP |
| Metric | Formula | Plain English |
| --- | --- | --- |
| Accuracy | (TP+TN)/N | how often I’m right — misleading under imbalance |
| Precision (PPV) | TP/(TP+FP) | when I say yes, how often am I right |
| Recall = Sensitivity = TPR | TP/(TP+FN) | of the real events, how many did I catch |
| Specificity = TNR | TN/(TN+FP) | of the non-events, how many did I correctly clear |
| FPR | FP/(TN+FP) = 1 − specificity | the x-axis of the ROC curve |
| F1 | 2·P·R/(P+R) | harmonic mean of precision and recall |

#### Worked example — memorize this shape, not the numbers. 1,000 policies, 50 are fraudulent. The model flags 80; 30 of those are actually fraud.

- TP = 30, FP = 50, FN = 20, TN = 900

- Accuracy = 930/1000 = 93%

- Precision = 30/80 = 37.5%

- Recall = 30/50 = 60%

- Specificity = 900/950 = 94.7%

- F1 = 2(0.375)(0.60)/(0.975) = 0.46

- And the baseline that never flags anything scores 95% accuracy - better than the model.

That last line is the whole argument for why you don't report accuracy on rare events. Have it ready.

Two more things to say:

- Precision depends on prevalence; recall does not. Deploy the same model on a population with half the fraud rate and precision halves while recall is unchanged. This is why precision degrades in production when the base rate drifts.
  > 📝 **[编辑注 6]** 「precision halves」是**示意性的近似**，不是恒等式。精确关系是 `precision = π·TPR / (π·TPR + (1−π)·FPR)` —— 只有当 `π·TPR` 远大于 `(1−π)·FPR` 时 precision 才近似正比于基础率 `π`。
  > **用上面那个算例验算**：原基础率 5% 时 `precision = 0.05×0.6/(0.05×0.6 + 0.95×0.0526) = 0.03/0.07997 = 37.5%`；基础率减半到 2.5% 时 `= 0.015/(0.015 + 0.975×0.0526) = 15/66.3 = 22.6%` —— **是下降 40%，不是恰好减半**（因为此例中假阳项与真阳项量级相当）。
  > **方向性结论完全正确**（precision 随基础率下降、recall 不变），面试照说即可；只是被追问「精确减半吗」时要知道它是近似。

- Every metric in that table depends on a chosen threshold. Only AUC / PR-AUC are threshold-free.

### Q: How do a linear model and a GLM differ?

> 📖 **对应讲解**：[[1.1 Ordinary Least Squares (OLS)]] ・ [[1.2 Generalized Linear Models (GLMs)]]

> 💬 **中文精讲**：考「GLM 到底把线性模型的什么泛化了」。骨架按表念五条：独立性一样；Y 从正态放宽到任意指数族；线性对象从「均值」换成「均值的某个函数（link）」；估计从最小二乘或 ML 变成**只能 ML**（IRLS / Fisher scoring）；方差从常数变成均值的函数 `Var(Y) = φ·V(μ)`。
> 两处点题最能加分：「线性」指对参数线性而不是对预测子线性（`β₁x + β₂x²` 仍然是线性模型）；GLM 不需要对方差做稳定化变换，因为异方差已经写进分布假设里了。追问几乎必来「为什么只能 ML」——非正态响应下最小二乘不是 ML 估计、会损失效率，而且没有闭式解，只能迭代求解。

|  | Linear Model | GLM |
| --- | --- | --- |
| 1 | Yᵢ independent | Yᵢ independent |
| 2 | Yᵢ ~ Normal | Yᵢ ~ any exponential family distribution |
| 3 | Mean is linear in the predictors | a function of the mean (the link) is linear in the predictors |
| 4 | Estimation: least squares or ML | Estimation: ML only (via IRLS / Fisher scoring) |
| — | Constant variance required | Variance is a function of the mean (Var(Y) = φ·V(μ)) |

- "Linear" in both cases means linear in the parameters, not in the predictors - `β₁x + β₂x²` is still a linear model. Good clarification to offer.

- **FOLLOW-UP:** Why ML only? Because with a non-normal response, least squares isn't the ML estimator and gives up efficiency; and there's no closed form, so it's solved iteratively (iteratively reweighted least squares).

- The variance-function row is the most under-used answer here. A GLM doesn't need a variance-stabilizing transform of Y because heteroscedasticity is built into the distribution choice.

### Q: What are the GLM assumptions and how do you check each?

> 📖 **对应讲解**：[[1.7 Assumptions]]

> 💬 **中文精讲**：考「一条假设配一条诊断」的成对能力——说得出诊断才算真用过。骨架就是这张表：响应分布 → deviance 残差 Q-Q 图加 deviance/df 看过散布；独立 → 研究设计、查重复保单与聚簇；link 正确 / link 尺度线性 → 残差 LOESS、分箱、empirical logit 图、偏残差图；无严重共线 → VIF、条件指数；无强影响点 → 杠杆值和 Cook 距离；同方差 → 残差图与 scale-location；设定正确 → 残差里还剩结构 + 模型没见过数据上的 actual-vs-expected。
> 最容易失分的是同方差那一行：它只属于线性模型，GLM 里方差本来就该随均值变，说反了等于当场露怯。追问常到 deviance/df ≈ 1 判过散布、VIF 的 > 5 / > 10，以及「怎么证明设定正确」——只有样本外 lift 加业务符号复核这两条路最实在。

| Assumption | Diagnostic |
| --- | --- |
| Response follows the assumed distribution | Q-Q plot of deviance residuals against the assumed exponential-family distribution; compare deviance/df to 1 for over-dispersion |
| Observations are independent | study design; check for repeated policies/claims, clustering, time correlation |
| Correct link / linearity on the link scale | residual analysis, LOESS smooth of residuals vs. each predictor, binning, empirical logit plots, partial residual plots |
| No severe multicollinearity | VIF (>5 concern, >10 serious), condition index |
| No influential outliers / high-leverage points | residual analysis, leverage (hat) statistics, Cook’s distance |
| Homoscedasticity — linear model only | residual plot, scale-location plot |
| Correct model specification | out-of-sample lift/actual-vs-expected, residuals vs. omitted candidate variables, link test, business review of every sign and shape |

- The homoscedasticity row is the one to flag verbally: it applies to the linear model. In a GLM the variance is supposed to change with the mean.

- That last row was blank in the source sheet - the answer above is the practical one: you detect misspecification by finding structure left in the residuals, and by checking actual-vs-expected holds up on data the model never saw.

### Q: Compare GLM to RF/GBM.

> 📖 **对应讲解**：[[1. Logistic Regression & GLMs]] ・ [[10.1 频率-严重度与纯保费]]

> 💬 **中文精讲**：考取舍意识，尤其是保险语境下的取舍。骨架按表念，主角是三条：非线性和交互要不要自己造、能不能外推到训练范围之外、监管报备能不能接受；单调性、精度、可解释性都是围绕这三条的细节。
> 最容易被忽略的是单调性那一行：GLM 对连续变量的效应按构造单调，监管和精算喜欢；树集成会给出波动的非单调形状，你必须解释它或者加约束。追问一般落到「费率模型你选哪个」——答案与 §7 第三题同一套：树做发现、把形状编码进要报备的 GLM。

|  | GLM | Random Forest | GBM |
| --- | --- | --- | --- |
| Non-linearity | No — you must engineer it | Yes, automatic | Yes, automatic |
| Interactions | No — you must specify them | Yes, automatic | Yes, automatic |
| Monotonicity | Yes, by construction | No | No (unless constrained) |
| Accuracy | Lower | High | Highest, typically |
| Interpretability | Coefficients, odds ratios, rate tables | Importance + PDP | Importance + SHAP |
| Extrapolation beyond training range | Yes (linear trend continues) | No | No |
| Filing / regulatory acceptance | Established | Harder | Hardest |

The monotonicity row is the one people miss. A GLM's effect for a continuous variable is monotone by construction, which regulators and actuaries like; a tree ensemble will happily produce a wiggly non-monotone relationship that you then have to explain or constrain.

## 2. Transformations

### Q: When do you need a transformation?

> 📖 **对应讲解**：[[2. Transformations]]

> On the predictor side, whenever the relationship isn't linear on the link scale, or the variable is too skewed/high-cardinality/outlier-prone to use raw. On the response side - in a GLM, almost never, because you choose the distribution and link instead.

> 💬 **中文精讲**：考你分不分得清**预测侧**与**响应侧**。骨架：引用句先把两侧切开，再各自列清单——预测侧要变换（与响应曲线的关系、U 形或带阈值、重尾带来的高杠杆点、高基数类别、要强加业务形状或单调性、想要弹性的乘法解释）；响应侧只在**线性模型**里才需要 log / Box-Cox。
> 加分句是「GLM 里几乎不动响应」：挑一个分布和 link 就已经把偏度和异方差处理掉了，这句话一出口就说明你懂 GLM 与「先变换再 OLS」的分工。被追问「那我到底什么时候动响应」时，答案是只有做线性模型时才动。

Predictor side:

- The relationship with the response is curved, U-shaped, or has a threshold.

- Heavy skew with a long tail creating high-leverage points → log, or cap.

- High-cardinality categorical → WoE, binning, level collapsing.

- You need to impose a business-sensible shape or monotonicity.

- You want multiplicative interpretability (log of a continuous variable with a log link gives an elasticity).

Response side:

- In a linear model: to fix skew and heteroscedasticity (log, Box-Cox).

- In a GLM: you pick Poisson/Gamma/Tweedie and the link instead. Volunteering this is the mark of someone who actually understands GLMs.

### Q: How do you identify which transformation you need?

> 📖 **对应讲解**：[[2. Transformations]]

> I let the data show me the shape first, then pick the simplest parametric form that reproduces it.

> 💬 **中文精讲**：考流程感而不是知识点：「你先看什么」。骨架正是引用句那两步——先让数据露出形状，再挑能复现它的最简参数形式；工具按顺序：empirical logit plot（二值目标的主力）→ 分箱均值响应图 / 残差 LOESS → Spearman 与 Hoeffding's D 一起看 → 先拟合 GAM 看平滑曲线再参数化 → 先拟合 GBM 读 PDP / SHAP 依赖图。
> 最想听到的是那句现代保险工作流：**树模型是发现工具，GLM 是交付物**。追问常给你一个 U 形变量，正解是单调变换救不了它——要分箱或样条；「低 Spearman + 高 Hoeffding's D」这类变量也正是在这一步该被捞回来，而不是删掉。

1. Empirical logit plot (binary target) - the workhorse. Bin the predictor into deciles, and per bin compute ln( (events + 0.5) / (non-events + 0.5)), then plot against the bin mean. A straight line means use the variable as-is; curvature tells you the transform; a U shape means no monotone transform will do and you need bins or a spline.

2. Binned mean-response / target analysis plot - same idea for continuous targets.

3. LOESS smooth of residuals vs. each predictor - leftover structure = missing transform.

4. Spearman + Hoeffding's D together - see §5; low Spearman with high D is the signature of a real but non-monotone relationship.

5. Fit a GAM first, look at the fitted smooth, then approximate it parametrically.

6. Fit a GBM, read the partial dependence / SHAP dependence plot, then encode that shape into the GLM. This is the standard modern insurance workflow - the tree model is the discovery tool, the GLM is the deliverable.

7. Box-Cox / Yeo-Johnson to let ML pick a power transform for you.

### Q: Explain Weight of Evidence coding.

> 📖 **对应讲解**：[[2.3.1 WOE & IV]]

> WoE replaces each bin of a predictor with the log-odds contribution of that bin, so a high-cardinality or non-linear variable becomes a single numeric column that is already on the logit scale.

> 💬 **中文精讲**：考「高基数类别变量怎么办」这个保险日常问题。骨架：先给定义（把每个箱替换成该箱的 log-odds 贡献，高基数或非线性变量就此塌缩成一个已经在 logit 尺度上的数值列），再给 IV 分档，最后主动交代劣处（用了目标所以是监督的、丢掉箱内分辨率、小箱会过拟合、零计数时 log(0) 无定义）。
> 加分项是那句「丢进 logistic 回归的系数应接近 ±1」，但务必先声明自己的 WoE 定义方向，因为两种常见约定互为相反数、理想符号也相反（见本题的编辑注）。追问一定落在泄漏上：分箱边界与 WoE 值只能在训练集上拟合再应用到验证 / 测试 / 生产，在全量数据上算就是泄漏——这与 §6 的 leakage 是同一个故事。→ 深潜：`2.3.1 WOE & IV`

For a binary target and a binned predictor:

```text
WoE_i = ln((Events_i / Total Events) / (NonEvents_i / Total NonEvents))
      = ln(% of all events in bin i / % of all non-events in bin i)
```

- Positive WoE → that bin has proportionally more events than the population. Negative → fewer.

- Because it's already on the log-odds scale, dropping a WoE-coded variable into a logistic regression should produce a coefficient near 1.0. If it comes out far from 1, your binning is off or the relationship isn't stable. Great detail to mention.
  > 📝 **[编辑注 3]** 上面用的是 `ln(%Events / %NonEvents)` ⇒ **正 WoE = 高风险**、理想系数 **≈ +1**。
  > **另一种同样常见的约定是 `ln(%Goods / %Bads)`（正 WoE = 低风险），它的理想系数是 −1 —— 两者互为相反数、都自洽。**
  > **关键是「定义方向与系数符号必须配套」**：回答时先声明自己用哪一种，就不会被面试官绕进去。库内深潜笔记（`2.3.1 WOE & IV`）用的是后者，读时按此换算。

Information Value - the strength of the whole variable:

`IV = Σ_i (%Events_i − %NonEvents_i) × WoE_i`

| IV | Interpretation |
| --- | --- |
| < 0.02 | useless for prediction |
| 0.02–0.1 | weak |
| 0.1–0.3 | medium |
| 0.3–0.5 | strong |
| > 0.5 | suspiciously strong — check for leakage |

Memorize those bands; they get asked directly.

Pros: collapses cardinality to one column, no dummy explosion; monotone-friendly; handles outliers via binning; missing gets its own bin with its own WoE, which is genuinely informative; comparable across variables via IV.

Cons - and these are the follow-ups:

- It's supervised. It uses the target, so computing WoE on the full dataset before splitting is leakage. Bins and WoE values must be fit on training data only and applied to validation/test/production.

- Loses within-bin variation - you've thrown away resolution.

- Overfits with small bins. Enforce a minimum bin size (commonly ≥5% of the population) and a minimum event count per bin.

- Undefined when a bin has zero events or zero non-events (log of 0 or division by 0). Fix by merging the bin or adding 0.5 smoothing to the counts. Note that this is the same problem as quasi-complete separation, viewed from a different angle.

### Q: Dummy coding - how, and what goes wrong?

> 📖 **对应讲解**：[[2. Transformations]]

> k levels become k-1 indicator columns plus a reference level; each coefficient is the contrast against that reference.

> 💬 **中文精讲**：考陷阱意识。骨架：k 个水平变成 k−1 个指示列 + 一个参照水平，每个系数都是对参照的对比；然后是四个坑——用满 k 个再加截距就是 dummy variable trap（完全共线、设计矩阵奇异）、参照水平要挑最大最稳的那一层、高基数下设计矩阵爆炸且稀疏水平估计不稳、空单元直接给出准完全分离。
> 「参照要挑估得准的那一层，因为其他每个估计都是对它做对比」这句最显功力；Greenacre 方法只要知道名字和用途（按最小化卡方关联损失的方式合并名义水平，先降基数再做哑变量）。追问会接到替代方案：WoE、target/mean encoding（同样有泄漏，要平滑加折外计算）、有序变量用 ordinal coding、超高基数用 hashing，以及 LightGBM / CatBoost 的原生类别处理。

- Use k-1, not k. All k dummies plus an intercept is the dummy variable trap - perfect multicollinearity, the design matrix is singular.

- Choose the reference level deliberately: the largest / most stable level, so every other estimate is a contrast against something well-estimated.

- Problems at high cardinality: the design matrix explodes, sparse levels give unstable estimates, and empty cells give quasi-complete separation.

- Greenacre method - an agglomerative way to collapse nominal levels by minimizing the loss of chi-square association, so you reduce cardinality before dummy coding. Know the name and what it's for.

- Alternatives: WoE, target/mean encoding (same leakage caveat, needs smoothing and out-of-fold computation), ordinal coding if the levels really are ordered, hashing for very high cardinality, and native categorical handling in LightGBM/CatBoost.

### Q: Polynomials and Box-Cox.

> 📖 **对应讲解**：[[2. Transformations]]

> 💬 **中文精讲**：两个话题拼成一题。骨架：多项式——简单、还留在 GLM 里，但项间严重共线（要中心化或用正交多项式）、外推极差、尾端为了拟合中段乱摆，超过 2 次优先用样条；Box-Cox——`(y^λ − 1)/λ`，λ 由极大似然选到「最接近正态且方差最稳」，λ = 1 不变换、0.5 平方根、0 取对数、−1 取倒数，且要求 y > 0（有 0 或负值改用 Yeo-Johnson）。
> 这题的杀手锏是反变换偏差：`E[g(Y)] ≠ g(E[Y])`，对 `log(y)` 的预测取指数得到的是**中位数**而不是均值，要用 smearing / Duan 校正才回得到均值；而 log link 的 GLM 直接建模 `log(E[Y])`，没有这个问题——这正是 GLM 在保险里取代「先变换再 OLS」的核心原因。

Basic polynomial — add `x²`, `x³`.

- Pro: simple, stays inside a GLM, captures curvature.

- Cons: terms are severely collinear (center the variable or use orthogonal polynomials); extrapolates terribly; the tails wag to fit the middle; hard to interpret a single coefficient. Prefer splines above degree 2.

Box-Cox — `y^(λ) = (y^λ − 1)/λ` for `λ ≠ 0`, and `ln(y)` for `λ = 0`; λ chosen by maximum likelihood to get closest to normality and constant variance.

- Requires y > 0. Use Yeo-Johnson if you have zeros or negatives.

- λ = 1 no transform, 0.5 square root, 0 log, −1 inverse.

- **TRAP / FOLLOW-UP:** back-transforming a prediction is biased, because E[g(Y)] ≠ g(E[Y]) . If you model log(y) and exponentiate the prediction, you get a median, not a mean, and you need a smearing/Duan correction to get back to the mean. A GLM with a log link models log (E[Y]) directly and has no such problem. This is the core reason GLMs displaced transform-then-OLS in insurance.

### Q: What is capping/flooring and why do it?

> 📖 **对应讲解**：[[2. Transformations]]

> Winsorizing - replace values above a high percentile with that percentile's value, and below a low one likewise. You keep the record but limit how much a single extreme value can move the fit.

> 💬 **中文精讲**：考工程细节加泄漏意识。骨架：先给定义（Winsorizing——把高低分位之外的取值换成该分位的值，记录保留、单个极端值的影响力被限住），再说常用切点（1/99 或 5/95 分位、或拟合关系明显走平处、或业务定义的上限），最后三条理由：限制杠杆与影响力、不必删行、打分时不会遇到训练里没见过的取值范围。
> 保险专属那句会加分：大额损失封顶，让一笔巨灾索赔别去带动严重度系数，超额部分单独作 excess layer 或巨灾加载。追问几乎必是泄漏——分位点是**从训练数据学来的参数**，用全量数据定阈值、或在测试集上重算一遍，都是泄漏；必须拟合在训练集、处处只应用。

- Typical cuts: 1st/99th or 5th/95th percentile; or cap where the fitted relationship visibly flattens; or at a business-defined limit.

- Why: bounds leverage and influence; you don't have to delete the row; and it protects you at score time from values outside anything the model was trained on.

- Insurance-specific: cap large losses so one catastrophic claim doesn't drive severity coefficients, and handle the excess separately as an excess layer or cat load. Saying this shows domain fluency.

- **TRAP:** the caps are parameters learned from the training data. Deriving caps from the full dataset, or recomputing them on the test set, is leakage. Fit on train, apply everywhere.

### Q: Splines and GAMs - what are they and when do you reach for them?

> 📖 **对应讲解**：[[2. Transformations]]

> A spline is a piecewise polynomial joined smoothly at knots; a GAM is a GLM where each predictor gets its own smooth function instead of a single coefficient.

> 💬 **中文精讲**：考「有没有一把工具既能抓曲线关系又不放弃可解释性」。骨架：样条是结点处光滑拼接的分段多项式（回归样条固定结点；**自然三次样条**在边界结点外线性、尾部行为远好于裸多项式，通常是最佳默认；平滑样条每个点都是结点、用粗糙度惩罚并以 GCV 选 λ）；GAM 是每个预测变量有自己的平滑函数、整体仍然可加的 GLM。
> 定位必须说清：GAM 夹在 GLM 与 GBM 之间——比 GLM 灵活，比 GBM 可解释、可加形状或单调约束，但它**不会替你找交互**，要显式加张量积平滑。追问常是「那为什么不直接上 GBM」——因为要报备，要给每个变量画一条能拿去给监管看的效应曲线。

- Regression spline: fixed knots, fit by least squares/ML. Knot count and placement control flexibility.

- Natural cubic spline: cubic, with the extra constraint of being linear beyond the boundary knots - much better tail behavior than a raw polynomial. Usually the right default.

- Smoothing spline: a knot at every point, with a roughness penalty; smoothing parameter λ chosen by generalized cross-validation.

- GAM: `g(E[Y]) = β₀ + f₁(x₁) + f₂(x₂) + ...`. Still additive, so you can plot each variable's effect and show it to a regulator - you keep the GLM interpretability story while dropping the linearity requirement. Does not capture interactions unless you add tensor/interaction smooths explicitly.

- When to reach for it: the relationship is clearly curved, you need interpretability, and you want monotonicity or shape constraints (monotone splines / shape-constrained GAMs).

- The trade-off to state: GAM sits between GLM and GBM - more flexible than GLM, more interpretable and constrainable than GBM, but it won't find interactions for you.

## 3. Missing Data

### Q: What are the missing-data mechanisms?

> 📖 **对应讲解**：[[4. Missing Data]]

> 💬 **中文精讲**：考三层定义，但真正的考点是那句结论——只有 MAR 下插补才站得住。骨架逐行念：MCAR 与任何数据都无关（后果是完整个案分析无偏、只是损失功效）、MAR 只依赖已观测变量（条件在已观测数据之后缺失机制可忽略，这是所有正经插补方法的假设）、MNAR 依赖没观测到的值本身（没有任何插补能修，只能加缺失指示、对机制建模、做敏感性分析）。
> 实务判断法比定义更常被追问：拿缺失指示变量去预测（能被已观测变量预测出来的，就不是 MCAR），再拿它去解释目标（能预测目标，这个标志本身就是特征，留下）。保险现实那句很加分：空白常常意味着「新司机没历史」「第三方数据没匹配上」「这个险种不适用」，所以默认答案是「missing 自成一层」，而不是插补。

| Mechanism | Definition | Example | Consequence |
| --- | --- | --- | --- |
| MCAR — Missing Completely At Random | Missingness is unrelated to any data, observed or not | a sensor randomly drops a reading; a form page is lost | Complete-case analysis is unbiased, just less powerful |
| MAR — Missing At Random | Missingness depends only on observed variables | income missing more often for younger applicants, and you observe age | Ignorable after conditioning on the observed data — this is the assumption every serious imputation method makes |
| MNAR — Missing Not At Random | Missingness depends on the unobserved value itself | high earners decline to state income; prior-claims blank because the insured is concealing it | No imputation fixes it. Add a missing indicator, model the mechanism, and do sensitivity analysis |

How do you tell which you have?

- You generally can't prove it. Little's MCAR test exists but is rarely decisive.

- Practical approach: model the missingness indicator against your other variables. If missingness is predictable from observed data, it isn't MCAR. If it isn't predictable, MCAR is plausible.

- Then: regress the target on the missingness indicator. If the fact of being missing predicts the outcome, that flag is a feature - keep it.

- Insurance reality worth volunteering: missing is usually informative. A blank field often means "new driver with no history," "third-party data didn't match," or "this coverage doesn't apply." So the default is frequently "missing is its own category/bin," not imputation.

### Q: What are the imputation methods and when do you use each?

> 📖 **对应讲解**：[[4. Missing Data]]

> 💬 **中文精讲**：考菜单广度与每条代价。骨架从最粗暴到最讲理念一遍：完整个案删行（只在 MCAR 且缺失很少时站得住）、删列（缺失约 50–70% 以上可考虑，但先看缺失指示有没有信号）、均值 / 中位数、众数或「Missing」单独一类、缺失指示 + 插补值（实战主力）、回归与随机回归插补、kNN、MICE 多重插补 + Rubin 规则（MAR 下唯一能把标准误算对的，代价是重、且在打分流水线里别扭）、树模型原生处理、hot-deck。
> 两条会扣分的红线：插补一定在拆分之后、只在训练集上学参数，在 CV 里属于 pipeline、每折重算；生产必须有一条确定的打分规则——「训练时我们把这些行删了」不是规则。追问常到「均值插补错在哪」：方差塌成尖峰、与其他变量的相关被稀释、标准误偏小、关系被衰减，还把「曾经缺失」这个信号抹掉了。

| Method | What it does | When / caveat |
| --- | --- | --- |
| Complete-case (listwise) deletion | drop any row with a missing value | Only defensible under MCAR with low missingness. With missingness spread across many columns you can lose most of the data |
| Drop the variable | remove the column | Reasonable above ~50–70% missing — but check the missing indicator for signal first |
| Mean / median imputation | fill with the center | Fast. Shrinks variance, attenuates correlations, understates standard errors. Median for skewed data. Only acceptable at low missingness, and always pair it with an indicator |
| Mode / explicit “Missing” level | treat missing as a category | The default for categoricals. Combine with WoE so the missing bin gets its own log-odds |
| Missing indicator + imputed value | add a 0/1 flag alongside the fill | The practical workhorse: retains the information that it was missing, so it partially survives MNAR |
| Regression / conditional mean imputation | predict the missing value from the other variables | Better than the mean; but imputed points sit exactly on the fit line, so variance is still understated |
| Stochastic regression imputation | regression prediction + a random residual | Fixes that variance understatement |
| kNN imputation | average of the k most similar records | Captures local structure. Needs scaling and a distance metric; expensive at scale |
| MICE / multiple imputation | impute each variable from the others, iterate, produce m complete datasets, fit m models, pool with Rubin’s rules | The statistically principled choice under MAR — it’s the only one that gets standard errors right, because it propagates imputation uncertainty. Heavy, and awkward in a production scoring pipeline |
| Model-native handling | XGBoost/LightGBM learn a default direction per split; CART uses surrogate splits | Simple and effective for tree models. You still need a defined rule at score time |
| Hot-deck | copy a value from a similar donor record | Survey-research tradition |

The two things that get you dinged

1. Imputing before splitting is leakage. Every imputation parameter — a mean, a median, a regression model, a kNN index - is learned from data. Learn it on train, apply it to validation/test/production. Inside CV it belongs in the pipeline, refit per fold.

2. Production needs the same rule. If a field can be missing at score time, the model must have a defined behavior for it. “We dropped those rows in training” is not a scoring rule.

**FOLLOW-UP:** What's wrong with mean imputation? It puts a spike of identical values at the mean: variance drops, the correlation with everything else is diluted toward zero, standard errors come out too small, and any relationship involving that variable is attenuated. It also destroys the signal in being missing - which is why the indicator matters.

## 4. Multicollinearity

### Q: Define it.

> 📖 **对应讲解**：[[3. Multicollinearity]]

> Two or more predictors are close to being a linear combination of each other. Collinearity is the pairwise case; multicollinearity includes relationships among three or more variables that no pairwise correlation will reveal.

> 💬 **中文精讲**：考定义精度：两个及以上预测变量彼此近似互为线性组合；两两的叫做共线性（collinearity），三个以上变量之间的关系常常连相关矩阵都看不出来，那才是多重共线性（multicollinearity）。
> 要顺手点出完全共线那种设计矩阵级别的故障：X′X 奇异、系数根本没被识别——dummy variable trap、同时放进一个变量和它的缩放、同时放进各个部分和它们的总和。一句话收口：实务上真正的麻烦是「接近共线」，它是程度问题、不是有或无。

- Perfect multicollinearity → X′X is singular and the coefficients aren't identified at all (the dummy variable trap; including a variable and a rescaling of it; including all components of a sum along with the sum).

- Near multicollinearity is the practical problem, and it’s a matter of degree.

### Q: What are its effects? Does it hurt predictions?

> 📖 **对应讲解**：[[3. Multicollinearity]]

> Estimates stay unbiased, but their variances blow up - so inference and interpretation break while predictive performance is essentially unaffected within the range of the training data.

> 💬 **中文精讲**：考「会不会把危害说过头」——面试官等的就是后半句。骨架：估计仍无偏、但方差爆炸，于是推断和解释坏掉（标准误大、置信区间宽、整体 F 显著而单个 t 都不显著、增删一个变量或重采样就翻符号），而**训练数据范围内的预测基本不受影响**。
> 两个避免显得天真的补充：生产里相关结构漂移时预测会变脆（等于在训练流形之外外推），预测区间也会变宽。树模型那边的对应现象也值得说：相关特征会互相分走功劳，置换重要性在相关下尤其误导——打乱一个变量，模型直接从它的孪生变量上读到信号，两边重要性都接近 0。

That second half is the part interviewers are listening for. Spell it out:

- Large standard errors, wide confidence intervals, and t-tests that come out non-significant even though the variables are jointly significant. The classic tell is a significant overall F-test with no significant individual coefficients.

- Signs flip and estimates swing wildly when you add or drop a variable or resample the data.

- Therefore you cannot interpret an individual coefficient as the effect of that variable. For a rate-filing model, that alone is disqualifying.

- It does not hurt model fit or predictive accuracy. The fitted surface is fine; it's the attribution among the correlated variables that's unstable.

- Two caveats to add so you don't sound naive: predictions become fragile if the correlation structure shifts in production (you're effectively extrapolating off the training manifold), and prediction intervals widen.

For non-GLM models (the sheet calls this out specifically):

- RF/GBM still predict fine, but correlated features split the credit between them, so both look less important than either really is - and you may wrongly drop one.

- Permutation importance is especially misleading under correlation: permute one variable and the model just reads the signal off its twin, so the measured importance is near zero for both.

- Fixes: cluster first and use one representative per cluster; grouped/clustered permutation importance; SHAP with awareness of the same caveat.

### Q: How do you detect it?

> 📖 **对应讲解**：[[3.4.2 Variance Inflation Factor (VIF)]] ・ [[3. Multicollinearity]]

> 💬 **中文精讲**：考工具与阈值，重点在「相关矩阵抓不到三个以上的关系」。骨架按表念：相关矩阵 → VIF = 1/(1−R²ⱼ)（1 表示不相关、> 5 关注、> 10 严重，√VIF 就是标准误被放大的倍数）→ 容差 = 1/VIF（< 0.1 等价于 VIF > 10）→ X′X 特征值算出的条件指数（> 10 中度、> 30 严重，方差分解比例告诉你哪几个变量共享那个坏维度）→ 症状检查（符号不稳、标准误巨大、F 显著而 t 不显著）。
> 顺序 VIF 流程要强调「删掉一个之后必须重新算」，VIF 会变；追问常到「VIF 高就一定要删吗」——如果模型只用于预测而不是解释费率，共线本身不致命，这是与 §5 相连的分寸感。

| Tool | Threshold / reading |
| --- | --- |
| Pairwise correlation matrix | catches collinearity; misses 3+-variable multicollinearity |
| VIF = 1/(1−R²ⱼ) from regressing xⱼ on all other predictors | 1 = uncorrelated, >5 concern, >10 serious. √VIF = the factor by which the standard error is inflated |
| Tolerance = 1/VIF | < 0.1 is the mirror of VIF > 10 |
| Condition index from the eigenvalues of X′X (SAS COLLIN) | >10 moderate, >30 severe. The variance-decomposition proportions tell you which variables share the bad dimension |
| Symptom check | unstable signs, huge SEs, significant F with insignificant t’s |

The sequential VIF procedure: compute VIFs → drop or combine the worst offender → recompute → repeat until all are under threshold. Emphasize the recompute; VIFs change once you remove a variable.

### Q: How do you fix it?

> 📖 **对应讲解**：[[3. Multicollinearity]] ・ [[3.5.2 SAS VARCLUS]] ・ [[1.6.3 Regularization]]

> 💬 **中文精讲**：考方案阶梯与每条的代价，而不是背名字。骨架四条加零散项：按 VIF 顺序删变量（最简单、最可解释、还省采集成本，但丢信息且「删哪个」常常说不清）、变量聚类 VARCLUS（保留原始可解释变量、能扩到上千个候选，但无监督、代表变量未必最会预测）、PCA 回归（成分正交、顺带降维，但完全没有可解释性、还照样要采集维护全部原始变量）、惩罚回归（ridge 稳、lasso 选、elastic net 兼顾相关群）。
> 两个细节最显功力：VARCLUS 的代表变量按 `(1−R²_own)/(1−R²_next)` 最小来挑，即最像自己簇、最不像别的簇；ridge 的本质是给 X′X 加 λI 让它可逆——它就是这个问题的发明物。追问常问「PCR 和 PLS 差别」：PCA 无监督，最大方差方向未必最会预测。

1. Variable removal, sequentially by VIF. Drop the less business-relevant member of each correlated group.

- Pro: simplest, fully interpretable, reduces data-collection cost.

- Con: discards information; which one you drop can be arbitrary and hard to defend.

2. Variable clustering - SAS PROC VARCLUS. Group variables into clusters of mutually correlated variables, then keep one representative per cluster or the cluster component.

- It's oblique principal component clustering: it splits a variable set using principal components, recursively, until each cluster is acceptably one-dimensional.

- Pick the representative by the lowest 1−R² ratio = `(1 − R²_own cluster) / (1 − R²_next closest cluster)` - the variable most representative of its own cluster and least like any other.

- Pro: keeps original, interpretable variables; decorrelates the input set; scales to thousands of candidates.

- Con: unsupervised - it ignores the target, so the most "representative" variable in a cluster isn't necessarily the best predictor in it. The fix is to choose the representative using cluster structure and univariate predictive strength (IV, univariate AUC) and business availability/cost.

3. PCA regression. Replace the predictors with the leading principal components.

- Pro: components are orthogonal by construction, so collinearity is gone by definition; also reduces dimension.

- Con: each component is a linear combination of all the inputs - no interpretability, and you still have to collect and maintain every original variable. It's unsupervised, so the directions of maximum variance need not be the most predictive (this is exactly why PCR can lose to PLS). Scale-dependent, so you must standardize. Sensitive to outliers.

4. Penalized regression.

- Ridge (L2): adds `λΣβ²`. It literally adds λI to X′X, making it invertible - ridge was invented for this problem. Shrinks correlated coefficients toward each other and stabilizes them; keeps all variables.

- Lasso (L1): adds `λΣ|β|`. Selects - but with a correlated group it picks one essentially arbitrarily and zeros the rest, and the choice is unstable across resamples.

- Elastic net: both penalties. Handles correlated groups better than pure lasso - it tends to select or drop them together.

- All three: standardize the inputs first (the penalty is scale-dependent), choose λ by CV, and accept bias in exchange for lower variance.

5. Others worth naming: center the variables (removes the structural collinearity created by polynomial and interaction terms), combine correlated variables into one business-meaningful ratio or index, or collect more data.

### Q: Explain PCA.

> 📖 **对应讲解**：[[3.5.4 PCA]]

> PCA finds a new orthogonal basis ordered by how much variance each direction explains, so you can keep a few components instead of many correlated variables.

> 💬 **中文精讲**：考机制顺序，外加能不能一句话说清它和 VarClus 的差别。骨架四步：标准化（强制，否则量纲最大的变量主导）→ 算相关阵做特征分解（或直接对 X 做 SVD）→ 特征向量是载荷（方向）、特征值是各方向解释的方差，PC1 是最大方差方向、之后每个都与前面正交 → 选 k（碎石图拐点、累计 80–95%、Kaiser 特征值 > 1、或 CV）。
> 最常被追问的是代价：成分是全部输入的线性组合、没有业务可解释性，而且每个原始变量仍然要采集与维护；它还是无监督的，最大方差方向未必最会预测（这正是 PCR 可能输给 PLS 的原因）。收尾用现成那句区分：PCA 是投影（造新变量）、VarClus 是选择（保留原变量），LDA 是有监督的投影对应物。→ 深潜：`3.5.4 PCA`

Mechanics, in order:

1. Standardize X to mean 0, sd 1. Mandatory - otherwise the largest-scale variable dominates.

2. Compute the correlation (or covariance) matrix and take its eigendecomposition - or equivalently the SVD of X directly.

3. Eigenvectors are the loadings (the directions); eigenvalues are the variance explained along each. PC1 is the direction of maximum variance; each subsequent PC is orthogonal to all previous ones and captures the most remaining variance.

4. Choose how many components: scree plot elbow, cumulative variance explained (80-95%), Kaiser criterion (keep eigenvalues > 1 on the correlation matrix), or cross-validation.

Properties to state: unsupervised; linear only; scale- and outlier-sensitive; components are uncorrelated but not statistically independent unless the data is Gaussian; the signs of loadings are arbitrary. SAS: PROC PRINCOMP.

The one-line distinction to have ready:

> PCA is feature projection - it creates new variables. VarClus is feature selection - it keeps original ones. Both are unsupervised; LDA is the supervised projection counterpart.

## 5. Feature Selection

### Q: Why do feature selection at all?

> 📖 **对应讲解**：[[5. Dimension Reduction]]

> Five reasons, and only one of them is about accuracy.

> 💬 **中文精讲**：考「除了精度还有别的理由」这个视角。骨架：五条理由，**只有一条与精度有关**——采集 / 购买 / 维护成本（第三方数据是真钱）、计算时间（训练、打分、刷新）、可解释性（400 个变量的模型没人审得动，报备要逐个解释）、过拟合（变量越多越有机会学到样本噪声）、参数精度下降（每加一个变量都抬高其他变量的标准误，这就是 §4 那条链）。
> 补上运维与合规两条会显得做过真事：变量少则受数据源故障的暴露更小、文档与报备更轻、公平性 / 差异性影响（disparate impact）审查的面更小。追问常接「那你怎么排优先级」——先砍采集成本高或打分时拿不到的变量，再按业务必要性排。

1. Cost of collecting, purchasing, and maintaining extra variables - real money for third-party data.

2. Computation time - training, scoring, and refresh cycles.

3. Interpretability - nobody can review a 400-variable model, and you have to explain every variable in a filing.

4. Overfitting - more variables means more opportunity to learn sample-specific noise.

5. Reduced parameter precision - every added variable inflates the standard errors of the others (and this is the multicollinearity link).

Add the operational ones: fewer variables means less exposure to a data-source outage, easier documentation and filing, and a smaller surface for fairness/disparate-impact review.

### Q: Walk me through your variable reduction workflow on a wide dataset.

> 📖 **对应讲解**：[[5. Dimension Reduction]] ・ [[5.2 Univariate Selection]] ・ [[5.3 Multivariate Selection]]

> 💬 **中文精讲**：这是几乎必考的「走一遍流程」，考顺序感，尤其是**先拆数据必须排在所有监督步骤之前**。骨架九步：业务与合规先筛（最便宜的一刀，砍掉禁用变量、打分时拿不到的、以及任何泄了目标信息的）→ 数据质量筛（近零方差、缺失过多、ID、重复、结果之后才填的变量）→ 先拆数据 → 单变量粗筛（放宽，只是让问题可处理，不是做决定）→ 无监督去冗余（VarClus 或相关聚类，每簇取一个代表）→ 多变量选择（elastic net 或前向 / 后向，用 CV 打分而不是样本内 p 值）→ 树模型重要性交叉校验 → 业务复核每个变量的符号与形状 → 在没动过的 holdout 上验证，并确认跨时段、跨群稳定。
> 第 3 步的位置是这题的暗礁：单变量筛选、WoE、插补、封顶全都是学参数的监督步骤，一旦发生在拆分之前，测试集就在替你选变量（§6 的泄漏）。追问常到「被单变量筛掉的变量就永远丢了吗」——树模型那一步就是专门捞回只在非线性或交互里才有用的变量。

This is a near-certain "walk me through" question. Have the ordered answer ready:

1. Business and regulatory screen first. Remove prohibited variables, anything unavailable at score time, and anything leaky. Cheapest cut you'll make.

2. Data-quality screen. Near-zero variance, excessive missingness, IDs, duplicates, variables whose values are populated after the outcome.

3. Split the data - before any supervised step. Everything below is learned on train only.

4. Univariate screening (supervised, coarse). Rank by Spearman, Hoeffding's D, IV, univariate AUC/R². Keep a generous top set - this is a filter to make the problem tractable, not a decision.

5. Unsupervised redundancy reduction. VarClus or correlation clustering; keep one representative per cluster, chosen using cluster structure plus univariate strength plus cost/availability.

6. Multivariate selection. Elastic net, or forward/backward selection on the reduced set - always scored by cross-validation, not in-sample p-values.

7. Model-based cross-check. GBM/RF importance or SHAP to catch variables that matter only non-linearly or in interaction, and to sanity-check anything you're about to drop.

8. Business review. Every retained variable's sign and shape must be explainable and directionally sensible.

9. Validate on the untouched holdout, and confirm the selection is stable across time periods and major segments.

### Q: Univariate methods - pros, cons, and what you actually use.

> 📖 **对应讲解**：[[5.2 Univariate Selection]]

> 💬 **中文精讲**：考你能不能张口说出单变量筛选的四个缺陷，以及会不会把 Spearman 与 Hoeffding 配对用。骨架：优点四条（快、能扩展到上千变量、与模型无关、好解释）；缺点四条（忽略联合效应，会丢掉只在交互里有用、或作为抑制变量才有用的变量；留冗余；看不见只在条件下出现的关系；多重检验——筛 2,000 个变量在 α = 0.05 下会白捡约 100 个假阳性）。
> 得分点是那两张配对表：**低 Spearman + 高 Hoeffding's D** 说明「有信号但非单调」，处理是分箱、样条或变换，**不是删掉**——这就是两个统计量要一起看的理由。追问常到 IV / 单变量 AUC / 卡方的分档口径。

Pros: fast, scales to thousands of variables, model-agnostic, trivially explainable.

Cons - say all four:

- Ignores joint effects: drops a variable that only matters in interaction, or a suppressor variable that only becomes useful once another is controlled for.

- Keeps redundancy: ten variables all correlated with the target and each other all score well.

- Can't see a relationship that only appears conditionally.

- Multiple testing - screen 2,000 variables at α = 0.05 and you get ~100 false positives.

The specific tools:

| Tool | What it measures | Range / note |
| --- | --- | --- |
| Spearman rank correlation | monotone association (Pearson computed on ranks) | −1 to 1. Robust to outliers and invariant to any monotone transform |
| Hoeffding’s D | any dependence, including non-monotone | ~−0.5 to 1. Detects U-shapes and non-monotone structure Spearman is blind to |
| Scatterplot of ranks | visual confirmation of the above |  |
| Empirical logit plot | the shape of the relationship, binary target |  |
| Univariate R² / AUC / IV / chi-square | strength | IV bands in §2 |

The pairing is the answer to give - this is why both are in the screening list:

| Spearman | Hoeffding’s D | Diagnosis | Action |
| --- | --- | --- | --- |
| High | High | clean monotone relationship | use as-is |
| Low | High | real signal, non-monotone shape | bin, spline, or transform — do not drop it |
| Low | Low | no detectable univariate signal | candidate to drop |
| High | Low | rare/unstable | investigate the data |

The low-Spearman/high-D cell is the one that earns you credit: a naive correlation screen throws those variables away, and they're often the good ones.

### Q: Multivariate methods and subset selection.

> 📖 **对应讲解**：[[5.3 Multivariate Selection]]

> 💬 **中文精讲**：方法本身是次要的，这题真正考的是「逐步法为什么不能交差」。骨架先给四条机制与限制：前向（贪心、不能撤销，可能永远加不进只在组合里才有用的变量，但 p > n 时仍可用）、后向（整体起步、更会抓联合效应，要求 n > p，而且贵）、逐步（继承前两者的问题）、最优子集（精确，但 2^p，靠 leaps-and-bounds 也就到 30–40 个变量）。
> 逐步法的批评要能主动交付六条：p 值与 R² 都乐观偏（同一份数据既选又测）、标准误偏小、置信区间覆盖不足、没有为成百上千次隐式比较做校正、bootstrap 一下选出来的变量就变、最后却被当成事先指定的模型。替代方案：惩罚回归（选择是单次优化的一部分、λ 由 CV 定），或把整个选择过程包进交叉验证。

Pros: accounts for joint effects and redundancy; selects a set that works together. Cons: computationally heavier, and unstable - small perturbations to the data give a different selected set.

| Method | How | Problem |
| --- | --- | --- |
| Forward | start empty, add the best variable each step | Greedy and can’t undo. May never add a variable that only helps in combination. But it works when p > n |
| Backward | start full, remove the weakest each step | Better at catching joint effects, since everything starts in the model. Requires n > p to start, and is expensive |
| Stepwise | add and drop at each step | Inherits the problems of both |
| Best subset | evaluate all 2^p combinations | Exact, but infeasible past ~30–40 variables even with leaps-and-bounds |

The criticism of stepwise you should be able to deliver unprompted:

- p-values and R² are optimistically biased, because the same data both chose and tested the model.

- Standard errors come out too small; confidence intervals under-cover.

- No correction for the hundreds of implicit comparisons.

- Selection is unstable - `bootstrap` the data and you get a different model.

- The final model gets treated as though it had been pre-specified, which it wasn't.

- Preferred alternatives: penalized regression (the selection is part of a single optimization with a CV-chosen λ), or wrapping the entire selection procedure inside cross-validation so the reported performance accounts for the selection.

### Q: How does L1 do selection? Why L1 and not L2?

> 📖 **对应讲解**：[[1.6.3 Regularization]]

> L1 produces exact zeros; L2 shrinks toward zero but never reaches it.

> 💬 **中文精讲**：考你能不能把同一件事用两种语言讲清楚，至少熟一种。骨架：先给结论句「L1 产生精确的 0，L2 只把系数往 0 压、永远到不了 0」，再二选一展开——几何：L1 的约束区是有角的菱形，椭圆的损失等高线通常先碰到角，角就意味着某些系数正好为 0；L2 是光滑球面，接触点几乎不会落在坐标轴上。微积分：L1 的导数是常数 `λ·sign(β)`，无论系数多小都有一份固定拉力、足以把它钉在 0；L2 的导数 `2λβ` 在 β → 0 时自己也消失了。
> 必须补的收尾是相关群的行为：lasso 在相关群上基本随机挑一个、把其余清零，而且跨重采样不稳定；elastic net 才会整组一起选或一起丢。另外别忘了「先标准化」（惩罚对尺度敏感）和「λ 由 CV 选」。

Two ways to explain it - know at least one cold:

- Geometric: minimizing loss subject to a budget on the penalty. The L1 constraint region is a diamond with corners on the axes; the elliptical loss contours typically first touch it at a corner, and a corner means some coefficients are exactly 0. The L2 region is a sphere - smooth, no corners - so the contact point almost never has a zero coordinate.

- Calculus: the L1 penalty's derivative is a constant λ·sign(β), so there's a fixed pull toward zero regardless of magnitude, strong enough to pin small coefficients at exactly 0. The L2 derivative 2λβ vanishes as β → 0, so the pull disappears before it gets there.

|  | Ridge (L2) | Lasso (L1) | Elastic net |
| --- | --- | --- | --- |
| Selects variables | No | Yes | Yes |
| Correlated group | shrinks them together, stable | picks one arbitrarily, unstable | selects/drops them together |
| Best for | collinearity, stability, p > n with all variables relevant | sparsity, interpretability | correlated groups + sparsity |

- λ = 0 recovers OLS/MLE; λ → ∞ drives everything to zero. Choose λ by cross-validation.

- Standardize first - the penalty is scale-dependent, so unscaled variables get penalized according to their units.

- Available for GLMs: SAS PROC HPGENSELECT with a LASSO selection method, Python sklearn with penalty='l1', R glmnet.

### Q: Would you use a GBM's feature importance to pick features for a GLM?

> 📖 **对应讲解**：[[7.4. Quantifying Feature Importance]]

> Yes - as a discovery tool, not as the selection rule.

> 💬 **中文精讲**：考分寸感：能不能既用它又不被它带走。骨架：引用句先定调——会，但只当**发现工具**，不当选择规则。为什么有用：它抓非线性和交互、不需要分布假设，能捞出被线性筛子丢掉的好变量。为什么不能直接取前 N：重要性不是统计显著性、更不是因果；头几名的排序本身不稳；相关变量会互相分走功劳；而且一个对 GBM 重要的变量，在 GLM 里可能要找到对的变换才起作用。
> 正解是读 PDP / SHAP 依赖图学「形状」和「交互」，再把它们**显式**写成变换项与交互项——一句话总结：树模型告诉你造什么，GLM 才是你真正交付的东西。追问常到「那你还做不做单变量筛选」——做，但只当粗筛。

- Why it helps: it captures non-linearity and interactions, so it surfaces variables a linear screen dismisses, and it needs no distributional assumptions.

- Why you can't just take the top N: importance isn't statistical significance and isn't causal; the ranking is unstable near the top; correlated features split credit; and a variable that's important to a GBM may only work in a GLM once you find the right transform.

- The right use: read the partial dependence / SHAP dependence plots to learn the shape and the interactions, then encode those explicitly as transforms and interaction terms in the GLM. The tree model tells you what to build; the GLM is what you file.

### Q: Selection vs. projection, supervised vs. unsupervised.

> 📖 **对应讲解**：[[5. Dimension Reduction]]

> 💬 **中文精讲**：考你能不能把六个东西装进一张 2×2。骨架：无监督的选择（VarClus、相关过滤、低方差过滤）、有监督的选择（单变量筛查、多变量 / 子集选择、树重要性、惩罚回归）、无监督的投影（PCA）、有监督的投影（LDA）；一句话记住「选择保留原变量，投影造新变量」。
> 再叠一层 filter / embedded / wrapper 三族的代价对比：filter 最便宜但与模型无关、看不见联合效应；embedded 高效但绑死在该模型族上；wrapper 最贴近最终模型，也最贵、最容易把「搜索」本身过拟合。追问常到「你实际怎么组合」——filter 粗筛 → 无监督去冗余 → embedded 精挑，全程在 CV 里。

|  | Unsupervised | Supervised |
| --- | --- | --- |
| Feature selection — keeps original variables | VarClus, correlation filters, low-variance filters | Univariate screening, multivariate/subset selection, Forest/GBM importance, penalized regression |
| Feature projection — creates new variables | PCA | LDA |

And the standard taxonomy of selection methods, which is the other way this gets asked:

| Family | How it works | Examples | Cost |
| --- | --- | --- | --- |
| Filter | rank variables by a statistic, independent of any model | correlation, chi-square, IV, low-variance filter, Spearman/Hoeffding | Cheapest; ignores the model and joint effects |
| Embedded | selection happens inside model fitting | LASSO/L1, elastic net, tree-based importance | Efficient; tied to that model family |
| Wrapper | search over subsets, scoring each by actually fitting the model | forward/backward/stepwise, RFE, best subset | Most faithful to the final model, most expensive, most prone to overfitting the search |

## 6. Model Assessment

### Q: How do you split your data, and why three sets?

> 📖 **对应讲解**：[[6.1 Data Preparation & Validation]]

> Train fits the parameters, validation makes the choices, and test gives one unbiased read on generalization.

> 💬 **中文精讲**：考的是「**为什么**要三个」，而不是三个是什么。骨架：训练拟合参数、验证做所有选择（调参、选变量、比模型、定阈值）、测试只给一次无偏读数；比例 60/20/20 或 70/15/15，数据小就用 CV 取代固定验证集。
> 核心句是「一旦某个数据集被用来做决策，它的误差估计就变乐观了」——被比较过 200 次超参的验证集已经不是任何东西的无偏估计。追问几乎必到时间结构：模型要上未来，就留出更晚的时段而不是随机切，保险里通常样本外与时间外两样都做（前者测拟合、后者测稳定）；再补实体完整性——同一张保单 / 家庭 / 索赔的所有行必须在同一侧。

- Train - fit coefficients / grow trees.

- Validation - tune hyperparameters, select features, compare models, pick the threshold.

- Test - a single final estimate. Honest only if you touch it once.

- Typical: 60/20/20 or 70/15/15. With small data, use CV instead of a fixed validation set.

- Why three: the moment you use a dataset to make a decision, its error estimate becomes optimistic. A validation set that's been used for 200 hyperparameter comparisons is no longer an unbiased estimate of anything.

- Stratify on the target for classification, especially with rare events.

- Out-of-time split. If the model will be applied to the future, hold out a later time period, not a random subset. Random-splitting time-ordered data leaks the future into the past. In insurance you generally want both out-of-sample and out-of-time validation - out-of-sample tests the fit, out-of-time tests stability.

- Group/entity integrity. Keep all rows for a policy, household, or claim in the same partition; otherwise the entity's identity leaks across the split.

### Q: What is cross-validation? Which flavor for which data?

> 📖 **对应讲解**：[[6.1 Data Preparation & Validation]]

> Rotate the validation role through k folds and average, so every record is used for both fitting and validating - you get a lower-variance performance estimate without sacrificing training data.

> 💬 **中文精讲**：考对号入座，「哪种数据用哪种折」比定义值钱。骨架：定义一句（让验证角色在 k 折之间轮换再平均，每条记录既参与拟合又参与验证，换来方差更低的性能估计），再逐个对号：k 折 5 或 10 是默认；分类尤其不平衡用分层；样本极少用 LOOCV（几乎无偏但方差高又贵）；想压掉折分配的运气用重复 k 折；有聚簇（同一保单多行）用 GroupKFold；时序用滚动起点、永远不 shuffle；既要调参又要报数用嵌套 CV。
> 两点收尾：k 的选择本身就是偏差-方差问题（k 小 → 每折训练数据少 → 偏悲观但折间方差小，5–10 是标准折中）；最大的陷阱是用同一个 CV 循环既调参又报数，那个数字一定乐观偏，修法是嵌套 CV 或留一个干净的最终 holdout。

| Flavor | Use it when |
| --- | --- |
| k-fold (k = 5 or 10) | the default |
| Stratified k-fold | classification, especially imbalanced — preserves class proportions per fold |
| LOOCV (k = n) | tiny datasets. Nearly unbiased but high-variance and expensive |
| Repeated k-fold | you want to average out the fold-assignment luck |
| GroupKFold | clustered data — multiple rows per policy/person |
| Time-series / rolling-origin CV | temporal data. Expanding or sliding window, always train on the past and validate on the future. Never shuffle |
| Nested CV | you are tuning and reporting. Inner loop tunes, outer loop measures |

- Choosing k is a bias-variance question: small k → each model trains on less data → more pessimistic bias, but lower variance across folds. Large k → less bias, more variance and more compute. 5-10 is the standard compromise.

- **TRAP:** using one CV loop for both tuning and reporting. That number is optimistically biased - you selected the hyperparameters because they did well on those folds. Nested CV, or a clean final holdout, is the fix.

### Q: How do you detect overfitting and underfitting?

> 📖 **对应讲解**：[[6.2 Model Diagnosis_Bias-Variance Tradeoff]]

> 💬 **中文精讲**：考诊断能力：看的是训练误差与验证误差之间的**差距**，不是任何单一数字。骨架按表念三行——欠拟合（两头都高、差距小，修法是更多 / 更好的特征、变换与交互、更弹性的模型、减正则）、合适（都低、差距小）、过拟合（训练低验证高、差距大，修法是更多数据、更少变量、正则、更简单的模型、早停、bagging、用 CV 调参）；再看三张曲线：学习曲线、验证曲线、boosting 的逐轮训练 / 验证跟踪。
> 学习曲线要能读出结论：两条曲线在高误差处收敛就是偏差问题、加数据没用；差距明显但随数据增大而收窄就是方差问题、加数据有用。验证曲线找的是那个 U 形的最低点；boosting 里验证误差拐上去的那一轮就是早停点。

|  | Training error | Validation / test error | Gap | Fix |
| --- | --- | --- | --- | --- |
| Underfitting (high bias) | High | High | Small | more/better features, transforms and interactions, a more flexible model, less regularization |
| Good fit | Low | Low | Small | ship it |
| Overfitting (high variance) | Low | High | Large | more data, fewer features, regularization, simpler model, early stopping, bagging, CV-driven tuning |

Tools:

- Learning curve — error vs. training set size. Both curves converging at a high error = bias problem, and more data won't help. A persistent gap that narrows as data grows = variance problem, and more data will help.

- Validation curve — error vs. a hyperparameter. You're looking for the U: the minimum of the validation curve is your setting.

- Per-iteration train/validation tracking for boosting — the iteration where validation error turns up is your early-stopping point.

### Q: What is data leakage? Give me examples.

> 📖 **对应讲解**：[[6.1 Data Preparation & Validation]]

> Leakage is any information in the training features that wouldn't be available at the moment you actually need to score - including information that leaked in through your own preprocessing.

> 💬 **中文精讲**：考实战警惕性——答得越具体越可信。骨架：先给定义（训练特征里含了真正打分那一刻拿不到的信息，也包括从自己预处理环节漏进来的信息），再分类举例：目标泄漏（结果之后才记录、或由结果产生的字段——预测「是否发生赔付」时用赔付笔数、已发生损失、销案原因码、理算员结案后才填的字段、被所预测事件更新的保单状态位）；训练 / 测试污染（全量标准化、全量均值插补、全量算 WoE 或 target encoding、全量分位封顶、拆分前先做特征选择）；再加时间泄漏、组泄漏、跨侧重复记录。
> 规则那句要背下来：任何**学参数**的步骤、尤其任何看过目标的步骤，都属于 CV 流水线内部、每折重算。追问常是「怎么发现」：性能好得不正常（难问题上 AUC 0.97，先找泄漏再庆祝）、看重要性头部有没有不合常理的强变量、对每个字段问「打分那一刻我手上有没有这个字段、是不是这个值」，并用时间外 holdout 照出随机 CV 会藏起来的时间泄漏。

Target leakage - a feature that is a consequence of, or recorded after, the outcome:

- "Number of claim payments" in a model predicting whether a claim occurs.

- Total incurred loss when predicting whether a loss happens.

- "Cancellation reason code" in a lapse model.

- Any field the adjuster only populates after the claim closes.

- A policy-status flag that gets updated by the event you're predicting.

Train/test contamination - a preprocessing step fit on all the data before splitting:

- Scaling or standardizing on the full dataset.

- Imputing with the full-data mean.

- WoE or target encoding computed on everything - the most common serious one.

- Capping at full-data percentiles.

- Feature selection run before the split - you've let the test set vote on which variables enter.

- The rule: any step that learns a parameter, and especially any step that looks at the target, belongs inside the CV pipeline and gets refit per fold.

Temporal leakage - using future information, or random-splitting time-ordered data.

Group leakage - the same policy, household, or VIN appearing in both train and test.

Duplicate records across splits.

How you catch it:

- Performance that's too good. AUC 0.97 on a genuinely hard problem means you look for leakage before you celebrate. Then it collapses in production and you find it the expensive way.

- Check the top feature importances for anything implausibly strong.

- For every feature, ask: "Would I have this field, with this value, at the moment I need to score?" Check the timestamp on every field.

- An out-of-time holdout catches temporal leakage that random CV will happily hide.

### Q: Define the metrics and tell me when you'd use each.

> 📖 **对应讲解**：[[6.3 Performance Metrics]]

> 💬 **中文精讲**：考广度，但真正评分的是收尾那句总结：**区分度与校准是两件事**——区分度看排序（AUC、Gini、lift），校准看水平对不对（log-loss、Brier、实际 vs 预期十分位），定价模型两样都要，分诊模型只要排序。骨架就是两张表：连续型（RMSE / MAE / MAPE / R² / deviance·AIC·BIC）与二值型（accuracy、precision-recall-F1-特异度、AUC、Gini、PR-AUC、log-loss / Brier、校准图、lift / gain）。
> 几条边界必须能解释：R² 加变量只会不降所以要看调整 R²，而且对 GLM 没意义（用 pseudo-R² 或 deviance）；MAPE 在 y → 0 时炸掉、y = 0 无定义，所以零多的目标（索赔次数）根本不能用；AUC 对类别比例不敏感既是优点也是陷阱——它可能看着体面，而在你真正的工作点上毫无用处。Gini = 2·AUC − 1 是保险惯例，0.75 → 0.50。

Continuous targets:

| Metric | Formula | Use / caveat |
| --- | --- | --- |
| RMSE | √(Σ(y−ŷ)²/n) | Same units as y; penalizes large errors quadratically. The default when big misses are disproportionately costly. Outlier-sensitive |
| MAE | `(1/n)·Σ|y − ŷ|` ← **由 OCR 片段恢复** | Same units as y; treats every error linearly, so it's less outlier-sensitive than RMSE. Use it when all misses cost about the same |
| MAPE | `(100/n)·Σ|(y − ŷ)/y|` ← **由 OCR 片段恢复** | Percentage error, comparable across series with different scales. Blows up as y → 0 and is undefined at y = 0 — unusable for zero-heavy targets like claim counts |
| R² | 1 − SSE/SST | Proportion of variance explained. Never decreases when you add a variable → use adjusted R². Not meaningful for GLMs — use pseudo-R² (McFadden) or deviance |
| Deviance / AIC / BIC | AIC = −2logL + 2k; BIC = −2logL + k·ln(n) | The right way to compare GLMs. BIC penalizes complexity harder, so it picks smaller models |

Binary targets:

| Metric | Use / caveat |
| --- | --- |
| Accuracy | Almost never, under imbalance. Always compare to the majority-class baseline |
| Precision / Recall / F1 / Specificity | When there’s an operating threshold and asymmetric costs. Precision moves with prevalence; recall doesn’t |
| AUC / ROC | Threshold-free ranking quality. AUC = P(random positive scores above random negative). 0.5 = random. Insensitive to class balance — a feature and a trap, since it can look respectable while the model is useless at your actual operating point |
| Gini = 2·AUC − 1 | The insurance convention. AUC 0.75 → Gini 0.50 |
| PR-AUC / average precision | Better than AUC under heavy imbalance, because it doesn’t get credit for the enormous true-negative count |
| Log-loss / Brier score | Proper scoring rules — they measure calibration as well as discrimination. Use these when you need probabilities, not just an ordering |
| Calibration plot / actual-vs-expected by decile | Essential for pricing. A model can rank perfectly and still be 30% off in level |
| Lift / gain | Translating the model into business value — see below |

The distinction to state explicitly: discrimination (can the model rank? AUC, Gini, lift) versus calibration (are the predicted levels right? log-loss, Brier, A-vs-E plots). A pricing model needs both. A triage model only needs the first.

### Q: Explain a lift chart and a gain chart. How do you read one?

> 📖 **对应讲解**：[[6.4 Diagnostic & Visualization Tools]] ・ [[6.3 Performance Metrics]]

> 💬 **中文精讲**：考「模型怎么翻译成业务语言」，以及你会不会批判地读图。骨架：构造四步（打分 → 按预测降序 → 切十分位 → 算每箱实际响应率）；gain（累计响应）图的 x 轴是累计触达比例、y 轴是累计捕获的事件比例，45° 线是随机基线，曲线越往左上角弓越好；lift = 箱内响应率 ÷ 整体响应率，顶部十分位 lift 是标准的一个数字总结。
> 三个批判性读法最值钱：lift 应从顶部往下单调衰减，非单调（第 3 箱压过第 2 箱）是稳定性或过拟合的红旗而不是噪声；必须按运营产能读图（SIU 只能查 2% 的案子，10% 处的 lift 无关）；连续型目标同样构造，顶箱与底箱实际损失成本之比就是定价里「多分出了多少段」的统计量。再加一句「箱内补上实际 vs 预期，顺带变成校准检查」就更完整。

Construction (identical for both):

1. Score the validation set.

2. Sort descending by predicted probability (or predicted loss cost).

3. Cut into deciles (or 20 equal bins).

4. Compute the actual response rate within each bin.

Gain (cumulative response) chart - x-axis is cumulative % of the population targeted, y-axis is cumulative % of all events captured. The 45° line is the **random-targeting baseline**: selecting 20% of the population at random gets you 20% of the events. The vertical gap between the curve and that line is the gain — a curve that hugs the 45° line means the model adds nothing, and the more it bows toward the top-left corner, the better. *(照片在此处截断，此句由通用定义补写。)*

Lift chart — lift in a bin = (response rate in the bin) / (overall response rate).

> "Cumulative lift at 20% is 55/20 = 2.75× random."

How to read one critically:

- Top-decile lift is the standard one-number summary.

- Lift should decay monotonically from the top decile down. Non-monotonicity - decile 3 outperforming decile 2 - is a red flag about stability or overfitting, not just noise.

- Read the chart at your operating capacity. If the SIU team can only investigate 2% of claims, the lift at 10% is irrelevant.

- For continuous targets (loss cost), build it the same way with the average actual value per bin; the ratio of top-decile to bottom-decile actual loss cost is the standard "how much segmentation did we gain" statistic in pricing.

- Adding the actual vs. expected value within each decile turns it into a calibration check too, not just an ordering check.

Related charts to name: Lorenz/concentration curve, and the double lift chart for comparing a proposed rating plan against the current one.

## 7. Bias-Variance & Ensembles

### Q: Explain the bias-variance tradeoff. Write the decomposition.

> 📖 **对应讲解**：[[6.2 Model Diagnosis_Bias-Variance Tradeoff]]

> `E[(y − f̂(x))²] = Bias[f̂]² + Var[f̂] + σ²`

> 💬 **中文精讲**：考能不能默写分解式并逐项解释，而不是背「高偏差欠拟合、高方差过拟合」。骨架：写出 `E[(y − f̂(x))²] = Bias² + Var + σ²`，然后逐项念——偏差是模型太简单或结构错，系统性的、加数据不会消失；方差是「换一个训练样本，拟合结果会变多少」；σ² 是不可约噪声，是地板。
> 关键句：复杂度上升时偏差降、方差升，总误差是 U 形，目标是它的最低点而**不是零偏差**。接着要说清什么在沿着曲线移动你：模型弹性、正则强度、变量个数、训练集大小（更多数据只降方差、不降偏差）。追问常给一个具体场景让你判断这是偏差问题还是方差问题。

- Bias — error from the model being too simple or structurally wrong. Systematic; it doesn't go away with more data. High bias = underfitting.

- Variance - how much the fitted model changes if you'd drawn a different training sample. High variance = overfitting.

- σ² — irreducible noise. The floor. No model beats it.

- As complexity increases, bias falls and variance rises. Total error is U-shaped, and the goal is its minimum — not zero bias.

- What moves you along the curve: model flexibility, regularization strength, feature count, and training-set size (more data lowers variance, not bias).

### Q: Bagging vs. boosting vs. stacking.

> 📖 **对应讲解**：[[8.3. Bagging (Bootstrap Aggregating)]] ・ [[8.4. Boosting]] ・ [[8.6 Bagging VS Boosting]]

> 💬 **中文精讲**：考三族对照加两段「为什么」。骨架先按表念六行（基学习器：强 / 低偏差高方差 vs 弱 / 高偏差 vs 异质模型；训练方式：并行独立 vs 顺序拟合前面集成的误差 vs 基模型并行再叠元模型；每个学习器看到的数据；主要降什么；加成员会不会过拟合；代表算法），再分别解释 bagging 为什么降方差、boosting 为什么降偏差（形式上是函数空间里的梯度下降）。
> 公式 `Var(平均) = ρσ² + (1−ρ)σ²/B` 是本题的枢纽：`1/B` 项随树数消失，但 `ρσ²` 是地板——这是随机森林必须做特征子采样的**唯一**理由（一句话同时答了三道题）。另两个必答细节：stacking 的元模型必须用折外预测训练，否则基模型的过拟合会直接泄进元模型；每个 bootstrap 约留下 1/e ≈ 36.8% 的行，就是免费的 OOB 验证。

|  | Bagging | Boosting | Stacking |
| --- | --- | --- | --- |
| Base learners | Strong, low-bias/high-variance (deep trees) | Weak, high-bias (shallow trees, stumps) | Heterogeneous models |
| Trained | In parallel, independently | Sequentially, each on the previous ensemble’s errors | Base models in parallel, then a meta-model on top |
| Data per learner | Bootstrap samples (with replacement) | Full data, reweighted or residualized | Full data; meta-model trained on out-of-fold predictions |
| Primarily reduces | Variance | Bias (plus variance via shrinkage/subsampling) | Both, by exploiting model diversity |
| Overfitting with more members | Low — more trees never hurts | Higher — will eventually overfit, needs early stopping | Moderate; leaks badly if done wrong |
| Examples | Random Forest, bagged trees | AdaBoost, GBM, XGBoost, LightGBM | Super learner, blended models |

Why bagging reduces variance - the formula worth knowing: averaging B estimators each with variance σ² and pairwise correlation ρ gives

`Var(average) = ρσ² + (1−ρ)σ²/B`

The 1/B term vanishes with more trees, but ρσ² is a floor. That's the entire justification for random feature subsetting in a random forest -the only way past the floor is to lower ρ, and `max_features` is the knob that does it. If you can say this, you've answered three questions at once.

Why boosting reduces bias: each new learner is fit to what the ensemble still gets wrong, so the approximation error shrinks by construction. Formally it's gradient descent in function space.

Stacking caveat: the meta-model must be trained on out-of-fold base predictions. Train it on in-fold predictions and the base models' overfitting leaks straight into the meta-model.

Bonus stat: each `bootstrap` sample omits about 1/e ≈ 36.8% of the rows. Those are the out-of-bag observations, and they give you a free validation estimate.

### Q: You work at an insurance company. GLM or GBM?

> 📖 **对应讲解**：[[10.1 频率-严重度与纯保费]] ・ [[1. Logistic Regression & GLMs]]

> It depends on whether the deliverable is a decision or a filed rate. For anything that goes into a rate, the GLM's interpretability and monotonicity usually win; for internal triage and targeting, the GBM's accuracy wins.

> 💬 **中文精讲**：考「你会不会按交付物选工具」，而且面试官在等你提监管。骨架：引用句先定调（看交付物是一个决策，还是一张要报备的费率表），再两栏算账——GLM：系数可解释、乘法结构直接映射成费率表、报备与精算复核有传统、按构造单调、稳定、标准误清楚；代价是每个非线性与交互都要手工造，精度上要放弃一些。GBM：精度更高、自动抓非线性与交互、对异常值和单调变换稳健、原生处理缺失；代价是黑箱、难报备、可能给出非单调甚至反直觉的形状（要靠单调约束压住）、要调参、不能外推、需要一整套可解释性材料。
> 落地的答法：用 GBM 发现哪些变量重要、效应是什么形状，再把发现编码进要报备的 GLM；或者在辖区允许时用单调约束加一份成文的可解释性包去报备 GBM。一定要主动提报备约束——那是通用机器学习候选人不会有的保险信号。→ 深潜：`10.9 项目实战：法语车险纯保费定价`

GLM: interpretable coefficients, multiplicative structure that maps directly onto a rate table, established for actuarial review and regulatory filings, monotone and explainable by construction, stable, with well-understood standard errors. Cost: you hand-engineer every non-linearity and interaction, and you leave accuracy on the table.

GBM: higher accuracy, finds non-linearity and interactions automatically, robust to outliers and monotone transformations of the predictors, handles missing values natively. Cost: black box, harder to file, can produce non-monotone and counterintuitive behavior that needs monotone constraints, more tuning, no extrapolation, needs SHAP and a full explainability package.

The answer that lands: use the GBM to discover which variables matter and what shape their effects take, then build and file the GLM that encodes those findings - or file the GBM with monotonic constraints plus a documented explainability package where the jurisdiction allows it. Always mention the filing constraint; that's the insurance-specific insight a generic ML candidate won't have.

## 8. Random Forest

### Q: Walk me through the algorithm.

> 📖 **对应讲解**：[[7.2. Building the Forest]] ・ [[7. Random Forest]]

> Bagged deep decision trees, with an extra trick: at every split, each tree only gets to consider a random subset of the features.

> 💬 **中文精讲**：考算法流畅度，分水岭在「两处随机性各自干什么」。骨架四步：对 b = 1…B 抽一个含 n 行的 bootstrap 样本（有放回）→ 在该样本上长树，每次分裂只在随机抽出的 `max_features` 个变量里找最优切分 → 长到纯或到 `min_samples_leaf`、不剪枝 → 回归取平均，分类用多数票或（更好，也是 sklearn 的实际做法）平均各类概率。
> 第二处随机性必须解释：没有它，一个主导变量会出现在几乎所有树的顶层分裂上、树长得几乎一模一样、ρ 变得很高，平均就几乎降不了方差——这就是 §7 的 ρσ² 地板。顺带记住分裂准则：分类用 Gini 不纯度或熵（Gini 更便宜、结果通常差别不大），回归用方差下降 / MSE。

1. For b = 1...B: draw a `bootstrap` sample of n rows, with replacement.

2. Grow a tree on that sample. At each split, randomly select `max_features` of the p predictors and find the best split only among those.

3. Grow deep - to purity or to `min_samples_leaf`. No pruning.

4. Predict: regression → average the B tree predictions. Classification → majority vote, or better (and what sklearn actually does) average the trees' predicted class probabilities.

The two sources of randomness, and why you need both:

- Bootstrap rows - this is the bagging part; it decorrelates the trees somewhat.

- Random feature subset at each split - this is what makes it a random forest rather than bagged trees. Without it, one dominant predictor would be the top split in nearly every tree, the trees would be near-identical, ρ would be high, and averaging would barely reduce variance. This is the ρσ² floor from §7.

Split criteria: Gini impurity `1 − Σpₖ²` or entropy `−Σp·log p` for classification (Gini is cheaper and they rarely disagree materially); variance reduction / MSE for regression.

### Q: What is OOB error?

> 📖 **对应讲解**：[[7. Random Forest]]

> Each `bootstrap` sample leaves out about 36.8% of the rows. Predict each row using only the trees that never saw it, aggregate, and you have a validation estimate for free.

> 💬 **中文精讲**：考 36.8% 从哪来，以及它的边界在哪。骨架：每个 bootstrap 样本约漏掉 36.8%（1/e）的行 → 用「从没见过这一行的那些树」去预测它再聚合 → 就得到一份免费的验证估计；用途是快速调参，以及判断 `n_estimators` 够不够（盯 OOB 误差趋平）。
> 边界一定要主动交代：OOB 仍在同一时段、仍在样本意义上，替代不了真正的时间外 holdout。这一句能挡住「那你为什么还做时间外验证」这类反问。追问常到「OOB 和 CV 差在哪」——OOB 免费，但只适用于 bagging 这一类，而且覆盖不了预处理环节的泄漏。

- Useful for quick tuning and for checking whether `n_estimators` is large enough (watch OOB error flatten).

- Not a substitute for a proper out-of-time holdout - OOB is still in-period and in-sample in the temporal sense.

### Q: Key hyperparameters - what they do and how they trade off.

> 📖 **对应讲解**：[[7. Random Forest]]

> 💬 **中文精讲**：考你调过没有——每一行都要给出「往哪边调、换来什么」。骨架按表念，重点四处：`n_estimators`（单调变好直到趋平、不会过拟合、成本线性，常用 300–1000，看 OOB 趋平就停）、`min_samples_leaf`（最直接的噪声控制，叶子小到一行就是在背答案，噪声大或不平衡时抬到 5 / 20 / 50 或 n 的某个比例）、`max_features`（RF 的招牌旋钮：低 → 树间更去相关但单树更弱，高 → 更强但更相关、退化成普通 bagging；默认分类 √p、回归 p/3）、`bootstrap`（留着——它同时是 OOB 的前提）。
> 能顺口说出「更多树不会过拟合是因为预测是平均，往平均里加项只降方差、不改期望」就接住了 §7；追问几乎必然是 `max_features` 与 `n_estimators` 要成对调（下一题）。

| Parameter | What it controls | Trade-off / typical |
| --- | --- | --- |
| `n_estimators` | number of trees | More is monotonically better until it plateaus, because it’s an average. More trees does not overfit — that’s the key difference from GBM. Cost is linear in time and memory. Typical 300–1000; raise until OOB/validation error flattens |
| `max_depth` | maximum tree depth | Deeper → lower bias, higher variance per tree, but averaging absorbs much of that. Often left unlimited in RF; cap it for speed or with very noisy data |
| `min_samples_split` | min rows required to attempt a split | Higher → shallower, more regularized trees |
| `min_samples_leaf` | min rows in a terminal node | The most direct noise control — a one-row leaf memorizes. Raise for noisy or imbalanced data (5, 20, 50, or a % of n) |
| `max_features` | candidates considered per split | The signature RF knob. Low → more decorrelation (lower ensemble variance) but each tree is weaker (higher bias). High → stronger but more correlated trees, converging to plain bagging. Defaults: √p classification, p/3 regression |
| `bootstrap` | sample rows with replacement | False means every tree sees all rows, so feature subsetting is the only randomness. Leave True — it’s also what enables OOB |
| `criterion` | split quality measure | gini / entropy / log_loss; squared_error / absolute_error |
| `class_weight` | reweight classes | 'balanced' for imbalance; 'balanced_subsample' recomputes per `bootstrap` |
| `max_samples` | `bootstrap` size below n | Speed, plus extra regularization |

### Q: How do those hyperparameters interact?

> 📖 **对应讲解**：[[7. Random Forest]]

> 💬 **中文精讲**：这题在源表里被明确要求，考的是「你真调过还是只背过默认值」。骨架五组，每组都要说机制：`max_features` × `n_estimators`（低 max_features 让单树更弱更去相关，多出来的那部分噪声要靠更多树平均掉，两者成对调）；`max_depth` × `min_samples_leaf`（一个从上面压复杂度、一个从下面压，同调基本冗余，定一个调另一个）；树复杂度 × `n_estimators`（深而高方差的树从加树里获益更多）；`class_weight` × `min_samples_leaf`（重类权重配极小叶子是过拟合稀有类最快的路，一旦加权就要抬高叶子下限）；`bootstrap`=False × `max_features`=p（得到 B 棵一模一样的树、集成收益为 0）。
> 最后那条是「你到底懂不懂」的题眼——它证明两处随机性一个都不能少，能顺口说出来等于证明你不是在背参数表。

The source sheet asks for this explicitly, so have real answers:

- `max_features` × `n_estimators` - lower `max_features` makes each tree weaker and more decorrelated, so you need more trees to average the added noise away. Tune them as a pair.

- `max_depth` × `min_samples_leaf` - both cap complexity from opposite ends (top-down vs. bottom-up). Tuning both is largely redundant: fix one, tune the other.

- Tree complexity × `n_estimators` - deeper, higher-variance trees gain more from additional trees than shallow ones do.

- `class_weight` × `min_samples_leaf` - heavy minority-class weighting combined with tiny leaves is the fastest way to overfit a rare class. If you upweight, raise the leaf minimum.

- `bootstrap`=False × `max_features`=p - that combination gives you B identical trees and zero ensemble benefit. A good “do you actually understand this” answer.

### Q: What tuning strategies do you use?

> 📖 **对应讲解**：[[7. Random Forest]]

> 💬 **中文精讲**：考方法选择与预算意识。骨架按表念四种「怎么工作 / 什么时候用」：网格搜索（参数少且已定位粗范围；成本是各维网格的乘积，超过 3–4 个参数就死）；随机搜索（第一轮默认；同样预算下通常赢过网格，因为真正重要的参数没几个，而随机采样能给每个参数更多不同的取值，而不是几个重复值）；贝叶斯优化（TPE / GP；每次拟合都贵、预算紧时用，缺点是串行不好并行、在便宜模型上开销不值）；逐次减半 / Hyperband（搜索空间大、早期信号便宜时：先便宜地跑一批、杀掉差的、把预算转给好的）。
> 收尾三句必须说：在验证 / CV 折上调、在没动过的测试集上报数、调得很狠就上嵌套 CV。

| Strategy | How it works | When to use it |
| --- | --- | --- |
| Grid search | exhaustive over a specified grid | Few parameters, and you’ve already located a coarse range. Cost is the product of the grid dimensions, so it dies past 3–4 parameters |
| Random search | sample combinations from distributions | The default first pass. For the same budget it usually beats grid search, because only a few parameters actually matter and random sampling gives you many distinct values of each rather than a few repeated ones |
| Bayesian optimization (TPE/GP — Optuna, Hyperopt) | builds a surrogate model of the objective and picks the next point by expected improvement | Each fit is expensive and the budget is tight. Con: sequential, so harder to parallelize; the overhead isn’t worth it on cheap models |
| Successive halving / Hyperband | start many configs cheaply, kill the losers, reallocate budget | Large search spaces with cheap early signals |

Always: tune on validation/CV folds, report on the untouched test set, and consider nested CV if the tuning is extensive.

### Q: How does RF compute feature importance?

> 📖 **对应讲解**：[[7.4. Quantifying Feature Importance]]

> 💬 **中文精讲**：考四种重要性的差别，重点在每种偏差来自哪里。骨架按表念：MDI / Gini 重要性（训练时免费算出来，但偏向高基数与连续变量，而且是在训练数据上算的）；置换重要性（模型无关、在留出数据上测，但相关特征下不可靠、更贵）；drop-column（最忠实、最贵）；SHAP（博弈论加性归因，一致、给方向和大小、局部与全局都能用，现代默认）。
> 那句 caveat 要主动说、不用人问：重要性不是统计显著性、也不是因果，相关特征还会互相分走功劳——这一句在 §8 与 §9 通用。追问常到「MDI 为什么偏向高基数」：因为可选的切分点更多，就有更多机会降低不纯度。→ 深潜：`7.4. Quantifying Feature Importance`

| Method | How | Caveat |
| --- | --- | --- |
| MDI — mean decrease in impurity (Gini importance) | total impurity reduction from all splits on that feature, averaged over trees | Free (computed during training), but biased toward high-cardinality and continuous variables, and computed on training data |
| Permutation importance | shuffle one feature, measure the drop in validation performance | Model-agnostic and measured out-of-sample — but unreliable under correlated features (the model reads the signal off the twin) and more expensive |
| Drop-column | refit without the feature | Most faithful, most expensive |
| SHAP | game-theoretic additive attributions | Consistent, gives direction and magnitude, works locally and globally. The modern default |

Say the caveat unprompted: importance is not statistical significance and is not causal, and correlated features split the credit between them.

### Q: Pros and cons.

> 📖 **对应讲解**：[[7. Random Forest]]

> 💬 **中文精讲**：考平衡感，也是 §7「GLM 还是 GBM」的素材库。骨架：优点一口气说完（开箱精度高、调参少、自动抓非线性与交互、对 X 里的异常值和无关变量稳健、不用缩放、对预测子的单调变换不变、可并行、免费 OOB、内置重要性、支持混合类型）；缺点（相对 GLM 是黑箱、模型大打分慢、**不能外推**——每个预测都是已观测 y 的加权平均，延续不了趋势，这在有趋势的保险数据上是真问题、重要性偏向高基数、在稀疏高维如文本上不如线性模型、严重不平衡时被多数类淹没、预测概率常需要校准）。
> 隐藏收尾是那条 FOLLOW-UP：为什么加树不会过拟合——预测是独立拟合的树的平均，往平均里加项只降方差、不动期望，它是收敛而不是退化；对比 boosting 每棵新树都在追当前残差，会把拟合函数一路推向训练数据。

Pros: strong accuracy out of the box with minimal tuning; captures non-linearity and interactions; robust to outliers in X and to irrelevant features; no scaling needed; invariant to monotone transforms of the predictors; parallelizable; free OOB estimate; built-in importance; handles mixed data types.

Cons: a black box relative to a GLM; large model size and slower scoring; cannot extrapolate beyond the training range - every prediction is an average of observed y values, so it can't continue a trend (a real problem for trended insurance data); importance biased toward high-cardinality variables; weaker than linear models on very sparse high-dimensional data like text; can be swamped by the majority class under severe imbalance; predicted probabilities often need calibration.

**FOLLOW-UP:** Why doesn't RF overfit as you add trees? Because the prediction is an average over independently-fit trees. Adding more terms to an average reduces its variance and leaves its expectation alone; it converges rather than degrading. In boosting, each new tree changes the fitted function by chasing the current residuals, so more trees keeps pushing toward the training data.

## 9. GBM & XGBoost

### Q: Walk me through gradient boosting.

> 📖 **对应讲解**：[[8.4.2.2 Gradient Boosting]]

> Fit a sequence of shallow trees, where each one is fit to the errors the current ensemble is still making, and add it in at a small learning rate.

> 💬 **中文精讲**：考算法细节，以及那条「为什么叫梯度」的解释。骨架六步：初始化为最优常数预测（平方误差下是均值、log-loss 下是基础率的 log-odds）→ 对 m = 1…M 算损失对当前预测的负梯度（伪残差，平方误差下就是 `y − F(x)`）→ 拿这些伪残差拟合一棵小回归树 → 每个叶节点求最优常数值（线搜索或牛顿步）→ `F_m = F_{m-1} + ν·h_m`，ν 是学习率 / 收缩 → 到 M，或按验证集早停。
> 「函数空间里的梯度下降」这个框架是本题的钥匙：它解释了为什么可以塞进任何可微损失，包括保险关键的 Poisson、Gamma、Tweedie deviance。还要点出结构差别——预测是初始常数加上 ν 乘每棵树输出的**加性求和**，不是平均，这正是与随机森林的分野。→ 深潜：`8.4.2.2 Gradient Boosting`

1. Initialize with the best constant prediction - the mean for squared error, the log-odds of the base rate for log-loss.

2. For m = 1...M: compute the negative gradient of the loss with respect to the current predictions - the "pseudo-residuals." For squared error these are literally y - F(x) .

3. Fit a small regression tree to those pseudo-residuals.

4. For each terminal node, compute the optimal constant value (a line search, or a Newton step).

5. Update: `F_m(x) = F_{m-1}(x) + ν · h_m(x)`, where ν is the learning rate / shrinkage.

6. Stop at M, or by early stopping on a validation set.

Where's the "gradient"? It's gradient descent in function space - each tree is a step in the direction that most reduces the loss. That framing is why you can plug in any differentiable loss: squared error, absolute error, log-loss, and - crucially for insurance - Poisson, Gamma, and Tweedie deviance. Being able to boost a Tweedie objective is a genuinely useful thing to mention.

Prediction is the initial constant plus ν times every tree's output - an additive sum, not an average. That's the structural difference from RF.

### Q: How is GBM different from random forest?

> 📖 **对应讲解**：[[8.6 Bagging VS Boosting]]

> 💬 **中文精讲**：考结构性对照，不是背差异清单。骨架按表念八行：树的形状（深、独立 vs 浅 2–8、每棵拟合前面集成的误差）、组合方式（平均 / 投票 vs 加权加性求和）、主要攻什么（方差 vs 偏差）、加树（趋平、不降 vs 会过拟合、M 必须调或早停）、并行性（跨树可并行 vs 天生顺序，只有树内的分裂搜索可并行）、超参数敏感度（低 vs 高）、表格数据上的精度（很好 vs 通常更好）、随机性（必需 vs 可选但有用）。
> 一句话根因是「**平均 vs 累加**」：平均让加成员只降方差、不改期望，所以 RF 的树数是单调的；累加让每棵树都在追上一轮的误差，错误会累积，所以 GBM 的树数有内部最优、早停是必需品。把这条根因说出来，这张表就不用死记了。

|  | Random Forest | GBM |
| --- | --- | --- |
| Trees | Deep, grown independently | Shallow (depth 2–8), each fit to the previous ensemble’s errors |
| Combination | Average / vote | Weighted additive sum |
| Primarily attacks | Variance | Bias |
| More trees | Plateaus, never overfits | Will overfit — M must be tuned or early-stopped |
| Parallelizable | Yes, across trees | No — sequential by construction (only the split search within a tree parallelizes) |
| Hyperparameter sensitivity | Low | High |
| Typical accuracy on tabular data | Very good | Usually better |
| Randomness | essential (`bootstrap` + feature subset) | optional (`subsample`, colsample) but helps |

### Q: What are the important hyperparameters?

> 📖 **对应讲解**：[[8.4.2.2 Gradient Boosting]]

> 💬 **中文精讲**：考参数的分组与优先级。骨架按三组念：核心一对——`learning_rate`（ν，典型 0.01–0.3）与 `n_estimators` 直接互换，学习率减半大致要把树数翻倍，低学习率加多树泛化更好、代价是时间线性增长（所以把树数设高、让早停替你选）；树本身——`max_depth` 典型 3–8，最重要的一句是「深度 d 允许最多 d 阶交互，depth 1 就是纯加性模型」，`min_child_weight` / `min_data_in_leaf` 是叶子噪声的主要控制（LightGBM 用 `num_leaves` 顶替深度），`gamma` / `min_split_loss` 剪掉弱分裂；正则与随机性——`subsample`、`colsample_bytree`、`reg_lambda` / `reg_alpha`、`scale_pos_weight`、`monotone_constraints`、`max_delta_step`。
> 两个加分点：`monotone_constraints` 要单独讲一句——它是让 GBM 能被监管和信贷报备接受的旋钮；「深度就是交互阶数」这个说法一句话就证明你理解树模型在表达什么。

Boosting parameters — the pair that matters most:

| Parameter | Effect |
| --- | --- |
| `n_estimators` / `num_boost_round` | Number of boosting iterations. Too few underfits, too many overfits. Set it high and let early stopping choose |
| `learning_rate` (ν, shrinkage) | Typically 0.01–0.3. Trades off directly with `n_estimators`: halve the learning rate and you need roughly twice the trees. Lower rate + more trees generalizes better, at linear cost in time |

Tree-specific parameters:

| Parameter | Effect |
| --- | --- |
| `max_depth` | 3–8 typical. Controls the interaction order the model can express — depth d permits up to d-way interactions, and depth 1 (stumps) gives a purely additive model. That framing is a strong thing to say |
| `min_child_weight` / `min_samples_leaf` / `min_data_in_leaf` | Minimum observations (or sum of Hessians) in a leaf. The main leaf-noise control |
| `num_leaves` (LightGBM) | LightGBM grows leaf-wise, so this replaces depth as the capacity knob |
| `gamma` / `min_split_loss` | Minimum loss reduction required to make a split — prunes weak splits |

Regularization and stochasticity:

| Parameter | Effect |
| --- | --- |
| `subsample` | Row sampling per tree (0.5–0.8) → stochastic gradient boosting. Reduces variance and speeds training |
| `colsample_bytree` / `_bylevel` / `_bynode` | Feature sampling — RF-style decorrelation |
| `reg_lambda` (L2), `reg_alpha` (L1) | Penalties on the leaf weights |
| `scale_pos_weight` | Class imbalance |
| `monotone_constraints` | Force a variable’s effect to be monotone. Critical for insurance and credit filings — this is how you make a GBM acceptable to a regulator |
| `max_delta_step` | Stabilizes logistic boosting under extreme imbalance |

### Q: What's your tuning strategy?

> 📖 **对应讲解**：[[8.4.2.2 Gradient Boosting]]

> 💬 **中文精讲**：考你有没有真跑过——要的是**有序配方**而不是清单。骨架六步：固定 `learning_rate` = 0.1，用验证集早停定一个合理的树数（快基线）→ 一起调树复杂度（`max_depth` 或 `num_leaves` 与 `min_child_weight`，这是偏差 / 方差的主旋钮）→ 调随机性（`subsample`、`colsample_bytree`）→ 调正则（`gamma`、`reg_lambda`、`reg_alpha`）→ 最后把学习率降到 0.01–0.05、加树加早停去榨最后几个点 → 第 2–4 步用随机搜索或贝叶斯优化代替全网格，全程配 CV 与早停。
> 顺序本身就是答案：先定树数，再复杂度、再随机性、最后正则与低学习率——低学习率放最后做，前面的搜索才便宜。被追问「为什么不用全网格」：维度爆炸，而且大量参数组合的收益可以忽略。

Give an ordered recipe, not a list - it shows you've actually done this:

1. Fix `learning_rate` = 0.1 and use early stopping on a validation set to find a reasonable number of trees. Fast baseline.

2. Tune tree complexity: `max_depth` (or `num_leaves`) and `min_child_weight` together - they're the primary bias/variance knobs.

3. Tune stochasticity: `subsample` and `colsample_bytree`.

4. Tune regularization: `gamma`, `reg_lambda`, `reg_alpha`.

5. Finally lower the learning rate to 0.01-0.05 and re-run with more trees and early stopping for the last few points of performance.

6. Use random search or Bayesian optimization over steps 2-4 rather than a full grid, always with CV and early stopping.

### Q: Why is GBM more sensitive to hyperparameters than RF?

> 📖 **对应讲解**：[[8.6 Bagging VS Boosting]]

> Because in a random forest the trees are fit independently and averaged, so mistakes are self-correcting - but in a GBM every tree is fit to the previous ensemble's errors, so mistakes compound.

> 💬 **中文精讲**：这一问其实在考「你理解 bagging 与 boosting 的本质差异吗」。骨架：引用句先给根因（RF 的树独立拟合再平均，错误会互相抵消；GBM 每棵树都拟合前面集成的误差，错误会累积），再三点展开——学习率过高或树太深就开始拟合噪声，之后每一棵树都在这层噪声上继续搭；`n_estimators` 有内部最优（少了欠拟合、多了过拟合），而 RF 里它是单调的，所以「树多加就完了」是免费的；因此早停对 GBM 必需、对 RF 无关，「更多树不会有坏处」的直觉不能照搬。
> 这其实是 §7「平均 vs 累加」的直接推论——答的时候显式接上那条根因，整篇会显得自洽，而不是在背两张清单。

- Too high a learning rate, or too deep a tree, and the ensemble starts fitting noise - and then every subsequent tree builds on top of that noise.

- There's no safe direction: too few trees underfits, too many overfits. `n_estimators` has an interior optimum. In RF it's monotone, so “just use more trees” is free.

- Which is why early stopping is mandatory for GBM and irrelevant for RF, and why RF’s “more trees can’t hurt” intuition does not transfer.

### Q: What does XGBoost add over traditional GBM?

> 📖 **对应讲解**：[[8.4.2.2 Gradient Boosting]]

> 💬 **中文精讲**：考增量知识，至少要能报出四件事。骨架三层：算法上——正则化目标（L1/L2 惩罚写进损失函数内部，正则成为优化的一部分而不是事后附加）、二阶（牛顿）优化（用梯度与 Hessian，叶值和分裂增益算得更准、迭代更少）、原生稀疏 / 缺失处理（每个分裂学一个默认方向，不必先插补）、加权分位草图的近似分裂查找、先长到 `max_depth` 再按 `gamma` 回头剪（能捞出藏在坏分裂后面的好分裂）；工程上——并行分裂查找、缓存友好的访存、out-of-core、分布式、GPU；实用上——内置 CV 与早停、单调与交互约束、含 Poisson / Gamma / Tweedie 的宽目标库。
> 措辞注意本题的编辑注：默认是 `grow_policy=depthwise`（按层生长）而不是 DFS 深度优先，实质结论（先长满再回头剪 vs 逐节点贪心早停）不变，但按官方参数名说更稳。收尾用一句对比 LightGBM（leaf-wise 生长、GOSS 采样、原生类别变量）与 CatBoost（ordered boosting 对抗编码泄漏、最强的类别处理）。

Algorithmic:

- Regularized objective - explicit L1/L2 penalties on the leaf weights are inside the loss function, so regularization is part of the optimization rather than bolted on afterward.

- Second-order (Newton) optimization - uses the gradient and the Hessian, so leaf values and split gains are computed more accurately and it converges in fewer iterations.

- Native sparsity / missing-value handling - learns a default direction per split for missing values, rather than requiring imputation.

- Approximate split finding via a weighted quantile sketch and histogram binning, instead of scanning every candidate split point.

- Grows the tree to `max_depth` **first, then prunes backward** by `gamma`, rather than greedily stopping at each node - so it can find a good split hiding behind a bad one.
  > 📝 **[编辑注 4]** 原文写作 "Depth-first growth"；XGBoost 默认是 `grow_policy=depthwise`（**按层生长**），不是 DFS 式深度优先。**实质结论不变**（先长满再回头剪 vs 逐节点贪心早停），但用词按官方参数名说更稳。

Engineering:

- Parallelized split finding, cache-aware access patterns, out-of-core computation for larger-than-memory data, distributed training, GPU support.

Practical:

- Built-in CV and early stopping; monotonic and interaction constraints; a wide objective library including Poisson, Gamma, and Tweedie.

Worth one sentence: LightGBM adds leaf-wise growth, GOSS sampling, and native categorical handling - usually faster on large data. CatBoost adds ordered boosting (to fight target leakage in encoding) and the strongest categorical handling.

### Q: How does GBM compute variable importance?

> 📖 **对应讲解**：[[7.4. Quantifying Feature Importance]]

> 💬 **中文精讲**：考你会不会挑对那个指标。骨架按表念：`gain`（该特征所有分裂带来的损失总下降）是默认、也是最该引用的一个；`cover`（被该特征的分裂覆盖的样本数）次要；`weight` / `frequency`（被用来分裂的次数）偏向高基数的连续变量，**不要引用**；外部校验用置换重要性（模型无关、相关下不可靠）与 SHAP。
> 这题真正要讲的是 gain 与 SHAP 的分工：gain 只说贡献多少、不说方向；SHAP 给大小和方向、能落到单条记录（所以能解释某一次拒保、或某一张保单的费率），还支持依赖图与交互图。要给监管或客户解释树模型，答案就是 SHAP 加单调约束；RF 那三条 caveat 在这里同样成立。

| Measure | What it is | Verdict |
| --- | --- | --- |
| gain | total improvement in the loss contributed by all splits on that feature. For squared-error loss this is the “squared error improvement” that H2O reports | The default and the one to quote |
| cover | number of observations touched by splits on that feature | Secondary |
| weight / frequency | how many times the feature was used to split | Biased toward high-cardinality continuous variables. Don’t quote this one |
| Permutation | out-of-sample drop when the feature is shuffled | Model-agnostic; unreliable under correlation |
| SHAP | additive per-record attributions | The modern answer |

The distinction to draw: gain tells you how much a feature contributed, but not in which direction. SHAP gives magnitude and direction, works at the individual-record level (so you can explain a single decline or a single rate), and supports dependence and interaction plots. If asked how you'd explain a tree model to a regulator or a customer, SHAP plus monotonic constraints is the answer.

Same caveats as RF: correlated features share credit, importance isn't significance, and none of it is causal.

### Q: How do you keep a GBM from overfitting?

> 📖 **对应讲解**：[[8.4.2.2 Gradient Boosting]]

> 💬 **中文精讲**：考清单的优先级，第一项必须是早停。骨架按影响力排序念八条：验证集早停（单条最重要）→ 低学习率配更多树 → 浅树（`max_depth` 3–6）→ `min_child_weight`，别让叶子太小 → `subsample` 与 `colsample_bytree` 注入随机性 → 叶权重的 L1 / L2 与 `gamma` 剪枝 → 每个调参决定都走 CV 而不是单次划分 → 单调约束（注入真先验知识，几乎零可信度代价）。
> 最后一条在保险语境下最值钱——它同时是正则和「让监管 / 客户接受」的手段，别只当技术项。被追问「哪一条最有效」就直接答早停，其余都是配合它工作的。

The checklist, roughly in order of impact:

1. Early stopping on a validation set - the single most important one.

2. Low learning rate with more trees.

3. Shallow trees (`max_depth` 3-6).

4. `min_child_weight` - don’t let leaves get tiny.

5. `subsample` and `colsample_bytree` - stochasticity.

6. L1/L2 on the leaf weights, and `gamma` pruning.

7. Cross-validation rather than a single split for every tuning decision.

8. Monotone constraints - injecting real prior knowledge is regularization that costs you nothing in credibility.

## 10. STAR Answers

> 📖 **对应讲解**：[[03. 答题模板与追问应对]]

### 10a. STAR adapted to technical questions

The source sheet's insight: use STAR to structure your answer about a method, not just about a past project. It stops you from rambling and guarantees you cover the parts the interviewer is scoring.

| Letter | Standard | Technical version |
| --- | --- | --- |
| S — Situation | The context | What kind of problem is this method for? |
| T — Task | What you had to do | What does it predict / how does it work? |
| A — Action | What you did | How does it work, what does it assume, and what do you do as a DS — transformations, diagnostics, tuning |
| R — Result | The outcome | What does it output, how do you interpret it, and what are the key values? |

#### Worked example — Logistic Regression

- S - Binary outcomes where I need an interpretable, well-calibrated probability and where stakeholders or regulators need to see what drives each effect: retention/lapse, quote conversion, fraud referral, large-loss propensity.

- T - It models the log-odds of the event as a linear function of the predictors, and outputs a probability between 0 and 1 through the logistic link.

- A - It assumes independent observations, a binomial response, linearity on the logit scale, no severe multicollinearity, and enough events per predictor. So in practice I check empirical logit plots and transform or bin anything non-linear, WoE-encode high-cardinality categoricals, screen VIF, watch for quasi-complete separation in sparse categorical levels, decide how to handle class imbalance, and pick the threshold from the business cost matrix rather than defaulting to 0.5.

- R - A coefficient per variable, where exp(β) is an odds ratio, plus a predicted probability per record. I evaluate with AUC/Gini, a lift chart read at the operating capacity, and a calibration plot - and before anyone sees it, I check that every coefficient's sign is directionally sensible to the business.

#### Worked example — Random Forest

- S - When I need accuracy on tabular data with non-linear effects and interactions, and full coefficient-level interpretability isn't the deliverable: internal triage, propensity scoring, or discovery work ahead of building a GLM.

- T - It fits many deep decision trees, each on a `bootstrap` sample and each considering only a random subset of features at every split, then averages them. It predicts by averaging the trees' outputs (or their class probabilities).

- A - Almost no distributional assumptions - no linearity, no normality, no scaling, and it's invariant to monotone transforms of the predictors. Its real assumption is that the trees are decorrelated enough for averaging to reduce variance, which is what the feature subsetting buys. As a DS I tune `max_features` and `n_estimators` together, set `min_samples_leaf` to control leaf noise, use `class_weight` for imbalance, watch OOB error to confirm I have enough trees, and keep in mind that it cannot extrapolate beyond the training range.

- R - A prediction per record plus feature importance. I read importance with care - MDI is biased toward high-cardinality variables and correlated features split their credit - so I use permutation importance or SHAP on held-out data, and partial dependence plots to see the shape of each effect rather than just its size.

Build one of these for logistic regression, GLM, random forest, GBM/XGBoost, PCA and regularization and you can handle any “tell me about X” question in the technical interview.

### 10b. Behavioral STAR

The template. Keep Action longest - it's what's actually being scored - and always quantify Result.

- S (1-2 sentences): the business context and why it mattered. Name the stakes.

- T (1 sentence): your specific responsibility. Not the team's.

- A (the bulk): what you did, in order, including the judgment calls and the tradeoffs you weighed. Use "I," not "we."

- R (2-3 sentences): the quantified outcome, plus what you learned or changed afterward.

Question bank - have a prepared story for each:

| Question | What they’re testing | Pick a story where… |
| --- | --- | --- |
| Tell me about a model you built end to end | Breadth, ownership | you touched data, modeling, validation, and handoff |
| A time you found a serious data quality problem | Rigor, skepticism | you caught it before it hit production, and quantified the impact |
| A time your model underperformed or was rejected | Humility, resilience | you diagnosed why and what you changed — never “the data was bad” |
| A time you had to explain a technical result to a non-technical audience | Communication | you changed a decision, and you can describe the visual or analogy you used |
| A time you disagreed with a stakeholder or a senior colleague | Conflict, judgment | you were persuaded by evidence, or you persuaded with it |
| A time you had to deliver under a hard deadline with incomplete data | Prioritization | you shipped something defensible and documented the limitations explicitly |
| A time you simplified something | Pragmatism | you gave up accuracy for interpretability or maintainability on purpose |
| A time you automated or improved a process | Initiative | you can quantify the hours or errors saved |
| Tell me about a time you were wrong | Self-awareness | you found it yourself and told someone |
| A time you mentored or unblocked someone | Collaboration | the other person’s outcome improved |

Mistakes that cost points: saying "we" throughout so nobody can tell what you did; a Situation that eats half the answer; no number in the Result; a "failure" story with no failure in it; and blaming other people or the data.

Self-practice prompt:

> “Act as a behavioral interview coach and ask me 5 questions using the STAR method. After each answer, give feedback on how well I covered Situation, Task, Action, and Result.”

## Numbers to Memorize

> 📖 **对应讲解**：[[04. 公式速查卡]]

These get asked as direct factual questions. There are about 25 of them.

| Quantity | Value |
| --- | --- |
| VIF — concern / serious | > 5 / > 10 |
| √VIF | the factor by which the standard error is inflated |
| Condition index — moderate / severe | > 10 / > 30 |
| Kaiser criterion for keeping a PC | eigenvalue > 1 (on the correlation matrix) |
| Cumulative variance to retain in PCA | 80–95% |
| Information Value bands | <0.02 useless · 0.02–0.1 weak · 0.1–0.3 medium · 0.3–0.5 strong · >0.5 check for leakage |
| Events per variable (EPV) for logistic regression | 10–20 |
| Minimum WoE bin size | ≥ 5% of the population |
| Typical winsorizing cuts | 1st/99th or 5th/95th percentile |
| Standard train/val/test split | 60/20/20 or 70/15/15 |
| Standard k for cross-validation | 5 or 10 |
| Out-of-bag fraction per `bootstrap` | 1/e ≈ 36.8% |
| AUC — random / useful / strong | 0.5 / > 0.7 / > 0.8（📝 **经验分档**，各机构口径不一致） |
| Gini from AUC | Gini = 2·AUC − 1 |
| RF `max_features` defaults | √p classification, p/3 regression |
| RF `n_estimators` typical | 300–1000 (more never overfits) |
| GBM `learning_rate` typical | 0.01–0.3 |
| GBM `max_depth` typical | 3–8 |
| GBM `subsample` / colsample typical | 0.5–0.8 |
| Learning rate ↔ tree count | halve the rate → roughly double the trees |
| GBM depth ↔ interactions | depth d → up to d-way interactions; depth 1 = additive |
| Bagging variance formula | ρσ² + (1−ρ)σ²/B |
| Bias-variance decomposition | E[(y−f̂)²] = Bias² + Var + σ² |
| AIC / BIC | −2logL + 2k / −2logL + k·ln(n) |
| Tweedie power for pure premium | p ≈ 1.5 (between 1 and 2) |
| Case-control intercept correction | β₀ − ln(r₁/r₀) |
| Coefficient on a well-binned WoE variable | ≈ 1.0（📝 按本文 `ln(%Events/%NonEvents)` 约定；**换成相反定义则为 −1**，见编辑注 3） |

> 💬 **中文精讲**：这 27 个数字要分三类用：**恒等式**（Gini = 2·AUC − 1、bagging 的 `ρσ² + (1−ρ)σ²/B`、偏差-方差分解、AIC / BIC、1/e ≈ 36.8%、√VIF 是标准误的放大倍数、容差 = 1/VIF、病例对照截距校正 `β₀ − ln(r₁/r₀)`）必须一字不差，说错就是硬伤；**经验起手值**（VIF > 5 / > 10、条件指数 > 10 / > 30、IV 分档、EPV 10–20、WoE 最小箱 ≥ 5%、winsorize 1/99 或 5/95、划分 60/20/20、k = 5 或 10、AUC 三档、各超参数的典型区间）要说成「常用惯例」而不是定律；**数据依赖的超参数**（Tweedie 的 p、学习率与树数这一对、`max_features` 的默认值）要主动声明它是调出来的。
> 不掉坑的答法是「报数字 + 报口径」：说经验值时补一句「随数据与场景可调」；说 WoE 系数 ≈ 1.0 时先声明自己的定义方向（见 §2 WoE 那题的编辑注）；说 Tweedie 的 p ≈ 1.5 时说明它是 `1 < p < 2` 上用 CV 选的（见 §1「保险目标选分布」那题的编辑注，实测项目选到 1.9）。数字答对再加口径清楚，比多背一个定义更能拉开差距。

## Cross-Topic Connections

> 📖 **对应讲解**：[[00 Index]]

The interview will test whether you see these as one subject rather than ten. Each line below is a place where two topics are the same idea:

- Quasi-complete separation (§1) and undefined WoE (§2) are the same problem — a categorical level with zero events. One shows up as an infinite coefficient, the other as a log of zero. Both are fixed by merging bins or adding a penalty.

- High cardinality is the bridge from §1 to §2. Separation is why WoE and level-collapsing exist.

- Multicollinearity (§4) is why feature selection (§5) exists, and penalized regression appears in both: ridge as the collinearity fix, lasso as the selection method, elastic net as both.

- PCA appears twice: as a collinearity fix (§4) and as feature projection (§5). VarClus is its selection counterpart - same PCA machinery, but it returns original variables instead of components.

- Spearman + Hoeffding's D (§5) is a transformation-detection tool (§2), not just a screening tool. Low Spearman with high D means “needs a transform,” not "drop it."

- Bias-variance (§7) is the theory; over/underfitting (§6) is the diagnostic; regularization and hyperparameters (§8, §9) are the controls. Same concept, three vocabularies.

- Bagging's variance formula (§7) explains `max_features` (§8). The ρσ² floor is why random forests subset features at all.

- Leakage (§6) is created by §2, §3, and §5 - WoE coding, imputation, capping, and feature selection are all supervised, parameter-learning steps, and every one of them leaks if fit before the split.

- Missing-as-informative (§3) and WoE's missing bin (§2) are the same practical answer: don't impute, let "missing" be its own level with its own log-odds.

- GLM vs GBM (§1, §7) resolves the same way every time: use the tree model to discover shape and interactions, encode them in the model you actually file.

> 💬 **中文精讲**：这 10 条连线的价值不在知识量，而在「把它当一个学科，而不是十个知识点」——每一条都是同一个想法在两个章节里出现。面试官往深里追问时，能主动接一句「这其实就是 §x 那个问题」的人，明显比只会单点定义的人高一层。最值钱的三条：准完全分离与 WoE 无定义是同一个病（某一层零事件，一个表现为系数跑到无穷、一个表现为 log(0)）；泄漏是 §2 / §3 / §5 造出来的（WoE、插补、封顶、特征选择全是学参数的监督步骤）；bagging 的 ρσ² 地板正是随机森林要抽特征的全部理由。
> 用法上：把这张表当「追问的出口」而不是背诵清单——每题被追到第二层、第三层时，接一条连线就是加分区。逐题的深潜落点见 `[[MAGNet 答案对照表]]`，本表只讲「为什么它们本质上是一件事」。

## Self-Test (No Answers)

> 📖 **对应讲解**：[[99.1 题库总览]]

Cover the document. If you can answer these out loud in 60-90 seconds each, you're ready.

### Tier 1 — you will be asked these

1. What is a logit, and what does exp(β) mean?

2. Why do we need a link function? Why is the log link standard in insurance?

3. Name the GLM assumptions and one diagnostic for each.

4. Your target is 0.5% positive. Walk me through how you'd handle it.

5. Compute precision, recall, and accuracy from a 2×2 I give you - and tell me which one to trust.

6. What does multicollinearity do to a model? Does it hurt predictions?

7. What's VIF, and what threshold worries you?

8. Explain WoE and IV. What are the IV cutoffs?

9. What is data leakage? Give me three concrete examples.

10. Explain the bias-variance tradeoff and write the decomposition.

11. Bagging vs. boosting - which reduces bias, which reduces variance, and why?

12. Walk me through the random forest algorithm.

13. Walk me through gradient boosting. Where is the "gradient"?

14. Name the key GBM hyperparameters and how you'd tune them, in order.

15. GLM or GBM for a rating model? Defend it.

### Tier 2 — the differentiators

16. What's quasi-complete separation, how do you spot it, and how do you fix it?

17. Why is back-transforming a log-transformed prediction biased, and what does a GLM do instead?

18. Difference between MAR and MNAR, with an insurance example of each.

19. What's wrong with mean imputation?

20. Why does L1 produce exact zeros and L2 not?

21. Spearman is near zero but Hoeffding's D is high. What do you do?

22. Why does a random forest subset features at every split? What would happen if it didn't?

23. Why doesn't RF overfit with more trees, while GBM does?

24. Why is GBM more hyperparameter-sensitive than RF?

25. What does XGBoost add over classical GBM? Name four things.

26. How do you read a lift chart, and what would make you distrust one?

27. Everything you'd list against using stepwise selection.

28. When is out-of-time validation required rather than random splitting?

29. PCA vs VarClus — what's the actual difference, and when would you pick each?

30. Your model has AUC 0.96 on a hard problem. What's your first thought?

### Tier 3 — behavioral

31. Give a STAR answer for a model you built end to end.

32. Give a STAR answer for a time your model was rejected.

33. Give the technical STAR for logistic regression, then for GBM.

---

## 导入说明（本次导入所做的改动）
### 技术核验修订（编辑注）

导入后对全文做了**技术核验**（依据 PSL / Loss Data Analytics / XGBoost 官方参数命名），发现 **6 处**需要标注但不改动原观点的地方，全部以 `> 📝 **[编辑注 N]**` 就地标注：

| # | 位置 | 原文 | 标注原因 |
|---|---|---|---|
| **1** | §1 分离的修复 · 第 3 条「加惩罚」 | "L2/ridge always yields a finite solution" | 就**固定 λ > 0** 而言正确，但 **CV 常选出 λ = 0**，此时惩罚失效 —— 分离的根因不是过拟合（PSL §10.3.1）。真正的解法是 Firth / 贝叶斯先验；且分离主要破坏**推断**而非决策边界 |
| **2** | §1 保险目标选分布 | Tweedie `p ≈ 1.5` | `p` 是 **`1 < p < 2` 上 CV 调的超参数**，不是常数；实测 678,013 份保单的项目选出 **p = 1.9** |
| **3** | §2 WoE | 理想系数 ≈ +1.0 | 这是本文 `ln(%Events/%NonEvents)` 约定下的结果；**相反约定 `ln(%Goods/%Bads)` 下是 −1** —— 两约定互为相反数，关键在「方向与符号配套」 |
| **4** | §9 XGBoost 增量 | "Depth-first growth" | XGBoost 默认 `grow_policy=depthwise`（**按层生长**），非 DFS；实质结论（先长满再剪 vs 贪心早停）不变，用词按官方参数名更稳 |
| **5** | Numbers to Memorize | AUC 三档 / WoE 系数 | AUC 的 0.5 / >0.7 / >0.8 是**经验分档**（各机构不一致）；WoE 系数 ≈ 1.0 需带约定说明 |

**未改动之处**：全部技术观点、55 道题的答案、27 个数字、10 条 Cross-Topic 连线均**保持原样**。经比对 35 张照片的原始 OCR（1628 个文本块），**合并稿没有丢内容** —— 唯一对不上的都是 OCR 识别噪声（如 `serlous` = serious、`teatures` = features、`bootstran samnle` = bootstrap sample）。


本文件由 `IMG_9774–IMG_9808`（35 张照片）的 OCR 结果合并而成，技术上未作修订。为便于使用，导入时只做了三处处理，全部在此声明：

| # | 位置 | 处理 | 依据 |
|---|---|---|---|
| 1 | §6 连续型指标表 · **MAE** | 原合并稿标注「公式排版残缺·未补写」。**已由 OCR 片段恢复**：OCR 该行给出 `Σ`、`y−ŷ`，据此补为 `(1/n)·Σ|y − ŷ|` | `.transcription/IMG_9794.json` 文本块 `MAE` / `Y-9` / `У-Ỹ` |
| 2 | §6 连续型指标表 · **MAPE** | 同上。**已由 OCR 片段恢复**：OCR 给出 `(100/n)Σ`，据此补为 `(100/n)·Σ|(y − ŷ)/y|` | `.transcription/IMG_9794.json` 文本块 `МАPЕ` / `* (100/n)ž` |
| 3 | §6 Gain chart 的 **45° 线** | **照片在此处真实截断**（IMG_9794 底部），无法从原图恢复。已用通用定义补写一句并加注 `[补写]`，请按需要复核 | 照片截断，非 OCR 问题 |
| 4 | 全篇 55 题 | **新增 `💬 中文精讲`**（逐题 2–4 句中文，讲清「问什么 / 答案骨架 / 会被追问到哪」） | 用户要求「中文讲解为主」；英文答案句保留不动，因为那是面试要说出口的话 |

**交叉引用**：本文件引用的 `magnet-study-guide.md`（resource/link index）**未随照片提供**，库内不存在该文件；如需请另行补入。

## 与库内其他笔记的分工

| 层 | 文件 | 定位 |
|---|---|---|
| **主线** | 本文件 | 面试真正会问什么、先说哪句、陷阱在哪 |
| **深潜** | 第 1–10 章 | 每个考点的推导、代码、保险语境 |
| **对照** | [[MAGNet 答案对照表]] | 逐题 → 深潜笔记的映射 |
| **自测** | [[99.1 题库总览]]（90 题） | 四块式：正确答案 / 错误答案 / 追问链 |
| **速览** | [[04. 公式速查卡]]、[[05. 高频追问 TOP 30]]、[[06. 中英术语对照表]] | 面试前 30 分钟 |
