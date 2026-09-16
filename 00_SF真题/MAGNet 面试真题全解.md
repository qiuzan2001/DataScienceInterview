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

> The logit is the natural log of the odds, and it's the transformation that lets us model a bounded probability with an unbounded linear predictor.

- If p = P(Y=1), then odds = p/(1-p), and logit(p) = ln(p/(1-p)).

- It maps (0,1) → (-∞, +∞). That's the whole point: Xβ can be any real number, but a probability can't, so we model the unbounded transform of p instead of p itself.

- Inverse (the logistic / sigmoid): p = 1/(1 + e^(-Xβ)) = e^(Xβ)/(1 + e^(Xβ)).

### Q: Write the logistic regression equation and interpret a coefficient.

> `ln(p/(1-p)) = β₀ + β₁x₁ + ... + βₖxₖ`. A one-unit increase in xⱼ adds βⱼ to the log-odds, which means it multiplies the odds by exp(βⱼ), holding everything else fixed.

- exp(βⱼ) is the odds ratio. exp(β) = 1.25 → a one-unit increase raises the odds by 25%.

- For a dummy variable, exp(β) is the odds ratio versus the reference level.

- **TRAP:** saying "βⱼ is the change in probability." It is not. The change in probability depends on where you sit on the S-curve - the same β moves p a lot near 0.5 and almost nothing near 0.02. If they want a probability effect, quote a marginal effect at a stated base rate, or just show two predicted probabilities.

### Q: What is a link function and why do we need one?

> A GLM has three pieces - a response distribution from the exponential family, a linear predictor η = Xβ, and a link function g that connects them via g(E[Y]) = η. The link is what keeps predictions in the valid range while letting the predictors act linearly.

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

| Target | Distribution | Why |
| --- | --- | --- |
| Claim frequency (count per exposure) | Poisson, log link, exposure as offset | counts, variance ≈ mean |
| Frequency, overdispersed | Negative Binomial, or Poisson with a scale parameter | variance > mean |
| Claim severity (avg cost per claim) | Gamma, log link | positive, right-skewed, variance ∝ μ² |
| Pure premium / loss cost in one step | Tweedie, log link, power p ≈ 1.5 | point mass at zero + continuous positive tail |
| Binary (lapse, conversion, fraud flag) | Binomial, logit | 0/1 |

- The offset point is worth volunteering: for frequency you model counts with log (exposure) as an offset (coefficient fixed at 1), not as a predictor, so the model is a rate per exposure.

- Tweedie is the one that impresses. It handles the fact that most policies have zero loss and the rest have a continuous positive amount, so you avoid fitting frequency and severity separately.

### Q: How do you deal with an unbalanced sample?

> My default is to not rebalance the data - I fix the decision threshold instead. What matters isn't the ratio, it's the absolute number of events.

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

> Separation means a predictor (or combination) perfectly predicts the outcome, so the maximum likelihood estimate doesn't exist - the likelihood keeps improving as the coefficient runs to infinity.

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

3. Add a penalty - L2/ridge always yields a finite solution; L1 also works.

4. Firth's penalized likelihood - purpose-built, keeps the variable.

5. Bayesian priors on the coefficients (weakly informative, e.g. a Cauchy prior).

6. Drop the variable if it adds nothing after binning.

7. Exact logistic regression for very small samples.

### Q: Explain a confusion matrix.

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

- Every metric in that table depends on a chosen threshold. Only AUC / PR-AUC are threshold-free.

### Q: How do a linear model and a GLM differ?

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

> On the predictor side, whenever the relationship isn't linear on the link scale, or the variable is too skewed/high-cardinality/outlier-prone to use raw. On the response side - in a GLM, almost never, because you choose the distribution and link instead.

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

> I let the data show me the shape first, then pick the simplest parametric form that reproduces it.

1. Empirical logit plot (binary target) - the workhorse. Bin the predictor into deciles, and per bin compute ln( (events + 0.5) / (non-events + 0.5)), then plot against the bin mean. A straight line means use the variable as-is; curvature tells you the transform; a U shape means no monotone transform will do and you need bins or a spline.

2. Binned mean-response / target analysis plot - same idea for continuous targets.

3. LOESS smooth of residuals vs. each predictor - leftover structure = missing transform.

4. Spearman + Hoeffding's D together - see §5; low Spearman with high D is the signature of a real but non-monotone relationship.

5. Fit a GAM first, look at the fitted smooth, then approximate it parametrically.

6. Fit a GBM, read the partial dependence / SHAP dependence plot, then encode that shape into the GLM. This is the standard modern insurance workflow - the tree model is the discovery tool, the GLM is the deliverable.

7. Box-Cox / Yeo-Johnson to let ML pick a power transform for you.

### Q: Explain Weight of Evidence coding.

> WoE replaces each bin of a predictor with the log-odds contribution of that bin, so a high-cardinality or non-linear variable becomes a single numeric column that is already on the logit scale.

For a binary target and a binned predictor:

```text
WoE_i = ln((Events_i / Total Events) / (NonEvents_i / Total NonEvents))
      = ln(% of all events in bin i / % of all non-events in bin i)
```

- Positive WoE → that bin has proportionally more events than the population. Negative → fewer.

- Because it's already on the log-odds scale, dropping a WoE-coded variable into a logistic regression should produce a coefficient near 1.0. If it comes out far from 1, your binning is off or the relationship isn't stable. Great detail to mention.

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

> k levels become k-1 indicator columns plus a reference level; each coefficient is the contrast against that reference.

- Use k-1, not k. All k dummies plus an intercept is the dummy variable trap - perfect multicollinearity, the design matrix is singular.

- Choose the reference level deliberately: the largest / most stable level, so every other estimate is a contrast against something well-estimated.

- Problems at high cardinality: the design matrix explodes, sparse levels give unstable estimates, and empty cells give quasi-complete separation.

- Greenacre method - an agglomerative way to collapse nominal levels by minimizing the loss of chi-square association, so you reduce cardinality before dummy coding. Know the name and what it's for.

- Alternatives: WoE, target/mean encoding (same leakage caveat, needs smoothing and out-of-fold computation), ordinal coding if the levels really are ordered, hashing for very high cardinality, and native categorical handling in LightGBM/CatBoost.

### Q: Polynomials and Box-Cox.

Basic polynomial — add `x²`, `x³`.

- Pro: simple, stays inside a GLM, captures curvature.

- Cons: terms are severely collinear (center the variable or use orthogonal polynomials); extrapolates terribly; the tails wag to fit the middle; hard to interpret a single coefficient. Prefer splines above degree 2.

Box-Cox — `y^(λ) = (y^λ − 1)/λ` for `λ ≠ 0`, and `ln(y)` for `λ = 0`; λ chosen by maximum likelihood to get closest to normality and constant variance.

- Requires y > 0. Use Yeo-Johnson if you have zeros or negatives.

- λ = 1 no transform, 0.5 square root, 0 log, −1 inverse.

- **TRAP / FOLLOW-UP:** back-transforming a prediction is biased, because E[g(Y)] ≠ g(E[Y]) . If you model log(y) and exponentiate the prediction, you get a median, not a mean, and you need a smearing/Duan correction to get back to the mean. A GLM with a log link models log (E[Y]) directly and has no such problem. This is the core reason GLMs displaced transform-then-OLS in insurance.

### Q: What is capping/flooring and why do it?

> Winsorizing - replace values above a high percentile with that percentile's value, and below a low one likewise. You keep the record but limit how much a single extreme value can move the fit.

- Typical cuts: 1st/99th or 5th/95th percentile; or cap where the fitted relationship visibly flattens; or at a business-defined limit.

- Why: bounds leverage and influence; you don't have to delete the row; and it protects you at score time from values outside anything the model was trained on.

- Insurance-specific: cap large losses so one catastrophic claim doesn't drive severity coefficients, and handle the excess separately as an excess layer or cat load. Saying this shows domain fluency.

- **TRAP:** the caps are parameters learned from the training data. Deriving caps from the full dataset, or recomputing them on the test set, is leakage. Fit on train, apply everywhere.

### Q: Splines and GAMs - what are they and when do you reach for them?

> A spline is a piecewise polynomial joined smoothly at knots; a GAM is a GLM where each predictor gets its own smooth function instead of a single coefficient.

- Regression spline: fixed knots, fit by least squares/ML. Knot count and placement control flexibility.

- Natural cubic spline: cubic, with the extra constraint of being linear beyond the boundary knots - much better tail behavior than a raw polynomial. Usually the right default.

- Smoothing spline: a knot at every point, with a roughness penalty; smoothing parameter λ chosen by generalized cross-validation.

- GAM: `g(E[Y]) = β₀ + f₁(x₁) + f₂(x₂) + ...`. Still additive, so you can plot each variable's effect and show it to a regulator - you keep the GLM interpretability story while dropping the linearity requirement. Does not capture interactions unless you add tensor/interaction smooths explicitly.

- When to reach for it: the relationship is clearly curved, you need interpretability, and you want monotonicity or shape constraints (monotone splines / shape-constrained GAMs).

- The trade-off to state: GAM sits between GLM and GBM - more flexible than GLM, more interpretable and constrainable than GBM, but it won't find interactions for you.

## 3. Missing Data

### Q: What are the missing-data mechanisms?

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

> Two or more predictors are close to being a linear combination of each other. Collinearity is the pairwise case; multicollinearity includes relationships among three or more variables that no pairwise correlation will reveal.

- Perfect multicollinearity → X′X is singular and the coefficients aren't identified at all (the dummy variable trap; including a variable and a rescaling of it; including all components of a sum along with the sum).

- Near multicollinearity is the practical problem, and it’s a matter of degree.

### Q: What are its effects? Does it hurt predictions?

> Estimates stay unbiased, but their variances blow up - so inference and interpretation break while predictive performance is essentially unaffected within the range of the training data.

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

| Tool | Threshold / reading |
| --- | --- |
| Pairwise correlation matrix | catches collinearity; misses 3+-variable multicollinearity |
| VIF = 1/(1−R²ⱼ) from regressing xⱼ on all other predictors | 1 = uncorrelated, >5 concern, >10 serious. √VIF = the factor by which the standard error is inflated |
| Tolerance = 1/VIF | < 0.1 is the mirror of VIF > 10 |
| Condition index from the eigenvalues of X′X (SAS COLLIN) | >10 moderate, >30 severe. The variance-decomposition proportions tell you which variables share the bad dimension |
| Symptom check | unstable signs, huge SEs, significant F with insignificant t’s |

The sequential VIF procedure: compute VIFs → drop or combine the worst offender → recompute → repeat until all are under threshold. Emphasize the recompute; VIFs change once you remove a variable.

### Q: How do you fix it?

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

> PCA finds a new orthogonal basis ordered by how much variance each direction explains, so you can keep a few components instead of many correlated variables.

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

> Five reasons, and only one of them is about accuracy.

1. Cost of collecting, purchasing, and maintaining extra variables - real money for third-party data.

2. Computation time - training, scoring, and refresh cycles.

3. Interpretability - nobody can review a 400-variable model, and you have to explain every variable in a filing.

4. Overfitting - more variables means more opportunity to learn sample-specific noise.

5. Reduced parameter precision - every added variable inflates the standard errors of the others (and this is the multicollinearity link).

Add the operational ones: fewer variables means less exposure to a data-source outage, easier documentation and filing, and a smaller surface for fairness/disparate-impact review.

### Q: Walk me through your variable reduction workflow on a wide dataset.

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

> L1 produces exact zeros; L2 shrinks toward zero but never reaches it.

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

> Yes - as a discovery tool, not as the selection rule.

- Why it helps: it captures non-linearity and interactions, so it surfaces variables a linear screen dismisses, and it needs no distributional assumptions.

- Why you can't just take the top N: importance isn't statistical significance and isn't causal; the ranking is unstable near the top; correlated features split credit; and a variable that's important to a GBM may only work in a GLM once you find the right transform.

- The right use: read the partial dependence / SHAP dependence plots to learn the shape and the interactions, then encode those explicitly as transforms and interaction terms in the GLM. The tree model tells you what to build; the GLM is what you file.

### Q: Selection vs. projection, supervised vs. unsupervised.

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

> Train fits the parameters, validation makes the choices, and test gives one unbiased read on generalization.

- Train - fit coefficients / grow trees.

- Validation - tune hyperparameters, select features, compare models, pick the threshold.

- Test - a single final estimate. Honest only if you touch it once.

- Typical: 60/20/20 or 70/15/15. With small data, use CV instead of a fixed validation set.

- Why three: the moment you use a dataset to make a decision, its error estimate becomes optimistic. A validation set that's been used for 200 hyperparameter comparisons is no longer an unbiased estimate of anything.

- Stratify on the target for classification, especially with rare events.

- Out-of-time split. If the model will be applied to the future, hold out a later time period, not a random subset. Random-splitting time-ordered data leaks the future into the past. In insurance you generally want both out-of-sample and out-of-time validation - out-of-sample tests the fit, out-of-time tests stability.

- Group/entity integrity. Keep all rows for a policy, household, or claim in the same partition; otherwise the entity's identity leaks across the split.

### Q: What is cross-validation? Which flavor for which data?

> Rotate the validation role through k folds and average, so every record is used for both fitting and validating - you get a lower-variance performance estimate without sacrificing training data.

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

> Leakage is any information in the training features that wouldn't be available at the moment you actually need to score - including information that leaked in through your own preprocessing.

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

> `E[(y − f̂(x))²] = Bias[f̂]² + Var[f̂] + σ²`

- Bias — error from the model being too simple or structurally wrong. Systematic; it doesn't go away with more data. High bias = underfitting.

- Variance - how much the fitted model changes if you'd drawn a different training sample. High variance = overfitting.

- σ² — irreducible noise. The floor. No model beats it.

- As complexity increases, bias falls and variance rises. Total error is U-shaped, and the goal is its minimum — not zero bias.

- What moves you along the curve: model flexibility, regularization strength, feature count, and training-set size (more data lowers variance, not bias).

### Q: Bagging vs. boosting vs. stacking.

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

> It depends on whether the deliverable is a decision or a filed rate. For anything that goes into a rate, the GLM's interpretability and monotonicity usually win; for internal triage and targeting, the GBM's accuracy wins.

GLM: interpretable coefficients, multiplicative structure that maps directly onto a rate table, established for actuarial review and regulatory filings, monotone and explainable by construction, stable, with well-understood standard errors. Cost: you hand-engineer every non-linearity and interaction, and you leave accuracy on the table.

GBM: higher accuracy, finds non-linearity and interactions automatically, robust to outliers and monotone transformations of the predictors, handles missing values natively. Cost: black box, harder to file, can produce non-monotone and counterintuitive behavior that needs monotone constraints, more tuning, no extrapolation, needs SHAP and a full explainability package.

The answer that lands: use the GBM to discover which variables matter and what shape their effects take, then build and file the GLM that encodes those findings - or file the GBM with monotonic constraints plus a documented explainability package where the jurisdiction allows it. Always mention the filing constraint; that's the insurance-specific insight a generic ML candidate won't have.

## 8. Random Forest

### Q: Walk me through the algorithm.

> Bagged deep decision trees, with an extra trick: at every split, each tree only gets to consider a random subset of the features.

1. For b = 1...B: draw a `bootstrap` sample of n rows, with replacement.

2. Grow a tree on that sample. At each split, randomly select `max_features` of the p predictors and find the best split only among those.

3. Grow deep - to purity or to `min_samples_leaf`. No pruning.

4. Predict: regression → average the B tree predictions. Classification → majority vote, or better (and what sklearn actually does) average the trees' predicted class probabilities.

The two sources of randomness, and why you need both:

- Bootstrap rows - this is the bagging part; it decorrelates the trees somewhat.

- Random feature subset at each split - this is what makes it a random forest rather than bagged trees. Without it, one dominant predictor would be the top split in nearly every tree, the trees would be near-identical, ρ would be high, and averaging would barely reduce variance. This is the ρσ² floor from §7.

Split criteria: Gini impurity `1 − Σpₖ²` or entropy `−Σp·log p` for classification (Gini is cheaper and they rarely disagree materially); variance reduction / MSE for regression.

### Q: What is OOB error?

> Each `bootstrap` sample leaves out about 36.8% of the rows. Predict each row using only the trees that never saw it, aggregate, and you have a validation estimate for free.

- Useful for quick tuning and for checking whether `n_estimators` is large enough (watch OOB error flatten).

- Not a substitute for a proper out-of-time holdout - OOB is still in-period and in-sample in the temporal sense.

### Q: Key hyperparameters - what they do and how they trade off.

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

The source sheet asks for this explicitly, so have real answers:

- `max_features` × `n_estimators` - lower `max_features` makes each tree weaker and more decorrelated, so you need more trees to average the added noise away. Tune them as a pair.

- `max_depth` × `min_samples_leaf` - both cap complexity from opposite ends (top-down vs. bottom-up). Tuning both is largely redundant: fix one, tune the other.

- Tree complexity × `n_estimators` - deeper, higher-variance trees gain more from additional trees than shallow ones do.

- `class_weight` × `min_samples_leaf` - heavy minority-class weighting combined with tiny leaves is the fastest way to overfit a rare class. If you upweight, raise the leaf minimum.

- `bootstrap`=False × `max_features`=p - that combination gives you B identical trees and zero ensemble benefit. A good “do you actually understand this” answer.

### Q: What tuning strategies do you use?

| Strategy | How it works | When to use it |
| --- | --- | --- |
| Grid search | exhaustive over a specified grid | Few parameters, and you’ve already located a coarse range. Cost is the product of the grid dimensions, so it dies past 3–4 parameters |
| Random search | sample combinations from distributions | The default first pass. For the same budget it usually beats grid search, because only a few parameters actually matter and random sampling gives you many distinct values of each rather than a few repeated ones |
| Bayesian optimization (TPE/GP — Optuna, Hyperopt) | builds a surrogate model of the objective and picks the next point by expected improvement | Each fit is expensive and the budget is tight. Con: sequential, so harder to parallelize; the overhead isn’t worth it on cheap models |
| Successive halving / Hyperband | start many configs cheaply, kill the losers, reallocate budget | Large search spaces with cheap early signals |

Always: tune on validation/CV folds, report on the untouched test set, and consider nested CV if the tuning is extensive.

### Q: How does RF compute feature importance?

| Method | How | Caveat |
| --- | --- | --- |
| MDI — mean decrease in impurity (Gini importance) | total impurity reduction from all splits on that feature, averaged over trees | Free (computed during training), but biased toward high-cardinality and continuous variables, and computed on training data |
| Permutation importance | shuffle one feature, measure the drop in validation performance | Model-agnostic and measured out-of-sample — but unreliable under correlated features (the model reads the signal off the twin) and more expensive |
| Drop-column | refit without the feature | Most faithful, most expensive |
| SHAP | game-theoretic additive attributions | Consistent, gives direction and magnitude, works locally and globally. The modern default |

Say the caveat unprompted: importance is not statistical significance and is not causal, and correlated features split the credit between them.

### Q: Pros and cons.

Pros: strong accuracy out of the box with minimal tuning; captures non-linearity and interactions; robust to outliers in X and to irrelevant features; no scaling needed; invariant to monotone transforms of the predictors; parallelizable; free OOB estimate; built-in importance; handles mixed data types.

Cons: a black box relative to a GLM; large model size and slower scoring; cannot extrapolate beyond the training range - every prediction is an average of observed y values, so it can't continue a trend (a real problem for trended insurance data); importance biased toward high-cardinality variables; weaker than linear models on very sparse high-dimensional data like text; can be swamped by the majority class under severe imbalance; predicted probabilities often need calibration.

**FOLLOW-UP:** Why doesn't RF overfit as you add trees? Because the prediction is an average over independently-fit trees. Adding more terms to an average reduces its variance and leaves its expectation alone; it converges rather than degrading. In boosting, each new tree changes the fitted function by chasing the current residuals, so more trees keeps pushing toward the training data.

## 9. GBM & XGBoost

### Q: Walk me through gradient boosting.

> Fit a sequence of shallow trees, where each one is fit to the errors the current ensemble is still making, and add it in at a small learning rate.

1. Initialize with the best constant prediction - the mean for squared error, the log-odds of the base rate for log-loss.

2. For m = 1...M: compute the negative gradient of the loss with respect to the current predictions - the "pseudo-residuals." For squared error these are literally y - F(x) .

3. Fit a small regression tree to those pseudo-residuals.

4. For each terminal node, compute the optimal constant value (a line search, or a Newton step).

5. Update: `F_m(x) = F_{m-1}(x) + ν · h_m(x)`, where ν is the learning rate / shrinkage.

6. Stop at M, or by early stopping on a validation set.

Where's the "gradient"? It's gradient descent in function space - each tree is a step in the direction that most reduces the loss. That framing is why you can plug in any differentiable loss: squared error, absolute error, log-loss, and - crucially for insurance - Poisson, Gamma, and Tweedie deviance. Being able to boost a Tweedie objective is a genuinely useful thing to mention.

Prediction is the initial constant plus ν times every tree's output - an additive sum, not an average. That's the structural difference from RF.

### Q: How is GBM different from random forest?

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

Give an ordered recipe, not a list - it shows you've actually done this:

1. Fix `learning_rate` = 0.1 and use early stopping on a validation set to find a reasonable number of trees. Fast baseline.

2. Tune tree complexity: `max_depth` (or `num_leaves`) and `min_child_weight` together - they're the primary bias/variance knobs.

3. Tune stochasticity: `subsample` and `colsample_bytree`.

4. Tune regularization: `gamma`, `reg_lambda`, `reg_alpha`.

5. Finally lower the learning rate to 0.01-0.05 and re-run with more trees and early stopping for the last few points of performance.

6. Use random search or Bayesian optimization over steps 2-4 rather than a full grid, always with CV and early stopping.

### Q: Why is GBM more sensitive to hyperparameters than RF?

> Because in a random forest the trees are fit independently and averaged, so mistakes are self-correcting - but in a GBM every tree is fit to the previous ensemble's errors, so mistakes compound.

- Too high a learning rate, or too deep a tree, and the ensemble starts fitting noise - and then every subsequent tree builds on top of that noise.

- There's no safe direction: too few trees underfits, too many overfits. `n_estimators` has an interior optimum. In RF it's monotone, so “just use more trees” is free.

- Which is why early stopping is mandatory for GBM and irrelevant for RF, and why RF’s “more trees can’t hurt” intuition does not transfer.

### Q: What does XGBoost add over traditional GBM?

Algorithmic:

- Regularized objective - explicit L1/L2 penalties on the leaf weights are inside the loss function, so regularization is part of the optimization rather than bolted on afterward.

- Second-order (Newton) optimization - uses the gradient and the Hessian, so leaf values and split gains are computed more accurately and it converges in fewer iterations.

- Native sparsity / missing-value handling - learns a default direction per split for missing values, rather than requiring imputation.

- Approximate split finding via a weighted quantile sketch and histogram binning, instead of scanning every candidate split point.

- Depth-first growth then pruning by `gamma`, rather than greedy early stopping at each node - so it can find a good split hiding behind a bad one.

Engineering:

- Parallelized split finding, cache-aware access patterns, out-of-core computation for larger-than-memory data, distributed training, GPU support.

Practical:

- Built-in CV and early stopping; monotonic and interaction constraints; a wide objective library including Poisson, Gamma, and Tweedie.

Worth one sentence: LightGBM adds leaf-wise growth, GOSS sampling, and native categorical handling - usually faster on large data. CatBoost adds ordered boosting (to fight target leakage in encoding) and the strongest categorical handling.

### Q: How does GBM compute variable importance?

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
| AUC — random / useful / strong | 0.5 / > 0.7 / > 0.8 |
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
| Coefficient on a well-binned WoE variable | ≈ 1.0 |

## Cross-Topic Connections

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

## Self-Test (No Answers)

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

本文件由 `IMG_9774–IMG_9808`（35 张照片）的 OCR 结果合并而成，技术上未作修订。为便于使用，导入时只做了三处处理，全部在此声明：

| # | 位置 | 处理 | 依据 |
|---|---|---|---|
| 1 | §6 连续型指标表 · **MAE** | 原合并稿标注「公式排版残缺·未补写」。**已由 OCR 片段恢复**：OCR 该行给出 `Σ`、`y−ŷ`，据此补为 `(1/n)·Σ|y − ŷ|` | `.transcription/IMG_9794.json` 文本块 `MAE` / `Y-9` / `У-Ỹ` |
| 2 | §6 连续型指标表 · **MAPE** | 同上。**已由 OCR 片段恢复**：OCR 给出 `(100/n)Σ`，据此补为 `(100/n)·Σ|(y − ŷ)/y|` | `.transcription/IMG_9794.json` 文本块 `МАPЕ` / `* (100/n)ž` |
| 3 | §6 Gain chart 的 **45° 线** | **照片在此处真实截断**（IMG_9794 底部），无法从原图恢复。已用通用定义补写一句并加注 `[补写]`，请按需要复核 | 照片截断，非 OCR 问题 |

**交叉引用**：本文件引用的 `magnet-study-guide.md`（resource/link index）**未随照片提供**，库内不存在该文件；如需请另行补入。

## 与库内其他笔记的分工

| 层 | 文件 | 定位 |
|---|---|---|
| **主线** | 本文件 | 面试真正会问什么、先说哪句、陷阱在哪 |
| **深潜** | 第 1–10 章 | 每个考点的推导、代码、保险语境 |
| **对照** | [[MAGNet 答案对照表]] | 逐题 → 深潜笔记的映射 |
| **自测** | [[99.1 题库总览]]（90 题） | 四块式：正确答案 / 错误答案 / 追问链 |
| **速览** | [[04. 公式速查卡]]、[[05. 高频追问 TOP 30]]、[[06. 中英术语对照表]] | 面试前 30 分钟 |
