# [[1. Logistic Regression & GLMs]]

## 🎯 Core Concept
A **Generalized Linear Model (GLM)** used when the outcome variable is **binary** (0 or 1). It models the probability of an event occurring.

- **Outcome**: Binary (e.g., Yes/No, Pass/Fail)
- **Fitting Method**: Maximum Likelihood Estimation (MLE)

## 📈 Model & Interpretation
The model uses a **logit link function** to connect the predictors to the outcome's probability.

#### Link Function: The Logit
- **What it is**: The natural log of the odds.
- **Formula**: `logit(p) = log(p / (1-p))`
- **Purpose**: Transforms probability `p` (from 0 to 1) to a continuous scale (from -∞ to +∞).

#### Model Equation
$$ \log\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 X_{i1} + \dots + \beta_k X_{ik} $$

- **Interpretation**: A one-unit change in `X_j` is associated with a `β_j` change in the **log-odds** of the outcome. Exponentiating the coefficient, `exp(β_j)`, gives the **odds ratio**.

## ✅ Key Assumptions
- **Binary Outcome**: The dependent variable must be binary.
- **Linearity of the Logit**: A linear relationship exists between the predictors and the log-odds of the outcome.
- **Independence of Observations**: Observations are not related to each other.
- **No Perfect Multicollinearity**: Predictors are not perfectly correlated.
- **Large Sample Size**: A rule of thumb is ≥10-20 cases for the *rarest* outcome class per predictor.

## ⚠️ Common Problems & Solutions

### 1. Imbalanced Samples (Rare Events)
- **Problem**: Model gets biased towards the majority class.
- **Solutions**:
  - **Resampling**: Oversample the minority (e.g., SMOTE) or undersample the majority.
  - **Class Weights**: Penalize errors on the minority class more heavily.
  - **Metrics**: Use **AUROC** or **Precision-Recall curves** instead of accuracy.

### 2. Separation
- **Problem**: A predictor perfectly (or nearly perfectly) separates the two outcome classes. This causes MLE to fail or produce huge standard errors.
- **Solutions**:
  - **Penalized Regression**: Use Ridge (L2) or Lasso (L1) regularization.
  - **Firth's Correction**: A bias-reduction method.
  - **Bayesian Priors**: Use informative priors to regularize coefficients.

## 📊 Evaluation
Primary evaluation is done using a **confusion matrix** and its derived metrics:

- **Accuracy**: Overall correctness (misleading for imbalanced data).
- **Precision**: `TP / (TP + FP)` - How many predicted positives were actually positive?
- **Recall (Sensitivity)**: `TP / (TP + FN)` - How many actual positives were found?
- **F1-Score**: Harmonic mean of Precision and Recall.
- **AUROC**: Ability of the model to rank a random positive case higher than a random negative one.
# [[2. Transformations]]
## 🎯 Why Transform Variables?
- **Improve Linearity**: Help linear models fit better.
- **Stabilize Variance**: Fix funnel shapes in residual plots (heteroscedasticity).
- **Normalize Distributions**: Make skewed data more symmetric for better inference.
- **Control Outliers**: Reduce the influence of extreme values.
- **Encode Categories**: Convert non-numeric data for algorithms.

## 🔍 When to Transform? (Key Signals)
- **Residual Plots**: Show curvature (non-linearity) or a funnel shape (variance issues).
- **Histograms / Q-Q Plots**: Show significant skew or heavy tails.

---

## 🛠️ Core Transformation Techniques Overview

| Technique | Applies to | Target-Aware? | Primary Goal |
| :--- | :--- | :--- | :--- |
| **Dummy / One-Hot Encoding** | Categorical | No | Convert categories to numeric format. |
| **Winsorization (Capping)** | Continuous | No | Reduce outlier impact. |
| **Binning (Unsupervised)** | Continuous | No | Simplify, handle non-linearity. |
| **Box-Cox / Power Transform** | Continuous | No | Normalize skew, stabilize variance. |
| **Polynomials / Splines** | Continuous | No | Model complex non-linear relationships. |
| **Weight of Evidence (WOE)** | Categorical | Yes (Binary) | Create a monotonic, predictive numeric score for categories. |
| **Supervised Binning** | Continuous | Yes | Create bins that best separate the target classes. |
| **GAM Smoothing** | Continuous | Yes | Automatically find and apply a smooth non-linear transformation. |

---

## ✨ Key Techniques Explained

### 1. Categorical Encoding

#### One-Hot / Dummy Coding
- **Goal**: Convert a category into binary (0/1) columns.
- **Dummy Coding**: Creates `k-1` columns, with one category as the baseline (intercept). Interpretable coefficients.
- **One-Hot Encoding**: Creates `k` columns. Can cause multicollinearity if all are used in a linear model.

#### Weight of Evidence (WOE) & Information Value (IV)
- **Goal**: Transform categories into a single numeric score based on their relationship with a binary target.
- **WOE Formula**: `WOE = ln(%Goods / %Bads)`
  - Positive WOE: Category is associated with higher odds of the event (Y=1).
  - Negative WOE: Category is associated with lower odds.
- **Information Value (IV)**: Measures the predictive power of the variable based on its WOE.
  - **Rule of Thumb**: `IV < 0.02` (useless), `0.1 - 0.3` (medium), `> 0.3` (strong).

### 2. Continuous Transformations

#### Binning
- **Unsupervised**: Bins based only on the feature's distribution.
  - **Equal-Width**: Simple, but sensitive to outliers.
  - **Quantile**: Each bin has the same number of observations; robust to outliers.
- **Supervised**: Bins are created to maximize the separation between target classes (e.g., using Chi-Square or Entropy).

#### Power Transforms (Box-Cox)
- **Goal**: Find the best power `λ` (lambda) to make data more normal.
- **Formula**: `(x^λ - 1) / λ`
- **Key Lambdas**: `λ=0` is log, `λ=0.5` is square root, `λ=-1` is reciprocal.
- **Requirement**: Data must be positive.

#### Polynomials & Splines (for non-linearity)
- **Polynomials**: Add powers of the feature (e.g., `x²`, `x³`) to the model to capture curves. Prone to overfitting and wild extrapolation.
- **Splines / GAMs**: Fit flexible, piecewise curves to the data. More stable and powerful than polynomials for capturing complex relationships without overfitting. A **GAM** automates this by fitting a smooth function `f(x)` for each predictor.

# [[3. Multicollinearity]]
## **Definition & Core Problem**
Multicollinearity occurs when predictor variables are highly correlated, making it impossible to determine individual variable effects. The mathematical root: when predictors are correlated, the matrix (X'X)⁻¹ becomes unstable, causing coefficient estimates to explode in variance.

**The Domino Effect:** High correlation → High VIF → Inflated standard errors → Unstable coefficients → High p-values → Unreliable interpretation

## **Model Impact**
| **Model Type** | **Effect** |
|:---|:---|
| **Linear/Logistic/GLMs** | 🚫 **Severely affected** - Unstable coefficients, inflated SEs, unreliable p-values |
| **Tree-Based Models** | ✅ **Largely immune** - No matrix inversion; naturally selects one variable from correlated groups |

## **Detection Methods**
1. **Correlation Matrix**: |r| ≥ 0.70 = problematic
2. **Variance Inflation Factor (VIF)**: VIF = 1/(1-R²)
   - VIF = 1: No multicollinearity
   - VIF > 5: Concerning
   - VIF > 10: Severe problem

## **Solutions

| **Method** | **Approach** | **Pros** | **Cons** |
|:---|:---|:---|:---|
| **Variable Removal** | Iteratively remove highest VIF variables | Simple, interpretable | Loses information |
| **Variable Clustering** | Group correlated variables, select representatives | Preserves meaning | More complex |
| **Penalized Regression** | Ridge/LASSO with coefficient penalties | Stable, automatic selection | Introduces tuning |
| **PCA Regression** | Transform to uncorrelated components | Mathematically elegant | Loses interpretability |

## **Key Takeaway**
- **For Interpretation**: Multicollinearity makes individual coefficients meaningless in GLMs
- **For Prediction**: Tree-based models handle it naturally; GLMs may need intervention
- **Quick Fix**: Remove variables with VIF > 10 sequentially, recalculating VIFs after each removal
# [[4. Missing Data]]

**Definition & Purpose**: Occurs when observations have absent entries for some variables; must handle thoughtfully to avoid bias.

### **🎲 MCAR (Missing Completely At Random)**

#### **Diagnosis: Pure Bad Luck**

| **What it is**                                                               | **How to Spot It**                                                                  | **Implication**                                                                 |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| The probability of a value being missing is **unrelated to everything**. It's a purely random event. | Gaps appear scattered with no pattern. Formal check: **Little's MCAR test** (p > 0.05). | The observed data is an **unbiased** (but smaller) subsample. The main risk is losing statistical power. |

#### **Fixes for MCAR**

| **Method**                        | **Core Idea**                                                   | **Pros**                                       | **Cons**                                                                                                     |
| --------------------------------- | --------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Complete Case Analysis (Dropping)** | Delete any row containing a missing value.                      | Simple, fast, and unbiased *if* data is truly MCAR. | Wastes data, reduces statistical power, widens confidence intervals. **Only recommended for <5% missingness.** |
| **Mean / Mode Imputation**        | Replace missing values with the column's average or most frequent category. | Keeps sample size intact, very easy to implement. | 🚨 **Destroys variance** and **distorts correlations**. Makes you overconfident in results.                  |

---

### **🔗 MAR (Missing At Random)**

#### **Diagnosis: Systematically Explainable**

| **What it is**                                                               | **How to Spot It**                                                                              | **Implication**                                                                                                   |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| The probability of a value being missing is explainable by **other observed variables**. | Patterns emerge in visualizations (e.g., missing `Income` correlates with `Age`). The cause is *in your data*. | **This is good!** You can use the relationships in your data to make an intelligent, unbiased estimate of the missing value. |

#### **Fixes for MAR**

| **Method**                   | **Core Idea**                                                                  | **Pros**                                                                      | **Cons**                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Regression Imputation**    | Predict the missing value using other columns as features in a regression model. | Preserves relationships between variables; more intelligent than mean imputation. | Assumes a specific model form (e.g., linearity). Underestimates variance unless random noise is added.         |
| **k-NN Imputation**          | Fill the gap using an average of the `k` most similar complete rows ("neighbors"). | Captures complex, non-linear relationships without needing a model.           | Can be slow on large datasets. Sensitive to feature scaling and the choice of `k`.                            |
| **Multiple Imputation (MICE)** | ✨ **Gold Standard** ✨ <br> Create `m` plausible completed datasets, analyze each, and pool the results. | **Properly accounts for uncertainty**, providing valid p-values and confidence intervals. | The most complex method to implement and understand. Computationally intensive.                              |

---

### **❓ MNAR (Missing Not At Random)**

#### **Diagnosis: The Unseen Cause**

| **What it is**                                                               | **How to Spot It**                                                                         | **Implication**                                                                                                           |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| The probability of a value being missing depends on the **missing value itself**. | Usually requires **domain knowledge**. You can't see it in the data alone (e.g., people with very high incomes hide them). | **This is the hardest case.** Standard imputation methods will introduce **bias**. The information needed to fix the gap is missing. |

#### **Fixes for MNAR**

| **Approach**              | **Core Idea**                                                                                                          | **Key Consideration**                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Model the Missingness** | Use advanced statistical techniques (e.g., Selection Models) that explicitly make an assumption about the missingness. | Requires strong statistical knowledge and justifiable assumptions about why the data is missing.            |
| **Sensitivity Analysis**  | Impute data under different "what-if" scenarios (e.g., all missing values are high, then all are low).                 | Doesn't give you one answer, but tests if your conclusion is robust under various plausible assumptions.    |
| **Collect More Data**     | Go back to the source and try to acquire the missing information through follow-up surveys or other means.             | Often impractical, but it is the only way to truly solve the problem without making untestable assumptions. |
# [[5. Dimension Reduction]]

**Definition & Purpose**: Techniques to reduce the number of variables while retaining most information, improving interpretability and performance.
**When & How to Identify**: Use when feature count is high relative to observations or multicollinearity is severe. Identify via exploratory analysis, variance explained, overfitting symptoms.

* **When & why**: reduce noise, improve interpretability, combat over‑fitting, lower storage/computation
* **Univariate screening**: IV, χ², ANOVA, AUROC to rank features individually
* **Multivariate screening**: Spearman, Hoeffding’s D, mutual information for pairs

**Types of Dimensionality Reduction Techniques**

|                                               | **Target-aware (supervised)**                                                                                                | **Target-unaware (unsupervised)**                                                                                           |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Dimension reduction (feature selection)**   | **Filter**: correlation, ANOVA, IV <br>**Wrapper**: RFE, forward/backward stepwise <br>**Embedded**: LASSO, tree importances | **Filter**: variance threshold, mutual info <br>**Wrapper**: k-NN with CV <br>**Embedded**: unsupervised feature importance |
| **Dimension projection (feature extraction)** | PCA supervised variants, supervised autoencoders                                                                             | PCA, ICA, t-SNE, unsupervised autoencoders                                                                                  |

# [[6. Model Assessment]]

**Definition & Purpose**: Framework for evaluating model performance and generalization on unseen data.
**When & How to Identify**: Use after model training to estimate skill. Identify via need for unbiased performance estimates and model comparison.

* **Data splitting**: train/validation/test, k‑fold (stratified), rolling window (time series)
* **Bias–variance trade-off**: under‑/over‑fitting, learning curves, regularization effects
* **Performance Metrics**:

| Task           | Primary Metrics                            |
| -------------- | ------------------------------------------ |
| Regression     | RMSE, MAE, MAPE, R²                        |
| Classification | Accuracy, Precision, Recall, F1, AUROC, KS |
| Profit-based   | Lift charts, Cumulative gains, Net revenue |

---

# [[7. Random Forest]]

**Definition**
An ensemble of unpruned decision trees trained on bootstrap samples with random feature subsets at each split. Combines bagging and random feature selection to reduce variance and improve generalization.

---

**Fit (Training Procedure)**

1. For each of *B* trees:

   * Draw a bootstrap sample from the training set.
   * Grow a full decision tree: at each node, select a random subset of features, choose the best split, and repeat until stopping criteria.
2. Aggregate the ensemble of trees for prediction.

---

**Feature Importance**

* **MDI (Mean Decrease in Impurity):** Sum of impurity reductions for each feature across all splits and trees, then normalize.
* **Permutation Importance:** Measure drop in performance (accuracy or MSE) when a feature’s values are randomly shuffled on out-of-bag or validation data.

---

### Key Parameters

| Parameter               | What It Controls                             | Trade-Off / Notes                                                       |
| :---------------------- | :------------------------------------------- | :---------------------------------------------------------------------- |
| **n\_estimators**       | Number of trees                              | ↑ reduces variance but ↑ computation & memory                           |
| **max\_depth**          | Maximum depth of each tree                   | ↓ simpler trees → bias ↑; ↑ complex trees → risk of overfitting         |
| **min\_samples\_split** | Min. samples to consider splitting a node    | ↑ value → fewer splits → bias ↑; ↓ value → more splits → variance ↑     |
| **min\_samples\_leaf**  | Min. samples required at a leaf              | Guards against tiny leaves; ↑ value → smoother predictions but ↑ bias   |
| **max\_features**       | Features considered per split                | Classification: √p; Regression: p/3; lower → more decorrelation, ↑ bias |
| **bootstrap**           | Use bootstrap sampling (True/False)          | False → no bagging → often ↑ variance                                   |
| **criterion**           | Split quality measure (Gini/entropy/MSE/MAE) | Affects split decisions; entropy slower but sometimes more informative  |
| **class\_weight**       | Class weights for imbalanced classification  | Weight minority class ↑ to reduce its misclassification                 |

---

### Pros & Cons

| Pros                                               | Cons                                              |
| :------------------------------------------------- | :------------------------------------------------ |
| High out-of-the-box accuracy                       | Less interpretable than a single tree             |
| Robust to overfitting (bagging + random splits)    | Computationally & memory intensive                |
| Handles non-linearities and mixed feature types    | MDI biased toward high-cardinality features       |
| Built-in OOB error estimate and feature importance | Cannot extrapolate beyond the training data range |

---

**Usage Tips**

* Use OOB error for quick validation.
* Profile error vs. number of trees to choose `n_estimators`.
* Tune core parameters (`max_depth`, `min_samples_leaf`, `max_features`) via randomized or Bayesian search within CV.
* Prefer permutation importance over MDI to avoid bias when guiding feature engineering.

Here's your formatted content using `###` for the main title ("Ensemble Learning") and `####` for the subheadings:

---

# [[8. Ensemble Learning]]

---

#### 1. What & Why

* **Ensemble Learning:** Combine multiple “weak” models to form a stronger predictor.
* **Goal:** Reduce variance (bagging), bias (boosting), or both (stacking) via model diversity.

---

#### 2. Core Concepts

* **Bias–Variance Tradeoff:**

  * Bagging cuts variance by averaging.
  * Boosting lowers bias by sequentially correcting errors.
  * Stacking can address both via meta-learning.

* **Diversity:** Key to success—achieved by resampling (bagging), reweighting (boosting), or heterogeneous learners (stacking).

---

#### 3. Methods

1. **Bagging (e.g., Random Forest)**

   * **How:** Train base learners on bootstrap samples + average/vote.
   * **Pros:** Parallelizable, strong variance reduction.
   * **Cons:** Doesn’t fix bias.

2. **Boosting (e.g., AdaBoost, Gradient Boosting)**

   * **How:** Sequentially fit learners to residuals or misclassified cases; aggregate with weighted sum.
   * **Pros:** Reduces bias & variance; flexible loss functions.
   * **Cons:** Sequential (less parallel), risks overfitting, many hyperparameters.

3. **Stacking**

   * **How:** First-level models → generate out-of-fold predictions → train a meta-learner on these “meta-features.”
   * **Pros:** Combines heterogeneous models; captures complex patterns.
   * **Cons:** Complex, computationally heavy, careful cross-validation needed.

---

#### 4. Quick Comparison

|              | Bias↓ | Variance↓ | Parallel  | Complexity |
| ------------ | ----- | --------- | --------- | ---------- |
| **Bagging**  | –     | ✔         | ✔         | Low        |
| **Boosting** | ✔     | ✔         | ✖️ (seq.) | Medium     |
| **Stacking** | ✔/✖️  | ✔         | Partial   | High       |

---

#### 5. Tips

* **Start Simple:** Random Forest → Gradient Boosting → Stacking.
* **Prevent Overfitting:** CV, early stopping, tree depth/regularization.
* **Interpretability:** Use SHAP or feature-importance tools.

---

Let me know if you'd like this adapted for a slide deck, blog, or technical report format.
