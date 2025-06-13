### [[1. Logistic Regression & GLMs]]
**Definition & Purpose**: Statistical models for binary or count outcomes using a link function to relate predictors to expected response. Ideal when the target variable is categorical (often binary).
**When & How to Identify**: Use when modeling classification tasks (e.g., churn, default). Identify via binary target variable and need for interpretable, probabilistic outputs.

**Core Assumptions and Diagnostics**

| Assumption                      | Linear Model        | GLM                     | Diagnostic Methods                                  |
| ------------------------------- | ------------------- | ----------------------- | --------------------------------------------------- |
| Yi distribution                 | Normal distribution | Exponential family      | Q-Q plot                                            |
| Linearity of predictors         | Linear in mean      | Linear in link function | Residual plots, LOESS                               |
| Multicollinearity               | Not expected        | Not expected            | VIF, correlation matrix                             |
| Outliers & high leverage points | Must check          | Must check              | Leverage stats, Cook's distance                     |
| Homoscedasticity                | Required            | Not required            | Residuals vs fitted, scale-location plot (LM only)  |
| Model specification correctness | Important           | Important               | Residual analysis, AIC/BIC, omitted variable checks |
|                                 |                     |                         |                                                     |

**GLM vs GBM: Model Capabilities**

| Feature       | GLM   | GBM    |
| ------------- | ----- | ------ |
| Non-linearity | No    | Yes    |
| Interactions  | No    | Yes    |
| Monotonicity  | Yes   | No     |
| Accuracy      | Lower | Higher |

* **Model form & interpretation**: logit link (log-odds), odds ratios, canonical vs non-canonical links
* **Estimation issues**: complete/quasi-separation, non-convergence, penalized approaches
* **Evaluation & imbalance**: confusion matrix, accuracy vs precision/recall trade-off, AUROC, resampling, class weights, threshold tuning

### [[2. Transformations]]

**Definition & Purpose**: Mathematical operations applied to input variables to improve model assumptions (linearity, normality, variance stability).
**When & How to Identify**: Use when predictor distributions are skewed or residuals show non-linearity. Identify via residual plots, skewness/kurtosis metrics, Q-Q plots.

| Transformation         | Purpose                                   | Use Case Example                       |
| ---------------------- | ----------------------------------------- | -------------------------------------- |
| Winsorisation          | Cap/floor outliers to reduce influence    | Handling salary outliers               |
| Box-Cox                | Normalize skewed positive-only data       | Right-skewed income values             |
| Yeo-Johnson            | Normalize skewed data including negatives | Net gain/loss columns                  |
| Polynomial Terms       | Model non-linearity explicitly            | U-shaped relationship in age vs. churn |
| Splines                | Piecewise smooth fitting                  | Non-linear patterns in time series     |
| GAM (Generalized Add.) | Flexible smoothing with regularization    | Continuous predictors with curvature   |

### [[3. Multicollinearity]]

**Definition & Purpose**: Situation where two or more predictors are highly correlated, inflating variance and destabilizing coefficient estimates.
**When & How to Identify**: Use when regression coefficients vary wildly or have unexpected signs. Identify via high VIF (>5–10), strong pairwise correlations, condition index.

* **Detection**: pairwise correlations, VIF, condition index, variance decomposition proportions
* **Impact**: inflated coefficient variance, unstable estimates, mis‑leading inference
* **Remedies**: drop/aggregate variables, hierarchical clustering (VARCLUS), PCA for projection, ridge/LASSO regularization

### [[4. Missing Data]]

**Definition & Purpose**: Occurs when observations have absent entries for some variables; must handle thoughtfully to avoid bias.
**When & How to Identify**: Use when dataset contains NA values. Identify via missing-value counts, patterns (MCAR/MAR/MNAR), visualizations (heatmaps).

| Mechanism | Description                                        | Example                                      |
| --------- | -------------------------------------------------- | -------------------------------------------- |
| MCAR      | Missing completely at random                       | Sensor failed randomly                       |
| MAR       | Missing at random, dependent on observed variables | Income missing more often for younger people |
| MNAR      | Missing not at random, depends on unobserved data  | People with high income skip salary question |

**Common Imputation Methods**:

| Method                | Description                               | Pros                     | Cons                         |
| --------------------- | ----------------------------------------- | ------------------------ | ---------------------------- |
| Mean/Mode Imputation  | Replace missing with column mean or mode  | Simple, fast             | Underestimates variance      |
| Hot-deck              | Impute from similar observed record       | Preserves distribution   | May introduce bias           |
| Regression Imputation | Predict missing using other features      | Leverages relationships  | Assumes linear relationships |
| k-NN Imputation       | Use nearest neighbors to impute           | Handles non-linearity    | Computationally expensive    |
| Multiple Imputation   | Generates multiple estimates and averages | Accounts for uncertainty | Complex implementation       |

**Considerations**: Bias risk from wrong assumptions, underestimation of variance, and impact on model interpretability.

### [[5. Dimension Reduction]]

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

### [[6. Model Assessment]]

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

### [[7. Random Forest]]

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

### [[8. Ensemble Learning]]

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
