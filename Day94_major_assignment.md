**Author Name:** Shahid Umar
**Email:** sshahidumarr@gmail.com
**Linkedin:** https://linkedin.com/in/shahidumar

---
# Day94 Major Assignment
### Make a document of all Machine Learning Algorithms as per following pattern
  1. Introduction
  2. Mathematical Formulation
  3. Assumptions
  4. Final Words (Conclusion)
---

## I wil discuss the following algorithms in this document
1. **Regression Algorithms**
   1. Simple Linear Regression Algorithm
   2. Multi-Linear Regression Algorithm
   3. Polynomial Regression Algorithm
   4. Ridge Regression Algorithm
   5. LASSO Regression Algorithm
   6. Support Vector Regressor (SVR) Algorithm (Type of SVM)
   7. KNeighborsRegressor Algorith (Type of KNN)
   8. DecisionTreeRegressor Algorithm (Type of Decision Tree)
   9. RandomForestRegressor Algorithm (Type of Ensemble -->> Bagging -->> Random Forest Algorithms)
   10. AdaBoostRegressor (Type of Ensemble -->> Boosting -->> Adaptive Boosting Algorithm)
   11. GradientBoostingRegressor Algorithm (Type of Ensemble -->> Boosting -->> Gradient Boosting Algorithm)
   12. LGBMRegressor (Type of Ensemble -->> Boosting -->> Light Gradient Boosting Algorithm)
   13. XGBRegressor Algorithm (Type of Ensemble -->> Boosting -->> Xtrem Gradient Boosting Algorithm)
   14. CatBoostRegressor Algorithm (Type of Ensemble -->> Boosting -->> Categorical Boosting Algorithm)
   15. Stacking for Regression (Type of Ensemble -->> Boosting -->> Stacked Generalization Algorithm)
   16. Pipeline Algorithms for Regression
   17. Evaluation Metrics for Regression
       1.  mean_squared_error (MSE) metrics Algorithm
       2.  root_mean_squared_error (RMSE) metrics Algorithm
       3.  R-Squared error metrics Algorithm
       4.  mean_absolute_error (MAE) metrics Algorithms
       5.  mean_absolute_percentage_error (MAPE) metrics
2. **Classification Algorithms**
   1. Logistic Regression Algorithm (`for use of classification, search why logistic is regression`)
   2. Support Vector Classifier (SVC) Algorithm (Type of SVM)
   3. KNeighborsClassifier Algorithm (Type of KNN)
   4. DecisionTreeClassifier Algorithm (Type of Decision Tree)
   5. RandomForestClassifier Algorithm (Type of Ensemble -->> Bagging -->> Random Forest Algorithm)
   6. AdaBoostClassifier (Type of Ensemble -->> Boosting -->> Adaptive Boosting Algorithm)
   7. GradientBoostingClassifier Algorithm (Type of Ensemble -->> Boosting -->> Gradient Boosting Algorithm)
   8. LGBMClassifier (Type of Ensemble -->> Boosting -->> Light Gradient Boosting Algorithm)
   9. XGBClassifier Algorithm (Type of Ensemble -->> Boosting -->> Xtrem Gradient Boosting Algorithm)
   10. CatBoostClassifier Algorithm (Type of Ensemble -->> Boosting -->> Categorical Boosting Algorithm)
   11. LPBoost Algorithm (Type of Ensemble -->> Boosting -->> Linear Programming Boosting Algorithm)
   12. Naive Bayes Algorithms
       1.  GaussianNB Algorithm `(19, 22 notebook)`
       2.  MultinomialNB Algorithm
       3.  BernoulliNB Algorithm
   13. Stochastic Gradient Boosting Algorithm
   14. TotalBoost (Total Boosting)
   15. Stacking for classification (Type of Ensemble -->> Boosting -->> Stacked Generalization Algorithm)
   16. Pipeline Algorithms for Regression
   17. Evaluation Metrics for Classification 
       1. accuracy_score metric Algorithm
       2. recall_score metric Algorithm
       3. precision_score metric Algorithm
       4. f1_score metric Algorithm
       5. confusion_matrix Algorithm
       6. classification_report Algorithm
3. **Scaling Algorithms**
   1. StandardScaler Algorithm
   2. MinMaxScaler Algorithm
   3. MaxAbsScaler Algorithm
   4. RobustScaler Algorithm
   5. PowerTransformer Algorithm
   6. QuantileTransformer Algorithm
   7. ColumnTransformer Algorithm `(10b notebook)`
4. **Encoding Algorithms**
   1. LabelEncoder Algorithm
   2. OneHotEncoder Algorithm
   3. OrdinalEncoder Algorithm
   4. BinaryEncoder Algorithm
   5. Frequency and Counting Encoding Algorithm
5.  **Cross Validation Algorithms**
    1.  Train_test_split Algorithm
    2.  GridSearchCV Algorithm
    3.  RandomizedSearchCV Algorithm
    4.  cross_val_score Algorithm `(19 notebook)`
    5.  KFold Algorithm `(23 notebook)`
6.  **Imputation Algorithms**
    1.  SimpleImputer Algorithm
    2.  KNNImputer Algorithm `(17 notebook)`
    3.  IterativeImputer Algorithm
7. **Discretization Algorithms**
   1. KBinsDiscretizer Algorithm
8. **Normalization Algorithms**
    1.  Normalizer Algorithm


---
# <span style="color:green;">1.) REGRESSION ALGORITHMS</span>
- Regression algorithms are a class of machine learning algorithms used to predict continuous numerical values based on input features. The goal of regression is to establish a mathematical relationship between a dependent variable (also known as the target or response variable) and one or more independent variables (also known as predictors or features).

- In regression, the dependent variable is a continuous variable, such as temperature, price, or sales volume, and the independent variables are used to make predictions or estimate the value of the dependent variable. Regression algorithms aim to find the best-fit line or curve that represents the relationship between the independent variables and the dependent variable.
## <span style="color:green;">1.1) SIMPLE LINEAR REGRESION ALGORITHM</span>
#### 1.1.1) Introduction:
Simple linear regression is a regression algorithm that assumes a linear relationship between a single independent variable and a dependent variable. It aims to find the best-fit line that represents the relationship between the variables. It is called "simple" because it deals with only one independent variable.

#### 1.1.2) Mathematical Formulation
In simple linear regression, we represent the relationship between the independent variable (x) and the dependent variable (y) using a linear equation of the form:
```
y = b0 + b1 * x
```

**where:**

- 'y' is the dependent variable (the variable we want to predict)
- 'x' is the independent variable (the variable used to make predictions)
- 'b0' is the y-intercept (the value of y when x = 0)
- 'b1' is the slope of the line (the change in y for a unit change in x)
- The goal of simple linear regression is to estimate the values of b0 and b1 that minimize the sum of squared differences between the observed values of y and the predicted values of y given by the linear equation.

#### 1.1.3) Assumptions
Simple linear regression relies on several assumptions:

**Linearity:** The relationship between the independent and dependent variables is assumed to be linear.
**Independence:** The observations are assumed to be independent of each other.
**Homoscedasticity:** The variance of the errors (the differences between the observed and predicted values of y) is assumed to be constant across all levels of x.
**Normality:** The errors are assumed to follow a normal distribution with a mean of zero.
**No multicollinearity:** The independent variable is not highly correlated with other independent variables.
>⚠️ **Tip:** Violations of these assumptions may affect the accuracy and reliability of the regression model.

#### 1.1.4) Conclusion
Simple linear regression is a straightforward algorithm that models the relationship between a single independent and dependent variable using a linear equation. By estimating the values of the y-intercept and slope, it provides a predictive model that can be used to make predictions or understand the relationship between the variables.
However, it is important to validate the assumptions and evaluate the model's performance before drawing conclusions or making predictions based on the regression results.

## <span style="color:green;">1.2) MULTI-LINEAR REGRESION ALGORITHM</span>
#### 1.2.1) Introduction
Multiple linear regression is a regression algorithm that extends simple linear regression to model the relationship between multiple independent variables and a dependent variable. It assumes a linear relationship between the variables and aims to find the best-fit hyperplane that represents this relationship. It is called "multiple" because it deals with more than one independent variable.

#### 1.2.2) Mathematical Formulation
In multiple linear regression, we represent the relationship between the dependent variable (y) and multiple independent variables (x₁, x₂, ..., xn) using a linear equation of the form:
```
y = b₀ + b₁ * x₁ + b₂ * x₂ + ... + bn * xn
```
**where:**
- 'y' is the dependent variable (the variable we want to predict)
- x₁, x₂, ..., xn are the independent variables (the variables used to make predictions)
- 'b₀' is the y-intercept (the value of y when all x variables are zero)
b₁, b₂, ..., bn are the slopes of the hyperplane (the change in y for a unit change in each x variable)
- The goal of multiple linear regression is to estimate the values of b₀, b₁, b₂, ..., bn that minimize the sum of squared differences between the observed values of y and the predicted values of y given by the linear equation.

#### 1.2.3) Assumptions
Multiple linear regression relies on several assumptions. Let's discuss them in detail:

**Linearity:** The relationship between the independent variables and the dependent variable is assumed to be linear. It means that the effect of each independent variable on the dependent variable is additive.

**Independence:** The observations are assumed to be independent of each other. Each observation should not be influenced by any other observation.

**Homoscedasticity:** The variance of the errors (the differences between the observed and predicted values of y) is assumed to be constant across all levels of the independent variables. This assumption implies that the spread of the residuals is consistent throughout the range of the independent variables.

**Normality:** The errors are assumed to follow a normal distribution with a mean of zero. This assumption allows for the calculation of confidence intervals and hypothesis testing.

**No multicollinearity:** The independent variables should not be highly correlated with each other. Multicollinearity occurs when there is a strong linear relationship between two or more independent variables, which can lead to difficulties in interpreting the individual effects of the variables.

>⚠️ **Tip:** To check for multicollinearity, you can calculate the correlation matrix of the independent variables and look for high correlation coefficients.

>⚠️ **Suggestion:** If multicollinearity is present, consider techniques such as feature selection or regularization (e.g., ridge regression or lasso regression) to mitigate its impact.

#### 1.2.4) Conclusion
Multi-linear regression is a powerful algorithm for modeling the relationship between multiple independent variables and a dependent variable. By estimating the values of the y-intercept and slopes, it provides a predictive model that can be used to make predictions or understand the influence of each independent variable on the dependent variable.
However, it is crucial to validate the assumptions, check for multicollinearity, and evaluate the model's performance before drawing conclusions or making predictions based on the regression results.


## <span style="color:green;">1.3) POLYNOMIAL REGRESION ALGORITHM</span>
#### 1.3.1) Introduction
Polynomial regression is a regression algorithm that extends the concept of linear regression by capturing non-linear relationships between the independent variables and the dependent variable. It fits a polynomial function to the data, allowing for curved relationships. Polynomial regression can be a powerful tool when the relationship between the variables cannot be adequately captured by a linear model.

#### 1.3.2) Mathematical Formulation
In polynomial regression, the relationship between the dependent variable (y) and the independent variable (x) is represented by a polynomial equation of the form:
```
y = b₀ + b₁x + b₂x² + ... + bₙxⁿ
```

**where:**
- 'y' is the dependent variable
- 'x' is the independent variable
- b₀, b₁, b₂, ..., bₙ are the coefficients of the polynomial equation
- 'n' is the degree of the polynomial, determining the complexity of the curve
- By adjusting the values of the coefficients, the polynomial equation can fit the data with varying degrees of flexibility, capturing different levels of curvature.

#### 1.3.3) Assumptions
Polynomial regression shares some assumptions with linear regression, but there are additional considerations:

**Linearity in coefficients:** Although the relationship between the variables is non-linear, polynomial regression assumes linearity in the coefficients. This means that the effect of increasing the value of x on y is additive, even though the relationship is curved.

**Independence:** The observations are assumed to be independent of each other, similar to other regression algorithms.

**Homoscedasticity:** As with linear regression, polynomial regression assumes constant variance of the errors across all levels of x.

**Normality:** The errors are assumed to follow a normal distribution with a mean of zero, allowing for statistical inference and hypothesis testing.

**No multicollinearity:** Polynomial regression assumes that the independent variables are not highly correlated with each other, as high multicollinearity can affect the stability and interpretability of the model.

> ⚠️ Tip: When working with polynomial regression, it is common to include interaction terms between the independent variables to account for potential interactions and capture more complex relationships.

> ⚠️ Suggestion: While polynomial regression can capture non-linear patterns, it is important to be cautious about overfitting. Higher-degree polynomials can lead to complex models that may not generalize well to new data. Regularization techniques, such as ridge regression or lasso regression, can help mitigate overfitting.

#### 1.3.4) Conclusion
Polynomial regression is a valuable technique for modeling non-linear relationships between variables. By using polynomial equations, it can capture curved patterns and provide a flexible modeling approach.
However, it is crucial to validate the assumptions, address multicollinearity, and consider the risk of overfitting when determining the degree of the polynomial.
By carefully selecting the degree and interpreting the coefficients, polynomial regression can be a powerful tool for gaining insights and making predictions.


## <span style="color:green;">1.4) RIDGE REGRESSION</span>

#### 1.4.1) Introduction
Ridge regression is a regularization technique used in linear regression to address the problems of multicollinearity and overfitting. It adds a penalty term to the ordinary least squares (OLS) objective function, which helps to stabilize the model and reduce the impact of highly correlated independent variables.

#### 1.4.2) Mathematical Formulation
In ridge regression, the objective function is modified by adding a penalty term that is proportional to the squared magnitudes of the coefficients. The modified objective function is given by:

minimize: (RSS + α * Σ(bᵢ²))

**where:**
RSS (Residual Sum of Squares) is the sum of squared differences between the observed and predicted values.
'α' (alpha) is the regularization parameter that controls the amount of shrinkage applied to the coefficients.
'bᵢ' (beta) represents the coefficients of the independent variables.
The term Σ(bᵢ²) represents the sum of squared coefficients, and α determines the trade-off between the fit of the model to the training data and the magnitude of the coefficients.

#### 1.4.3) Assumptions
Ridge regression relies on the following assumptions:

**Linearity:** The relationship between the independent variables and the dependent variable is assumed to be linear.

**Independence:** The observations are assumed to be independent of each other.

**Homoscedasticity:** The variance of the errors is assumed to be constant across all levels of the independent variables.

**Normality:** The errors are assumed to follow a normal distribution with zero mean.

**No perfect multicollinearity:** Ridge regression assumes that the independent variables are not perfectly correlated. Perfect multicollinearity occurs when there is an exact linear relationship between two or more independent variables. This assumption is important because ridge regression uses the sum of squared coefficients as a penalty term, and perfect multicollinearity would lead to an infinite penalty.

#### 1.4.4) Conclusion
Ridge regression is a valuable regularization technique for linear regression models. By introducing a penalty term based on the magnitude of the coefficients, it helps to address multicollinearity and reduce the risk of overfitting. However, it is important to validate the assumptions, standardize the variables, and choose an appropriate value for the regularization parameter α. Ridge regression provides a robust approach for improving the stability and generalization performance of linear regression models.


## <span style="color:green;">1.5) LASSO REGRESSION ALGORITHM</span>

#### 1.5.1) Introduction
Lasso (Least Absolute Shrinkage and Selection Operator) regression is a linear regression method that performs both variable selection and regularization. It is a popular technique in machine learning and statistics for feature selection and parameter estimation. Lasso regression can be used to effectively handle high-dimensional datasets by promoting sparsity in the model, i.e., it encourages some of the coefficients to be exactly zero.

#### 1.5.2) Mathematical Formulation
In the context of linear regression, the Lasso algorithm aims to find the best linear model that minimizes the sum of squared residuals, subject to a constraint on the sum of the absolute values of the coefficients. Mathematically, the Lasso optimization problem can be formulated as:

minimize: ||Y - Xβ||^2 + λ * ||β||_1
subject to: ||β||_1 <= t

**where:**
'Y' is the vector of observed responses
'X' is the design matrix of predictors
'β' is the vector of regression coefficients
'λ' is the regularization parameter that controls the amount of shrinkage
||.||^2 denotes the L2-norm (Euclidean norm) of a vector
||.||_1 denotes the L1-norm (sum of absolute values) of a vector
t is a positive constant representing an upper bound on the L1-norm of the coefficients
The objective function consists of two terms: the residual sum of squares (RSS) and the L1-norm penalty term. The parameter λ controls the trade-off between model fit and sparsity. A larger value of λ results in more coefficients being pushed towards zero, promoting sparsity in the model.

#### 1.5.3) Assumptions
**Linear Relationship:** Lasso regression assumes a linear relationship between the predictors and the response variable. It assumes that the relationship can be adequately represented by a linear model.
**No Perfect Multicollinearity:** Lasso regression assumes that there is no perfect multicollinearity among the predictors. Perfect multicollinearity occurs when there is an exact linear relationship between two or more predictors, which can lead to unstable coefficient estimates.
**Independence of Observations:** Lasso regression assumes that the observations are independent of each other. This assumption ensures that each observation provides unique information and avoids issues of dependence or autocorrelation.
**Homoscedasticity:** Lasso regression assumes that the variance of the error term is constant across all levels of the predictors. In other words, it assumes that the residuals have constant variance.
Normality of Residuals: Lasso regression assumes that the residuals follow a normal distribution with a mean of zero. This assumption allows for valid statistical inference and hypothesis testing.
#### 1.5.4) Conclusion
Lasso regression is a powerful technique for feature selection and regularization in linear regression models. It combines the benefits of variable selection and regularization by shrinking some coefficients to exactly zero, effectively performing automatic feature selection. The Lasso algorithm offers a flexible framework for handling high-dimensional datasets and can be used in various applications, such as predictive modeling, variable importance analysis, and model interpretability.


## <span style="color:green;">1.6) SUPPORT VECTOR REGRESSOR ALGORITHM</span>

#### 1.6.1) Introduction
Support Vector Regression (SVR) is a machine learning algorithm used for regression tasks. It is a variant of Support Vector Machines (SVM) and is particularly useful when dealing with continuous target variables. SVR aims to find a regression function that best fits the training data while minimizing the error.

#### 1.6.2) Mathematical Formulation
The mathematical formulation of the Support Vector Regressor algorithm involves finding an optimal hyperplane, which acts as a decision boundary, that maximizes the margin around the training data points. The hyperplane is defined by a linear function:

f(x) = w^T x + b
where w represents the weight vector, x is the input vector, and b is the bias term. The goal is to find w and b that minimize the error between the predicted outputs and the actual outputs for the training data.
To incorporate errors, SVR introduces a loss function that penalizes deviations from the desired outputs. The most commonly used loss function is the epsilon-insensitive loss function, defined as:

L(y, f(x)) = max(0, |y - f(x)| - ε)
where y is the true output, f(x) is the predicted output, and ε is a user-defined parameter that determines the width of the insensitive zone. The objective of SVR is to minimize the sum of the loss function and a regularization term, which controls the complexity of the model.

#### 1.6.3) Assumptions
The Support Vector Regressor algorithm makes the following assumptions:

**Linearity:** SVR assumes that the relationship between the input variables and the target variable is linear. If the relationship is highly nonlinear, SVR may not perform well without appropriate transformations or kernel functions.

**Independence:** The training data points are assumed to be independent and identically distributed (i.i.d). Violation of this assumption may affect the performance of the algorithm.

**Homoscedasticity:** The variance of the errors is assumed to be constant across all input variables. If the variance is heteroscedastic, preprocessing techniques such as data normalization or transformation may be required.

**Noisy Data:** SVR assumes that the training data may contain some degree of noise. The epsilon-insensitive loss function helps in handling noisy data by allowing a certain tolerance level (ε) for deviations.

**Large Dataset:** For large datasets consider using LinearSVR or SGDRegressor instead

**Method Relevance for Estimation:** This method is only relevant if this estimator is used as a sub-estimator of a meta-estimator, e.g. used inside a Pipeline. Otherwise it has no effect.

#### 1.6.4) Conclusion
Support Vector Regression (SVR) is a powerful algorithm for solving regression problems. It combines the principles of SVM with regression techniques to find an optimal hyperplane that best fits the training data. By making appropriate assumptions and utilizing the mathematical formulation, SVR can effectively model and predict continuous target variables. However, it is important to validate these assumptions and preprocess the data accordingly to achieve accurate results.For more detail visit [SVR Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)


## <span style="color:green;">1.7) K-NEIGHBORS REGRESSOR ALGORITHM</span>

#### 1.7.1) Introduction
KNeighborsRegressor is a machine learning algorithm used for regression tasks. It belongs to the family of k-nearest neighbors (KNN) algorithms and is based on the principle of finding the k closest instances in the training data to make predictions for new instances. KNeighborsRegressor estimates the target variable by taking the average (or weighted average) of the target values of its k nearest neighbors.

#### 1.7.2) Mathematical Formulation
The mathematical formulation of the KNeighborsRegressor algorithm involves finding the k nearest neighbors to a given test instance based on a distance metric, such as Euclidean distance. Once the nearest neighbors are identified, the algorithm estimates the target variable by taking the average (or weighted average) of the target values of those neighbors.

Let x_i be an instance in the training data with corresponding target value y_i, and x be a test instance for which we want to predict the target value. The predicted target value y_pred can be calculated as follows:

y_pred = (1/k) * sum(y_i)
where k is the number of nearest neighbors considered.

#### 1.7.3) Assumptions
The KNeighborsRegressor algorithm makes the following assumptions:

**Locality:** KNeighborsRegressor assumes that instances with similar feature values tend to have similar target values. It relies on the assumption that instances close to each other in the feature space are likely to have similar target values.

**Stationarity:** The algorithm assumes that the underlying relationship between the input variables and the target variable is stationary, meaning it does not change over time or across different regions of the feature space. If the relationship is non-stationary, the algorithm may not perform well.

**Feature Scaling:** KNeighborsRegressor assumes that the input features are on the same scale. If the features have different scales, it is recommended to perform feature scaling, such as normalization or standardization, to ensure that no single feature dominates the distance calculation.

**Noisy Data:** The algorithm assumes that the training data may contain some degree of noise. Outliers or noisy instances in the training data can significantly affect the predictions. It is important to preprocess the data and handle outliers appropriately.

**Relevance of Features:** KNeighborsRegressor assumes that all features are equally relevant in predicting the target variable. If some features are irrelevant or have low predictive power, it may be necessary to perform feature selection or dimensionality reduction techniques.
> ⚠️ **Warning:** Regarding the Nearest Neighbors algorithms, if it is found that two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on the ordering of the training data.

#### 1.7.4) Conclusion
KNeighborsRegressor is a versatile algorithm for regression tasks that relies on the concept of finding the nearest neighbors to make predictions. By making appropriate assumptions and utilizing the mathematical formulation, KNeighborsRegressor can effectively estimate the target variable based on the values of its neighbors. However, it is important to validate these assumptions and preprocess the data accordingly for optimal performance. For more information and detailed documentation of [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)


## <span style="color:green;">1.8) DECISION TREE REGRESSOR ALGORITHM</span>

#### 1.8.1) Introduction
DecisionTreeRegressor is a machine learning algorithm used for regression tasks. It belongs to the family of decision tree algorithms and is based on the concept of recursively partitioning the feature space into regions, where each region corresponds to a leaf node in the decision tree. DecisionTreeRegressor predicts the target variable by estimating the value in each leaf node.

#### 1.8.2) Mathematical Formulation
The mathematical formulation of the DecisionTreeRegressor algorithm involves recursively partitioning the feature space based on the feature values. The goal is to create a decision tree that minimizes the error between the predicted values and the actual target values.

Decision trees are constructed by selecting the best split point at each node based on a criterion such as mean squared error (MSE) or mean absolute error (MAE). The chosen criterion evaluates the quality of a split by measuring the impurity or variance reduction achieved by the split. The splitting process continues until a stopping criterion is met, such as reaching a maximum tree depth or a minimum number of samples required to split a node.

The predicted target value y_pred for a given instance can be obtained by traversing the decision tree and finding the corresponding leaf node. The value in that leaf node represents the estimated target value.

#### 1.8.3) Assumptions
The DecisionTreeRegressor algorithm makes the following assumptions:

**Feature Relevance:** DecisionTreeRegressor assumes that the input features are relevant and informative for predicting the target variable. If some features are irrelevant or have low predictive power, it may be necessary to perform feature selection or dimensionality reduction techniques.

**Independence:** The algorithm assumes that the training instances are independent and identically distributed (i.i.d). Violation of this assumption may lead to biased or inefficient models.

**Locality:** DecisionTreeRegressor assumes that instances with similar feature values tend to have similar target values. It aims to create partitions in the feature space that group similar instances together.

**Noisy Data:** The algorithm assumes that the training data may contain some degree of noise. Outliers or noisy instances in the training data can affect the structure of the decision tree and the resulting predictions. It is important to preprocess the data and handle outliers appropriately.

**Interactions:** DecisionTreeRegressor assumes that the relationship between the input features and the target variable can be adequately captured using axis-parallel splits. If the relationship involves complex interactions between features, decision trees may struggle to model it effectively.

#### 1.8.4) Conclusion
DecisionTreeRegressor is a versatile algorithm for regression tasks that constructs a decision tree to predict the target variable based on the feature values. By making appropriate assumptions and utilizing the mathematical formulation, DecisionTreeRegressor can effectively estimate the target variable by recursively partitioning the feature space. However, it is important to validate these assumptions and preprocess the data accordingly to achieve accurate and reliable predictions. For more information and detailed documentation on [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)


## <span style="color:green;">1.9) RANDOMFORESTREGRESSOR ALGORITHM</span>

#### 1.9.1) Introduction
RandomForestRegressor is a machine learning algorithm used for regression tasks. It belongs to the family of ensemble methods and is based on the concept of aggregating the predictions of multiple individual decision trees. RandomForestRegressor improves upon the decision tree algorithm by reducing overfitting and increasing prediction accuracy through the use of randomization.

#### 1.9.2) Mathematical Formulation
The mathematical formulation of the RandomForestRegressor algorithm involves constructing an ensemble of decision trees and aggregating their predictions. Each decision tree is built using a random subset of the training data and a random subset of the features.

To make a prediction for a new instance, RandomForestRegressor takes the average (or weighted average) of the predictions made by the individual decision trees in the ensemble. The randomization introduced in the algorithm helps to reduce the variance and improve the generalization ability of the model.

#### 1.9.3) Assumptions
The RandomForestRegressor algorithm makes the following assumptions:

**Feature Relevance:** RandomForestRegressor assumes that the input features are relevant and informative for predicting the target variable. If some features are irrelevant or have low predictive power, they may not be selected during the random feature subset sampling, and their impact on the predictions may be diminished.

**Independence:** The algorithm assumes that the training instances are independent and identically distributed (i.i.d). Violation of this assumption may lead to biased or inefficient models.

**Locality:** RandomForestRegressor assumes that instances with similar feature values tend to have similar target values. It aims to create partitions in the feature space that group similar instances together. The random feature subset sampling helps to capture different aspects of the data and improve the diversity among the decision trees.

**Noisy Data:** The algorithm assumes that the training data may contain some degree of noise. Outliers or noisy instances in the training data can affect the structure of individual decision trees and the resulting predictions. However, the ensemble nature of RandomForestRegressor helps to mitigate the impact of outliers by averaging the predictions of multiple trees.

**Interactions:** RandomForestRegressor assumes that the relationship between the input features and the target variable can be captured by a combination of simple interactions. While decision trees can model complex interactions, the random feature subset sampling may limit the ability of individual trees to capture higher-order interactions.

**Default Parameters:**
- criterion: squared_error
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1
- min_weight_fraction_leaf: 0.0
- max_features: 1.0
- max_leaf_nodes: None
- min_impurity_decrease: 0.0
- bootstrap: True
- oob_score: False (Out-Of-Bag Sample Score)
- n_jobs: None
- random_state: None
- verbose: 0 (Controls the verbosity when fitting and predicting.)
- warm_start: False
- ccp_alpha: 0.0 (Cost-Complexity Pruning)
- max_samples: None
- max_samples: None
  

    > - **Note:** 
    > 1. The default values for the parameters controlling the size of the trees (e.g. max_depth, min_samples_leaf, etc.) lead to fully grown and unpruned trees which can potentially be very large on some data sets.
    > 2. TTo reduce memory consumption, the complexity and size of the trees should be controlled by setting those parameter values.
    > 3. The features are always randomly permuted at each split.

#### 1.9.4) Conclusion
RandomForestRegressor is a powerful algorithm for regression tasks that leverages the strength of multiple decision trees through ensemble learning. By making appropriate assumptions and utilizing the mathematical formulation, RandomForestRegressor can effectively estimate the target variable by aggregating the predictions of individual trees. It provides improved prediction accuracy and robustness compared to a single decision tree. However, it is important to validate these assumptions and preprocess the data accordingly for optimal performance. For more information and detailed documentation, follow link [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)


## <span style="color:green;">1.10) ADABOOSTREGRESSOR ALGORITHM</span>

#### 1.10.1) Introduction
AdaBoostRegressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases.
It is a machine learning algorithm used for regression tasks. It belongs to the family of boosting algorithms and is based on the concept of combining multiple weak learners to create a strong learner. AdaBoostRegressor focuses on iteratively adjusting the weights of training instances to improve the overall predictive performance.

#### 1.10.2) Mathematical Formulation
The mathematical formulation of the AdaBoostRegressor algorithm involves iteratively training a sequence of weak regressors and combining their predictions to obtain the final prediction. Each weak regressor is trained on a modified version of the training data, where the weights of the instances are adjusted based on their previous performance.

At each iteration, AdaBoostRegressor assigns higher weights to instances that were poorly predicted by the previous weak regressors. This emphasizes the importance of these instances and forces subsequent weak regressors to focus on improving their predictions. The final prediction is obtained by taking a weighted average of the predictions made by the weak regressors, where the weights are determined by their performance during training.

#### 1.10.3) Assumptions
The AdaBoostRegressor algorithm makes the following assumptions:

**Weak Learner Availability:** AdaBoostRegressor assumes the availability of a set of weak regressors. These weak regressors can be any regression model capable of providing predictions slightly better than random guessing. The algorithm combines the predictions of these weak regressors to obtain an accurate overall prediction.

**Feature Relevance:** AdaBoostRegressor assumes that the input features are relevant and informative for predicting the target variable. The algorithm relies on the weak regressors' ability to learn from the features and make accurate predictions. If some features are irrelevant or have low predictive power, their impact on the overall prediction may be diminished.

**Independence:** The algorithm assumes that the training instances are independent and identically distributed (IID). Violation of this assumption may lead to biased or inefficient models.

**Weak Regressor Quality:** AdaBoostRegressor assumes that the weak regressors perform better than random guessing, even if only slightly. The algorithm relies on the iterative process of adjusting instance weights to boost the performance of the weak regressors. If the weak regressors consistently perform poorly, AdaBoostRegressor may struggle to improve the overall prediction accuracy.

**Noisy Data:** The algorithm assumes that the training data may contain some degree of noise. Outliers or noisy instances in the training data can affect the performance of individual weak regressors and the resulting predictions. The iterative nature of AdaBoostRegressor allows it to adapt to the noisy instances by assigning them higher weights during training.

#### 1.10.4) Conclusion
AdaBoostRegressor is a powerful algorithm for regression tasks that combines the predictions of multiple weak regressors to obtain an accurate overall prediction. By making appropriate assumptions and utilizing the mathematical formulation, AdaBoostRegressor iteratively adjusts instance weights to emphasize the importance of poorly predicted instances. It provides improved prediction accuracy compared to a single weak regressor. However, it is important to validate these assumptions and preprocess the data accordingly for optimal performance. For more information visit [AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)


## <span style="color:green;">1.11) GRADIENTBOOSTINGREGRESSOR ALGORITHM</span>

#### 1.11.1) Introduction
GradientBoostingRegressor is a machine learning algorithm used for regression tasks. It belongs to the family of boosting algorithms and is based on the concept of sequentially adding weak learners to create a strong learner. GradientBoostingRegressor focuses on minimizing a loss function by iteratively optimizing the model's parameters.

#### 1.11.2) Mathematical Formulation
The mathematical formulation of the GradientBoostingRegressor algorithm involves iteratively training a sequence of weak regressors and combining their predictions to obtain the final prediction. At each iteration, a weak regressor is trained to approximate the negative gradient of the loss function with respect to the target variable. The weak regressor's predictions are then combined with the predictions made by the previous weak regressors to update the overall prediction.

To update the prediction at each iteration, the algorithm uses a gradient descent-like procedure. It adjusts the parameters of each weak regressor to minimize the loss function with respect to the current predictions and the true target values. The final prediction is obtained by aggregating the predictions of all the weak regressors.

#### 1.11.3) Assumptions
The GradientBoostingRegressor algorithm makes the following assumptions:

**Weak Learner Availability:** GradientBoostingRegressor assumes the availability of a set of weak regressors. These weak regressors can be any regression model capable of providing predictions slightly better than random guessing. The algorithm combines the predictions of these weak regressors to obtain an accurate overall prediction.

**Feature Relevance:** GradientBoostingRegressor assumes that the input features are relevant and informative for predicting the target variable. The algorithm relies on the weak regressors' ability to learn from the features and make accurate predictions. If some features are irrelevant or have low predictive power, their impact on the overall prediction may be diminished.

**Independence:** The algorithm assumes that the training instances are independent and identically distributed (IID). Violation of this assumption may lead to biased or inefficient models.

**Weak Regressor Quality:** GradientBoostingRegressor assumes that the weak regressors perform better than random guessing, even if only slightly. The algorithm relies on the iterative process of updating the predictions based on the negative gradient of the loss function. If the weak regressors consistently perform poorly, GradientBoostingRegressor may struggle to improve the overall prediction accuracy.

**Differentiability of Loss Function:** GradientBoostingRegressor assumes that the loss function used is differentiable with respect to the predicted values. The algorithm relies on calculating the negative gradient of the loss function to update the predictions. If the loss function is not differentiable or not well-defined, GradientBoostingRegressor may encounter difficulties during the optimization process.
> **Notes:** 
The features are always randomly permuted at each split. Therefore, the best found split may vary, even with the same training data and max_features=n_features, if the improvement of the criterion is identical for several splits enumerated during the search of the best split. To obtain a deterministic behaviour during fitting, random_state has to be fixed.

> **Tip:**
> HistGradientBoostingRegressor is a much faster variant of this algorithm for intermediate datasets (n_samples >= 10000). For detail [visit](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

#### 1.11.4) Conclusion
GradientBoostingRegressor is a powerful algorithm for regression tasks that combines the predictions of multiple weak regressors to obtain an accurate overall prediction. By making appropriate assumptions and utilizing the mathematical formulation, GradientBoostingRegressor iteratively updates the predictions based on the negative gradient of the loss function. It provides improved prediction accuracy compared to a single weak regressor. However, it is important to validate these assumptions and preprocess the data accordingly for optimal performance. For more information and detailed documentation on the [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)


## <span style="color:green;">1.12) LGBMREGRESSOR ALGORITHM</span>

#### 1.12.1) Introduction
LGBMRegressor is a machine learning algorithm used for regression tasks. It is based on the LightGBM framework, which is a gradient boosting framework that uses tree-based learning algorithms. LGBMRegressor is known for its high efficiency and accuracy, making it a popular choice for regression problems.

#### 1.12.2) Mathematical Formulation
The mathematical formulation of the LGBMRegressor algorithm involves building an ensemble of decision trees and optimizing them to minimize a loss function. LGBMRegressor uses a gradient-based approach to train the ensemble of trees in a sequential manner.

At each iteration, LGBMRegressor fits a decision tree to the negative gradient of the loss function with respect to the predicted values. The trees are built in a leaf-wise manner, where each split is chosen to maximize the decrease in the loss function. This strategy leads to faster convergence and better accuracy compared to traditional depth-wise tree growth.

The final prediction is obtained by aggregating the predictions made by all the trees in the ensemble. LGBMRegressor also incorporates regularization techniques like L1 and L2 regularization to prevent overfitting and improve generalization.

#### 1.12.3) Assumptions
The LGBMRegressor algorithm makes the following assumptions:

Feature Relevance: LGBMRegressor assumes that the input features are relevant and informative for predicting the target variable. The algorithm relies on the decision trees' ability to learn from the features and make accurate predictions. If some features are irrelevant or have low predictive power, their impact on the overall prediction may be diminished.

**Independence:** The algorithm assumes that the training instances are independent and identically distributed (i.i.d). Violation of this assumption may lead to biased or inefficient models.

**Weak Learner Quality:** LGBMRegressor assumes that the decision trees used as weak learners can capture complex patterns and dependencies in the data. The algorithm relies on the ensemble of trees to learn from the data and make accurate predictions. If the weak learners are not capable of capturing the underlying relationships in the data, LGBMRegressor may struggle to improve the prediction accuracy.

**Differentiability of Loss Function:** LGBMRegressor assumes that the loss function used is differentiable with respect to the predicted values. The algorithm optimizes the ensemble of trees by minimizing the loss function using gradient-based optimization techniques. If the loss function is not differentiable or not well-defined, LGBMRegressor may encounter difficulties during the optimization process.

**Noisy Data:** The algorithm assumes that the training data may contain some degree of noise. LGBMRegressor incorporates regularization techniques like L1 and L2 regularization to mitigate the impact of noisy instances and prevent overfitting.

#### 1.12.4) Conclusion
LGBMRegressor is a highly efficient and accurate algorithm for regression tasks. By making appropriate assumptions and utilizing the mathematical formulation, LGBMRegressor builds an ensemble of decision trees in a leaf-wise manner, optimizing them to minimize a loss function. It provides improved prediction accuracy and faster convergence compared to traditional gradient boosting algorithms. However, it is important to validate these assumptions and preprocess the data accordingly for optimal performance. For more information and detailed documentation [LGBMRegressor](https://github.com/microsoft/LightGBM)