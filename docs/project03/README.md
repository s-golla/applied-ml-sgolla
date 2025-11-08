# P3: Building a Classifier - Titanic Survival Prediction

**Author:** Saratchandra Golla  
**Dataset:** Titanic (from seaborn)

## Introduction

This project aims to build and evaluate machine learning classification models to predict passenger survival on the Titanic, using the publicly available seaborn Titanic dataset. We will compare the performance of three different classifier types are Decision Tree (DT), Support Vector Machine (SVM), and a Neural Network (NN) across three distinct feature sets to determine the most effective approach for this prediction task. The entire process will follow a structured methodology: data preparation, feature selection, model training, and performance evaluation using metrics like accuracy, precision, recall, and F1-score.

## 1. Project Setup and Data Loading

We begin by importing all necessary libraries for data manipulation, modeling, and visualization. The Titanic dataset is then loaded directly from the `seaborn` library, and we perform an initial inspection to understand its structure, data types, and identify any missing values.

## 2. Data Exploration and Preparation

### 2.1 Handle Missing Values and Clean Data
Missing values for continuous features like `age` are imputed using the median, while missing values for categorical features like `embark_town` are filled using the mode.

### 2.2 Feature Engineering
A new feature, `family_size`, is created. Categorical features (`sex`, `embarked`, `alone`) are converted into numerical formats for model training.

## 3. Feature Selection and Justification

We evaluate model performance across three different input feature sets with the target being `survived`.

- **Case 1:** `alone` (A simple binary feature)
- **Case 2:** `age` (A continuous, potentially predictive feature)
- **Case 3:** `age` and `family_size` (A combination of two numerical features)

## 4. Model Training and Evaluation

Three types of models are trained and evaluated on the feature sets defined above.

- **Decision Tree Classifier**
- **Support Vector Classifier (SVC) with RBF Kernel**
- **Multi-Layer Perceptron (MLP) Neural Network**

## 5. Summary of Results

The table below summarizes the key performance metrics (Accuracy, Precision, Recall, F1-Score) from the test data for the target class (Survival=1).

| Model Type           | Case   | Features Used     | Accuracy (Test) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|:---------------------|:-------|:------------------|:---------------:|:-------------------:|:----------------:|:------------------:|
| Decision Tree        | Case 1 | `alone`           |      63.00%     |        51.00%       |      58.00%      |       54.00%       |
|                      | Case 2 | `age`             |      61.00%     |        50.00%       |      17.00%      |       26.00%       |
|                      | Case 3 | `age + family_size` |      59.00%     |        45.00%       |      33.00%      |       38.00%       |
| SVM (RBF Kernel)     | Case 1 | `alone`           |      63.00%     |        51.00%       |      58.00%      |       54.00%       |
|                      | Case 2 | `age`             |      63.00%     |        71.00%       |      7.00%       |       13.00%       |
|                      | Case 3 | `age + family_size` |      63.00%     |        71.00%       |      7.00%       |       13.00%       |
| Neural Network (MLP) | Case 3 | `age + family_size` |      66.00%     |        57.00%       |      46.00%      |       51.00%       |

## 6. Reflection

1.  **Which model performed the best and on which case?**
    Based on the F1-score, the **Decision Tree and SVM models on Case 1 (`alone`)** performed the best, each achieving an F1-score of **54.00%**. Although the Neural Network (Case 3) achieved the highest accuracy at 66.00%, its F1-score was slightly lower. This highlights that the simple binary feature `alone` was a surprisingly powerful predictor.

2.  **How did the feature sets impact performance?**
    The impact of feature sets was highly dependent on the model. For the Decision Tree and SVM, the single-feature **Case 1 (`alone`)** outperformed the multi-feature cases. For the Neural Network, the combination of features in Case 3 was what enabled it to achieve the highest accuracy, showing its ability to model more complex relationships.

3.  **What is a possible next step for model improvement?**
    The most critical next step is to incorporate the highly predictive **`sex` and `pclass` features**. Following that, **hyperparameter tuning** (e.g., using `GridSearchCV`) and applying **feature scaling** would be essential to optimize each model further.