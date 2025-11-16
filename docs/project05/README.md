# Project 5: Red Wine Quality Classification with Ensemble Learning 🍷

This repository contains the Jupyter Notebook (`ml05_sgolla.ipynb`) for Project 5, focusing on applying and evaluating **ensemble machine learning models** to classify the quality of red wine.

The goal of this project is to compare different ensemble strategies—specifically **Random Forest (Bagging)** and a **Voting Classifier (Heterogeneous Ensemble)**—to determine the most effective and robust approach for predicting wine quality based on physicochemical features.

## 💾 Dataset

The project utilizes the **Red Wine Quality Dataset** (part of the UCI Wine Quality Data Set).

  * **File:** `winequality-red.csv`
  * **Features:** 11 physicochemical properties (e.g., fixed acidity, volatile acidity, alcohol, pH).
  * **Target:** The original `quality` score (0-10) is transformed into a **3-class categorical variable**: **Low** ($\le 4$), **Medium** ($\le 6$), and **High** ($\ge 7$).

## ⚙️ Methodology

The analysis follows a standard machine learning workflow, with a focus on evaluating ensemble models:

1.  **Data Loading & Preparation:** Load the data and convert the multi-class target variable into a balanced 3-class system.
2.  **Train/Test Split:** Data is split 80/20 with **stratification** to maintain class proportions.
3.  **Model Selection (Chosen 2):**
      * **Random Forest (100 Trees):** A powerful **Bagging** ensemble method.
      * **Voting Classifier (Soft):** A **Heterogeneous** ensemble combining a Decision Tree (DT), Support Vector Machine (SVC), and a Neural Network (MLP).
4.  **Evaluation:** Models are trained and compared using standard metrics, with a strong emphasis on **Test F1 Score** (weighted) and the **Accuracy/F1 Gap** between training and testing performance to assess generalization and overfitting.

## 📊 Key Results

The table below summarizes the performance of the two primary models evaluated in the notebook.

| Model | Train Accuracy | Test Accuracy | Accuracy Gap | Train F1 | Test F1 | F1 Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (100)** | $1.0000$ | **$0.8875$** | $0.1125$ | $1.0000$ | **$0.8661$** | $0.1339$ |
| **Voting (DT + SVM + NN)** | $0.9226$ | $0.8656$ | **$0.0570$** | $0.9061$ | $0.8434$ | **$0.0626$** |

### **Conclusion**

  * **Best Predictive Power:** The **Random Forest** achieved the highest Test F1 Score ($\mathbf{0.8661}$), indicating it is the most effective at classifying the wine quality categories.
  * **Best Generalization/Stability:** The **Voting Classifier** demonstrated superior robustness, showing a significantly smaller **Accuracy Gap** ($\mathbf{0.0570}$) compared to the Random Forest ($\mathbf{0.1125}$). This suggests the Voting Classifier is less prone to overfitting and would be more reliable on truly unseen data.

## 🚀 Next Steps

To improve performance further, the following steps were identified:

1.  **Hyperparameter Tuning:** Systematically tune the Random Forest (`max_depth`, `min_samples_split`) to reduce the overfitting gap while maintaining high test performance.
2.  **Data Scaling:** Apply standardization to the features and re-evaluate the **Voting Classifier**. Scaling is crucial for the SVM and Neural Network base estimators and is likely to improve the overall ensemble performance.
3.  **Class Imbalance Handling:** Implement **SMOTE** (Synthetic Minority Over-sampling Technique) on the training data to better address the minority classes (Low and High quality wines), which would boost the weighted F1 score.

## 🛠️ Prerequisites

To run this notebook, you will need Python 3.x and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## 📝 Usage

1.  Clone this repository.
2.  Ensure `winequality-red.csv` is in the project directory.
3.  Open the notebook in JupyterLab or VS Code:
    ```bash
    jupyter notebook ml05_sgolla.ipynb
    ```
4.  Run all cells sequentially to reproduce the data preparation, model training, evaluation, and conclusion sections.