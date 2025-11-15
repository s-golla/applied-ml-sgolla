# Project 4: Predicting Fare on the Titanic with Regression

## Objective
The goal of this project was to build and evaluate various regression models to predict the `fare` paid by passengers on the Titanic. The project explored how different features and model types could be used to explain the variance in ticket prices.

## Data Preparation
The analysis began with loading the Titanic dataset. Key data preparation steps included:
- **Imputing Missing Values**: Missing `age` values were filled using the median age of all passengers.
- **Feature Engineering**: A `family_size` feature was created by combining the `sibsp` (siblings/spouses) and `parch` (parents/children) columns.
- **Data Conversion**: The categorical `sex` feature was converted into a numeric format (`sex_numeric`) for use in the models.

## Modeling and Evaluation
Four distinct cases were defined to test the predictive power of different feature combinations using Linear Regression:
- **Case 1**: `age` only
- **Case 2**: `family_size` only
- **Case 3**: `age` and `family_size`
- **Case 4**: `sex` and `pclass`

### Initial Findings
- **Case 4 (`sex` and `pclass`)** was the best-performing simple linear model, achieving a **Test R² of approximately 0.34**. This indicates that passenger class and sex are significant predictors of fare.
- The other cases performed poorly, with R² scores close to zero, suggesting that `age` and `family_size` have a weak linear relationship with `fare`.

### Advanced Models
Using the features from the best case (`sex` and `pclass`), several other regression models were trained and evaluated:
- **Ridge Regression**: Performed similarly to the linear model, suggesting that regularization was not critical for this simple feature set.
- **Elastic Net**: Showed a slight improvement over the linear and Ridge models.
- **Polynomial Regression (Degree 3)**: This model yielded the best performance, capturing non-linear relationships between the features and the target.

## Results
The final comparison of models trained on the `sex` and `pclass` features revealed the following:

| Model                  | Test R² | Test RMSE | Test MAE |
|------------------------|---------|-----------|----------|
| Linear Regression      | 0.340   | 30.90     | 20.40    |
| Ridge Regression       | 0.340   | 30.89     | 20.36    |
| ElasticNet             | 0.369   | 30.23     | 19.18    |
| **Polynomial (Deg 3)** | **0.446**   | **28.30**     | **17.61**    |

The **Polynomial Regression (Degree 3) model was the top performer**, explaining approximately 44.6% of the variance in fare prices.

## Conclusion
This project successfully demonstrated that `pclass` and `sex` are the most influential features for predicting `fare` in the Titanic dataset. While simple linear models provided a decent baseline, a **Polynomial Regression model was able to capture more complexity and deliver the best predictive performance**. The analysis also highlighted the challenges of predicting a skewed target variable like `fare`, where outliers can significantly impact model performance.