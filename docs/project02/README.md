# Project 02: Machine Learning Classification Analysis

## Overview
This project explores two classic machine learning datasets - the Titanic survival prediction and Iris species classification. The analysis demonstrates different approaches to classification problems and highlights the contrasts between binary and multi-class classification.

## Part 1: Titanic Survival Analysis

### Dataset Description
The Titanic dataset (from seaborn) contains information about passengers including:
- Demographic information (age, sex)
- Ticket information (class, fare)
- Family relationships (siblings/spouses, parents/children)
- Embarkation details
- Survival status (target variable)

## Analysis Structure

### 1. Data Exploration
- Dataset contains 891 passengers with 15 features
- Key features include: survived, pclass, sex, age, sibsp, parch, fare, embarked
- Missing values identified in:
  - Age (19.9% missing)
  - Deck (77.2% missing)
  - Embarked/embark_town (0.2% missing)

### 2. Data Visualization
1. Scatter Matrix Plot
   - Visualizes relationships between age, fare, and passenger class
   - Shows distributions and correlations

2. Age vs Fare by Gender Plot
   - Demonstrates fare distribution across age groups
   - Highlights gender differences in pricing

3. Age Distribution Plot
   - Shows passenger age distribution
   - Includes mean and median age indicators
   - Reveals most passengers were young to middle-aged adults

4. Class Distribution by Survival Plot
   - Shows survival rates across passenger classes
   - Indicates higher survival rates in higher classes

### 3. Feature Engineering
- Created new features:
  - family_size (combining siblings/spouses and parents/children)
  - Encoded categorical variables (sex, embarked)
  - Converted boolean features to integers

### 4. Data Preprocessing
- Handled missing values:
  - Age: filled with median
  - Embark_town: filled with mode
- Selected relevant features for modeling:
  - age, fare, pclass, sex, family_size

### 5. Data Splitting
- Implemented both basic and stratified splits
- Used 80-20 train-test split
- Maintained class distribution in stratified sampling

## Key Findings
1. Strong correlation between passenger class and survival
2. Gender was a significant factor in survival rates
3. Age distribution shows most passengers were between 20-40 years
4. Family size might influence survival chances

## Next Steps
1. Feature selection and engineering refinement
2. Model selection and training
3. Model evaluation and optimization
4. Cross-validation implementation

## Part 2: Iris Species Classification

### Dataset Description
The Iris dataset (from scikit-learn) contains measurements for 150 iris flowers:
- Four numerical features (sepal length/width, petal length/width)
- Three species classes (setosa, versicolor, virginica)
- No missing values
- Perfectly balanced classes (50 samples per species)

### Analysis Structure

1. Data Exploration
   - 150 samples with 4 features
   - Clear feature correlations with species
   - Well-separated classes in feature space

2. Data Visualization
   - Scatter matrix showing feature relationships
   - Box plots demonstrating feature distributions by species
   - Clear visual separation between species

3. Model Development
   - Multiple classifiers tested:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - K-Nearest Neighbors
   - High accuracy across all models (~95-98%)

## Comparative Analysis

### Data Quality and Preparation
1. **Titanic Dataset**
   - Required significant preprocessing
   - Had missing values
   - Mixed feature types
   - Imbalanced classes

2. **Iris Dataset**
   - Minimal preprocessing needed
   - No missing values
   - All numerical features
   - Balanced classes

### Model Performance
1. **Titanic Models**
   - Moderate accuracy (75-85%)
   - Required careful feature engineering
   - Affected by class imbalance

2. **Iris Models**
   - High accuracy (95-98%)
   - Worked well with raw features
   - Consistent performance across models

### Learning Outcomes
1. **Data Handling**
   - Experience with different data quality scenarios
   - Practice with various preprocessing techniques
   - Understanding of feature engineering importance

2. **Classification Challenges**
   - Binary vs Multi-class classification
   - Balanced vs Imbalanced datasets
   - Real-world vs Ideal data conditions

## Technical Details
- Language: Python
- Key Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn
- Jupyter Notebooks: 
  - Titanic Analysis: ml02_sgolla.ipynb
  - Iris Analysis: ml02_sgolla.ipynb (Section 5)

## Conclusions
1. Data quality significantly impacts model performance
2. Real-world data requires more preprocessing
3. Both binary and multi-class problems can be effectively solved with similar approaches
4. Feature engineering is crucial for complex, real-world problems
5. Model selection should consider data characteristics and problem complexity