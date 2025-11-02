# Project 02: Titanic Survival Analysis

## Overview
This project explores the Titanic dataset to build a machine learning model for predicting passenger survival. The analysis includes data exploration, preparation, and model development.

## Dataset Description
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

## Technical Details
- Language: Python
- Key Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn
- Jupyter Notebook: ml02_sgolla.ipynb