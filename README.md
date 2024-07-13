# Shipment Time Prediction with OLS, LASSO, Ridge, Elastic Net and Random Forest

The objective of this project is to analyze and predict delivery times using a dataset from an e-commerce platform, focusing on data cleaning, variable transformation, and model selection.

## Author

Sandro Gassner

## Date

21.01.2022

## Methodology

### Data Source

The data used in this project is from the [Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/Brazilian-ecommerce) available on Kaggle.

### Data Cleaning

The data cleaning process involved several steps to ensure the quality and usability of the dataset:

1. **Removing Missing Values**: Rows with missing values were removed to ensure complete data for analysis.
2. **Filtering Unrealistic Values**: Entries with zero or negative values for key variables such as weight, freight rate, distance, and shipping time were filtered out.
3. **Handling Categorical Variables**: Categorical variables were converted into dummy variables using one-hot encoding to prepare the data for various modeling approaches.
4. **Removing Small Groups**: Product categories with fewer than 20 observations were excluded to avoid issues with small sample sizes.

### Variable Transformation

To address skewness and improve model performance, the following transformations were applied:

1. **Log-Transformation**: Highly skewed variables were log-transformed to normalize their distribution.
2. **Outlier Removal**: Outliers were identified and removed based on the interquartile range (IQR) to enhance model accuracy.

### Modeling Approach

Several predictive models were tested to identify the best approach for predicting shipping time:

1. **Ordinary Least Squares (OLS) Regression**: A basic linear regression model.
2. **LASSO Regression**: A regularization technique that performs variable selection and regularization to enhance prediction accuracy.
3. **Ridge Regression**: Another regularization method that addresses multicollinearity by shrinking coefficients.
4. **Elastic Net**: Combines the penalties of LASSO and Ridge to balance between variable selection and coefficient shrinkage.
5. **Random Forest**: An ensemble method that builds multiple decision trees and merges their predictions for improved accuracy and robustness.

### Preferred Model

The **Random Forest** model was selected as the preferred approach due to its superior performance. This model offers several advantages:

- Automatically handles interactions between variables.
- Non-parametric nature makes it robust to outliers.
- Provides high predictive accuracy through ensemble learning.

The Random Forest model achieved the highest out-of-sample R-squared value, making it the best model for predicting delivery times.

### Results

The project results demonstrate that the Random Forest model provides the best predictive performance for delivery times. The following table summarizes the out-of-sample R-squared values for the models tested:

| Model          | Out-of-Sample R² |
|----------------|------------------|
| Random Forest  | 0.5351           |
| Elastic Net    | 0.4883           |
| OLS            | 0.4882           |

### Additional Methods

Other methods and approaches tested include:

- **Elastic Net Regression**: Explored as a combination of LASSO and Ridge regression techniques.
- **Alternative Transformations**: Square Root and Cube Root transformations were considered but performed worse than log-transformation.
- **Different Train/Test Splits**: Various splits and random seeds were tested to ensure robustness and consistency of results.
