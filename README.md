# petrophysical-analysis-for-well-logging-and-hydrocarbon-exploration
Machine Learning Models for Well Log Analysis

# Internship Project at ONGC

## Developed Machine Learning Models for Well Log Analysis

### Project Description

- **Predictive Model**: Created a model to estimate log curves, such as density porosity, by leveraging relationships with other manually obtainable logs.
- **Classification Model**: Developed a model to analyze oil well logs and categorize them into specific lithology types at various depths.

### Tools/Technologies Used

- Python
- Jupyter
- Machine Learning Algorithms
- Data Analysis

## 1.DPOR Prediction:

### Data Description
The dataset includes well log data with the following columns:
DPOR - Density Porosity
GR - Gamma Ray
RHOB - Bulk Density
RILD - Deep Resistivity
RILM - Medium Resistivity
RLL3 - Shallow Resistivity
SP - Spontaneous Potential
SPOR - Sonic Porosity

### Models and Results:
- Linear Regression
Description: Linear Regression is a simple and interpretable model that assumes a linear relationship between the input features and the target variable.
Performance:
Mean Absolute Error: 1.41
Mean Squared Error: 8.55
R-squared: 0.92
Pros:
Easy to implement and interpret.
Works well with linearly separable data.
Cons:
May underperform if the relationship between features and target is not linear.

- Random Forest
Description: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees.
Performance:
Best Parameters: {'n_estimators': 200, 'min_samples_split': 2, 'max_depth': None}
Mean Absolute Error: 1.15
Mean Squared Error: 3.49
R-squared: 0.96
Pros:
Handles large datasets with higher dimensionality well.
Reduces overfitting compared to individual decision trees.
Cons:
Computationally intensive, especially with large datasets and many trees.
Less interpretable than linear models.

- Neural Network
Description: Neural Networks are a class of models inspired by the human brain, capable of capturing complex relationships between inputs and outputs through multiple layers of interconnected nodes.
Performance:
The performance metrics of the neural network were less favorable compared to Random Forest.
Learning Curves: The plot of training and validation loss shows the learning behavior of the model over epochs.
Pros:
Can model complex, non-linear relationships.
Scalable with increasing data and computational power.
Cons:
Prone to overfitting, especially with small datasets.
Requires more computational resources and careful tuning of hyperparameters.
Less interpretable compared to simpler models.

### Visualization
Actual vs. Predicted DPOR: Scatter plots and line plots comparing the actual and predicted DPOR values for both Linear Regression and Random Forest models.
![dpor_actual_vs_predicted](https://github.com/vaish414/petrophysical-analysis-for-well-logging-and-hydrocarbon-exploration/assets/106098796/bedcffdd-e79b-4d4d-ad19-036bba41cf68)

Learning Curves: Plots of training and validation loss for the neural network, helping in understanding the model's learning behavior and identifying issues like overfitting or underfitting.
![dpor_loss](https://github.com/vaish414/petrophysical-analysis-for-well-logging-and-hydrocarbon-exploration/assets/106098796/6a07451b-30a2-4011-8ab5-4744f50ac107)

![FNN_DPOR](https://github.com/vaish414/petrophysical-analysis-for-well-logging-and-hydrocarbon-exploration/assets/106098796/7e5efd7f-aebc-4780-a1f3-ad461ef67821)


### Conclusion
Linear Regression provided a good baseline with interpretable results.
Random Forest outperformed other models in terms of accuracy (R-squared = 0.96), suggesting it is well-suited for this prediction task due to its ability to handle non-linear relationships and large feature sets.
Neural Network required careful tuning and more computational resources but did not perform as well as Random Forest in this case, indicating potential issues with overfitting or the need for more data.

-Future Work
Hyperparameter Tuning: Further refinement of hyperparameters for neural networks to improve performance.
Feature Engineering: Exploring additional features or transformations to enhance model accuracy.
Model Interpretability: Using techniques like SHAP values to interpret model predictions, especially for complex models like Random Forest and Neural Networks.
