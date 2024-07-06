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

## 2.Lithology Classification:

### Project Overview
Lithology classification is crucial in geophysical studies for understanding subsurface geology. In this project, we use machine learning techniques to predict lithologies from geophysical features. The project involves data preprocessing, feature extraction, model training, and evaluation.

### Dataset
The dataset used in this project contains depth intervals and geophysical features. The target variable is the lithology class.
The dataset includes well log data with the following columns:
TopDepth
BotDepth	
_CAL - Caliper	
_GR	- Gamma Ray
_SP	- Spontaneous Potential
_LLD - Deep Resistivity
_LLS - Shallow Resistivity	
_AC - Accoustic Log	
_DEN - Density log	
_PEF - Photoelectric effect
Lith_Section

### Models Used
We explored various machine learning models for lithology classification. The models and their respective characteristics are listed below:

### 1. Long Short-Term Memory (LSTM) Neural Network
- **Description**: LSTM is a type of recurrent neural network (RNN) that is well-suited for sequence modeling tasks. It can capture long-term dependencies and temporal relationships within the data.
- **Advantages**:
  - Effective in capturing temporal dependencies in sequential data.
  - Can handle varying sequence lengths.
- **Disadvantages**:
  - Computationally intensive and time-consuming to train.
  - Requires a large amount of data to perform well.
- **Accuracy**: 65.8%

### 2. XGBoost Classifier
- **Description**: XGBoost is an efficient and scalable implementation of gradient boosting framework. It is known for its speed and performance.
- **Advantages**:
  - High performance and accuracy.
  - Efficient handling of missing data.
  - Supports regularization to prevent overfitting.
- **Disadvantages**:
  - Requires careful tuning of hyperparameters.
  - Can be complex to interpret compared to simpler models.
- **Accuracy**: TBD (This should be updated with the final accuracy after hyperparameter tuning)

### 3. Random Forest Classifier
- **Description**: Random Forest is an ensemble learning method that constructs multiple decision trees and merges their results to improve accuracy and reduce overfitting.
- **Advantages**:
  - Robust to overfitting due to ensemble averaging.
  - Can handle high-dimensional data well.
- **Disadvantages**:
  - Can be computationally expensive for large datasets.
  - May require significant memory.
- **Accuracy**: TBD

### Support Vector Machine (SVM)
- **Description**: SVM is a supervised learning model that finds the optimal hyperplane to separate different classes in the feature space.
- **Advantages**:
  - Effective in high-dimensional spaces.
  - Robust to overfitting, especially in high-dimensional data.
- **Disadvantages**:
  - Not well-suited for very large datasets.
  - Can be sensitive to the choice of kernel and hyperparameters.
- **Accuracy**: TBD

### Results
- The LSTM model achieved an accuracy of 65.8%.
- The XGBoost model is under hyperparameter tuning to determine its final accuracy.
- Random Forest and SVM models' accuracies will be determined after evaluation.
