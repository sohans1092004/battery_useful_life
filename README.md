# Battery Remaining Useful Life (RUL) Prediction for Electric Vehicles

This project predicts the **Remaining Useful Life (RUL)** of EV batteries using machine learning.  
It involves comprehensive **data analysis**, **feature engineering**, and the application of multiple regression and ensemble learning techniques to build an accurate predictive model.

---

## Overview

Electric vehicle (EV) batteries degrade over time due to repeated charge–discharge cycles.  
Accurately estimating their **Remaining Useful Life (RUL)** helps optimize performance, schedule maintenance, and improve overall reliability.

In this project, various regression and ensemble algorithms are applied to predict RUL from battery health parameters.  
The workflow covers every step — from exploratory data analysis (EDA) to model evaluation.

---

## Project Workflow

### 1. Data Preprocessing
- Loaded and cleaned dataset using **pandas**.
- Handled missing values, dropped irrelevant columns, and separated features (`X`) and target (`y`).
- Scaled features using **StandardScaler** for zero-mean, unit-variance normalization.

### 2. Exploratory Data Analysis (EDA)
- Analyzed feature distributions, summary statistics, and correlations.
- Visualized relationships between features and RUL using:
  - **Histograms & Boxplots** (to detect outliers)
  - **Heatmaps** (to identify correlated features)
  - **Pairplots** (to observe feature interactions)
- Derived insights on how voltage, current, and temperature affect RUL.

### 3. Feature Engineering
- Selected key predictors based on correlation and domain relevance.
- Prepared training and testing datasets for regression modeling.

### 4. Model Training
Trained and compared multiple **regression models** using `scikit-learn`:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- AdaBoost & Gradient Boosting  
- Support Vector Regressor (SVR)  
- K-Nearest Neighbors (KNN)

### 5. Ensemble Learning
Implemented advanced ensemble methods to improve performance:
- **Bagging Regressor:** Reduced model variance by averaging multiple estimators.  
- **Stacking Regressor:** Combined predictions of diverse base models using a meta-learner.

### 6. Model Evaluation
Evaluated models using key metrics:
- **R² Score** (Model Fit)
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**  
Compared all models and identified the best-performing approach for RUL prediction.

---

## Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python 3.x |
| Data Analysis | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Environment | Jupyter Notebook |

---

## Results & Insights
- Key features like **voltage drop rate** and **discharge time** strongly influence RUL.  
- Ensemble methods outperformed individual models, achieving the most stable predictions.  
- The final stacked model showed improved accuracy and generalization.

---

## Key Highlights
- Performed **comprehensive EDA** and visualizations for insight extraction.  
- Built and tuned **multiple regression models** for RUL prediction.  
- Applied **Bagging & Stacking ensembles** to improve predictive accuracy.  
- Gained actionable insights into battery degradation patterns.

---

## Future Scope
- Integrate **deep learning models** (e.g., LSTMs) for sequential battery data.  
- Use **real-time sensor data** from EVs for online RUL estimation.  
- Deploy the trained model via a web dashboard for interactive predictions.

---
