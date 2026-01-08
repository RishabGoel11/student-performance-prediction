# Student Performance Prediction (Pass/Fail)

## 🌐 Live Demo

- Frontend (Lovable): https://student-ml-predictor.lovable.app
- Backend API (Render): https://student-performance-prediction-kva6.onrender.com

## 🏗 Architecture

Lovable Frontend → Flask API → Machine Learning Model (Logistic Regression)

This project is deployed as an API-only backend with a separate frontend interface.


## 📌 Problem Statement
The goal of this project is to predict whether a student will **pass or fail** based on academic, demographic, and socio-economic factors.  
This is formulated as a **binary classification problem**.

---

## 📊 Dataset
The dataset was created by **merging two datasets**:
- Mathematics student performance
- Portuguese student performance

Merging was done to:
- Increase sample size
- Improve generalization

### Target Variable
- `pass = 1` → Final grade (G3) ≥ 10  
- `pass = 0` → Final grade (G3) < 10

---

## 🛠️ Approach

### 1. Data Preprocessing
- Merged both datasets
- Handled categorical variables using **one-hot encoding**
- Prevented **data leakage** by removing grade-related predictors (`G1`, `G2`, `G3`)

### 2. Exploratory Data Analysis (EDA)
- Class distribution (Pass vs Fail)
- Scatter plots for:
  - Study time vs Absences
  - Absences vs Pass/Fail
- Analysis of attendance impact on performance

### 3. Model Building
- Baseline Logistic Regression
- Hyperparameter tuning using **GridSearchCV**
- Cross-validation with F1-score optimization

### 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve

---

## 📈 Results
- Logistic Regression achieved good generalization
- Tuned model performed better than baseline
- Attendance (`absences`) and study-related features were strong predictors

---

## 📊 Visualizations
- Class distribution plot
- Confusion matrix
- ROC curve
- Feature importance using model coefficients

---

## Model Selection
The final deployed model is a leakage-free Logistic Regression pipeline trained only on pre-exam numeric features.
Early exam grades (G1, G2) were intentionally excluded to ensure realistic and deployable predictions.

A scikit-learn Pipeline was used to guarantee consistency between training and inference.

## 💾 Model Persistence
The trained and tuned model was saved using `pickle` for reuse and deployment.

---

## 🚀 Future Improvements
- Try tree-based models (Random Forest, XGBoost)
- Handle outliers in absences
- Deploy as a web application using Flask
- Add real-time prediction UI

---

## 🧠 Key Learnings
- Importance of data leakage prevention
- Effect of class imbalance on model performance
- Role of hyperparameter tuning in improving generalization

---

## 🧪 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
