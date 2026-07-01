# 📈 DrLoE: Employee Attrition Prediction App

DrLoE (Life of an Employee) is a modern, premium-grade web application built to predict employee attrition using machine learning. Driven by a Random Forest Classifier, it provides organizations with actionable insights to anticipate talent churn, understand burnout factors, and optimize workforce retention strategies.

---

## ✨ Features

- **🎯 Individual Risk Profiling**:
  - Predicts whether a specific employee is likely to stay or leave the organization.
  - Generates a **Churn Risk Score (%)** using model probability outputs.
  - Dynamically displays **💡 Retention Insights** based on employee metrics (e.g., workload warning, low satisfaction, compensation gaps).

- **📂 Bulk Processing & Analytics Dashboard**:
  - Upload a batch CSV file to analyze retention risk across multiple teams.
  - Displays high-level workforce metrics (Total Evaluated, Predicted Attrition, General Attrition Rate, and Average Risk Score).
  - Includes interactive visualizations:
    - **Attrition Forecast Distribution** (Pie Chart)
    - **Attrition Rate by Department** (Seaborn Bar Plot)
  - Exports annotated predictions as a downloadable CSV.

- **🎨 Premium User Experience**:
  - Styled with the modern **Outfit** typography.
  - Glassmorphic metric cards, subtle gradients, and reactive layout design.

---

## 🛠️ Tech Stack & Dependencies

- **Core Application**: [Streamlit](https://streamlit.io/) (Interactive Web UI)
- **Data Engineering & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `Pillow`
- **Machine Learning**: `scikit-learn` (v1.1.3), `category-encoders`, `joblib`

---

## 📂 Directory Structure

```
├── app.py                     # Streamlit web application source code
├── requirements.txt           # Python package dependencies
├── clean_ohe.pkl              # Pre-trained Category Encoder (One-Hot Encoder)
├── emp_rf.pkl                 # Pre-trained Random Forest Classifier model
├── sample.png                 # Sample image showcasing correct CSV schema
├── existing_employee.csv      # Reference dataset for active employees
└── employee_who_left.csv      # Reference dataset for departed employees
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Snehamakharia/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!NOTE]
> `scikit-learn==1.1.3` is required to deserialize the pre-trained model (`emp_rf.pkl`) successfully.

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📊 Machine Learning Pipeline

1. **Data Pre-processing**:
   - Merged the datasets of active employees and departed employees.
   - Target variable (`Churn`): `1` for employees who left, `0` for existing employees.
   - Categorical variables (`dept`, `salary`) encoded using a cached Category Encoder.

2. **Model Selection**:
   - Compared multiple models (SVM, Decision Trees, KNN, and Random Forest).
   - Chosen model: **Random Forest Classifier** due to its superior capability in handling non-linear decision boundaries and imbalanced classes.
   - Primary evaluation metric: **Cohen's Kappa Statistics** (giving robust weight to prediction accuracy on minority classes).

---

## 📋 Batch Schema Format

When uploading a CSV for batch predictions, ensure the following columns are present:

| Column Name | Type | Description |
|---|---|---|
| `Satisfaction_level` | Float | Job satisfaction score between `0.0` and `1.0` |
| `Last_evaluation` | Float | Last evaluation score between `0.0` and `1.0` |
| `number_project` | Integer | Number of active projects assigned |
| `average_montly_hours` | Float | Average hours worked per month |
| `time_spend_company` | Integer | Number of years spent in the organization |
| `Work_accident` | Integer | `0` = No work accident, `1` = Experienced work accident |
| `promotion_last_5years` | Integer | `0` = No promotion, `1` = Promoted in last 5 years |
| `dept` | String | Department (e.g., `sales`, `technical`, `support`, `IT`, etc.) |
| `salary` | String | Salary tier (`low`, `medium`, `high`) |
