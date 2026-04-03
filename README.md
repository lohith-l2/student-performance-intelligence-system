# 🎓 Student Performance Intelligence System

A machine learning-powered web application that predicts student performance, identifies at-risk students, and provides actionable insights through an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project simulates a real-world student performance analysis system used by educational institutions to:
- Predict whether a student will **Pass or Fail**
- Identify **at-risk students** who need early intervention
- Analyze performance trends through **interactive visualizations**
- Allow educators to **add new student records** manually or via CSV upload

---

## 🖥️ Live Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | KPI metrics and project overview |
| 📊 Data Explorer | Filter, search, and download student data |
| 📈 EDA & Insights | 6 analytical charts with key findings |
| 🤖 Model Performance | Accuracy, ROC-AUC, confusion matrix |
| 🔮 Predict Student | Real-time prediction form |
| ⚠️ At-Risk Students | Configurable risk detection dashboard |
| ➕ Add New Student | Manual entry or CSV bulk upload |

---

## 🧠 Machine Learning

| Item | Detail |
|------|--------|
| Algorithm | Random Forest Classifier |
| Accuracy | 90.00% |
| ROC-AUC | 0.9176 |
| Cross-Validation | 97.50% ± 3.06% |
| Features Used | 8 (attendance, study hours, grades, etc.) |
| Target | Pass / Fail (binary classification) |

### Features Used
- `attendance_pct` — percentage of classes attended
- `study_hours_per_day` — daily study hours
- `prev_grade` — previous semester grade
- `assignments_submitted` — out of 10
- `parent_education` — ordinal encoded (None → Postgraduate)
- `internet_access` — binary encoded
- `gender` — binary encoded
- `age` — numeric

---

## 📁 Project Structure
---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-intelligence-system.git
cd student-performance-intelligence-system
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Dataset
```bash
python3 data/generate_data.py
```

### 5. Train the Model
```bash
python3 train.py
```

### 6. Launch the Dashboard
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Key Insights from EDA

- Students with **attendance > 75%** are 3x more likely to pass
- Passing students study an average of **2 more hours per day**
- **Previous grade** is the strongest predictor of final performance
- Students with **no internet access** have a 15% lower pass rate
- Higher **parent education level** correlates with better outcomes

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | Machine learning |
| matplotlib / seaborn | Visualizations |
| streamlit | Web dashboard |
| joblib | Model serialization |

---

## 👤 Author

**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 📄 License

This project is licensed under the MIT License.
