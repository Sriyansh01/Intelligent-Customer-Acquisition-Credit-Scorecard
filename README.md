# 💳 Intelligent Customer Acquisition Credit Scorecard

An **Explainable AI-powered credit risk and customer acquisition system** that uses machine learning to estimate customer credit risk, support lending decisions, and identify suitable acquisition opportunities.

The project combines **credit-risk modeling, class-imbalance handling, gradient-boosting models, model comparison, and an interactive application** to transform customer financial data into actionable credit insights.

---

## 🚀 Project Overview

Financial institutions need to assess the creditworthiness of customers before approving loans or extending credit.

Traditional credit scoring approaches may rely heavily on fixed rules or manually designed scorecards. This project explores a machine-learning-based approach that can learn patterns from historical credit data and generate risk predictions.

The system is designed to:

* Analyze customer financial and demographic information.
* Predict credit-risk outcomes.
* Handle imbalanced credit datasets.
* Train and compare machine-learning models.
* Generate customer-level risk predictions.
* Support intelligent customer acquisition and lending decisions.
* Provide an application interface for interacting with the trained models.
* Explore model behavior and performance through research notebooks and visualizations.

---

## 🎯 Objectives

The main objectives of the project are:

1. Build machine-learning models for credit-risk prediction.
2. Handle class imbalance using **ADASYN oversampling**.
3. Compare different gradient-boosting approaches.
4. Engineer and select relevant credit-risk features.
5. Evaluate model performance using appropriate classification metrics.
6. Save trained models for application use.
7. Integrate predictions into an interactive application.
8. Support explainable, data-driven customer acquisition decisions.

---

## 🧠 Machine Learning Approach

The project explores multiple machine-learning approaches, including:

* **LightGBM**
* **XGBoost**
* Logistic Regression / baseline approaches where applicable
* ADASYN for class balancing

The main production-oriented training scripts use **LightGBM**, while XGBoost is also trained as a baseline for comparison in the bank-credit workflow.

### Why Gradient Boosting?

Gradient-boosting models are well suited for structured/tabular financial datasets because they can capture nonlinear relationships and interactions between customer attributes.

---

## ⚖️ Handling Class Imbalance

Credit-risk datasets are often highly imbalanced, with significantly fewer high-risk/default cases than normal cases.

To address this, the project uses:

### ADASYN — Adaptive Synthetic Sampling

ADASYN generates synthetic minority-class observations to create a more balanced training dataset.

The workflow is:

```text
Original Dataset
       ↓
Identify Minority Class
       ↓
ADASYN Oversampling
       ↓
Balanced Training Data
       ↓
Machine Learning Model
```

This helps the model learn patterns associated with higher-risk customers instead of being dominated by the majority class.

The training scripts explicitly apply ADASYN before model training.

---

# 📊 Datasets

The project works with credit-risk datasets including:

### 1. German Credit Dataset

The project contains a cleaned German Credit dataset used by the application and training workflow.

The final application-oriented model uses a selected set of features including:

* Checking Status
* Duration
* Credit History
* Credit Amount
* Savings Status
* Age

These features are used to generate credit-risk predictions.

### 2. Home Credit / Bank Credit Dataset

The repository also contains a bank-credit modeling workflow based on the **Home Credit Default Risk** dataset.

The full dataset is too large for GitHub and is therefore hosted externally.

Dataset:

https://www.kaggle.com/competitions/home-credit-default-risk/data

The repository contains smaller/sample data for development and demonstration.

---

# 🔄 Project Workflow

```text
                 CREDIT DATA
                      │
                      ▼
               Data Collection
                      │
                      ▼
               Data Cleaning
                      │
                      ▼
             Exploratory Analysis
                      │
                      ▼
             Feature Engineering
                      │
                      ▼
             Class Imbalance Check
                      │
                      ▼
                 ADASYN
                      │
                      ▼
            Model Training
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
          LightGBM          XGBoost
             │                 │
             └────────┬────────┘
                      ▼
             Model Evaluation
                      │
                      ▼
              Saved ML Models
                      │
                      ▼
             Credit Risk Score
                      │
                      ▼
          Customer Acquisition /
             Lending Insights
                      │
                      ▼
             Interactive App
```

---

# 🏦 Business Problem

A financial institution needs to answer questions such as:

* Is this customer likely to default?
* Should credit be offered to this customer?
* What level of risk does the customer represent?
* Which customers should be targeted for acquisition?
* Which customers require additional verification?
* How can the bank reduce potential credit losses?

This project uses machine learning to provide a **data-driven risk assessment layer** that can support these decisions.

---

# 💡 Intelligent Customer Acquisition

The system can be used as part of a customer acquisition strategy.

Instead of targeting every potential customer equally, financial institutions can prioritize customers based on predicted credit risk.

Conceptually:

```text
Potential Customers
        ↓
Credit Risk Prediction
        ↓
 ┌──────┼──────────┐
 │      │          │
 ▼      ▼          ▼
Low    Medium      High
Risk    Risk       Risk
 │       │          │
 ▼       ▼          ▼
Target   Review    Restrict /
More     Further   Additional
        Verification
```

This can help organizations balance:

**Customer Growth + Credit Risk Management**

---

# 🤖 Models

## LightGBM

LightGBM is used as the primary gradient-boosting model in the application-oriented training workflow.

The model is trained using selected credit-risk features and balanced training data.

---

## XGBoost

XGBoost is used as a comparison/baseline model in the bank-credit workflow.

The project trains both LightGBM and XGBoost on the processed dataset and saves the resulting models for further evaluation.

---

# 🔬 Model Research & Comparison

The repository contains dedicated research scripts and notebooks for experimenting with:

* Model performance
* Confusion matrices
* German Credit modeling
* Bank-credit modeling
* LightGBM
* XGBoost
* Feature selection
* Class balancing
* Model visualization

This separates **research experimentation** from the final application-oriented training pipeline.

---

# 📈 Model Evaluation

The project evaluates models using classification-oriented metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC where applicable

For credit-risk problems, **recall and false-negative analysis are particularly important**, because incorrectly classifying a high-risk customer as low-risk can lead to financial losses.

---

# 🖥️ Application

The repository contains an `app/` directory containing the application layer.

The application is designed to use the trained credit-risk models and selected customer features to generate predictions.

The model-training workflow saves trained models into the `models/` directory so that they can be loaded by the application rather than retraining the model every time.

---

# 📁 Project Structure

```text
Intelligent-Customer-Acquisition-Credit-Scorecard/
│
├── app/
│   └── Application files
│
├── data/
│   └── Sample / processed datasets
│
├── models/
│   └── Saved trained models
│
├── notebooks/
│   └── Development and experimentation notebooks
│
├── bank_research_results.png
│
├── env.py
├── import_data.py
│
├── research.py
├── research_confusion.py
├── research_german.py
│
├── train_final.py
├── train_final_bank.py
├── train_german.py
├── train_german_research.py
│
├── plot_lgbm_tree.py
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

The repository currently contains separate folders for the application, data, models, and notebooks, along with dedicated training and research scripts.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Sriyansh01/Intelligent-Customer-Acquisition-Credit-Scorecard.git
```

Navigate into the project:

```bash
cd Intelligent-Customer-Acquisition-Credit-Scorecard
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Setup

The full Home Credit dataset is too large to store directly in this repository.

Download it from:

https://www.kaggle.com/competitions/home-credit-default-risk/data

Place the required files in the appropriate `data/` or project location expected by the training scripts.

A smaller sample/processed dataset is included in the repository for demonstration and development.

---

# ▶️ Running the Project

## German Credit Model

Run:

```bash
python train_german.py
```

This trains the German Credit model using the selected application features and ADASYN balancing.

The trained model is saved in the `models/` directory.

---

## Final Application Model

Run:

```bash
python train_final.py
```

This workflow:

1. Loads the cleaned German Credit data.
2. Selects the features used by the application.
3. Converts categorical features.
4. Applies ADASYN.
5. Trains the LightGBM model.
6. Saves the trained model.

The current script saves the model as:

```text
models/xgboost_model.pkl
```

Note: the filename reflects the existing repository implementation; the training model in this script is LightGBM.

---

## Bank Credit Model

Run:

```bash
python train_final_bank.py
```

This workflow:

1. Loads the bank-credit dataset.
2. Handles missing values using median imputation.
3. Separates the target variable.
4. Applies ADASYN.
5. Splits the data into training and testing sets.
6. Trains LightGBM.
7. Trains XGBoost as a baseline.
8. Saves both trained models.

---

# 📊 Model Files

The trained models are stored inside:

```text
models/
```

Examples include:

```text
kaggle_model.pkl
xgb_model.pkl
xgboost_model.pkl
```

These serialized models can be loaded by the application for prediction without retraining.

---

# 🔍 Explainability

Explainability is an important consideration for credit-risk systems.

Financial decisions can have significant consequences, so understanding the reasoning behind model predictions is valuable.

Future iterations of this project can extend the current modeling workflow with:

* SHAP feature importance
* Individual prediction explanations
* Feature contribution plots
* Risk reason codes
* Global model interpretation

This would allow the system to answer not only:

> **"What is the customer's predicted risk?"**

but also:

> **"Which customer characteristics contributed to that risk?"**

---

# 📌 Key Features

### Machine Learning

* LightGBM
* XGBoost
* Classification modeling
* Probability-based risk prediction

### Data Processing

* Pandas
* NumPy
* Missing-value handling
* Feature selection
* Categorical encoding

### Imbalanced Learning

* ADASYN oversampling

### Model Development

* Train/test split
* Model comparison
* Confusion matrix analysis
* Research notebooks

### Application

* Saved model artifacts
* Application-oriented feature selection
* Interactive credit-risk prediction workflow

---

# 📈 Business Impact

A production-ready version of this system could help financial institutions:

* Reduce potential credit losses.
* Improve customer screening.
* Prioritize safer acquisition opportunities.
* Identify potentially high-risk applicants.
* Improve risk-based decision making.
* Automate parts of the initial credit assessment.
* Support more targeted customer acquisition campaigns.

The system is intended as a **decision-support tool**, rather than a replacement for regulatory, financial, or human review processes.

---

# 🔮 Future Improvements

Potential improvements include:

* Hyperparameter tuning for LightGBM and XGBoost.
* Cross-validation for more reliable model evaluation.
* Probability calibration.
* Cost-sensitive threshold optimization.
* SHAP-based explainability in the application.
* Interactive risk-score visualization.
* Automated model monitoring.
* Model drift detection.
* Fairness and bias analysis.
* REST API deployment using FastAPI or Flask.
* Cloud deployment.
* Automated data pipelines.
* Integration with a production database.
* Automated retraining.

---

# 🧪 Research Direction

The project can be extended into a more comprehensive **Explainable Credit Risk Scorecard** by combining:

```text
Credit Risk Prediction
        +
Model Explainability
        +
Risk Segmentation
        +
Customer Acquisition
        +
Business Decision Support
```

This creates a complete machine-learning workflow from **raw financial data to actionable customer-level credit insights**.

---

# 🛠️ Tech Stack

| Category            | Technologies                    |
| ------------------- | ------------------------------- |
| Language            | Python                          |
| Data Processing     | Pandas, NumPy                   |
| Machine Learning    | LightGBM, XGBoost, Scikit-learn |
| Imbalanced Learning | imbalanced-learn / ADASYN       |
| Visualization       | Matplotlib, Seaborn             |
| Model Persistence   | Joblib                          |
| Development         | Jupyter Notebook, VS Code       |
| Application         | Python-based application        |

---

# 👨‍💻 Author

**Sriyansh Mishra**
**Saharsh Srivastava**
**N Prashant Deo**

GitHub:
https://github.com/Sriyansh01

---

## ⭐ Project Highlights

**Machine Learning + Credit Risk + Explainable AI + Customer Acquisition**

This project demonstrates an end-to-end approach to using machine learning for financial risk assessment, from data preprocessing and class balancing to model training, evaluation, model persistence, and application integration.

---

⭐ If you found this project useful, consider giving the repository a star.
