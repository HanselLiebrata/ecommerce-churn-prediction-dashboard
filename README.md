### 🎯 Furniture E-commerce Customer Churn Prediction - ML Pipeline

## Project Overview
Build an end-to-end machine learning pipeline to predict customer churn for an online furniture retailer using delivery, assembly, and customer behavior data. The goal is to identify customers at risk of not making future purchases and quantify the business impact of retention strategies.

## Dataset Information
- **Source**: Online Furniture Orders: Delivery & Assembly Dataset (Kaggle)
- **Size**: 1,938 rows, 14 columns
- **Domain**: E-commerce furniture retail with delivery and assembly services
- **Target Variable**: Define churn as customers unlikely to make repeat purchases

## Step-by-Step Process

### 📊 1. Data Exploration & Understanding
File: `notebooks/01_furniture_churn_analysis.ipynb`
- Load the furniture orders dataset and examine structure
- Analyze data types, missing values, and distributions
- Create visualizations for key business metrics:
  - Order value distribution by product category
  - Delivery success rates by delivery window
  - Assembly service adoption and impact on satisfaction
  - Customer rating patterns by delivery status
- Identify potential churn indicators from delivery failures, low ratings, high costs
- Generate business insights about customer behavior patterns

### 🛠️ 2. Data Preprocessing & Feature Engineering
File: `notebooks/01_furniture_churn_analysis.ipynb` (continued)
Supporting Code: `src/data_preprocessing.py` & `src/feature_engineering.py`
- Handle missing values in brand, shipping_cost, assembly_cost, customer_rating
- Define churn target variable via multiple methods
- Create features: `total_delivery_cost`, `price_per_day`, `delivery_complexity_score`, `customer_value_tier`, `delivery_efficiency`
- Encode categoricals and scale numerics (done in model pipeline)
- Save processed data to `data/processed/cleaned_data.csv` & `data/processed/engineered_features.csv`

### 🤖 3. Model Development & Training
File: `notebooks/01_furniture_churn_analysis.ipynb` (continued)
Supporting Code: `src/model_training.py`
- Split data 80/20 with stratification
- Handle class imbalance with SMOTE or class weights
- Train Logistic Regression, Random Forest, XGBoost, Gradient Boosting with CV
- Hyperparameter tuning via GridSearchCV
- Save best model and preprocessors to `models/`

### 📈 4. Model Evaluation & Validation
File: `notebooks/01_furniture_churn_analysis.ipynb` (continued)
Supporting Code: `src/model_evaluation.py`
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- Feature importance and interpretation guidance
- Business impact calculation examples

### 💼 5. Business Insights & Recommendations
File: `notebooks/01_furniture_churn_analysis.ipynb` (final section)
Output: `reports/executive_summary.pdf`
- Segmentation of customer risk groups and recommended actions

### 🔄 6. Data Export for Dashboard
File: `notebooks/02_powerbi_data_preparation.ipynb`
Output: `data/exports/powerbi_data.csv` & `data/exports/summary_metrics.csv`
- Prepare predictions and KPIs for Power BI

## Quickstart
1) Place raw CSV at `data/raw/furniture_orders_dataset.csv`
2) Install dependencies: `pip install -r requirements.txt`
3) Run pipeline: `python scripts/run_pipeline.py`
4) Exports for Power BI appear under `data/exports/`
