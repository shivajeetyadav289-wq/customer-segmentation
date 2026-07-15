# Customer Segmentation using Machine Learning

## Overview

Customer segmentation is one of the most important applications of machine learning in marketing and customer relationship management. This project builds an end-to-end machine learning pipeline that segments customers based on their purchasing behavior, demographics, and shopping patterns.

The project follows a modular architecture covering the complete machine learning lifecycle—from data ingestion and validation to model training, evaluation, cluster analysis, prediction, and deployment through a Streamlit web application.

---

# Objectives

* Build an end-to-end customer segmentation pipeline.
* Perform comprehensive Exploratory Data Analysis (EDA).
* Engineer meaningful customer features.
* Train and evaluate clustering models.
* Identify meaningful customer segments.
* Predict the segment of new customers.
* Provide business recommendations for each customer segment.
* Deploy the project using Streamlit.

---

# Project Workflow

```text
Customer Segmentation Pipeline

MongoDB
    │
    ▼
Data Ingestion
    │
    ▼
Data Validation
    │
    ▼
Exploratory Data Analysis (EDA)
    │
    ▼
Data Transformation
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Cluster Analysis
    │
    ▼
Prediction Pipeline
    │
    ▼
Streamlit Web Application
```

---

# Project Structure

```text
customer_segmentation/

│── artifacts/
│
├── src/
│   ├── components/
│   │      data_ingestion.py
│   │      data_validation.py
│   │      data_transformation.py
│   │      model_trainer.py
│   │      model_evaluation.py
│   │      cluster_analysis.py
│   │
│   ├── pipeline/
│   │      prediction_pipeline.py
│   │
│   ├── entity/
│   ├── constant/
│   ├── data_access/
│   └── __init__.py
│
├── app.py
├── main.py
├── EDA.ipynb
├── requirements.txt
└── README.md
```

---

# Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### Database

* MongoDB

### Model Serialization

* Joblib

### Web Application

* Streamlit

---

# Dataset

The project uses the **Customer Personality Analysis** dataset containing customer demographic information, purchasing behavior, campaign responses, and spending patterns.

The original dataset contains customer information such as:

* Income
* Education
* Marital Status
* Product spending
* Web purchases
* Store purchases
* Catalog purchases
* Campaign acceptance
* Customer recency
* Website visits

---

# Exploratory Data Analysis (EDA)

A detailed Exploratory Data Analysis (EDA) was performed before model building to understand the dataset and identify the most informative features.

## EDA Tasks Performed

* Dataset overview
* Data type inspection
* Missing value analysis
* Duplicate record detection
* Statistical summary
* Distribution analysis
* Outlier analysis
* Correlation analysis
* Feature engineering
* Variance Inflation Factor (VIF) analysis
* Feature selection

---

## Feature Engineering

The following features were created during EDA:

* Age
* Children
* Family Size
* Days as Customer
* Total Spending
* Total Promotions
* Offers Responded To
* Parental Status

---

## Feature Selection

After correlation analysis and VIF analysis, redundant variables were removed.

The final model was trained using the following features:

* Age
* Income
* Days_as_Customer
* Recency
* Wines
* Fruits
* Meat
* Fish
* Sweets
* Gold
* Web
* Catalog
* Store
* Discount Purchases
* NumWebVisitsMonth

This reduced multicollinearity and improved clustering performance.

---

## Visualizations Performed

The following visualizations were created during EDA:

* Histograms
* Boxplots
* Countplots
* Correlation Heatmap
* Feature Distribution Plots
* Customer Spending Analysis
* Purchase Behaviour Analysis
* VIF Analysis

---

# Data Ingestion

The data ingestion module performs:

* Reading customer data from MongoDB
* Exporting raw data
* Train-test split
* Saving datasets into the artifacts directory

Generated files:

* raw.csv
* train.csv
* test.csv

---

# Data Validation

The validation module checks:

* Missing files
* Missing columns
* Missing values
* Duplicate records

A YAML validation report is automatically generated.

Generated file:

* report.yaml

---

# Data Transformation

The transformation module performs:

## Feature Engineering

* Customer age calculation
* Customer tenure calculation
* Feature selection
* Data preprocessing

## Data Preprocessing

* Missing value imputation
* Standard scaling

Generated artifacts:

* preprocessor.pkl
* transformed_train.npy
* transformed_test.npy

---

# Model Training

Customer segmentation was performed using **K-Means Clustering**.

The optimal number of clusters was determined using the Silhouette Score.

Generated artifacts:

* kmeans_model.pkl
* clustered_data.csv

---

# Model Evaluation

Three clustering algorithms were evaluated:

* K-Means
* Agglomerative Clustering
* DBSCAN

Evaluation metric:

* Silhouette Score

The best-performing model was selected based on clustering quality.

---

# Cluster Analysis

Cluster analysis was performed to understand customer behavior.

Generated outputs include:

* Cluster summary
* Cluster distribution
* Income analysis
* Spending analysis
* Cluster visualizations

---

# Prediction Pipeline

The prediction pipeline allows new customer records to be classified into an existing customer segment.

Workflow:

* Accept customer information
* Apply preprocessing
* Load trained model
* Predict customer cluster
* Return customer segment
* Generate marketing recommendation

---

# Streamlit Web Application

A Streamlit application was developed to make the trained model interactive.

The application allows users to:

* Enter customer information
* Predict customer segment
* View customer segment
* Receive marketing recommendations

---

# Customer Segments

## Cluster 0 – Budget Customers

Characteristics:

* Lower average income
* Lower spending
* More website visits
* Price-sensitive purchasing behavior

Recommended Strategy:

* Discount coupons
* Promotional campaigns
* Seasonal offers
* Price-based marketing

---

## Cluster 1 – High Value Customers

Characteristics:

* Higher average income
* Higher catalog purchases
* Higher store purchases
* Loyal purchasing behavior

Recommended Strategy:

* Premium product recommendations
* Loyalty rewards
* Exclusive memberships
* Personalized marketing campaigns

---

# Model Performance

| Metric                  | Value      |
| ----------------------- | ---------- |
| Algorithm               | K-Means    |
| Best Number of Clusters | 2          |
| Silhouette Score        | **0.3132** |

---

# Future Improvements

* Hyperparameter optimization
* Additional clustering algorithms
* Interactive dashboards
* REST API deployment
* Cloud deployment
* Automated model retraining
* CI/CD pipeline integration

---

# Author

**Shivajeet Yadav**

**M.Sc. Bioinformatics**

**Skills**

* Machine Learning
* Data Analytics
* Python
* MongoDB
* Data Visualization
* Bioinformatics

---

# License

This project is intended for educational, learning, and portfolio purposes.

