# TelcoX Communications: Customer Churn Analysis and Insights
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

*An in-depth case study on reducing customer churn through data-driven insights and predictive modeling.*

## Table of Contents

- [Introduction](#introduction)
- [Business Challenge](#business-challenge)
- [Solution Overview](#solution-overview)
- [Key Findings](#key-findings)
- [Interactive Dashboard](#interactive-dashboard)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)
- [Disclaimer](#disclaimer)
- [Project Structure](#project-structure)

## Introduction

TelcoX Communications is a leading telecommunications provider facing challenges with customer churn, which impacts revenue and growth. Retaining existing customers is crucial, as acquiring new ones is often more costly.

This case study leverages data analytics and machine learning to:

- **Identify patterns and segments** within the customer base.
- **Predict customer churn** using advanced predictive modeling.
- **Provide actionable insights** to reduce churn rates and enhance customer satisfaction.

## Business Challenge

**Objective**: Reduce customer churn by identifying at-risk customers and understanding the factors contributing to their decision to leave.

**Key Questions**:

1. Which customers are most likely to churn?
2. What are the common characteristics of customers who churn?
3. How can TelcoX tailor its services and marketing strategies to retain customers?

## Solution Overview

To address the challenge, the following steps were undertaken:

1. **Data Analysis**: Explored customer demographics, account information, service usage, and churn status to uncover patterns.
2. **Customer Segmentation**: Applied clustering techniques to segment customers into distinct groups based on similarities.
3. **Churn Prediction**: Built a predictive model to estimate the likelihood of each customer churning.
4. **Interactive Dashboard**: Developed a user-friendly dashboard for stakeholders to visualize insights and make informed decisions.

## Key Findings

- **High-Risk Customers**: Identified customers with a high probability of churning, allowing for targeted retention efforts.
- **Significant Factors Influencing Churn**:
  - **Contract Type**: Month-to-month contracts have higher churn rates.
  - **Payment Method**: Customers using electronic checks are more likely to churn.
  - **Tenure**: Shorter tenure customers are at higher risk.
- **Customer Segments**:
  - **Price-Sensitive Customers**: Prefer flexible plans but are more likely to churn.
  - **Loyal Customers**: Long-term customers with lower churn risk.
  - **Service-Oriented Customers**: Subscribe to multiple services but may churn if not satisfied.

## Interactive Dashboard

Explore the detailed analysis and insights through our interactive dashboard:

### **[Access the TelcoX Customer Churn Dashboard](https://churnprediction-4cl6vyj5yyguu5w76xzuoe.streamlit.app)**

**Dashboard Features**:

- **Cluster Overview**: Summarizes customer segments with key metrics.
- **Demographics**: Analyzes customer demographics across clusters.
- **Services**: Examines service subscriptions and preferences.
- **Contract & Payment**: Reviews contract types and payment methods.
- **Churn Analysis**: Provides insights into churn patterns and risk factors.
- **Customer Profiles**: Allows exploration of individual customer data.
- **Recommendations**: Offers marketing strategies tailored to each cluster.
- **Churn Prediction**: Displays churn probabilities and identifies high-risk customers.

## Recommendations

Based on the analysis, the following strategies are recommended for TelcoX Communications:

### **For High-Risk Customers**

- **Personalized Retention Offers**: Provide incentives to encourage contract renewals or upgrades.
- **Customer Engagement**: Proactively reach out to gather feedback and address concerns.
- **Flexible Contract Options**: Offer customizable plans to meet individual needs.

### **Cluster-Specific Strategies**

- **Price-Sensitive Customers**:
  - Offer competitive pricing and highlight cost savings.
  - Introduce bundled packages with discounts.

- **Loyal Customers**:
  - Implement loyalty programs with exclusive benefits.
  - Upsell additional services that enhance their experience.

- **Service-Oriented Customers**:
  - Ensure high-quality service and support.
  - Provide early access to new features or services.

## Conclusion

By leveraging data analytics and predictive modeling, TelcoX Communications can proactively address customer churn. The insights from clustering and churn prediction models enable targeted interventions, leading to improved customer retention and increased revenue.

The interactive dashboard serves as a valuable tool for stakeholders to explore data, understand customer segments, and make informed decisions.

## Disclaimer

This is a simulated company case study for educational purposes. TelcoX Communications is a fictional entity, and the dataset used is publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).


## Project Structure

Below is the organization of this repository:

```{bash}
.
├── data/
│   ├── data.csv                         # Original dataset for analysis
│   └── data_cluster.csv                 # Dataset with clustering results
├── models/
│   ├── classification_pipeline.joblib   # Classification pipeline model
│   ├── clustering_pipeline.joblib       # Clustering pipeline model
│   └── target_transformer.joblib        # Target variable transformer for model preprocessing
├── notebooks/
│   ├── nathan.ipynb                     # Analysis and work by Nathan
│   ├── nathan_full_pipeline.ipynb       # Full pipeline implementation by Nathan
│   ├── nathan_model_optimization.ipynb  # Model optimization notebook by Nathan
│   ├── pierre.ipynb                     # Analysis and work by Pierre
│   ├── sigurd.ipynb                     # Analysis and work by Sigurd
│   └── theo.ipynb                       # Analysis and work by Theo
├── scripts/
│   ├── predict.py                       # Script for making predictions with trained models
│   └── train_models.py                  # Script for training and saving models
└── utils/
    ├── classification_pipeline.py       # Code for the classification pipeline
    ├── clustering_pipeline.py           # Code for the clustering pipeline
    ├── custom_transformers.py           # Custom transformation functions
    ├── pipeline.py                      # End-to-end pipeline configuration
    └── preprocessing_pipeline.py        # Preprocessing pipeline implementation
├── .gitignore
├── CUSTOMER_SEGMENTS.md                 # Documentation on customer segmentation results
├── MIT-LICENSE.txt
├── README.md
├── TSE_Project_guidelines_M2DSSS.pdf
├── dashboard_streamlit.py               # Streamlit dashboard for interactive data visualization
├── requirements.txt
```


---

*Developed as a case study for TelcoX Communications by TSE students.*
