# Project Overview: Credit Card Fraud Detection

## Overview

This project aims to detect fraudulent transactions in credit card usage using machine learning techniques. Fraud detection is a classic example of anomaly detection and is crucial for minimizing financial losses and ensuring the security of financial systems.

## Project Lifecycle

1. **Problem Definition**
   - Define goals, objectives

2. **Data Acquisition & Understanding**
   - Source publicly available credit card transaction data
   - Understand data characteristics and distributions

3. **Exploratory Data Analysis (EDA)**
   - Analyze trends, patterns, and anomalies
   - Understand class imbalance

4. **Modeling**
   - Apply unsupervised and semi-supervised algorithms such as Isolation Forest and One-Class SVM

5. **Evaluation**
   - Compare model performance using metrics suitable for imbalanced classification

6. **Deployment**
   - Deploy the best model using a Streamlit web app

7. **Monitoring**
   - Set up systems for performance tracking and model drift

## Tools & Technologies

- **Languages**: Python
- **Libraries**: scikit-learn, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn
- **Deployment**: Streamlit

## Success Criteria

- Achieve F1-score > 0.85 on test set
- Real-time prediction latency under 1 second
- Model retraining and monitoring pipeline in place
