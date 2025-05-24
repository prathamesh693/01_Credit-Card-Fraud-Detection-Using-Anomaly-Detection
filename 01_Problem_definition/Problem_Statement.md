# Problem Statement: Credit Card Fraud Detection
Credit card fraud is a significant issue in the financial industry, resulting in billions of dollars in losses annually. The key challenge in fraud detection is identifying fraudulent transactions from an extremely imbalanced dataset, where fraudulent cases represent a tiny fraction of all transactions.

The primary objective of this project is to build a robust machine learning system that can accurately identify potentially fraudulent credit card transactions in real-time, minimizing false positives while ensuring that actual frauds are detected quickly.

The dataset used in this project contains anonymized credit card transaction data, including a binary classification label indicating whether a transaction is fraudulent.

## Objectives

- Analyze the distribution and characteristics of fraudulent vs. legitimate transactions.
- Develop and compare multiple anomaly detection models:
  - Isolation Forest
  - One-Class SVM
- Evaluate the models using appropriate metrics (Precision, Recall, F1 Score, AUC-ROC).
- Deploy the best-performing model through a web application interface for real-time detection.

## Challenges

- Highly imbalanced dataset
- Lack of feature interpretability due to anonymization
- Need for real-time or near real-time performance
- Trade-off between sensitivity and specificity
