# Credit Card Fraud Detection Using Anomaly Detection

## 📗 Project Overview

This project implements a **Credit Card Fraud Detection** system using **Anomaly Detection** techniques. It utilizes machine learning models like **Isolation Forest** and **One-Class SVM** to identify fraudulent credit card transactions from a dataset. The project covers the entire machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and real-time predictions.

### Key Features:
- **Data Preprocessing**: Upload and preprocess credit card transaction data.
- **Exploratory Data Analysis**: Perform visual analysis of the dataset using interactive charts and metrics.
- **Model Training**: Train fraud detection models (Isolation Forest, One-Class SVM) using the processed data.
- **Model Evaluation**: Evaluate model performance using confusion matrix, classification report, and anomaly score distribution.
- **Real-time Prediction**: Predict whether a new credit card transaction is fraudulent or not in real time.

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** for the web application interface
- **Scikit-learn** for machine learning models
- **Pandas** for data manipulation
- **NumPy** for numerical operations
- **Matplotlib** and **Seaborn** for data visualization
- **Plotly** for interactive visualizations
- **Joblib** for saving and loading trained models

---

## 📊 Dataset

The dataset used in this project is based on credit card transactions and contains features like:

- **Time**: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: 28 anonymized features resulting from a PCA transformation.
- **Amount**: The transaction amount.
- **Class**: 1 for fraud and 0 for a normal transaction.

The dataset is available for download from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

---

## ⚙️ How to Run the Project Locally

### Clone the Repository:
First, clone the repository to your local machine using the following command:
```bash
git clone https://github.com/prathamesh693/01_Credit-Card-Fraud-Detection-Using-Anomaly-Detection.git
