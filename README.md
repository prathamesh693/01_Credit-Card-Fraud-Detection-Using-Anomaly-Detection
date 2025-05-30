# 💳 Credit Card Fraud Detection Using Anomaly Detection  
**⚠️ Identify fraudulent transactions in real-time using ML**

This project aims to detect fraudulent transactions in credit card usage using machine learning techniques. Fraud detection is a classic example of anomaly detection and is crucial for minimizing financial losses and ensuring the security of financial systems.

---

## 📚 Table of Contents  
- [Problem Statement](#problem-statement)  
- [Objectives](#objectives)  
- [Challenges](#challenges)  
- [Project Lifecycle](#project-lifecycle)  
- [Tools and Technologies](#tools-and-technologies)  
- [Success Criteria](#success-criteria)  
- [How to Run the Project Locally](#how-to-run-the-project-locally)  
- [References](#references)  
- [Connect With Me](#connect-with-me)

---

## 📌 Problem Statement  
Credit card fraud is a major concern in the financial industry, with billions of dollars lost annually. The key challenge in fraud detection is to identify fraudulent transactions from highly imbalanced datasets where fraud represents a tiny fraction of all records.

The goal is to build an anomaly detection system using unsupervised learning techniques that can accurately identify fraudulent transactions while minimizing false positives, suitable for real-time deployment.

---

## 🎯 Objectives  
- Analyze characteristics of fraudulent vs. legitimate transactions  
- Build models using:  
  - Isolation Forest  
  - One-Class SVM  
- Evaluate with precision, recall, F1-score, AUC-ROC  
- Deploy a real-time detection interface using Streamlit

---

## ⚠️ Challenges  
- Extreme class imbalance  
- Anonymized dataset features (less interpretability)  
- Need for real-time inference  
- Managing false positives vs. detection rate trade-off

---

## 🛠️ Project Lifecycle  

1. **Problem Definition**  
   - Define use case and success criteria  
2. **Data Acquisition & Understanding**  
   - Use public Kaggle dataset on credit card transactions  
3. **Exploratory Data Analysis (EDA)**  
   - Analyze transaction patterns, detect outliers  
4. **Modeling**  
   - Apply Isolation Forest and One-Class SVM  
5. **Evaluation**  
   - Use precision, recall, F1, ROC-AUC for comparison  
6. **Deployment**  
   - Deploy best model using a Streamlit web app  
7. **Monitoring**  
   - Prepare retraining and drift detection pipeline  

---

## 💻 Tools and Technologies  

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-44A8B3?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Joblib-008000?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

---

## ✔️ Success Criteria  
- F1-score > 0.85 on test data  
- Real-time prediction latency < 1 second  
- Streamlit interface for live testing  
- Monitoring and retraining ready for production scaling  

---

## 🔗 References  
The dataset is available for download from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

---

## 🤝 Connect With Me  
**[![LinkedIn](https://img.shields.io/badge/LinkedIn-Prathamesh%20Jadhav-blue?logo=linkedin)](https://www.linkedin.com/in/prathamesh-jadhav-78b02523a/)** **[![GitHub](https://img.shields.io/badge/GitHub-Prathamesh%20Jadhav-2b3137?logo=github)](https://github.com/prathamesh693)**