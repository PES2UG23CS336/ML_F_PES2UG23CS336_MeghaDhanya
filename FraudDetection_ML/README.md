# Detecting Payment Fraud Using Machine Learning

This project aims to detect fraudulent credit card transactions using various machine learning models. It handles the problem of class imbalance with SMOTE oversampling and compares models such as Logistic Regression, Random Forest, SVM, ANN etc.

## Dataset
The dataset used is publicly available on Kaggle:  
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Models Implemented
* Logistic Regression
* K-Nearest Neighbors  
* Decision Tree  
* Random Forest  
* Gradient Boosting  
* Naive Bayes  
* Support Vector Machine  
* Artificial Neural Network (ANN)

##  Best Model
Random Forest achieved the most balanced performance with:
* Precision: 1.0000  
* Recall: 0.8750  
* F1 Score: 0.9333  
* AUC: 1.0000  

## Run the App
1. Install required libraries:  
pip install -r requirements.txt

2. Run the Streamlit app:  
streamlit run app.py
