🛡️Payment Fraud Detection using Machine Learning




This project implements a Machine Learning based payment fraud detection system using a verified transaction dataset.
The goal is to classify online transactions as fraud or legitimate based on extracted numerical and categorical features such as
account age, number of items, local time, payment method, category, and behavioral indicators.

The model is trained using a Random Forest Classifier, saved using joblib, and can be used to predict
fraud status on demand.

📌Features

Preprocessed payment fraud dataset

Random Forest based ML model

Feature based transaction classification

Training script + prediction script

Saved model files for fast inference

Clean and organized project structure

📁Project Structure

project/

│
├── train_payment_fraud_model.py # Script to train the ML model
├── predict_fraud.py # Script to run predictions
│
├── dataset_payments.csv # Verified dataset
├── fraud_rf_model.joblib # Saved Random Forest model
├── fraud_label_encoder.joblib # Label encoder for output labels
│
└── README.md # Project documentation

📦Dataset

The dataset contains:

Structured numeric features

Encoded categorical attributes

Behavioral indicators (weekend, count, local time)

Target label: fraud or legitimate

Only cleaned and encoded features are used during training.


📄License

This project is licensed under the MIT License.
