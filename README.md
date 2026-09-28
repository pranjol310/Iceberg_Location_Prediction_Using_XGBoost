```
Iceberg Location Prediction Using XGBoost

A machine learning project for predicting the location/class of icebergs from input data using the XGBoost algorithm.

The project is organized into three main stages:

Data Preprocessing – Cleaning and preparing the raw dataset for machine learning.
Training – Training an XGBoost model on the processed data.
Prediction – Using the trained model to make predictions on new/unseen data.

Project Overview

The complete workflow follows a standard machine learning pipeline:

Raw Data
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Preparation
   │
   ▼
XGBoost Model Training
   │
   ▼
Trained Model
   │
   ▼
Prediction on New Data


iceberg-location-prediction/
│
├── data_preprocessing/
│   ├── README_preprocessing.md
│   ├── build_dataset.py
│   ├── make_tracks.py
│   └── training_data.csv
│
├── train/
│   ├── README_train.md
│   └── train.py
│
├── prediction/
│   ├── README_prediction.md
│   └── predict.py
│
├── README.md
```
