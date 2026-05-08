# Identifying Age-Related Medical Conditions
## CS4662 Advanced Machine and Deep Learning

## Project Overview

This project builds machine learning models to predict whether an individual has age-related medical conditions using anonymized biomedical data. 

Our data and project guidelines came from the following competition on Kaggle.com:

[text](https://www.kaggle.com/competitions/icr-identify-age-related-conditions)

We implemented and compared multiple approaches, including:
- Traditional Machine Learning (Logistic Regression, SVM, Random Forest)
- Boosting (XGBoost)
- Deep Learning (Neural Networks)
- Ensemble Methods (Stacking, Blending)

---

## Team Members

- Joe Miranda – Project Manager  
- Huilin Zhang – Data & Preprocessing  
- Luis Chavez – Deep & Ensemble Learning  
- Haonan Ma – Boosting & Advanced Models  
- Lcndr Aquino – Traditional ML Models  

---

## Dependencies

Install required packages:
pandas numpy matplotlib scikit-learn xgboost tensorflow

## Project Structure & How to Run

Data/ → Raw data is contained here
Notebooks/ → Jupyter notebooks
Reports/ → Final reports and output from data processing


### Run notebooks in this order:

1. `data_cleaning.ipynb`  
   → Cleans and prepares the dataset  

2. `traditional_ml.ipynb`  
   → Runs baseline and traditional ML models  

3. `boosting_xgboost.ipynb`  
   → Trains and evaluates XGBoost  

4. `deep_ensemble_learning.ipynb`  
   → Builds neural networks and ensemble models  

5. `final_model_comparison.ipynb`  
   → Combines all results and generates final comparisons  
