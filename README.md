# Identifying Age-Related Medical Conditions
## CS4662 Advance Machine and Deep Learning

## Project Overview

This project builds machine learning models to predict whether an individual has age-related medical conditions using anonymized biomedical data. 

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
- Data/
- train.csv
- test.csv
- greeks.csv

- Notebooks/
- data_cleaning.ipynb
- traditional_ml.ipynb
- boosting_xgboost.ipynb
- deep_ensemble_learning.ipynb
- final_model_comparison.ipynb


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