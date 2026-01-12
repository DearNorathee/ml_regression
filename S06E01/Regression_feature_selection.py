# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:13:29 2023

@author: Heng2020
"""
# from pydantic.fields import ModelField

import autogluon.core as ag
ag.__version__
from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import sys
import autogluon.eda.analysis as eda
import dataframe_short as ds
from sklearn.metrics import classification_report
from playsound import playsound
from autogluon.core.metrics import make_scorer
from sklearn.metrics import mean_tweedie_deviance
from functools import partial
from sklearn.metrics import roc_auc_score

def score_func(eval_name):
    from sklearn.metrics import roc_auc_score

    if eval_name in ["roc_auc"]:
        return roc_auc_score

df_path = r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Modeling\Modeling 01\Dataset Regression\01 Insurance Premium.csv"
y_name = "fraud_reported"
saved_model_name = "AgModel Insurance Premium_v03_feature_select"



folder_path = r"C:/Users/Heng2020/OneDrive/D_Code/Python/Python Modeling/Modeling 01/Code Classification/ml_binary_classification/Classify 02/Ag_Models"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.mp3"


drop_col01 = ["_c39",]
drop_col02 = []
drop_col03 = []
drop_col = drop_col01

feature_selection = [
     (0, ["incident_severity","insured_hobbies"])
    ,(1, ["incident_severity","insured_hobbies", "vehicle_claim"])
    ,(2, ["incident_severity","insured_hobbies", "vehicle_claim", "collision_type", "witnesses", "auto_year", "incident_type"])
    ,(3, ["incident_severity","insured_hobbies", "vehicle_claim", "collision_type", "witnesses", "auto_year", "incident_type", "policy_number", "insured_occupation", "auto_make", "total_claim_amount", "months_as_customer", "capital-loss"])
    ,(4, ["incident_severity","insured_hobbies", "vehicle_claim", "collision_type", "witnesses", "auto_year", "incident_type", "policy_number", "insured_occupation", "auto_make", "total_claim_amount", "months_as_customer", "capital-loss", "incident_city", "umbrella_limit"])
    ,(5, ["incident_severity","insured_hobbies", "vehicle_claim", "collision_type", "witnesses", "auto_year", "incident_type", "policy_number", "insured_occupation", "auto_make", "total_claim_amount", "months_as_customer", "capital-loss", "incident_city", "umbrella_limit", "bodily_injuries", "insured_sex", "age", "insured_relationship", "injury_claim", "authorities_contacted"] )
    ,(6, ["incident_severity","insured_hobbies", "vehicle_claim", "collision_type", "witnesses", "auto_year", "incident_type", "policy_number", "insured_occupation", "auto_make", "total_claim_amount", "months_as_customer", "capital-loss", "incident_city", "umbrella_limit", "bodily_injuries", "insured_sex", "age", "insured_relationship", "injury_claim", "authorities_contacted", "policy_csl", "policy_deductable", "auto_model", "capital-gains", "policy_state", "policy_annual_premium"])
    ,(7, ["incident_severity","insured_hobbies", "vehicle_claim", "collision_type", "witnesses", "auto_year", "incident_type", "policy_number", "insured_occupation", "auto_make", "total_claim_amount", "months_as_customer", "capital-loss", "incident_city", "umbrella_limit", "bodily_injuries", "insured_sex", "age", "insured_relationship", "injury_claim", "authorities_contacted", "policy_csl", "policy_deductable", "auto_model", "capital-gains", "policy_state", "policy_annual_premium", "number_of_vehicles_involved", "insured_education_level", "police_report_available", "incident_hour_of_the_day", "property_claim", "incident_state", "incident_date", "property_damage", "insured_zip", "policy_bind_date"])
    ,
    ]

data_ori = pd.read_csv(df_path,header=0)

mySeed = 20
# 'prem_ops_ilf_table' could be num or cat


num_to_cat_col = [
                  ]

# YEAR_COL = 'contract_effective_yr'
# # OOT_YEAR needs to be string
# OOT_YEAR = '2019'
eval_metric = 'roc_auc'
n_data = 30_000

if isinstance(n_data, str) or data_ori.shape[0] < n_data :
    data = data_ori
else:
    data = data_ori.sample(n=n_data,random_state=mySeed)


saved_model_path = folder_path + "/" + saved_model_name
data = data.drop(drop_col,axis=1)

################################### Pre processing - specific to this data(Blood Pressure) ##################
def pd_preprocess(data):
    df_cleaned = data

    return df_cleaned

data = ds.num_to_cat(data,num_to_cat_col)
data_types = pd.DataFrame(data.dtypes).reset_index()
data_types.columns = ["feature","dtype"]

cat_col = ds.cat_col(data)
ds.to_datetime(data,date_cols)
data = ds.to_category(data)

#---------------------------------- Pre processing - specific to this data(Blood Pressure) -------------------

X_train, X_test, y_train, y_test = train_test_split(
                                        data.drop(y_name, axis=1), 
                                        data[y_name], 
                                        test_size=0.2, 
                                        random_state=mySeed)


train_df_all = pd.concat([X_train,y_train],axis=1)
test_df_all = pd.concat([X_test,y_test],axis=1)
# Load data
train_df_list = []
test_df_list = []

for _, feature in feature_selection:
    X_train_selected = X_train[feature]
    train_df = pd.concat([X_train_selected,y_train],axis=1)
    train_df_list.append(train_df)

for _, feature in feature_selection:
    X_test_selected = X_test[feature]
    test_df = pd.concat([X_test_selected,y_test],axis=1)
    test_df_list.append(test_df)


train_data_list = []

train_data = TabularDataset(train_df) 

for train_df in train_df_list:
    train_data = TabularDataset(train_df) 
    train_data_list.append(train_data)

predictor:TabularPredictor
predictor_list = []

for i, train_data in enumerate(train_data_list):
    # Initialize predictor # train using precision
    saved_model_path = folder_path + "/" + saved_model_name + f"_feature_{str(i).zfill(2)}"
    predictor = TabularPredictor(label=y_name, path=saved_model_path,eval_metric=eval_metric).fit(
        train_data,
                # presets = 'best_quality',
        )
    predictor_list.append(predictor)
playsound(alarm_path)

for i, train_data in enumerate(train_data_list):
    # Initialize predictor # train using precision
    saved_model_path = folder_path + "/" + saved_model_name + f"_feature_{str(i).zfill(2)}"
    predictor = TabularPredictor.load(saved_model_path)
    predictor_list.append(predictor)


train_metrics_list = []

metric_score_func = score_func(eval_metric)

for i, predictor in enumerate(predictor_list):
    #### prediction on training data
    predictions = predictor.predict(train_df_all)
    predictions_prob = predictor.predict_proba(train_df_all)
    train_prediction = train_df_all.copy()
    
    train_prediction[y_name + "_predict"] = predictions
    train_prediction[y_name + "_predict_prob"] = predictions
    
    
    if eval_metric in ["roc_auc"]:
        train_metrics = metric_score_func(train_prediction[y_name],train_prediction[y_name + "_predict_prob"])
    train_metrics_list.append(train_metrics)


test_metrics_list = []
# Making predictions on new data
for i, predictor in enumerate(predictor_list):
    predictions = predictor.predict(test_df_all)
    test_prediction = test_df_all.copy()
    test_prediction[y_name + "_predict"] = predictions
    
    test_prediction[y_name + "_predict"] = predictions
    test_prediction[y_name + "_predict_prob"] = predictions
    
    if eval_metric in ["roc_auc"]:
        test_metrics = metric_score_func(test_prediction[y_name],train_prediction[y_name + "_predict_prob"])
        
    test_metrics_list.append(test_metrics)

features_list = [item[1] for item in feature_selection]
feature_metrics = pd.DataFrame({
    'features': features_list,
    'train_metrics': train_metrics_list,
    'test_metrics': test_metrics_list
})
permutation_importances = predictor.feature_importance(data=train_data,model = predictor.get_model_best())




best_model_name = predictor.get_model_best()
# predictor.model_best is for ag version >= 1.1.0
best_model_name = predictor.model_best
print(best_model_name)
best_model_params = predictor.fit_hyperparameters_
print(best_model_params)


##### try to improve the model by groupping things

explainer = shap.Explainer(ag_predict, X_train)
shap_values = explainer(X_test)



