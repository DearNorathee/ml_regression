# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:50:11 2023

@author: Heng2020
"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
import optuna
# joblib: for save&load pipeline
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 01/01 Insurance Premium.csv"
data = pd.read_csv(df_path,header=0)
print(data.head(5))
y_name = "expenses"
saved_model_name = "Model 01"
mySeed = 20
num_to_cat_col = ['children']
drop_col01 = ['sex']
drop_col02 = ['region']
drop_col03 = ['sex','region']
saved_model_name = "reg01_01"
drop_col = drop_col01

def create_result_df(X,y_actual,y_predict):
    df_result = X.copy()
    df_result['y_actual'] = y_actual
    df_result['y_model'] = y_predict
    
    df_result['y_diff'] = abs(df_result['y_model']-df_result['y_actual'])

    df_result['y_diff%1'] = df_result['y_diff']/((df_result['y_model']+df_result['y_actual'])/2)
    df_result['y_diff%2'] = df_result['y_diff']/df_result['y_actual']
    df_result['y_diff%3'] = df_result['y_diff']/df_result['y_model']
    return df_result
def train_dev_test_split(X,y,seed=0):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
                                                    data.drop(y_name, axis=1), 
                                                    data[y_name], 
                                                    test_size=0.3,
                                                    random_state=mySeed
                                                    )
    
    X_dev,X_test,y_dev,y_test = train_test_split(
                                    X_dev_test,
                                    y_dev_test,
                                    test_size=1/3,
                                    random_state=mySeed
                                        )
    return [X_train,X_dev,X_test,y_train,y_dev,y_test]
    
def xgb_predict(xgb_pipeline,X):
    y_model = xgb_pipeline.predict(X.to_dict("records"))
    return y_model
# what if I change children to cat?
# data['children'] = data['children'].astype(str)
# data = data.drop('children', axis=1)

# ############################ change some numeric columns to categorical instead ############################

for col_name in num_to_cat_col:
    data[col_name] = data[col_name].astype(str)


X_train_full,X_dev_full, X_test_full, y_train,y_dev, y_test = train_dev_test_split(
                                                    data.drop(y_name, axis=1),
                                                    data[y_name],
                                                    seed=mySeed
                                                        )
# X_train_full,X_dev_full, X_test_full  are for creating final model report 
# They have full columns(X_train... has droped columns)

X_train = X_train_full.drop(drop_col,axis=1)
X_dev = X_dev_full.drop(drop_col,axis=1)
X_test = X_test_full.drop(drop_col,axis=1)


best_params_val01 = {
    'learning_rate': 0.28213699353598115,
    'n_estimators': 806,
    'subsample': 0.8557493246298656,
    'max_depth': 9}
best_params_val02 = {
    'learning_rate': 0.29250783606455044, 
    'n_estimators': 210, 
    'subsample': 0.47317158951583094, 
    'max_depth': 8}
best_params_val03 = {
    'learning_rate': 0.0037943253426943394, 
    'n_estimators': 833, 
    'subsample': 0.7532000685669987, 
    'max_depth': 8}
best_params_val04 = {
    'learning_rate': 0.25, 
    'n_estimators': 20, 
    'max_depth': 3,
    "num_boost_round": 100 
    }
best_params_val05 = {
    'learning_rate': 0.25, 
    'n_estimators': 15, 
    'max_depth': 3,
    "num_boost_round": 100 
    }

chosen_params = best_params_val05

steps01 = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(**chosen_params))]



# Create the pipeline: xgb_pipeline
# xgb_pipe01 => is the pipeline object
xgb_pipe01 = Pipeline(steps01)
# Fit the pipeline

xgb_pipe01.fit(X_train.to_dict("records"), y_train)

y_train_pred01 = xgb_predict(xgb_pipe01,X_train)
y_dev_pred01 = xgb_predict(xgb_pipe01,X_dev)
y_test_pred01 = xgb_predict(xgb_pipe01,X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred01))
rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred01))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred01))


rmse_train_format = '{:,.3f}'.format(rmse_train)
rmse_dev_format = '{:,.0f}'.format(rmse_dev)
rmse_test_format = '{:,.0f}'.format(rmse_test)

print("train: " + rmse_train_format)
print("dev: " + rmse_dev_format)
print("test: " + rmse_test_format)

xgb_model01 = xgb_pipe01.named_steps['xgb_model']

y_train_result = create_result_df(X_train_full, y_train, y_train_pred01)
y_dev_result = create_result_df(X_dev_full, y_dev, y_dev_pred01)
y_test_result = create_result_df(X_test_full, y_test, y_test_pred01)

joblib.dump(xgb_pipe01,saved_model_name +".joblib")