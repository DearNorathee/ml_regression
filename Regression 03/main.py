# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:10:41 2023

@author: Heng2020
"""
# this is for testing function in ungroupped01
import joblib
import myLib01.ungroupped01 as f
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
import time
import xgboost as xgb


df_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Regression/Regression 03/04 Home Insurance.csv"
data = pd.read_csv(df_path,header=0)
print(data.head(5))
y_name = "charges"
saved_model_name = "Model 01"
mySeed = 20
num_to_cat_col = []

drop_col01 = ['sex']
drop_col02 = ['region']
drop_col03 = ['sex','region']
drop_col = None


saved_model_name = "Model 02"
folder_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"
#########################################################

saved_model_path = folder_path + "/" + saved_model_name + ".joblib"

data = f.num_to_cat(data,num_to_cat_col)
X_train_full,X_dev_full, X_test_full, y_train,y_dev, y_test = f.train_dev_test_split(
                                                    data.drop(y_name, axis=1),
                                                    data[y_name],
                                                    seed=mySeed
                                                    )


if drop_col:
    X_train = X_train_full
    X_dev = X_dev_full
    X_test = X_test_full
else:
    X_train = X_train_full.drop(drop_col,axis=1)
    X_dev = X_dev_full.drop(drop_col,axis=1)
    X_test = X_test_full.drop(drop_col,axis=1)

# insight:   n = 100*t,   t is the time in minutes

############################# Code for manual tuning #################################

param_dict = {
    # np.arange(0, 10, 1)
    # from 0 to 9 step 1
    'learning_rate': [0.2], 
    'n_estimators': [18], 
    'max_depth': [3],
    'subsample': [0.6]
    }


param_df = f.param_combination(param_dict)
print(f"Total test: {param_df.shape[0]} ")
param_test_name = f.tune_param_name(param_dict)
print(param_test_name)


rmse_df = f.xgb_RMSE(X_train,y_train,param_df)
sns.lineplot(x=param_test_name, y='RMSE',data=rmse_df)

#----------------------------- Code for manual tuning ------------------------------------

param_grid = {
    # np.arange(0, 10, 1)
    # from 0 to 9 step 1
    'learning_rate': np.arange(0.16, 0.26, 0.02), 
    'n_estimators': [18], 
    'max_depth': [4],
    'subsample': [0.6]
    }

ans02 = f.xgb_ParamSearch(X_train, y_train, param_grid)


best_params_val01 = {
    'learning_rate': 0.25, 
    'n_estimators': 20, 
    'max_depth': 3,
    #"num_boost_round": 100 
    }

best_params_val02 = {
    'learning_rate': 0.25, 
    'n_estimators': 25, 
    'max_depth': 3,
    #"num_boost_round": 100 
    }
chosen_params = best_params_val01

#xgb_model00 = f.xgb_tune(X_train, y_train, X_dev, y_dev, X_test, y_test, chosen_params)
xgb_model01 = f._xgb_RMSE_H1(X_train, y_train, chosen_params)
xgb_model01 = f.xgb_tune02(X_train, y_train, chosen_params)

################### save pipeline
joblib.dump(xgb_model01,saved_model_path)

###################### load model
load_pipeline = joblib.load(model_path)

y_load_pred = f.xgb_predict(load_pipeline, X_test)
df_result = f.create_result_df(X_test, y_test, y_load_pred)


sns.scatterplot(x='y_actual', y='y_diff%1',data=df_result,hue="smoker")
sns.scatterplot(x='y_actual', y='y_diff%2',data=df_result,hue="sex")
sns.scatterplot(x='y_actual', y='y_diff%3',data=df_result,hue="region")


