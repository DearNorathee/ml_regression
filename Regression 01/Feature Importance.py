# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 09:41:12 2023

@author: Heng2020
"""
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
from playsound import playsound
import csv
# put below to add alarm sound
# playsound(sound_path)

# This is very slow when I manually tuning
# (This might be because of many columns)


df_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Regression/Regression 06/06 Molecular Retention Time.csv"
data = pd.read_csv(df_path,header=0)
print(data.head(5))
y_name = "rt"
saved_model_name = "Model 06.01"
mySeed = 20
num_to_cat_col = []
# n_data = (string means used all data) || eg. n_data = "all"
n_data = 10000
used_col = None
drop_col = None
# from score >= 9
used_col = ['MaxEStateIndex',  'MinEStateIndex', 'MinAbsEStateIndex', 
            'MaxPartialCharge','BCUT2D_MWHI',    'BertzCT', 
            'PEOE_VSA2',       'PEOE_VSA6',      'PEOE_VSA8', 
            'SMR_VSA3',        'NHOHCount',      'MolLogP',
            y_name
            ]


# drop_col01 = ['sex']
# drop_col02 = ['region']
# drop_col03 = ['sex','region']
# drop_col = drop_col01


saved_model_name = "Model 02"
folder_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 06"
sound_path = r"H:\D_Music\Women Laugh.wav"
# model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"


null_report = f.count_null(data)
saved_model_path = folder_path + "/" + saved_model_name + ".joblib"
data = f.num_to_cat(data,num_to_cat_col)

################################### New Function #################################

if isinstance(n_data, str):
    data_used = data
else:
    data_used = data.sample(n=n_data,random_state=mySeed)
# temporary code below
data_used = data_used[used_col]

#!!!!!!!!!!!!!!!!!!! to be continued(Don't run this below part) !!!!!!!!!!!!!!!!!!!!!!!!!!


if used_col:
    if drop_col:
        data_used02 = data_used
    else:
        data_used02 = data_used.drop(drop_col,axis=1)
else:
    if drop_col:
        data_used02 = data_used[used_col]
    else:
        
        data_used02 = data_used
#-------!!!!!!!!! to be continued(Don't run this above part) !!!!!!!!!!!!!!!---------
#--------------------------- New Function  --------------------------------


X_train_full,X_dev_full, X_test_full, y_train,y_dev, y_test = f.train_dev_test_split(
                                                    data_used.drop(y_name, axis=1),
                                                    data_used[y_name],
                                                    seed=mySeed
                                                    )

# if drop_col:
#     X_train = X_train_full.drop(drop_col,axis=1)
#     X_dev = X_dev_full.drop(drop_col,axis=1)
#     X_test = X_test_full.drop(drop_col,axis=1)
# else:
#     # if drop_col is None
#     X_train = X_train_full
#     X_dev = X_dev_full
#     X_test = X_test_full

X_train = X_train_full
X_dev = X_dev_full
X_test = X_test_full

# insight:   n = 100*t,   t is the time in minutes

################################### New Function #################################
# X_01,y_01 = data_used[data_used.columns.tolist()[:-1]], \
#             data_used[data_used.columns.tolist()[-1]]

  
param_dict = {
    # np.arange(0, 10, 1)
    # from 0 to 9 step 1
    'learning_rate': 0.05,
    # n_estimators = 330 is best(so far)
    # n_estimators range is 300-400
    #'n_estimators': np.arange(250, 400, 20), 
    'max_depth': 4,
    'subsample': 0.7
    }

cv_result = f.xgb_DTrain_Tune(X_train,y_train,param_dict)

data_DMatrix = xgb.DMatrix(data=X_train, label=y_train)

xgb_model = xgb.train(dtrain=data_DMatrix,params = param_dict)

importance_scores =  xgb_model.get_score(importance_type='weight')


def create_result_df(X,y_actual,y_predict):
    df_result = X.copy()
    df_result['y_actual'] = y_actual
    df_result['y_model'] = y_predict
    
    df_result['y_diff'] = abs(df_result['y_model']-df_result['y_actual'])

    df_result['y_diff%1'] = df_result['y_diff']/((df_result['y_model']+df_result['y_actual'])/2)
    df_result['y_diff%2'] = df_result['y_diff']/df_result['y_actual']
    df_result['y_diff%3'] = df_result['y_diff']/df_result['y_model']
    return df_result

# what if I change children to cat?
# data['children'] = data['children'].astype(str)
# data = data.drop('children', axis=1)

# ############################ change some numeric columns to categorical instead ############################

for col_name in num_to_cat_col:
    data[col_name] = data[col_name].astype(str)


X_train,X_dev, X_test, y_train,y_dev, y_test = train_dev_test_split(
                                                    data.drop(y_name, axis=1),
                                                    data[y_name],
                                                    seed=mySeed
                                                        )
# train to see the feature importance
best_params_val04 = {
    'learning_rate': 0.25, 
    'n_estimators': 20, 
    'max_depth': 3,
    "num_boost_round": 100 
    }

###################### Look at only train ##################
X_train_encoded = pd.get_dummies(X_train,drop_first=True,columns=cat_col)
X_train_DMatrix = xgb.DMatrix(data=X_train_encoded,label=y_train)
xgb_model = xgb.train(dtrain=X_train_DMatrix,params = best_params_val04)
xgb.plot_importance(xgb_model)
importance_scores =  xgb_model.get_score(importance_type='weight')
###################### Look at all data ###############################
