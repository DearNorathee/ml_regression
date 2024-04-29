# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 11:29:38 2023

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
from collections import defaultdict
from itertools import product
######################################## Change Below #####################################
y_name = "expenses"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Model 01.joblib"
df_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Code Regression/Regression 01/01 Insurance Premium.csv"
num_to_cat_col = ['children']
output_name = "model_val01"
params_list = ['learning_rate','max_depth','n_estimators','subsample']
#-------------------------------------- Change Above -------------------------------

def xgb_predict(xgb_pipeline,X):
    y_model = xgb_pipeline.predict(X.to_dict("records"))
    return y_model

def xgb_predict_append(xgb_pipeline,X):
    y_model = xgb_predict(xgb_pipeline,X)
    X["y_model"] = y_model
    return X


def df_combination(dict_in):
    # Get all combinations of values for each key in the dictionary
    combinations = product(*dict_in.values())
    
    # Create a list of dictionaries with all combinations of key-value pairs
    list_of_dicts = [dict(zip(dict_in.keys(), combo)) for combo in combinations]
    
    # Convert the list of dictionaries to a pandas DataFrame
    df_combinations = pd.DataFrame(list_of_dicts)
    
    return df_combinations

def df_cat_combi(df_in):

    cat_dict = defaultdict(list)

    for col in df_in.columns:
        if df_in[col].dtype == 'object':
            for elem in df_in[col].unique():
                cat_dict[col].append(elem)

    cat_combi = df_combination(cat_dict)
    return cat_combi

def df_num_combi(df_in,n_sample = 30):
    num_dict = defaultdict(list)
    # n_sample = # of sample to generate
    numeric_cols = df_in.select_dtypes(include=['number']).columns.tolist()
    num = 30
    for col in numeric_cols:
        min_val = df_in[col].min()
        max_val = df_in[col].max()
        out_list = np.linspace(start = min_val, stop = max_val,num=num)
        num_dict[col] = list(out_list)
    num_combi = df_combination(num_dict)
    return num_combi
def make_testing_val(df_in,n_sample = 30):
    # n_sample = # of sample generate for each of numeric columns
    cat_combi = df_cat_combi(df_in)
    num_combi = df_num_combi(df_in,n_sample)
    
    out_df = merge_df(cat_combi,num_combi)
    return out_df
def merge_df(df1, df2):
    # not sure what it does
    """
    Merge two dataframes into all combinations from every row of df1 to every row of df2.
    """
    result = pd.merge(df1.assign(key=1), df2.assign(key=1), on='key').drop('key', axis=1)
    return result
def model_values(xgb_pipeline,X,y_name=""):
    # In case y_name="" means that X doesn't have the y values already
    
    if y_name != "":
        X_NO_y = X.drop(y_name,axis=1)
    else:
        X_NO_y = X
    test_val = make_testing_val(X_NO_y)
    model_val = xgb_predict_append(xgb_pipeline, test_val)
    return model_val
    
        


# load pipeline
load_pipeline = joblib.load(model_path)
data = pd.read_csv(df_path,header=0)
if ".csv" not in output_name:
    output_name = output_name + ".csv"

for col_name in num_to_cat_col:
    data[col_name] = data[col_name].astype(str)

params = load_pipeline.get_params()

# print the hyperparameters and their values
for param, value in params.items():
    for want_param in params_list:
        if want_param in param:
            print(want_param, "=", value)


model_prediction = model_values(load_pipeline, data, y_name) 
model_prediction.to_csv(output_name, index=False)   
             
            

