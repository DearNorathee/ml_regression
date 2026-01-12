# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:13:29 2023

@author: Heng2020
"""


import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor

import simpleaudio as sa

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import sys
import modeling_tool as ml
import dataframe_short as ds
from sklearn.metrics import classification_report
from playsound import playsound

def simple_playsound(file_path):
    # doesn't work
    import simpleaudio as sa
    # Load the sound
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    
    # Play the sound
    play_obj = wave_obj.play()
    
    # Wait until sound has finished playing
    play_obj.wait_done()

def cat_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    out_dict = {}
    out_dict['accuracy'] = accuracy_score(y_true,y_pred)
    out_dict['f1'] = f1_score(y_true,y_pred)
    out_dict['precision'] = precision_score(y_true,y_pred)
    out_dict['recall'] = recall_score(y_true,y_pred)
    
    return out_dict

df_path = r"C:/Users/Heng2020/OneDrive/D_Code/Python/Python Modeling/Modeling 01/Dataset Binary Classification/02 InsuranceFraud_test.csv"
y_name = "fraud_reported"
saved_model_name = "AgModel InsuranceFraud_v04_no_policy_no"

out_folder = "C:/Users/Heng2020/OneDrive/D_Code/Python/Python Modeling/Modeling 01/Dataset Binary Classification"
out_scored_name = "AgModel Insurance_fraud_v07_0.7667.csv"

out_scored_path = out_folder + "/" + out_scored_name
# positive_class = "High"
eval_metric='roc_auc'


model_path = r"C:/Users/Heng2020/OneDrive/D_Code/Python/Python Modeling/Modeling 01/Code Classification/ml_binary_classification/Classify 02/Ag_Models/AgModel Insurance Premium_v03_feature_select_feature_07"
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.wav"


drop_col01 = []
drop_col02 = []
drop_col03 = []
drop_col = drop_col01

date_cols = ["policy_bind_date","incident_date"]

data_ori = pd.read_csv(df_path,header=0)

mySeed = 20
# seems like for this data_set
# number_of_vehicles_involved, bodily_injuries better off being categorical
# while witnesses better off being numerical

num_to_cat_col = []
num_to_cat_col = ['number_of_vehicles_involved','bodily_injuries']
keep_cols = ["policy_number"]
n_data = 30_000

if isinstance(n_data, str) or data_ori.shape[0] < n_data :
    data = data_ori
else:
    data = data_ori.sample(n=n_data,random_state=mySeed)


data = data.drop(drop_col,axis=1)

################################### Pre processing - specific to this data(Blood Pressure) ##################
def pd_preprocess(data):
    df_cleaned = data

    return df_cleaned

null_report = ds.count_null(data)

#---------------------------------- Pre processing - specific to this data(Blood Pressure) -------------------
data = pd_preprocess(data)

data = ds.num_to_cat(data,num_to_cat_col)
data_types = pd.DataFrame(data.dtypes).reset_index()
data_types.columns = ["feature","dtype"]

cat_col = ds.cat_col(data)
ds.to_datetime(data,date_cols)
data = ds.to_category(data)



# Initialize predictor # train using precision
predictor =  TabularPredictor.load(model_path)
playsound(alarm_path)



#### prediction on training data
predictions = predictor.predict(data)
data_prediction = data.copy()
data_prediction[y_name + "_predict"] = predictions

# output like
test_prediction = data_prediction[keep_cols + [y_name + "_predict",]]
test_prediction = test_prediction.rename(columns = {y_name + "_predict":y_name})
test_prediction.to_csv(out_scored_path,index=False)






