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

df_path = r"C:/Users/Heng2020/OneDrive/D_Code/Python/Python Modeling/Modeling 01/Dataset Binary Classification/02 InsuranceFraud_train.csv"
y_name = "fraud_reported"
saved_model_name = "AgModel InsuranceFraud_v04_no_policy_no"

# positive_class = "High"
eval_metric='roc_auc'


folder_path = r"C:/Users/Heng2020/OneDrive/D_Code/Python/Python Modeling/Modeling 01/Code Classification/ml_binary_classification/Classify 02/Ag_Models"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.wav"


drop_col01 = ["_c39","incident_severity","policy_number"]
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

null_report = ds.count_null(data)

#---------------------------------- Pre processing - specific to this data(Blood Pressure) -------------------
data = pd_preprocess(data)

data = ds.num_to_cat(data,num_to_cat_col)
data_types = pd.DataFrame(data.dtypes).reset_index()
data_types.columns = ["feature","dtype"]

cat_col = ds.cat_col(data)
ds.to_datetime(data,date_cols)
data = ds.to_category(data)

# convert positive class to 1, negative class to 0
# as of now autogluon, doesn't have a way to specify 
# autogluon version 0.8.2, Jan 6, 2024
# data[y_name] = np.where(data[y_name] == positive_class,1,0 )

X_train, X_test, y_train, y_test = train_test_split(
                                        data.drop(y_name, axis=1), 
                                        data[y_name], 
                                        test_size=0.2, 
                                        random_state=mySeed)


train_df = pd.concat([X_train,y_train],axis=1)
test_df = pd.concat([X_test,y_test],axis=1)
# Load data
train_data = TabularDataset(train_df) 


# Initialize predictor # train using precision
predictor = TabularPredictor(label=y_name, path=saved_model_path,eval_metric=eval_metric).fit(train_data)
playsound(alarm_path)

results = predictor.fit_summary(show_plot=True)
results_train = predictor.leaderboard(train_df, silent=True)
results_test = predictor.leaderboard(test_df, silent=True)

#### prediction on training data
predictions = predictor.predict(train_df)
train_prediction = train_df.copy()
train_prediction[y_name + "_predict"] = predictions
train_metric = classification_report(train_prediction[y_name], train_prediction[y_name + "_predict"])


# Making predictions on new data
predictions = predictor.predict(test_df)
test_prediction = test_df.copy()
test_prediction[y_name + "_predict"] = predictions
test_metric = classification_report(train_prediction[y_name], train_prediction[y_name + "_predict"])

# feature_importance from autogluon is actually permutation_importances
permutation_importances = predictor.feature_importance(data=test_df,model = predictor.get_model_best())


best_model_name = predictor.get_model_best()
best_model_params = predictor.fit_hyperparameters_
print(best_model_params)



