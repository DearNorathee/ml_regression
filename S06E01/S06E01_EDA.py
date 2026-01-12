# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 09:51:34 2026

@author: Norat
"""

#%%
import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor

# import simpleaudio as sa

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import sys
import modeling_tool as ml
import dataframe_short as ds
from sklearn.metrics import classification_report
from playsound import playsound


#%%
df_path = r"C:\Users\Norat\OneDrive\D_Code\Python\Python Modeling\Modeling 01\Dataset Regression\S06E01\S06E01_train.csv"
target_name = "exam_score"
saved_model_name = "AgModel S06E01_v01"

# positive_class = "High"
eval_metric='rmse'


folder_path = r"C:/Users/Heng2020/OneDrive/D_Code/Python/Python Modeling/Modeling 01/Code Classification/ml_binary_classification/Classify 02/Ag_Models"
model_path = r"C:/Users/Heng2020/OneDrive/Python Modeling/Modeling 01/Regression 02/Model 02.joblib"
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.wav"

#%%
data_ori = pd.read_csv(df_path,header=0)



