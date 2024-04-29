# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:57:41 2023

@author: Heng2020
"""
# Import necessary modules
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

df_path = r"C:\Users\Heng2020\OneDrive\Python 01\Python 01\Regression 01\01 Insurance Premium.csv"
data = pd.read_csv(df_path,header=0)
print(data.head(5))
y_name = "expenses"
saved_model_name = "Model 01"
mySeed = 10

def train_dev_test_split(X,y,seed=0):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
                                                    X, 
                                                    y, 
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
    

# what if I change children to cat?
data['children'] = data['children'].astype(str)
data = data.drop('children', axis=1)

X_train,X_dev, X_test, y_train,y_dev, y_test = train_dev_test_split(
                                                    data.drop(y_name, axis=1),
                                                    data[y_name],
                                                    seed=mySeed
                                                        )


def xgb_predict(xgb_pipeline,X):
    y_model = xgb_pipeline.predict(X.to_dict("records"))
    return y_model





def objective(trial):
    learning_rate = trial.suggest_float("learning_rate",0,1)
    n_estimators = trial.suggest_int("n_estimators",50,1000)
    subsample = trial.suggest_float("subsample",0,1)
    max_depth = trial.suggest_int("max_depth",3,10)
    
    # Setup the pipeline steps: steps
    steps = [("ohe_onestep", DictVectorizer(sparse=False)),
             ("xgb_model", xgb.XGBRegressor(
                                 learning_rate = learning_rate,
                                 n_estimators = n_estimators,
                                 subsample = subsample,
                                 max_depth=max_depth, 
                                 objective="reg:squarederror"))]
    # Create the pipeline: xgb_pipeline
    xgb_pipeline = Pipeline(steps)

    # Fit the pipeline
    xgb_pipeline.fit(X_train.to_dict("records"), y_train)


    # Cross-validate the model
    # cross_val_scores_train = cross_val_score(
    #                         xgb_pipeline,
    #                         X_train.to_dict("records"),
    #                         y_train,
    #                         scoring = "neg_mean_squared_error",
    #                         cv=10)

    # RMSE_train = np.mean(np.sqrt(np.abs(cross_val_scores_train)))
    
    # Print the 10-fold RMSE
    # print("10-fold RMSE: ",RMSE_train )
    y_dev_pred = xgb_predict(xgb_pipeline,X_dev)
    RMSE_dev02 = mean_squared_error(y_dev, y_dev_pred)
#  I don't know why they chose neg_mean_squared_error instead of rmse
    cross_val_scores_dev = cross_val_score(
                            xgb_pipeline,
                            X_dev.to_dict("records"),
                            y_dev,
                            scoring = "neg_mean_squared_error",
                            cv=10)
    RMSE_dev = np.mean(np.sqrt(np.abs(cross_val_scores_dev)))
    
    
    
    return RMSE_dev02
# ####################################### Use Optuna for tuning ################################
# Create a study object and optimize the objective function
# about 15 sec/ trial
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)



def trial_info(study):
    all_trial_results = study.get_trials()
    trials_data = []
    for each_trial in all_trial_results:
        param = each_trial.params
        rmse = each_trial.values[0]
        param['RMSE'] = rmse
        trials_data.append(param)
    df = pd.DataFrame(trials_data)
    return df[['learning_rate', 'n_estimators', 'subsample', 'max_depth', 'RMSE']]

all_trial = trial_info(study)
    


# Print the best hyperparameters and accuracy
best_params = study.best_params
best_accuracy = study.best_value
print("Best hyperparameters: ", best_params)
print("Best accuracy: ", best_accuracy)

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



def create_result_df(X,y_actual,y_predict):
    df_result = X.copy()
    df_result['y_actual'] = y_actual
    df_result['y_model'] = y_predict
    
    df_result['y_diff'] = abs(df_result['y_model']-df_result['y_actual'])

    df_result['y_diff%1'] = df_result['y_diff']/((df_result['y_model']+df_result['y_actual'])/2)
    df_result['y_diff%2'] = df_result['y_diff']/df_result['y_actual']
    df_result['y_diff%3'] = df_result['y_diff']/df_result['y_model']
    return df_result
    
    

steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(**best_params))]



# Create the pipeline: xgb_pipeline
# xgb_model01 => is the pipeline object
xgb_model01 = Pipeline(steps)
# Fit the pipeline

xgb_model01.fit(X_train.to_dict("records"), y_train)

y_train_pred = xgb_predict(xgb_model01,X_train)
y_dev_pred = xgb_predict(xgb_model01,X_dev)
y_test_pred = xgb_predict(xgb_model01,X_test)

rmse_train = mean_squared_error(y_train, y_train_pred)
rmse_dev = mean_squared_error(y_dev, y_dev_pred)
rmse_test = mean_squared_error(y_test, y_test_pred)

print("train: " + str(int(rmse_train)))
print("dev: " + str(int(rmse_dev)))
print("test: " + str(int(rmse_test)))


y_train_result = create_result_df(X_train, y_train, y_train_pred)
y_dev_result = create_result_df(X_dev, y_dev, y_dev_pred)
y_test_result = create_result_df(X_test, y_test, y_test)


#####################################################
param02 = {
    'learning_rate': 0.1, 
    'n_estimators': 500, 
    'max_depth': 4
    }
steps02 = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(**param02))]

xgb_model02 = Pipeline(steps02)
# Fit the pipeline

xgb_model02.fit(X_train.to_dict("records"), y_train)

y_train_pred02 = xgb_predict(xgb_model02,X_train)
y_dev_pred02 = xgb_predict(xgb_model02,X_dev)
y_test_pred02 = xgb_predict(xgb_model02,X_test)

rmse_train02 = mean_squared_error(y_train, y_train_pred02)
rmse_dev02 = mean_squared_error(y_dev, y_dev_pred02)
rmse_test02 = mean_squared_error(y_test, y_test_pred02)

print("train: " + str(int(rmse_train02)))
print("dev: " + str(int(rmse_dev02)))
print("test: " + str(int(rmse_test02)))


y_train_result02 = create_result_df(X_train, y_train, y_train_pred02)
y_dev_result02 = create_result_df(X_dev, y_dev, y_dev_pred02)
y_test_result02 = create_result_df(X_test, y_test, y_test_pred02)






# save pipeline
joblib.dump(xgb_model01,saved_model_name +".joblib")


# load pipeline
model_path = r"Model 01.joblib"
load_pipeline = joblib.load(model_path)

y_load_pred = xgb_predict(load_pipeline, X_test)



    


