import os
import sys
import numpy as np
import pandas as pd
import dill 

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)#create pickle file
    
    except Exception as e:
        raise CustomException(e,sys)
    
def train_and_evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        
        for i in range(len(list(models))):
            # loop through each model
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            #apply GridSearch or randomiseCV (to perform hyperparam tuning on the model)
            gs= GridSearchCV(model,para,cv=3)# initialiase GridSearchCV object ; do 3-fold cross-validation
            gs.fit(X_train,y_train) # fits the GridSearchCV to training data; Performs grid search to find the best hyperparam for the model using this train data. ( training of the model occurs for every combination of hyperparameters for 3 times(cv=3), and the model's performance is evaluated using cross-validation to determine the best hyperparameters.)

            #select the best hyperparam and set them to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) #Train model with the best hyperparams obtained from the grid search.

            # Make predictions
            y_train_pred =model.predict(X_train)
            y_test_pred =model.predict(X_test)

            # Evaluate Train and Test dataset
            train_model_score = r2_score(y_train,y_train_pred) #gt-label, prediction output
            test_model_score = r2_score(y_test,y_test_pred) 

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)