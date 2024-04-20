import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, train_and_evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data, into X and y")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], #train dataset's X; get all the row, and all the column except the last column
                train_array[:,-1], #train dataset's y; get all the row, and the last column
                test_array[:,:-1], #test dataset's X
                test_array[:,-1] #test dataset's y
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()    
                # "Lasso":Lasso(),
                # "Ridge":Ridge(),
            } 

            #Do Hyperparameter tuning; Can also write these hyperparameter in a config/yaml file and import in here when needed
            params={
                "Random Forest":{
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    #'max_features':['sqrt','log2','None'],  
                    'n_estimators':[8,16,32,64,128,256]                  
                },
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    #'loss':['squared_error','huber','absolute_error','quantile']
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    #'max_features':['sqrt','log2','None'],  
                    'n_estimators':[8,16,32,64,128,256]   
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    #'weights':['uniform','distance'],
                    #'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]  
                    
                },
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]                      
                },
                "AdaBoost Regressor": {
                    'learning_rate':[.1,.01,.05,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators':[8,16,32,64,128,256]  

                }   
            }

            #Train and evaluate the model
            model_report:dict=train_and_evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            #To get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            #To get best model name from dict
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model =  models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            #Can load the preprocessor pickle file in to do some data transformation if new data coming in

            logging.info(f"Saved trained model object.")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            #To get the prediction output of best model
            predicted= best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)

            return r2_square


        except:
            pass