#feature engineering, data cleaning , 
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer #for missing value
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')#path to save the preprocessor pipeline/model pickle file

class DataTransformation:
    def __init__(self):
        self.data_transfomation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        Create all the pickle files that will do the data transformation such as:
        convert categorical to numerical, perform standard scalar
        '''
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            
            #create preprocessor/preprocess pipeline (just do on train dataset, just like we do fit_transform() on train dataset; and only transform() on test dataset)
            #1. create a numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),#handle missing value, use median to fill in missing values (becs from EDA we found that the data has outlier)
                    ("scaler",StandardScaler())#do standard scalar
                ]
            )
                        
            #2. create a categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),#handle missing value, use mode to fill in missing values
                    ("one_hot_encoder",OneHotEncoder()),#do OneHotEncoding  
                    ("scaler",StandardScaler(with_mean=False))#do standard scalar(Standardization)
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            #combine the numerical and categorical pipeline
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns), #pipeline name, what pipeline, input to the pipeline
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor           
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try :
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column_name="math_score" #y-label, dependent variable
            numerical_columns=["reading_score","writing_score"]

            # Get X and y of train and test dataset
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) # X-input feature, independent variable 
            target_feature_train_df=train_df[target_column_name] #y label feature, ground-truth, dependent variable
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) # X-input feature, independent variable
            target_feature_test_df=test_df[target_column_name] #y label feature, ground-truth, dependent variable
                
            logging.info("Applying preprocessing object on training dataframe and testing dataframe") 
            #apply .fit_transform() on train dataset and transform() on test dataset                        
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[ 
                input_feature_train_arr,np.array(target_feature_train_df) #train dataset's : X,y
            ]
            test_arr = np.c_[ 
                input_feature_test_arr,np.array(target_feature_test_df) #test dataset's : X,y
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transfomation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transfomation_config.preprocessor_obj_file_path,
            )        

        except Exception as e:
            raise CustomException(e,sys)

