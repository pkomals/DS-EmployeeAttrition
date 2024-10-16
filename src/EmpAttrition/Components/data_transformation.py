import sys
from dataclasses import dataclass

from imblearn.combine import SMOTETomek
from collections import Counter
import pandas as pd
import numpy as np

import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.EmpAttrition.exception import CustomException
from src.EmpAttrition.logger import logging
from src.EmpAttrition.utils import save_object
from src.EmpAttrition.Components.data_ingestion import DataIngestionConfig,DataIngestion
from sklearn.base import BaseEstimator, TransformerMixin

import os 



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifact','preprocessor.pkl')


class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        data = pd.read_csv(DataIngestionConfig.raw_data_path)

    def get_data_tranformer_obj(self):
        '''
        Data Transformation
        '''
        try:
            data = pd.read_csv(DataIngestionConfig.raw_data_path)
            Numerical_Attributes=[attr for attr in data.columns if data[attr].dtype!='O']
            Categorical_Attributes=[attr for attr in data.columns if data[attr].dtype=='O']
            Numerical_Attributes.remove('left')
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',MinMaxScaler())

            ])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder()),
            ("scalar",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{Categorical_Attributes}")
            logging.info(f"Numerical Columns:{Numerical_Attributes}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,Numerical_Attributes),
                    ("cat_pipeline",cat_pipeline,Categorical_Attributes)
                ]

            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj=self.get_data_tranformer_obj()

            
            target_attr='left'

            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_attr],axis=1)
            target_feature_train_df=train_df[target_attr]
            

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_attr],axis=1)
            target_feature_test_df=test_df[target_attr]

            logging.info("Applying Preprocessing on training and test dataframe")            

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # counter = Counter(target_feature_train_df)
            # # print('Before', counter)
            # smtom = SMOTETomek(random_state=139)
            # input_feature_train_arr, target_feature_train_df = smtom.fit_resample(input_feature_train_arr, target_feature_train_df)

            # counter = Counter(target_feature_train_df)
            # # print('After', counter)
            
            # logging.info("Done Oversampling")



            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            #logging.info(f"train_arr{train_arr}")


            

            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object")


            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )


        except Exception as e:
            raise CustomException(e,sys)
    