import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.EmpAttrition.exception import CustomException
from src.EmpAttrition.logger import logging
from src.EmpAttrition.utils import save_object, evaluate_models
from src.EmpAttrition.Components.data_ingestion import DataIngestionConfig,DataIngestion
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Splitting done")
            # Perform PCA for dimensionality reduction
            pca = PCA(n_components=10)  # Adjust number of components
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Feature selection using RFE
            logging.info("Starting Recursive Feature Elimination (RFE)")
            rfe_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rfe = RFE(estimator=rfe_model, n_features_to_select=9) 
            X_train_rfe = rfe.fit_transform(X_train_pca, y_train)
            X_test_rfe = rfe.transform(X_test_pca)
            logging.info(f"Feature selection using RFE")

            models = {
                "Random Forest": RandomForestClassifier(),
                # "Decision Tree": DecisionTreeRegressor(),
                # "Gradient Boosting": GradientBoostingClassifier(),
                # "Linear Regression": LinearRegression(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                # "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                # "Decision Tree": {
                #     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                #     # 'splitter':['best','random'],
                #     # 'max_features':['sqrt','log2'],
                # },
                "Random Forest":{
                    'criterion':['gini','entropy','log_loss'],#,'squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'class_weight':['balanced','balanced_subsample'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "Gradient Boosting":{
                #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     'learning_rate':[.1,.01,.05,.001],
                #     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #     # 'criterion':['squared_error', 'friedman_mse'],
                #     # 'max_features':['auto','sqrt','log2'],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                # "Linear Regression":{},
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                # "AdaBoost Regressor":{
                #     'learning_rate':[.1,.01,0.5,.001],
                #     # 'loss':['linear','square','exponential'],
                #     'n_estimators': [8,16,32,64,128,256]
                # }
                
            }
            # Apply RandomizedSearchCV for hyperparameter tuning
            best_models = {}
            # for model_name, model in models.items():
            #     if model_name in params:
            #         logging.info(f"Tuning hyperparameters for {model_name}")
            #         search = RandomizedSearchCV(model, params[model_name], n_iter=10, cv=3, random_state=42, n_jobs=-1)
            #         search.fit(X_train_rfe, y_train)
            #         best_models[model_name] = search.best_estimator_
            #     else:
            #         best_models[model_name] = model

            model_report:dict=evaluate_models(X_train_rfe,y_train,X_test_rfe,y_test,models,params)
            print(f"Mddel Report{model_report}:")

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)
            print(best_model_score)

            model_names = list(params.keys())

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test_rfe)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)