import sys
from src.EmpAttrition.logger import logging
from src.EmpAttrition.exception import CustomException
from src.EmpAttrition.Components.data_ingestion import DataIngestion
from src.EmpAttrition.Components.data_ingestion import DataIngestion, DataIngestionConfig
from src.EmpAttrition.Components.data_transformation import DataTransformation, DataTransformationConfig
from src.EmpAttrition.Components.model_trainer import Model_Trainer,ModelTrainingConfig




if __name__=='__main__':
    #logging.info("Logging test") 

    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        # model Training
        model_trainer= Model_Trainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Exception occured")
        raise CustomException(e,sys)