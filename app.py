import sys
from src.EmotionRecog.logger import logging
from src.EmotionRecog.exception import CustomException
from src.EmotionRecog.Components.data_ingestion import DataIngestion
from src.EmotionRecog.Components.data_ingestion import DataIngestionConfig


if __name__=='__main__':
    #logging.info("Logging test") 

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Exception occured")
        raise CustomException(e,sys)