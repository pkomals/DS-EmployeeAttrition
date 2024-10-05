import os 
import sys
from src.EmotionRecog.exception import CustomException
from src.EmotionRecog.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql


load_dotenv()
host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Connecting--> mysql")
    try:
        #print(f"Host: {host}, User: {user}, Password: {password}, DB: {db}")
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection established")
        df=pd.read_sql_query("select * from employee_attrition",mydb)
        print(df.head())

        return df #returning df to data ingestion(raw)
    except Exception as e:
        raise CustomException(e,sys)

