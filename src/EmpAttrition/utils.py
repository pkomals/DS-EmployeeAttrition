import os 
import sys
from src.EmpAttrition.exception import CustomException
from src.EmpAttrition.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle


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


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
