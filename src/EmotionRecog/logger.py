import os #this helps retriving the relative path of files whenever needed.
import logging
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"

log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True )

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

#format to print log message
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, # could be logging.warning/logging.error as per the logging type
    
)