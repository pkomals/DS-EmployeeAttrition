import sys
from src.EmotionRecog.logger import logging


def error_message_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

'''
Key Points:

error_details.exc_info(): This function retrieves the current exception information, including the exception type, value, and traceback.
exc_tb.tb_frame.f_code.co_filename: This extracts the filename from the traceback information.
exc_tb.tb_lineno: This extracts the line number from the traceback information.
str(error): Converts the error object to a string for inclusion in the error message.

'''


class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message