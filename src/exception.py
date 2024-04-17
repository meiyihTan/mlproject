import sys
import logging

def error_message_detail(error,error_detail:sys): #error_details will be present in sys
    ''' 
    exc_tb tells which file and which line the exception occurs
    '''
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))

    return error_message

class CustomException(Exception): #inherit from default Exception class
    def __init__(self,error_message,error_detail:sys): #error_detail is sys type, error_detail will be track by sys
        super().__init__(error_message) #call the parent class constructor
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self): #__str__() inherit from Exception class
        '''will get all the error_message when we try to print this function'''
        return self.error_message


# if __name__=="__main__":

#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by Zero")
#         raise CustomException(e,sys)