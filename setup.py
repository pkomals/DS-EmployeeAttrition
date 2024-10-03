from setuptools import find_packages, setup
from typing import List



def get_requirements(file_path:str)-> List[str]:
    '''
    This function will return the list of requirements in list format
    ['numpy', 'panda',...]
    '''
    HYPHENe= '-e .'

    requirements=[]

    with open(file_path)as file_obj:
        '''
        The with statement in combination with open() is commonly used to work with files in Python 
        because it guarantees that the file will be closed regardless of whether an exception occurs or not. 
        This helps prevent resource leaks and ensures proper file handling.
        '''
        # to read the content of file_obj
        requirements= file_obj.readlines()
        
        # to eleminate the newline character 
        requirements=[req.replace('\n',"") for req in requirements]

        if HYPHENe in requirements:
            requirements.remove(HYPHENe) # this is to avoid -e . getting read by get_requirement function 

    return requirements




setup(
    name='Emotion Recognition',
    version='0.0.1',
    author='Komal Patil',
    packages=find_packages(),
    #install_requires=['numpy','pandas'],
    install_requires= get_requirements('requirements.txt'), #to avoid manual listing of requirements


)