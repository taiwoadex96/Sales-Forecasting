from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''Reads requirements.txt and strips out tracking flags'''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

setup(
    name='walmart_sales_forecasting',
    version='0.0.1',
    author='Adex',
    author_email='taiwoadex96@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)