import os 
import sys 
from pathlib import Path
import logging 
logging.basicConfig(level=logging.INFO, format='[%(asctime)s : %(message)s]')
PROJECT_NAME = 'mlapp'

list_of_files = [

    'data/__init__.py',
    'docs/test_doc.pdf',
    f'src/{PROJECT_NAME}/__init__.py',
    f'src/{PROJECT_NAME}/README.md',
    'notebooks/research.ipynb',
    'config/config.yaml',
    'config/Version.txt',
    'README.md',
    'requirements.txt',
    'setup.py',   
    'project.yaml',   
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)
 
    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'file directory {filedir} created')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath ,'w') as f :
            pass 
            logging.info(f'creating empty file {filename}')
            
    else:
        logging.info(f'file {filename} alredy exists')

    
