from mlapp import logger 
from common import read_yaml, create_directories
from pathlib import Path 
import urllib.request as requests 
import zipfile
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import json 
import joblib

import re 

test_yamlfilepath  = Path('test_config.yaml')

# read the yaml file 
test_config = read_yaml(test_yamlfilepath)
#create the directory
create_directories([test_config.artifacts_root])

# read the data ingestion parameters 
data_ingestion = test_config.data_ingestion
logger.info(data_ingestion)

#create the data ingestion directory
create_directories([data_ingestion.root_dir])

# download the data 
filename , header = requests.urlretrieve(
    url=data_ingestion.data_url,
    filename=data_ingestion.data_local_file,
)
logger.info(f'Data downloaded with name {filename}')

# unzipping the data 
with zipfile.ZipFile(data_ingestion.data_local_file, 'r') as zip_ref :
    zip_ref.extractall(data_ingestion.data_unzip_dir)
    logger.info('data unziped ')

#create the data validation directory
data_validation = test_config.data_validation 

create_directories([data_validation.root_dir])
logger.info('data_validation directory created')
# check the data file exists 

schema = read_yaml(Path('schema.yaml'))
logger.info('data schema loaded ')
if os.path.exists(data_validation.data_local_file)  :
    data = pd.read_csv(data_validation.data_local_file)
    # check all the columns except the target 
    data_columns = data.columns[:-1]
    for column_name in data_columns :
        if column_name not in schema.COLUMNS.keys():
            logger.info(f'column dosenot exists in dataset {column_name}')
            with open(data_validation.data_validation_status_file, 'w') as file :
                file.write('Valid : False')
        else:
            with open(data_validation.data_validation_status_file, 'w') as file :
                file.write('Valid : True')
    logger.info('data validated')

# split the data and perform some of data transformation operation 
data_transformation = test_config.data_transformation
#create the data transformatin directory 
create_directories([data_transformation.root_dir])
if os.path.exists(data_validation.data_local_file)  :
    data = pd.read_csv(data_validation.data_local_file)

    # check the the data is valid or not 
    with open( data_validation.data_validation_status_file,'r' ) as file :
        # read the data 
        data_validation_status = file.read()
        if  'True' in data_validation_status :
            train, test = train_test_split(data, train_size=0.7, random_state=2)
            train.to_csv(os.path.join(data_transformation.root_dir, "train.csv"),index = False)
            test.to_csv(os.path.join(data_transformation.root_dir, "test.csv"),index = False)
            logger.info(f'Data splitted into the train and test \n test size {test.shape} train size {train.shape}')
        elif 'False':
            logger.info('Data is not validated')
        else:
            logger.info('Something wrong with the data')

        logger.info('data transformation completed')

logger.info('Model Training')
model_trainer = test_config.model_trainer
# create the directory for model trainer 
create_directories([model_trainer.root_dir])

params = read_yaml(Path('params.yaml'))
train_data = pd.read_csv(model_trainer.train_data_path)
test_data = pd.read_csv(model_trainer.test_data_path)

train_x = train_data.drop([schema.TARGET], axis=1)
test_x = test_data.drop([schema.TARGET], axis=1)
train_y = train_data[[schema.TARGET]]
test_y = test_data[[schema.TARGET]]

lr = ElasticNet(alpha=params.ElasticNet.alpha, l1_ratio=params.ElasticNet.l1_ratio, random_state=42)
lr.fit(train_x, train_y)
joblib.dump(lr, os.path.join(model_trainer.root_dir, model_trainer.model_name))
logger.info('Model Training is completed')

# model Evaluation
model_evaluation = test_config.model_evaluation
#create the data evaluvation directory 
create_directories([model_evaluation.root_dir])
test_data = pd.read_csv(model_evaluation.test_data_path)
test_x = test_data.drop([schema.TARGET], axis=1)
test_y = test_data[[schema.TARGET]]
logger.info('test data loaded')
model = joblib.load(model_evaluation.model_path)
y_pred = model.predict(test_x)

r2  = r2_score(test_y, y_pred)
mse  = mean_squared_error(test_y, y_pred)
metrics = {
    "r2":r2,
    "mse":mse
}

with open(model_evaluation.metric_file_name, 'w') as file :
    json.dump(metrics, file)
    
logger.info(f'model r2 : {r2} model mse : {mse}')
