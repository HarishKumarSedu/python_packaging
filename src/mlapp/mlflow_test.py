# mlflow imports 
import mlflow
import mlflow.sklearn

import pandas as pd 
from pathlib import Path
import urllib.request as request
import zipfile
import os 
import argparse

#sklearn imports 
from sklearn import metrics, ensemble
from sklearn.model_selection import train_test_split
from src.mlapp import logger

#
data_url = 'https://github.com/HarishKumarSedu/End_to_End_Ml/raw/main/artifacts/data_ingestion/data.zip'
local_data_path = 'data/winequalit-red.csv'
def data_ingestion(url=''):
    data_path = Path('data/data.zip')
    filename, header = request.urlretrieve(
        url=url,
        filename=data_path
    )
    logger.info(f'data downloaded with filename {filename}')
    data_status = False
    if os.path.exists(data_path) and (os.path.getsize(data_path) != 0 ) :
        unzip_dir,filename = os.path.split(data_path)
        with zipfile.ZipFile(data_path, 'r') as zip_ref :
            zip_ref.extractall(unzip_dir)

            data_status =  True

            logger.info(f'data {filename} extracted into {unzip_dir} folder')
    
    return data_status

def data_preporcessing(data:pd.DataFrame):

    train,test = train_test_split(data, train_size=0.7)
    logger.info(f'data splited \n\t train size {test.shape} test size {test.shape}')

    train.to_csv(os.path.join('data', "train.csv"),index = False)
    test.to_csv(os.path.join('data', "test.csv"),index = False)
    logger.info(f'train and test data stored ')
    return test, train

# Define the main 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="input data path", type=str)
    ### Extra parameters to operate the model 
    # parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
    # parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
    args = parser.parse_args()

    # Read the csv file
    try:
        if data_status := data_ingestion(url=data_url): # extract the data 
            data = pd.read_csv(args.data)
            # applyt the data preprocessing 
            train_data, test_data = data_preporcessing(data=data)
            
            #define the prameters 
            params = {"n_estimators": 5, "learning_rate": 0.1}
            model = ensemble.GradientBoostingClassifier(n_estimators=params.get('n_estimators'), learning_rate= params.get('learning_rate'))

        mlflow.set_experiment('test')
        with mlflow.start_run(run_name='test') as run :
            mlflow.log_params(params=params)
            # Train the model and log the metrics
            mlflow.log_artifacts('data')
            X_train = train_data.iloc[:,:-1]
            y_train = train_data['quality']

            X_test = test_data.iloc[:,:-1]
            y_test = test_data['quality']

            model.fit(X_train, y_train)
            predicted_probs = model.predict_proba(X_test)
           
            # training code ...
            # log experiment tags, parameters and result metrics
            logger.info(f'mlflow run id {run.info.experiment_id}')
            # roc_auc = metrics.roc_auc_score(y_test, predicted_probs)
            # mlflow.log_metric("test_auc", roc_auc)
            mlflow.set_tag("framework", "sklearn")
            # mlflow.log_param("dropout", args.dropout)
            # mlflow.log_param("lr", args.lr)
            # log data and model artifacts
            # Log the sklearn model and register as version 1
            mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-reg-model"
            )
    except Exception as e:
        raise ValueError(f"Unable to read the training CSV, {e}")


