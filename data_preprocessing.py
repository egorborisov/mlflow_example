# data_preprocessing
import argparse
import mlflow
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

from config import config, logger
from utils import upload_dataset_as_artifact


def get_cancer_df():
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target)
    logger.info(f'Cancer data downloaded')
    return X, y


if __name__ == '__main__':

    TEST_SIZE = config.default_test_size
    # get arguments if running not in ipykernel
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", default=config.default_test_size, type=float)
    TEST_SIZE = parser.parse_args().test_size
        

    warnings.filterwarnings('ignore')
    
    logger.info('Data preprocessing started')
    
    # create or use an experiment
    experiment_id = mlflow.set_experiment(config.experiment_name).experiment_id
    
    with mlflow.start_run(run_name=config.data_preprocessing_run_name):
            
        # download cancer dataset
        X, y = get_cancer_df()
    
        # add additional features
        X['additional_feature'] = X['mean symmetry'] / X['mean texture']
        logger.info('Additional features added')
    
        # log dataset size and features count
        mlflow.log_metric('full_data_size', X.shape[0])
        mlflow.log_metric('features_count', X.shape[1])
    
        # split dataset to train and test part and log sizes to mlflow
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        mlflow.log_metric('train_size', X_train.shape[0])
        mlflow.log_metric('test_size', X_test.shape[0])
        
        # log datasets
        upload_dataset_as_artifact(X_train.assign(target=y_train), 'train')
        upload_dataset_as_artifact(X_test.assign(target=y_test), 'test')
        
        logger.info('Data preprocessing finished')