# data_preprocessing
import sys
import argparse
import mlflow
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from loguru import logger


# set up logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')

def get_cancer_df():
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target)
    logger.info(f'Cancer data downloaded')
    return X, y


if __name__ == '__main__':
    
    TEST_SIZE = 0.33
    # get arguments if running not in ipykernel
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", default=TEST_SIZE, type=float)
    TEST_SIZE = parser.parse_args().test_size
        
    logger.info(f'Data preprocessing started with test size: {TEST_SIZE}')
    
    # create or use an experiment
    experiment_id = mlflow.set_experiment('Cancer_Classification').experiment_id
    
    with mlflow.start_run(run_name='Data_Preprocessing'):
            
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
        
        # log and register datasets
        train = X_train.assign(target=y_train)
        mlflow.log_text(train.to_csv(index=False),'datasets/train.csv')
        dataset_source_link = mlflow.get_artifact_uri('datasets/train.csv')
        dataset = mlflow.data.from_pandas(train, name='train', targets="target", source=dataset_source_link)
        mlflow.log_input(dataset)

        test = X_test.assign(target=y_test)
        mlflow.log_text(test.to_csv(index=False),'datasets/test.csv')
        dataset_source_link = mlflow.get_artifact_uri('datasets/test.csv')
        dataset = mlflow.data.from_pandas(train, name='test', targets="target", source=dataset_source_link)
        mlflow.log_input(dataset)
        
        logger.info('Data preprocessing finished')