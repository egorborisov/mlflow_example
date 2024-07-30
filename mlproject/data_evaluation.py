# data_evaluation
import sys
import os
import argparse
import warnings
import logging
import mlflow
import pandas as pd
from loguru import logger


logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)


if __name__ == '__main__':

    logger.info('Evaluation started')

    experiment_id = mlflow.set_experiment('Cancer_Classification').experiment_id
    
    if 'test.csv' in os.listdir():
        eval_dataset = pd.read_csv('test.csv')
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=str)
    eval_dataset = pd.read_csv(parser.parse_args().eval_dataset)
        
    with mlflow.start_run(run_name='Data_Evaluation') as run:
        
        eval_dataset = mlflow.data.from_pandas(
            eval_dataset, targets="target"
        )
        last_version = mlflow.MlflowClient().get_registered_model('CancerModel').latest_versions[0].version
        mlflow.evaluate(
            data=eval_dataset, model_type="classifier", model=f'models:/CancerModel/{last_version}'
        )
        logger.success('Evaluation finished')