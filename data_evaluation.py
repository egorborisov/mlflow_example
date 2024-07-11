# data_evaluation
import argparse
import warnings
import logging
import mlflow
import pandas as pd

from utils import get_last_run, download_dataset_as_artifact
from config import config, logger

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    logging.getLogger('mlflow').setLevel(logging.ERROR)
    
    logger.info('Evaluation started')

    experiment_id = mlflow.set_experiment(config.experiment_name).experiment_id

    last_data_run_id = get_last_run(experiment_id, config.data_preprocessing_run_name).run_id
    eval_dataset = download_dataset_as_artifact(last_data_run_id, 'test')
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=str)
    eval_dataset = pd.read_csv(parser.parse_args().eval_dataset)
        
    with mlflow.start_run(run_name=config.data_evaluation_run_name) as run:
        
        eval_dataset = mlflow.data.from_pandas(
            eval_dataset, targets="target"
        )
        last_version = mlflow.MlflowClient().get_registered_model(config.registered_model_name).latest_versions[0].version
        mlflow.evaluate(
            data=eval_dataset, model_type="classifier", model=f'models:/{config.registered_model_name}/{last_version}'
        )
        logger.success('Evaluation finished')