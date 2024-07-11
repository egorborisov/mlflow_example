import sys
from loguru import logger
from pathlib import Path
from pydantic import BaseSettings


class Config(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent
    mlflow_project: Path = project_root / 'mlflow_project'
    experiment_name: str = 'Cancer_Classification'
    data_preprocessing_run_name: str = 'Data_Preprocessing'
    hyperparameter_search_run_name: str = 'Hyperparameters_Search'
    training_run_name: str = 'Model_Training'
    data_evaluation_run_name: str = 'Data_Evaluation'
    registered_model_name: str = 'CancerModel'
    default_test_size: float = 0.33
    default_n_trials: int = 10


# start config
config = Config()

# set up logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


if __name__ == '__main__':
    import pydantic
    print(config)
    print(pydantic.version)