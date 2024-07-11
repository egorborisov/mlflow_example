# hyperparameters_tuning
import argparse
import logging
import warnings
import mlflow
import optuna
import xgboost as xgb
from xgboost.callback import TrainingCallback

from config import config, logger
from utils import download_dataset_as_artifact, get_last_run


# Custom callback for logging metrics
class LoggingCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        for metric_name, metric_vals in evals_log['test'].items():
            mlflow.log_metric(f"{metric_name}", metric_vals[-1][0], step=epoch)
        return False


# Define an objective function for Optuna
def objective(trial):
    global dtrain

    # hyperparameters
    params = {
        "objective": trial.suggest_categorical('objective', ['binary:logistic']),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "alpha": trial.suggest_float("alpha", 0.001, 0.05),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
        "num_boost_round": trial.suggest_int("num_boost_round", 30, 300),
    }

    with mlflow.start_run(nested=True):

        mlflow.log_params(params)
        params.update(eval_metric=['auc', 'error'])
        num_boost_round = params["num_boost_round"]
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=3,
            callbacks=[LoggingCallback()],
            verbose_eval=False,
        )
        
        error = cv_results['test-error-mean'].iloc[-1]
        mlflow.log_metric("accuracy", (1 - error))
        logger.info(f"Attempt: {trial.number}, Accuracy: {1 - error}")

        return error


if __name__ == '__main__':

    N_TRIALS = config.default_n_trials
    # get arguments if running not in ipykernel
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", default=config.default_n_trials, type=float)
    N_TRIALS = parser.parse_args().n_trials

    # set up logging
    warnings.filterwarnings('ignore')
    logging.getLogger('mlflow').setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    logger.info('Hyperparameters tuning started')

    # start experiment
    experiment_id = mlflow.set_experiment(config.experiment_name).experiment_id

    with mlflow.start_run(run_name=config.hyperparameter_search_run_name, log_system_metrics=True):
  
        last_run = get_last_run(experiment_id, config.data_preprocessing_run_name)
        last_run_id = last_run.run_id
        
        train = download_dataset_as_artifact(last_run_id, 'train')

        # convert to DMatrix format
        features = [i for i in train.columns if i != 'target']
        dtrain = xgb.DMatrix(data=train.loc[:, features], label=train['target'])
        
        logger.info('Starting optuna study')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=N_TRIALS)
        best_trial = study.best_trial
        
        logger.info(f'Optimization finished, best params: {best_trial.params}')
        mlflow.log_params(best_trial.params)
        
        logger.info(f'Best trial Accuracy: {1 - best_trial.value}')
        mlflow.log_metric('accuracy', 1 - study.best_value)