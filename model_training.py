# model_training
import mlflow
import warnings
import logging
import xgboost as xgb
from mlflow.models import infer_signature

from config import config, logger
from utils import download_dataset_as_artifact, get_last_run, upload_dataset_as_artifact


if __name__ == '__main__':

    # set up logging
    warnings.filterwarnings('ignore')
    logging.getLogger('mlflow').setLevel(logging.ERROR)

    logger.info('Model training started')
    
    mlflow.xgboost.autolog()

    experiment_id = mlflow.set_experiment(config.experiment_name).experiment_id

    with mlflow.start_run(run_name=config.training_run_name) as run:
        
        # get data
        last_data_run_id = get_last_run(experiment_id, config.data_preprocessing_run_name).run_id
        train = download_dataset_as_artifact(last_data_run_id, 'train')
        test = download_dataset_as_artifact(last_data_run_id, 'test')

        # convert to DMatrix format
        features = [i for i in train.columns if i != 'target']
        dtrain = xgb.DMatrix(data=train.loc[:, features], label=train['target'])
        dtest = xgb.DMatrix(data=test.loc[:, features], label=test['target'])
        
        # get params
        last_tuning_run = get_last_run(experiment_id, config.hyperparameter_search_run_name)
        params = {col.split('.')[1]: last_tuning_run[col] for col in last_tuning_run.index if 'params' in col}
        params.update(eval_metric=['auc', 'error'])

        mlflow.log_params(params)
        
        model = xgb.train(
            dtrain=dtrain,
            num_boost_round=int(params["num_boost_round"]),
            params=params,
            evals=[(dtest, 'test')],
            verbose_eval=False,
            early_stopping_rounds=10
        )

        mlflow.log_metric("accuracy", 1 - model.best_score)

        signature = infer_signature(test.loc[:,features], model.predict(dtest))
        mlflow.xgboost.log_model(model, "booster", signature=signature)

        # Register xgboost model
        model_uri = f"runs:/{run.info.run_id}/booster"
        mlflow.register_model(model_uri, config.registered_model_name + 'Booster')
        
        params.update(num_boost_round=model.best_iteration)
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(train.loc[:, features], train['target'])
        signature = infer_signature(test.loc[:, features], final_model.predict(test.loc[:, features]))
            
        mlflow.xgboost.log_model(final_model, "model", signature=signature)

        # Log the datasets as artifacts
        upload_dataset_as_artifact(train, 'train')
        upload_dataset_as_artifact(test, 'test')

        logger.info('Model training finished')

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, config.registered_model_name)

        logger.info('Model registered')