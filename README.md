# Mlflow Local Workflow Example

Here we show a few steps of the machine learning lifecycle using an XGBoost example, focusing on integration with MLflow. We'll structure MLflow experiments and runs, perform hyperparameter optimizations with Optuna, and track all runs. We will use MLflow's capabilities to compare runs and adjust the best parameters. Additionally, we'll cover options to log the model and use it with different flavours, as well as ways to collaborate within a team without a remote tracking server.

> You have two options to work with this example: you can either clone it and follow all the steps on your local machine, or simply go through this README file.

> Our goal here is not to demonstrate best machine learning practices, but to highlight the capabilities of MLflow.

## Prepare env

You will need `Python 3.10.11` and `conda` installed on your machine firts. Then run the folllowing commands in you terminal. This script sets up the conda environment and converts `modeling.py` back to a Jupyter notebook (`modeling.ipynb`) using jupytext.

```shell
conda env create -f conda.yaml
conda activate mlflow-example
jupytext --to notebook modeling.py -o modeling.ipynb
```

You may then use this environment in IDE or just call `jupyter notebook` and then navigate to `modeling.ipynb`.

> Conda produce platform specific specification so you will probably need to install dependencies manually

> It's not necessary to create an environment with conda; you can also use poetry or manually install all packages from a `requirements.txt`. However, MLflow has built-in compatibility with conda, which is why we're using it here.

## Run MLflow UI
run from the terminal `mlflow ui --workers 1` to run MLflow with 1 worker on localhost:5000
> By default, MLflow creates the mlruns folder in the project root to store all project files. If this folder already exists with data, consider deleting it first: `rm -rf mlruns`

![](img/mlflow_main.png)


## Experiments and tuning
Here, we upload an open-source cancer dataset, build a classification model, use Optuna for hyperparameter selection with cross-validation, and log metrics and all steps in MLflow.

### Data preparation
This step is usually more complicated, but here we'll just download the dataset, split it into training and testing sets, log a few metrics into MLflow (like the number of samples and features), and pass the datasets themselves to MLflow artifacts.


```python
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
    # hide parser = argparse.ArgumentParser()
    # hide parser.add_argument("--test-size", default=config.default_test_size, type=float)
    # hide TEST_SIZE = parser.parse_args().test_size
        

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
```

    2024-07-12 00:39:06 | INFO | Data preprocessing started
    2024-07-12 00:39:06 | INFO | Cancer data downloaded
    2024-07-12 00:39:06 | INFO | Additional features added
    2024-07-12 00:39:06 | INFO | train uploaded
    2024-07-12 00:39:06 | INFO | test uploaded
    2024-07-12 00:39:06 | INFO | Data preprocessing finished


### Hyperparameters tuning


In this part, we use Optuna to find the best hyperparameters for XGBoost, leveraging its built-in cross-validation for training and evaluation. Additionally, we'll show how to track metrics during the model fitting process using a custom callback.


```python
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
    # hide parser = argparse.ArgumentParser()
    # hide parser.add_argument("--n-trials", default=config.default_n_trials, type=float)
    # hide N_TRIALS = parser.parse_args().n_trials

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
```

    2024-07-12 00:39:09 | INFO | Hyperparameters tuning started



    Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]


    2024-07-12 00:39:09 | INFO | train downloaded
    2024-07-12 00:39:09 | INFO | Starting optuna study
    2024-07-12 00:39:10 | INFO | Attempt: 0, Accuracy: 0.9553805774278216
    2024-07-12 00:39:10 | INFO | Attempt: 1, Accuracy: 0.9553805774278216
    2024-07-12 00:39:11 | INFO | Attempt: 2, Accuracy: 0.9606299212598425
    2024-07-12 00:39:11 | INFO | Attempt: 3, Accuracy: 0.9606299212598425
    2024-07-12 00:39:11 | INFO | Attempt: 4, Accuracy: 0.963254593175853
    2024-07-12 00:39:12 | INFO | Attempt: 5, Accuracy: 0.958005249343832
    2024-07-12 00:39:12 | INFO | Attempt: 6, Accuracy: 0.9422572178477691
    2024-07-12 00:39:12 | INFO | Attempt: 7, Accuracy: 0.9448818897637795
    2024-07-12 00:39:13 | INFO | Attempt: 8, Accuracy: 0.958005249343832
    2024-07-12 00:39:13 | INFO | Attempt: 9, Accuracy: 0.9606299212598425
    2024-07-12 00:39:13 | INFO | Optimization finished, best params: {'objective': 'binary:logistic', 'max_depth': 4, 'alpha': 0.03623937669892902, 'learning_rate': 0.45680940256230484, 'num_boost_round': 156}
    2024-07-12 00:39:13 | INFO | Best trial Accuracy: 0.963254593175853


### Review experiment resulst from MLflow UI

![](img/runs_list.png)

*Here, we use nested run capabilities to organize our runs within experiments with custom columns and ordering*

![](img/runs_charts.png)

*We can use the chart view to compare runs and set up different plots. Here, we use XGBoost callbacks to log metrics during the model fitting process, allowing us to create plots with the number of trees on the x-axis.*

![](img/compare_runs_counter_plot.png)

*MLflow has an amazing feature to compare runs. To do this, you can select a few runs and push the compare button. Then, select the most useful view. This feature is especially valuable when trying to find optimal hyperparameters because you can guess and adjust the boundaries of possible intervals based on the comparison results.*

![Relative Path Image](img/system_metrics.png)

*We can also track system metrics throughout the run. While this may not provide an exact estimate of the real project requirements, it can still be useful in some cases.*


### Final model training

It is possible, but not necessary, to save the model for each experiment and run. In most scenarios, it is better to save the parameters and then, once the final parameters are selected, perform an additional run to save the model. Here, we follow the same logic: we use the parameters from the best run to save the final model and register it for versioning and usage via a short link.


```python
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
```

    2024-07-12 00:39:16 | INFO | Model training started



    Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]


    2024-07-12 00:39:16 | INFO | train downloaded



    Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]


    2024-07-12 00:39:16 | INFO | test downloaded


    Successfully registered model 'CancerModelBooster'.
    Created version '1' of model 'CancerModelBooster'.


    2024-07-12 00:39:21 | INFO | train uploaded
    2024-07-12 00:39:21 | INFO | test uploaded
    2024-07-12 00:39:21 | INFO | Model training finished
    2024-07-12 00:39:21 | INFO | Model registered


    Successfully registered model 'CancerModel'.
    Created version '1' of model 'CancerModel'.


After saving a model, we can access the artifacts page within the run and also check the model using the model tracking UI. We can save as many artifacts as we want, such as custom plots, text files, images, datasets, python scripts, or jupyter notebooks.

![](img/model_artifacts.png)

*Here, we have the model's input and output specifications, as well as the environment specification files used to create the model. This ensures the model will work as expected later on. Additionally, we manually add the training and test datasets to enable fully repeatable results. XGBoost autologging also adds feature importance as a plot and image.*

![](img/model_metrics.png)

*Thanks to the `mlflow.xgboost.autolog()` feature, we have all parameters and metrics available without manual steps, callbacks, or additional configuration.*

### Built in evaluation

We can leverage MLflow's built-in ability to evaluate additional datasets, even if they were not available during training. By using the evaluate method, we can also enhance it by logging additional metrics if desired.


```python
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
    # hide parser = argparse.ArgumentParser()
    # hide parser.add_argument("--eval-dataset", type=str)
    # hide eval_dataset = pd.read_csv(parser.parse_args().eval_dataset)
        
    with mlflow.start_run(run_name=config.data_evaluation_run_name) as run:
        
        eval_dataset = mlflow.data.from_pandas(
            eval_dataset, targets="target"
        )
        last_version = mlflow.MlflowClient().get_registered_model(config.registered_model_name).latest_versions[0].version
        mlflow.evaluate(
            data=eval_dataset, model_type="classifier", model=f'models:/{config.registered_model_name}/{last_version}'
        )
        logger.success('Evaluation finished')
```

    2024-07-12 00:39:26 | INFO | Evaluation started



    Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]


    2024-07-12 00:39:26 | INFO | test downloaded
    2024-07-12 00:39:35 | SUCCESS | Evaluation finished


> Here we utilize `MLflow Dataset` functionality. Datasets allow tracking metadata for files in runs, including columns, hash, and source link. We can also run the `mlflow.evaluate` command for an already created `MLflow dataset`.

We can then observe the results in the `mlflow ui`. Here, mlflow generates numerous metrics and plots for us, including `roc-auc`, `confusion matrices`, and `shap plots` (if SHAP is installed)

![](img/shap_plot.png)


## MLflow Projects

The next step might be to share your project with other data scientists or to automate the running of your model training pipeline. This is where MLflow Projects come in.

> An MLflow Project packages data science code in a reusable and reproducible way, following conventions that make it easy for others (or automated tools) to run. Each project is a directory of files or a Git repository containing your code. MLflow can run these projects based on specific conventions for organizing files in the directory. 

First, we'll convert our main code cells into Python files using the `nbformat` library. We'll create a separate Python file for each cell, based on comment lines at the top of each cell that specify the predefined names for the files.


```python
import nbformat

from config import logger

def extract_and_save_cell(notebook_path, comment):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Search for the cell that starts with the specific comment
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.source.strip().startswith(f'# {comment}'):
            code_content = cell.source.strip()
            break
    else:
        raise ValueError(f"No cell starting with comment '{comment}' found in the notebook.")

    # Process each line to remove "# hide" and convert it to plain text while preserving indentation
    processed_lines = []
    for line in code_content.splitlines():
        stripped_line = line.lstrip()
        if stripped_line.startswith('# hide'):
            # Calculate leading whitespace
            leading_whitespace = len(line) - len(stripped_line)
            # Remove '# hide' and keep the leading whitespace
            new_line = ' ' * leading_whitespace + stripped_line.replace('# hide', '', 1).strip()
            processed_lines.append(new_line)
        else:
            processed_lines.append(line)

    # Join the processed lines back into a single string
    processed_content = '\n'.join(processed_lines)

    # Save the extracted and processed content to a Python file
    with open(f'{comment}.py', 'w', encoding='utf-8') as f:
        f.write(processed_content)

    logger.info(f'{comment}.py saved')


if __name__ == '__main__':
    for comment in ['data_preprocessing', 'hyperparameters_tuning', 'model_training', 'data_evaluation']:
        extract_and_save_cell('modeling.ipynb', comment)
```

    2024-07-12 01:40:11 | INFO | data_preprocessing.py saved
    2024-07-12 01:40:11 | INFO | hyperparameters_tuning.py saved
    2024-07-12 01:40:11 | INFO | model_training.py saved
    2024-07-12 01:40:11 | INFO | data_evaluation.py saved


### Conda env export
We already have a conda.yaml file, but its creation can be complex and require manual steps. Exporting the current environment with `conda env export | grep -v '^prefix: ' > conda.yaml` may not be suitable for sharing or `Docker` use due to platform-specific issues. Adding the `--from-history` flag lists only explicitly requested packages but may fail with pip-installed packages. Using `pip freeze` includes local package links. Thus, manually creating a `requirements.txt` file or `conda.yaml` might be the best solution.

### MLproject file

The `MLproject` file helps MLflow and others understand and run your project by specifying the environment, entry points, and possible parameters for customization. Let's review the `MLproject` created for this project


```python
from IPython.display import Markdown, display
with open('MLproject', 'r') as file:
    mlproject_content = file.read()

# Display the contents as a Markdown code snippet
display(Markdown(f"```yaml\n{mlproject_content}\n```"))
```


```yaml
name: Cancer_Modeling

conda_env: conda.yaml

entry_points:
  data-preprocessing:
    parameters:
      test-size: {type: float, default: 0.33}
    command: "python data_preprocessing.py --test-size {test-size}"
  hyperparameters-tuning:
    parameters:
      n-trials: {type: int, default: 10}
    command: "python hyperparameters_tuning.py --n-trials {n-trials}"
  model-training:
    command: "python model_training.py"
  data-evaluation:
    parameters:
      eval-dataset: {type: str}
    command: "python data_evaluation.py --eval-dataset {eval-dataset}"

```


### Mlflow run
We can run project endpoints using either the CLI or the Python API. In this instance, we run the first endpoint with different test size parameters in a local environment.

> The `mlflow run` command sets the experiment and creates a run before executing the python script. Therefore, if we use the same commands inside our Python code with specified names, it is important to use the same names in this command.


```python
mlflow.run(
    uri = '.',
    entry_point = 'data-preprocessing',
    env_manager='local',
    experiment_name=config.experiment_name,
    run_name=config.data_preprocessing_run_name,
    parameters={'test-size': 0.5},
)
```

    2024-07-12 00:39:52 | INFO | Data preprocessing started
    2024-07-12 00:39:52 | INFO | Cancer data downloaded
    2024-07-12 00:39:52 | INFO | Additional features added
    2024-07-12 00:39:52 | INFO | train uploaded
    2024-07-12 00:39:52 | INFO | test uploaded
    2024-07-12 00:39:52 | INFO | Data preprocessing finished





    <mlflow.projects.submitted_run.LocalSubmittedRun at 0x168865cd0>



Here, we run a second endpoint with a conda environment, creating an additional conda environment. We can verify its creation using the conda env list command.


```python
mlflow.run(
    uri = '.',
    entry_point = 'hyperparameters-tuning',
    env_manager='conda',
    experiment_name=config.experiment_name,
    run_name=config.hyperparameter_search_run_name,
    parameters={'n-trials': 3},
)
```

    Channels:
     - conda-forge
     - defaults
    Platform: osx-arm64
    Collecting package metadata (repodata.json): ...working... done
    Solving environment: ...working... done
    Preparing transaction: ...working... done
    Verifying transaction: ...working... done
    Executing transaction: ...working... done
    Installing pip dependencies: ...working... done
    2024-07-12 00:40:51 | INFO | Hyperparameters tuning started
    2024-07-12 00:40:51 | INFO | train downloaded
    2024-07-12 00:40:51 | INFO | Starting optuna study


    Downloading artifacts: 100%|██████████| 1/1 [00:00<00:00, 2711.25it/s]


    2024-07-12 00:40:51 | INFO | Attempt: 0, Accuracy: 0.9541246733855916
    2024-07-12 00:40:52 | INFO | Attempt: 1, Accuracy: 0.9399402762224711
    2024-07-12 00:40:53 | INFO | Attempt: 2, Accuracy: 0.9576334453154162
    2024-07-12 00:40:53 | INFO | Optimization finished, best params: {'objective': 'binary:logistic', 'max_depth': 3, 'alpha': 0.03348833961166779, 'learning_rate': 0.31237953375320565, 'num_boost_round': 134}
    2024-07-12 00:40:53 | INFO | Best trial Accuracy: 0.9576334453154162





    <mlflow.projects.submitted_run.LocalSubmittedRun at 0x17e7e7c10>




```python
mlflow.run(
    uri = '.',
    entry_point = 'model-training',
    env_manager='conda',
    experiment_name=config.experiment_name,
    run_name=config.training_run_name,
)
```

    2024-07-12 00:40:59 | INFO | Model training started
    2024-07-12 00:41:00 | INFO | train downloaded
    2024-07-12 00:41:00 | INFO | test downloaded


    Downloading artifacts: 100%|██████████| 1/1 [00:00<00:00, 3597.17it/s]
    Downloading artifacts: 100%|██████████| 1/1 [00:00<00:00, 1844.46it/s]
    Registered model 'CancerModelBooster' already exists. Creating a new version of this model...
    Created version '2' of model 'CancerModelBooster'.


    2024-07-12 00:41:05 | INFO | train uploaded
    2024-07-12 00:41:05 | INFO | test uploaded
    2024-07-12 00:41:05 | INFO | Model training finished
    2024-07-12 00:41:05 | INFO | Model registered


    Registered model 'CancerModel' already exists. Creating a new version of this model...
    Created version '2' of model 'CancerModel'.





    <mlflow.projects.submitted_run.LocalSubmittedRun at 0x168dba410>




```python
# get data
last_data_run_id = get_last_run(experiment_id, config.data_preprocessing_run_name).run_id
test = download_dataset_as_artifact(last_data_run_id, 'test')
test.to_csv('test.csv')
path = str(config.project_root / 'test.csv')

mlflow.run(
    uri = '.',
    entry_point = 'data-evaluation',
    env_manager='conda',
    experiment_name=config.experiment_name,
    run_name=config.data_evaluation_run_name,
    parameters={'eval-dataset': path},
)
```


    Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]


    2024-07-12 00:41:07 | INFO | test downloaded
    2024-07-12 00:41:09 | INFO | Evaluation started
    2024-07-12 00:41:09 | INFO | test downloaded


    Downloading artifacts: 100%|██████████| 1/1 [00:00<00:00, 5949.37it/s]


    2024-07-12 00:41:24 | SUCCESS | Evaluation finished





    <mlflow.projects.submitted_run.LocalSubmittedRun at 0x17fae4b90>



### Docker setup

`Dockerfile` and `docker-compose` stored in the mlproject_docker folder. Docker image based on slim Python and install dependencies from a manually created `requirments.txt`. `docker-compose` mounts working directory to a volume to access files and log all MLflow activities in the `mlruns` folder. In a production environment, we might run this command from an orchestration tool and provide the MLFLOW_TRACKING_URI to a remote MLflow server. You can run `docker-compose -f mlproject_docker/docker-compose.yml build` to build the image and then `docker-compose -f mlproject_docker/docker-compose.yml` up to run it.

> We need to mount the absolute path to the `mlruns` folder in the project root to log and retrieve artifacts. This is necessary because local MLflow tracking uses absolute paths.


```python
# Change error reporting mode to minimal
%xmode Minimal
```

    Exception reporting mode: Minimal


## MLmodel: flavours

In MLflow, models can be loaded using different flavors specified in the `MLmodel` file. We save two versions of the model both with xgboost, each have two flavors: `python_function` and `xgboost`. The difference lies in the model class: for the `booster`, it is `xgboost.core.Booster`, and for the `model`, it is `xgboost.sklearn.XGBClassifier`, which supports a scikit-learn compatible API. These two cases differ in how the predict method works, so it is important to review the MLmodel file and check the model signature before using it.


When loading the `booster` model with the `xgboost`, the model expects the input data to be in the form of a `DMatrix` object and `predict` method will produce scores (not classes) in our case.


```python
import mlflow
import xgboost as xgb

from config import config, logger
from utils import get_last_run, download_dataset_as_artifact

# prepare data
experiment_id = mlflow.set_experiment(config.experiment_name).experiment_id
last_data_run_id = get_last_run(experiment_id, config.data_preprocessing_run_name).run_id
test = download_dataset_as_artifact(last_data_run_id, 'test')
features = [i for i in test.columns if i != 'target']
dtest = xgb.DMatrix(data=test.loc[:, features], label=test['target'])
test.drop('target', axis=1, inplace=True)

# download booster with xgboost flavour
logged_model = 'models:/CancerModelBooster/1'
xgboost_booster = mlflow.xgboost.load_model(logged_model)
```


    Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]


    2024-07-12 00:59:08 | INFO | test downloaded



```python
# error with pandas input
xgboost_booster.predict(test)
```


    TypeError: ('Expecting data to be a DMatrix object, got: ', <class 'pandas.core.frame.DataFrame'>)




```python
# work with DMatrix like predict proba
xgboost_booster.predict(dtest)[:3]
```




    array([0.9881982 , 0.9896479 , 0.01500983], dtype=float32)




```python
# download booster with pyfunc flavour
pyfunc_booster = mlflow.pyfunc.load_model(logged_model)
```


```python
# work with pandas, produce probs
pyfunc_booster.predict(test)[:3]
```




    array([0.9881982 , 0.9896479 , 0.01500983], dtype=float32)




```python
# error with DMatrix
pyfunc_booster.predict(dtest)
```


    MlflowException: Expected input to be DataFrame. Found: DMatrix


    
    During handling of the above exception, another exception occurred:


    MlflowException: Failed to enforce schema of data '<xgboost.core.DMatrix object at 0x168fe3bd0>' with schema '['mean radius': double (required), 'mean texture': double (required), 'mean perimeter': double (required), 'mean area': double (required), 'mean smoothness': double (required), 'mean compactness': double (required), 'mean concavity': double (required), 'mean concave points': double (required), 'mean symmetry': double (required), 'mean fractal dimension': double (required), 'radius error': double (required), 'texture error': double (required), 'perimeter error': double (required), 'area error': double (required), 'smoothness error': double (required), 'compactness error': double (required), 'concavity error': double (required), 'concave points error': double (required), 'symmetry error': double (required), 'fractal dimension error': double (required), 'worst radius': double (required), 'worst texture': double (required), 'worst perimeter': double (required), 'worst area': double (required), 'worst smoothness': double (required), 'worst compactness': double (required), 'worst concavity': double (required), 'worst concave points': double (required), 'worst symmetry': double (required), 'worst fractal dimension': double (required), 'additional_feature': double (required)]'. Error: Expected input to be DataFrame. Found: DMatrix




```python
# but we can still reach booster object and use it with DMatrix
pyfunc_booster._model_impl.xgb_model.predict(dtest)[:3]
```




    array([0.9881982 , 0.9896479 , 0.01500983], dtype=float32)



Let's examine with `xgboost.sklearn.XGBClassifier`


```python
logged_model = 'models:/CancerModel/1'
xgboost_model = mlflow.xgboost.load_model(logged_model)
```


```python
# predict method produce classes not probs - work with pandas
xgboost_model.predict(test)[:3]
```




    array([1, 1, 0])




```python
# not able to work with DMatrix
xgboost_model.predict(dtest)[:3]
```


    TypeError: Not supported type for data.<class 'xgboost.core.DMatrix'>




```python
pyfunc_model = mlflow.pyfunc.load_model(logged_model)
```

### Performace comparison
Since the pyfunc model has some additional overhead, performance is slightly worse in our case. However, this can vary based on the data and model, and the difference may be greater.


```python
%%timeit
xgboost_booster.predict(xgb.DMatrix(test))
```

    1.66 ms ± 67.7 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%%timeit
pyfunc_booster.predict(test)
```

    1.92 ms ± 46.3 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%%timeit
xgboost_model.predict(test)
```

    1.5 ms ± 8.89 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)



```python
%%timeit
pyfunc_model.predict(test)
```

    1.81 ms ± 31.7 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


## Model Serving
MLflow has built-in capabilities to serve models locally (though it also integrates with SageMaker and Kubernetes, which we won't cover here). Serving a model with flask is pretty straightforward: `mlflow models serve -m models:/CancerModel/1 -p 5050 --env-manager local`. But we also may utilize `mlserver` and to do it properly we may first install `mlserver`, `mlserver-mlflow`, `mlserver-xgboost` and create json config file.

You may also start from docker setup: `docker-compose -f docker/mlserve/docker-compose.yml build` and `docker-compose -f docker/mlserve/docker-compose.yml up`



```python
from IPython.display import Markdown, display
with open('docker/mlserve/model-settings.json', 'r') as file:
    content = file.read()

# Display the contents as a Markdown code snippet
display(Markdown(f"```json\n{content}\n```"))
```


```json
{
  "name": "cancer-model",
  "implementation": "mlserver_mlflow.MLflowRuntime",
  "parameters": {
    "uri": "models:/CancerModel/1"
  }
}
```


Then, we can run mlserver start and it's done. After that, we can check the documentation for our model and inspect the expected data structure via swagger:

![](img/mlserver.png)

And query it with our dataset:


```python
import requests
import json

url = "http://127.0.0.1:8080/invocations"

# convert df do split format and then to json
input_data = json.dumps({
    'dataframe_split': {
        "columns": test.columns.tolist(),
        "data": test.values.tolist()
    }
})

# Send a POST request to the MLflow model server
response = requests.post(url, data=input_data, headers={"Content-Type": "application/json"})

if response.status_code == 200:
    prediction = response.json()
    print("Prediction:", prediction)
else:
    print("Error:", response.status_code, response.text)
```

    Prediction: {'predictions': [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]}


We already serve the model and can query it!

### Customize model serving
We may then decide to customize our model, for example, to respond with probabilities and add additional logging. We can implement this using a custom wrapper.


```python
# Step 1: Download the Existing Model from MLflow
model_uri = "models:/CancerModel/1"
model = mlflow.xgboost.load_model(model_uri)

# Step 2: Create a Custom Wrapper Around the Model
class CustomModelWrapper:
    
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        probabilities = self.model.predict_proba(X)[:,1]
        return probabilities

# Wrap the downloaded model
custom_model = CustomModelWrapper(model)

# Step 3: Define the Custom PyFunc Model with `loguru` Setup in `load_context`
class CustomPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        pass


    def predict(self, context, model_input):
        probabilities = self.model.predict(model_input)
        return probabilities

# Step 4: Save the Wrapped Model Back to MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="custom_model",
        python_model=CustomPyFuncModel(custom_model),
        registered_model_name="CustomCancerModel"
    )
```

    Successfully registered model 'CustomCancerModel'.
    Created version '1' of model 'CustomCancerModel'.


After changing the model name in the config and rerunning the container, we may observe that the model starts responding with probabilities


```python
import requests
import json

url = "http://127.0.0.1:8080/invocations"

# convert df do split format and then to json
input_data = json.dumps({
    'dataframe_split': {
        "columns": test.columns.tolist(),
        "data": test.values.tolist()
    }
})

# Send a POST request to the MLflow model server
response = requests.post(url, data=input_data, headers={"Content-Type": "application/json"})

if response.status_code == 200:
    prediction = response.json()
    print("Prediction:", prediction['predictions'][:10])
else:
    print("Error:", response.status_code, response.text)
```

    Prediction: [0.9957305788993835, 0.9979894161224365, 0.004697034601122141, 0.9994956254959106, 0.9846009612083435, 0.019780341535806656, 0.006149278022348881, 0.9992685914039612, 0.9983139038085938, 0.9988442659378052]


> While this method works, it might be more straightforward to set up a custom web server if we want to incorporate more complex logic rather than using the built-in tools.

## Other tricks
A better option for collaboration is to set up a remote MLflow tracking server to work together. However, if that's not possible, you can share your `mlruns` folder with colleagues and use it together to share experiments. You can add it to a git repository if it doesn't contain sensitive information. If you use different experiments and models, it will probably not cause any issues. However, keep in mind that this approach might lead to conflicts and may require additional communication and rules within the team.

You may also change your local tracking folder if you would like to have multiple separate MLflow instances. Let's see how it works for a local setup.

1. Start the MLflow UI with the new folder as the backend store: `poetry run mlflow ui --backend-store-uri ./mlruns_new --workers 1`
2. chnage tracking uri in you code: `mlflow.set_tracking_uri("file:./mlruns_new")`

Then, check the MLflow UI to ensure it not contains the experiment and model information.


```python
# create README.md based on this notebook
import nbformat
from nbconvert import MarkdownExporter


def process_markdown(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    with open(file_name, 'w') as f:
        inside_code_block = False
        for line in lines:
            if line.startswith('```'):
                if inside_code_block:
                    inside_code_block = False
                    f.write(line)
                else:
                    if line.strip() == '```':
                        f.write('```python\n')
                    else:
                        f.write(line)
                    inside_code_block = True
            else:
                f.write(line)


if __name__ == '__main__':
    
    # Convert to Markdown
    markdown_exporter = MarkdownExporter()
    markdown_body, markdown_resources = markdown_exporter.from_filename('modeling.ipynb')
    with open('README.md', 'w') as f:
        f.write(markdown_body)

    process_markdown('README.md')
```


```python
# convert notebook to python
import jupytext
notebook = jupytext.read('modeling.ipynb')
jupytext.write(notebook, 'modeling.py')
```
