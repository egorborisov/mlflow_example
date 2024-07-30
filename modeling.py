# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Introduction to Mlflow
#
# This guide demonstrates key steps in the machine learning lifecycle using an `XGBoost` example, 
# focusing on integration with `MLflow`. The process includes structuring MLflow experiments and runs, 
# performing hyperparameter optimization with `Optuna`, and tracking all runs. Leveraging MLflow's capabilities, 
# it shows how to compare runs and fine-tune parameters for optimal performance. Additionally, 
# it explores options for logging the model and utilizing it with different flavors, 
# as well as covering MLproject packaging and MLflow's model serving capabilities.
#
# > Two options are available for working with this example: either clone it and follow all the steps on your local machine, or simply review this `README`.

# ## Prepare env
#
# Ensure conda is installed on your machine. Then, execute the following commands in your terminal. 
# This script sets up the conda environment from conda.yaml, activates it, and converts modeling.py 
# back to a Jupyter notebook (modeling.ipynb) using jupytext.
#
# ```bash
# conda env create -f conda.yaml
# conda activate mlflow-example
# jupytext --to notebook modeling.py -o modeling.ipynb
# ```
#
# This environment can be used in an IDE or by calling `jupyter notebook` and navigating to `modeling.ipynb`.

# ## Run MLflow UI
# Start MLflow with one worker on localhost:5000 by running `mlflow ui` in the terminal. Then check it in your browser.
#
# ![](img/mlflow_main.png)
#

# ## Modeling
#
# We will upload an open-source cancer dataset, develop a classification model, utilize `Optuna` for hyperparameter optimization with cross-validation, and log metrics along with all procedural steps in `MLflow`.

# ### Data preparation
# This step is usually more complicated, but here we will simply download the dataset, split it into training and testing sets, log a few metrics into MLflow (such as the number of samples and features), and pass the datasets themselves to MLflow artifacts.
# > MLflow tracks your modeling with the concepts of [runs and experiments](https://mlflow.org/docs/latest/tracking.html). `Runs` represent executions of data science code, recording metadata (metrics, parameters, start and end times) and artifacts (output files like model weights and images). `Experiments` group runs for specific tasks.

import os
if 'MLFLOW_TRACKING_URI' in os.environ:
    del os.environ['MLFLOW_TRACKING_URI']
if 'MLFLOW_S3_ENDPOINT_URL' in os.environ:
    del os.environ['MLFLOW_S3_ENDPOINT_URL']

# +
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
    # hide parser = argparse.ArgumentParser()
    # hide parser.add_argument("--test-size", default=TEST_SIZE, type=float)
    # hide TEST_SIZE = parser.parse_args().test_size
        
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
# -

# save test locally for later use
test.to_csv('test.csv', index=False)

# This run must now be accessible through the UI, allowing us to review and verify that the metrics, parameters, and artifacts are correctly in place and ready for use. In the experiment, we see used datasets filed filled with datasets due to MLflow's dataset capabilities. However, it's important to understand that this represents metadata, not the actual data itself.
#  
# > The `mlflow.data` module tracks dataset information during model training and evaluation, storing metadata such as features, targets, predictions, name, schema, and source. Log this metadata using the `mlflow.log_input()` API.
#
#
# ![](img/data_preprocessing.png)

# ### Hyperparameters tuning
#
# In this part, we use Optuna to find the best hyperparameters for `XGBoost`, leveraging its built-in cross-validation for training and evaluation. Additionally, we'll demonstrate how to track metrics during the model fitting process with a custom callback.

# +
# hyperparameters_tuning
import tempfile
import sys
import psutil
import os
import argparse
import logging
import warnings
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from xgboost.callback import TrainingCallback
from loguru import logger


# set up logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)


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

    N_TRIALS = 10
    # get arguments if running not in ipykernel
    # hide parser = argparse.ArgumentParser()
    # hide parser.add_argument("--n-trials", default=N_TRIALS, type=float)
    # hide N_TRIALS = parser.parse_args().n_trials

    logger.info(f'Hyperparameters tuning started with: {N_TRIALS} trials')

    # start experiment
    experiment_id = mlflow.set_experiment('Cancer_Classification').experiment_id

    with mlflow.start_run(run_name='Hyperparameters_Search', log_system_metrics=True):
        
        # get last finished run for data preprocessing
        last_run_id = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Data_Preprocessing' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, 'run_id']
        
        # download train data from last run
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.artifacts.download_artifacts(run_id=last_run_id, artifact_path='datasets/train.csv', dst_path=tmpdir)
            train = pd.read_csv(os.path.join(tmpdir, 'datasets/train.csv'))
    
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
# -

# ### Review results from MLflow UI
#
#
# ![](img/runs_list.png)
#
# It is possible to utilize nested run capabilities to structure project, as demonstrated in this example. Here, we have one parent run for hyperparameter tuning and collect all trials as nested runs. MLflow also provides the ability to customize the columns and order of rows in this view, enhancing the organization and readability of experiment data.
#
# ![](img/runs_charts.png)
#
# The chart view allows for the comparison of runs and the setup of various plots. Using XGBoost callbacks to log metrics during the model fitting process enables the creation of plots with the number of trees on the x-axis.
#
# ![](img/compare_runs_counter_plot.png)
#
# MLflow allows for the comparison of runs. Select multiple runs, click the compare button, and choose the most useful view. This feature is particularly valuable when optimizing hyperparameters, as it helps refine and adjust the boundaries of possible intervals based on the comparison results.
#
# ![](img/system_metrics.png)
#
# System metrics can also be tracked throughout the run. While this may not provide an exact estimate of the real project requirements, it can still be useful in certain cases.
#

# ### Log and register model
#
# It is possible, but not necessary, to save the model for each experiment and run. In most scenarios, it is better to save the parameters and then, once the final parameters are selected, perform an additional run to save the model. Here, we follow the same logic: using the parameters from the best run to save the final model and register it for versioning and usage via a short link.

# convert this notebook to html in order to log it
# !jupyter nbconvert --to html modeling.ipynb

# +
# model_training
import os
import sys
import tempfile
import mlflow
import warnings
import logging
import xgboost as xgb
import pandas as pd
from loguru import logger


# set up logging
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


if __name__ == '__main__':

    logger.info('Model training started')
 
    mlflow.xgboost.autolog()

    experiment_id = mlflow.set_experiment('Cancer_Classification').experiment_id

    with mlflow.start_run(run_name='Model_Training') as run:
        
        run_id = run.info.run_id
        logger.info(f'Start mlflow run: {run_id}')
        
        # get last finished run for data preprocessing
        last_data_run_id = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Data_Preprocessing' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, 'run_id']
    
        # download train and test data from last run
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.artifacts.download_artifacts(run_id=last_data_run_id, artifact_path='datasets/train.csv', dst_path=tmpdir)
            mlflow.artifacts.download_artifacts(run_id=last_data_run_id, artifact_path='datasets/test.csv', dst_path=tmpdir)
            train = pd.read_csv(os.path.join(tmpdir, 'datasets/train.csv'))
            test = pd.read_csv(os.path.join(tmpdir, 'datasets/test.csv'))

        # convert to DMatrix format
        features = [i for i in train.columns if i != 'target']
        dtrain = xgb.DMatrix(data=train.loc[:, features], label=train['target'])
        dtest = xgb.DMatrix(data=test.loc[:, features], label=test['target'])

        # get last finished run for hyperparameters tuning
        last_tuning_run = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Hyperparameters_Search' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, :]
        
        # get best params
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
        
        # Log model as Booster
        input_example = test.loc[0:10, features]
        predictions_example = pd.DataFrame(model.predict(xgb.DMatrix(input_example)), columns=['predictions'])
        mlflow.xgboost.log_model(model, "booster", input_example=input_example)
        mlflow.log_text(predictions_example.to_json(orient='split', index=False), 'booster/predictions_example.json')

        # Register model
        model_uri = f"runs:/{run.info.run_id}/booster"
        mlflow.register_model(model_uri, 'CancerModelBooster')
        
        # Log model as sklearn completable XGBClassifier
        params.update(num_boost_round=model.best_iteration)
        model = xgb.XGBClassifier(**params)
        model.fit(train.loc[:, features], train['target'])
        mlflow.xgboost.log_model(model, "model", input_example=input_example)

        # log datasets
        mlflow.log_text(train.to_csv(index=False), 'datasets/train.csv')
        mlflow.log_text(test.to_csv(index=False),'datasets/test.csv')
        
        # log html with training notebook
        mlflow.log_artifact(local_path='modeling.html')
        
        logger.info('Model training finished')

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, 'CancerModel')
        
        logger.info('Model registered')
# -

# Thanks to the `mlflow.xgboost.autolog()` feature, which works fine with the XGBoost training API, all metrics are automatically saved without the need for custom callbacks.
#
# ![](img/model_metrics.png)
#
#
# Once we save a model, we can access the `artifacts` page within the run. 
#
# ![](img/model_artifacts.png)
#
#
# It is possible to store any type of file in artifacts, such as custom plots, text files, images, datasets, python scripts. For instance, I converted this notebook to `html` and saved it as an artifact, allowing it to be viewed directly in `mlflow ui`:
#
# ![](img/logged_html.png)
#
# For each model, MLflow automatically creates a yaml configuration file called `MLmodel`. This file can be viewed directly in the MLflow UI or downloaded and inspected:
#

# +
from IPython.display import display, Markdown
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='model/MLmodel', dst_path=tmpdir)
    with open(Path(tmpdir) / 'model/MLmodel', 'rb') as f:
        content = f.read().decode('utf-8')
        display(Markdown(f"```yaml\n{content}\n```"))
# -

# The `MLmodel` file supports multiple deployment `flavors`, including a generic python function and xgboost. It includes environment setups with both Conda (`conda.yaml`) and virtualenv (`python_env.yaml`). The model is an XGBoost classifier compatible with sklearn API, saved in the XGBoost format version 2.0.3. It tracks details such as model size, UUID, run ID, and creation time. We also provide an `example input` linked with the model and its `signature` - input and output specification. While the signature can be manually created and saved with the model, MLflow automatically generates the signature when an input example is provided.
#
# > In the MLflow ecosystem, `flavors` are wrappers for specific machine learning libraries, allowing consistent saving, logging, and retrieval of models. This ensures uniform `predict` method behavior across different frameworks for streamlined model management and deployment.
#
# > In addition to the input example, it is beneficial to include calculated predictions for this example. This allows immediate testing of the model after loading it, ensuring it performs correctly in different environment setups.
#

# ### Built in evaluation
#
# MLflow's built-in capability allows evaluation (`mlflow.evaluate`) of models on additional datasets, even those not available during training

# +
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
    # hide parser = argparse.ArgumentParser()
    # hide parser.add_argument("--eval-dataset", type=str)
    # hide eval_dataset = pd.read_csv(parser.parse_args().eval_dataset)
        
    with mlflow.start_run(run_name='Data_Evaluation') as run:
        
        eval_dataset = mlflow.data.from_pandas(
            eval_dataset, targets="target"
        )
        last_version = mlflow.MlflowClient().get_registered_model('CancerModel').latest_versions[0].version
        mlflow.evaluate(
            data=eval_dataset, model_type="classifier", model=f'models:/CancerModel/{last_version}'
        )
        logger.success('Evaluation finished')
# -

# The results can be viewed in the `mlflow ui`, where various metrics and plots are provided, including `roc-auc`, `confusion matrices`, and `shap plots` (if shap is installed).
#
# ![](img/shap_plot.png)
#

# ## MLflow Projects
#
#
# Next, you might want to share your project with other data scientists or automate your model training pipeline. This is where MLflow Projects come in.
#
# > An MLflow Project packages data science code in a reusable and reproducible way, following conventions that make it easy for others (or automated tools) to run. Each project is a directory of files or a git repository containing your code. MLflow can run these projects based on specific conventions for organizing files in the directory. 
#
# First, the main code cells will be converted into Python files using the nbformat library. A separate Python file will be created for each cell, based on comment lines at the top of each cell that specify the predefined names for the files.

# +
import nbformat
from pathlib import Path

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
    with open(Path('mlproject') / f'{comment}.py', 'w', encoding='utf-8') as f:
        f.write(processed_content)

    print(f'{comment}.py saved')


if __name__ == '__main__':
    for comment in ['data_preprocessing', 'hyperparameters_tuning', 'model_training', 'data_evaluation']:
        extract_and_save_cell('modeling.ipynb', comment)
# -

# # copy config and env files to mlproject folder
# !cp conda.yaml modeling.html test.csv mlproject

# ### Conda env export
#
# Creating a `conda.yaml` file can be complex and often requires manual steps. Exporting the current environment with `conda env export` may not be ideal for sharing or docker use due to platform-specific issues. Adding the `--from-history` flag lists only explicitly requested packages but fail with `pip-installed` packages. Using `pip freeze` includes local package links. Therefore, manually creating a `requirements.txt` file or `conda.yaml` might be the best solution.

# ### MLproject file
#
# The `MLproject` file helps MLflow and others understand and run your project by specifying the environment, entry points, and possible parameters for customization. Let's review the `MLproject` created for this project

# +
from IPython.display import Markdown, display
with open('mlproject/MLproject', 'r') as file:
    mlproject_content = file.read()

# Display the contents as a Markdown code snippet
display(Markdown(f"```yaml\n{mlproject_content}\n```"))
# -

# ### Mlflow run
# Project endpoints can be executed via the cli (`mlflow run`) or the python api (`mlflow.run`).
#
# > The `mlflow run` command sets the experiment and creates a run before executing the python script. Therefore, if we use the same commands inside our Python code with specified names, it is important to use the same names in this command.

mlflow.run(
    uri = 'mlproject',
    entry_point = 'data-preprocessing',
    env_manager='local',
    experiment_name='Cancer_Classification',
    run_name='Data_Preprocessing',
    parameters={'test-size': 0.5},
)

# Here, we run a second endpoint with a conda environment, creating an additional conda environment. We can verify its creation using the conda env list command.

mlflow.run(
    uri = 'mlproject',
    entry_point = 'hyperparameters-tuning',
    env_manager='conda',
    experiment_name='Cancer_Classification',
    run_name='Hyperparameters_Search',
    parameters={'n-trials': 3},
)

mlflow.run(
    uri = 'mlproject',
    entry_point = 'model-training',
    env_manager='conda',
    experiment_name='Cancer_Classification',
    run_name='Model_Training',
)

# +
# get data
import os
path = os.path.abspath('test.csv')

mlflow.run(
    uri = 'mlproject',
    entry_point = 'data-evaluation',
    env_manager='conda',
    experiment_name='Cancer_Classification',
    run_name='Data_Evaluation',
    parameters={'eval-dataset': path},
)
# -

# ### Docker setup
#
# `Dockerfile` and `docker compose` stored in the `mlproject/docker` folder. Docker image based on slim Python and install dependencies from a manually created `requirments.txt`. `docker compose` mounts `mlruns` directory to a volume to log all MLflow activities in the sa,e `mlruns` folder. In a production environment, we might run this command from an orchestration tool and provide MLFLOW_TRACKING_URI to the remote MLflow server. You can run `docker compose -f mlproject/docker/docker-compose.yml build` to build the image and then `docker compose -f mlproject/docker/docker-compose.yml up` to run it.
#
# > We need to mount the absolute path to the `mlruns` folder in the project root to log and retrieve artifacts. This is necessary because local MLflow artifact methods uses absolute paths.

# ## MLmodel: flavours
#
# In MLflow, models can be loaded using different flavors specified in the `MLmodel` file. We save two versions of the model both with xgboost, each have two flavors: `python_function` and `xgboost`. The difference lies in the model class: for the `booster`, it is `xgboost.core.Booster`, and for the `model`, it is `xgboost.sklearn.XGBClassifier`, which supports a scikit-learn compatible API. These differences affect how the predict method works, so it is important to review the `MLmodel` file and check the model signature before using it.
#
#
# When loading the `booster` model with the `xgboost`, the model expects the input data to be in the form of a `DMatrix` object and `predict` method will produce scores (not classes) in our case.

# Change error reporting mode to minimal
# %xmode Minimal

# +
import mlflow
import xgboost as xgb

test = pd.read_csv('test.csv')
dtest = xgb.DMatrix(data=test.loc[:, [i for i in test.columns if i != 'target']], label=test['target'])
test.drop('target', axis=1, inplace=True)

# download booster with xgboost flavour
logged_model = 'models:/CancerModelBooster/1'
xgboost_booster = mlflow.xgboost.load_model(logged_model)
# -

# error with pandas input
try:
    xgboost_booster.predict(test)
except Exception as e:
    print(e)

# work with DMatrix like predict proba
xgboost_booster.predict(dtest)[:3]

# download booster with pyfunc flavour
pyfunc_booster = mlflow.pyfunc.load_model(logged_model)

# work with pandas, produce probs
pyfunc_booster.predict(test)[:3]

# error with DMatrix
try:
    pyfunc_booster.predict(dtest)
except Exception as e:
    print(e)

# but we can still reach booster object and use it with DMatrix
pyfunc_booster._model_impl.xgb_model.predict(dtest)[:3]

# Let's examine with `xgboost.sklearn.XGBClassifier`

logged_model = 'models:/CancerModel/1'
xgboost_model = mlflow.xgboost.load_model(logged_model)

# predict method produce classes not probs - work with pandas
try:
    xgboost_model.predict(test.loc[:,features])[:3]
except Exception as e:
    print(e)

# not able to work with DMatrix
try:
    xgboost_model.predict(dtest)[:3]
except Exception as e:
    print(e)

pyfunc_model = mlflow.pyfunc.load_model(logged_model)

# ### Performace comparison
# Since the pyfunc model has some additional overhead, performance is slightly worse in our case. However, this can vary based on the data and model, and the difference may be greater.

# %%timeit
xgboost_booster.predict(xgb.DMatrix(test))

# %%timeit
pyfunc_booster.predict(test)

# %%timeit
xgboost_model.predict(test)

# %%timeit
pyfunc_model.predict(test)

# ## Model Serving
#
# MLflow has built-in capabilities to serve models. Serving a model with flask is pretty straightforward: `mlflow models serve -m models:/CancerModel/1 --env-manager local`. But we also may utilize `mlserver` and to do it properly we may first install  and create json config file:

# ### Mlserver
# It is also possible to use built-in integration with MLServer
#
# > The `mlserver` package facilitates efficient deployment and serving of machine learning models with support for multiple frameworks, using REST and gRPC interfaces. It integrates with seldon core for scalable and robust model management and monitoring.
#
# You might need to install the following packages: `mlserver`, `mlserver-mlflow`, `mlserver-xgboost` if you use your own environment. We can then set up a config file for `MLServer`. This allows us to modify how the API will work; here, we just set up a alias for the model:

# +
# %%writefile mlserve/model-settings.json

{
    "name": "cancer-model",
    "implementation": "mlserver_mlflow.MLflowRuntime",
    "parameters": {
        "uri": "models:/CancerModel/1"
    }
}
# -

# To start from local env we can use `mlserver start mlserve`, to start from the Docker setup, you can use the following commands: `docker compose -f mlserve/docker/docker-compose.yml build` and `docker compose -f mlserve/docker/docker-compose.yml up`. 
#
# > It's great that we have a working API with `openapi` documentation, request validation, `HTTP` and `gRPC` servers, and prometheus scrapping endpoint. And the best part is that all of this is achieved without any coding, just by providing a simple JSON configuration.
#
# > An alternative to this approach is writing a new API from scratch. This might be preferable when we need more flexibility, additional functionality, or when fitting within time constraints, as the MLServer approach can introduce some overhead and may be slightly slower.
#
# We can check the documentation for our model and inspect the expected data structure via swagger `/v2/models/cancer/model/docs`
#
# ![](img/mlserver.png)
#
# We can access the metrics endpoint or configure Prometheus to scrape it in production environments.
#
# ![](img/metrics_endpoint.png)
#
# *`mlserver` offers various customization please refer to the [documentation](https://mlserver.readthedocs.io/en/stable/) for more details.*

# Then it is possible to query the served model from a given endpoint and port:

# +
import requests
import json

url = "http://127.0.0.1:8080/invocations"

# convert df do split format and then to json
input_data = json.dumps({
    "params": {
      'method': 'proba',  
    },
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
# -

# ### Customize model
# We can customize our model to provide probabilities or include specific logging features. To do this, we will first obtain the model and then encapsulate it with a custom wrapper.

# +
import mlflow
import mlflow.xgboost
import mlflow.pyfunc

# Step 1: Download the Existing Model from MLflow
model_uri = "models:/CancerModel/1"
model = mlflow.xgboost.load_model(model_uri)


# Step 2: Define the Custom PyFunc Model with `loguru` Setup in `load_context`
class CustomPyFuncModel(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        
    def get_logger(self):
        from loguru import logger
        logger.remove()
        logger.add("mlserve/mlserver_logs.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        return logger
        
    def load_context(self, context):
        self.logger = self.get_logger()

    def predict(self, context, model_input):
        
        self.logger.info(f"start request")
        self.logger.info(f"batch size: {len(model_input)}")
        
        predict =  self.model.predict_proba(model_input)[:,1]
        
        self.logger.success(f"Finish request")
        
        return predict
        

# Step 3: Save the Wrapped Model Back to MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="custom_model",
        python_model=CustomPyFuncModel(model),
        registered_model_name="CustomCancerModel",
    )

# +
# %%writefile mlserve/model-settings.json

{
    "name": "cancer-model",
    "implementation": "mlserver_mlflow.MLflowRuntime",
    "parameters": {
        "uri": "models:/CustomCancerModel/1"
    }
}
# -

# We can then run `mlserver` again, query the `API` to obtain probabilities instead of classes, and implement our custom `logging`.

# +
# Send a POST request to the MLflow model server
response = requests.post(url, data=input_data, headers={"Content-Type": "application/json"})

if response.status_code == 200:
    prediction = response.json()
    print("Prediction:", prediction['predictions'][:10])
else:
    print("Error:", response.status_code, response.text)

# +
from IPython.display import Markdown, display
with open('mlserve/mlserver_logs.log', 'r') as file:
    mlproject_content = file.read()

# Display the contents as a Markdown code snippet
display(Markdown(f"```\n{mlproject_content}\n```"))
# -

# > While this method works, it might be more straightforward to set up a custom web server if we want to incorporate more complex logic rather than using the built-in tools.
#
# > In case of a regular model training pipeline, redeploying the service or updating the model version in the current API is not handled by the open-source version of MLflow. The custom Databricks version includes a webhook feature that allows MLflow to notify the API serving remotely about new versions. Another option is to trigger deployment when the model is updated. We could also expose an additional endpoint in the server and call it within a DAG, or we could have the server query MLflow periodically for updates.

# ## MLflow Tracking Server

# ### MLflow Local Setup
# We've been working on setting up MLflow locally, with metadata stored in the default `mlruns` folder, along with artifacts. You can check this folder on your machine if you've successfully completed all the previous steps. We can change the storage location for MLflow metadata by specifying a different `backend-store-uri` when running the MLflow UI command. For example, to use a different folder (`mlruns_new`), you would run: `mlflow ui --backend-store-uri ./mlruns_new` and set the tracking URI in your project with:
# `mlflow.set_tracking_uri("file:./mlruns_new")`.

# ### Remote Tracking
#
# In production environments, we typically set up a remote tracking server with artifact storage and a database for MLflow metadata. We can simulate this configuration using MinIO for artifact storage and PostgreSQL for the database. Here's a simple Docker Compose file to achieve this:
#
# 1. Set up MinIO.
# 2. Use MinIO client (`minio/mc`) to create a bucket for MLflow.
# 3. Run `PostgreSQL` as the database.
# 4. Start the `MLflow UI`.

# +
from IPython.display import Markdown, display
with open('tracking_server/docker-compose.yml', 'r') as file:
    mlproject_content = file.read()

# Display the contents as a Markdown code snippet
display(Markdown(f"```yaml\n{mlproject_content}\n```"))
# -

# You may run following commands to build and run it: `docker compose -f tracking_server/docker-compose.yml build` and  `docker compose -f tracking_server/docker-compose.yml up`.
#
# > MLflow uses a specific logic for handling artifacts. Instead of processing artifacts, MLflow provides a link to the client, allowing the client to save and download artifacts directly from the artifact storage. Therefore, you need to set up access keys and the tracking server host to log artifacts properly.

import os
os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5050'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

# run data preprocessing step one more time
mlflow.run(
    uri = 'mlproject',
    entry_point = 'data-preprocessing',
    env_manager='local',
    experiment_name='Cancer_Classification',
    run_name='Data_Preprocessing',
    parameters={'test-size': 0.5},
)

#
# After this step, we can access the MLflow tracking server to verify that your run and artifacts have been successfully logged:
#
# ![](img/tracking_server.png)
#
# And verify through the MinIO UI that the artifacts have been successfully stored in the bucket:
#
# ![](img/minio.png)
#
# And also query our PostgreSQL database to ensure it is being used for metadata:

# +
import psycopg2
import pandas as pd

conn = psycopg2.connect(dbname='mlflow', user='mlflow', password='password', host='localhost', port='5432')
try:
    query = "SELECT experiment_id, name FROM experiments"
    experiments_df = pd.read_sql(query, conn)
except Exception as e:
    print(e)
else:
    print(experiments_df)
finally:
    conn.close()
# -

# > For production deployment, you might need authentication, different user permissions, and monitoring for MLflow, the bucket, database, and specific artifacts. We won't cover these aspects here, and some of them don't have built-in capabilities in MLflow.
#
# So this concludes our tutorial where we explored various MLflow features and demonstrated how to utilize them. MLflow offers many other capabilities, which you can learn about by referring to the official documentation: [MLflow Documentation](https://mlflow.org/docs/latest/index.html).

# +
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
# -

# convert notebook to python
import jupytext
notebook = jupytext.read('modeling.ipynb')
jupytext.write(notebook, 'modeling.py')


