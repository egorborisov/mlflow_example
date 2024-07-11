import mlflow
import os
import tempfile
import pandas as pd
from pathlib import Path
from loguru import logger


def get_last_run(experiment_id: str, run_name: str):

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and status = 'FINISHED'",
        order_by=["start_time DESC"]
    )

    if runs.empty:
        logger.error(f'No {run_name} runs found')
        raise Exception(f'No {run_name} runs found')

    return runs.loc[0,:]


def upload_dataset_as_artifact(data: pd.DataFrame, file_name: str):

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(tmpdir, f"{file_name}.csv")

        # Save the datasets to the temporary directory
        data.to_csv(path, index=False)

        # Log the datasets as artifacts
        mlflow.log_artifact(path, artifact_path="datasets")

        logger.info(f'{file_name} uploaded')

        dataset_source_link = mlflow.get_artifact_uri(f'datasets/{file_name}.csv')
        dataset = mlflow.data.from_pandas(data, name=file_name, targets="target", source=dataset_source_link)
        mlflow.log_input(dataset)


def download_dataset_as_artifact(last_run_id, file_name: str):

    with (tempfile.TemporaryDirectory() as tmpdir):
        mlflow.artifacts.download_artifacts(
            run_id=last_run_id, artifact_path=f"datasets/{file_name}.csv", dst_path=tmpdir
        )
        data = pd.read_csv(Path(tmpdir) / "datasets" / f"{file_name}.csv")

    logger.info(f'{file_name} downloaded')

    return data

#%%
