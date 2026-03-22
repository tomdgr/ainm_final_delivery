import os
import sys

import mlflow


def is_running_in_azure_ml() -> bool:
    return os.environ.get("AZUREML_RUN_ID") is not None


def get_output_dir() -> str:
    return "./outputs"


def initialize_aml_logging(experiment_name: str = "nm_ai_image", run_name: str | None = None):
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stdout, format="{time} | {level} | {message}", level="INFO")
    except ImportError:
        pass

    if is_running_in_azure_ml():
        return

    if run_name is None:
        run_name = experiment_name

    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        ws = ml_client.workspaces.get(ml_client.workspace_name)
        tracking_uri = ws.mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")
