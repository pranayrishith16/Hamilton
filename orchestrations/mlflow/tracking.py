# mlops/tracking.py

import os
from typing import Any
import yaml
import mlflow
from loguru import logger

# Load configuration once
with open("configs/mlops.yaml", "r") as f:
    _config = yaml.safe_load(f)

_ml_cfg = _config.get("mlflow", {})
_DEFAULT_TAGS = _ml_cfg.get("default_tags", {})

def _configure_mlflow():
    # Allow override via environment
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", _ml_cfg.get("tracking_uri"))
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if _ml_cfg.get("experiment_name"):
        mlflow.set_experiment(_ml_cfg["experiment_name"])

def start_run(run_name: str,nested:bool=False, **params):
    """
    Configure MLflow, then start a run, apply default tags and log params.
    """
    _configure_mlflow()
    mlflow.start_run(run_name=run_name,nested=nested)
    for k, v in _DEFAULT_TAGS.items():
        mlflow.set_tag(k, v)
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_param(key: str, value: Any):
    """
    Log a single parameter to MLflow.
    """
    try:
        mlflow.log_param(key, value)
    except Exception as e:
        logger.error(f"Failed to log param {key}: {e}")

def log_artifact(local_path: str, artifact_path: str = None):
    """
    Log a local file or directory as an MLflow artifact.
    """
    mlflow.log_artifact(local_path, artifact_path=artifact_path)

def log_metric(key: str, value: float, step: int = None):
    """
    Log a metric value to MLflow.
    """
    mlflow.log_metric(key, value, step=step)

def end_run():
    """
    End the active MLflow run.
    """
    mlflow.end_run()
