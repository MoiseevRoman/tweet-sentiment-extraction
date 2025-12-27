import logging
import subprocess
from typing import Any, Optional

import mlflow

logger = logging.getLogger(__name__)


def get_git_commit_id() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def init_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Закрываем любые активные runs
    try:
        mlflow.end_run()
    except Exception:
        pass

    run = mlflow.start_run(run_name=run_name)
    commit_id = get_git_commit_id()
    if commit_id:
        mlflow.log_param("git_commit_id", commit_id)
        logger.info(f"Logged git commit ID: {commit_id}")

    return {"run_id": run.info.run_id, "run": run}


def log_hyperparameters(params: dict[str, Any]) -> None:
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)):
                    mlflow.log_param(f"{key}.{sub_key}", sub_value)
