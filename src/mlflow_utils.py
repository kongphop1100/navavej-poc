import json
import tempfile
from contextlib import nullcontext
from pathlib import Path


def get_mlflow():
    try:
        import mlflow  # type: ignore
    except ImportError:
        return None
    return mlflow


def start_run_if_available(run_name: str, experiment_name: str | None = None):
    mlflow = get_mlflow()
    if mlflow is None:
        return None, nullcontext()

    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return mlflow, mlflow.start_run(run_name=run_name)


def log_dict_artifact(mlflow, data: dict, artifact_file: str):
    if mlflow is None:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / artifact_file
        artifact_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(artifact_path))
