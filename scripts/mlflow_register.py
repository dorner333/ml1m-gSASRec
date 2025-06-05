import mlflow
import mlflow.pyfunc
from pathlib import Path
import sys
from gsasrec.mlflow_inference import ModelWrapper

base = Path(__file__).resolve().parents[1]
sys.path.append(str(base))


with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="gsasrec_model",
        python_model=ModelWrapper(),
        artifacts={"model_path": str(base / "model_artifact")},
    )
    print("RUN_ID=", run.info.run_id)
