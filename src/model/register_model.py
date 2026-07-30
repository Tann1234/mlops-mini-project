# register model

import json
import mlflow
import logging
import os
import dagshub
from mlflow.tracking import MlflowClient

# set up dagshub credentials for MLFlow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri('https://dagshub.com/guptatannu538/mlops-mini-project.mlflow')

def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def register_model(model_name: str, model_info: dict):
    # Build the model URI from run_id and artifact path
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    client = MlflowClient()

    # Explicitly create a model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=model_info['run_id']
    )

    version_number = model_version.version
    print(f"✅ Registered {model_name} as version {version_number}")

    # Transition stage
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Staging"
    )
    print(f"🚀 Model {model_name} version {version_number} transitioned to Staging")

def main():
    model_info = load_model_info("reports/experiment_info.json")
    register_model("my_model", model_info)

if __name__ == "__main__":
    main()





