import json
import logging
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import os


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri('https://dagshub.com/guptatannu538/mlops-mini-project.mlflow')

logger = logging.getLogger('register_model')
logger.setLevel(logging.INFO)

def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def promote_model(model_name: str, model_info: dict):
    client = MlflowClient()
    
    # Fetch the latest version of the registered model
    latest_versions = client.get_latest_versions(model_name)
    if latest_versions:
        latest_version = latest_versions[-1].version
        
        # Transition the model to the "Staging" stage
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True # Moves older 'Staging' models to 'Archived'
            )
            logger.info(f"Model '{model_name}' version {latest_version} transitioned to 'Production' stage!")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")

def main():
    try:
        model_info = load_model_info('reports/experiment_info.json')
        promote_model("my_model", model_info)
    except Exception as e:
        logger.error(f"Error during model promotion: {e}")
        raise

if __name__ == '__main__':
    main()