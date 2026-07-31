# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
       import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import time

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Check for DAGSHUB_PAT
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise unittest.SkipTest("DAGSHUB_PAT not set, skipping model loading tests")

        # Use username + token for authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = "guptatannu538"  # replace with your DagsHub username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri('https://dagshub.com/guptatannu538/mlops-mini-project.mlflow')
        # Load the new model from MLflow model registry
        cls.new_model_name = "model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

