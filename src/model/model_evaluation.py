# updated model evaluation

import numpy as np
import pandas as pd
import yaml
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import dagshub
import mlflow
import logging
import mlflow.sklearn
import os



mlflow.set_tracking_uri('https://dagshub.com/guptatannu538/mlops-mini-project.mlflow')
dagshub.init(repo_owner='guptatannu538', repo_name='mlops-mini-project', mlflow=True)

def load_model(model_path):
    return pickle.load(open(model_path, 'rb'))

def load_data(data_path):
    test_data = pd.read_csv(data_path)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return X_test, y_test

def make_prediction(clf, X_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba

def compute_metrics(y_test, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

def save_metrics(path, metrics_dict):
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def main():
    mlflow.set_experiment('dvc-pipeline')
    mlflow.start_run()
    try:
        clf = load_model('models/model.pkl')
        X_test, y_test = load_data('./data/processed/test_bow.csv')
        y_pred, y_pred_proba = make_prediction(clf, X_test)
        metrics_dict = compute_metrics(y_test, y_pred, y_pred_proba)

        # Load parameters
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)

        # Save metrics to file
        save_metrics('reports/metrics.json', metrics_dict)

        # log metrics to MLflow
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value)

        # log model parameters to MLflow
        if hasattr(clf, 'get_params'):
            params=clf.get_params()
            for param_name,param_value in params.items():
                mlflow.log_params(param_name, param_value)

        mlflow.sklearn.log_model(clf, "model")
            
            
        # Log the metrics file to MLflow
        mlflow.log_artifact('reports/metrics.json')

        # Log the model info file to MLflow
        mlflow.log_artifact('reports/model_info.json')

        # Log the evaluation errors log file to MLflow
        mlflow.log_artifact('model_evaluation_errors.log')
    except Exception as  e:
        print(f'error is occured:{e}')
if __name__ == '__main__':
    main()
