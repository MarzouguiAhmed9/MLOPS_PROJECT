import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live
import datetime

# ----------------------------
# Logging setup with timestamp
# ----------------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(log_dir, f"model_evaluation_{timestamp}.log")

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def close_logger():
    """Close all logger handlers to release file locks."""
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

# ----------------------------
# Functions
# ----------------------------
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving metrics: %s', e)
        raise

# ----------------------------
# Main
# ----------------------------
def main():
    try:
        params = load_params('params.yaml')
        clf = load_model(r'C:\Users\21650\Desktop\MLOPS PROJECT\MLOPS_PROJECT\src\models\model.pkl')

        test_data = load_data('./data/processed/test_tfidf.csv')
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        # Track experiment with dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('recall', metrics['recall'])
            live.log_metric('auc', metrics['auc'])
            live.log_params(params)

        save_metrics(metrics, 'reports/metrics.json')

        logger.info('Model evaluation completed successfully')

    except Exception as e:
        logger.error('Failed to complete model evaluation: %s', e)
        print(f"Error: {e}")
    finally:
        close_logger()  # Ensure all handlers are closed

# ----------------------------
# Entry point
# ----------------------------
if __name__ == '__main__':
    main()
