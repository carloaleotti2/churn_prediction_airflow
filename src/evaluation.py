"""
Task 3 - evaluation
Carica il modello allenato e il test set, calcola le metriche sull'hold-out,
salva un log in JSON.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
import mlflow
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
)

MLFLOW_TRACKING_URI = "http://mlflow:5000"
PROCESSED_DIR = Path("/opt/airflow/data/processed")
MODEL_DIR = Path("/opt/airflow/models")
METRICS_DIR = Path("/opt/airflow/logs/metrics")


import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
)
import mlflow

MLFLOW_TRACKING_URI = "http://mlflow:5000"
PROCESSED_DIR = Path("/opt/airflow/data/processed")
MODEL_DIR = Path("/opt/airflow/models")
METRICS_DIR = Path("/opt/airflow/logs/metrics")


def evaluate_model(mlflow_run_id: str,
                    processed_dir: Path = PROCESSED_DIR,
                    model_dir: Path = MODEL_DIR,
                    metrics_dir: Path = METRICS_DIR):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with open(model_dir / 'churn_xgb_model.pkl', 'rb') as f:
        artifact = pickle.load(f)
    model = artifact['model']
    threshold = artifact.get('decision_threshold', 0.5)

    X_test = pd.read_parquet(processed_dir / 'X_test.parquet')
    y_test = pd.read_parquet(processed_dir / 'y_test.parquet')['churn']

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'auc': roc_auc_score(y_test, proba),
        'recall': recall_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'threshold': threshold,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'n_test_samples': len(y_test),
    }

    # Riapro lo STESSO run del training (non ne creo uno nuovo) per
    # aggiungere le metriche di valutazione allo stesso esperimento
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics({
            'holdout_auc': metrics['auc'],
            'holdout_recall': metrics['recall'],
            'holdout_precision': metrics['precision'],
            'holdout_f1': metrics['f1'],
        })

    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    metrics_path = metrics_dir / f'metrics_{run_id}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metriche salvate in {metrics_path}")
    print(json.dumps(metrics, indent=2))

    return metrics

if __name__ == '__main__':
    evaluate_model()