"""
Task 2 - training
Carica i dati preparati dal Task 1, allena XGBoost con iperparametri già ottimizzati
(niente RandomizedSearchCV ad ogni run: quello si fa una volta in sviluppo),
salva il pickle finale con model + artifact di preprocessing.
"""
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

MLFLOW_TRACKING_URI = "http://mlflow:5000"
PROCESSED_DIR = Path("/opt/airflow/data/processed")
MODEL_DIR = Path("/opt/airflow/models")

# Iperparametri congelati: risultato del RandomizedSearchCV fatto in sviluppo
# (best_params_ stampati dallo script originale). Da aggiornare manualmente
# se si rifà il tuning.
BEST_PARAMS = {
    'subsample': 0.9,
    'n_estimators': 400,
    'min_child_weight': 5,
    'max_depth': 7,
    'learning_rate': 0.03,
    'colsample_bytree': 0.6,
}


def train_model(processed_dir: Path = PROCESSED_DIR, model_dir: Path = MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction_v4")

    X_train = pd.read_parquet(processed_dir / 'X_train.parquet')
    y_train = pd.read_parquet(processed_dir / 'y_train.parquet')['churn']

    with mlflow.start_run() as run:
        xgb = XGBClassifier(**BEST_PARAMS, eval_metric='auc', random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)

        mlflow.log_params(BEST_PARAMS)
        mlflow.xgboost.log_model(xgb, "model")

        with open(processed_dir / 'preprocessing_artifact.pkl', 'rb') as f:
            artifact = pickle.load(f)
        artifact['model'] = xgb
        artifact['decision_threshold'] = 0.5

        model_path = model_dir / 'churn_xgb_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(artifact, f)
        mlflow.log_artifact(str(model_path))

        print(f"Modello salvato in {model_path}, MLflow run_id={run.info.run_id}")
        return {"model_path": str(model_path), "mlflow_run_id": run.info.run_id}


if __name__ == '__main__':
    train_model()